#!/usr/bin/env python3
"""Ersatz SLURM: Distribute jobs across Vast.ai GPU instances."""

import argparse
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from setup_vast import (
    NoInstancesAvailableError,
    VastInstanceResult,
    VastInstanceSetupError,
    VastNonZeroExit,
    create_vast_instance,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# YAML format version
YAML_VERSION = "0.2"


@dataclass
class Job:
    """Represents a job to be executed."""

    job_id: str
    script: str
    attempt_number: int = 1  # Track retry attempts


def validate_job_id(job_id: str) -> bool:
    """Validate that job_id contains only alphanumeric, underscore, and hyphen."""
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", job_id))


def validate_version(version: str) -> bool:
    """
    Validate that version matches the required YAML_VERSION.

    Args:
        version: Version string from YAML.

    Returns:
        True if version matches YAML_VERSION, False otherwise.
    """
    return str(version) == YAML_VERSION


def load_jobs(yaml_path: Path) -> tuple[list[Job], str]:
    """
    Load and validate jobs from YAML file.

    Args:
        yaml_path: Path to YAML file containing job definitions.

    Returns:
        Tuple of (list of Job objects, setup command string).

    Raises:
        ValueError: If YAML format is invalid, version is incompatible, or job IDs are invalid/not unique.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML must be a dict with 'version', 'setup', and 'jobs' keys")

    # Validate version
    version = data.get("version")
    if not version:
        raise ValueError("YAML missing required 'version' key")
    if not validate_version(str(version)):
        raise ValueError(f"Incompatible version '{version}': expected '{YAML_VERSION}'")

    # Get setup command
    setup = data.get("setup", "")
    if not isinstance(setup, str):
        raise ValueError("'setup' must be a string")

    # Get jobs list
    jobs_data = data.get("jobs")
    if not jobs_data:
        raise ValueError("YAML missing required 'jobs' key")
    if not isinstance(jobs_data, list):
        raise ValueError("'jobs' must be a list")

    jobs = []
    seen_ids = set()

    for item in jobs_data:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid job entry: {item}")

        job_id = item.get("job_id")
        script = item.get("script")

        if not job_id:
            raise ValueError(f"Job missing job_id: {item}")
        if not script:
            raise ValueError(f"Job {job_id} missing script")

        if not validate_job_id(job_id):
            raise ValueError(
                f"Invalid job_id '{job_id}': must contain only alphanumeric, underscore, and hyphen"
            )

        if job_id in seen_ids:
            raise ValueError(f"Duplicate job_id: {job_id}")

        seen_ids.add(job_id)
        jobs.append(Job(job_id=job_id, script=script))

    logger.info(f"Loaded {len(jobs)} jobs from {yaml_path} (version {version})")
    return jobs, setup


def run_ssh_command(ssh_config_name: str, command: str, log_file: Path) -> bool:
    """
    Run a command via SSH and log the output.

    Args:
        ssh_config_name: SSH config name (e.g., 'vast06').
        command: Command to execute.
        log_file: Path to log file.

    Returns:
        True if command succeeded, False otherwise.
    """
    with open(log_file, "a") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Running: {command}\n")
        f.write(f"{'=' * 80}\n\n")

    result = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            command,
        ],
        capture_output=True,
        text=True,
    )

    with open(log_file, "a") as f:
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
        f.write(f"\nReturn code: {result.returncode}\n")

    return result.returncode == 0


def check_for_machine_error(log_file: Path) -> bool:
    """
    Check if the log file contains machine errors that warrant a retry on a different instance.

    This includes SSH connection errors and hardware errors (like CUDA illegal instruction).

    Args:
        log_file: Path to log file to check.

    Returns:
        True if machine error detected, False otherwise.
    """
    try:
        with open(log_file, "r") as f:
            log_content = f.read()

        # Check for connection closed by remote host pattern
        if re.search(r"Connection to [\d.]+ closed by remote host", log_content):
            return True

        # Check for broken pipe pattern
        if re.search(
            r"client_loop: ssh_packet_write_poll: Connection to [\d.]+ port \d+: Broken pipe",
            log_content,
        ):
            return True

        # Check for CUDA illegal instruction error (buggy GPU/machine)
        if re.search(
            r"torch\.AcceleratorError: CUDA error: an illegal instruction was encountered",
            log_content,
        ):
            return True

        return False
    except Exception as e:
        logger.warning(f"Failed to check log file for machine errors: {e}")
        return False


def get_base_job_id(job_id: str) -> str:
    """
    Extract the base job ID by removing attempt suffix.

    Args:
        job_id: Job ID potentially containing _attempt_N suffix.

    Returns:
        Base job ID without attempt suffix.
    """
    # Remove _attempt_N suffix if present
    return re.sub(r"_attempt_\d+$", "", job_id)


def create_single_instance(
    instance_num: int,
    pool_size: int,
    gpu_types: list[str],
    min_driver_version: str,
    retry_interval: int,
    repo_user: str | None = None,
    repo_name: str | None = None,
    instance_log_file: Path | None = None,
    instance_log_lock=None,
    ctx: "WorkerContext | None" = None,
) -> VastInstanceResult | None:
    """
    Create a single Vast.ai instance, retrying until successful or all jobs are done.

    Args:
        instance_num: Instance number (1-indexed for logging).
        pool_size: Total pool size (for logging).
        gpu_types: List of acceptable GPU types.
        min_driver_version: Minimum GPU driver version.
        retry_interval: Seconds to wait between GPU spawn retries.
        repo_user: GitHub username for repository to clone.
        repo_name: GitHub repository name to clone.
        instance_log_file: Path to file for logging created instance IDs.
        instance_log_lock: Lock for thread-safe writes to instance_log_file.
        ctx: WorkerContext to check if all jobs are completed.

    Returns:
        VastInstanceResult object, or None if all jobs are completed before instance is created.
    """
    while True:
        for gpu_type in gpu_types:
            try:
                logger.info(
                    f"[Pool {instance_num}/{pool_size}] Creating {gpu_type} instance"
                )
                instance = create_vast_instance(
                    gpu_type=gpu_type,
                    repo_user=repo_user,
                    repo_name=repo_name,
                    min_driver_version=min_driver_version,
                    full_ssh_name=True,
                    instance_log_file=instance_log_file,
                    instance_log_lock=instance_log_lock,
                )
                logger.info(
                    f"[Pool {instance_num}/{pool_size}] Created instance {instance.instance_id} ({instance.ssh_config_name})"
                )
                return instance
            except NoInstancesAvailableError:
                logger.warning(
                    f"[Pool {instance_num}/{pool_size}] No {gpu_type} instances available, trying next GPU type"
                )
                continue
            except VastNonZeroExit as e:
                logger.warning(
                    f"[Pool {instance_num}/{pool_size}] Command failed for {gpu_type} (before instance created): {e}"
                )
                logger.warning(
                    f"[Pool {instance_num}/{pool_size}] Return code: {e.returncode}"
                )
                if e.stdout:
                    logger.warning(
                        f"[Pool {instance_num}/{pool_size}] Stdout: {e.stdout}"
                    )
                if e.stderr:
                    logger.warning(
                        f"[Pool {instance_num}/{pool_size}] Stderr: {e.stderr}"
                    )
                logger.warning(
                    f"[Pool {instance_num}/{pool_size}] Trying next GPU type"
                )
                continue
            except VastInstanceSetupError as e:
                logger.warning(
                    f"[Pool {instance_num}/{pool_size}] Setup failed for {gpu_type}: {e}"
                )
                logger.info(
                    f"[Pool {instance_num}/{pool_size}] Destroying failed instance {e.instance_id}"
                )
                try:
                    subprocess.run(
                        ["uvx", "vastai", "destroy", "instance", e.instance_id],
                        capture_output=True,
                        timeout=30,
                    )
                    logger.info(
                        f"[Pool {instance_num}/{pool_size}] Instance {e.instance_id} destroyed"
                    )
                except Exception as destroy_error:
                    logger.error(
                        f"[Pool {instance_num}/{pool_size}] Failed to destroy instance {e.instance_id}: {destroy_error}"
                    )
                logger.warning(
                    f"[Pool {instance_num}/{pool_size}] Trying next GPU type"
                )
                continue

        # No GPUs available, check if all jobs are done before retrying
        if ctx is not None:
            with ctx.completed_jobs_lock:
                if ctx.completed_jobs_count[0] >= ctx.total_jobs:
                    logger.info(
                        f"[Pool {instance_num}/{pool_size}] All jobs completed while waiting for instance, exiting without creating instance"
                    )
                    return None

        logger.warning(
            f"[Pool {instance_num}/{pool_size}] No instances available, retrying in {retry_interval}s"
        )
        time.sleep(retry_interval)


def prepare_instance_for_next_job(
    instance: VastInstanceResult,
    log_file: Path,
    repo_user: str,
    repo_name: str,
    use_gh_for_clone: bool,
) -> bool:
    """
    Prepare an instance for the next job by cleaning up and re-cloning.

    Args:
        instance: Instance to prepare.
        log_file: Path to log file for this operation.
        repo_user: GitHub username for repository.
        repo_name: GitHub repository name.
        use_gh_for_clone: Whether to use gh (if authenticated) or git clone.

    Returns:
        True if preparation succeeded, False otherwise.
    """
    logger.info(f"[{instance.ssh_config_name}] Preparing instance for next job")

    # Remove the repository
    cleanup_cmd = f"rm -rf {repo_name}"
    if not run_ssh_command(instance.ssh_config_name, cleanup_cmd, log_file):
        logger.error(f"[{instance.ssh_config_name}] Failed to cleanup repository")
        return False

    # Re-clone the repository
    if use_gh_for_clone:
        clone_cmd = f"gh repo clone {repo_user}/{repo_name}"
    else:
        clone_cmd = f"git clone https://github.com/{repo_user}/{repo_name}.git"

    if not run_ssh_command(instance.ssh_config_name, clone_cmd, log_file):
        logger.error(f"[{instance.ssh_config_name}] Failed to re-clone repository")
        return False

    logger.info(f"[{instance.ssh_config_name}] Instance prepared for next job")
    return True


def destroy_single_instance(instance: VastInstanceResult):
    """
    Destroy a single instance.

    Args:
        instance: Instance to destroy.
    """
    try:
        logger.info(f"Destroying instance {instance.instance_id}")
        subprocess.run(
            ["uvx", "vastai", "destroy", "instance", instance.instance_id],
            check=True,
            capture_output=True,
            timeout=30,
        )
        logger.info(f"Instance {instance.instance_id} destroyed")
    except Exception as e:
        logger.error(f"Failed to destroy instance {instance.instance_id}: {e}")


@dataclass
class WorkerContext:
    """Shared context for worker threads."""

    worker_id: int
    total_workers: int
    job_queue: queue.Queue
    total_jobs: int
    completed_jobs_count: list[int]  # Single-element list for mutability
    completed_jobs_lock: threading.Lock
    failed_jobs: list[str]
    failed_jobs_lock: threading.Lock
    results_base_dir: Path
    timestamp: str
    setup_command: str
    gpu_types: list[str]
    min_driver_version: str
    retry_interval: int
    instance_log_file: Path | None
    instance_log_lock: "threading.Lock | None"
    s3_bucket: str | None
    remote_results_dir: str | None
    rsync_enabled: bool
    repo_user: str | None
    repo_name: str | None
    use_gh_for_clone: bool


def worker_thread(ctx: WorkerContext):
    """
    Worker thread that creates its own instance and processes jobs.

    Each worker:
    1. Creates a Vast.ai instance (retrying indefinitely)
    2. Gets jobs from the queue and executes them
    3. On setup failure: re-queues job, destroys instance, creates new one
    4. If queue is empty, waits up to 5 minutes (checking every 10s) for new jobs
    5. After 5 minutes of no work, destroys instance and exits
    6. Exits when all jobs are completed
    """
    instance: VastInstanceResult | None = None
    idle_start_time: float | None = None  # Track when we started being idle

    try:
        while True:
            # Check if all jobs are done
            with ctx.completed_jobs_lock:
                if ctx.completed_jobs_count[0] >= ctx.total_jobs:
                    logger.info(f"[Worker {ctx.worker_id}] All jobs completed, exiting")
                    break

            # Create instance if we don't have one
            if instance is None:
                logger.info(f"[Worker {ctx.worker_id}] Creating instance...")
                instance = create_single_instance(
                    instance_num=ctx.worker_id,
                    pool_size=ctx.total_workers,
                    gpu_types=ctx.gpu_types,
                    min_driver_version=ctx.min_driver_version,
                    retry_interval=ctx.retry_interval,
                    repo_user=ctx.repo_user,
                    repo_name=ctx.repo_name,
                    instance_log_file=ctx.instance_log_file,
                    instance_log_lock=ctx.instance_log_lock,
                    ctx=ctx,
                )
                # If instance is None, all jobs were completed while waiting
                if instance is None:
                    logger.info(
                        f"[Worker {ctx.worker_id}] All jobs completed while waiting for instance, exiting"
                    )
                    break
                logger.info(
                    f"[Worker {ctx.worker_id}] Instance ready: {instance.instance_id} ({instance.ssh_config_name})"
                )
                idle_start_time = None  # Reset idle timer when we get a new instance

            # Try to get a job (checking every 10 seconds)
            try:
                job = ctx.job_queue.get(timeout=10.0)
                # Got a job, reset idle timer
                idle_start_time = None
            except queue.Empty:
                # No job available
                # Check if all jobs are done
                with ctx.completed_jobs_lock:
                    if ctx.completed_jobs_count[0] >= ctx.total_jobs:
                        logger.info(
                            f"[Worker {ctx.worker_id}] All jobs completed, exiting"
                        )
                        break

                # Start or continue idle timer
                if idle_start_time is None:
                    idle_start_time = time.time()
                    logger.info(
                        f"[Worker {ctx.worker_id}] Queue empty, starting 5-minute idle timer"
                    )
                else:
                    idle_duration = time.time() - idle_start_time
                    if idle_duration >= 300:  # 5 minutes
                        logger.info(
                            f"[Worker {ctx.worker_id}] No work for 5 minutes, destroying instance and exiting"
                        )
                        break

                # Continue waiting
                continue

            logger.info(
                f"[Worker {ctx.worker_id}] Got job {job.job_id}, running on {instance.ssh_config_name}"
            )

            # Set up job logging
            job_results_dir = ctx.results_base_dir / ctx.timestamp / job.job_id
            job_results_dir.mkdir(parents=True, exist_ok=True)
            log_file = job_results_dir / "output.log"

            # Execute setup command if provided
            if ctx.setup_command:
                logger.info(
                    f"[Worker {ctx.worker_id}] [{job.job_id}] Running setup command"
                )
                setup_cmd = f"cd {ctx.repo_name} && {ctx.setup_command}"
                setup_success = run_ssh_command(
                    instance.ssh_config_name, setup_cmd, log_file
                )
                if not setup_success:
                    # Check if this is a machine error (SSH or hardware issue)
                    if check_for_machine_error(log_file):
                        logger.warning(
                            f"[Worker {ctx.worker_id}] [{job.job_id}] Setup failed due to machine error (SSH/hardware issue), re-queuing with new attempt number"
                        )
                        # Create new job with incremented attempt number
                        new_attempt = job.attempt_number + 1
                        base_job_id = get_base_job_id(job.job_id)
                        new_job_id = f"{base_job_id}_attempt_{new_attempt}"
                        retry_job = Job(
                            job_id=new_job_id,
                            script=job.script,
                            attempt_number=new_attempt,
                        )
                        ctx.job_queue.put(retry_job)
                        logger.info(
                            f"[Worker {ctx.worker_id}] Re-queued job as {new_job_id}"
                        )
                    else:
                        logger.error(
                            f"[Worker {ctx.worker_id}] [{job.job_id}] Setup failed (not SSH error), re-queuing job"
                        )
                        # Re-queue job with same ID (non-SSH setup failure)
                        ctx.job_queue.put(job)
                    # Destroy instance
                    destroy_single_instance(instance)
                    instance = None
                    continue

            # Execute job commands
            logger.info(f"[Worker {ctx.worker_id}] [{job.job_id}] Executing script")
            commands = f"cd {ctx.repo_name} && {job.script}"
            success = run_ssh_command(instance.ssh_config_name, commands, log_file)

            if success:
                logger.info(
                    f"[Worker {ctx.worker_id}] [{job.job_id}] SUCCESS - Job script completed"
                )

            if not success:
                # Check if this is a machine error (SSH or hardware) that warrants a retry
                if check_for_machine_error(log_file):
                    logger.warning(
                        f"[Worker {ctx.worker_id}] [{job.job_id}] Machine error detected (SSH/hardware issue), re-queuing job and recreating instance"
                    )
                    # Create new job with incremented attempt number
                    new_attempt = job.attempt_number + 1
                    base_job_id = get_base_job_id(job.job_id)
                    new_job_id = f"{base_job_id}_attempt_{new_attempt}"
                    retry_job = Job(
                        job_id=new_job_id, script=job.script, attempt_number=new_attempt
                    )
                    # Re-queue the job
                    ctx.job_queue.put(retry_job)
                    logger.info(
                        f"[Worker {ctx.worker_id}] Re-queued job as {new_job_id}"
                    )
                    # Destroy instance
                    destroy_single_instance(instance)
                    instance = None
                    # Don't increment completed_jobs_count since we're retrying
                    continue

                error_msg = f"Job {job.job_id} commands failed (see {log_file})"
                logger.error(f"[Worker {ctx.worker_id}] {error_msg}")
                with ctx.failed_jobs_lock:
                    ctx.failed_jobs.append(error_msg)

            # Rsync results back
            if ctx.rsync_enabled and ctx.remote_results_dir:
                logger.info(f"[Worker {ctx.worker_id}] [{job.job_id}] Syncing results")
                rsync_result = subprocess.run(
                    [
                        "rsync",
                        "-avz",
                        "--exclude=*.safetensors",
                        f"{instance.ssh_config_name}:{ctx.repo_name}/{ctx.remote_results_dir}/",
                        str(job_results_dir / ctx.remote_results_dir),
                    ],
                    capture_output=True,
                    text=True,
                )

                with open(log_file, "a") as f:
                    f.write(f"\n{'=' * 80}\n")
                    f.write("Rsync output:\n")
                    f.write(f"{'=' * 80}\n")
                    if rsync_result.stdout:
                        f.write(rsync_result.stdout)
                    if rsync_result.stderr:
                        f.write(rsync_result.stderr)

                if rsync_result.returncode != 0:
                    error_msg = f"Job {job.job_id} rsync failed (return code {rsync_result.returncode})"
                    logger.error(f"[Worker {ctx.worker_id}] {error_msg}")
                    with ctx.failed_jobs_lock:
                        ctx.failed_jobs.append(error_msg)

            # Upload results to S3 from the remote instance
            if ctx.s3_bucket and ctx.remote_results_dir:
                logger.info(f"[Worker {ctx.worker_id}] [{job.job_id}] Uploading to S3")
                s3_path = f"s3://{ctx.s3_bucket}/{ctx.timestamp}/{job.job_id}/"
                s3_cmd = (
                    f"cd {ctx.repo_name} && aws s3 sync {ctx.remote_results_dir}/ {s3_path}"
                )
                s3_success = run_ssh_command(instance.ssh_config_name, s3_cmd, log_file)
                if not s3_success:
                    error_msg = f"Job {job.job_id} S3 upload failed"
                    logger.error(f"[Worker {ctx.worker_id}] {error_msg}")
                    with ctx.failed_jobs_lock:
                        ctx.failed_jobs.append(error_msg)

                # Upload job log file to S3
                logger.info(
                    f"[Worker {ctx.worker_id}] [{job.job_id}] Uploading job log to S3"
                )
                try:
                    log_upload = subprocess.run(
                        ["aws", "s3", "cp", str(log_file), s3_path],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if log_upload.returncode != 0:
                        error_msg = (
                            f"Job {job.job_id} log S3 upload failed: {log_upload.stderr}"
                        )
                        logger.error(f"[Worker {ctx.worker_id}] {error_msg}")
                        with ctx.failed_jobs_lock:
                            ctx.failed_jobs.append(error_msg)
                except Exception as e:
                    error_msg = f"Job {job.job_id} log S3 upload error: {e}"
                    logger.error(f"[Worker {ctx.worker_id}] {error_msg}")
                    with ctx.failed_jobs_lock:
                        ctx.failed_jobs.append(error_msg)

            logger.info(
                f"[Worker {ctx.worker_id}] [{job.job_id}] Job completed, results in {job_results_dir}"
            )

            # Prepare instance for next job
            logger.info(
                f"[Worker {ctx.worker_id}] [{job.job_id}] Preparing instance for reuse"
            )
            if not prepare_instance_for_next_job(
                instance, log_file, ctx.repo_user, ctx.repo_name, ctx.use_gh_for_clone
            ):
                error_msg = f"Job {job.job_id} failed to prepare instance {instance.instance_id} for reuse"
                logger.error(f"[Worker {ctx.worker_id}] {error_msg}")
                with ctx.failed_jobs_lock:
                    ctx.failed_jobs.append(error_msg)
                # Destroy the instance and create a new one next iteration
                destroy_single_instance(instance)
                instance = None

            # Mark job as completed
            with ctx.completed_jobs_lock:
                ctx.completed_jobs_count[0] += 1
                logger.info(
                    f"[Worker {ctx.worker_id}] Completed {ctx.completed_jobs_count[0]}/{ctx.total_jobs} jobs"
                )

    except Exception as e:
        logger.error(f"[Worker {ctx.worker_id}] Fatal error: {e}", exc_info=True)

    finally:
        # Clean up instance
        if instance is not None:
            logger.info(
                f"[Worker {ctx.worker_id}] Destroying instance {instance.instance_id}"
            )
            destroy_single_instance(instance)


def get_instance_statuses() -> dict[str, str]:
    """
    Get current status of all instances with our docker image.

    Returns:
        Dict mapping instance_id to status for instances with theodoreehrenborg/ubuntu-uv image.
    """
    try:
        result = subprocess.run(
            ["uvx", "vastai", "show", "instances"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to get instances: {result.stderr}")
            return {}

        instance_statuses = {}
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            fields = line.split()
            if len(fields) < 14:
                continue

            instance_id = fields[0]
            status = fields[2]
            # Image name is typically around field 13, look for our docker image
            if "theodoreehrenborg/ubuntu-uv" in line and status in [
                "loading",
                "created",
            ]:
                instance_statuses[instance_id] = status

        return instance_statuses
    except Exception as e:
        logger.warning(f"Error checking instance statuses: {e}")
        return {}


def stuck_instance_monitor_thread(stop_event: threading.Event):
    """
    Monitor instances and destroy any stuck in 'loading' or 'created' state for 5+ minutes.

    Runs continuously until stop_event is set. Checks every 30 seconds and destroys
    instances that maintain the same status (loading or created) for 5 minutes.

    Args:
        stop_event: Event to signal thread should stop.
    """
    # Track when we first saw each instance in its current status
    instance_status_times: dict[
        str, tuple[str, float]
    ] = {}  # instance_id -> (status, first_seen_time)
    check_interval = 30  # Check every 30 seconds
    stuck_threshold = 300  # 5 minutes

    while not stop_event.is_set():
        try:
            current_statuses = get_instance_statuses()
            current_time = time.time()

            # Update tracking for all instances
            for instance_id, current_status in current_statuses.items():
                if instance_id in instance_status_times:
                    prev_status, first_seen = instance_status_times[instance_id]
                    if prev_status == current_status:
                        # Same status, check if it's been too long
                        if current_time - first_seen >= stuck_threshold:
                            logger.warning(
                                f"Instance {instance_id} stuck in '{current_status}' state for 5+ minutes, destroying"
                            )
                            try:
                                subprocess.run(
                                    [
                                        "uvx",
                                        "vastai",
                                        "destroy",
                                        "instance",
                                        instance_id,
                                    ],
                                    capture_output=True,
                                    timeout=30,
                                )
                                logger.info(f"Destroyed stuck instance {instance_id}")
                                # Remove from tracking
                                del instance_status_times[instance_id]
                            except Exception as e:
                                logger.error(
                                    f"Failed to destroy instance {instance_id}: {e}"
                                )
                    else:
                        # Status changed, reset tracking
                        instance_status_times[instance_id] = (
                            current_status,
                            current_time,
                        )
                else:
                    # New instance or status, start tracking
                    instance_status_times[instance_id] = (current_status, current_time)

            # Remove instances that are no longer in loading/created state
            tracked_ids = set(instance_status_times.keys())
            current_ids = set(current_statuses.keys())
            for instance_id in tracked_ids - current_ids:
                del instance_status_times[instance_id]

        except Exception as e:
            logger.error(f"Error in stuck instance monitor: {e}")

        # Wait for check_interval or until stop_event is set
        stop_event.wait(check_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Ersatz SLURM: Distribute jobs across Vast.ai GPU instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Example YAML format:
  version: "{YAML_VERSION}"
  setup: |
    /root/.local/bin/uv sync
  jobs:
    - job_id: exp1
      script: |
        /root/.local/bin/uv run python train.py --config exp1.yaml

    - job_id: exp2
      script: |
        /root/.local/bin/uv run python train.py --config exp2.yaml

The script will:
1. Start N worker threads (N = min(max_concurrent_gpus, num_jobs))
2. Each worker creates its own Vast.ai instance and pulls jobs from a queue
3. For each job: run setup, execute script, rsync results back
4. Workers exit and destroy their instances when all jobs are done

Note: version must be exactly "{YAML_VERSION}"
        """,
    )

    parser.add_argument(
        "yaml_file",
        type=Path,
        help="YAML file containing job definitions",
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository in format user/repo-name",
    )

    parser.add_argument(
        "--gpu-types",
        type=str,
        nargs="+",
        required=True,
        help="GPU types to use for jobs (e.g., RTX_5090 RTX_PRO_6000_WS)",
    )

    parser.add_argument(
        "--max-concurrent-gpus",
        type=int,
        default=4,
        help="Maximum number of concurrent GPU instances (default: 4)",
    )

    parser.add_argument(
        "--retry-interval",
        type=int,
        default=10,
        help="Seconds to wait between GPU spawn retries (default: 10)",
    )

    parser.add_argument(
        "--min-driver-version",
        type=str,
        default="575.0.0",
        help="Minimum GPU driver version (default: 575.0.0)",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Local directory to rsync results to. If not provided, rsync will be skipped.",
    )

    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket name for uploading results. If not provided, S3 uploads will be skipped.",
    )

    parser.add_argument(
        "--remote-results-dir",
        type=str,
        default=None,
        help="Path to results directory on remote instance (e.g., eval_results). Required if --s3-bucket or --results-dir is provided.",
    )

    args = parser.parse_args()

    # Parse repository
    if "/" not in args.repo:
        parser.error("--repo must be in format user/repo-name")
    repo_user, repo_name = args.repo.split("/", 1)

    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    # Validate argument dependencies
    if (args.s3_bucket or args.results_dir) and not args.remote_results_dir:
        parser.error(
            "--remote-results-dir is required when --s3-bucket or --results-dir is provided"
        )

    if args.s3_bucket:
        if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ):
            parser.error(
                "--s3-bucket requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "in .env or environment"
            )

    if not args.s3_bucket:
        logger.warning("No --s3-bucket provided, S3 uploads will be skipped")
    if not args.results_dir:
        logger.warning("No --results-dir provided, local rsync will be skipped")

    # Load jobs
    try:
        jobs, setup_command = load_jobs(args.yaml_file)
    except Exception as e:
        logger.error(f"Failed to load jobs: {e}")
        sys.exit(1)

    # Get current git commit
    try:
        git_commit_result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        git_commit = (
            git_commit_result.stdout.strip()
            if git_commit_result.returncode == 0
            else "unknown"
        )
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
        git_commit = "unknown"

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rsync_enabled = args.results_dir is not None
    if args.results_dir:
        results_base_dir = args.results_dir
    else:
        results_base_dir = Path(tempfile.mkdtemp(prefix="ersatz_slurm_"))
        logger.info(f"Using temporary directory for logs: {results_base_dir}")
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # Create instance log file for tracking created instances
    instance_log_file = results_base_dir / timestamp / "vast_instances.txt"
    instance_log_file.parent.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging
    log_file = results_base_dir / timestamp / "ersatz_slurm.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Git commit: {git_commit}")
    logger.info(f"Logs will be saved to: {results_base_dir / timestamp}")
    logger.info(f"Instance IDs will be logged to: {instance_log_file}")
    logger.info(
        f"Running {len(jobs)} jobs with up to {args.max_concurrent_gpus} concurrent GPUs"
    )
    logger.info(f"GPU types: {', '.join(args.gpu_types)}")
    if args.s3_bucket:
        logger.info(f"S3 bucket: {args.s3_bucket}")
    if args.results_dir:
        logger.info(f"Results dir: {args.results_dir}")
    if args.remote_results_dir:
        logger.info(f"Remote results dir: {args.remote_results_dir}")

    # Copy YAML file to results directory
    yaml_dest = results_base_dir / timestamp / args.yaml_file.name
    yaml_copied = False
    try:
        shutil.copy2(args.yaml_file, yaml_dest)
        logger.info(f"Copied YAML file to: {yaml_dest}")
        yaml_copied = True
    except Exception as e:
        logger.warning(f"Failed to copy YAML file: {e}")

    # Start stuck instance monitor thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=stuck_instance_monitor_thread,
        args=(stop_monitor,),
        daemon=True,
        name="StuckInstanceMonitor",
    )
    monitor_thread.start()
    logger.info("Started stuck instance monitor thread")

    # Create job queue
    job_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    # Shared state
    completed_jobs_count = [0]
    completed_jobs_lock = threading.Lock()
    failed_jobs: list[str] = []
    failed_jobs_lock = threading.Lock()
    instance_log_lock = threading.Lock()

    # Check if GitHub token is available for using gh CLI
    use_gh_for_clone = bool(os.environ.get("GITHUB_TOKEN"))
    if use_gh_for_clone:
        logger.info("GITHUB_TOKEN found, will use 'gh repo clone' for cloning")
    else:
        logger.info("No GITHUB_TOKEN found, will use 'git clone' for public repos")

    # Create worker contexts
    num_workers = min(args.max_concurrent_gpus, len(jobs))
    worker_contexts = [
        WorkerContext(
            worker_id=i + 1,
            total_workers=num_workers,
            job_queue=job_queue,
            total_jobs=len(jobs),
            completed_jobs_count=completed_jobs_count,
            completed_jobs_lock=completed_jobs_lock,
            failed_jobs=failed_jobs,
            failed_jobs_lock=failed_jobs_lock,
            results_base_dir=results_base_dir,
            timestamp=timestamp,
            setup_command=setup_command,
            gpu_types=args.gpu_types,
            min_driver_version=args.min_driver_version,
            retry_interval=args.retry_interval,
            instance_log_file=instance_log_file,
            instance_log_lock=instance_log_lock,
            s3_bucket=args.s3_bucket,
            remote_results_dir=args.remote_results_dir,
            rsync_enabled=rsync_enabled,
            repo_user=repo_user,
            repo_name=repo_name,
            use_gh_for_clone=use_gh_for_clone,
        )
        for i in range(num_workers)
    ]

    try:
        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_thread, ctx) for ctx in worker_contexts]

            # Wait for all workers to complete
            for future in futures:
                future.result()

    except KeyboardInterrupt:
        logger.error("\nExecution interrupted by user")
        stop_monitor.set()
        monitor_thread.join(timeout=5)
        sys.exit(1)

    # Stop the stuck instance monitor thread
    logger.info("Stopping stuck instance monitor thread...")
    stop_monitor.set()
    monitor_thread.join(timeout=10)

    # Upload log files to S3
    if args.s3_bucket:
        logger.info("\nUploading log files to S3...")
        s3_logs_path = f"s3://{args.s3_bucket}/{timestamp}/"
        try:
            # Upload ersatz_slurm.log
            log_upload = subprocess.run(
                ["aws", "s3", "cp", str(log_file), s3_logs_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if log_upload.returncode == 0:
                logger.info(f"Uploaded ersatz_slurm.log to {s3_logs_path}")
            else:
                logger.error(f"Failed to upload ersatz_slurm.log: {log_upload.stderr}")

            # Upload vast_instances.txt
            instances_upload = subprocess.run(
                ["aws", "s3", "cp", str(instance_log_file), s3_logs_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if instances_upload.returncode == 0:
                logger.info(f"Uploaded vast_instances.txt to {s3_logs_path}")
            else:
                logger.error(
                    f"Failed to upload vast_instances.txt: {instances_upload.stderr}"
                )

            # Upload YAML file if it was copied successfully
            if yaml_copied:
                yaml_upload = subprocess.run(
                    ["aws", "s3", "cp", str(yaml_dest), s3_logs_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if yaml_upload.returncode == 0:
                    logger.info(f"Uploaded {args.yaml_file.name} to {s3_logs_path}")
                else:
                    logger.error(f"Failed to upload YAML file: {yaml_upload.stderr}")
        except Exception as e:
            logger.error(f"Error uploading logs to S3: {e}")

    # Destroy all instances created during this run
    logger.info("\nDestroying all instances created during this run...")
    try:
        if instance_log_file.exists():
            # Get list of currently existing instances
            logger.info("Checking which instances currently exist...")
            show_result = subprocess.run(
                ["uvx", "vastai", "show", "instances"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            existing_instance_ids = set()
            if show_result.returncode == 0:
                for line in show_result.stdout.strip().split("\n")[1:]:  # Skip header
                    fields = line.split()
                    if fields and fields[0].isdigit():
                        existing_instance_ids.add(fields[0])
                logger.info(
                    f"Found {len(existing_instance_ids)} total existing instances"
                )
            else:
                logger.warning(
                    f"Failed to get existing instances: {show_result.stderr}"
                )

            # Read our instance log file
            with open(instance_log_file) as f:
                lines = f.readlines()

            our_instance_ids = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:  # timestamp (date + time) + instance_id
                    instance_id = parts[2]
                    our_instance_ids.add(instance_id)

            # Only destroy instances that both exist and are ours
            to_destroy = our_instance_ids & existing_instance_ids
            already_gone = our_instance_ids - existing_instance_ids

            if already_gone:
                logger.info(
                    f"{len(already_gone)} instances already destroyed: {', '.join(sorted(already_gone))}"
                )

            if to_destroy:
                logger.info(f"Destroying {len(to_destroy)} instances...")
                for instance_id in to_destroy:
                    try:
                        logger.info(f"Destroying instance {instance_id}...")
                        result = subprocess.run(
                            ["uvx", "vastai", "destroy", "instance", instance_id],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Successfully destroyed instance {instance_id}"
                            )
                        else:
                            logger.error(
                                f"Failed to destroy instance {instance_id}: return code {result.returncode}, stderr: {result.stderr}"
                            )
                    except Exception as e:
                        logger.error(f"Error destroying instance {instance_id}: {e}")
            else:
                logger.info("No instances to destroy")
        else:
            logger.info("No instance log file found, skipping cleanup")
    except Exception as e:
        logger.error(f"Error during instance cleanup: {e}")

    # Report failures
    if failed_jobs:
        logger.error("\n" + "=" * 80)
        logger.error("FAILED JOBS SUMMARY:")
        logger.error("=" * 80)
        for error in failed_jobs:
            logger.error(error)
        logger.error("=" * 80)
        sys.exit(1)
    else:
        logger.info("\nAll jobs completed successfully!")
        logger.info(f"Results saved to: {results_base_dir / timestamp}")


if __name__ == "__main__":
    main()
