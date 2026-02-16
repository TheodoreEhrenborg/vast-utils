#!/usr/bin/env python3
"""Python version of setup_vast.sh for setting up Vast.ai instances."""

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class VastInstanceResult:
    """Result of creating a Vast.ai instance."""

    ssh_config_name: str
    instance_id: str


class NoInstancesAvailableError(Exception):
    """Raised when no instances matching the criteria are available."""

    pass


class VastNonZeroExit(Exception):
    """Raised when a command fails before instance creation."""

    def __init__(
        self, message: str, cmd: list[str], returncode: int, stdout: str, stderr: str
    ):
        super().__init__(message)
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class VastInstanceSetupError(Exception):
    """Raised when instance setup fails after creation (includes instance_id for cleanup)."""

    def __init__(self, message: str, instance_id: str):
        super().__init__(message)
        self.instance_id = instance_id


def run_command(
    cmd: list[str],
    instance_id: str = None,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command and return the result.

    Args:
        cmd: Command to run.
        instance_id: Instance ID for error reporting
        capture_output: Whether to capture stdout/stderr.
        check: Whether to check return code.

    Returns:
        CompletedProcess result.

    Raises:
        VastNonZeroExit: If check=True, instance_id is empty, and command fails.
        VastInstanceSetupError: If check=True, instance_id is provided, and command fails.
    """
    try:
        return subprocess.run(
            cmd, capture_output=capture_output, text=True, check=check
        )
    except subprocess.CalledProcessError as e:
        if instance_id:
            raise VastInstanceSetupError(
                f"Command failed: {' '.join(cmd)}\nReturn code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}",
                instance_id,
            )
        else:
            raise VastNonZeroExit(
                f"Command failed: {' '.join(cmd)}\nReturn code: {e.returncode}",
                cmd=cmd,
                returncode=e.returncode,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            )


def ssh_retry(cmd: list[str], instance_id: str, max_attempts: int = 5) -> None:
    """Retry SSH commands up to max_attempts times."""
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout, end="")
            return
        print(f"SSH command failed (attempt {attempt}/{max_attempts})")
        print(f"  Return code: {result.returncode}")
        if result.stdout:
            print(f"  stdout: {result.stdout}")
        if result.stderr:
            print(f"  stderr: {result.stderr}")
        if attempt < max_attempts:
            print("  Retrying...")
            time.sleep(2)
    raise VastInstanceSetupError(
        f"SSH command failed after {max_attempts} attempts: {' '.join(cmd)}",
        instance_id,
    )


def create_vast_instance(
    gpu_type: str | None = None,
    instance_id: str | None = None,
    repo_user: str | None = None,
    repo_name: str | None = None,
    min_driver_version: str = "575.0.0",
    full_ssh_name: bool = False,
    instance_log_file: Path | None = None,
    instance_log_lock=None,
) -> VastInstanceResult:
    """
    Create and setup a Vast.ai instance.

    Args:
        gpu_type: GPU type to search for (e.g., 'RTX_5090'). Mutually exclusive with instance_id.
        instance_id: Specific instance ID (8 digits). Mutually exclusive with gpu_type.
        repo_user: GitHub username for repository to clone.
        repo_name: GitHub repository name to clone.
        min_driver_version: Minimum GPU driver version required.
        full_ssh_name: If True, use full instance ID in SSH config name (vastXYZ..W).
                      If False, use last 2 digits (vastXY). Default: False.
        instance_log_file: Path to file for logging created instance IDs.
        instance_log_lock: Lock for thread-safe writes to instance_log_file.

    Returns:
        VastInstanceResult with ssh_config_name and instance_id.

    Raises:
        NoInstancesAvailableError: If no instances matching the criteria are available.
        RuntimeError: For other errors during instance creation.
        FileNotFoundError: If .env file is missing.
    """
    if not gpu_type and not instance_id:
        raise ValueError("Either gpu_type or instance_id must be provided")
    if gpu_type and instance_id:
        raise ValueError("Cannot specify both gpu_type and instance_id")

    print("Starting Vast.ai instance setup...")

    # Determine instance search ID
    if instance_id:
        instance_search_id = instance_id
        print(f"Using instance ID: {instance_search_id}")
    else:
        print(f"Searching for cheapest {gpu_type} instance...")

        result = run_command(
            [
                "uvx",
                "vastai",
                "search",
                "offers",
                f"reliability > 0.99 num_gpus=1 gpu_name={gpu_type} rented=False cpu_ram>200 driver_version >= {min_driver_version}",
                "-o",
                "dph-",
            ],
        )

        search_result = result.stdout
        # Get the last instance ID (cheapest)
        instance_search_id = None
        for line in search_result.strip().split("\n")[1:]:  # Skip header
            fields = line.split()
            if fields and fields[0].isdigit():
                instance_search_id = fields[0]

        if not instance_search_id:
            raise NoInstancesAvailableError(
                f"No instances found for GPU type {gpu_type}"
            )

        print(f"Selected cheapest instance: {instance_search_id}")

    # Load environment variables if .env exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
    else:
        log.info(".env file not found, continuing without environment variables")

    # Create instance
    print("Creating instance...")
    result = run_command(
        [
            "uvx",
            "vastai",
            "create",
            "instance",
            instance_search_id,
            "--image",
            "theodoreehrenborg/ubuntu-uv",
            "--disk",
            "256",
            "--ssh",
            "--direct",
            "--env",
            "-p 8501:8501",
            "--raw",
        ],
    )

    try:
        create_json = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print("Error: Failed to parse create instance response as JSON")
        print(f"JSONDecodeError: {e}")
        print(f"Command output (stdout): {repr(result.stdout)}")
        print(f"Command output (stderr): {repr(result.stderr)}")
        raise NoInstancesAvailableError(
            f"Failed to parse create instance response: {e}"
        )

    if not create_json.get("success"):
        raise NoInstancesAvailableError(
            f"Failed to create instance (likely already claimed): {json.dumps(create_json, indent=2)}"
        )

    created_instance_id = str(create_json["new_contract"])
    print(f"Instance created: {created_instance_id}")

    # Record instance ID immediately (thread-safe with lock)
    if instance_log_file and instance_log_lock:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with instance_log_lock:
            with open(instance_log_file, "a") as f:
                f.write(f"{timestamp} {created_instance_id}\n")

    if full_ssh_name:
        ssh_config_name = f"vast{created_instance_id}"
    else:
        ssh_config_name = f"vast{created_instance_id[-2:]}"
    print(f"SSH config name: {ssh_config_name}")

    # Wait for instance to be running (doubled wait time)
    print("Waiting for instance to start...")
    for i in range(1, 61):
        result = run_command(
            ["uvx", "vastai", "show", "instance", created_instance_id, "--raw"],
            instance_id=created_instance_id,
        )
        try:
            instance_json = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"\nWarning: Failed to parse instance status (attempt {i}/60)")
            print(f"JSONDecodeError: {e}")
            print(f"Command output (stdout): {repr(result.stdout)}")
            if i == 60:
                raise VastInstanceSetupError(
                    f"Failed to parse instance status after 60 attempts: {e}",
                    created_instance_id,
                )
            print(".", end="", flush=True)
            time.sleep(2)
            continue

        status = instance_json.get("actual_status")

        if status == "running":
            print("\nInstance running")
            break

        print(".", end="", flush=True)
        time.sleep(2)

        if i == 60:
            raise VastInstanceSetupError(
                "Timeout waiting for instance to start", created_instance_id
            )

    # Attach SSH key
    print("Attaching SSH key...")
    ssh_pub_key = Path.home() / ".ssh" / "id_ed25519.pub"
    with open(ssh_pub_key) as f:
        pub_key = f.read().strip()

    run_command(
        ["uvx", "vastai", "attach", "ssh", created_instance_id, pub_key],
        instance_id=created_instance_id,
    )

    # Get instance details
    result = run_command(
        ["uvx", "vastai", "show", "instance", created_instance_id, "--raw"],
        instance_id=created_instance_id,
    )
    try:
        instance_json = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print("Error: Failed to parse instance details as JSON")
        print(f"JSONDecodeError: {e}")
        print(f"Command output (stdout): {repr(result.stdout)}")
        print(f"Command output (stderr): {repr(result.stderr)}")
        raise VastInstanceSetupError(
            f"Failed to parse instance details: {e}", created_instance_id
        )

    try:
        ssh_host = instance_json["public_ipaddr"]
        ssh_port = instance_json["ports"]["22/tcp"][0]["HostPort"]
        streamlit_port = instance_json["ports"]["8501/tcp"][0]["HostPort"]
    except KeyError as e:
        print(f"Error: Missing expected key in instance details: {e}")
        print(f"Instance JSON: {json.dumps(instance_json, indent=2)}")
        raise VastInstanceSetupError(
            f"Failed to extract instance connection details, missing key: {e}",
            created_instance_id,
        )

    print(f"SSH: {ssh_host}:{ssh_port}")

    # Add to SSH config
    ssh_config_path = Path.home() / ".ssh" / "config"

    # Remove existing entry if present
    if ssh_config_path.exists():
        with open(ssh_config_path) as f:
            lines = f.readlines()

        with open(ssh_config_path, "w") as f:
            skip = False
            for line in lines:
                if line.strip() == f"Host {ssh_config_name}":
                    skip = True
                elif skip and line.strip() == "":
                    skip = False
                    continue
                elif not skip:
                    f.write(line)

    # Append new entry
    with open(ssh_config_path, "a") as f:
        f.write(f"\n#   {ssh_host}:{streamlit_port} -> 8501/tcp\n")
        f.write(f"Host {ssh_config_name}\n")
        f.write(f"    HostName {ssh_host}\n")
        f.write("    User root\n")
        f.write(f"    Port {ssh_port}\n")

    # Wait for SSH (doubled wait time)
    print("Waiting for SSH...")
    for i in range(1, 61):
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "StrictHostKeyChecking=no",
                ssh_config_name,
                "exit",
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            break
        print(".", end="", flush=True)
        time.sleep(2)
    print()

    # Health checks
    print("Health checks...")
    health_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        ssh_config_name,
        "df -h . | grep -v Filesystem; free -h | grep Mem; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader",
    ]
    ssh_retry(health_cmd, created_instance_id)

    # Install gh, python3-dev, gcc (needed for vllm), and unzip
    print("Installing gh, python3-dev, gcc, and unzip...")
    install_packages_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        ssh_config_name,
        "apt-get update -qq && apt-get install -y -qq gh python3-dev gcc unzip > /dev/null 2>&1",
    ]
    ssh_retry(install_packages_cmd, created_instance_id)

    # Install AWS CLI v2
    print("Installing AWS CLI v2...")
    install_awscli_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        ssh_config_name,
        'curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip -q awscliv2.zip && ./aws/install > /dev/null 2>&1 && rm -rf awscliv2.zip aws',
    ]
    ssh_retry(install_awscli_cmd, created_instance_id)

    # Fix libcuda.so symlink (needed for vllm torch.compile)
    print("Creating libcuda.so symlink...")
    fix_cuda_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        ssh_config_name,
        "cd /usr/lib/x86_64-linux-gnu && ln -sf libcuda.so.1 libcuda.so",
    ]
    ssh_retry(fix_cuda_cmd, created_instance_id)

    # Configure AWS credentials
    print("Configuring AWS credentials...")
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if aws_access_key and aws_secret_key:
        # Create .aws directory
        mkdir_aws_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            "mkdir -p ~/.aws",
        ]
        ssh_retry(mkdir_aws_cmd, created_instance_id)

        # Write credentials file
        aws_creds_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            f"cat > ~/.aws/credentials << EOF\n[default]\naws_access_key_id = {aws_access_key}\naws_secret_access_key = {aws_secret_key}\nEOF",
        ]
        ssh_retry(aws_creds_cmd, created_instance_id)
    else:
        print(
            "Warning: AWS credentials not found in environment, skipping AWS configuration"
        )

    # Authenticate with GitHub
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        print("Authenticating with GitHub...")
        gh_auth_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            f'echo "{github_token}" | gh auth login --with-token',
        ]
        ssh_retry(gh_auth_cmd, created_instance_id)
    else:
        log.info("No GITHUB_TOKEN found, skipping GitHub authentication (assuming public repo)")

    # Clone repository
    if repo_user and repo_name:
        print(f"Cloning repository {repo_user}/{repo_name}...")
        # Use gh if authenticated, otherwise use git clone for public repos
        if github_token:
            clone_command = f"gh repo clone {repo_user}/{repo_name}"
        else:
            clone_command = f"git clone https://github.com/{repo_user}/{repo_name}.git"

        clone_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            clone_command,
        ]
        ssh_retry(clone_cmd, created_instance_id)
    else:
        print("No repository specified, skipping clone...")

    # Authenticate with HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace...")
        hf_auth_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            f'/root/.local/bin/uvx hf auth login --token "{hf_token}"',
        ]
        ssh_retry(hf_auth_cmd, created_instance_id)
    else:
        log.info("No HF_TOKEN found, skipping HuggingFace authentication")

    # Install dependencies
    if repo_name:
        print("Installing dependencies...")
        install_deps_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            f"cd {repo_name} && /root/.local/bin/uv sync",
        ]
        ssh_retry(install_deps_cmd, created_instance_id)
    else:
        print("No repository specified, skipping dependency installation...")

    # Load gemma
    hf_token = os.environ.get("HF_TOKEN")
    if repo_name and hf_token:
        print("Load gemma")
        scp_cmd = ["scp", "scripts/load_gemma.py", f"{ssh_config_name}:"]
        ssh_retry(scp_cmd, created_instance_id)

        load_gemma_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            "/root/.local/bin/uv run --with transformers --with torch load_gemma.py",
        ]
        ssh_retry(load_gemma_cmd, created_instance_id)
    elif repo_name and not hf_token:
        log.info("No HF_TOKEN found, skipping load_gemma step")
    else:
        log.info("No repository specified, skipping load_gemma...")


    # Copy pingme script if it exists locally
    pingme_path = Path.home() / ".local" / "bin" / "pingme"
    if pingme_path.exists():
        print("Copying pingme script...")
        # Create .local/bin directory on remote if needed
        mkdir_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_config_name,
            "mkdir -p ~/.local/bin",
        ]
        ssh_retry(mkdir_cmd, created_instance_id)

        # Copy pingme script
        copy_pingme_cmd = ["scp", str(pingme_path), f"{ssh_config_name}:.local/bin/"]
        ssh_retry(copy_pingme_cmd, created_instance_id)
    else:
        print("No local pingme script found, skipping...")

    print()
    print("Setup complete!")
    print(f"Connect: ssh {ssh_config_name}")
    print(f"Instance ID: {created_instance_id}")
    print(f"Once you start streamlit, view it at http://{ssh_host}:{streamlit_port}")
    print()
    print(f"To stop: uvx vastai stop instance {created_instance_id}")
    print(f"To restart: uvx vastai start instance {created_instance_id}")

    return VastInstanceResult(
        ssh_config_name=ssh_config_name, instance_id=created_instance_id
    )


def main():
    parser = argparse.ArgumentParser(
        description="Setup Vast.ai instances for GPU computing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_vast.py --gpu RTX_5090 --repo user/repo-name
  python setup_vast.py --gpu RTX_PRO_6000_WS --repo user/repo-name
  python setup_vast.py --id 28396515 --repo user/repo-name

Suggested GPU types: RTX_5090, RTX_PRO_6000_WS
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gpu", type=str, help="GPU type to search for (e.g., RTX_5090)"
    )
    group.add_argument("--id", type=str, help="Specific instance ID (8 digits)")

    parser.add_argument(
        "--repo",
        type=str,
        help="GitHub repository in format user/repo-name",
    )

    parser.add_argument(
        "--min-driver-version",
        type=str,
        default="575.0.0",
        help="Minimum GPU driver version (default: 575.0.0)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full instance ID in SSH config name (vastXYZ..W) instead of short form (vastXY)",
    )

    args = parser.parse_args()

    # Parse repository if provided
    repo_user = None
    repo_name = None
    if args.repo:
        if "/" not in args.repo:
            print("Error: --repo must be in format user/repo-name")
            exit(1)
        repo_user, repo_name = args.repo.split("/", 1)

    try:
        create_vast_instance(
            gpu_type=args.gpu,
            instance_id=args.id,
            repo_user=repo_user,
            repo_name=repo_name,
            min_driver_version=args.min_driver_version,
            full_ssh_name=args.full,
        )
    except NoInstancesAvailableError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
