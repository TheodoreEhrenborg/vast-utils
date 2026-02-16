# vast-utils

Scripts for automating launching Vast.ai instances and scheduling jobs on them

This is roughly imitating the interface of terraform/AWS batch. 
If you can use a high-level tool like those, do. 
See https://github.com/Beneficial-AI-Foundation/vericoding/pull/311 for an example of how to use those tools.

**Warning**: 
I extracted this from a larger repo and changed it to not depend on details of that repo, but hence those changes may have broken something. 
I've tested most but not all features since then (in particular, I haven't tested the s3 upload). 
Think of this more as a template, where you can delete parts you don't need and add parts you want.

## Requirements

Vast.ai API key in `~/.config/vastai/vast_api_key`

Putting a `HF_TOKEN` in `.env` will allow the script to preload gemma on the instance. 
Putting `GITHUB_TOKEN` in `.env` will allow the script to clone private repos.
Putting `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env` will allow the script to upload to s3 
(you'll have to make the bucket first).

## Examples


### Setup command
- Make a 4090
- Takes about 2 minutes
``` sh
uv run src/vast-utils/setup_vast.py --gpu RTX_4090
```

### Ersatz slurm command
- This script runs all the jobs in a yaml
  - See `tests/test_jobs.yaml` for an example
- It creates a pool of instances, destroys them at the end, and uses heuristics to detect if an instance has a hardware problem
- This example takes about 8 minutes
  - It tries the cheaper GPUs first
``` sh
uv run src/vast-utils/ersatz_slurm.py tests/test_jobs.yaml --gpu-types RTX_4090 RTX_5090 RTX_PRO_6000_WS H200 H200_NVL --max-concurrent-gpus 2 --repo TheodoreEhrenborg/pivotal_sae --results-dir /tmp/pivotal_talk/ --remote-results-dir eval_results
```
