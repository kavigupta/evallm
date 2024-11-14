# Following https://sebastian-petrus.medium.com/how-to-run-mistral-nemo-12b-locally-step-by-step-002fc35dbaa1
# adapted slightly
# You will need to run `pip install vllm` then run the command
# python -m vllm.entrypoints.openai.api_server --model /scratch/kavig/mistral_models/Nemo-Instruct --port 52372 --max-model-len 4096
# or
# python -m vllm.entrypoints.openai.api_server --model /scratch/kavig/mistral_models/Nemo-Base --port 52372 --max-model-len 4096

from pathlib import Path

from huggingface_hub import snapshot_download

model_path = Path.home().joinpath("mistral_models", "Nemo-Instruct")
model_path.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    # allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
    local_dir=model_path,
)

model_path = Path.home().joinpath("mistral_models", "Nemo-Base")
model_path.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="mistralai/Mistral-Nemo-Base-2407",
    # allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
    local_dir=model_path,
)
