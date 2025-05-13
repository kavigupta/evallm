import os

model_by_display_key = {
    "llama3-8B": "meta-llama/Meta-Llama-3-8B",
    "llama3-70B": "meta-llama/Meta-Llama-3-70B",
    "llama3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-nemo-minitron-8B": "nvidia/Mistral-NeMo-Minitron-8B-Base",
    "mistral-nemo-base-12B": "mistral_models/Nemo-Base",
    "mistral-nemo-instruct-12B": "mistral_models/Nemo-Instruct",
    "starcoder2-15b": "bigcode/starcoder2-15b",
    "codestral-22B": "mistralai/Codestral-22B-v0.1",
    "deepseek-coder-33b-instruct": "deepseek-ai/deepseek-coder-33b-instruct",
    "qwen-2.5-coder-7B": "Qwen/Qwen2.5-Coder-7B",
    "qwen-2.5-coder-instruct-7B": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen-2.5-coder-instruct-32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "gemma-7b": "google/gemma-7b",
    "falcon-7b": "tiiuae/falcon-7b",
    "gpt-3.5-instruct": "gpt-3.5-turbo-instruct",
    "gpt-3.5-chat": "gpt-3.5-turbo-0125",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-05-13",
    # "o1-preview": "o1-preview-2024-09-12",
    "o3-mini": "o3-mini-2025-01-31",
    "claude-3.5": "claude-3-5-sonnet-20241022",
}

in_homedir = {"mistral_models/Nemo-Base", "mistral_models/Nemo-Instruct"}


def full_path(path):
    if path in in_homedir:
        path = os.path.expanduser(f"~/{path}")
    return path
