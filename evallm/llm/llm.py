import functools
import multiprocessing
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import tqdm.auto as tqdm
from anthropic import Anthropic, InternalServerError
from openai import OpenAI, RateLimitError
from permacache import permacache

KEY_DIR = "/mnt/md0"
if not os.path.exists(KEY_DIR):
    KEY_DIR = os.path.expanduser("~")


def key(path):
    openai_key_path = f"{KEY_DIR}/{path}"
    if not os.path.exists(openai_key_path):
        return "EMPTY"
    with open(openai_key_path) as f:
        return f.read().strip()


def openai_key():
    path = ".openaikey"
    return key(path)


def anthropic_key():
    path = ".anthropickey"
    return key(path)


openai_client = OpenAI(
    api_key=openai_key(),
    base_url="https://api.openai.com/v1",
)

anthropic_client = Anthropic(
    api_key=anthropic_key(),
    # base_url="https://api.anthropic.com/v1",
)

local_client = OpenAI(
    api_key="EMPTY",
    base_url=key(".local-vllm-server"),
)


@dataclass
class ModelSpec:
    client: OpenAI
    is_chat: bool


model_specs = {
    "none": ModelSpec(client=None, is_chat=False),
    # open source completion models
    "meta-llama/Meta-Llama-3-8B": ModelSpec(client=local_client, is_chat=False),
    "meta-llama/Meta-Llama-3-70B": ModelSpec(client=local_client, is_chat=False),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(client=local_client, is_chat=False),
    "meta-llama/Meta-Llama-3.1-70B": ModelSpec(client=local_client, is_chat=False),
    "meta-llama/Llama-3.1-8B": ModelSpec(client=local_client, is_chat=False),
    "nvidia/Mistral-NeMo-Minitron-8B-Base": ModelSpec(
        client=local_client, is_chat=False
    ),
    os.path.expanduser("~/mistral_models/Nemo-Instruct"): ModelSpec(
        client=local_client, is_chat=False
    ),
    os.path.expanduser("~/mistral_models/Nemo-Base"): ModelSpec(
        client=local_client, is_chat=False
    ),
    "google/gemma-7b": ModelSpec(client=local_client, is_chat=False),
    "tiiuae/falcon-7b": ModelSpec(client=local_client, is_chat=False),
    # open source code models
    "bigcode/starcoder2-15b": ModelSpec(client=local_client, is_chat=False),
    "mistralai/Codestral-22B-v0.1": ModelSpec(client=local_client, is_chat=False),
    "deepseek-ai/deepseek-coder-33b-instruct": ModelSpec(
        client=local_client, is_chat=False
    ),
    "Qwen/Qwen2.5-Coder-7B": ModelSpec(client=local_client, is_chat=False),
    "Qwen/Qwen2.5-Coder-7B-Instruct": ModelSpec(client=local_client, is_chat=False),
    "Qwen/Qwen2.5-7B": ModelSpec(client=local_client, is_chat=False),
    "Qwen/Qwen2.5-32B": ModelSpec(client=local_client, is_chat=False),
    "Qwen/Qwen2.5-Coder-32B-Instruct": ModelSpec(client=local_client, is_chat=False),
    # openai models
    "gpt-3.5-turbo-instruct": ModelSpec(client=openai_client, is_chat=False),
    "gpt-3.5-turbo-0125": ModelSpec(client=openai_client, is_chat=True),
    "gpt-4o-mini-2024-07-18": ModelSpec(client=openai_client, is_chat=True),
    "gpt-4o-2024-05-13": ModelSpec(client=openai_client, is_chat=True),
    "o1-preview-2024-09-12": ModelSpec(client=openai_client, is_chat=True),
    "o3-mini-2025-01-31": ModelSpec(client=openai_client, is_chat=True),
    # anthropic models
    "claude-3-5-sonnet-20241022": ModelSpec(client=anthropic_client, is_chat=True),
}


def anthropic_create(*, messages, **kwargs):
    filtered_messages = []
    for message in messages:
        if message["role"] == "system":
            assert not message["content"]
            continue
        filtered_messages.append(message)
    while True:
        try:
            message = anthropic_client.messages.create(
                messages=filtered_messages, **kwargs
            )
            break
        except InternalServerError as e:
            if "overloaded_error" in e.message:
                print(e)
                time.sleep(10)
                continue
            raise
    message_text = "".join([x.text for x in message.content if hasattr(x, "text")])
    message = SimpleNamespace(content=message_text)
    return [SimpleNamespace(message=message)]


def get_create_method(model):
    client = model_specs[model].client
    if client == openai_client:
        return lambda *args, **kwargs: client.chat.completions.create(
            *args, **kwargs
        ).choices
    if client == anthropic_client:
        return anthropic_create
    raise NotImplementedError(f"Model {model} does not support chat")


def to_messages(prompt):
    return [
        {
            "role": role,
            "content": content,
        }
        for role, content in prompt.items()
    ]


MAX_CHUNKSIZE = 100


def run_prompt_chunked(model: str, prompt: List[str], kwargs: dict):
    choices = []
    for i in tqdm.trange(0, len(prompt), MAX_CHUNKSIZE):
        print(i)
        x = run_prompt(model, prompt[i : i + MAX_CHUNKSIZE], kwargs)
        choices += x.choices
    return SimpleNamespace(choices=choices)


@permacache("evallm/llm/llm/run_prompt_2", multiprocess_safe=False)
def run_prompt(model: str, prompt: List[str], kwargs: dict):
    if len(prompt) > MAX_CHUNKSIZE:
        return run_prompt_chunked(model, prompt, kwargs)
    num_parallel = 200
    if model == "gpt-3.5-turbo-instruct":
        num_parallel = 10
    if model == "claude-3-5-sonnet-20241022":
        # anthropic has extremely low rate limits
        num_parallel = 1
    if model.startswith("o1"):
        # expensive so lets not churn through $20 instantly
        num_parallel = 1
    assert isinstance(prompt, (list, tuple))
    client = model_specs[model].client
    if model_specs[model].is_chat:
        assert client in (openai_client, anthropic_client)
        with multiprocessing.Pool(num_parallel) as p:
            map_fn = (
                p.map
                if num_parallel > 1
                else (
                    (lambda fn, xs: map(fn, tqdm.tqdm(xs))) if len(prompt) > 1 else map
                )
            )
            choices_each = map_fn(
                functools.partial(create_openai_completion, model, kwargs),
                prompt,
            )
            choices_each = list(choices_each)
        choices = []
        for x in choices_each:
            choices += x
        return SimpleNamespace(choices=choices)

    chunk_size = num_parallel
    choices = []
    for start in range(0, len(prompt), chunk_size):
        chunk = prompt[start : start + chunk_size]
        while True:
            try:
                from_chunk = client.completions.create(
                    model=model,
                    prompt=chunk,
                    **kwargs,
                ).choices
                break
            except RateLimitError:
                time.sleep(1)
        choices += from_chunk
    return SimpleNamespace(choices=choices)


def create_openai_completion(model, kwargs, prompt):
    if model == "gpt-4o-2024-05-13" and kwargs["max_tokens"] == 5000:
        kwargs = kwargs.copy()
        kwargs["max_tokens"] = 4096
    if model.startswith("o1") or model.startswith("o3"):
        assert set(prompt.keys()) == {"system", "user"}
        prompt = {"user": prompt["user"]}
        kwargs = kwargs.copy()
        del kwargs["max_tokens"]  # this causes issues
        del kwargs["temperature"]  # non-default temperature is not supported
    create = get_create_method(model)
    return create(model=model, messages=to_messages(prompt), **kwargs)
