import functools
import multiprocessing
import os
from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import List

from openai import OpenAI, RateLimitError
from permacache import permacache

sketch5_client = OpenAI(
    api_key="EMPTY",
    base_url="http://sketch5.csail.mit.edu:52372/v1",
)


def openai_key():
    openai_key_path = "/mnt/md0/.openaikey"
    if not os.path.exists(openai_key_path):
        return "EMPTY"
    with open(openai_key_path) as f:
        return f.read().strip()


openai_client = OpenAI(
    api_key=openai_key(),
    base_url="https://api.openai.com/v1",
)


@dataclass
class ModelSpec:
    client: OpenAI
    is_chat: bool


model_specs = {
    "none": ModelSpec(client=None, is_chat=False),
    "meta-llama/Meta-Llama-3-8B": ModelSpec(client=sketch5_client, is_chat=False),
    "gpt-3.5-turbo-instruct": ModelSpec(client=openai_client, is_chat=False),
    "gpt-3.5-turbo-0125": ModelSpec(client=openai_client, is_chat=True),
    "gpt-4o-mini-2024-07-18": ModelSpec(client=openai_client, is_chat=True),
    "gpt-4o-2024-05-13": ModelSpec(client=openai_client, is_chat=True),
}


def to_messages(prompt):
    return [
        {
            "role": role,
            "content": content,
        }
        for role, content in prompt.items()
    ]


@permacache("evallm/llm/llm/run_prompt_2", multiprocess_safe=True)
def run_prompt(model: str, prompt: List[str], kwargs: dict):
    num_parallel = 200 if model != "gpt-3.5-turbo-instruct" else 10
    assert isinstance(prompt, (list, tuple))
    client = model_specs[model].client
    if model_specs[model].is_chat:
        assert client == openai_client
        with multiprocessing.Pool(num_parallel) as p:
            choices_each = p.map(
                functools.partial(create_openai_completion, model, kwargs),
                prompt,
            )
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
    return openai_client.chat.completions.create(
        model=model, messages=to_messages(prompt), **kwargs
    ).choices
