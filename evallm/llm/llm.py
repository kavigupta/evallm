from dataclasses import dataclass
from typing import List

from openai import OpenAI
from permacache import permacache

sketch5_client = OpenAI(
    api_key="EMPTY",
    base_url="http://sketch5.csail.mit.edu:52372/v1",
)


@dataclass
class ModelSpec:
    client: OpenAI
    is_chat: bool


model_specs = {
    "meta-llama/Meta-Llama-3-8B": ModelSpec(client=sketch5_client, is_chat=False),
}


@permacache("evallm/llm/llm/run_prompt", multiprocess_safe=True)
def run_prompt(model: str, prompt: List[str], kwargs: dict):
    assert isinstance(prompt, (list, tuple))
    client = model_specs[model].client
    if model_specs[model].is_chat:
        raise NotImplementedError("Chat models are not supported.")

    completion = client.completions.create(
        model=model,
        prompt=[p.strip() for p in prompt],
        **kwargs,
    )
    return completion
