from dataclasses import dataclass
from typing import List

from openai import OpenAI
from permacache import permacache

sketch5_client = OpenAI(
    api_key="EMPTY",
    base_url="http://sketch5.csail.mit.edu:52372/v1",
)
openai_client = OpenAI(
    api_key=open("/mnt/md0/.openaikey").read().strip(),
    base_url="https://api.openai.com/v1",
)


@dataclass
class ModelSpec:
    client: OpenAI
    is_chat: bool


model_specs = {
    "meta-llama/Meta-Llama-3-8B": ModelSpec(client=sketch5_client, is_chat=False),
    "gpt-3.5-turbo-0125": ModelSpec(client=openai_client, is_chat=True),
}


@permacache("evallm/llm/llm/run_prompt")
def run_prompt(model: str, prompt: List[str], kwargs: dict):
    assert isinstance(prompt, (list, tuple))
    client = model_specs[model].client
    completion = client.completions.create(
        model=model,
        prompt=[p.strip() for p in prompt],
        **kwargs,
    )
    return completion
