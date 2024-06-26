from typing import List

from openai import OpenAI
from permacache import permacache

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://sketch5.csail.mit.edu:52372/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


@permacache("evallm/llm/llm/run_prompt")
def run_prompt(model: str, prompt: List[str], kwargs: dict):
    assert isinstance(prompt, (list, tuple))
    completion = client.completions.create(
        model=model,
        prompt=[p.strip() for p in prompt],
        **kwargs,
    )
    return completion
