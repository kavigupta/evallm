import codecs
import io
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List
from uuid import uuid4

import tqdm.auto as tqdm
from openai import OpenAI
from permacache import permacache

sketch5_client = OpenAI(
    api_key="EMPTY",
    base_url="http://sketch5.csail.mit.edu:52372/v1",
)


def openai_key():
    with open("/mnt/md0/.openaikey") as f:
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
    "meta-llama/Meta-Llama-3-8B": ModelSpec(client=sketch5_client, is_chat=False),
    "gpt-3.5-turbo-0125": ModelSpec(client=openai_client, is_chat=True),
}


def create_batch_request_line(request_id, model, prompt, **kwargs):
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": role,
                    "content": content,
                }
                for role, content in prompt.items()
            ],
        },
        **kwargs,
    }


def create_batch_request(model, prompts, **kwargs):
    ids = [uuid4().hex for _ in prompts]
    jsonl = [
        create_batch_request_line(request_id, model, prompt, **kwargs)
        for request_id, prompt in zip(ids, prompts)
    ]
    f = io.BytesIO()
    f_string = codecs.getwriter("utf-8")(f)
    for line in jsonl:
        f_string.write(json.dumps(line) + "\n")
    f.seek(0)
    return ids, f


def execute_batch_request(model, prompts, pbar, **kwargs):
    ids, jsonl = create_batch_request(model, prompts, **kwargs)
    batch_input_file = openai_client.files.create(file=jsonl, purpose="batch")
    batch = openai_client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    if pbar is not None:
        pbar = pbar(total=len(ids))
    while True:
        batch_status = openai_client.batches.retrieve(batch.id)
        if batch_status.status == "completed":
            if pbar is not None:
                pbar.close()
            break
        if pbar is not None:
            pbar.update(
                batch_status.request_counts.completed
                + batch_status.request_counts.failed
                - pbar.n
            )
        time.sleep(1)
    if batch_status.error_file_id is not None:
        print(openai_client.files.content(batch_status.error_file_id).text)
    output = openai_client.files.content(batch_status.output_file_id).text
    output = [json.loads(x) for x in output.split("\n") if x]
    id_to_output = {x["custom_id"]: x for x in output}
    return [id_to_output[id] for id in ids]


@permacache("evallm/llm/llm/run_prompt")
def run_prompt(model: str, prompt: List[str], kwargs: dict):
    assert isinstance(prompt, (list, tuple))
    client = model_specs[model].client
    if model_specs[model].is_chat:
        assert client == openai_client
        output = execute_batch_request(model, prompt, tqdm.tqdm, **kwargs)
        choices = []
        for x in output:
            choices += x["response"]["body"]["choices"]
        return SimpleNamespace(choices=choices)

    completion = client.completions.create(
        model=model,
        prompt=[p.strip() for p in prompt],
        **kwargs,
    )
    return completion
