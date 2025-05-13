
## Installation

```bash
pip install -e .
```

## Reproducing the graphs

All the graphs should be reproducible by running the notebooks under `notebooks/`. They will pull from saved intermediate files in `cache/`.

## Reproducing the experiments

First, delete `cache/`. This will force the code to recompute all intermediate files. Then, run the notebooks under `notebooks/`. The script is expecting the following files:

```
~/.openaikey: the OpenAI API key
~/.anthropickey: the Anthropic API key
~/.local-vllm-server: the local-vllm server's URL, ending in /v1
```

To actually run the local models, you need to run vllm servers for them. All except nemo are run by running vllm

```bash
python -m vllm.entrypoints.openai.api_server --model $MODEL --port $PORT --tensor-parallel-size 4
```

where MODEL is just the model path. It should download the model automatically; though you might need to gain access to the model first.

For nemo specifically, you should run `python scripts/nemo.py` to download the files, then run the above command but with the full path to the model.
