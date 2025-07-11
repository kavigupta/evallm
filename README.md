
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

## Running the benchmark on your own model.

### The simple way

Follow the pattern established in `scripts/lib_example.py` to run the benchmark on your own model.

This allows you to run the benchmark on your own model without needing to modify any code, and you can use any technique you want to take a prompt string and get a response string.

The main downside to this approach is it does not integrate into any of the rest of the analyses in the codebase.

### The way that integrates with the rest of the codebase

**Note**: only do this if you want to add your model to the tables and graphs as generated by the codebase. This is probably not what you want if you just want to run the benchmark on your own model without modifying the codebase.

These instructions assume you have a model that is compatible with the `vllm` library and can be run as an API server. If you want to benchmark your own model, you can follow these steps:

First, follow the installation instructions above. Then, do the following:

1. Add your model to `evallm/experiments/models_display.py`. The key should be a short name for your model, and the value should be the internal name that VLLM uses to refer to the model.
2. Add your model to `evallm/llm/llm.py`. The key should be the internal name and the value should be a dictionary specifying the client (you should specify this  in `~/.local-vllm-server`), and whether the model is a chat model or not.
3. Update `metadata_for_models` in `evallm/experiments/main_tables.py` to include your model's metadata. This is used to display the model in the graphs and tables. Also, update `evallm/experiments/transducer_summary.py` and `evallm/experiments/sequence_completion_summary.py` to include your model. You can now re-run the notebooks to see your model in the graphs and tables.
