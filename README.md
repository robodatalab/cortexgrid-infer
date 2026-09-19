# cortexgrid-infer

The model-specific pieces that [cortexgrid](https://github.com/robodatalab/cortexgrid)
needs but cannot work out for itself, plus the clients that talk to what it deploys.

cortexgrid stores whatever weights it is handed under whatever key it is given, and
serves them with whatever class it was told to bundle. It deliberately knows nothing
about where a model came from or what shape it is. This library supplies exactly that
missing knowledge for one family of models at a time, and drives no lifecycle of its
own: importing, deploying and deleting stay your calls against the cortexgrid SDK.

## Division of labour

| | Owns |
|---|---|
| **cortexgrid** | Registry identity, weights, serve bundle, hardware placement, deployment, jobs. Never knows what kind of model it holds. |
| **cortexgrid-infer** | Per family: how to name it, fetch it, run it, size it, talk to it. Never drives a lifecycle. |
| **you** | Policy: which model, which requirements, when to import, deploy and tear down. |

An **importer** is the whole of this library's cluster-side surface. It gives you what
`cortexgrid.import_model` and `cortexgrid.deploy_model` need:

| | |
|---|---|
| `.family` / `.suffix` | how this model id names itself in the registry |
| `.serve_app` | the class cortexgrid bundles and runs on the cluster |
| `.requirements()` | what one replica needs to be placed and to run |
| `.config()` | settings the serve app reads at construction, if it needs any |
| `.client(url)` | a client that speaks the deployed app's routes |
| `.source` | *(models with weights)* a callable that downloads them and returns the directory |

| Importer | Serves | Client |
|---|---|---|
| `HuggingFaceCompletingImport(hf_id, token=None)` | HuggingFace causal LM, via `AutoModelForCausalLM` | `ServedCompletingModel` (`complete`) |
| `HuggingFaceImageImport(hf_id, token=None, ignore_patterns=None)` | HuggingFace diffusers pipeline | `HuggingFaceImageModel` (`generate`) |
| `AnthropicImport(model_id, api_key_secret=...)` | the Anthropic API, forwarded from the cluster | `ServedCompletingModel` (`complete`) |

Not every model has weights, so `.source` belongs to the HuggingFace importers
rather than to all of them: one that has weights goes through
`cortexgrid.import_model`, one that has none through `cortexgrid.register_model`.
Everything else is common, which is why one client serves all three — every
completion app speaks the same `/complete` protocol.

## Quick start

```python
import asyncio
import cortexgrid
import cortexgrid_infer as mg

cortexgrid.Experiment.init("img-gen")
imp = mg.HuggingFaceImageImport("black-forest-labs/FLUX.2-klein-base-4B")

# 1. Import — weights go HuggingFace ─▶ cluster node ─▶ registry, never through
#    your machine, so this runs as a job rather than in-process.
def import_weights(importer, requirements):
    with importer:
        cortexgrid.import_model(
            importer.source, importer.serve_app,
            family=importer.family, suffix=importer.suffix,
            requirements=requirements,
        )

# remote() hands back a JobFuture; result() blocks and re-raises whatever the
# job raised, so a failed import surfaces here with its own traceback.
cortexgrid.remote(
    import_weights, imp, imp.requirements(), num_gpus=0, num_cpus=2
).result()

# 2. Deploy — blocks until the Ray Serve app is running.
deployment = cortexgrid.deploy_model(
    imp.family, imp.suffix, cortexgrid.IMPORTED, wait=True, timeout=1800
)

# 3. Inference.
model = imp.client(deployment.url)

async def run():
    result = await mg.generate(model, "a red bicycle on a beach at sunrise")
    open("out.png", "wb").write(result.image)   # PNG bytes; also .width/.height/.params

asyncio.run(run())

# 4. Teardown.
cortexgrid.undeploy_model(imp.family, imp.suffix, cortexgrid.IMPORTED)  # free the GPU
cortexgrid.delete_model(imp.family, imp.suffix, cortexgrid.IMPORTED)    # and the weights
```

See [`examples/deploy_text_model`](examples/deploy_text_model/main.py) for the whole
lifecycle end to end.

An importer carries no open resources until it is entered, so it can be handed to
`cortexgrid.remote` and rebuilt on the node that runs the import. Entering it opens
the scratch directory `source` downloads into; `import_model` reads that directory
after `source` returns, which is why the lifetime spans the import call rather than
the download.

Call the import on every run. The weights are imported once and shared by every run:
on a model that is already `ready`, `import_model` downloads nothing, re-bundles the
serve code if it changed, and tags the run with the model.

## Hardware requirements

`cortexgrid.ModelRequirements` decides which node Ray will place a replica on, and
nothing in the registry can derive it — cortexgrid stores whatever weights it is
handed, and an importer imports whatever repo id it is handed. `requirements()`
estimates it from the repo's own metadata: safetensors headers where the repo has
them, weight-file sizes otherwise. Both are read over HTTP; no weights are
downloaded.

```python
>>> mg.HuggingFaceCompletingImport("Qwen/Qwen2.5-0.5B-Instruct").requirements()
ModelRequirements(num_gpus=1, ram_gb=3.0, vram_gb=3.2)
```

These are estimates. Pass a `ModelRequirements` of your own to `import_model` where
you know better, or edit the figures on the model card in the dashboard afterwards —
`import_model` only fills in requirements a version does not already have, so a
hand-set value is never overwritten.

The estimate always asks for exactly one GPU, because both HuggingFace serve apps move
the whole model onto a single device. A model too large for any one card needs a serve
app that shards it, not a larger `num_gpus`: the requirement would simply never be
placed. (`AnthropicImport` asks for none — it holds no weights.)

## Compilation

Both HuggingFace serve apps run `torch.compile` over the model they load, because
eager PyTorch bills per *operation* and a transformer forward is mostly glue —
views, broadcasts, permutes, elementwise adds — with only a small fraction of the
dispatches being the matrix multiplies that do the arithmetic. While the tensors
are large the host stays ahead of the accelerator and none of that shows; once
they are small, the accelerator drains its queue faster than the host can fill it
and step time tracks the operation count rather than the work. Feeding the model
less then stops helping, because the op count does not shrink with the input.
Inductor fuses the glue away; `reduce-overhead` additionally captures a CUDA graph,
so a step replays one graph instead of thousands of launches.

`cortexgrid_infer.compiling` offers two modes, and a serve app picks the one that
fits the work it does:

| | | for |
|---|---|---|
| `Mode.GRAPHED` (`reduce-overhead`) | fuses **and** captures a CUDA graph, so a forward replays one graph instead of issuing its launches | a loop of identically shaped forwards, where the capture is reused every iteration. A graph cannot go dynamic, so each distinct shape captures separately |
| `Mode.FUSED` (`default`) | fuses only | work whose shapes move under it, where there is nothing stable to capture and dynamo is free to compile one dynamic kernel set for all of them |

Both serve apps are tuned rather than defaulted:

- **`HuggingFaceImageDeployment`** compiles the denoiser (`transformer` or `unet`)
  and every `text_encoder*` as `Mode.GRAPHED`. A denoising schedule is the ideal case —
  every step is the same shapes — so the capture is made once and replayed for the
  whole loop. The VAE is left alone: smallest win of the three, and the one whose
  shapes move most, especially with tiling on.
- **`HuggingFaceCompletingDeployment`** does *not* compile the module. Decode has
  no stable shape until the KV cache is static, so it sets
  `cache_implementation="static"` and a `CompileConfig(mode=Mode.GRAPHED)`, and
  transformers compiles `generate` itself — capturing prefill and decode
  separately, which it is better placed to do. Compiling the module here as well
  would only compile the same forward twice. The cost is VRAM: a static cache is
  preallocated to the length asked for, where a dynamic one grows into it.

Neither asks to be configured, for the same reason the serve apps do not ask which
dtype to load in. Both happen on CUDA and nowhere else — inductor is weakest off
it, and a host outrunning its accelerator is a GPU problem to begin with.

Two things worth knowing:

- **Warm-up is per input shape, and lands on a request.** Compilation is lazy, so
  the first request at each image size — or each prompt and generation length —
  pays for it.
- **Compilation never becomes load-bearing.** A backend that cannot compile a model
  falls back to eager rather than failing the deployment.

It does not fix everything. Every forward re-reads the model's whole weight tensor
from memory, and no amount of fusion makes that read smaller. Below some input size
it is the floor; only quantizing the weights or running fewer forwards moves it.

## Inference

`complete` streams `CompletionChunk`s (`.content`, `.tool_calls`, `.finish_reason`):

```python
imp = mg.HuggingFaceCompletingImport("Qwen/Qwen2.5-7B-Instruct")
model = imp.client(deployment.url)
messages = [{"role": "user", "content": "Explain RAG in one sentence."}]

async for chunk in mg.complete(model, messages, max_new_tokens=512, temperature=0.7):
    if chunk.has_text:
        print(chunk.content, end="", flush=True)
```

`generate` returns one `GeneratedImage` (`.image` PNG bytes, `.width`, `.height`,
`.params`); pass `image=` bytes for img2img.

### Anthropic

An Anthropic model is a registry entry like any other — the difference is that
there are no weights to stage, so it is imported with no source, and the serve
app forwards to the API instead of loading anything:

```python
cortexgrid.set_secret("ANTHROPIC_API_KEY", "sk-ant-...")   # once

imp = mg.AnthropicImport("claude-sonnet-5")
cortexgrid.register_model(
    imp.serve_app,
    family=imp.family, suffix=imp.suffix,
    requirements=imp.requirements(), config=imp.config(),
)
```

`register_model` is `import_model` for a model that stages nothing: only the
serve bundle is stored, and the entry is indistinguishable from an imported one
to `deploy_model`, `list_models` and the dashboard.

`config` carries the Anthropic model name and the **name of** the cortexgrid
secret holding the key — never the key itself, since a registry entry is readable
by anyone who can see the model. The serve app reads it with
`cortexgrid.model_config` at construction, and both values are editable on the
model card, so pointing a deployment at a different model or key is an edit plus
a re-deploy rather than a re-registration.

A replica asks for no hardware at all, so it is placed on any node, CPU-only
included. From there it deploys and streams exactly like a cluster-served model:
`imp.client(deployment.url)` returns the same `ServedCompletingModel`. Anthropic
reports tool calls as structured blocks rather than as generated text, so the
serve app re-encodes them into the text form the client parses.

## API

| | |
|---|---|
| `HuggingFaceCompletingImport(hf_id, token=None)` | Importer for a causal LM. |
| `HuggingFaceImageImport(hf_id, token=None, ignore_patterns=None)` | Importer for a diffusers pipeline. |
| `AnthropicImport(model_id, api_key_secret="ANTHROPIC_API_KEY")` | Importer for an Anthropic model; no weights. |
| `ModelImport` | Base of all three: `family`, `suffix`, `serve_app`, `requirements()`, `config()`, `client(url)`. |
| `HuggingFaceImport` | Adds the download half — `source` and the scratch directory. Subclass it for another HuggingFace family. |
| `complete(model, messages, tools=None, max_new_tokens=2048, temperature=0.7, **kw)` | Async stream of `CompletionChunk` for a `CompletingModel`. |
| `generate(model, prompt, *, image=None, **kw) -> GeneratedImage` | One image from a `GeneratingModel`. |
| `ServedCompletingModel(url, model_id)` | Client for any deployed completion app. |
| `HuggingFaceImageModel(url, model_id)` | Client for a deployed diffusers app. |
| `split_model_id(model_id) -> (family, suffix)` | The registry identity an importer derives. |
| `detect_device()` | The torch device a serve app should load onto. |
| `compiling.Mode.GRAPHED` / `compiling.Mode.FUSED` | The two modes a serve app chooses between. See [Compilation](#compilation). |
| `compiling.supported(device)` / `compile_module(module, device, mode)` / `compile_pipeline(pipe, device, mode)` | Whether to compile here, and what a serve app calls to do it. |
| `ModelDeployFailed` | Re-exported from cortexgrid; subclasses `RuntimeError`. |

Types: `DeployedModel`, `CompletingModel`, `GeneratingModel`, `CompletionChunk`,
`GeneratedImage`, `ToolCall`, `Message`, `Tool`, `ToolSpec`.

Adding a provider means one importer and one serve app. A completion app only has
to stream text from `POST /complete`, inlining tool calls as
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (there is an
`encode_tool_call` for upstreams that report them structurally) — then
`ServedCompletingModel` is its client, unchanged.

## Notes

- **HuggingFace auth.** For gated/private repos, pass `token=` to the importer (or
  read `HF_TOKEN` yourself and pass it); it travels with the importer to the node
  that downloads.
- **Image weight pruning.** `HuggingFaceImageImport` skips example images, docs and
  `.gitattributes` by default — nothing `from_pretrained` reads. It does **not**
  skip a repo's consolidated single-file checkpoint, which is weights it cannot
  tell apart from the component ones: name it per-model in
  `HF_IMAGE_SNAPSHOT_IGNORE` (comma-separated globs, e.g.
  `flux-2-klein-base-4b.safetensors`) to save staging it, or pass
  `ignore_patterns=` to replace the defaults outright. Whatever the globs, the
  download and the hardware estimate use the same list, so the figures always
  describe what actually gets staged:

  ```python
  >>> mg.HuggingFaceImageImport("black-forest-labs/FLUX.2-klein-base-4B").requirements()
  ModelRequirements(num_gpus=1, ram_gb=26.3, vram_gb=29.6)   # single-file checkpoint included
  ```
- **Only model files are stored.** HuggingFace's download bookkeeping
  (`.cache/huggingface/` inside the download folder) is dropped before the upload.
- **Shared across runs.** A model's identity is `(family, suffix, cortexgrid.IMPORTED)`,
  derived from the repo id alone, so every run imports, deploys and deletes the same
  copy — deleting removes it for all runs. Importing needs an active cortexgrid run,
  which it tags with the model.

See the [cortexgrid model-serving docs](https://github.com/robodatalab/cortexgrid/blob/main/docs/cortexgrid/model-serving.md)
for the registry and serving state machines.
