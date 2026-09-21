# cortexgrid-infer

The model-specific pieces that [cortexgrid](https://github.com/robodatalab/cortexgrid)
needs but cannot work out for itself, plus the clients that talk to what it deploys.

cortexgrid stores whatever weights it is handed under whatever key it is given, and
serves them with whatever class it was told to bundle. It deliberately knows nothing
about where a model came from or what shape it is. This library supplies exactly that
missing knowledge, and drives no lifecycle of its own: importing, deploying and
deleting stay your calls against the cortexgrid SDK.

Upgrading from 0.2.x? See [MIGRATION.md](https://github.com/robodatalab/cortexgrid-infer/blob/main/MIGRATION.md).

## Division of labour

| | Owns |
|---|---|
| **cortexgrid** | Registry identity, weights, serve bundle, hardware placement, deployment, jobs. Never knows what kind of model it holds. |
| **cortexgrid-infer** | Per source: how to name a model, fetch it and size it. Per task: how to run it and talk to it. Never drives a lifecycle. |
| **you** | Policy: which model, which requirements, when to import, deploy and tear down. |

A model is two things, and the library keeps them apart:

- **What runs it** — a serve app in `cortexgrid_infer.models`, named for the task it
  runs, not for where the model came from. It loads the weights, answers the task's
  routes, and names the client that speaks them. One serve app runs every model of
  its task, whichever source staged it.
- **Where its weights come from** — an importer in `cortexgrid_infer.importers`, one
  per source. It names the model in the registry, downloads the weights, and sizes
  them from the source's metadata. It is handed the serve app, so one importer takes
  any model its source holds.

| Serve app | Task | Runs | Client |
|---|---|---|---|
| `Text2Text` | text to text | any causal LM in the transformers layout, via `AutoModelForCausalLM` | `ServedCompletingModel` (`complete`) |
| `Text2Image` | text (and image) to image | any diffusers pipeline | `ServedGeneratingModel` (`generate`) |
| `Image2Mesh` — a base: subclass it with the model's `load` and `make_mesh` | image to mesh | the model the subclass loads, behind the task's `POST /mesh` | `ServedMeshingModel` (`mesh`: one picture in, a `GeneratedMesh` out) |
| `AnthropicText2Text` | text to text | the Anthropic API, forwarded from the cluster | `ServedCompletingModel` (`complete`) |

| Importer | Source |
|---|---|
| `HuggingFaceImporter(hf_id, serve_app, token=None, ignore_patterns=None)` | the HuggingFace Hub |

A model hosted elsewhere has no weights, and so no importer: its serve app forwards
to it, and a `Hosted(model_id, serve_app, **settings)` entry registers it.

An importer and a `Hosted` entry are both a `ModelEntry`, which gives you what
`cortexgrid.import_model`, `cortexgrid.register_model` and `cortexgrid.deploy_model`
need:

| | |
|---|---|
| `.family` / `.suffix` | how this model id names itself in the registry |
| `.serve_app` | the class cortexgrid bundles and runs on the cluster |
| `.requirements()` | what one replica needs to be placed and to run |
| `.config()` | settings the serve app reads at construction, if it needs any |
| `.client(url)` | a client that speaks the deployed app's routes |
| `.source` | *(importers only)* a callable that downloads the weights and returns the directory |

One with weights goes through `cortexgrid.import_model`, one without through
`cortexgrid.register_model`.

## Quick start

```python
import asyncio
import cortexgrid
import cortexgrid_infer as mg

cortexgrid.Experiment.init("img-gen")
imp = mg.HuggingFaceImporter("black-forest-labs/FLUX.2-klein-base-4B", mg.Text2Image)

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
splits the estimate along the same line as the rest: the importer reads what the
source records about the weights — for HuggingFace, safetensors headers where the
repo has them, weight-file sizes otherwise, both over HTTP with no weights
downloaded — and the serve app turns that into what a replica needs, since it knows
the dtype it loads in and what its work takes on top.

```python
>>> mg.HuggingFaceImporter("Qwen/Qwen2.5-0.5B-Instruct", mg.Text2Text).requirements()
ModelRequirements(num_gpus=1, ram_gb=3.0, vram_gb=3.2)
```

These are estimates. Pass a `ModelRequirements` of your own to `import_model` where
you know better, or edit the figures on the model card in the dashboard afterwards —
`import_model` only fills in requirements a version does not already have, so a
hand-set value is never overwritten.

The estimate always asks for exactly one GPU, because every serve app that runs weights
moves the whole model onto a single device. A model too large for any one card needs a
serve app that shards it, not a larger `num_gpus`: the requirement would simply never
be placed. An `Image2Mesh` subclass can set `min_vram_gb` for the memory meshing takes
beyond the weights. (A `Hosted` entry asks for none — it holds no weights.)

## Compilation

`Text2Text` and `Text2Image` run `torch.compile` over the model they load, because
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

- **`Text2Image`** compiles the denoiser (`transformer` or `unet`)
  and every `text_encoder*` as `Mode.GRAPHED`. A denoising schedule is the ideal case —
  every step is the same shapes — so the capture is made once and replayed for the
  whole loop. The VAE is left alone: smallest win of the three, and the one whose
  shapes move most, especially with tiling on.
- **`Text2Text`** does *not* compile the module. Decode has
  no stable shape of its own — the sequence grows a token per forward — so it
  pins one: `cache_implementation="static"` with a fixed `max_cache_len`, plus a
  `CompileConfig(mode=Mode.GRAPHED)`. transformers then compiles `generate`
  itself, capturing prefill and decode separately, which it is better placed to
  do; compiling the module here as well would compile the same forward twice.

  The fixed length is the point, and how long it is decides what decode costs:
  every token's attention reads the whole cache, filled or not. That is why it is
  *not* pinned to the model's context — 32k slots of a Qwen2.5-3B KV cache is
  1.2 GB of keys and values re-read per generated token, several times the
  traffic of the model's own weights, to make room for a prompt nobody sent. It
  is pinned instead to the prompt and reply a caller plausibly sends: 4096 tokens,
  or `max_total_tokens` on the model card, never past the model's own
  `max_position_embeddings`. A reply takes at most half of it, so an
  over-optimistic `max_new_tokens` cannot squeeze the prompt to nothing, and a
  prompt that will not fit in the rest is truncated — keeping the end, since a
  chat template puts the turn to answer last — with a warning naming both
  lengths. Serving it whole would resize the cache and recompile the decode loop,
  which costs more than the generation itself.

The image app asks to be configured for nothing, for the same reason neither asks
which dtype to load in; the completion app takes `max_total_tokens` alone, because
how much cache a caller needs is the one thing the model cannot tell it. Both
happen on CUDA and nowhere else — inductor is weakest off it, and a host
outrunning its accelerator is a GPU problem to begin with.

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
imp = mg.HuggingFaceImporter("Qwen/Qwen2.5-7B-Instruct", mg.Text2Text)
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
there are no weights to stage, so there is no importer: the serve app forwards to
the API instead of loading anything, and a `Hosted` entry registers it:

```python
cortexgrid.set_secret("ANTHROPIC_API_KEY", "sk-ant-...")   # once

entry = mg.Hosted("claude-sonnet-5", mg.AnthropicText2Text)   # api_key_secret="..." to use another secret
cortexgrid.register_model(
    entry.serve_app,
    family=entry.family, suffix=entry.suffix,
    requirements=entry.requirements(), config=entry.config(),
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
`entry.client(deployment.url)` returns the same `ServedCompletingModel`. Anthropic
reports tool calls as structured blocks rather than as generated text, so the
serve app re-encodes them into the text form the client parses.

## API

| | |
|---|---|
| `HuggingFaceImporter(hf_id, serve_app, token=None, ignore_patterns=None)` | Importer for a model on the HuggingFace Hub, run by `serve_app`. |
| `Importer` | Base of the importers: identity, scratch directory, `source`, and `requirements()` / `client(url)` taken from the serve app. A new source implements `download(local_dir)` and `weights()`. |
| `Hosted(model_id, serve_app, **settings)` | Entry for a model hosted elsewhere; no weights. `settings` are the serve app's `config` arguments. |
| `ModelEntry` | Base of both: `model_id`, `family`, `suffix`, `serve_app`, `requirements()`, `config()`, `client(url)`. |
| `Text2Text`, `Text2Image`, `Image2Mesh` | Serve apps that run weights (`LocalModel`s). |
| `AnthropicText2Text` | Serve app that forwards to the Anthropic API (a `HostedModel`). |
| `LocalModel` | Base of the serve apps that run weights: `ignore_patterns()`, `bytes_per_param`, `min_vram_gb`, `requirements(weights)`, `client(url, name)`. |
| `HostedModel` | Base of the serve apps that forward: `config(model_id, **settings)`, `requirements()`, `client(url, name)`. |
| `Weights(params=None, file_bytes=0)` | What an importer reads about the weights, handed to the serve app to size a replica. |
| `complete(model, messages, tools=None, max_new_tokens=2048, temperature=0.7, **kw)` | Async stream of `CompletionChunk` for a `CompletingModel`. |
| `generate(model, prompt, *, image=None, **kw) -> GeneratedImage` | One image from a `GeneratingModel`. |
| `ServedCompletingModel(url, model_id)` | Client for any deployed completion app. |
| `ServedGeneratingModel(url, model_id)` | Client for any deployed text-to-image app. |
| `ServedMeshingModel(url, model_id)` | Client for any deployed image-to-mesh app. |
| `split_model_id(model_id) -> (family, suffix)` | The registry identity an importer derives. |
| `detect_device()` | The torch device a serve app should load onto. |
| `compiling.Mode.GRAPHED` / `compiling.Mode.FUSED` | The two modes a serve app chooses between. See [Compilation](#compilation). |
| `compiling.supported(device)` / `compile_module(module, device, mode)` / `compile_pipeline(pipe, device, mode)` | Whether to compile here, and what a serve app calls to do it. |
| `ModelDeployFailed` | Re-exported from cortexgrid; subclasses `RuntimeError`. |

Types: `DeployedModel`, `CompletingModel`, `GeneratingModel`, `MeshingModel`,
`CompletionChunk`, `GeneratedImage`, `GeneratedMesh`, `ToolCall`, `Message`, `Tool`,
`ToolSpec`.

Extending it:

- **A new source** is one importer: subclass `Importer` with `download(local_dir)`
  and `weights()`, and every serve app whose weight format it delivers runs its
  models unchanged.
- **A new hosted API** is one serve app: subclass `HostedModel`, and register it
  with `Hosted`. A text-to-text one only has to stream text from `POST /complete`,
  inlining tool calls as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
  (there is an `encode_tool_call` for upstreams that report them structurally) —
  then `ServedCompletingModel` is its client, unchanged.
- **A model no task's serve app can load** gets its own serve app — an `Image2Mesh`
  subclass, say — and, if its source is unusual too, its own importer.

## Notes

- **HuggingFace auth.** For gated/private repos, pass `token=` to the importer (or
  read `HF_TOKEN` yourself and pass it); it travels with the importer to the node
  that downloads.
- **Image weight pruning.** `Text2Image` tells the importer to skip example images,
  docs and `.gitattributes` — nothing `from_pretrained` reads. It does **not**
  skip a repo's consolidated single-file checkpoint, which is weights it cannot
  tell apart from the component ones: name it per-model in
  `HF_IMAGE_SNAPSHOT_IGNORE` (comma-separated globs, e.g.
  `flux-2-klein-base-4b.safetensors`) to save staging it, or pass
  `ignore_patterns=` to replace the defaults outright. Whatever the globs, the
  download and the hardware estimate use the same list, so the figures always
  describe what actually gets staged:

  ```python
  >>> mg.HuggingFaceImporter("black-forest-labs/FLUX.2-klein-base-4B", mg.Text2Image).requirements()
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
