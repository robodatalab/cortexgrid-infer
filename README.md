# cortexgrid-infer

One API to run inference across providers — hosted APIs (Anthropic) and models you
deploy yourself on the [cortexgrid](https://github.com/robodatalab/cortexgrid)
cluster (HuggingFace LLMs and diffusion pipelines). You address every model by a
single **model id** string; the gateway routes it to the right provider by prefix.

| Prefix | Provider | Weights live on the cluster? |
|---|---|---|
| `Anthropic/` | Anthropic hosted API (e.g. `Anthropic/claude-sonnet-5`) | no — hosted |
| `hf:` | HuggingFace causal-LM, served via cortexgrid (e.g. `hf:Qwen/Qwen2.5-7B-Instruct`) | yes |
| `hf-image:` | HuggingFace diffusion pipeline, served via cortexgrid (e.g. `hf-image:black-forest-labs/FLUX.2-klein-base-4B`) | yes |

## The model lifecycle

Hosted models (`Anthropic/`) have nothing to stage — call `deploy_model` and go.
Cluster models (`hf:`, `hf-image:`) move through three separate, independently
observable stages, because staging tens of GB of weights should not block a deploy:

```
upload_model ──▶ [ registry: uploading ─▶ ready ] ──▶ deploy_model ──▶ [ serving: deploying ─▶ running ] ──▶ inference
     │                                                                                                          │
     └── weights ingested HF ─▶ cluster node ─▶ registry (never through your machine)      undeploy_model / delete_model
```

- **Upload** stages the weights into the cortexgrid registry. The download from
  HuggingFace and the upload to the registry both run **on the cluster** (submitted
  as a `cortexgrid.remote` job), so large weights never round-trip through your
  machine. Only the model files are stored: HuggingFace's download bookkeeping
  (`.cache/huggingface/` inside the download folder) is dropped first. Returns a job
  id; poll until the model is `ready`.
- **Deploy** blocks until the model is ready to serve and returns a client. It
  requires the model to be registry-`ready` (raises otherwise), reuses a running
  Ray Serve app, waits on one still coming up, and replaces a failed one. See
  [Deploying](#deploying).
- **Query** reports one combined phase across both lifecycles — registry while the
  weights upload, then serving once an app exists.
- **Teardown**: `model.undeploy()` frees the GPU but keeps the weights (re-deploy is
  cheap). `delete_model(id)` removes everything — it undeploys, then deletes the
  weights and serve bundle from the registry.

## Quick start

Every session starts a cortexgrid run; the model's registry/serving identity is
scoped to it.

```python
import asyncio, time
import cortexgrid
import cortexgrid_infer as mg

cortexgrid.init(experiment="img-gen")   # one call per process; starts an MLflow run
mid = "hf-image:black-forest-labs/FLUX.2-klein-base-4B"

# 1. Upload — ingest weights into the registry, on the cluster.
mg.upload_model(mid)                     # returns a job id (None if already staged)

# 2. Query — wait until the weights are registered and deployable. status is None
#    for a brief window after upload_model, before the ingest job registers the version.
status = mg.deployment_status(mid)
while status is None or status.phase != "ready":
    time.sleep(10)
    status = mg.deployment_status(mid)

# 3. Deploy — blocks until the Ray Serve app is running; get a client back.
#    Raises mg.ModelDeployFailed if the deploy fails, TimeoutError after 30 min.
model = mg.deploy_model(mid, timeout=1800)

# 4. Inference.
async def run():
    result = await mg.generate(model, "a red bicycle on a beach at sunrise")
    open("out.png", "wb").write(result.image)   # PNG bytes; also .width/.height/.params

asyncio.run(run())

# 5. Teardown.
model.undeploy()        # free the GPU, keep the weights (fast re-deploy later)
mg.delete_model(mid)    # OR remove everything: undeploy + delete weights & bundle
```

### Text completion (`hf:` and `Anthropic/`)

`complete` streams `CompletionChunk`s (`.content`, `.tool_calls`, `.finish_reason`):

```python
model = mg.deploy_model("hf:Qwen/Qwen2.5-7B-Instruct")   # after upload → ready
messages = [{"role": "user", "content": "Explain RAG in one sentence."}]

async def run():
    async for chunk in mg.complete(model, messages, max_new_tokens=512, temperature=0.7):
        if chunk.has_text:
            print(chunk.content, end="", flush=True)
```

Hosted models skip staging entirely — `upload_model` / `deployment_status` /
`delete_model` are no-ops that return `None`, so just deploy and call:

```python
model = mg.deploy_model("Anthropic/claude-sonnet-5")
async for chunk in mg.complete(model, messages):
    ...
```

## Deploying

`deploy_model` is the one call that makes a cluster model servable. It reads the
model's current Ray Serve app and does what that state needs:

| Serving phase | `deploy_model` |
|---|---|
| `running` | returns a client straight away |
| `not_started`, `deploying`, `unhealthy` | waits for `running`; it never re-sends the spec, which would restart a build in progress |
| no app yet, `failed`, `deleting` | deploys afresh; cortexgrid removes a failed app first, so the retry really restarts it |

It raises `ModelDeployFailed` when the app fails to deploy, with Ray's message
(including the replica's traceback), and `TimeoutError` when the model is not
running within `timeout` seconds. `timeout=None` (the default) waits indefinitely,
so a model waiting for a free GPU blocks forever. `ModelDeployFailed` is
re-exported from cortexgrid and subclasses `RuntimeError`.

```python
try:
    model = mg.deploy_model(mid, timeout=3600)
except mg.ModelDeployFailed as e:
    log.error("deploying %s failed: %s", mid, e)
    raise
```

A failed app is left in place, so its phase and message stay visible (e.g. in the
cortexgrid UI) until the next `deploy_model` replaces it. There is no need to tear
it down yourself. Retrying only helps with transient failures; a serve-app that
fails deterministically fails again until it is fixed and re-uploaded.

To follow a deploy without blocking - say, a status endpoint polled while another
thread deploys - poll `deployment_status(id)` instead (see
[Status phases](#status-phases)). Hosted models (`Anthropic/`) have nothing to
deploy: `deploy_model` returns immediately and ignores `timeout`.

## API

All exported from `cortexgrid_infer.*`. Each generic entry point routes by model-id
prefix to the provider that registered for it.

| Function | Purpose |
|---|---|
| `upload_model(id) -> str \| None` | Start ingesting the weights into the registry (a cluster job); returns its job id, or `None` if already staged or nothing to stage (hosted models). |
| `deployment_status(id) -> status \| None` | Combined phase: the registry lifecycle (`uploading`/`ready`/`upload_failed`/`broken`) while staging, then the serving lifecycle (`deploying`/`running`/`failed`/…) once a Serve app exists. `None` for hosted models. |
| `deploy_model(id, timeout=None) -> DeployedModel` | Return a client once the model is ready to serve (must be registry-`ready`). Blocks; see [Deploying](#deploying). Raises `ModelDeployFailed` when the deploy fails, `TimeoutError` past `timeout` seconds (`None` waits indefinitely). |
| `complete(model, messages, tools=None, max_new_tokens=2048, temperature=0.7, **kw)` | Async stream of `CompletionChunk` for a `CompletingModel` (`hf:`, `Anthropic/`). |
| `generate(model, prompt, *, image=None, **kw) -> GeneratedImage` | One image from a `GeneratingModel` (`hf-image:`). Pass `image=` bytes for img2img. |
| `model.undeploy()` | Free the GPU (tear down the Serve app); weights stay registered. |
| `delete_model(id)` | Full cleanup: undeploy if running, then delete the weights + serve bundle from the registry. Idempotent; no-op for hosted models. |

Extending to a new provider means registering four prefix handlers:
`register_provider` (deploy), `register_uploader` (upload), `register_status_provider`
(status), `register_deleter` (cleanup). A deploy factory is called as
`factory(model_id, timeout)` and returns a model that is ready to serve, or `None`
to pass the id to the next provider. A cluster-backed provider gets the behaviour in
[Deploying](#deploying) from
`cortexgrid_infer.providers.serving.ensure_serving(family, suffix, run_name, timeout)`,
which returns the Serve app's URL once it is running.

### Status phases

`deployment_status(id).phase` surfaces cortexgrid's lifecycle phases directly:

- **Registry** (while uploading): `uploading` → `ready`; `upload_failed` / `broken` on error.
- **Serving** (once deployed): `not_started` → `deploying` → `running`; `unhealthy` / `failed` / `deleting`.

See the [cortexgrid model-serving docs](https://github.com/robodatalab/cortexgrid/blob/main/docs/cortexgrid/model-serving.md)
for the full state machine and error table.

## Notes

- **HuggingFace auth.** For gated/private repos, set `HF_TOKEN` in your environment;
  it is read at upload time and passed to the cluster ingest job.
- **Image weight pruning.** The `hf-image:` upload skips example images, docs, and a
  repo's consolidated single-file checkpoint by default (only the component weights
  the pipeline loads are staged). Extend per-model with `HF_IMAGE_SNAPSHOT_IGNORE`
  (comma-separated globs).
- **Run scoping.** A model's identity is `(family, suffix, run_name)`, derived from
  the model id and the active cortexgrid run. Upload, deploy, status, and delete all
  resolve the same identity, so they must run against the same `cortexgrid.init`
  experiment/run.
