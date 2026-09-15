# model-gateway

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
- **Deploy** schedules the model as a Ray Serve app and returns a client. It
  requires the model to be registry-`ready` (raises otherwise).
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
import model_gateway as mg

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

# 3. Deploy — schedule the Ray Serve app; get a client back.
model = mg.deploy_model(mid)

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

## API

All exported from `model_gateway.*`. Each generic entry point routes by model-id
prefix to the provider that registered for it.

| Function | Purpose |
|---|---|
| `upload_model(id) -> str \| None` | Start ingesting the weights into the registry (a cluster job); returns its job id, or `None` if already staged or nothing to stage (hosted models). |
| `deployment_status(id) -> status \| None` | Combined phase: the registry lifecycle (`uploading`/`ready`/`upload_failed`/`broken`) while staging, then the serving lifecycle (`deploying`/`running`/`failed`/…) once a Serve app exists. `None` for hosted models. |
| `deploy_model(id) -> DeployedModel` | Schedule the model (must be registry-`ready`) and return a client. Idempotent — reuses a running deployment. |
| `complete(model, messages, tools=None, max_new_tokens=2048, temperature=0.7, **kw)` | Async stream of `CompletionChunk` for a `CompletingModel` (`hf:`, `Anthropic/`). |
| `generate(model, prompt, *, image=None, **kw) -> GeneratedImage` | One image from a `GeneratingModel` (`hf-image:`). Pass `image=` bytes for img2img. |
| `model.undeploy()` | Free the GPU (tear down the Serve app); weights stay registered. |
| `delete_model(id)` | Full cleanup: undeploy if running, then delete the weights + serve bundle from the registry. Idempotent; no-op for hosted models. |

Extending to a new provider means registering four prefix handlers:
`register_provider` (deploy), `register_uploader` (upload), `register_status_provider`
(status), `register_deleter` (cleanup).

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
