# Migrating from 0.2.x to 0.3.0

0.3.0 splits what an importer used to be into two separate pieces:

- **Serve apps** (`cortexgrid_infer.models`) are named for the task they run —
  `Text2Text`, `Text2Image`, `Image2Mesh` — and run any model of that task,
  whatever its source. The serve app now carries what depends on the model: its
  client, the files it never loads, and how much hardware a replica needs.
- **Importers** (`cortexgrid_infer.importers`) are one per source. They handle the
  registry name, the download and the weight size, and nothing else. You pass one
  the serve app, so a single `HuggingFaceImporter` replaces the three
  HuggingFace importer classes.

A model hosted elsewhere (Anthropic) has no weights, so it has no importer. Its
serve app forwards requests, and a `Hosted` entry registers it.

Everything you pass to cortexgrid afterwards has the same shape: `.family`,
`.suffix`, `.serve_app`, `.source`, `.requirements()`, `.config()`,
`.client(url)`. In most code, the constructor is the only line that changes.

## Constructors

| 0.2.x | 0.3.0 |
|---|---|
| `HuggingFaceCompletingImport(hf_id, token=None)` | `HuggingFaceImporter(hf_id, Text2Text, token=None)` |
| `HuggingFaceImageImport(hf_id, token=None, ignore_patterns=None)` | `HuggingFaceImporter(hf_id, Text2Image, token=None, ignore_patterns=None)` |
| a `HuggingFaceMeshImport` subclass naming `serve_app` and `vram_gb` | `HuggingFaceImporter(hf_id, YourImage2MeshSubclass)`, with `min_vram_gb` on the serve app |
| `AnthropicImport(model_id, api_key_secret="ANTHROPIC_API_KEY")` | `Hosted(model_id, AnthropicText2Text, api_key_secret="ANTHROPIC_API_KEY")` |

All of these are still exported from the top-level `cortexgrid_infer` package.

## Other renames

| 0.2.x | 0.3.0 |
|---|---|
| `ModelImport` | `ModelEntry` (`cortexgrid_infer.registry`) |
| `HuggingFaceImport` | `HuggingFaceImporter`; for any source, the `Importer` base |
| `imp.hf_id` | `imp.model_id` |
| `HuggingFaceCompletingDeployment` | `Text2Text` |
| `HuggingFaceImageDeployment` | `Text2Image` |
| `MeshingDeployment` | `Image2Mesh` |
| `AnthropicDeployment` | `AnthropicText2Text` |
| `HuggingFaceMeshImport.vram_gb` | `Image2Mesh.min_vram_gb` |
| `HuggingFaceImageModel` | `ServedGeneratingModel` |
| `requirements.estimate(hf_id, token, ignore)` | `HuggingFaceImporter(hf_id, serve_app, token, ignore).requirements()` |
| `requirements.weight_gb(hf_id, token, ignore)` | `serve_app.weight_gb(repo_weights(hf_id, token, ignore))` |

Module paths, for code that imports from submodules:

| 0.2.x | 0.3.0 |
|---|---|
| `cortexgrid_infer.importing` | `cortexgrid_infer.registry` (`ModelEntry`, `split_model_id`), `cortexgrid_infer.importers` |
| `cortexgrid_infer.requirements` | `cortexgrid_infer.importers.huggingface` (`repo_weights`), `cortexgrid_infer.models.base` (sizing) |
| `cortexgrid_infer.providers.huggingface_complete[_serve]` | `cortexgrid_infer.models.text2text` |
| `cortexgrid_infer.providers.huggingface_image` | `cortexgrid_infer.models.text2image`, `cortexgrid_infer.imaging` (client) |
| `cortexgrid_infer.providers.huggingface_image_serve` | `cortexgrid_infer.models.text2image` |
| `cortexgrid_infer.providers.huggingface_mesh`, `cortexgrid_infer.meshing_serve` | `cortexgrid_infer.models.image2mesh` |
| `cortexgrid_infer.providers.anthropic[_serve]` | `cortexgrid_infer.models.anthropic` |

## Examples

### A text or image model

```python
# 0.2.x
imp = mg.HuggingFaceCompletingImport("Qwen/Qwen2.5-0.5B-Instruct", token=token)
imp = mg.HuggingFaceImageImport("black-forest-labs/FLUX.2-klein-base-4B")

# 0.3.0
imp = mg.HuggingFaceImporter("Qwen/Qwen2.5-0.5B-Instruct", mg.Text2Text, token=token)
imp = mg.HuggingFaceImporter("black-forest-labs/FLUX.2-klein-base-4B", mg.Text2Image)
```

The code after this line stays the same:
`cortexgrid.import_model(imp.source, imp.serve_app, ...)`, `imp.requirements()`
and `imp.client(url)`.

### An image-to-mesh model

In 0.2.x, the serve app and the importer were two classes. In 0.3.0 the serve app
is the only class you write.

```python
# 0.2.x
class TripoSRDeployment(mg.MeshingDeployment):
    def load(self, path, device): ...
    def make_mesh(self, image, **options): ...

class TripoSRImport(mg.HuggingFaceMeshImport):
    serve_app = TripoSRDeployment
    vram_gb = 6.0

imp = TripoSRImport("stabilityai/TripoSR")

# 0.3.0
class TripoSRDeployment(mg.Image2Mesh):
    min_vram_gb = 6.0
    def load(self, path, device): ...
    def make_mesh(self, image, **options): ...

imp = mg.HuggingFaceImporter("stabilityai/TripoSR", TripoSRDeployment)
```

### Anthropic

```python
# 0.2.x
imp = mg.AnthropicImport("claude-sonnet-5", api_key_secret="TEAM_KEY")

# 0.3.0
imp = mg.Hosted("claude-sonnet-5", mg.AnthropicText2Text, api_key_secret="TEAM_KEY")
```

`cortexgrid.register_model(imp.serve_app, ...)` is unchanged. If you misspell a
setting, `Hosted` raises `TypeError` at construction, before anything reaches the
registry.

### Type hints

Code that takes an importer as an argument usually only needs the new name:

```python
# 0.2.x
def import_weights(importer: mg.HuggingFaceImport, requirements): ...

# 0.3.0
def import_weights(importer: mg.HuggingFaceImporter, requirements): ...
```

To accept an importer from any source, use `mg.Importer`. To accept anything
registrable, including `Hosted`, use `mg.ModelEntry`.

## Your own serve apps and importers

**A custom importer that only named a serve app and a client** (a
`HuggingFaceImport` subclass setting `serve_app` and overriding `client`) becomes
the serve app itself. Subclass `LocalModel`, move `client` onto it as a
classmethod `client(cls, url, name)`, and pass the class to `HuggingFaceImporter`.
If it overrode `requirements()`, use one of these instead:

- set `min_vram_gb` or `bytes_per_param`, or
- override the serve app's `requirements(cls, weights)` classmethod.

If it skipped files, override the serve app's `ignore_patterns()` classmethod.

**A new source** is a subclass of `Importer` that implements
`download(local_dir)` and `weights()`. Every serve app whose weight format that
source delivers can then run its models.

**A new hosted API** is a subclass of `HostedModel` with `config(model_id,
**settings)` and `client(url, name)`, registered through `Hosted`.

## What has not changed

- **Wire protocols and routes.** `/complete`, `/generate` and `/mesh` are
  unchanged, so a 0.3.0 client talks to an app deployed by 0.2.x, and the other
  way round.
- **Hardware estimates.** They produce the same figures as before: the same
  16-bit sizing, headroom, floors and single GPU.
- **Download behaviour.** The download works as before: the scratch directory only
  exists inside `with imp:`, HuggingFace's `.cache/huggingface/` bookkeeping is
  dropped, and an importer that hasn't been entered survives `cloudpickle` into
  `cortexgrid.remote`.
- **`HF_IMAGE_SNAPSHOT_IGNORE`** is still read, now by `Text2Image.ignore_patterns()`.
  An explicit `ignore_patterns=` still replaces the defaults.
- **`split_model_id`**, and therefore every registry key, is the same. Models
  already in the registry keep their `(family, suffix)`.

## Models already in the registry

Each registry entry's bundle records the serve app's import path, for example
`cortexgrid_infer.providers.huggingface_complete_serve:HuggingFaceCompletingDeployment`.

- **Existing entries keep working.** They run the code they were bundled with.
- **The next import rebuilds the bundle.** The first `import_model` (or
  `register_model`) call with a 0.3.0 serve app replaces the bundle. It keeps the
  weights and downloads nothing, and it keeps any requirements or config you set
  on the model card.
- **Running deployments need a redeploy.** A running deployment keeps serving the
  old code until you deploy it again.
