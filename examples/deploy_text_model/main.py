"""Deploy a HuggingFace text model on the cortexgrid cluster and chat with it.

Runs the whole lifecycle in one go: import the weights into the registry, deploy
the Ray Serve app, chat in the terminal, then delete the model from the cluster.

The lifecycle is driven with the cortexgrid SDK directly - `remote`,
`import_model`, `deploy_model`, `delete_model`. cortexgrid-infer only supplies
the pieces those calls need: the text-to-text serve app that will run the
weights, and a HuggingFace importer that fetches them, names them in the
registry, estimates the hardware one replica needs, and hands back the serve
app's HTTP client for the deployed app.

Requires CORTEXGRID_HEAD_URL (e.g. in the repo-root .env), and HF_TOKEN for gated
models.
"""

import asyncio
import os

import cortexgrid
from dotenv import load_dotenv

import cortexgrid_infer as mg

HF_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def import_weights(
    importer: mg.HuggingFaceImporter, requirements: cortexgrid.ModelRequirements
) -> None:
    """Import the model into the registry under the hardware it needs to be served.

    Runs on a cluster node, not the caller's machine, so the weights travel
    HuggingFace -> node -> registry and never transit the client. The importer
    owns the scratch directory the download lands in, and hands `import_model` a
    source callable that fills it - called only when the weights actually have
    to be uploaded, so a model that is already `ready` downloads nothing and
    just re-bundles changed serve code.

    Nothing is caught here: `JobFuture.result()` re-raises whatever this raises,
    so a failed import reaches the caller with its own traceback intact."""
    with importer:
        cortexgrid.import_model(
            importer.source,
            importer.serve_app,
            family=importer.family,
            suffix=importer.suffix,
            requirements=requirements,
            config=importer.config(),
        )


async def stream_reply(model: mg.CompletingModel, messages: list[mg.Message]) -> str:
    reply = ""
    async for chunk in mg.complete(model, messages, max_new_tokens=512):
        if chunk.has_text:
            print(chunk.content, end="", flush=True)
            reply += chunk.content
    print()
    return reply


def chat(model: mg.CompletingModel) -> None:
    print("\nChat with the model. Empty line or Ctrl-D to quit.")
    messages: list[mg.Message] = []
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except EOFError:
            return
        if not prompt:
            return
        messages.append({"role": "user", "content": prompt})
        print("model> ", end="", flush=True)
        reply = asyncio.run(stream_reply(model, messages))
        messages.append({"role": "assistant", "content": reply})


def main() -> None:
    load_dotenv()
    exp = cortexgrid.Experiment.init("cortexgrid-infer-examples")
    print(f"run: {exp.run_name()}")

    token = os.environ.get("HF_TOKEN")
    imp = mg.HuggingFaceImporter(HF_ID, mg.Text2Text, token=token)

    # Reads the repo's safetensors headers over HTTP - no weights downloaded - so
    # the registry entry carries the hardware a replica needs and Ray places it
    # on a node that has it. An estimate: override it with a
    # `cortexgrid.ModelRequirements(...)` of your own where you know better, or
    # edit the figures on the model card in the dashboard afterwards.
    requirements = imp.requirements()
    print(f"requirements: {requirements}")

    try:
        print(f"importing {HF_ID} as {imp.family}/{imp.suffix}")
        job = cortexgrid.remote(
            import_weights,
            imp,
            requirements,
            num_gpus=0,
            num_cpus=2,
        )
        print(f"  import job: {job.job_id}")
        # Blocks until the job finishes, and re-raises whatever it raised. The
        # job's exit is the whole answer - a returned call means the weights are
        # in the registry, so there is no second lifecycle to poll.
        job.result()

        print("deploying")
        deployment = cortexgrid.deploy_model(
            family=imp.family,
            suffix=imp.suffix,
            run_name=cortexgrid.IMPORTED,
            wait=True,
        )

        chat(imp.client(deployment.url))
    finally:
        print("deleting")
        cortexgrid.undeploy_model(imp.family, imp.suffix, cortexgrid.IMPORTED)
        cortexgrid.delete_model(imp.family, imp.suffix, cortexgrid.IMPORTED)


if __name__ == "__main__":
    main()
