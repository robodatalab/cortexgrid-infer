"""Deploy a HuggingFace text model on the cortexgrid cluster and chat with it.

Runs the whole lifecycle in one go: upload the weights to the registry, deploy the
Ray Serve app, chat in the terminal, then delete the model from the cluster.

Requires CORTEXGRID_HEAD_URL (e.g. in the repo-root .env), and HF_TOKEN for gated
models.
"""

import asyncio
import time

import cortexgrid
from dotenv import load_dotenv

import cortexgrid_infer as mg

MODEL_ID = "hf:Qwen/Qwen2.5-0.5B-Instruct"
POLL_INTERVAL_S = 10


def wait_until_uploaded(model_id: str) -> None:
    while True:
        # None for a brief window before the ingest job registers the version.
        status = mg.deployment_status(model_id)
        phase = None if status is None else status.phase
        print(f"  phase: {phase}")
        if phase == "ready":
            return
        if phase in ("upload_failed", "broken"):
            raise RuntimeError(f"upload of {model_id} ended in phase '{phase}'")
        time.sleep(POLL_INTERVAL_S)


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

    try:
        print(f"uploading {MODEL_ID}")
        mg.upload_model(MODEL_ID)
        wait_until_uploaded(MODEL_ID)

        print("deploying")
        model = mg.deploy_model(MODEL_ID)
        assert isinstance(model, mg.CompletingModel)

        chat(model)
    finally:
        print("deleting")
        mg.delete_model(MODEL_ID)


if __name__ == "__main__":
    main()
