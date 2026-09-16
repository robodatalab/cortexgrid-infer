# cortexgrid-infer examples
#
# Prerequisites:
#   - `uv sync` at the repo root
#   - CORTEXGRID_HEAD_URL set (env or repo-root .env), e.g. http://robolab-head:7700
#   - HF_TOKEN set for gated/private models
#
# Usage:
#   make -C examples deploy_text_model

.PHONY: deploy_text_model

deploy_text_model:
	cd deploy_text_model && uv run python main.py
