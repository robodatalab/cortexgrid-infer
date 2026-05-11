# RoboLab  Model Gateway

Library for running inference on a wide range of models

## Related repos

- `robolab-infra` — infrastructure used to deploy models. It hosts the `cortexflow` SDK that is used to access the infrastructure programmaticaly.
- `model-training` — Training facilities (SFT, LoRA), depends on `cortexflow`

## Important rules

WORK in small increments, always consulting everything with the user.

Respond succintly, and always to the specifically asked question. Do not add unnecessary details unless asked.

Never use local/inline imports (imports inside functions, methods, or conditional blocks). All imports must be unconditional and at the top of the file. If this creates a circular dependency, restructure the code (e.g. move a function to a different module) rather than working around it with a lazy import.