# MNIST SageMaker Inference Endpoint

Deploy an MNIST inference endpoint on AWS SageMaker with Triton Inference Server. The pipeline downloads a pre-trained MNIST PyTorch model, converts it to ONNX, packages it in Triton model repository format, and uploads the artifact to S3.

## Setup

Requires Python >= 3.10. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For development (includes pytest and hypothesis):

```bash
uv sync --extra dev
```

## Project Structure

```
.
├── main.py                          # Entry point (placeholder)
├── pyproject.toml                   # Project metadata, dependencies, build config (hatchling)
├── uv.lock                          # Locked dependency versions
├── .python-version                  # Pinned Python version for uv
├── src/
│   ├── __init__.py                  # Package marker
│   ├── config.py                    # PackagerConfig and DeployerConfig dataclasses
│   ├── exceptions.py                # Custom exception hierarchy (PipelineError base)
│   └── model_packager.py           # ModelPackager class — full pipeline orchestration
├── tests/
│   ├── __init__.py                  # Package marker
│   ├── test_model_packager.py      # Tests for download, ONNX conversion, validation
│   └── test_model_packager_upload.py # Tests for S3 upload retry logic and run() orchestration
├── design_decisions/
│   ├── architecture-options.md     # Comparison of Lambda vs SageMaker deployment options
│   └── model-format-comparison.md  # ONNX vs CatBoost Binary vs JSON format trade-offs
└── .kiro/
    ├── hooks/
    │   ├── format-on-create.json   # Auto-format new .py files with ruff
    │   ├── format-on-save.json     # Auto-format saved .py files with ruff
    │   └── update-readme-on-save.json # Prompt agent to keep README in sync on save
    └── specs/
        └── mnist-inference-endpoint/
            ├── requirements.md     # Feature requirements
            ├── design.md           # Technical design document
            ├── tasks.md            # Implementation task breakdown
            ├── tasks.meta.json     # Task metadata/status tracking
            └── .config.kiro        # Spec configuration
```

## File Descriptions

### Source (`src/`)

| File | Purpose |
|------|---------|
| `config.py` | Defines `PackagerConfig` (model source URL, S3 bucket/prefix, ONNX opset version) and `DeployerConfig` (endpoint name, instance type, region). |
| `exceptions.py` | Custom exception hierarchy: `PipelineError` base class with `DownloadError`, `ConversionError`, `ValidationError`, `UploadError`, and `DeploymentError`. |
| `model_packager.py` | `ModelPackager` class that implements the full pipeline: download model → convert to ONNX → validate → create Triton repo → package as tar.gz → upload to S3 with retry logic. |

### Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_model_packager.py` | Unit tests for download (network errors, HTTP errors, timeouts), ONNX conversion (valid/invalid models, opset config), and ONNX validation. |
| `test_model_packager_upload.py` | Unit tests for S3 upload retry behavior (success after retries, failure after 3 attempts, delay timing) and `run()` pipeline orchestration order. |

### Design Decisions (`design_decisions/`)

| File | Purpose |
|------|---------|
| `architecture-options.md` | Compares four deployment architectures: Lambda, Lambda + Provisioned Concurrency, SageMaker Serverless, and SageMaker Real-Time. Includes cost/latency trade-offs and decision flowcharts. |
| `model-format-comparison.md` | Compares model formats (ONNX, CatBoost binary, JSON), container options (Triton, pre-built, extended, BYOC), and deployment modes. |

### Root Files

| File | Purpose |
|------|---------|
| `main.py` | Application entry point (currently a placeholder). |
| `pyproject.toml` | Project definition — name, version, Python requirement (>=3.10), runtime deps (torch, onnx, requests, boto3), dev deps (pytest, hypothesis), and hatchling build system. |
| `uv.lock` | Lock file pinning exact dependency versions for reproducible installs. |
| `.python-version` | Tells uv which Python version to use. |

## Agent Hooks

This project uses Kiro agent hooks (`.kiro/hooks/`) to automate development workflows.

### Format on Create — `format-on-create.json`

Formats Python files with [ruff](https://docs.astral.sh/ruff/) whenever the agent creates a new `.py` file.

- **Trigger:** `PostFileCreate` | **Matcher:** `\.(py)$`
- **Command:** `uvx ruff format "$KIRO_FILE_PATH"`

### Format on Save — `format-on-save.json`

Formats Python files with ruff whenever a `.py` file is saved.

- **Trigger:** `PostFileSave` | **Matcher:** `\.(py)$`
- **Command:** `uvx ruff format "$KIRO_FILE_PATH"`

### Update README on Save — `update-readme-on-save.json`

An agent-type hook that prompts Kiro to review and update the README after every file save, keeping documentation in sync with code changes.

- **Trigger:** `PostFileSave`
- **Action type:** `agent`

## Running Tests

```bash
uv run pytest
```
