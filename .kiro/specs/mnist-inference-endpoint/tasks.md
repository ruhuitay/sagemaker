# Implementation Plan: MNIST Inference Endpoint

## Overview

This plan implements an MNIST inference endpoint on SageMaker using Triton Inference Server with ONNX, following an incremental layered approach. Infrastructure is defined declaratively using **AWS CDK (Python)** — all cloud resources (SageMaker endpoint, API Gateway) are deployed via CloudFormation stacks. Cleanup is handled by `cdk destroy`.

The architecture uses **direct API Gateway → SageMaker integration** (no Lambda). API Gateway calls the SageMaker endpoint directly using an AWS service integration with IAM role-based authentication. Clients send and receive **raw Triton V2 protocol JSON** — no request/response transformation is performed.

The offline model preparation pipeline (`model_packager.py`) runs before `cdk deploy` to produce the S3 model artifact. CDK stacks then reference that artifact URI to deploy the infrastructure.

The CDK app deploys 3 stacks in order: `StorageStack` (S3 bucket) → `SageMakerStack` (endpoint) → `ApiStack` (API Gateway with direct SageMaker integration).

Implementation language: Python 3.12. Property-based tests use Hypothesis. CDK tests use `aws_cdk.assertions`.

## Tasks

- [x] 1. Layer 1 — Model Packaging and S3 Upload
  - [x] 1.1 Create project structure and configuration dataclasses
    - Create `src/` directory with `__init__.py`
    - Define `PackagerConfig` dataclass in `src/config.py`
    - Define custom exception classes (`ConversionError`, `ValidationError`, `UploadError`) in `src/exceptions.py`
    - Set up `pyproject.toml` with project dependencies
    - _Requirements: 1.1–1.9, 2.1–2.5_

  - [x] 1.2 Implement model download and ONNX conversion
    - Create `src/model_packager.py` with `ModelPackager` class
    - Implement `convert_to_onnx()` — convert PyTorch model to ONNX (opset >= 11) using `torch.onnx.export`
    - Implement `validate_onnx()` — validate with `onnx.checker.check_model`
    - Handle errors: raise typed exceptions (`ModelLoadError`, `ConversionError`, `ValidationError`)
    - _Requirements: 1.1, 1.2, 1.3, 1.7, 1.8_

  - [x] 1.3 Implement Triton model repository creation and packaging
    - Implement `create_model_repository()` — create directory structure (`mnist/config.pbtxt`, `mnist/1/model.onnx`)
    - Generate `config.pbtxt` with platform `onnxruntime_onnx`, input shape [1, 28, 28] FP32, output shape [10] FP32, max_batch_size 8
    - Implement `package_artifact()` — create `model.tar.gz` preserving Triton directory hierarchy
    - _Requirements: 1.4, 1.5, 1.6, 1.9_

  - [ ]* 1.4 Write property test for model artifact packaging round-trip
    - **Property 1: Model artifact packaging round-trip**
    - Generate random valid Triton repository structures, package into tar.gz, extract, verify identical file paths and contents
    - **Validates: Requirements 1.6**

  - [x] 1.5 Implement S3 upload with retry logic
    - Implement `upload_to_s3()` — upload `model.tar.gz` to configured bucket/prefix with retry (3 attempts, 1s delay)
    - Construct and return S3 URI (`s3://{bucket}/{prefix}{filename}`)
    - Implement `run()` method orchestrating the full pipeline
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 1.6 Write property tests for S3 upload behavior
    - **Property 2: S3 upload retry behavior**
    - Generate random failure sequences (0–4 failures), verify retry with delay and final outcome
    - **Validates: Requirements 2.3, 2.5**
    - **Property 3: S3 URI construction**
    - Generate random valid bucket names and prefixes, verify URI format `s3://{bucket}/{prefix}{filename}`
    - **Validates: Requirements 2.4**

- [x] 2. Layer 1 — CDK Infrastructure Stacks
  - [x] 2.1 Set up CDK app structure
    - Create `infra/` directory with `app.py` (CDK app entry point) and `stacks/__init__.py`
    - Create `cdk.json` pointing to `infra/app.py`
    - Add CDK dependencies to `pyproject.toml` or a separate `infra/requirements.txt` (`aws-cdk-lib`, `constructs`)
    - Wire `app.py` to instantiate 3 stacks in order: `StorageStack` → `SageMakerStack` → `ApiStack`
    - Accept `model_key` and `instance_type` via CDK context, set `env` to eu-west-1
    - _Requirements: 3.1_

  - [x] 2.2 Implement StorageStack
    - Create `infra/stacks/storage_stack.py` with `StorageStack(Stack)` class
    - Create S3 bucket with SSE-S3 encryption
    - Set removal policy to DESTROY with auto-delete objects (dev/test)
    - Export bucket name and bucket ARN as CloudFormation outputs (`CfnOutput`)
    - _Requirements: 2.1, 2.2_

  - [x] 2.3 Implement SageMakerStack with CfnModel, CfnEndpointConfig, CfnEndpoint
    - Create `infra/stacks/sagemaker_stack.py` with `SageMakerStack(Stack)` class
    - Accept `model_bucket` and `model_key` as constructor params
    - Construct S3 URI internally as `s3://{model_bucket}/{model_key}`
    - Implement `CfnModel` — reference Triton CPU container image URI for eu-west-1 and constructed S3 model data URL
    - Implement `CfnEndpointConfig` — production variant with specified instance type (default `ml.c5.large`), initial instance count = 1
    - Implement `CfnEndpoint` — real-time inference endpoint
    - Expose `endpoint_name` as a CloudFormation output for cross-stack reference
    - Add instance type validation in constructor (`_validate_instance_type`) — reject GPU/accelerator types with `ValueError`
    - _Requirements: 3.1, 3.2, 3.7, 3.8, 3.9, 6.1, 6.4, 6.5_

  - [ ]* 2.4 Write CDK assertion tests for StorageStack and SageMakerStack
    - Use `aws_cdk.assertions.Template` to verify synthesized templates contain correct resources
    - Test: StorageStack creates S3 bucket with SSE-S3 encryption configured
    - Test: StorageStack has CloudFormation outputs for bucket name and ARN
    - Test: SageMakerStack correctly constructs S3 URI from `model_bucket` + `model_key`
    - Test: CfnModel references correct container image and constructed S3 URI
    - Test: CfnEndpointConfig uses specified instance type with 1 instance
    - Test: CfnEndpoint resource exists
    - Test: constructor raises `ValueError` for GPU instance types (ml.p3.2xlarge, ml.g4dn.xlarge)
    - _Requirements: 2.1, 2.2, 3.1, 3.8, 3.9, 6.1, 6.4, 6.5_

- [ ] 3. Checkpoint — Layer 1 complete
  - Ensure all tests pass, ask the user if questions arise.
  - At this point the deployment flow is:
    ```
    cdk deploy MnistStorageStack
    uv run src/model_packager.py
    cdk deploy MnistSageMakerStack
    ```
  - After StorageStack + model upload + SageMakerStack, the endpoint is invokable via `boto3.invoke_endpoint()`.

- [ ] 4. Layer 2 — External Access (CDK ApiStack)
  - [ ] 4.1 Implement CDK ApiStack with direct SageMaker integration
    - Create `infra/stacks/api_stack.py` with `ApiStack(Stack)` class
    - Accept `sagemaker_endpoint_name` as constructor parameter
    - Create REST API with `POST /predict` method
    - Create IAM role with `sagemaker:InvokeEndpoint` permission on the specific endpoint ARN
    - Configure `AwsIntegration` with `service="runtime.sagemaker"`, `path="endpoints/{endpoint_name}/invocations"`
    - Set up integration response mappings: 200 for success, `4\d{2}` → 400, `5\d{2}` → 503
    - Add API key requirement on the method
    - Create usage plan (10 rps rate limit, 20 burst limit)
    - Add CloudFormation outputs: invoke URL, API key value
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.7, 5.9_

  - [ ]* 4.2 Write CDK assertion tests for ApiStack
    - Verify IAM policy contains only `sagemaker:InvokeEndpoint`
    - Verify REST API with POST method and API key required
    - Verify usage plan with correct rate/burst limits (10 rps, 20 burst)
    - Verify `AwsIntegration` configuration (service, path, integration responses)
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 5. Checkpoint — Layer 2 complete
  - Ensure all tests pass, ask the user if questions arise.
  - At this point the deployment flow is:
    ```
    cdk deploy MnistStorageStack
    uv run src/model_packager.py
    cdk deploy MnistSageMakerStack
    cdk deploy MnistApiStack
    ```
  - External applications can call `POST /predict` via API Gateway with an API key, sending raw Triton V2 JSON without needing AWS credentials.

- [ ] 6. Layer 3 — Production Hardening
  - [ ] 6.1 Add auto-scaling configuration to SageMakerStack
    - Add Application Auto Scaling resources to `infra/stacks/sagemaker_stack.py`
    - Configure scalable target for SageMaker endpoint variant (min 1, max 10)
    - Add target tracking scaling policy on `SageMakerVariantInvocationsPerInstance`
    - _Requirements: 6.2, 6.3_

  - [ ]* 6.2 Write property test for instance type validation
    - **Property 4: Instance type validation**
    - Generate random instance type strings (from allowed CPU prefixes, GPU prefixes, and random strings), verify correct accept/reject classification
    - Test the `_validate_instance_type` logic extracted as a standalone testable function
    - **Validates: Requirements 6.1, 6.4, 6.5**

  - [ ]* 6.3 Write CDK assertion tests for auto-scaling
    - Verify synthesized template contains Application Auto Scaling resources
    - Verify min/max capacity values (1, 10)
    - Verify target tracking policy metric is `SageMakerVariantInvocationsPerInstance`
    - _Requirements: 6.2, 6.3_

- [ ] 7. Final checkpoint — All layers complete
  - Ensure all tests pass, ask the user if questions arise.
  - Verify full system: model packaging → S3 upload → `cdk deploy` (StorageStack → SageMakerStack → ApiStack) → API key auth → auto-scaling
  - Cleanup via `cdk destroy` removes all resources in dependency order

## Notes

- **No Lambda, no formatters, no input validator** — API Gateway integrates directly with SageMaker using AWS service integration.
- Clients send and receive **raw Triton V2 protocol JSON**. No request/response transformation is performed by the infrastructure. Triton handles input validation natively.
- Infrastructure is managed by AWS CDK (Python). All AWS resources are deployed via `cdk deploy` and removed via `cdk destroy`.
- CDK deploys 3 stacks in order: `StorageStack` (S3 bucket) → `SageMakerStack` (endpoint) → `ApiStack` (API Gateway with direct SageMaker integration).
- `model_packager.py` is an offline tool that runs between StorageStack and SageMakerStack deployment to upload the model artifact to the CDK-managed bucket.
- The `model/` folder (train.py, draw_digit.py, test_predict.py, test_predict_sample.py) already exists and is not modified by this plan.
- Tasks marked with `*` are optional and can be skipped for faster MVP.
- Each task references specific requirements for traceability.
- Checkpoints ensure incremental validation after each layer.
- Only 4 correctness properties (Properties 1–4) from the design document. Property tests use Hypothesis.
- CDK assertion tests use `aws_cdk.assertions.Template` for snapshot and fine-grained assertions.
- Region is eu-west-1 for all AWS resources.
- No custom cleanup orchestrator is needed — CDK/CloudFormation handles dependency-ordered deletion natively.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.5"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["1.4", "1.6", "2.2"] },
    { "id": 4, "tasks": ["2.3"] },
    { "id": 5, "tasks": ["2.4"] },
    { "id": 6, "tasks": ["4.1"] },
    { "id": 7, "tasks": ["4.2"] },
    { "id": 8, "tasks": ["6.1"] },
    { "id": 9, "tasks": ["6.2", "6.3"] }
  ]
}
```
