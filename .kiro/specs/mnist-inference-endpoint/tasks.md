# Implementation Plan: MNIST Inference Endpoint

## Overview

This plan implements an MNIST inference endpoint on SageMaker using Triton Inference Server with ONNX, following an incremental layered approach. Each layer builds on the previous one, so that after Layer 1 you have a working (minimal) endpoint invokable via boto3, and subsequent layers add formatting, external access, and production hardening.

Implementation language: Python. Property-based tests use Hypothesis.

## Tasks

- [ ] 1. Layer 1 — Model Packaging and S3 Upload
  - [x] 1.1 Create project structure and configuration dataclasses
    - Create `src/` directory with `__init__.py`
    - Define `PackagerConfig` and `DeployerConfig` dataclasses in `src/config.py`
    - Define custom exception classes (`DownloadError`, `ConversionError`, `ValidationError`, `UploadError`, `DeploymentError`) in `src/exceptions.py`
    - _Requirements: 1.1–1.9, 2.1–2.5_

  - [-] 1.2 Implement model download and ONNX conversion
    - Create `src/model_packager.py` with `ModelPackager` class
    - Implement `download_model()` — download pre-trained MNIST PyTorch model from configured source
    - Implement `convert_to_onnx()` — convert PyTorch model to ONNX (opset >= 11) using `torch.onnx.export`
    - Implement `validate_onnx()` — validate with `onnx.checker.check_model`
    - Handle errors: log to stdout, raise typed exceptions
    - _Requirements: 1.1, 1.2, 1.3, 1.7, 1.8_

  - [~] 1.3 Implement Triton model repository creation and packaging
    - Implement `create_model_repository()` — create directory structure (`mnist/config.pbtxt`, `mnist/1/model.onnx`)
    - Generate `config.pbtxt` with platform `onnxruntime_onnx`, input shape [1, 28, 28] FP32, output shape [10] FP32, max_batch_size 8
    - Implement `package_artifact()` — create `model.tar.gz` preserving Triton directory hierarchy
    - _Requirements: 1.4, 1.5, 1.6, 1.9_

  - [ ]* 1.4 Write property test for model artifact packaging round-trip
    - **Property 1: Model artifact packaging round-trip**
    - Generate random valid Triton repository structures, package into tar.gz, extract, verify identical file paths and contents
    - **Validates: Requirements 1.6**

  - [-] 1.5 Implement S3 upload with retry logic
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

- [ ] 2. Layer 1 — SageMaker Endpoint Deployment
  - [~] 2.1 Implement basic endpoint deployer
    - Create `src/endpoint_deployer.py` with `EndpointDeployer` class
    - Implement `get_triton_image_uri()` — return CPU Triton container image URI for eu-west-1
    - Implement `create_model()` — create SageMaker model resource
    - Implement `create_endpoint_config()` — create endpoint config with specified instance type (default: ml.c5.large, 1 instance)
    - Implement `create_endpoint()` — create endpoint, poll until InService (timeout 15 min), handle failures by cleaning partial resources
    - _Requirements: 3.1, 3.2, 3.4, 3.7, 3.8, 3.9_

  - [ ]* 2.2 Write unit tests for endpoint deployer
    - Test Triton image URI retrieval for eu-west-1
    - Test default configuration (single instance, ml.c5.large)
    - Mock SageMaker client to verify API call sequences
    - _Requirements: 3.1, 3.8, 3.9_

- [~] 3. Checkpoint — Layer 1 complete
  - Ensure all tests pass, ask the user if questions arise.
  - At this point, the system can: download model → convert to ONNX → package → upload to S3 → deploy SageMaker endpoint invokable via `boto3.invoke_endpoint()`

- [ ] 4. Layer 2 — Inference Protocol and Validation
  - [~] 4.1 Implement request formatter
    - Create `src/request_formatter.py` with `RequestFormatter` class
    - Implement `format_request()` — convert flat 784-element FP32 array to Triton V2 inference protocol JSON (shape [1, 1, 28, 28], datatype "FP32", name "input")
    - _Requirements: 4.3_

  - [ ]* 4.2 Write property test for Triton request formatting
    - **Property 4: Triton request formatting**
    - Generate random 784-element FP32 arrays, verify output conforms to Triton V2 protocol with correct shape, datatype, name, and all values preserved
    - **Validates: Requirements 4.3**

  - [~] 4.3 Implement response formatter
    - Create `src/response_formatter.py` with `ResponseFormatter` class and `PredictionResponse` dataclass
    - Implement `format_prediction()` — extract argmax as predicted digit, max value as confidence, return full probability distribution
    - _Requirements: 4.1, 4.4, 4.7_

  - [ ]* 4.4 Write property test for response formatting
    - **Property 5: Response formatting produces valid predictions**
    - Generate random 10-element probability distributions (non-negative, sum ≈ 1.0), verify predicted_digit == argmax index and confidence == max value in [0.0, 1.0]
    - **Validates: Requirements 4.1, 4.4**

  - [~] 4.5 Implement input validator
    - Create `src/input_validator.py` with `InputValidator` class and `ValidationResult` dataclass
    - Validate: tensor shape [1, 28, 28], datatype FP32, required fields present, payload size <= 1 MB
    - Return specific error messages identifying which constraint was violated
    - _Requirements: 4.5, 4.6_

  - [ ]* 4.6 Write property test for input validation
    - **Property 6: Input validation rejects all invalid payloads**
    - Generate random invalid inputs (wrong shapes, wrong types, missing fields, oversized payloads), verify all are rejected with specific error messages
    - **Validates: Requirements 4.5**

- [~] 5. Checkpoint — Layer 2 complete
  - Ensure all tests pass, ask the user if questions arise.
  - At this point, the system includes request/response formatting and input validation on top of the working endpoint.

- [ ] 6. Layer 3 — External Access (Lambda Proxy + API Gateway)
  - [~] 6.1 Implement Lambda proxy handler
    - Create `src/lambda_handler.py` with `handler(event, context)` function
    - Extract request body from API Gateway proxy event
    - Invoke SageMaker endpoint via boto3 (IAM SigV4 automatic)
    - Format response for API Gateway proxy integration (statusCode, headers, body)
    - Handle errors: 502 for Lambda/SageMaker failure, pass-through for SageMaker errors
    - _Requirements: 5.1, 5.5, 5.6, 5.8_

  - [ ]* 6.2 Write unit tests for Lambda handler
    - Test event extraction from API Gateway proxy format
    - Test error formatting (502, pass-through)
    - Mock SageMaker runtime client
    - _Requirements: 5.5, 5.8_

  - [~] 6.3 Implement API Gateway setup
    - Create `src/api_gateway_setup.py` with `ApiGatewaySetup` class
    - Implement `deploy()` — create REST API, POST /predict resource, Lambda integration, API key requirement, usage plan (10 rps, burst 20)
    - Output invoke URL and API key value
    - Implement `delete()` — remove all API Gateway resources in dependency order
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.7, 5.9, 5.10_

  - [ ]* 6.4 Write unit tests for API Gateway setup
    - Test resource creation sequence
    - Test default usage plan limits (10 rps, burst 20)
    - Test deletion order
    - _Requirements: 5.3, 5.4, 5.10_

- [~] 7. Checkpoint — Layer 3 complete
  - Ensure all tests pass, ask the user if questions arise.
  - At this point, external applications can call the endpoint via API Gateway with an API key, without needing AWS credentials.

- [ ] 8. Layer 4 — Production Hardening
  - [~] 8.1 Implement instance type validation
    - Add `validate_instance_type()` to `EndpointDeployer`
    - Accept only CPU families: ml.c4, ml.c5, ml.c5d, ml.m4, ml.m5, ml.m5d, ml.t2, ml.t3
    - Reject GPU families (ml.p2, ml.p3, ml.p4, ml.g4dn, ml.g5, ml.inf1) and unrecognized types with descriptive error
    - Wire validation into `create_endpoint_config()` flow
    - _Requirements: 6.1, 6.4, 6.5_

  - [ ]* 8.2 Write property test for instance type validation
    - **Property 7: Instance type validation**
    - Generate random instance type strings (from allowed CPU prefixes, GPU prefixes, and random strings), verify correct accept/reject classification
    - **Validates: Requirements 6.1, 6.4, 6.5**

  - [~] 8.3 Implement auto-scaling configuration
    - Add `configure_auto_scaling()` to `EndpointDeployer`
    - Configure Application Auto Scaling for SageMaker endpoint variant (min 1, max 10, target: invocations per instance)
    - _Requirements: 6.2, 6.3_

  - [~] 8.4 Implement endpoint cleanup with ordered deletion
    - Create `src/cleanup.py` with `CleanupOrchestrator` class
    - Add `delete_endpoint()` to `EndpointDeployer` — delete endpoint → endpoint config → model in dependency order
    - Implement `delete_all()` — delete API Gateway resources, Lambda, SageMaker resources in correct order
    - Continue on individual failures, treat non-existent resources as already deleted
    - Return `DeletionSummary` with per-resource status
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 8.5 Write property test for resilient ordered deletion
    - **Property 8: Resilient ordered deletion with complete summary**
    - Generate random resource sets with random success/failure outcomes, verify: (a) dependency-order deletion, (b) continuation on failure, (c) non-existent treated as success, (d) complete summary
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

- [~] 9. Final checkpoint — All layers complete
  - Ensure all tests pass, ask the user if questions arise.
  - Verify full system: model packaging → S3 upload → SageMaker deployment → request/response formatting → API Gateway with API key auth → auto-scaling → cleanup

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each layer
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases
- The existing `download_mnist.py` in the project root may be referenced or extended for model download logic
- Region is eu-west-1 for all AWS resources

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.5"] },
    { "id": 2, "tasks": ["1.3"] },
    { "id": 3, "tasks": ["1.4", "1.6", "2.1"] },
    { "id": 4, "tasks": ["2.2"] },
    { "id": 5, "tasks": ["4.1", "4.3", "4.5"] },
    { "id": 6, "tasks": ["4.2", "4.4", "4.6"] },
    { "id": 7, "tasks": ["6.1", "6.3"] },
    { "id": 8, "tasks": ["6.2", "6.4"] },
    { "id": 9, "tasks": ["8.1", "8.3", "8.4"] },
    { "id": 10, "tasks": ["8.2", "8.5"] }
  ]
}
```
