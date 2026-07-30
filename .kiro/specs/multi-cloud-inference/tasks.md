# Implementation Plan: Multi-Cloud Inference

## Overview

This plan implements multi-cloud MNIST inference with ROS CDK infrastructure as the first priority, followed by the shared inference client and desktop application. All Alibaba Cloud infrastructure is managed via ROS CDK stacks. The imperative deployer has been removed.

Tasks are ordered: cleanup obsolete files -> upload model to OSS -> deploy infrastructure (CDK stacks) -> verify endpoints -> build inference client -> build desktop app.

## Tasks

- [x] 1. Clean up obsolete files from imperative approach
  - [x] 1.1 Remove imperative deployer and rename scripts
    - Delete `src/alicloud_deployer.py` (imperative SDK deployer - replaced by ROS CDK stacks)
    - Delete `src/__pycache__/` directory (stale bytecode)
    - Delete `infra_alicloud/__pycache__/` directory (stale bytecode)
    - Delete `cdk.out/` at project root (stale CDK output - real output is at `infra_aws/cdk.out/`)
    - Rename `scripts/test_endpoint.py` to `scripts/test_endpoint_aws.py`
    - Create `scripts/test_endpoint_alicloud.py` (Triton V2 test against PAI-EAS endpoint)
    - Verify `infra_alicloud/config.py` still contains `ALLOWED_ALICLOUD_GPU_FAMILIES` and `VALID_ALICLOUD_REGIONS` (used by CDK stacks)

- [x] 2. Create model upload script for Alibaba Cloud OSS
  - [x] 2.1 Create scripts/upload_model_alicloud.py
    - Equivalent of `scripts/package_model.py` for AWS, but uploads to Alibaba Cloud OSS
    - Reads OSS bucket name and region from env vars (`OSS_BUCKET`, `ALICLOUD_REGION`) or CLI args
    - Uses `oss2` SDK to upload the Triton model repository (`model/triton_repo/`) to OSS
    - Uploads all files preserving directory structure: `models/mnist/config.pbtxt`, `models/mnist/1/model.onnx`
    - Prints the OSS URI on success (e.g. `oss://bucket-name/models/mnist/`)
    - Exits with non-zero code on failure
    - **Run before deploying CDK stacks** since EasStack references the model path in OSS
    - Usage:
      ```bash
      export OSS_BUCKET="mnist-model-artifacts-alicloud"
      export ALICLOUD_REGION="cn-hangzhou"
      export ALIBABA_CLOUD_ACCESS_KEY_ID="..."
      export ALIBABA_CLOUD_ACCESS_KEY_SECRET="..."
      uv run python scripts/upload_model_alicloud.py
      ```
  - [x] 2.2 Add oss2 to pyproject.toml optional dependencies
    - Add `alicloud` optional dependency group: `oss2>=2.18.0`
    - Install with: `uv sync --extra alicloud`

- [ ] 3. Implement Alibaba Cloud ROS CDK infrastructure stacks
  - [x] 3.1 Add cost-tracking tags to all ROS CDK stacks
    - Apply tags to all resources across all stacks for cost tracking:
      - `project: mnist-inference`
      - `environment: dev`
      - `managed-by: ros-cdk`
      - `team: platform-engineering`
    - Use `ros.Tags.of(stack).add(key, value)` or equivalent ROS CDK tagging mechanism
    - Ensure tags propagate to all child resources (OSS bucket, PAI-EAS service, etc.)
    - Also add tags to the existing AWS CDK stacks in `infra_aws/` for consistency:
      - `project: mnist-inference`
      - `environment: dev`
      - `managed-by: aws-cdk`
      - `team: platform-engineering`

  - [x] 3.2 Implement storage stack (OSS bucket)
    - Create `infra_alicloud/stacks/storage_stack.py` with `StorageStack` class using ROS CDK
    - Provision OSS bucket with server-side encryption (AES256)
    - Add lifecycle rules for cost management (e.g. expire incomplete multipart uploads after 7 days)
    - Export `bucket_name` as ROS Output for cross-stack reference
    - _Requirements: 4.2, 4.6_

  - [X] 3.3 Implement inference stack (PAI-EAS service)
    - Create `infra_alicloud/stacks/eas_stack.py` with `EasStack` class using ROS CDK
    - Accept `oss_bucket_name` and `model_key` as inputs from storage stack
    - Configure PAI-EAS service with Triton Inference Server image, GPU instance, spot preference, auto-scaling
    - Export `service_name` and `endpoint_url` as ROS Outputs
    - _Requirements: 4.3, 4.5, 4.6_

  - [X] 3.4 Implement access stack (authentication and network)
    - Create `infra_alicloud/stacks/access_stack.py` with `AccessStack` class using ROS CDK
    - Accept `service_name` from inference stack
    - Configure token-based authentication and public HTTPS access
    - Export `access_token` and `public_endpoint` as ROS Outputs
    - _Requirements: 4.4, 4.5, 4.6_

  - [X] 3.5 Wire stacks together in app.py
    - Update `infra_alicloud/app.py` to instantiate StorageStack, EasStack, AccessStack with correct cross-stack references
    - Ensure stack dependency order: StorageStack -> EasStack -> AccessStack
    - _Requirements: 4.6_

  - [ ]* 3.6 Write unit tests for ROS CDK stack synthesis
    - Test storage stack synthesizes ROS template with encrypted OSS bucket
    - Test EAS stack synthesizes template with correct PAI-EAS service configuration
    - Test access stack outputs endpoint URL and token
    - Test cross-stack references are correctly wired
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 4. Deploy and verify Alibaba Cloud infrastructure
  - [ ] 4.1 Deploy ROS CDK stacks
    - **Prerequisite**: Model must already be uploaded to OSS (task 2)
      ```bash
      cd infra_alicloud
      ros-cdk deploy --all
      ```
    - Note the stack outputs: `endpoint_url`, `access_token`, `bucket_name`
    - **Verify**: Check Alibaba Cloud console that OSS bucket exists and PAI-EAS service is Running

  - [ ] 4.2 Create and run test_endpoint_alicloud.py
    - Create `scripts/test_endpoint_alicloud.py` that:
      - Reads endpoint URL and token from env vars (`ALICLOUD_ENDPOINT_URL`, `ALICLOUD_API_TOKEN`)
      - Builds a Triton V2 JSON payload with a sample 784-float input (e.g. a known digit)
      - Sends POST with `Authorization` header
      - Prints predicted digit, confidence, and response time
      - Exits 0 on success, 1 on failure
    - **Run**:
      ```bash
      export ALICLOUD_ENDPOINT_URL="<endpoint_url from stack output>"
      export ALICLOUD_API_TOKEN="<access_token from stack output>"
      uv run python scripts/test_endpoint_alicloud.py
      ```
    - **Expected**: Prints predicted digit and confidence percentage

  - [ ] 4.3 Rename and verify AWS test script
    - Rename `scripts/test_endpoint.py` to `scripts/test_endpoint_aws.py`
    - Verify it still works:
      ```bash
      uv run python scripts/test_endpoint_aws.py
      ```
    - **Expected**: Prints predicted digit and confidence from AWS SageMaker endpoint

  - [ ] 4.4 Verify both endpoints side-by-side
    - Run both test scripts with the same sample input
    - Confirm both return the same predicted digit (may differ slightly in confidence)
    - **This confirms the infrastructure layer is complete and both backends serve the same model**

- [ ] 5. Implement shared Triton V2 payload builder
  - [ ] 5.1 Create payload builder module
    - Create `src/inference/payload.py` with `build_triton_payload(pixel_data: list[float]) -> dict`
    - Returns `{"inputs": [{"name": "input", "shape": [1, 1, 28, 28], "datatype": "FP32", "data": pixel_data}]}`
    - Add `parse_triton_response(response_json: dict) -> tuple[int, float, list[float]]` to extract digit, confidence, probabilities from response
    - _Requirements: 5.4_

- [ ] 6. Implement provider backends and inference client
  - [ ] 6.1 Update provider backend ABC to accept pre-built payload
    - Update `src/inference/providers/base.py`: `send_request(self, payload: dict) -> UnifiedResponse`
    - Providers receive the Triton V2 payload dict and only add their auth header + send
    - _Requirements: 5.1_

  - [ ] 6.2 Update AWS backend to use shared payload builder
    - Update `src/inference/providers/aws.py` to call `build_triton_payload()` and send with `x-api-key` header
    - Use `parse_triton_response()` to construct UnifiedResponse
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

  - [ ] 6.3 Update Alicloud backend to use shared payload builder
    - Update `src/inference/providers/alicloud.py` to call `build_triton_payload()` and send with `Authorization` header
    - Use `parse_triton_response()` to construct UnifiedResponse
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

  - [ ] 6.4 Implement InferenceClient class
    - Create `src/inference/client.py` with `InferenceClient` class
    - `predict(input_data)`: validate input -> build payload -> send via provider -> return UnifiedResponse
    - `switch_provider(provider)`: in-memory provider swap
    - `health_check()`: delegate to active provider with 5s timeout
    - Wrap all provider exceptions with `categorize_error`
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.7_

  - [ ]* 6.5 Write property tests for response normalization and error categorization
    - **Property 5: Response normalization** - random 10-element arrays, assert argmax/max/list
    - **Property 6: Error categorization** - random status codes, assert correct categories
    - **Validates: Requirements 5.3, 5.5, 5.6**

- [ ] 7. Checkpoint - Verify inference client works end-to-end
  - Run a quick integration check:
    ```python
    from src.inference.client import InferenceClient
    from src.inference.providers.aws import AWSBackend
    
    backend = AWSBackend(endpoint_url="...", api_key="...")
    client = InferenceClient(backend)
    # Use a sample 784-float input
    result = client.predict(sample_data)
    print(f"Digit: {result.predicted_digit}, Confidence: {result.confidence}")
    ```
  - Ensure all existing tests still pass: `uv run pytest`

- [ ] 8. Implement configuration management
  - [ ] 8.1 Implement ConfigManager with keyring integration
    - Create `src/app/config_manager.py` with `ConfigManager` class
    - Determine OS-specific config path (~/Library/Application Support/mnist-inference/ on macOS)
    - Implement `load()`, `save()`, `store_credential()`, `get_credential()`, `delete_credential()`
    - Handle missing/corrupt config file gracefully (return empty config)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 6.8_

  - [ ]* 8.2 Write property tests for config round-trip and URL validation
    - **Property 7: Config serialization round-trip**
    - **Property 10: URL format validation**
    - **Validates: Requirements 6.1, 9.7**

- [ ] 9. Implement desktop application
  - [ ] 9.1 Implement provider switcher UI component
    - Create `src/app/provider_switcher.py` with `ProviderSwitcher` class
    - Radio buttons with availability status indicators
    - _Requirements: 8.1, 8.2, 8.5_

  - [ ] 9.2 Implement settings dialog
    - Create `src/app/settings_dialog.py` with `SettingsDialog` class
    - Add/edit/remove providers, masked credential fields, connectivity test
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ] 9.3 Implement main application window
    - Create `src/app/main.py` with `MnistApp` class
    - 280x280 canvas, provider switcher, predict/clear buttons, results display
    - On predict: preprocess -> build_triton_payload -> send via InferenceClient -> display
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 8.1, 8.2, 8.3, 8.5, 8.6, 8.8, 8.9_

  - [ ] 9.4 Implement provider switching behavior
    - Wire to InferenceClient.switch_provider(), health check on switch
    - _Requirements: 8.2, 8.3, 8.4, 8.7_

- [ ] 10. Implement error handling and logging
  - [ ] 10.1 Implement error display and user actions
    - Categorized error messages in UI with retry/switch provider options
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.9_

  - [ ] 10.2 Implement error logging to local file
    - JSON log entries to ~/Library/Application Support/mnist-inference/inference.log
    - _Requirements: 10.8_

  - [ ]* 10.3 Write property test for error log entry completeness
    - **Property 9: Error log entry completeness**
    - **Validates: Requirements 10.8**

- [ ] 11. Final checkpoint and cleanup
  - Run full test suite: `uv run pytest`
  - Verify desktop app launches: `uv run python -m src.app.main`
  - Test with both providers connected
  - Verify `ros-cdk destroy --all` cleans up Alicloud resources when done

## Deployment Instructions

### Prerequisites

- Python >= 3.10 with uv installed
- AWS CDK already deployed (existing stacks: MnistStorageStack, MnistSageMakerStack, MnistApiStack)
- Alibaba Cloud CLI configured with credentials (`aliyun configure`)
- ROS CDK installed: `pip install ros-cdk-cli`

### Deploy Alibaba Cloud Infrastructure

```bash
# 1. Install dependencies (includes oss2 for model upload)
uv sync --extra alicloud
cd infra_alicloud
pip install ros-cdk-core ros-cdk-oss ros-cdk-pai

# 2. Upload model to OSS first (required before CDK deploy)
cd ..
export OSS_BUCKET="mnist-model-artifacts-alicloud"
export ALICLOUD_REGION="cn-hangzhou"
export ALIBABA_CLOUD_ACCESS_KEY_ID="..."
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="..."
uv run python scripts/upload_model_alicloud.py

# 3. Synthesize templates (dry run)
cd infra_alicloud
ros-cdk synth

# 4. Deploy all stacks
ros-cdk deploy --all

# 5. Note the outputs
# StorageStack.bucket_name = mnist-model-artifacts-alicloud
# EasStack.endpoint_url = https://xxx.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/mnist_triton
# AccessStack.access_token = <token>
```

### Verify Endpoints

```bash
# Test Alicloud
export ALICLOUD_ENDPOINT_URL="<endpoint_url from deploy output>"
export ALICLOUD_API_TOKEN="<token from deploy output>"
uv run python scripts/test_endpoint_alicloud.py

# Test AWS (existing)
uv run python scripts/test_endpoint_aws.py
```

### Tear Down (when done testing)

```bash
# Destroy Alicloud resources (stops billing)
cd infra_alicloud
ros-cdk destroy --all

# AWS (optional - only if you want to stop AWS billing too)
cd infra_aws
cdk destroy --all
```

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- **CDK stacks have top priority** - deploy infrastructure first so you can verify endpoints before writing client code
- `scripts/upload_model_alicloud.py` uploads model files to OSS (equivalent of package_model.py for AWS)
- All CDK resources are tagged with `project`, `environment`, `managed-by`, and `team` for cost tracking
- Model must be uploaded to OSS BEFORE deploying CDK stacks (PAI-EAS references the model path)
- `scripts/test_endpoint_aws.py` (renamed from test_endpoint.py) tests the AWS SageMaker endpoint
- `scripts/test_endpoint_alicloud.py` (new) tests the Alibaba Cloud PAI-EAS endpoint
- Both test scripts use the same Triton V2 payload format - only URL and auth header differ
- The `src/alicloud_deployer.py` file is deleted - all infrastructure is managed via ROS CDK stacks
- Alibaba Cloud ROS CDK dependencies: `ros-cdk-core`, `ros-cdk-oss`, `ros-cdk-pai`
- Desktop app dependencies: `keyring` (credential storage), `Pillow` (image processing), `numpy`
- Both AWS and Alicloud use identical Triton V2 inference protocol - only auth headers differ
- The shared `build_triton_payload()` function in `src/inference/payload.py` eliminates code duplication between providers

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["3.1", "3.2"] },
    { "id": 3, "tasks": ["3.3", "3.4", "3.5"] },
    { "id": 4, "tasks": ["3.6", "4.1", "4.2", "4.3"] },
    { "id": 5, "tasks": ["4.4", "5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3", "5.4"] },
    { "id": 7, "tasks": ["6.1", "6.2"] },
    { "id": 8, "tasks": ["7.1"] },
    { "id": 9, "tasks": ["7.2", "8.1", "8.2"] },
    { "id": 10, "tasks": ["8.3", "8.4"] },
    { "id": 11, "tasks": ["9.1", "9.2", "9.3"] }
  ]
}
```
