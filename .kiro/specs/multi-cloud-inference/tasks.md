# Implementation Plan: Multi-Cloud Inference

## Overview

This plan implements multi-cloud MNIST inference across four layers: Alibaba Cloud infrastructure (ROS CDK stacks for OSS, PAI-EAS, and access), unified inference client abstraction, tkinter desktop application with provider switching, and reliability/error handling. Tasks are ordered so that foundational abstractions (data models, interfaces) and infrastructure deployment come first, followed by provider implementations, then the desktop app that wires everything together. All Alibaba Cloud infrastructure is managed exclusively via ROS CDK - there is no imperative deployer.

## Tasks

- [x] 1. Set up project structure and core data models
  - [x] 1.1 Create inference module package structure and data models
    - Create `src/inference/__init__.py`, `src/inference/providers/__init__.py`
    - Create `src/inference/models.py` with `UnifiedResponse` and `ProviderConfig` dataclasses
    - Create `src/inference/errors.py` with `ErrorCategory` enum, `InferenceError` exception, and `categorize_error` function
    - Create `src/app/__init__.py`
    - _Requirements: 5.3, 5.6_

  - [x] 1.2 Create Alibaba Cloud infrastructure package structure
    - Create `infra_alicloud/__init__.py`, `infra_alicloud/stacks/__init__.py`
    - Create `infra_alicloud/cdk.json` with ROS CDK project configuration
    - Create `infra_alicloud/app.py` as the ROS CDK app entry point
    - Create `infra_alicloud/config.py` with `AlicloudDeployConfig`, `ALLOWED_ALICLOUD_GPU_FAMILIES`, and `VALID_ALICLOUD_REGIONS` constants
    - _Requirements: 4.1, 2.1, 2.2_

- [ ] 2. Implement Alibaba Cloud ROS CDK infrastructure stacks
  - [ ] 2.1 Implement storage stack (OSS bucket)
    - Create `infra_alicloud/stacks/storage_stack.py` with `StorageStack` class using ROS CDK
    - Provision OSS bucket with server-side encryption (AES256)
    - Export `bucket_name` as ROS Output for cross-stack reference
    - _Requirements: 4.2, 4.6_

  - [ ] 2.2 Implement inference stack (PAI-EAS service)
    - Create `infra_alicloud/stacks/eas_stack.py` with `EasStack` class using ROS CDK
    - Accept `oss_bucket_name` and `model_key` as inputs from storage stack
    - Configure PAI-EAS service with Triton Inference Server image, GPU instance, spot preference, auto-scaling
    - Export `service_name` and `endpoint_url` as ROS Outputs
    - _Requirements: 4.3, 4.5, 4.6_

  - [ ] 2.3 Implement access stack (authentication and network)
    - Create `infra_alicloud/stacks/access_stack.py` with `AccessStack` class using ROS CDK
    - Accept `service_name` from inference stack
    - Configure token-based authentication and public HTTPS access
    - Export `access_token` and `public_endpoint` as ROS Outputs
    - _Requirements: 4.4, 4.5, 4.6_

  - [ ]* 2.4 Write unit tests for ROS CDK stack synthesis
    - Test storage stack synthesizes ROS template with encrypted OSS bucket
    - Test EAS stack synthesizes template with correct PAI-EAS service configuration
    - Test access stack outputs endpoint URL and token
    - Test cross-stack references are correctly wired
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 3. Implement input validation and preprocessing
  - [x] 3.1 Implement input validation logic
    - Create `src/inference/validation.py` with `validate_input` function
    - Accept numpy arrays of shape (28, 28), (1, 28, 28), or flat list of exactly 784 numeric values
    - Validate all values are in range [0.0, 1.0]
    - Raise `InputValidationError` with descriptive message for invalid input (wrong shape or out-of-range values)
    - Return flat list of 784 floats for valid input
    - _Requirements: 5.2, 5.8, 5.9_

  - [ ]* 3.2 Write property test for input validation
    - **Property 5: Inference client input validation**
    - Generate random numpy arrays of valid shapes (28,28), (1,28,28) and flat lists of 784 values with values in [0.0, 1.0] - assert acceptance and correct output length
    - Generate random inputs with wrong shapes or out-of-range values - assert InputValidationError is raised
    - **Validates: Requirements 5.2, 5.8, 5.9**

  - [x] 3.3 Implement canvas preprocessing
    - Create `src/app/preprocessing.py` with `preprocess_canvas` function
    - Extract logic from existing `model/draw_digit.py` `DigitCanvas.preprocess()` method
    - Resize PIL image to 28x28 using LANCZOS interpolation
    - Convert to numpy float32, normalize by dividing by 255.0, flatten to 784 floats
    - _Requirements: 7.3_

  - [ ]* 3.4 Write property test for canvas preprocessing
    - **Property 10: Canvas preprocessing invariant**
    - Generate random PIL images of arbitrary dimensions with pixel values 0-255
    - Assert output is always a flat list of exactly 784 floats, each in [0.0, 1.0]
    - **Validates: Requirements 7.3**

- [ ] 4. Implement provider backends
  - [x] 4.1 Implement provider backend abstract base class
    - Create `src/inference/providers/base.py` with `ProviderBackend` ABC
    - Define abstract methods: `send_request(pixel_data: list[float]) -> UnifiedResponse`, `health_check(timeout: float = 5.0) -> bool`, `provider_name() -> str`
    - _Requirements: 5.1_

  - [x] 4.2 Implement AWS backend provider
    - Create `src/inference/providers/aws.py` with `AWSBackend` class
    - Build Triton V2 JSON payload with shape [1, 1, 28, 28] and datatype FP32
    - Send via HTTPS with `x-api-key` header and `Content-Type: application/json`
    - Parse response `outputs[0].data` into 10-element probability array
    - Construct `UnifiedResponse` from probability array (argmax for digit, max for confidence)
    - Implement health check via lightweight request with timeout
    - Use `categorize_error` for all exceptions
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

  - [x] 4.3 Implement Alibaba Cloud backend provider
    - Create `src/inference/providers/alicloud.py` with `AlicloudBackend` class
    - Build same Triton V2 JSON payload as AWS (shape [1, 1, 28, 28], FP32)
    - Send via HTTPS with `Authorization` header and `Content-Type: application/json`
    - Parse response `outputs[0].data` into 10-element probability array
    - Construct `UnifiedResponse` from probability array
    - Implement health check via GET to `/v2/health/ready` with timeout
    - Use `categorize_error` for all exceptions
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

  - [ ]* 4.4 Write property test for response normalization
    - **Property 6: Response normalization**
    - Generate random 10-element non-negative arrays summing to ~1.0
    - Assert `predicted_digit` equals argmax, `confidence` equals max value, `probabilities` is full 10-element list
    - **Validates: Requirements 5.3, 5.5**

  - [ ]* 4.5 Write property test for request format translation
    - **Property 7: Request format translation**
    - Generate random 784-element float arrays in [0.0, 1.0]
    - Assert both AWS and Alicloud translators produce valid Triton V2 JSON with shape [1,1,28,28] and FP32 datatype
    - Assert payloads are structurally identical and preserve exact input values
    - **Validates: Requirements 5.4**

- [ ] 5. Implement inference client and error handling
  - [ ] 5.1 Implement the InferenceClient class
    - Create `src/inference/client.py` with `InferenceClient` class
    - Accept a `ProviderBackend` at initialization
    - Implement `predict(input_data)`: validate input, call provider `send_request`, return `UnifiedResponse`
    - Implement `switch_provider(provider)` for in-memory provider swap
    - Implement `health_check()` delegating to active provider with 5s timeout
    - Expose `active_provider` property returning the provider name string
    - Wrap all provider exceptions with `categorize_error` before raising
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.7_

  - [ ]* 5.2 Write property test for error categorization
    - **Property 8: Error categorization**
    - Generate random HTTP status codes and exception types (ConnectionError, TimeoutError, etc.)
    - Assert 401/403 -> AUTHENTICATION, timeout -> TIMEOUT, connection errors -> NETWORK, 5xx -> SERVER_ERROR, others -> UNKNOWN
    - Assert error message never contains provider-specific URLs or tokens
    - **Validates: Requirements 5.6, 10.1, 10.3, 10.4, 10.7**

- [ ] 6. Checkpoint - Core inference client + infrastructure
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement configuration management
  - [ ] 7.1 Implement ConfigManager with keyring integration
    - Create `src/app/config_manager.py` with `ConfigManager` class
    - Determine OS-specific config path (XDG_CONFIG_HOME, %APPDATA%, ~/Library/Application Support)
    - Implement `load()`: read JSON config file, retrieve credentials from OS keyring, return dict of ProviderConfig objects
    - Implement `save()`: write JSON atomically (write to temp file, rename), store credentials in OS keyring
    - Implement `store_credential`, `get_credential`, `delete_credential` using `keyring` library
    - Handle missing/corrupt config file gracefully (return empty config)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 6.8_

  - [ ]* 7.2 Write property test for config serialization round-trip
    - **Property 9: Provider config serialization round-trip**
    - Generate random valid ProviderConfig objects (valid provider_id, endpoint_url up to 2048 chars, region up to 64 chars)
    - Assert serialize then deserialize produces equivalent object
    - **Validates: Requirements 6.1**

  - [ ]* 7.3 Write property test for URL validation
    - **Property 13: URL format validation**
    - Generate random valid URLs (scheme://host with optional port/path) and invalid strings
    - Assert valid URLs are accepted, invalid strings are rejected
    - **Validates: Requirements 9.7**

  - [ ]* 7.4 Write property test for form field validation
    - **Property 14: Provider form required fields validation**
    - Generate random pairs of (endpoint_url, credential) strings including empty and whitespace-only
    - Assert save is enabled only when both are non-empty after trimming
    - **Validates: Requirements 6.6, 9.2, 9.3**

- [ ] 8. Implement desktop application
  - [ ] 8.1 Implement provider switcher UI component
    - Create `src/app/provider_switcher.py` with `ProviderSwitcher` class
    - Render radio buttons for each configured provider with availability status indicator
    - Call `on_switch` callback when selection changes
    - Implement `update_status(provider_id, available)` to update indicator
    - Implement `refresh_providers(providers)` to rebuild list from configuration
    - _Requirements: 8.1, 8.2, 8.5_

  - [ ] 8.2 Implement settings dialog
    - Create `src/app/settings_dialog.py` with `SettingsDialog` class (tkinter Toplevel modal)
    - Support add/edit/remove provider configurations (up to 10)
    - Show AWS fields: endpoint URL, API key (masked by default with toggle)
    - Show Alicloud fields: endpoint URL, access token (masked by default with toggle)
    - Validate URL format inline, disable save for empty required fields
    - Test connectivity on save (10s timeout), show success/failure with response time
    - Allow saving even if connectivity fails (for offline setup)
    - Persist via ConfigManager on save
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ] 8.3 Implement main application window
    - Create `src/app/main.py` with `MnistApp` class based on existing `model/draw_digit.py` DigitCanvas
    - Render 280x280 canvas with white-on-black drawing (stroke width 15-25px)
    - Integrate `ProviderSwitcher`, results display area, Predict/Clear buttons
    - Wire Settings button to open `SettingsDialog`
    - On Predict: check canvas not blank, check provider configured, preprocess, send inference, display result
    - On Clear: reset canvas and results
    - Display predicted digit, confidence percentage (1 decimal), and provider name in results
    - On startup: load config, select first available provider, perform health check
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 8.1, 8.2, 8.3, 8.5, 8.6, 8.8, 8.9_

  - [ ] 8.4 Implement provider switching behavior
    - Wire provider switcher to `InferenceClient.switch_provider()`
    - On switch: perform health check (5s timeout), update status indicator
    - If unreachable: show warning with Retry and Switch Provider options
    - If inference in-flight during switch: complete on original provider, then route subsequent to new
    - Provider switch completes within 100ms (in-memory state change)
    - _Requirements: 8.2, 8.3, 8.4, 8.7_

- [ ] 9. Implement error handling and logging in desktop app
  - [ ] 9.1 Implement error display and user actions
    - On 401/403: display credentials invalid message, suggest checking config, re-enable Predict within 1s
    - On timeout (10s): display timeout message with Retry and Switch Provider buttons
    - On network error: display connection problem message
    - On 5xx: display service unavailable with Retry button
    - On unknown: display generic error with error code if available
    - Show loading indicator and disable Predict while request in-flight
    - Replace previous error message on new error
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.9_

  - [ ] 9.2 Implement error logging to local file
    - Determine platform-appropriate log directory (XDG_DATA_HOME, %LOCALAPPDATA%, ~/Library/Application Support)
    - Write JSON log entries with: ISO 8601 timestamp, provider name, endpoint URL, error code, error category, message
    - Create log directory if it does not exist
    - _Requirements: 10.8_

  - [ ]* 9.3 Write property test for error log entry completeness
    - **Property 12: Error log entry completeness**
    - Generate random inference error events
    - Assert log entry contains all required fields: non-empty provider name, valid URL, error code (int or null), valid ErrorCategory, ISO 8601 timestamp
    - **Validates: Requirements 10.8**

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The desktop app (Layer 3) builds on the inference client (Layer 2), which builds on provider backends and validation
- ROS CDK stacks (Layer 1) should be deployed early so infrastructure is ready when the inference client needs to connect
- The existing `model/draw_digit.py` DigitCanvas class is the reference for canvas drawing, preprocessing, and Triton V2 payload logic
- Alibaba Cloud ROS CDK dependencies: `ros-cdk-core`, `ros-cdk-oss`, `ros-cdk-pai`
- Desktop app dependencies: `keyring` (credential storage), `Pillow` (image processing), `numpy`
- Both AWS and Alicloud use identical Triton V2 inference protocol - only auth headers differ
- The `src/alicloud_deployer.py` file from the imperative approach should be deleted - all infrastructure is managed via ROS CDK stacks

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "3.3", "4.1"] },
    { "id": 3, "tasks": ["2.4", "3.2", "3.4", "4.2", "4.3"] },
    { "id": 4, "tasks": ["4.4", "4.5", "5.1"] },
    { "id": 5, "tasks": ["5.2", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4", "8.1", "8.2"] },
    { "id": 7, "tasks": ["8.3", "8.4"] },
    { "id": 8, "tasks": ["9.1", "9.2"] },
    { "id": 9, "tasks": ["9.3"] }
  ]
}
```
