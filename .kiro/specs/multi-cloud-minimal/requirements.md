# Requirements Document

## Introduction

Minimal extension to the existing MNIST DigitCanvas tkinter application to support multi-cloud inference. The app currently calls a hardcoded local Triton URL. This feature adds the ability to call either AWS API Gateway (with x-api-key auth) or Alibaba Cloud PAI-EAS (with Authorization token) endpoints. Both use the identical Triton V2 JSON protocol. Configuration is via a simple JSON config file or environment variables.

## Glossary

- **DigitCanvas**: The existing tkinter application class that provides a drawing surface and inference button
- **Provider**: A cloud inference backend (AWS or Alicloud) identified by its URL and auth header
- **Config_File**: A JSON file (`config.json`) at the project root containing provider URLs and API keys
- **Triton_V2_Payload**: The JSON body format `{"inputs": [{"name": "input", "shape": [1,1,28,28], "datatype": "FP32", "data": [...]}]}`
- **PAI_EAS**: Alibaba Cloud Platform for AI - Elastic Algorithm Service, hosting Triton endpoints

## Requirements

### Requirement 1: Provider Configuration

**User Story:** As a developer, I want to store provider endpoint URLs and API keys in a simple config file, so that I can switch between cloud providers without modifying code.

#### Acceptance Criteria

1. THE Config_Loader SHALL read provider settings from a `config.json` file at the project root
2. WHEN `config.json` is missing or malformed, THE Config_Loader SHALL fall back to environment variables (`AWS_ENDPOINT_URL`, `AWS_API_KEY`, `ALICLOUD_ENDPOINT_URL`, `ALICLOUD_API_TOKEN`)
3. THE Config_File SHALL contain a JSON object with keys `aws` and `alicloud`, each having `url` and `key` string fields
4. WHEN both `config.json` and environment variables are absent for a provider, THE Config_Loader SHALL return empty strings for that provider's settings

### Requirement 2: Provider Selection in UI

**User Story:** As a user, I want to select which cloud provider to use for inference directly in the drawing app, so that I can easily switch between AWS and Alicloud.

#### Acceptance Criteria

1. WHEN the DigitCanvas starts, THE DigitCanvas SHALL display radio buttons for provider selection (AWS, Alicloud)
2. THE DigitCanvas SHALL default the provider selection to the first provider that has a non-empty URL configured
3. WHEN a user selects a different provider, THE DigitCanvas SHALL use that provider for the next prediction request

### Requirement 3: Multi-Cloud Inference Request

**User Story:** As a user, I want my drawn digit to be sent to the selected cloud provider for inference, so that I get predictions regardless of which cloud I use.

#### Acceptance Criteria

1. WHEN the user triggers prediction with AWS selected, THE DigitCanvas SHALL send the Triton_V2_Payload to the AWS endpoint URL with an `x-api-key` header
2. WHEN the user triggers prediction with Alicloud selected, THE DigitCanvas SHALL send the Triton_V2_Payload to the Alicloud endpoint URL with an `Authorization` header
3. THE DigitCanvas SHALL display the predicted digit and confidence percentage in the result label
4. IF the HTTP request fails or times out, THEN THE DigitCanvas SHALL display a user-friendly error message in the result label

### Requirement 4: Alicloud PAI-EAS Deploy Script

**User Story:** As a developer, I want a simple Python script to deploy the ONNX model to Alibaba Cloud PAI-EAS, so that I can set up the Alicloud endpoint quickly without a full IaC framework.

#### Acceptance Criteria

1. THE Deploy_Script SHALL use the Alibaba Cloud Python SDK to create a PAI-EAS Triton inference service
2. THE Deploy_Script SHALL accept the model S3/OSS path and service name as command-line arguments or environment variables
3. WHEN deployment succeeds, THE Deploy_Script SHALL print the service endpoint URL and access token
4. IF deployment fails, THEN THE Deploy_Script SHALL print the error message and exit with a non-zero code
