# Requirements Document

## Introduction

This feature extends the existing MNIST inference endpoint to support multiple cloud providers. Currently the system deploys an MNIST model on AWS SageMaker with Triton Inference Server, exposed via API Gateway with API key authentication. This feature adds equivalent inference capability on Alibaba Cloud using PAI-EAS (Platform for AI - Elastic Algorithm Service) and provides a desktop application that allows users to switch between AWS and Alibaba Cloud providers with a single click, presenting a unified interface for sending inference requests regardless of which provider is active.

**Note:** This is a test/development deployment optimized for cost. Cost optimization via spot/preemptible instances takes priority over high availability. Services may experience brief interruptions when spot instances are reclaimed, which is acceptable for this non-production environment.

## Glossary

- **Desktop_App**: A cross-platform desktop application (built with Python and a GUI framework) that provides a unified interface for drawing digits and sending inference requests to either AWS or Alibaba Cloud
- **Provider_Switcher**: The component within the Desktop_App responsible for managing active cloud provider selection and routing inference requests to the correct backend
- **Cloud_Provider**: An abstraction representing a cloud inference backend (either AWS SageMaker or Alibaba Cloud PAI-EAS)
- **PAI_EAS**: Alibaba Cloud Platform for AI - Elastic Algorithm Service, a managed model serving platform that hosts ONNX models and serves predictions via HTTP endpoints
- **Alicloud_Deployer**: The component responsible for deploying the MNIST ONNX model to Alibaba Cloud PAI-EAS and managing the service lifecycle
- **Inference_Client**: The abstraction layer within the Desktop_App that sends inference requests to the active Cloud_Provider and normalizes responses into a common format
- **Provider_Config**: A configuration object storing endpoint URLs, authentication credentials, and connection parameters for a specific Cloud_Provider
- **Unified_Response**: A normalized prediction result containing the predicted digit (0-9) and confidence scores, independent of which Cloud_Provider produced it
- **Canvas_Widget**: The drawing area in the Desktop_App where users draw digits with a mouse or stylus for inference
- **AWS_Backend**: The existing AWS SageMaker endpoint with API Gateway and API key authentication serving MNIST predictions via Triton V2 protocol
- **Alicloud_Backend**: The Alibaba Cloud PAI-EAS service serving MNIST predictions via its HTTP inference API
- **Spot_Instance**: A discounted compute instance (called "preemptible instance" on Alibaba Cloud) that can be reclaimed by the cloud provider with short notice when capacity is needed, offering significant cost savings over on-demand instances in exchange for reduced availability guarantees

## Requirements

Requirements are organized in layers. Layer 1 delivers the Alibaba Cloud deployment capability. Layer 2 provides the unified inference client abstraction. Layer 3 delivers the desktop application with provider switching.

---

### Layer 1 - Alibaba Cloud Deployment

### Requirement 1: Deploy MNIST Model to Alibaba Cloud PAI-EAS

**User Story:** As an ML engineer, I want to deploy the same MNIST ONNX model to Alibaba Cloud PAI-EAS, so that I have an alternative cloud provider for inference.

#### Acceptance Criteria

1. WHEN the Alicloud_Deployer is executed with a valid ONNX model path, THE Alicloud_Deployer SHALL upload the MNIST ONNX model to Alibaba Cloud OSS (Object Storage Service) in a configured bucket and prefix
2. WHEN the model is uploaded to OSS, THE Alicloud_Deployer SHALL create a PAI-EAS inference service configured to serve the ONNX model using the ONNX Runtime backend
3. THE PAI-EAS service SHALL accept inference requests containing a flat array of exactly 784 FP32 values in the range 0.0 to 1.0 representing a 28x28 grayscale digit image
4. WHEN the PAI-EAS service receives a valid inference request, THE PAI-EAS service SHALL return a JSON response containing a 10-element probability distribution array corresponding to digits 0-9 within 5 seconds
5. THE PAI-EAS service SHALL be accessible via an HTTPS endpoint with token-based authentication
6. IF the OSS upload fails, THEN THE Alicloud_Deployer SHALL retry up to 3 times with a minimum delay of 1 second between attempts before raising an error
7. IF the PAI-EAS service fails to reach a running state within 10 minutes, THEN THE Alicloud_Deployer SHALL report the failure reason and clean up any partially created resources
8. WHEN deployment completes successfully, THE Alicloud_Deployer SHALL output the service endpoint URL and access token to standard output
9. IF the PAI-EAS service receives a request with an array length other than 784 or containing values outside the range 0.0 to 1.0, THEN THE PAI-EAS service SHALL return an error response indicating the input validation failure without processing the inference
10. IF the PAI-EAS service receives a request with a missing or invalid authentication token, THEN THE PAI-EAS service SHALL reject the request with an error response indicating authentication failure

### Requirement 2: Alibaba Cloud Infrastructure Configuration

**User Story:** As an ML engineer, I want to configure Alibaba Cloud resources (region, instance type, scaling) for the inference service, so that I can control cost and performance.

#### Acceptance Criteria

1. THE Alicloud_Deployer SHALL accept a user-specified Alibaba Cloud region for deployment (default: cn-hangzhou)
2. THE Alicloud_Deployer SHALL accept a user-specified EAS instance type for the inference service (default: ecs.gn6i-c4g1.xlarge), restricted to GPU instance families (ecs.gn6i, ecs.gn6v, ecs.gn7i, ecs.gn7e, or equivalent families with NVIDIA GPU support), and SHALL default to using preemptible (spot) instances to minimize cost for this test deployment
3. WHERE auto-scaling is configured, THE PAI-EAS service SHALL scale between a configured minimum (at least 1) and a configured maximum (no greater than 3) number of replicas, using average QPS per instance as the scaling metric with a user-specified target threshold
4. WHEN deployment configuration is provided, THE Alicloud_Deployer SHALL validate that the specified instance type belongs to one of the allowed GPU instance families before creating the service
5. IF an invalid or unsupported instance type is specified, THEN THE Alicloud_Deployer SHALL reject the deployment with an error message indicating the instance type is not supported for ONNX inference
6. IF an invalid or unrecognized Alibaba Cloud region is specified, THEN THE Alicloud_Deployer SHALL reject the deployment with an error message indicating the region is not available
7. IF auto-scaling is configured with a minimum replica count greater than the maximum replica count, THEN THE Alicloud_Deployer SHALL reject the configuration with an error message indicating the invalid scaling bounds
8. IF a preemptible instance is reclaimed by Alibaba Cloud, THEN THE Alicloud_Deployer SHALL automatically attempt to acquire a new preemptible instance within 60 seconds, and log the interruption event with a timestamp and instance identifier
9. IF no preemptible instance capacity is available after 3 retry attempts (each spaced at least 30 seconds apart), THEN THE Alicloud_Deployer SHALL fall back to provisioning an on-demand instance and log a warning indicating the fallback, so that service availability is maintained

### Requirement 3: Alibaba Cloud Service Cleanup

**User Story:** As an ML engineer, I want to delete the PAI-EAS service and associated resources when no longer needed, so that I avoid unnecessary costs.

#### Acceptance Criteria

1. WHEN a delete operation is initiated, THE Alicloud_Deployer SHALL remove the PAI-EAS service and release the associated compute instances provisioned for that service within 300 seconds
2. WHEN deletion completes, THE Alicloud_Deployer SHALL return a status summary containing the service identifier, the deletion outcome (succeeded or failed), and a list of resources that were removed
3. IF deletion of the PAI-EAS service fails, THEN THE Alicloud_Deployer SHALL log the service identifier and failure reason
4. IF the cleanup flag is set to true, THEN THE Alicloud_Deployer SHALL delete the model artifact from OSS after the PAI-EAS service is removed
5. IF the PAI-EAS service does not exist when deletion is initiated, THEN THE Alicloud_Deployer SHALL return a status summary indicating that no service was found for the given identifier
6. IF the OSS model artifact deletion fails while the cleanup flag is set to true, THEN THE Alicloud_Deployer SHALL log the OSS object key and failure reason and include the failure in the status summary

### Requirement 4: Alibaba Cloud Infrastructure Stack Separation

**User Story:** As an ML engineer, I want the Alibaba Cloud deployment infrastructure code to be organized into separate modular stacks mirroring the AWS CDK separation pattern, so that the codebase is maintainable and each concern can be deployed or updated independently.

#### Acceptance Criteria

1. THE Alicloud infrastructure code SHALL be implemented using ROS CDK (Python) and SHALL be organized into three separate stacks mirroring the existing AWS CDK stack separation
2. THE Alicloud infrastructure code SHALL include a storage module responsible for provisioning and managing the OSS bucket used for model artifacts, equivalent to the AWS storage_stack.py
3. THE Alicloud infrastructure code SHALL include an inference endpoint module responsible for provisioning and managing the PAI-EAS service configuration and compute resources, equivalent to the AWS sagemaker_stack.py
4. THE Alicloud infrastructure code SHALL include an API/access module responsible for configuring endpoint access, authentication tokens, and network access policies for the PAI-EAS service, equivalent to the AWS api_stack.py
5. WHEN a single module is updated, THE other modules SHALL remain unaffected and SHALL NOT require redeployment unless their inputs have changed
6. THE Alicloud infrastructure modules SHALL pass resource references between them (e.g., the storage module outputs the OSS bucket name consumed by the inference endpoint module) using the IaC tool's native cross-module reference mechanism

---

### Layer 2 - Unified Inference Client

### Requirement 5: Provider Abstraction Layer

**User Story:** As a developer, I want a unified interface for sending inference requests to either cloud provider, so that the desktop app does not need provider-specific logic.

#### Acceptance Criteria

1. THE Inference_Client SHALL define a common interface with methods for sending inference requests, checking provider health, and selecting the active Cloud_Provider by provider identifier (aws or alicloud) at initialization time
2. WHEN an inference request is sent through the Inference_Client, THE Inference_Client SHALL accept input as a 28x28 grayscale image (numpy array or flat list of 784 FP32 values normalized to the range 0.0-1.0) regardless of which Cloud_Provider is active
3. WHEN a prediction is received from any Cloud_Provider, THE Inference_Client SHALL normalize it into a Unified_Response containing the predicted digit (integer 0-9), the confidence score (float 0.0-1.0 representing the highest probability among the 10 classes), and the full 10-class probability distribution (list of 10 floats each in range 0.0-1.0 summing to 1.0)
4. THE Inference_Client SHALL translate the input format to the provider-specific protocol: Triton V2 JSON for AWS_Backend and the PAI-EAS request format for Alicloud_Backend
5. THE Inference_Client SHALL translate provider-specific responses back to the Unified_Response format
6. IF the active Cloud_Provider returns an error or does not respond within 30 seconds, THEN THE Inference_Client SHALL raise a provider-agnostic exception containing the error category (authentication, timeout, validation, server error) and a message indicating the failure reason without exposing provider-specific details
7. WHEN a health check is requested, THE Inference_Client SHALL query the active Cloud_Provider endpoint and return a boolean indicating availability, returning false if the provider does not respond within 5 seconds
8. IF the Inference_Client receives input that is not a numpy array of shape (28, 28) or (1, 28, 28) and not a flat list of exactly 784 numeric values, THEN THE Inference_Client SHALL raise a validation exception indicating the expected input format and the actual format received
9. IF the Inference_Client receives input containing values outside the range 0.0-1.0, THEN THE Inference_Client SHALL raise a validation exception indicating that pixel values must be normalized to the range 0.0-1.0

### Requirement 6: Provider Configuration Management

**User Story:** As a user, I want to configure and persist cloud provider credentials and endpoints, so that I do not need to re-enter them each time the application starts.

#### Acceptance Criteria

1. THE Provider_Config SHALL store endpoint URL (maximum 2048 characters), authentication credentials (API key or authentication token, maximum 256 characters), and region (maximum 64 characters) for each configured Cloud_Provider
2. THE Provider_Config SHALL persist configuration to a local JSON file in the OS-standard user configuration directory (XDG_CONFIG_HOME on Linux, %APPDATA% on Windows, ~/Library/Application Support on macOS), writing the file atomically so that a crash during save does not corrupt existing configuration
3. WHEN the Desktop_App starts, THE Provider_Config SHALL load previously saved provider configurations from the configuration file within 5 seconds
4. THE Provider_Config SHALL support storing configurations for at minimum 2 and at maximum 20 providers simultaneously, including at minimum AWS and Alibaba Cloud
5. IF the configuration file is missing, cannot be parsed as valid JSON, or does not conform to the expected provider configuration schema, THEN THE Provider_Config SHALL discard the invalid file, start with empty configuration, and prompt the user to configure providers
6. IF a user attempts an inference request and the selected provider configuration is missing a non-empty endpoint URL or a non-empty authentication token or API key (after trimming whitespace), THEN THE Provider_Config SHALL reject the request and display an error message indicating which required fields are missing
7. WHEN the user adds, updates, or removes a provider configuration through the UI, THE Provider_Config SHALL persist the updated configuration to disk before confirming the change to the user
8. THE Provider_Config SHALL store authentication credentials with OS-level credential protection (Keychain on macOS, Credential Manager on Windows, Secret Service on Linux) rather than in plaintext within the JSON configuration file

---

### Layer 3 - Desktop Application

### Requirement 7: Desktop Application with Drawing Canvas

**User Story:** As a user, I want a desktop application where I can draw digits and get predictions, so that I can visually test the inference endpoint.

#### Acceptance Criteria

1. THE Desktop_App SHALL display a Canvas_Widget (minimum 280x280 pixels) where users draw digits using mouse input with a stroke width between 15 and 25 pixels (scaled proportionally to canvas size) to produce strokes visually consistent with MNIST digit thickness
2. THE Canvas_Widget SHALL render drawn strokes as white lines on a black background, matching MNIST training data conventions
3. WHEN the user clicks a "Predict" button, THE Desktop_App SHALL preprocess the Canvas_Widget content into a 28x28 grayscale image using area-based interpolation for downsampling, normalize pixel values to the range 0.0-1.0, and send it to the active Cloud_Provider via the Inference_Client
4. WHEN a Unified_Response is received, THE Desktop_App SHALL display the predicted digit (integer 0-9) and confidence score (formatted to 1 decimal place as a percentage, e.g. "92.3%") in a results area below the canvas
5. THE Desktop_App SHALL provide a "Clear" button that resets the Canvas_Widget to a blank black state and clears any previously displayed prediction results
6. THE Desktop_App SHALL run on macOS, Windows, and Linux with Python 3.9 or later without requiring platform-specific installation steps beyond Python and pip
7. IF the user clicks the "Predict" button while the Canvas_Widget is blank (all pixels are black), THEN THE Desktop_App SHALL display a message indicating that the user must draw a digit before requesting a prediction, without sending a request to the Cloud_Provider
8. IF no Cloud_Provider is configured or active when the user clicks "Predict", THEN THE Desktop_App SHALL display a message directing the user to configure a provider in settings, without sending a request

### Requirement 8: Provider Switching in Desktop Application

**User Story:** As a user, I want to switch between AWS and Alibaba Cloud with a single click, so that I can compare predictions or use whichever provider is available.

#### Acceptance Criteria

1. THE Desktop_App SHALL display a provider selector (radio buttons or dropdown) showing all configured Cloud_Providers with their availability status, refreshed each time the selector is opened or a switch occurs
2. WHEN the user selects a different Cloud_Provider, THE Provider_Switcher SHALL update the active provider within 100 milliseconds without requiring application restart
3. WHEN the active provider is switched, THE Desktop_App SHALL perform a health check (with a 5-second timeout) on the newly selected Cloud_Provider and display the result (available/unavailable) next to the provider name
4. IF the selected Cloud_Provider is unreachable after the health check, THEN THE Desktop_App SHALL display a warning message indicating the provider failed the connectivity check and allow the user to switch to another provider or retry the health check
5. THE Desktop_App SHALL visually indicate which Cloud_Provider is currently active using a distinct highlight or label
6. WHEN an inference request completes, THE Desktop_App SHALL display which Cloud_Provider produced the result alongside the prediction
7. IF an inference request is in progress when the user switches providers, THEN THE Desktop_App SHALL complete the in-flight request on the original provider and display its result before routing subsequent requests to the newly selected provider
8. WHEN the Desktop_App starts with at least one configured Cloud_Provider, THE Provider_Switcher SHALL select the first available provider (based on configuration order) as the active provider and perform a health check on it
9. IF no Cloud_Providers are configured, THEN THE Desktop_App SHALL disable the provider selector and display a message directing the user to configure a provider in the settings dialog

### Requirement 9: Provider Configuration UI

**User Story:** As a user, I want to configure cloud provider endpoints and credentials within the desktop app, so that I can set up providers without editing files manually.

#### Acceptance Criteria

1. THE Desktop_App SHALL provide a settings dialog where users can add, edit, or remove Cloud_Provider configurations, supporting up to 10 saved configurations
2. WHEN configuring the AWS_Backend, THE settings dialog SHALL accept the API Gateway endpoint URL (up to 2048 characters) and API key (up to 128 characters), and SHALL require both fields to be non-empty before enabling save
3. WHEN configuring the Alicloud_Backend, THE settings dialog SHALL accept the PAI-EAS service endpoint URL (up to 2048 characters) and access token (up to 128 characters), and SHALL require both fields to be non-empty before enabling save
4. WHEN the user saves a provider configuration, THE Desktop_App SHALL validate connectivity by performing a health check with a timeout of 10 seconds and display a success or failure indicator with response time
5. IF connectivity validation fails, THEN THE Desktop_App SHALL display the error reason and allow the user to save the configuration anyway (for offline setup)
6. THE settings dialog SHALL mask credential fields (API keys, tokens) by default, with a toggle to reveal them
7. IF the user enters a URL that does not conform to valid URL format, THEN THE Desktop_App SHALL display an inline validation error and disable the save action until corrected
8. WHEN the user saves a provider configuration successfully, THE Desktop_App SHALL persist the configuration so that it remains available after application restart

---

### Layer 4 - Reliability and Error Handling

### Requirement 10: Inference Request Error Handling

**User Story:** As a user, I want clear error messages when inference fails, so that I can understand and resolve issues.

#### Acceptance Criteria

1. IF the active Cloud_Provider returns an authentication error (401/403), THEN THE Desktop_App SHALL display a message indicating the credentials are invalid or expired, suggest checking the provider configuration, and re-enable the "Predict" button within 1 second of receiving the error
2. IF the active Cloud_Provider does not respond within 10 seconds, THEN THE Desktop_App SHALL display a timeout message and present "Retry" and "Switch Provider" action buttons that the user can click to re-attempt the request or navigate to the provider selector
3. IF the inference request fails due to a network error (DNS resolution failure, connection refused, connection reset, or TLS handshake failure), THEN THE Desktop_App SHALL display a message indicating a connection problem and suggest checking network connectivity
4. IF the active Cloud_Provider returns a server error (5xx), THEN THE Desktop_App SHALL display a message indicating the service is temporarily unavailable and present a "Retry" action button
5. WHILE an inference request is in progress, THE Desktop_App SHALL display a loading indicator and disable the "Predict" button to prevent duplicate requests
6. WHEN an inference request completes with an error, THE Desktop_App SHALL re-enable the "Predict" button and display the error message in the results area, replacing any previous error message
7. IF the active Cloud_Provider returns an error that does not match authentication (401/403), timeout, network, or server error (5xx) categories, THEN THE Desktop_App SHALL display a generic error message indicating the request failed, include the error code if available, and re-enable the "Predict" button
8. WHEN an error occurs during an inference request, THE Desktop_App SHALL log the full error details (provider name, endpoint URL, error code, error category, and ISO 8601 timestamp) to a local log file in the platform-appropriate user data directory
9. WHEN a new error message is displayed in the results area, THE Desktop_App SHALL replace any previously displayed error message so that only the most recent error is visible
