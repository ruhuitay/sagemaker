# Requirements Document

## Introduction

This feature covers downloading a pre-trained MNIST handwriting recognition model, converting it to ONNX format, packaging it in a Triton model repository structure, uploading the model artifact to a test S3 bucket, and deploying it as a real-time inference endpoint using NVIDIA Triton Inference Server on SageMaker. The system enables users to submit handwritten digit images and receive classification predictions with low latency.

## Glossary

- **Model_Packager**: The component responsible for downloading a pre-trained MNIST model, converting it to ONNX format, and packaging it into a Triton-compatible model repository structure
- **Model_Artifact**: The compressed archive (model.tar.gz) containing the Triton Model_Repository structure with the ONNX model and config.pbtxt, ready for SageMaker Triton deployment
- **Model_Repository**: The directory structure required by Triton_Inference_Server, containing a model directory with a config.pbtxt file and a numbered version subdirectory holding the ONNX model file
- **config.pbtxt**: The Triton model configuration file specifying model name, platform (onnxruntime_onnx), input/output tensor shapes, data types, and batching settings
- **ONNX**: Open Neural Network Exchange format, a portable model representation that enables interoperability across ML frameworks and optimized inference via Triton
- **Triton_Inference_Server**: NVIDIA Triton Inference Server, a high-performance inference serving platform available as a SageMaker container image that supports ONNX models natively
- **Inference_Endpoint**: The real-time HTTPS endpoint hosted on SageMaker using the Triton_Inference_Server container image that loads the ONNX model and serves predictions
- **Digit_Image**: A grayscale image of a handwritten digit (0-9), sized 28x28 pixels
- **Prediction_Response**: The JSON response containing the predicted digit class and confidence scores, formatted according to Triton inference protocol
- **S3_Bucket**: The Amazon S3 test bucket used to store the packaged model artifact
- **SageMaker**: AWS SageMaker service used for endpoint deployment and hosting

## Requirements

Requirements are organized in layers of increasing complexity. Layer 1 delivers a minimal working inference endpoint. Each subsequent layer adds capabilities on top of the working base.

---

### Layer 1 — Minimal Working Endpoint

### Requirement 1: Download, Convert to ONNX, and Package Pre-Trained MNIST Model

**User Story:** As an ML engineer, I want to download an existing pre-trained MNIST model, convert it to ONNX format, and package it in a Triton model repository structure, so that I can deploy it on Triton Inference Server without training from scratch.

#### Acceptance Criteria

1. WHEN the Model_Packager is executed, THE Model_Packager SHALL download a pre-trained MNIST model from a configured source URL or repository
2. WHEN the download completes, THE Model_Packager SHALL convert the pre-trained model to ONNX format with an opset version of at least 11
3. WHEN the ONNX conversion completes, THE Model_Packager SHALL validate the ONNX model using the onnx.checker module to confirm structural correctness
4. WHEN the ONNX model is validated, THE Model_Packager SHALL create a Triton Model_Repository structure containing a model directory with a config.pbtxt file and a version subdirectory (named "1") holding the ONNX model file (model.onnx)
5. THE config.pbtxt SHALL specify the platform as "onnxruntime_onnx", define the input tensor shape as 1x28x28 (single-channel grayscale), define the output tensor shape matching the 10-class digit prediction, and set the data types for input (FP32) and output (FP32)
6. WHEN the Model_Repository structure is created, THE Model_Packager SHALL package it into a model.tar.gz archive preserving the directory hierarchy expected by Triton_Inference_Server
7. IF the download fails due to network issues or an unreachable source, THEN THE Model_Packager SHALL log the error details to standard output and exit with a non-zero status code
8. IF the downloaded model file is corrupted or cannot be converted to ONNX format, THEN THE Model_Packager SHALL log a validation error and exit with a non-zero status code
9. WHEN packaging completes successfully, THE Model_Packager SHALL output the local file path of the generated model.tar.gz archive

### Requirement 2: Upload Model Artifact to S3

**User Story:** As an ML engineer, I want the packaged Triton model repository uploaded to a test S3 bucket, so that SageMaker Triton Inference Server can access it for deployment.

#### Acceptance Criteria

1. WHEN packaging completes successfully, THE Model_Packager SHALL upload the model.tar.gz containing the Triton Model_Repository structure to the configured S3_Bucket
2. WHEN uploading the Model_Artifact, THE Model_Packager SHALL store it at a path that includes a descriptive prefix (e.g., models/mnist/) such that the artifact is identifiable
3. IF the S3 upload fails, THEN THE Model_Packager SHALL retry up to 3 times with a minimum delay of 1 second between attempts before raising an error that includes the failure reason
4. WHEN the Model_Artifact is uploaded successfully, THE Model_Packager SHALL return the full S3 URI (s3://bucket/path) of the stored artifact
5. IF all 3 retry attempts are exhausted, THEN THE Model_Packager SHALL terminate with a non-zero exit status and an error message indicating the upload failure cause

### Requirement 3: Deploy Real-Time Inference Endpoint Using Triton Inference Server

**User Story:** As an ML engineer, I want to deploy the ONNX model as a real-time SageMaker endpoint using the Triton Inference Server container, so that applications can get digit predictions with low latency and optimized inference.

#### Acceptance Criteria

1. WHEN a deployment is initiated with a Model_Artifact S3 URI, THE Inference_Endpoint SHALL create a SageMaker real-time endpoint using the NVIDIA Triton_Inference_Server container image from the SageMaker Deep Learning Containers registry
2. THE Inference_Endpoint SHALL be accessible via HTTPS
3. WHEN the endpoint status is "InService", THE Inference_Endpoint SHALL respond to health check requests with a 200 status code within 3 seconds
4. IF deployment fails or does not reach "InService" status within 15 minutes, THEN THE Inference_Endpoint SHALL report the failure reason to the caller and delete any partially created endpoint resources within 5 minutes
5. WHEN the endpoint receives an inference request containing a valid input tensor, THE Inference_Endpoint SHALL return a predicted digit (0-9) with a confidence score between 0.0 and 1.0
6. WHEN the endpoint is in "InService" status and receives an inference request, THE Inference_Endpoint SHALL return a prediction response within 1 second at the 95th percentile
7. THE Inference_Endpoint SHALL use the SageMaker Triton container image version compatible with the ONNX opset version used during model conversion
8. THE Inference_Endpoint SHALL use a CPU-only SageMaker instance type (e.g., ml.c5.large or ml.m5.large) for deployment
9. THE Inference_Endpoint SHALL use the CPU variant of the Triton_Inference_Server container image from the SageMaker Deep Learning Containers registry

---

### Layer 2 — Inference Protocol and Validation

### Requirement 4: Serve Predictions via Triton Inference Protocol

**User Story:** As an application developer, I want to send digit images to the Triton-backed endpoint and receive predictions using the Triton inference protocol, so that I can integrate handwriting recognition into my application.

#### Acceptance Criteria

1. WHEN a valid Digit_Image is submitted to the Inference_Endpoint, THE Inference_Endpoint SHALL return a Prediction_Response containing the predicted digit as an integer in the range 0-9 and a confidence score as a decimal value between 0.0 and 1.0
2. WHEN a Digit_Image is submitted, THE Inference_Endpoint SHALL return the Prediction_Response within 500 milliseconds under normal load of up to 10 concurrent requests
3. THE Inference_Endpoint SHALL accept inference requests formatted as JSON following the Triton inference protocol, with input tensor data provided as a flat array of FP32 values representing a 1x28x28 grayscale image
4. THE Inference_Endpoint SHALL return responses in JSON format following the Triton inference protocol, with output tensor data containing the 10-class probability distribution
5. IF an invalid input is submitted with wrong tensor shape, wrong data type, missing required fields, or payload size exceeding 1 MB, THEN THE Inference_Endpoint SHALL return a 400 status code with an error message indicating the specific validation failure
6. IF the Inference_Endpoint receives a request while the model is unavailable or fails to produce a prediction, THEN THE Inference_Endpoint SHALL return a 503 status code with an error message indicating the service is temporarily unavailable
7. WHEN a valid Digit_Image is submitted to the Inference_Endpoint, THE Inference_Endpoint SHALL return the Prediction_Response with Content-Type application/json

---

### Layer 3 — External Access

### Requirement 5: API Gateway Authentication for External Applications

**User Story:** As an ML engineer, I want to expose the SageMaker inference endpoint through API Gateway with API key authentication, so that external applications can securely call the endpoint without needing AWS credentials.

#### Acceptance Criteria

1. WHEN the authentication layer is deployed, THE system SHALL create an API Gateway REST API with a POST method on a `/predict` resource that proxies requests to the SageMaker Inference_Endpoint via a Lambda function
2. THE API Gateway `/predict` resource SHALL require an API key for access, rejecting requests without a valid `x-api-key` header with a 403 status code
3. WHEN the API Gateway is created, THE system SHALL create at least one API key and associate it with a usage plan that defines rate limiting (requests per second) and throttling (burst limit) settings
4. THE usage plan SHALL default to 10 requests per second with a burst limit of 20 unless otherwise specified by the user
5. WHEN a valid request with a correct API key is received, THE Lambda function SHALL forward the request payload to the SageMaker Inference_Endpoint using IAM authentication (SigV4) and return the prediction response to the caller
6. THE Lambda function SHALL have an IAM execution role with least-privilege permissions: only `sagemaker:InvokeEndpoint` on the specific endpoint resource
7. WHEN the API Gateway is deployed, THE system SHALL output the invoke URL (e.g., `https://{api-id}.execute-api.{region}.amazonaws.com/{stage}/predict`) and the generated API key value
8. IF the Lambda function fails to invoke the SageMaker endpoint or the endpoint returns an error, THE API Gateway SHALL return the appropriate HTTP error code (502 for Lambda failure, or the status code from SageMaker) with an error message
9. THE API Gateway SHALL enforce a maximum request payload size of 1 MB, returning a 413 status code for oversized payloads
10. WHEN a delete operation is initiated for the authentication layer, THE system SHALL remove the API Gateway, usage plan, API key, Lambda function, and associated IAM role in dependency order

---

### Layer 4 — Production Hardening

### Requirement 6: Endpoint Configuration and Scaling

**User Story:** As an ML engineer, I want to configure the endpoint instance type and scaling behavior, so that I can balance cost and performance.

#### Acceptance Criteria

1. THE Inference_Endpoint SHALL be deployable on a user-specified SageMaker instance type provided at creation time, restricted to CPU-only instance families (ml.c4, ml.c5, ml.c5d, ml.m4, ml.m5, ml.m5d, ml.t2, ml.t3)
2. WHEN the endpoint is created, THE Inference_Endpoint SHALL use a single instance by default
3. WHERE auto-scaling is configured, THE Inference_Endpoint SHALL scale the number of instances between a configured minimum (at least 1) and a configured maximum (no greater than 10) based on the average number of invocations per instance per minute
4. IF a GPU instance type is specified (ml.p2, ml.p3, ml.p4, ml.g4dn, ml.g5, ml.inf1, or any other GPU/accelerator instance family), THEN THE Inference_Endpoint SHALL reject the deployment request with an error message indicating that only CPU-only instance types are supported
5. IF an unsupported or invalid instance type is specified, THEN THE Inference_Endpoint SHALL reject the deployment request with an error message indicating the invalid instance type

### Requirement 7: Endpoint Cleanup

**User Story:** As an ML engineer, I want to delete the endpoint when it is no longer needed, so that I avoid unnecessary costs.

#### Acceptance Criteria

1. WHEN a delete operation is initiated, THE Inference_Endpoint SHALL remove the SageMaker endpoint, endpoint configuration, and model resources in dependency order: endpoint first, then endpoint configuration, then model
2. WHEN deletion completes, THE Inference_Endpoint SHALL return a status summary indicating the deletion result for each resource (endpoint, endpoint configuration, model)
3. IF deletion of a resource fails, THEN THE Inference_Endpoint SHALL log the resource identifier and failure reason, and continue attempting to delete remaining resources
4. IF a resource does not exist at the time of deletion, THEN THE Inference_Endpoint SHALL treat the resource as successfully deleted
5. IF deletion of one or more resources fails after all deletion attempts complete, THEN THE Inference_Endpoint SHALL return a summary indicating which resources were successfully deleted and which failed with the corresponding reason
