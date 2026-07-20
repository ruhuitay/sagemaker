"""Configuration dataclasses for model packaging and endpoint deployment."""

from dataclasses import dataclass


@dataclass
class PackagerConfig:
    """Configuration for the model packaging pipeline.

    Attributes:
        model_source_url: URL to download the pre-trained MNIST model from.
        s3_bucket: S3 bucket name for uploading the model artifact.
        s3_prefix: S3 key prefix for the uploaded artifact.
        onnx_opset_version: ONNX opset version to use during conversion.
    """

    model_source_url: str
    s3_bucket: str
    s3_prefix: str = "models/mnist/"
    onnx_opset_version: int = 11


@dataclass
class DeployerConfig:
    """Configuration for the SageMaker endpoint deployment.

    Attributes:
        endpoint_name: Name for the SageMaker endpoint.
        instance_type: SageMaker instance type (CPU-only).
        initial_instance_count: Number of instances to launch.
        region: AWS region for deployment.
        model_name: Name for the SageMaker model resource.
    """

    endpoint_name: str
    instance_type: str = "ml.c5.large"
    initial_instance_count: int = 1
    region: str = "eu-west-1"
    model_name: str = "mnist-triton"
