"""Custom exception classes for the MNIST inference endpoint pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class DownloadError(PipelineError):
    """Raised when model download fails due to network issues or unreachable source."""


class ConversionError(PipelineError):
    """Raised when model conversion to ONNX format fails."""


class ValidationError(PipelineError):
    """Raised when ONNX model validation fails."""


class UploadError(PipelineError):
    """Raised when S3 upload fails after all retry attempts are exhausted."""


class DeploymentError(PipelineError):
    """Raised when SageMaker endpoint deployment fails."""
