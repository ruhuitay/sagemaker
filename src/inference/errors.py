"""Provider-agnostic error handling for inference requests."""

from enum import Enum


class ErrorCategory(Enum):
    """Categories of inference errors."""

    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    NETWORK = "network"
    SERVER_ERROR = "server_error"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class InferenceError(Exception):
    """Provider-agnostic inference error.

    Exposes only the error category and a user-facing message.
    No provider-specific details (URLs, tokens) are included.
    """

    def __init__(self, category: ErrorCategory, message: str) -> None:
        self.category = category
        self.message = message
        super().__init__(message)


def categorize_error(
    exception: Exception, status_code: int | None = None
) -> InferenceError:
    """Categorize a provider error into a standard error category.

    Mapping rules:
    - 401/403 status codes -> AUTHENTICATION
    - Timeout exceptions -> TIMEOUT
    - ConnectionError, DNS, TLS errors -> NETWORK
    - 5xx status codes -> SERVER_ERROR
    - Everything else -> UNKNOWN

    Args:
        exception: The original exception from the provider.
        status_code: HTTP status code if available.

    Returns:
        An InferenceError with the appropriate category and a
        provider-agnostic message.
    """
    # Check status code first if provided
    if status_code is not None:
        if status_code in (401, 403):
            return InferenceError(
                category=ErrorCategory.AUTHENTICATION,
                message="Credentials are invalid or expired. Check provider configuration.",
            )
        if 500 <= status_code < 600:
            return InferenceError(
                category=ErrorCategory.SERVER_ERROR,
                message="Service is temporarily unavailable.",
            )

    # Check exception type for timeout
    import requests.exceptions

    if isinstance(exception, (TimeoutError, requests.exceptions.Timeout)):
        return InferenceError(
            category=ErrorCategory.TIMEOUT,
            message="Request timed out. The service may be unavailable.",
        )

    # Check exception type for network errors
    if isinstance(
        exception,
        (
            ConnectionError,
            OSError,
            requests.exceptions.ConnectionError,
        ),
    ):
        return InferenceError(
            category=ErrorCategory.NETWORK,
            message="Connection problem. Check your network connectivity.",
        )

    # Default to unknown
    error_code_info = f" (error code: {status_code})" if status_code is not None else ""
    return InferenceError(
        category=ErrorCategory.UNKNOWN,
        message=f"Request failed{error_code_info}.",
    )
