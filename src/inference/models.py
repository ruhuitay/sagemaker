"""Data models for the unified inference client."""

from dataclasses import dataclass


@dataclass
class UnifiedResponse:
    """Normalized prediction result from any provider."""

    predicted_digit: int  # 0-9, argmax of probabilities
    confidence: float  # 0.0-1.0, max of probabilities
    probabilities: list[float]  # 10-element probability distribution
    provider: str  # Name of provider that produced the result
    latency_ms: float  # Request round-trip time in milliseconds


@dataclass
class ProviderConfig:
    """Configuration for a single cloud provider."""

    provider_id: str  # e.g., "aws", "alicloud"
    endpoint_url: str  # API endpoint URL (max 2048 chars)
    region: str  # Provider region (max 64 chars)
    credential_key: str  # Keyring lookup key, not the actual credential
