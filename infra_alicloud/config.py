"""Alibaba Cloud deployment configuration for MNIST inference."""

from dataclasses import dataclass


ALLOWED_ALICLOUD_GPU_FAMILIES = [
    "ecs.gn6i",
    "ecs.gn6v",
    "ecs.gn7i",
    "ecs.gn7e",
]

VALID_ALICLOUD_REGIONS = [
    "cn-shanghai",
    "cn-beijing",
    "cn-hangzhou",
    "cn-shenzhen",
    "cn-guangzhou",
    "cn-chengdu",
    "cn-hongkong",
]


@dataclass
class AlicloudDeployConfig:
    """Configuration for Alibaba Cloud deployment."""

    model_path: str  # Path to local ONNX model
    region: str = "cn-hangzhou"  # Alibaba Cloud region
    instance_type: str = "ecs.gn6i-c4g1.xlarge"  # GPU instance
    use_spot: bool = True  # Use preemptible instances
    min_replicas: int = 1  # Min auto-scaling replicas
    max_replicas: int = 3  # Max auto-scaling replicas
    scaling_target_qps: int = 10  # Target QPS per instance
    oss_bucket: str = ""  # OSS bucket name
    oss_prefix: str = "models/mnist/"  # OSS key prefix
