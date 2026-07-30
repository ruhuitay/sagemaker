"""ROS CDK stack for the PAI-EAS Triton inference service."""

import ros_cdk_core as ros
import ros_cdk_pai as pai

from config import COMMON_TAGS

# Official PAI-EAS Triton Inference Server image (GPU, cn-hangzhou registry)
TRITON_IMAGE = (
    "registry.cn-hangzhou.aliyuncs.com/pai-dlc/tritonserver:23.05-py3"
)

ALLOWED_GPU_FAMILIES = [
    "ecs.gn6i",
    "ecs.gn6v",
    "ecs.gn7i",
    "ecs.gn7e",
]


def validate_instance_type(instance_type: str) -> None:
    """Validate that an instance type is a supported Alibaba Cloud GPU family.

    Args:
        instance_type: ECS instance type string (e.g. 'ecs.gn6i-c4g1.xlarge').

    Raises:
        ValueError: If the instance type is not from a supported GPU family.
    """
    for family in ALLOWED_GPU_FAMILIES:
        if instance_type.startswith(family):
            return

    raise ValueError(
        f"Instance type '{instance_type}' is not a supported GPU type. "
        f"PAI-EAS Triton inference requires a GPU instance. "
        f"Allowed families: {', '.join(ALLOWED_GPU_FAMILIES)}"
    )


class EasStack(ros.Stack):
    """ROS CDK stack for PAI-EAS Triton inference service.

    Creates:
    - PAI-EAS service configured with Triton Inference Server
    - GPU instance with optional spot/preemptible preference
    - Auto-scaling from min_replicas to max_replicas based on QPS

    Exports:
    - service_name: The PAI-EAS service name for cross-stack reference
    - endpoint_url: The HTTPS endpoint URL for inference requests
    """

    def __init__(
        self,
        scope: ros.Construct,
        id: str,
        oss_bucket_name: str,
        model_key: str,
        instance_type: str = "ecs.gn6i-c4g1.xlarge",
        use_spot: bool = True,
        min_replicas: int = 1,
        max_replicas: int = 3,
        **kwargs,
    ) -> None:
        """
        Args:
            scope: ROS CDK construct scope.
            id: Stack identifier.
            oss_bucket_name: OSS bucket containing the model artifact.
            model_key: OSS object key/prefix for the model (e.g. 'models/mnist/').
            instance_type: GPU ECS instance type (default: ecs.gn6i-c4g1.xlarge).
            use_spot: Whether to use preemptible/spot instances (default: True).
            min_replicas: Minimum number of service replicas (default: 1).
            max_replicas: Maximum number of service replicas (default: 3).

        Raises:
            ValueError: If instance_type is not a supported GPU type.
            ValueError: If min_replicas > max_replicas.
        """
        super().__init__(scope, id, **kwargs)

        # Validate inputs
        validate_instance_type(instance_type)

        if min_replicas > max_replicas:
            raise ValueError(
                f"min_replicas ({min_replicas}) cannot exceed "
                f"max_replicas ({max_replicas})."
            )

        # Construct the OSS model path
        model_oss_path = f"oss://{oss_bucket_name}/{model_key}"

        # Service name derived from stack id
        service_name = "mnist-triton-eas"

        # Build the PAI-EAS service configuration
        service_config = {
            "name": service_name,
            "model_path": model_oss_path,
            "processor": "triton",
            "image": TRITON_IMAGE,
            "instance_type": instance_type,
            "cloud": {
                "computing": {
                    "instance_type": instance_type,
                    "use_spot_instance": use_spot,
                },
            },
            "scaling": {
                "min_replica": min_replicas,
                "max_replica": max_replicas,
                "auto_scaling": {
                    "enabled": True,
                    "metric": "qps",
                    "target": 10,
                    "min_replica": min_replicas,
                    "max_replica": max_replicas,
                },
            },
            "metadata": {
                "instance": instance_type,
                "enable_webservice": True,
            },
        }

        # Create the PAI-EAS service resource (ALIYUN::PAI::Service)
        self._service = pai.Service(
            self,
            "MnistTritonService",
            props=pai.ServiceProps(
                service_config=service_config,
                labels=COMMON_TAGS,
            ),
        )

        # Store values for outputs and properties
        self._service_name = service_name
        # PAI-EAS endpoint URL pattern for cn-hangzhou region
        self._endpoint_url = (
            f"https://{service_name}.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/content"
        )

        # ROS Outputs for cross-stack reference
        ros.RosOutput(
            self,
            "ServiceName",
            value=self._service_name,
            description="PAI-EAS service name",
        )

        ros.RosOutput(
            self,
            "EndpointUrl",
            value=self._endpoint_url,
            description="PAI-EAS inference endpoint URL",
        )

    @property
    def service_name(self) -> str:
        """The PAI-EAS service name (for cross-stack references)."""
        return self._service_name

    @property
    def endpoint_url(self) -> str:
        """The HTTPS endpoint URL for inference requests."""
        return self._endpoint_url
