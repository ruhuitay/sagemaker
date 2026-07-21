"""CDK stack for API Gateway with direct SageMaker integration (Layer 2)."""

from aws_cdk import Stack
from constructs import Construct


class ApiStack(Stack):
    """CDK stack for API Gateway with direct SageMaker AWS integration.

    This stack is implemented in Layer 2. Placeholder for now.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        sagemaker_endpoint_name: str,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)
        self._sagemaker_endpoint_name = sagemaker_endpoint_name
