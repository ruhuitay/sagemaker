"""CDK stack for SageMaker model, endpoint config, endpoint, and auto-scaling."""

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

# NVIDIA Triton Inference Server DLC (GPU) for eu-west-1
# The sagemaker-tritonserver image supports GPU instances; there is no CPU-only variant.
# Account 763104351884 is the AWS DLC ECR account for eu-west-1.
# See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
TRITON_IMAGE_EU_WEST_1 = (
    "763104351884.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-tritonserver:24.05-py3"
)

ALLOWED_GPU_FAMILIES = [
    "ml.g4dn",
    "ml.g5",
    "ml.g6",
    "ml.p3",
    "ml.p4d",
]


def validate_instance_type(instance_type: str) -> None:
    """Validate that an instance type is a supported GPU family for Triton.

    Args:
        instance_type: SageMaker instance type string (e.g. 'ml.g4dn.xlarge').

    Raises:
        ValueError: If the instance type is not from a supported GPU family.
    """
    for family in ALLOWED_GPU_FAMILIES:
        if instance_type.startswith(family):
            return

    raise ValueError(
        f"Instance type '{instance_type}' is not a supported GPU type. "
        f"The Triton inference container requires a GPU instance. "
        f"Allowed families: {', '.join(ALLOWED_GPU_FAMILIES)}"
    )


class SageMakerStack(Stack):
    """CDK stack for SageMaker model, endpoint config, and endpoint.

    Creates:
    - IAM execution role for SageMaker with S3 read access
    - CfnModel referencing Triton GPU container image and S3 model artifact
    - CfnEndpointConfig with GPU instance type (ml.g4dn.xlarge default)
    - CfnEndpoint for real-time inference

    Auto-scaling is added in Layer 3.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        model_bucket: str,
        model_key: str,
        instance_type: str = "ml.g4dn.xlarge",
        **kwargs,
    ):
        """
        Args:
            model_bucket: S3 bucket name containing the model artifact.
            model_key: S3 object key for the model.tar.gz artifact.
            instance_type: GPU SageMaker instance type (default: ml.g4dn.xlarge).

        Raises:
            ValueError: If instance_type is not a supported GPU type.
        """
        super().__init__(scope, id, **kwargs)

        # Validate instance type before creating any resources
        validate_instance_type(instance_type)

        # Construct S3 URI for the model artifact
        model_data_url = f"s3://{model_bucket}/{model_key}"

        # IAM execution role for SageMaker
        execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
            inline_policies={
                "ModelBucketAccess": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["s3:GetObject", "s3:ListBucket"],
                            resources=[
                                f"arn:aws:s3:::{model_bucket}",
                                f"arn:aws:s3:::{model_bucket}/*",
                            ],
                        )
                    ]
                )
            },
        )

        # CfnModel - references Triton GPU container and S3 model artifact
        model = sagemaker.CfnModel(
            self,
            "MnistModel",
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=TRITON_IMAGE_EU_WEST_1,
                model_data_url=model_data_url,
            ),
        )

        # CfnEndpointConfig - production variant with GPU instance
        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "MnistEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="primary",
                    model_name=model.attr_model_name,
                    instance_type=instance_type,
                    initial_instance_count=1,
                )
            ],
        )
        endpoint_config.add_dependency(model)

        # CfnEndpoint - real-time inference endpoint
        endpoint = sagemaker.CfnEndpoint(
            self,
            "MnistEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )
        endpoint.add_dependency(endpoint_config)

        # Store for cross-stack reference and auto-scaling (Layer 3)
        self._endpoint = endpoint
        self._endpoint_config = endpoint_config
        self._model = model

        # CloudFormation output
        CfnOutput(
            self,
            "EndpointName",
            value=endpoint.attr_endpoint_name,
        )

    @property
    def endpoint_name(self) -> str:
        """The name of the SageMaker endpoint (for cross-stack references)."""
        return self._endpoint.attr_endpoint_name
