"""CDK assertion tests for StorageStack and SageMakerStack.

Uses aws_cdk.assertions.Template to verify synthesized CloudFormation templates
contain correct resources and configurations.
"""

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match

from infra.stacks.storage_stack import StorageStack
from infra.stacks.sagemaker_stack import SageMakerStack


@pytest.fixture
def storage_template():
    """Synthesize StorageStack and return its Template for assertions."""
    app = cdk.App()
    stack = StorageStack(app, "TestStorageStack")
    return Template.from_stack(stack)


@pytest.fixture
def sagemaker_template():
    """Synthesize SageMakerStack and return its Template for assertions."""
    app = cdk.App()
    stack = SageMakerStack(
        app,
        "TestSageMakerStack",
        model_bucket="my-model-bucket",
        model_key="models/mnist/model.tar.gz",
        instance_type="ml.c5.large",
    )
    return Template.from_stack(stack)


class TestStorageStack:
    """Tests for StorageStack synthesized template."""

    def test_creates_s3_bucket_with_sse_s3_encryption(self, storage_template):
        """StorageStack creates S3 bucket with SSE-S3 encryption configured."""
        storage_template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketEncryption": {
                    "ServerSideEncryptionConfiguration": [
                        {
                            "ServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "AES256",
                            }
                        }
                    ]
                }
            },
        )

    def test_has_cloudformation_output_for_bucket_name(self, storage_template):
        """StorageStack has CloudFormation output for bucket name."""
        storage_template.has_output(
            "BucketName",
            {"Value": Match.any_value()},
        )

    def test_has_cloudformation_output_for_bucket_arn(self, storage_template):
        """StorageStack has CloudFormation output for bucket ARN."""
        storage_template.has_output(
            "BucketArn",
            {"Value": Match.any_value()},
        )


class TestSageMakerStack:
    """Tests for SageMakerStack synthesized template."""

    def test_constructs_s3_uri_from_bucket_and_key(self, sagemaker_template):
        """SageMakerStack correctly constructs S3 URI from model_bucket + model_key."""
        sagemaker_template.has_resource_properties(
            "AWS::SageMaker::Model",
            {
                "PrimaryContainer": {
                    "ModelDataUrl": "s3://my-model-bucket/models/mnist/model.tar.gz",
                }
            },
        )

    def test_cfn_model_references_correct_container_image(self, sagemaker_template):
        """CfnModel references correct Triton CPU container image."""
        sagemaker_template.has_resource_properties(
            "AWS::SageMaker::Model",
            {
                "PrimaryContainer": {
                    "Image": (
                        "785573368785.dkr.ecr.eu-west-1.amazonaws.com/"
                        "sagemaker-tritonserver:23.12-py3-cpu"
                    ),
                }
            },
        )

    def test_cfn_model_references_correct_image_and_s3_uri(self, sagemaker_template):
        """CfnModel references both the correct container image and constructed S3 URI."""
        sagemaker_template.has_resource_properties(
            "AWS::SageMaker::Model",
            {
                "PrimaryContainer": {
                    "Image": (
                        "785573368785.dkr.ecr.eu-west-1.amazonaws.com/"
                        "sagemaker-tritonserver:23.12-py3-cpu"
                    ),
                    "ModelDataUrl": "s3://my-model-bucket/models/mnist/model.tar.gz",
                }
            },
        )

    def test_endpoint_config_uses_specified_instance_type(self, sagemaker_template):
        """CfnEndpointConfig uses specified instance type with 1 instance."""
        sagemaker_template.has_resource_properties(
            "AWS::SageMaker::EndpointConfig",
            {
                "ProductionVariants": [
                    Match.object_like(
                        {
                            "InstanceType": "ml.c5.large",
                            "InitialInstanceCount": 1,
                        }
                    )
                ]
            },
        )

    def test_endpoint_resource_exists(self, sagemaker_template):
        """CfnEndpoint resource exists in the synthesized template."""
        sagemaker_template.resource_count_is("AWS::SageMaker::Endpoint", 1)

    def test_raises_value_error_for_gpu_instance_p3(self):
        """Constructor raises ValueError for GPU instance type ml.p3.2xlarge."""
        app = cdk.App()
        with pytest.raises(ValueError, match="GPU/accelerator instance type"):
            SageMakerStack(
                app,
                "TestGpuStack",
                model_bucket="my-bucket",
                model_key="models/model.tar.gz",
                instance_type="ml.p3.2xlarge",
            )

    def test_raises_value_error_for_gpu_instance_g4dn(self):
        """Constructor raises ValueError for GPU instance type ml.g4dn.xlarge."""
        app = cdk.App()
        with pytest.raises(ValueError, match="GPU/accelerator instance type"):
            SageMakerStack(
                app,
                "TestGpuStack2",
                model_bucket="my-bucket",
                model_key="models/model.tar.gz",
                instance_type="ml.g4dn.xlarge",
            )
