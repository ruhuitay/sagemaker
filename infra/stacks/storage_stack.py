"""CDK stack for the S3 bucket that stores model artifacts."""

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_s3 as s3
from constructs import Construct


class StorageStack(Stack):
    """Creates an S3 bucket for storing MNIST model artifacts.

    The bucket is configured with:
    - Server-side encryption (SSE-S3)
    - Removal policy DESTROY (dev/test — allows bucket deletion)
    - Auto-delete objects enabled (dev/test — empties bucket on stack deletion)
    """

    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        self._bucket = s3.Bucket(
            self,
            "ModelArtifactsBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # CloudFormation outputs for cross-stack reference
        CfnOutput(self, "BucketName", value=self._bucket.bucket_name)
        CfnOutput(self, "BucketArn", value=self._bucket.bucket_arn)

    @property
    def bucket_name(self) -> str:
        """The name of the S3 bucket (for cross-stack references)."""
        return self._bucket.bucket_name

    @property
    def bucket_arn(self) -> str:
        """The ARN of the S3 bucket (for cross-stack references)."""
        return self._bucket.bucket_arn
