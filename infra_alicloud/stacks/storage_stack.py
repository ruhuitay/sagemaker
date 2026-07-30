"""ROS CDK stack for the OSS bucket that stores model artifacts."""

import ros_cdk_core as ros
import ros_cdk_oss as oss

from config import COMMON_TAGS


class StorageStack(ros.Stack):
    """Creates an OSS bucket for storing MNIST model artifacts.

    The bucket is configured with:
    - Server-side encryption (AES256)
    - Lifecycle rules to abort incomplete multipart uploads after 7 days
    - Force deletion enabled for dev/test cleanup
    """

    BUCKET_NAME = "mnist-model-artifacts-alicloud"

    def __init__(self, scope: ros.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self._bucket = oss.Bucket(
            self,
            "ModelArtifactsBucket",
            props=oss.BucketProps(
                bucket_name=self.BUCKET_NAME,
                tags=COMMON_TAGS,
                server_side_encryption_configuration=oss.RosBucket.ServerSideEncryptionConfigurationProperty(
                    sse_algorithm="AES256",
                ),
                lifecycle_configuration=oss.RosBucket.LifecycleConfigurationProperty(
                    rule=[
                        oss.RosBucket.RuleProperty(
                            prefix="",
                            status="Enabled",
                            id="AbortIncompleteMultipartUpload",
                            abort_multipart_upload=oss.RosBucket.AbortMultipartUploadProperty(
                                days=7,
                            ),
                        ),
                    ],
                ),
                deletion_force=True,
            ),
        )

        # ROS Output for cross-stack reference
        ros.RosOutput(
            self,
            "BucketName",
            value=self.BUCKET_NAME,
            description="OSS bucket name for model artifacts",
        )

    @property
    def bucket_name(self) -> str:
        """The name of the OSS bucket (for cross-stack references)."""
        return self.BUCKET_NAME
