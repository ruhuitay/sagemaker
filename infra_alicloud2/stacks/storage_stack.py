import ros_cdk_core as ros
import ros_cdk_oss as oss
from config import COMMON_TAGS

class StorageStack(ros.Stack):        

    def __init__(self, scope: ros.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here
        self.BUCKET_NAME = "mnist-model-artifacts-alicloud"

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
    
