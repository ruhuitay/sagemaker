"""ROS CDK stack for PAI-EAS access configuration (authentication and network)."""

import ros_cdk_core as ros


class AccessStack(ros.Stack):
    """Configures authentication and network access for a PAI-EAS service.

    PAI-EAS services automatically provision a public HTTPS endpoint and
    access token upon creation. This stack documents and exports those
    values as ROS Outputs for cross-stack reference.

    Configured with:
    - Token-based authentication (PAI-EAS built-in)
    - Public HTTPS endpoint access (no VPC restriction for test deployment)
    """

    # PAI-EAS endpoint URL pattern for the cn-hangzhou region
    _ENDPOINT_PATTERN = (
        "https://{service_name}.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/{service_name}"
    )

    def __init__(
        self,
        scope: ros.Construct,
        id: str,
        service_name: str,
        **kwargs,
    ) -> None:
        """Create access configuration for a PAI-EAS service.

        Args:
            scope: ROS CDK app or parent construct.
            id: Logical stack ID.
            service_name: Name of the PAI-EAS service (from EasStack).
        """
        super().__init__(scope, id, **kwargs)

        self._service_name = service_name

        # PAI-EAS generates an access token when the service is created.
        # In a real deployment, the token is retrieved from the PAI-EAS API
        # after service creation. For IaC purposes, we reference it as a
        # placeholder that will be populated at deploy time.
        self._access_token = ros.RosParameter(
            self,
            "AccessToken",
            type=ros.RosParameterType.STRING,
            description="PAI-EAS service access token for API authentication",
            no_echo=True,
            default_value="",
        )

        # Construct the public HTTPS endpoint URL for the service
        self._public_endpoint = self._ENDPOINT_PATTERN.format(
            service_name=service_name,
        )

        # Export access token as ROS Output
        ros.RosOutput(
            self,
            "AccessTokenOutput",
            value=self._access_token.value_as_string,
            description="Access token for PAI-EAS service authentication",
        )

        # Export public endpoint as ROS Output
        ros.RosOutput(
            self,
            "PublicEndpoint",
            value=self._public_endpoint,
            description="Public HTTPS endpoint URL for inference requests",
        )

    @property
    def access_token(self) -> str:
        """The access token for PAI-EAS API authentication."""
        return self._access_token.value_as_string

    @property
    def public_endpoint(self) -> str:
        """The public HTTPS endpoint URL for the PAI-EAS service."""
        return self._public_endpoint
