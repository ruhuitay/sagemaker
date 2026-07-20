"""Unit tests for ModelPackager.upload_to_s3() and run() methods."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from botocore.exceptions import ClientError

from src.config import PackagerConfig
from src.exceptions import UploadError
from src.model_packager import ModelPackager


@pytest.fixture
def config():
    return PackagerConfig(
        model_source_url="https://example.com/mnist.pt",
        s3_bucket="test-bucket",
        s3_prefix="models/mnist/",
    )


@pytest.fixture
def packager(config):
    return ModelPackager(config)


@pytest.fixture
def artifact_file(tmp_path):
    artifact = tmp_path / "model.tar.gz"
    artifact.write_bytes(b"fake archive content")
    return artifact


class TestUploadToS3:
    """Tests for upload_to_s3() method."""

    def test_successful_upload_returns_s3_uri(self, packager, artifact_file):
        """Upload succeeds on first attempt and returns correct S3 URI."""
        with patch("src.model_packager.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3

            result = packager.upload_to_s3(artifact_file)

            assert result == "s3://test-bucket/models/mnist/model.tar.gz"
            mock_s3.upload_file.assert_called_once_with(
                str(artifact_file),
                "test-bucket",
                "models/mnist/model.tar.gz",
            )

    def test_successful_upload_after_retries(self, packager, artifact_file):
        """Upload succeeds on third attempt after two failures."""
        error = ClientError(
            {"Error": {"Code": "500", "Message": "Internal Error"}},
            "PutObject",
        )
        with patch("src.model_packager.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3
            mock_s3.upload_file.side_effect = [error, error, None]

            with patch("src.model_packager.time.sleep") as mock_sleep:
                result = packager.upload_to_s3(artifact_file)

            assert result == "s3://test-bucket/models/mnist/model.tar.gz"
            assert mock_s3.upload_file.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_has_calls([call(1), call(1)])

    def test_raises_upload_error_after_3_failures(self, packager, artifact_file):
        """All 3 attempts fail, raises UploadError with failure reason."""
        error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutObject",
        )
        with patch("src.model_packager.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3
            mock_s3.upload_file.side_effect = [error, error, error]

            with patch("src.model_packager.time.sleep"):
                with pytest.raises(UploadError) as exc_info:
                    packager.upload_to_s3(artifact_file)

            assert "after 3 attempts" in str(exc_info.value)
            assert "Access Denied" in str(exc_info.value)
            assert mock_s3.upload_file.call_count == 3

    def test_retry_delay_minimum_1_second(self, packager, artifact_file):
        """Verifies minimum 1-second delay between retry attempts."""
        error = ClientError(
            {"Error": {"Code": "500", "Message": "Server Error"}},
            "PutObject",
        )
        with patch("src.model_packager.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3
            mock_s3.upload_file.side_effect = [error, None]

            with patch("src.model_packager.time.sleep") as mock_sleep:
                packager.upload_to_s3(artifact_file)

            mock_sleep.assert_called_once_with(1)

    def test_s3_uri_format_with_custom_prefix(self, config, artifact_file):
        """S3 URI correctly uses configured prefix."""
        config.s3_prefix = "custom/path/"
        packager = ModelPackager(config)

        with patch("src.model_packager.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3

            result = packager.upload_to_s3(artifact_file)

        assert result == "s3://test-bucket/custom/path/model.tar.gz"

    def test_no_sleep_after_last_failed_attempt(self, packager, artifact_file):
        """No sleep is called after the third (last) failed attempt."""
        error = ClientError(
            {"Error": {"Code": "500", "Message": "Server Error"}},
            "PutObject",
        )
        with patch("src.model_packager.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3
            mock_s3.upload_file.side_effect = [error, error, error]

            with patch("src.model_packager.time.sleep") as mock_sleep:
                with pytest.raises(UploadError):
                    packager.upload_to_s3(artifact_file)

            # Only 2 sleeps (after attempt 1 and 2, not after attempt 3)
            assert mock_sleep.call_count == 2


class TestRun:
    """Tests for run() method pipeline orchestration."""

    def test_run_calls_pipeline_in_order(self, packager):
        """run() calls all pipeline methods in the correct order."""
        call_order = []

        def make_tracker(name, return_val=None):
            def tracker(*args, **kwargs):
                call_order.append(name)
                return return_val
            return tracker

        with patch.object(packager, "download_model", make_tracker("download", Path("/tmp/model.pt"))):
            with patch.object(packager, "convert_to_onnx", make_tracker("convert", Path("/tmp/model.onnx"))):
                with patch.object(packager, "validate_onnx", make_tracker("validate")):
                    with patch.object(packager, "create_model_repository", make_tracker("create_repo", Path("/tmp/repo"))):
                        with patch.object(packager, "package_artifact", make_tracker("package", Path("/tmp/model.tar.gz"))):
                            with patch.object(packager, "upload_to_s3", make_tracker("upload", "s3://bucket/key")):
                                result = packager.run()

        assert call_order == [
            "download",
            "convert",
            "validate",
            "create_repo",
            "package",
            "upload",
        ]
        assert result == "s3://bucket/key"

    def test_run_returns_s3_uri(self, packager):
        """run() returns the S3 URI from upload_to_s3."""
        expected_uri = "s3://test-bucket/models/mnist/model.tar.gz"

        with patch.object(packager, "download_model", return_value=Path("/tmp/model.pt")):
            with patch.object(packager, "convert_to_onnx", return_value=Path("/tmp/model.onnx")):
                with patch.object(packager, "validate_onnx"):
                    with patch.object(packager, "create_model_repository", return_value=Path("/tmp/repo")):
                        with patch.object(packager, "package_artifact", return_value=Path("/tmp/model.tar.gz")):
                            with patch.object(packager, "upload_to_s3", return_value=expected_uri):
                                result = packager.run()

        assert result == expected_uri

    def test_run_passes_outputs_between_steps(self, packager):
        """run() passes each step's output as input to the next step."""
        model_path = Path("/tmp/model.pt")
        onnx_path = Path("/tmp/model.onnx")
        repo_path = Path("/tmp/repo")
        artifact_path = Path("/tmp/model.tar.gz")

        with patch.object(packager, "download_model", return_value=model_path) as mock_download:
            with patch.object(packager, "convert_to_onnx", return_value=onnx_path) as mock_convert:
                with patch.object(packager, "validate_onnx") as mock_validate:
                    with patch.object(packager, "create_model_repository", return_value=repo_path) as mock_create:
                        with patch.object(packager, "package_artifact", return_value=artifact_path) as mock_package:
                            with patch.object(packager, "upload_to_s3", return_value="s3://b/k") as mock_upload:
                                packager.run()

        mock_convert.assert_called_once_with(model_path)
        mock_validate.assert_called_once_with(onnx_path)
        mock_create.assert_called_once_with(onnx_path)
        mock_package.assert_called_once_with(repo_path)
        mock_upload.assert_called_once_with(artifact_path)
