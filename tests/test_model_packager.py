"""Unit tests for ModelPackager download, conversion, and validation."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from src.config import PackagerConfig
from src.exceptions import ConversionError, DownloadError, ValidationError
from src.model_packager import MNISTNet, ModelPackager


@pytest.fixture
def config():
    """Create a test PackagerConfig."""
    return PackagerConfig(
        model_source_url="http://example.com/mnist_model.pt",
        s3_bucket="test-bucket",
        s3_prefix="models/mnist/",
        onnx_opset_version=11,
    )


@pytest.fixture
def packager(config):
    """Create a ModelPackager instance with test config."""
    return ModelPackager(config)


@pytest.fixture
def valid_model_path(tmp_path):
    """Create a valid PyTorch model file for testing."""
    model = MNISTNet()
    model_path = tmp_path / "mnist_model.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


class TestDownloadModel:
    """Tests for ModelPackager.download_model()."""

    def test_successful_download(self, packager):
        """Test model is downloaded and saved to temp directory."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake_model_data"]
        mock_response.raise_for_status.return_value = None

        with patch("src.model_packager.requests.get", return_value=mock_response):
            result = packager.download_model()

        assert result.exists()
        assert result.name == "mnist_model.pt"
        assert result.read_bytes() == b"fake_model_data"

    def test_download_network_error_raises_download_error(self, packager):
        """Test that network errors raise DownloadError."""
        import requests as req

        with patch(
            "src.model_packager.requests.get",
            side_effect=req.exceptions.ConnectionError("Connection refused"),
        ):
            with pytest.raises(DownloadError, match="Failed to download model"):
                packager.download_model()

    def test_download_http_error_raises_download_error(self, packager):
        """Test that HTTP errors (404, 500) raise DownloadError."""
        import requests as req

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError(
            "404 Not Found"
        )

        with patch("src.model_packager.requests.get", return_value=mock_response):
            with pytest.raises(DownloadError, match="Failed to download model"):
                packager.download_model()

    def test_download_timeout_raises_download_error(self, packager):
        """Test that timeout raises DownloadError."""
        import requests as req

        with patch(
            "src.model_packager.requests.get",
            side_effect=req.exceptions.Timeout("Request timed out"),
        ):
            with pytest.raises(DownloadError, match="Failed to download model"):
                packager.download_model()


class TestConvertToOnnx:
    """Tests for ModelPackager.convert_to_onnx()."""

    def test_successful_conversion(self, packager, valid_model_path):
        """Test successful conversion from PyTorch to ONNX."""
        onnx_path = packager.convert_to_onnx(valid_model_path)

        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_conversion_produces_valid_onnx(self, packager, valid_model_path):
        """Test that the converted model passes ONNX validation."""
        import onnx

        onnx_path = packager.convert_to_onnx(valid_model_path)
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_conversion_uses_configured_opset(self, packager, valid_model_path):
        """Test that ONNX export uses the configured opset version."""
        import onnx

        onnx_path = packager.convert_to_onnx(valid_model_path)
        model = onnx.load(str(onnx_path))
        assert model.opset_import[0].version >= packager.config.onnx_opset_version

    def test_conversion_invalid_model_raises_conversion_error(self, packager, tmp_path):
        """Test that an invalid model file raises ConversionError."""
        bad_model_path = tmp_path / "bad_model.pt"
        bad_model_path.write_bytes(b"not a valid model")

        with pytest.raises(ConversionError, match="Failed to load PyTorch model"):
            packager.convert_to_onnx(bad_model_path)

    def test_conversion_nonexistent_file_raises_conversion_error(self, packager):
        """Test that a non-existent file raises ConversionError."""
        with pytest.raises(ConversionError, match="Failed to load PyTorch model"):
            packager.convert_to_onnx(Path("/nonexistent/model.pt"))


class TestValidateOnnx:
    """Tests for ModelPackager.validate_onnx()."""

    def test_valid_onnx_passes(self, packager, valid_model_path):
        """Test that a valid ONNX model passes validation."""
        onnx_path = packager.convert_to_onnx(valid_model_path)
        # Should not raise
        packager.validate_onnx(onnx_path)

    def test_invalid_onnx_raises_validation_error(self, packager, tmp_path):
        """Test that an invalid ONNX file raises ValidationError."""
        bad_onnx_path = tmp_path / "bad_model.onnx"
        bad_onnx_path.write_bytes(b"not a valid onnx model")

        with pytest.raises(ValidationError, match="Failed to validate ONNX model"):
            packager.validate_onnx(bad_onnx_path)

    def test_nonexistent_onnx_raises_validation_error(self, packager):
        """Test that a non-existent ONNX file raises ValidationError."""
        with pytest.raises(ValidationError):
            packager.validate_onnx(Path("/nonexistent/model.onnx"))
