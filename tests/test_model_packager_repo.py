"""Unit tests for ModelPackager.create_model_repository() and package_artifact()."""

import tarfile
from pathlib import Path

import pytest

from src.config import PackagerConfig
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
def onnx_file(tmp_path):
    """Create a fake ONNX model file for testing."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx model content for testing")
    return onnx_path


class TestCreateModelRepository:
    """Tests for create_model_repository() method."""

    def test_returns_path_to_model_repository(self, packager, onnx_file):
        """create_model_repository() returns path ending with model_repository."""
        repo_path = packager.create_model_repository(onnx_file)
        assert repo_path.name == "model_repository"
        assert repo_path.exists()

    def test_creates_mnist_directory(self, packager, onnx_file):
        """Creates mnist/ directory inside model_repository."""
        repo_path = packager.create_model_repository(onnx_file)
        mnist_dir = repo_path / "mnist"
        assert mnist_dir.exists()
        assert mnist_dir.is_dir()

    def test_creates_config_pbtxt(self, packager, onnx_file):
        """Creates config.pbtxt in the mnist/ directory."""
        repo_path = packager.create_model_repository(onnx_file)
        config_path = repo_path / "mnist" / "config.pbtxt"
        assert config_path.exists()
        assert config_path.is_file()

    def test_config_pbtxt_contains_correct_platform(self, packager, onnx_file):
        """config.pbtxt specifies platform as onnxruntime_onnx."""
        repo_path = packager.create_model_repository(onnx_file)
        config_content = (repo_path / "mnist" / "config.pbtxt").read_text()
        assert 'platform: "onnxruntime_onnx"' in config_content

    def test_config_pbtxt_contains_correct_model_name(self, packager, onnx_file):
        """config.pbtxt specifies model name as mnist."""
        repo_path = packager.create_model_repository(onnx_file)
        config_content = (repo_path / "mnist" / "config.pbtxt").read_text()
        assert 'name: "mnist"' in config_content

    def test_config_pbtxt_contains_max_batch_size(self, packager, onnx_file):
        """config.pbtxt sets max_batch_size to 8."""
        repo_path = packager.create_model_repository(onnx_file)
        config_content = (repo_path / "mnist" / "config.pbtxt").read_text()
        assert "max_batch_size: 8" in config_content

    def test_config_pbtxt_contains_input_shape(self, packager, onnx_file):
        """config.pbtxt defines input with shape [1, 28, 28] and FP32 type."""
        repo_path = packager.create_model_repository(onnx_file)
        config_content = (repo_path / "mnist" / "config.pbtxt").read_text()
        assert "dims: [1, 28, 28]" in config_content
        assert "data_type: TYPE_FP32" in config_content

    def test_config_pbtxt_contains_output_shape(self, packager, onnx_file):
        """config.pbtxt defines output with shape [10] and FP32 type."""
        repo_path = packager.create_model_repository(onnx_file)
        config_content = (repo_path / "mnist" / "config.pbtxt").read_text()
        assert "dims: [10]" in config_content

    def test_creates_version_directory(self, packager, onnx_file):
        """Creates 1/ version subdirectory inside mnist/."""
        repo_path = packager.create_model_repository(onnx_file)
        version_dir = repo_path / "mnist" / "1"
        assert version_dir.exists()
        assert version_dir.is_dir()

    def test_copies_onnx_model_to_version_dir(self, packager, onnx_file):
        """Copies the ONNX model as model.onnx inside version directory."""
        repo_path = packager.create_model_repository(onnx_file)
        model_file = repo_path / "mnist" / "1" / "model.onnx"
        assert model_file.exists()
        assert model_file.read_bytes() == onnx_file.read_bytes()


class TestPackageArtifact:
    """Tests for package_artifact() method."""

    def test_returns_path_to_tar_gz(self, packager, onnx_file):
        """package_artifact() returns path to model.tar.gz file."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)
        assert artifact_path.name == "model.tar.gz"
        assert artifact_path.exists()

    def test_creates_valid_tar_gz_archive(self, packager, onnx_file):
        """Created archive is a valid gzipped tar file."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)
        assert tarfile.is_tarfile(str(artifact_path))

    def test_archive_contains_config_pbtxt(self, packager, onnx_file):
        """Archive contains mnist/config.pbtxt."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)

        with tarfile.open(str(artifact_path), "r:gz") as tar:
            names = tar.getnames()
        assert "mnist/config.pbtxt" in names

    def test_archive_contains_model_onnx(self, packager, onnx_file):
        """Archive contains mnist/1/model.onnx."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)

        with tarfile.open(str(artifact_path), "r:gz") as tar:
            names = tar.getnames()
        assert "mnist/1/model.onnx" in names

    def test_archive_preserves_directory_hierarchy(self, packager, onnx_file):
        """Archive preserves the Triton directory hierarchy when extracted."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)

        with tarfile.open(str(artifact_path), "r:gz") as tar:
            names = tar.getnames()

        # Verify the Triton directory structure is preserved
        assert any(n.startswith("mnist/") for n in names)
        assert "mnist/config.pbtxt" in names
        assert "mnist/1/model.onnx" in names

    def test_extracted_model_matches_original(self, packager, onnx_file, tmp_path):
        """Extracting the archive produces files with identical content."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(str(artifact_path), "r:gz") as tar:
            tar.extractall(path=str(extract_dir))

        extracted_model = extract_dir / "mnist" / "1" / "model.onnx"
        assert extracted_model.exists()
        assert extracted_model.read_bytes() == onnx_file.read_bytes()

    def test_extracted_config_matches_original(self, packager, onnx_file, tmp_path):
        """Extracting the archive produces config.pbtxt with expected content."""
        repo_path = packager.create_model_repository(onnx_file)
        artifact_path = packager.package_artifact(repo_path)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(str(artifact_path), "r:gz") as tar:
            tar.extractall(path=str(extract_dir))

        extracted_config = extract_dir / "mnist" / "config.pbtxt"
        original_config = repo_path / "mnist" / "config.pbtxt"
        assert extracted_config.exists()
        assert extracted_config.read_text() == original_config.read_text()
