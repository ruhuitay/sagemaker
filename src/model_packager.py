"""Model packaging pipeline for MNIST inference endpoint.

Loads a locally-trained MNIST model, converts to ONNX, packages in Triton
model repository format, and uploads to S3.
"""

import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config import PackagerConfig
from src.exceptions import ModelLoadError, ConversionError, ValidationError, UploadError

import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    """Simple CNN for MNIST digit classification (28x28 grayscale → 10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelPackager:
    """Loads, converts, validates, packages, and uploads the MNIST model."""

    def __init__(self, config: PackagerConfig):
        """Initialize with configuration (model path, S3 bucket, prefix)."""
        self.config = config

    def convert_to_onnx(self, model_path: Path) -> Path:
        """Convert PyTorch model to ONNX format (opset >= 11).

        Args:
            model_path: Path to the local PyTorch model (.pt file).

        Returns:
            Path to the converted ONNX model file.

        Raises:
            ModelLoadError: If the model file does not exist.
            ConversionError: If model cannot be loaded or converted.
        """
        if not model_path.exists():
            raise ModelLoadError(
                f"Model file not found at: {model_path}"
            )

        try:
            model = MNISTNet()
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Failed to load PyTorch model from {model_path}: {e}")
            raise ConversionError(f"Failed to load PyTorch model: {e}")

        onnx_path = model_path.parent / "mnist_model.onnx"
        onnx_data_path = model_path.parent / "mnist_model.onnx.data"
        dummy_input = torch.randn(1, 1, 28, 28)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=self.config.onnx_opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        except Exception as e:
            print(f"Failed to export model to ONNX: {e}")
            raise ConversionError(f"Failed to convert model to ONNX: {e}")

        print(f"Model converted to ONNX: {onnx_path}")
        return onnx_path, onnx_data_path

    def validate_onnx(self, onnx_path: Path) -> None:
        """Validate ONNX model using onnx.checker.check_model.

        Args:
            onnx_path: Path to the ONNX model file.

        Raises:
            ValidationError: If model is structurally invalid.
        """
        import onnx

        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
        except Exception as e:
            print(f"Failed to validate ONNX model at {onnx_path}: {e}")
            raise ValidationError(f"Failed to validate ONNX model: {e}")

        print(f"ONNX model validated: {onnx_path}")

    def create_model_repository(self, onnx_path: Path, onnx_data_path: Path) -> Path:
        """Create Triton model repository directory structure.

        Creates the following layout:
            model_repository/
              mnist/
                config.pbtxt
                1/
                  model.onnx

        Args:
            onnx_path: Path to the validated ONNX model file.

        Returns:
            Path to the repository root directory (model_repository/).
        """
        repo_root = Path(tempfile.mkdtemp()) / "model_repository"
        model_dir = repo_root / "mnist"
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True)

        # Write config.pbtxt
        config_content = (
            'name: "mnist"\n'
            'backend: "onnxruntime"\n'
            'max_batch_size: 8\n'
            'input [\n'
            '  {\n'
            '    name: "input"\n'
            '    data_type: TYPE_FP32\n'
            '    dims: [1, 28, 28]\n'
            '  }\n'
            ']\n'
            'output [\n'
            '  {\n'
            '    name: "output"\n'
            '    data_type: TYPE_FP32\n'
            '    dims: [10]\n'
            '  }\n'
            ']\n'
        )
        config_path = model_dir / "config.pbtxt"
        config_path.write_text(config_content)

        # Copy ONNX model into version directory
        shutil.copy2(str(onnx_path), str(version_dir / "model.onnx"))
        shutil.copy2(str(onnx_data_path), str(version_dir / "mnist_model.onnx.data"))
       

        print(f"Model repository created at: {repo_root}")
        return repo_root

    def package_artifact(self, repo_path: Path) -> Path:
        """Package model repository as model.tar.gz.

        Creates a tar.gz archive that preserves the Triton directory hierarchy.
        When extracted, the archive contents start from the model_repository root,
        so Triton sees the expected structure (mnist/config.pbtxt, mnist/1/model.onnx).

        Args:
            repo_path: Path to the Triton model repository root (model_repository/).

        Returns:
            Path to the created model.tar.gz archive.
        """
        artifact_path = repo_path.parent / "model.tar.gz"

        with tarfile.open(str(artifact_path), "w:gz") as tar:
            # Add contents relative to repo_path so the archive preserves
            # the Triton directory hierarchy (mnist/config.pbtxt, mnist/1/model.onnx)
            for item in repo_path.iterdir():
                tar.add(str(item), arcname=item.name)

        print(f"Model artifact packaged: {artifact_path}")
        return artifact_path

    def upload_to_s3(self, artifact_path: Path) -> str:
        """Upload artifact to S3 with retry logic (3 attempts, 1s delay).

        Args:
            artifact_path: Path to the model.tar.gz file to upload.

        Returns:
            S3 URI of the uploaded artifact (s3://bucket/key).

        Raises:
            UploadError: After 3 failed upload attempts.
        """
        filename = artifact_path.name
        s3_key = f"{self.config.s3_prefix}{filename}"
        s3_client = boto3.client("s3")

        last_error = None
        for attempt in range(1, 4):
            try:
                s3_client.upload_file(
                    str(artifact_path),
                    self.config.s3_bucket,
                    s3_key,
                )
                s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
                print(f"Upload successful: {s3_uri}")
                return s3_uri
            except (ClientError, BotoCoreError) as e:
                last_error = e
                print(
                    f"Upload attempt {attempt}/3 failed: {e}"
                )
                if attempt < 3:
                    time.sleep(1)

        raise UploadError(
            f"Failed to upload {filename} to s3://{self.config.s3_bucket}/{s3_key} "
            f"after 3 attempts: {last_error}"
        )

    def run(self) -> str:
        """Execute full pipeline: load, convert, validate, create repo, package, upload.

        Returns:
            S3 URI of the uploaded model artifact.
        """
        model_path = Path(self.config.model_path)
        onnx_path, onnx_data_path = self.convert_to_onnx(model_path)
        self.validate_onnx(onnx_path)
        repo_path = self.create_model_repository(onnx_path, onnx_data_path)
        artifact_path = self.package_artifact(repo_path)
        s3_uri = self.upload_to_s3(artifact_path)
        return s3_uri
