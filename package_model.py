"""Run the model packaging pipeline locally (no S3 upload).

Creates the Triton model repository and model.tar.gz artifact.
Use this to verify the packaging works before deploying.

Usage:
    uv run python package_model.py
"""

from pathlib import Path

from src.config import PackagerConfig
from src.model_packager import ModelPackager


def main():
    # Configure — S3 bucket is a placeholder since we skip upload
    config = PackagerConfig(
        model_path="model/mnist_model.pt",
        s3_bucket="placeholder",
        s3_prefix="models/mnist/",
    )

    packager = ModelPackager(config)
    model_path = Path(config.model_path)

    # Step 1: Convert to ONNX
    print("=" * 50)
    print("Step 1: Converting PyTorch model to ONNX...")
    onnx_path = packager.convert_to_onnx(model_path)

    # Step 2: Validate ONNX
    print("=" * 50)
    print("Step 2: Validating ONNX model...")
    packager.validate_onnx(onnx_path)

    # Step 3: Create Triton model repository
    print("=" * 50)
    print("Step 3: Creating Triton model repository...")
    repo_path = packager.create_model_repository(onnx_path)

    # Step 4: Package as model.tar.gz
    print("=" * 50)
    print("Step 4: Packaging model artifact...")
    artifact_path = packager.package_artifact(repo_path)

    # Summary
    print("=" * 50)
    print("Done! Artifact ready for S3 upload.")
    print(f"  ONNX model:      {onnx_path}")
    print(f"  Model repo:      {repo_path}")
    print(f"  Artifact:        {artifact_path}")
    print(f"  Artifact size:   {artifact_path.stat().st_size / 1024:.1f} KB")

    # Show archive contents
    import tarfile
    print("\nArchive contents:")
    with tarfile.open(str(artifact_path), "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  {member.name} ({member.size} bytes)")


if __name__ == "__main__":
    main()
