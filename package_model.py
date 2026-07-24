"""Run the full model packaging pipeline and upload to S3.

Converts PyTorch model to ONNX, validates, packages as model.tar.gz,
and uploads to the MnistStorageStack S3 bucket.

Usage:
    uv run python package_model.py
"""

from src.config import PackagerConfig
from src.model_packager import ModelPackager


def main():
    # Configure
    config = PackagerConfig(
        model_path="model/mnist_model.pt",
        s3_bucket="mniststoragestack-modelartifactsbucket80acad84-13pktmijk8yc",
        s3_prefix="models/mnist/",
    )

    packager = ModelPackager(config)

    # Run full pipeline (convert, validate, package, upload)
    print("=" * 50)
    print("Running full model packaging pipeline...")
    s3_uri = packager.run()

    print("=" * 50)
    print(f"Done! Model uploaded to: {s3_uri}")
    print("You can now deploy MnistSageMakerStack.")


if __name__ == "__main__":
    main()
