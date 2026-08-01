"""Upload Triton model repository to Alibaba Cloud OSS.

Walks the local model/triton_repo/ directory and uploads all files to OSS,
preserving directory structure under a 'models/' prefix.

For example:
    model/triton_repo/mnist/config.pbtxt  ->  models/mnist/config.pbtxt
    model/triton_repo/mnist/1/model.onnx  ->  models/mnist/1/model.onnx

Run this BEFORE deploying ROS CDK stacks since EasStack references the model path in OSS.

Usage:
    export OSS_BUCKET="mnist-model-artifacts-alicloud"
    export ALICLOUD_REGION="eu-central-1"
    export ALIBABA_CLOUD_ACCESS_KEY_ID="..."
    export ALIBABA_CLOUD_ACCESS_KEY_SECRET="..."
    uv run python scripts/upload_model_alicloud.py
"""

import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv package is not installed.")
    print("Install with: uv add python-dotenv")
    sys.exit(1)

try:
    import oss2
except ImportError:
    print("ERROR: oss2 package is not installed.")
    print("Install with: uv sync --extra alicloud")
    sys.exit(1)

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# Local path to the Triton model repository
TRITON_REPO_DIR = Path(__file__).resolve().parent.parent / "model" / "triton_repo"

# OSS prefix for uploaded model files
OSS_PREFIX = "models"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1


def get_config() -> tuple[str, str, str, str]:
    """Read configuration from environment variables.

    Returns:
        Tuple of (bucket_name, region, access_key_id, access_key_secret)

    Exits with non-zero code if required variables are missing.
    """
    bucket_name = os.environ.get("OSS_BUCKET")
    region = os.environ.get("ALICLOUD_REGION")
    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    missing = []
    if not bucket_name:
        missing.append("OSS_BUCKET")
    if not region:
        missing.append("ALICLOUD_REGION")
    if not access_key_id:
        missing.append("ALIBABA_CLOUD_ACCESS_KEY_ID")
    if not access_key_secret:
        missing.append("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    return bucket_name, region, access_key_id, access_key_secret


def create_bucket_client(
    bucket_name: str, region: str, access_key_id: str, access_key_secret: str
) -> oss2.Bucket:
    """Create an OSS Bucket client.

    Args:
        bucket_name: Name of the OSS bucket.
        region: Alibaba Cloud region (e.g. cn-hangzhou).
        access_key_id: Alibaba Cloud access key ID.
        access_key_secret: Alibaba Cloud access key secret.

    Returns:
        oss2.Bucket instance ready for uploads.
    """
    endpoint = f"https://oss-{region}.aliyuncs.com"
    auth = oss2.Auth(access_key_id, access_key_secret)
    return oss2.Bucket(auth, endpoint, bucket_name)


def upload_file_with_retry(
    bucket: oss2.Bucket, oss_key: str, local_path: Path
) -> None:
    """Upload a single file to OSS with retry logic.

    Args:
        bucket: oss2.Bucket client.
        oss_key: Destination object key in OSS.
        local_path: Path to the local file.

    Raises:
        oss2.exceptions.OssError: If all retry attempts fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bucket.put_object_from_file(oss_key, str(local_path))
            return
        except oss2.exceptions.OssError as e:
            if attempt < MAX_RETRIES:
                print(
                    f"  WARNING: Upload failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {RETRY_DELAY_SECONDS}s..."
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise


def upload_triton_repo(bucket: oss2.Bucket) -> list[str]:
    """Walk the Triton model repository and upload all files to OSS.

    Files in model/triton_repo/ are uploaded under the 'models/' prefix.
    For example: model/triton_repo/mnist/config.pbtxt -> models/mnist/config.pbtxt

    Args:
        bucket: oss2.Bucket client.

    Returns:
        List of uploaded OSS keys.

    Raises:
        FileNotFoundError: If the Triton repo directory does not exist.
        oss2.exceptions.OssError: If upload fails after retries.
    """
    if not TRITON_REPO_DIR.exists():
        raise FileNotFoundError(
            f"Triton model repository not found at: {TRITON_REPO_DIR}"
        )

    uploaded_keys: list[str] = []

    for local_path in sorted(TRITON_REPO_DIR.rglob("*")):
        if not local_path.is_file():
            continue

        # Skip hidden files (e.g. .DS_Store)
        if any(part.startswith(".") for part in local_path.relative_to(TRITON_REPO_DIR).parts):
            continue

        # Build OSS key: model/triton_repo/mnist/X -> models/mnist/X
        relative_path = local_path.relative_to(TRITON_REPO_DIR)
        oss_key = f"{OSS_PREFIX}/{relative_path}"

        print(f"  Uploading: {relative_path} -> oss://{bucket.bucket_name}/{oss_key}")
        upload_file_with_retry(bucket, oss_key, local_path)
        uploaded_keys.append(oss_key)

    return uploaded_keys


def main() -> None:
    """Entry point - upload Triton model repository to OSS."""
    print("=" * 60)
    print("Alibaba Cloud OSS Model Upload")
    print("=" * 60)

    # Read configuration
    bucket_name, region, access_key_id, access_key_secret = get_config()

    print(f"Bucket:  {bucket_name}")
    print(f"Region:  {region}")
    print(f"Source:  {TRITON_REPO_DIR}")
    print(f"Prefix:  {OSS_PREFIX}/")
    print("-" * 60)

    # Create OSS client
    bucket = create_bucket_client(bucket_name, region, access_key_id, access_key_secret)

    # Upload all model files
    try:
        uploaded_keys = upload_triton_repo(bucket)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except oss2.exceptions.OssError as e:
        print(f"ERROR: OSS upload failed after {MAX_RETRIES} attempts: {e}")
        sys.exit(1)

    if not uploaded_keys:
        print("ERROR: No files found to upload in Triton repo directory.")
        sys.exit(1)

    # Print success summary
    print("-" * 60)
    print(f"Successfully uploaded {len(uploaded_keys)} file(s).")
    print(f"OSS URI: oss://{bucket_name}/{OSS_PREFIX}/mnist/")
    print("=" * 60)


if __name__ == "__main__":
    main()
