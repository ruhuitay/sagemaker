"""Test invoke the Alibaba Cloud PAI-EAS MNIST endpoint.

Sends a sample 28x28 image to the Triton V2 endpoint on PAI-EAS
and prints the predicted digit, confidence, and response time.

Usage:
    export ALICLOUD_ENDPOINT_URL="https://xxx.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/mnist"
    export ALICLOUD_API_TOKEN="your-access-token"
    uv run python scripts/test_endpoint_alicloud.py
"""

import json
import os
import sys
import time

import numpy as np
import requests


def create_sample_input() -> list[float]:
    """Create a sample MNIST-like input (a rough '7' shape).

    Returns a flat list of 784 FP32 values.
    """
    image = np.zeros((1, 1, 28, 28), dtype=np.float32)
    # Draw a simple '7' pattern
    image[0, 0, 5, 8:20] = 1.0  # top horizontal
    image[0, 0, 6, 16:18] = 1.0
    image[0, 0, 7, 15:17] = 1.0
    image[0, 0, 8, 14:16] = 1.0
    image[0, 0, 9, 13:15] = 1.0
    image[0, 0, 10, 12:14] = 1.0
    image[0, 0, 11, 12:14] = 1.0
    image[0, 0, 12, 11:13] = 1.0
    image[0, 0, 13, 11:13] = 1.0
    image[0, 0, 14, 10:12] = 1.0
    image[0, 0, 15, 10:12] = 1.0
    image[0, 0, 16, 10:12] = 1.0
    image[0, 0, 17, 10:12] = 1.0
    image[0, 0, 18, 10:12] = 1.0
    image[0, 0, 19, 10:12] = 1.0
    image[0, 0, 20, 10:12] = 1.0
    return image.flatten().tolist()


def invoke_endpoint(endpoint_url: str, token: str) -> None:
    """Invoke the PAI-EAS Triton endpoint with a sample input."""
    # Build Triton V2 JSON payload
    pixel_data = create_sample_input()
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 1, 28, 28],
                "datatype": "FP32",
                "data": pixel_data,
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": token,
    }

    print(f"Invoking endpoint: {endpoint_url}")
    print("=" * 50)

    start_time = time.time()
    response = requests.post(
        endpoint_url,
        headers=headers,
        json=payload,
        timeout=30,
    )
    elapsed_ms = (time.time() - start_time) * 1000

    if response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)

    result = response.json()

    # Parse prediction from Triton V2 response
    outputs = result.get("outputs", [])
    if not outputs:
        print("ERROR: No outputs in response")
        print(f"Raw response: {json.dumps(result, indent=2)}")
        sys.exit(1)

    logits = outputs[0]["data"]
    predicted_digit = int(np.argmax(logits))
    confidence = float(np.max(logits))

    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.1f}%)")
    print(f"Response time: {elapsed_ms:.0f}ms")
    print(f"\nAll scores: {[f'{x:.4f}' for x in logits]}")


def main() -> None:
    """Entry point - read config from env vars and invoke endpoint."""
    endpoint_url = os.environ.get("ALICLOUD_ENDPOINT_URL")
    token = os.environ.get("ALICLOUD_API_TOKEN")

    if not endpoint_url:
        print("ERROR: ALICLOUD_ENDPOINT_URL environment variable is not set")
        sys.exit(1)

    if not token:
        print("ERROR: ALICLOUD_API_TOKEN environment variable is not set")
        sys.exit(1)

    try:
        invoke_endpoint(endpoint_url, token)
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out (30s)")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Connection failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
