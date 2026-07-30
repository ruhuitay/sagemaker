"""Test invoke the SageMaker MNIST endpoint.

Sends a sample 28x28 image to the Triton endpoint and prints the predicted digit.

Usage:
    uv run python main.py
"""

import json

import boto3
import numpy as np


def create_sample_input():
    """Create a sample MNIST-like input (a rough '7' shape)."""
    image = np.zeros((1, 1, 28, 28), dtype=np.float32)
    # Draw a simple '7' pattern
    image[0, 0, 5, 8:20] = 1.0   # top horizontal
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
    return image


def invoke_endpoint(endpoint_name: str):
    """Invoke the SageMaker endpoint with a sample input."""
    client = boto3.client("sagemaker-runtime", region_name="eu-west-1")

    # Triton expects a specific JSON payload format
    sample = create_sample_input()
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 1, 28, 28],
                "datatype": "FP32",
                "data": sample.flatten().tolist(),
            }
        ]
    }

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode())
    print("Raw response:", json.dumps(result, indent=2))

    # Parse prediction
    outputs = result.get("outputs", [])
    if outputs:
        logits = outputs[0]["data"]
        predicted_digit = np.argmax(logits)
        print(f"\nPredicted digit: {predicted_digit}")
        print(f"Confidence scores: {[f'{x:.3f}' for x in logits]}")
    else:
        print("No outputs in response")


def main():
    # Replace with your actual endpoint name from CDK output
    endpoint_name = "MnistSageMakerStack-MnistEndpoint"

    print(f"Invoking endpoint: {endpoint_name}")
    print("=" * 50)
    invoke_endpoint(endpoint_name)


if __name__ == "__main__":
    main()
