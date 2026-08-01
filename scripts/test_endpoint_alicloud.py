import numpy as np
from dotenv import load_dotenv
import os
# To install the tritonclient package, run the following command: pip install tritonclient
import tritonclient.http as httpclient

load_dotenv() 

# The public endpoint generated after the service is deployed. Do not include the http:// prefix.
url = os.environ["EAS_URL"]
token = os.environ["EAS_TOKEN"]

triton_client = httpclient.InferenceServerClient(url=url)

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
    return image

image = create_sample_input()
print(image)
image = image.astype(np.float32)

inputs = []
inputs.append(httpclient.InferInput('input', image.shape, "FP32"))
inputs[0].set_data_from_numpy(image, binary_data=False)
outputs = []
outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))  # Get a 1000-dimensional vector.

# Specify the model name, request token, inputs, and outputs.
results = triton_client.infer(
    model_name="mnist",
    model_version="1",
    inputs=inputs,
    outputs=outputs,
    headers={"Authorization": token},
)
output_data0 = results.as_numpy('output')
print(output_data0.shape)
print(output_data0)