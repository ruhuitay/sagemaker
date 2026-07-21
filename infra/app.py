#!/usr/bin/env python3
"""CDK app entry point for the MNIST inference endpoint infrastructure."""

import aws_cdk as cdk

from stacks.storage_stack import StorageStack
from stacks.sagemaker_stack import SageMakerStack
from stacks.api_stack import ApiStack

app = cdk.App()

# Context values (passed via cdk.json or --context)
model_key = app.node.try_get_context("model_key") or "models/mnist/model.tar.gz"
instance_type = app.node.try_get_context("instance_type") or "ml.c5.large"

env = cdk.Environment(region="eu-west-1")

# Stack 1: S3 bucket for model artifacts
storage_stack = StorageStack(app, "MnistStorageStack", env=env)

# Stack 2: SageMaker endpoint (depends on StorageStack's bucket)
sagemaker_stack = SageMakerStack(
    app,
    "MnistSageMakerStack",
    model_bucket=storage_stack.bucket_name,
    model_key=model_key,
    instance_type=instance_type,
    env=env,
)

# Stack 3: API Gateway with direct SageMaker integration
api_stack = ApiStack(
    app,
    "MnistApiStack",
    sagemaker_endpoint_name=sagemaker_stack.endpoint_name,
    env=env,
)

app.synth()
