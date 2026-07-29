#!/usr/bin/env python3
"""ROS CDK app entry point for the MNIST inference endpoint infrastructure on Alibaba Cloud."""

import ros_cdk_core as ros

from config import AlicloudDeployConfig

# Stack imports - stacks will be implemented in subsequent tasks
# from stacks.storage_stack import StorageStack
# from stacks.eas_stack import EasStack
# from stacks.access_stack import AccessStack

app = ros.App()

# Context values (passed via cdk.json or --context)
model_key = app.node.try_get_context("model_key") or "models/mnist/model.onnx"
instance_type = app.node.try_get_context("instance_type") or "ecs.gn6i-c4g1.xlarge"
region = app.node.try_get_context("region") or "cn-hangzhou"

# Configuration
config = AlicloudDeployConfig(
    model_path=model_key,
    region=region,
    instance_type=instance_type,
)

# Stack 1: OSS bucket for model artifacts
# storage_stack = StorageStack(app, "MnistStorageStack")

# Stack 2: PAI-EAS Triton inference service (depends on StorageStack's bucket)
# eas_stack = EasStack(
#     app,
#     "MnistEasStack",
#     oss_bucket_name=storage_stack.bucket_name,
#     model_key=model_key,
#     instance_type=instance_type,
#     use_spot=config.use_spot,
#     min_replicas=config.min_replicas,
#     max_replicas=config.max_replicas,
# )

# Stack 3: Auth and network access configuration
# access_stack = AccessStack(
#     app,
#     "MnistAccessStack",
#     service_name=eas_stack.service_name,
# )

# Cost tracking tags applied to all resources
# ros.Tags.of(app).add("Project", "mnist-inference")
# ros.Tags.of(app).add("Owner", "ruhuitay")
# ros.Tags.of(app).add("Cloud", "alicloud")

app.synth()
