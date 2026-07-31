#!/usr/bin/env python3

import ros_cdk_core as core

from stacks.storage_stack import StorageStack
from stacks.eas_stack import EasStack
from stacks.access_stack import AccessStack


app = core.App()

StorageStack(app, "StorageStack")
EasStack(app, "EasStack")
AccessStack(app, "AccessStack")

app.synth()
