#!/usr/bin/env python3

import ros_cdk_core as core

from test_ali.test_ali_stack import TestAliStack


app = core.App()

TestAliStack(app, "test-ali")

app.synth()
