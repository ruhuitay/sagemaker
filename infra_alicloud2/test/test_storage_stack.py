#!/usr/bin/env python3
import unittest
import ros_cdk_core as core
from stacks.storage_stack import StorageStack


class TestStack(unittest.TestCase):
    def setUp(self):
        pass

    def test_stack(self):
        app = core.App()
        stack = StorageStack(app, "testinfra-alicloud2")
        artifact = app.synth().get_stack_artifact(stack.artifact_id).template
        expect = {
           "Metadata": {
             "ALIYUN::ROS::Interface": {
               "TemplateTags": [
                 "Create by ROS CDK"
               ]
             }
           },
           "ROSTemplateFormatVersion": "2015-09-01"
         }
        self.assertDictContainsSubset(artifact, expect)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()