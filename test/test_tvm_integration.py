import unittest
from test.util import TVMTest
import torch_tvm

class IntegrationTest(TVMTest):
    def test_basic(self):
        assert torch_tvm.test() == 1337

if __name__ == "__main__":
    unittest.main()

