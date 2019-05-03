from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tvm import relay # This registers all the schedules

from ._torch_tvm import *

from tvm._ffi.function import _init_api # This lets us use PackedFunc with torch_tvm
_init_api("torch_tvm")
