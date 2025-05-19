# ===-- xegpu.py - XeGPUTransformOps bindings -----------------*- Python -*-===#
#
# Copyright 2025 Intel Corporation
# Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

from ..._mlir_libs import get_dialect_registry
from ..._mlir_libs._imex_mlir.transform.xegpu import register_dialect_extension

register_dialect_extension(get_dialect_registry())
