//===- MatcherUtils.h - Matcher utils ---------------------------*- C++ -*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for matching tensor ops.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_MATCHERUTILS_H
#define _IMEX_MATCHERUTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace imex {
namespace structured_match {

// Returns true if the linalg operation is a 2d eltwise floating point addition.
bool isTwoDAddOp(
    mlir::linalg::LinalgOp linalgOp,
    mlir::SmallVectorImpl<mlir::Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic is a 2d eltwise floating point relu
// operation.
bool isTwoDReluOp(
    mlir::linalg::LinalgOp linalgOp,
    mlir::SmallVectorImpl<mlir::Value> *capturedOperands = nullptr);

} // namespace structured_match
} // namespace imex

#endif // _IMEX_MATCHERUTILS_H
