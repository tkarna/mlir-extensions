//===- ValueUtils.h - Zero-checking utilities -------------------*- C++ -*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for writing passes.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_VALUEUTILS_H
#define _IMEX_VALUEUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace imex {

// Returns true if the value is a constant float or integer.
bool isValConstZero(mlir::Value val);

// Returns true if the op defining `val` represents a zero filled tensor.
bool isZeroTensor(mlir::Value val);

// Returns the strides of `val`. The method returns something useful
// only if the `val` type is a strided memref.
mlir::FailureOr<mlir::SmallVector<int64_t>> getStrides(mlir::Value val);

// Returns the strides of `val`. The method returns something useful
// only if the `val` type is a strided memref and the strides are statically
// known.
mlir::FailureOr<mlir::SmallVector<int64_t>> getStaticStrides(mlir::Value val);

// Return the offset and ptr for `val`. Assert if `val`
// is not a memref.
std::pair<mlir::Value, mlir::Value> getPtrAndOffset(mlir::OpBuilder &builder,
                                                    mlir::Value operand);

// Create a 'mlir::vector' constant from a list of values.
template <typename T>
mlir::Value createTypedVector(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::ArrayRef<T> values,
                              mlir::Type elementType) {
  mlir::VectorType vectorType =
      mlir::VectorType::get({static_cast<int64_t>(values.size())}, elementType);
  mlir::DenseElementsAttr denseAttr =
      mlir::DenseElementsAttr::get(vectorType, values);
  auto vector =
      rewriter.create<mlir::arith::ConstantOp>(loc, vectorType, denseAttr)
          .getResult();
  return vector;
}

// Flatten a 2D memref to a 1D memref.
mlir::Value flattenMemref(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value srcMemref);

// Return true if the memref has shared memory space.
bool hasSharedMemSpace(mlir::Value memref);

// Go through all parent 'memref.subview' ops for the given `memref`
// and return the folded offsets of all subviews and the root memref.
void computeSubviewOffsets(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Value memref,
                           mlir::SmallVector<mlir::Value> &resultOffsets,
                           mlir::Value &resultRootMemref);

// Return the strides of the memref
mlir::SmallVector<mlir::OpFoldResult>
getMemrefStrides(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value memref);

// Squeeze the leading dimensions of a given memref up to 'maxDims'.
mlir::FailureOr<mlir::Value> reduceMemrefDims(mlir::PatternRewriter &rewriter,
                                              mlir::Location loc,
                                              mlir::Value memref,
                                              size_t maxDims = 2);

// Squeeze the leading dimensions of memref operands of a given 'linalgOp'.
mlir::LogicalResult maybeSqueezeDims(mlir::PatternRewriter &rewriter,
                                     mlir::linalg::LinalgOp linalgOp,
                                     size_t maxDims = 2);

// Return if a memref with the given shape can be squeezed to the shape of
// 'maxDims'. Only leading dimensions are considered squeezable.
bool canSqueezeDims(llvm::ArrayRef<int64_t> shape, size_t maxDims = 2);

} // namespace imex

#endif // _IMEX_VALUEUTILS_H
