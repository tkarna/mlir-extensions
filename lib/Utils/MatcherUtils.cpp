//===- MatcherUtils.cpp - Matcher utils -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Utils/StructuredOpMatcher.h"
#include "imex/Utils/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace imex {
namespace structured_match {

// Return true if all the operand have the same type, i.e., no implicit
// conversion in the linalgOp.
static mlir::LogicalResult hasEqualOperandTypes(mlir::Operation *operation) {
  if (!mlir::isa<mlir::linalg::LinalgOp>(operation))
    return failure();
  auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(operation);
  mlir::OpOperand &outputOperand = linalgOp.getDpsInitsMutable()[0];
  auto elemType = getElementTypeOrSelf(outputOperand.get().getType());

  if (!llvm::all_of(linalgOp.getDpsInitsMutable(),
                    [&](mlir::OpOperand &operand) {
                      auto currentOperandType =
                          getElementTypeOrSelf(operand.get().getType());
                      return currentOperandType == elemType;
                    })) {
    return failure();
  }

  if (!llvm::all_of(linalgOp.getDpsInputOperands(),
                    [&](mlir::OpOperand *operand) {
                      auto currentOperandType =
                          getElementTypeOrSelf(operand->get().getType());
                      return currentOperandType == elemType;
                    })) {
    return failure();
  }
  return success();
}

static bool isTppOp(mlir::linalg::LinalgOp linalgOp) {
  // clang-format off
  auto tppMatcher =
    StructuredOpMatcher::make<mlir::linalg::LinalgOp>()
      .output(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasStaticShape())
      .operation(NumRegions(EqualsTo(1)))
      .output(MatchAll(), HasStaticStrides())
      .output(MatchAll(), HasElementType<mlir::FloatType>())
      .input(MatchAll(), HasStaticStrides())
      .operation(VerifyOpProperty(hasEqualOperandTypes));
  // clang-format on
  return tppMatcher.match(linalgOp);
}

static bool isTppBinaryOp(mlir::linalg::LinalgOp linalgOp) {
  // clang-format off
  auto binaryMatcher =
      StructuredOpMatcher::make<mlir::linalg::LinalgOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(1), EqualsTo(2))))
          .output(MatchAll(), HasRank({2}))
          // TODO: (lorenzo) When we introduce broadcast op we
          // will restrict the input to 2d tiles.
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)))
          .output(MatchAll(), HasMap(Identity()))
          .input(MatchAll(), HasMap(BroadcastableProjectedPermutation()));
  // clang-format on
  return isTppOp(linalgOp) && binaryMatcher.match(linalgOp);
}

static bool isTppUnaryOp(mlir::linalg::LinalgOp linalgOp) {
  // clang-format off
  auto unaryMatcher =
      StructuredOpMatcher::make<mlir::linalg::LinalgOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(0), EqualsTo(1))))
          // TODO: (lorenzo) When we introduce reduce operations
          // we will relax this constraint, and allow SCALAR, 1d
          // and 2d.
          .output(MatchAll(), HasRank({2}))
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)));
  // clang-format on
  return isTppOp(linalgOp) && unaryMatcher.match(linalgOp);
}

template <typename OpTy>
static bool
isTwoDEltWiseOpOfTypeTy(mlir::linalg::LinalgOp linalgOp,
                        mlir::SmallVectorImpl<mlir::Value> *operands) {
  // clang-format off
  auto matcher =
    StructuredOpMatcher::make<mlir::linalg::LinalgOp>().region(
      MatchOne(0), WithSingleOp<OpTy>(operands));
  // clang-format on
  return isTppBinaryOp(linalgOp) && matcher.match(linalgOp);
}

bool isTwoDAddOp(mlir::linalg::LinalgOp linalgOp,
                 mlir::SmallVectorImpl<mlir::Value> *operands) {
  return isTwoDEltWiseOpOfTypeTy<mlir::arith::AddFOp>(linalgOp, operands);
}

static bool hasReluBody(mlir::Operation *op,
                        mlir::SmallVectorImpl<mlir::Value> *captured) {
  if (!mlir::isa<mlir::linalg::LinalgOp>(op))
    return false;
  auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(op);
  mlir::Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;
  mlir::Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;
  mlir::Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());

  // If lhs is a zero get rhs as input for the relu if it is a block argument,
  // return false otherwise.
  auto getOperand = [&](mlir::Value lhs, mlir::Value rhs) -> bool {
    if (isZeroTensor(lhs)) {
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(rhs);
      if (!blockArg || blockArg.getParentBlock() != linalgOp.getBlock())
        return false;
      mlir::OpOperand *operand =
          linalgOp.getMatchingOpOperand(mlir::cast<mlir::BlockArgument>(rhs));
      if (captured) {
        captured->push_back(operand->get());
        captured->push_back(linalgOp.getDpsInitOperand(0)->get());
      }
      return true;
    }
    return false;
  };

  // Multiple patterns map to Relu.
  if (auto maxfOp = mlir::dyn_cast<mlir::arith::MaximumFOp>(innerOp)) {
    // Pattern 1 - arith.maximumf(tensor, const 0)
    if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
      return false;
    mlir::Value maxfLhs = maxfOp.getLhs();
    mlir::Value maxfRhs = maxfOp.getRhs();

    return (getOperand(maxfLhs, maxfRhs) || getOperand(maxfRhs, maxfLhs));
  }
  if (auto cmpfOp = mlir::dyn_cast<mlir::arith::CmpFOp>(innerOp)) {
    // Pattern 2 - arith.cmpf, arith.select
    //
    // x = arith.cmpf ugt, value, 0
    // y = arith.select x, value, 0
    //
    // NOTE: ugt - unsigned greater than, one of the predicates
    if (linalgOp.getBlock()->getOperations().size() != 3)
      return false;

    auto opIterator = linalgOp.getBlock()->getOperations().begin();
    mlir::Operation *cmpOp = &*opIterator;
    opIterator++;

    if (!mlir::isa<mlir::arith::SelectOp>(&*opIterator))
      return false;

    mlir::Operation *selectOp = &*opIterator;
    opIterator++;
    if (yieldOp != &*opIterator)
      return false;

    if (yieldOp->getOperand(0).getDefiningOp() != selectOp)
      return false;

    if (selectOp->getOperand(0).getDefiningOp() != cmpOp)
      return false;

    auto cmpPredicate = cmpfOp.getPredicate();
    mlir::Value cmpLhs = cmpfOp.getLhs();
    mlir::Value cmpRhs = cmpfOp.getRhs();

    auto selOp = mlir::cast<mlir::arith::SelectOp>(selectOp);
    auto trueVal = selOp.getTrueValue();
    auto falseVal = selOp.getFalseValue();

    if (cmpPredicate == mlir::arith::CmpFPredicate::UGT ||
        cmpPredicate == mlir::arith::CmpFPredicate::UGE) {
      if (cmpLhs == trueVal && isZeroTensor(cmpRhs) && isZeroTensor(falseVal)) {
        // case: %in > 0 ? %in : 0
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
      if (isZeroTensor(cmpLhs) && isZeroTensor(trueVal) && cmpRhs == falseVal) {
        // case: 0 > %in ? 0 : %in
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
    } else if (cmpPredicate == mlir::arith::CmpFPredicate::ULT ||
               cmpPredicate == mlir::arith::CmpFPredicate::ULE) {
      if (cmpLhs == falseVal && isZeroTensor(cmpRhs) && isZeroTensor(trueVal)) {
        // case: %in < 0 ? 0 : %in
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
      if (isZeroTensor(cmpLhs) && isZeroTensor(falseVal) && cmpRhs == trueVal) {
        // case: 0 < %in ? %in : 0
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
    }
  }
  return false;
}

namespace {
// Helper matcher functor for relu detection.
struct WithReluBody {
  WithReluBody() = delete;
  WithReluBody(mlir::SmallVectorImpl<mlir::Value> *captures)
      : captures(captures){};

  bool operator()(mlir::Region *region, mlir::Operation *op) {
    auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op);
    if (!linalgOp)
      return false;

    return hasReluBody(linalgOp, captures);
  }

private:
  mlir::SmallVectorImpl<mlir::Value> *captures;
};
} // namespace

// Return true if the linalg.generic can be mapped to a tpp.relu.
bool isTwoDReluOp(mlir::linalg::LinalgOp linalgOp,
                  mlir::SmallVectorImpl<mlir::Value> *operands) {
  // clang-format off
  auto reluMatcher =
    StructuredOpMatcher::make<mlir::linalg::LinalgOp>()
    .output(MatchAll(), HasMap(Identity()))
    .input(MatchAll(), HasMap(BroadcastableProjectedPermutation()))
    .region(MatchOne(0), WithReluBody(operands));
  // clang-format on
  return isTppUnaryOp(linalgOp) && reluMatcher.match(linalgOp);
}

} // namespace structured_match
} // namespace imex
