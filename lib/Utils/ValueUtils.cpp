//===-- ValueUtils.cpp - Zero-checking utilities ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

namespace imex {

// Returns true if the value is a constant float or integer.
bool isValConstZero(mlir::Value val) {
  return mlir::matchPattern(val, mlir::m_AnyZeroFloat()) ||
         mlir::matchPattern(val, mlir::m_Zero());
}

// Returns true if the attribute represent "all zeros"
static bool isZeroAttr(mlir::Attribute attribute) {
  return mlir::TypeSwitch<mlir::Attribute, bool>(attribute)
      .Case<mlir::FloatAttr>(
          [](auto attr) { return attr.getValueAsDouble() == 0.0; })
      .Case<mlir::IntegerAttr>([](auto attr) { return attr.getInt() == 0; })
      .Case<mlir::DenseElementsAttr>([](auto attr) {
        if (!attr.getElementType().isIntOrFloat())
          return false;
        if (!attr.isSplat())
          return false;
        auto splat = attr.template getSplatValue<mlir::Attribute>();
        return isZeroAttr(splat);
      })
      .Default([](auto attr) { return false; });
}

// Prototypes
static bool isZeroOp(mlir::Operation *);

// Returns true if the value represents a zero filled tensor.
// Recurse into isZeroOp for defining ops if not immediately obvious
// Looks past linalg generic's argument (which don't have defining ops)
bool isZeroTensor(mlir::Value val) {
  if (!val)
    return false;
  if (isValConstZero(val))
    return true;

  mlir::Operation *defOp = nullptr;

  // Block arguments don't have a defining op, but they do have an op arg
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    // We need to find the argument to the linalg on the same order as this one
    auto *linalgOp = arg.getParentRegion()->getParentOp();
    if (!mlir::isa<mlir::linalg::GenericOp>(linalgOp))
      return false;
    auto index = arg.getArgNumber();
    auto linalgArg = linalgOp->getOperand(index);
    defOp = linalgArg.getDefiningOp();
  } else {
    defOp = val.getDefiningOp();
  }
  return isZeroOp(defOp);
}

// Returns true if the operation represents a zero filled tensor
// Recurses into isZeroTensor for operands and isZeroAttr for attributes
static bool isZeroOp(mlir::Operation *defOp) {
  if (!defOp)
    return false;

  return mlir::TypeSwitch<mlir::Operation *, bool>(defOp)
      .Case<mlir::arith::ConstantOp>([&](auto op) {
        // Dense attributes don't match APFloat.isZero()
        auto attr = op.getValue();
        return isZeroAttr(attr);
      })
      .Case<mlir::linalg::FillOp, mlir::linalg::CopyOp>([&](auto op) {
        if (op.getInputs().size() != 1)
          return false;
        return isZeroTensor(op.getInputs()[0]);
      })
      .Case<mlir::memref::CopyOp, mlir::memref::SubViewOp, mlir::tensor::CastOp,
            mlir::tensor::ExtractSliceOp>(
          [&](auto op) { return isZeroTensor(op.getSource()); })
      .Case<mlir::memref::GetGlobalOp>([&](auto op) {
        auto name = op.getName();
        auto module = defOp->getParentOfType<mlir::ModuleOp>();
        auto global = module.lookupSymbol<mlir::memref::GlobalOp>(name);
        auto attr = global.getInitialValueAttr();
        return isZeroAttr(attr);
      })
      .Default([&](mlir::Operation *op) { return false; });
}

mlir::FailureOr<mlir::SmallVector<int64_t>> getStrides(mlir::Value value) {
  auto valueType = value.getType();
  if (!mlir::isa<mlir::MemRefType>(valueType))
    return mlir::failure();
  auto memrefType = mlir::cast<mlir::MemRefType>(valueType);
  mlir::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(memrefType.getStridesAndOffset(strides, offset)))
    return mlir::failure();
  return strides;
}

mlir::FailureOr<mlir::SmallVector<int64_t>>
getStaticStrides(mlir::Value value) {
  auto strides = getStrides(value);
  if (mlir::failed(strides))
    return mlir::failure();
  if (llvm::any_of(*strides, [](int64_t stride) {
        return stride == mlir::ShapedType::kDynamic;
      }))
    return mlir::failure();
  return strides;
}

std::pair<mlir::Value, mlir::Value> getPtrAndOffset(mlir::OpBuilder &builder,
                                                    mlir::Value operand) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
  assert(memrefType && "Expect a memref value");

  mlir::Location loc = operand.getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert right after operand producer for better opt chances.
  builder.setInsertionPointAfterValue(operand);

  mlir::MemRefType baseMemrefType =
      mlir::MemRefType::get({}, memrefType.getElementType());
  mlir::Type basePtrType = builder.getIndexType();
  mlir::Type offsetType = builder.getIndexType();
  mlir::SmallVector<mlir::Type> sizesTypes(memrefType.getRank(), offsetType);
  mlir::SmallVector<mlir::Type> stridesTypes(memrefType.getRank(), offsetType);
  auto meta = builder.create<mlir::memref::ExtractStridedMetadataOp>(
      loc, baseMemrefType, offsetType, sizesTypes, stridesTypes, operand);
  mlir::Value alignedPointerAsIndex =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
          loc, basePtrType, operand);
  mlir::Value alignedPointerAsI64 = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getIntegerType(64), alignedPointerAsIndex);
  // TODO: non-POD will require an LLVMTypeConverter.
  mlir::Value alignedPointer = builder.create<mlir::LLVM::IntToPtrOp>(
      loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
      alignedPointerAsI64);
  mlir::Value offset = meta.getOffset();
  return std::make_pair(alignedPointer, offset);
}

mlir::Value flattenMemref(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value srcMemref) {
  auto srcType = mlir::cast<mlir::MemRefType>(srcMemref.getType());

  assert(srcType && "Expected a memref type");

  auto shapeNd = srcType.getShape();
  int64_t flatSize =
      std::accumulate(shapeNd.begin(), shapeNd.end(), 1, std::multiplies<>());

  mlir::Value offset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value size =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, flatSize);
  mlir::Value stride = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

  // Use memref.reinterpret_cast to flatten the memref
  auto flatMemRefType = mlir::MemRefType::get(
      {flatSize}, srcType.getElementType(), nullptr, srcType.getMemorySpace());
  auto flatMemref =
      rewriter
          .create<mlir::memref::ReinterpretCastOp>(
              loc, flatMemRefType, srcMemref, offset, size, stride)
          .getResult();
  return flatMemref;
}

bool hasSharedMemSpace(mlir::Value memref) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!type)
    return false;

  auto memSpace = type.getMemorySpace();
  if (!memSpace)
    return false;

  if (auto gpuAttr = mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(memSpace))
    return gpuAttr.getValue() == mlir::gpu::AddressSpace::Private;

  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(memSpace))
    return intAttr.getValue() ==
           static_cast<int64_t>(mlir::gpu::AddressSpace::Private);

  return false;
}

void computeSubviewOffsets(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Value memref,
                           mlir::SmallVector<mlir::Value> &resultOffsets,
                           mlir::Value &resultRootMemref) {
  auto fillVal = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  assert(type && "Expected a memref type");

  auto origShape = type.getShape();

  resultOffsets.clear();
  resultOffsets.append(origShape.size(), fillVal);
  resultRootMemref = memref;

  while (auto subViewOp =
             resultRootMemref.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto currentOffsets = getAsOpFoldResult(resultOffsets);
    resultOffsets.clear();

    mlir::affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, resultRootMemref.getLoc(), subViewOp.getMixedOffsets(),
        subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), currentOffsets,
        resultOffsets);
    resultRootMemref = subViewOp.getOperand(0);
  }
}

mlir::SmallVector<mlir::OpFoldResult>
getMemrefStrides(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value memref) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());

  auto stridedLayout =
      mlir::dyn_cast<mlir::StridedLayoutAttr>(type.getLayout());
  if (stridedLayout) {
    auto strides = stridedLayout.getStrides();
    return getMixedValues(strides, {}, rewriter);
  }

  auto sizes = getMixedValues(type.getShape(), {}, rewriter);
  auto strides = mlir::memref::computeStridesIRBlock(loc, rewriter, sizes);
  return strides;
}

mlir::FailureOr<mlir::Value> reduceMemrefDims(mlir::PatternRewriter &rewriter,
                                              mlir::Location loc,
                                              mlir::Value memref,
                                              size_t maxDims) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  auto shape = type.getShape();

  if (shape.size() <= maxDims)
    return memref;

  for (size_t i = 0; i < shape.size() - maxDims; i++)
    if (shape[i] != 1)
      return mlir::failure();

  auto offsets =
      getMixedValues(mlir::SmallVector<int64_t>(shape.size(), 0), {}, rewriter);
  auto sizes = getMixedValues(shape, {}, rewriter);
  auto staticStrides = getStaticStrides(memref).value();
  auto strides =
      getMixedValues(mlir::SmallVector<int64_t>(shape.size(), 1), {}, rewriter);

  mlir::SmallVector<int64_t> newShape(shape.begin() + shape.size() - maxDims,
                                      shape.end());
  mlir::SmallVector<int64_t> newStrides(
      staticStrides.begin() + shape.size() - maxDims, staticStrides.end());

  int64_t newOffset = 0;
  if (auto memrefLayout =
          mlir::dyn_cast<mlir::StridedLayoutAttr>(type.getLayout()))
    newOffset = memrefLayout.getOffset();

  auto newLayout = mlir::StridedLayoutAttr::get(
      rewriter.getContext(), /*offset=*/newOffset, /*strides=*/newStrides);
  mlir::MemRefType newMemRefType = mlir::MemRefType::get(
      newShape, type.getElementType(), newLayout, type.getMemorySpace());

  auto squeezedSubview =
      rewriter
          .create<mlir::memref::SubViewOp>(loc, newMemRefType, memref, offsets,
                                           sizes, strides)
          .getResult();
  return squeezedSubview;
}

mlir::LogicalResult maybeSqueezeDims(mlir::PatternRewriter &rewriter,
                                     mlir::linalg::LinalgOp linalgOp,
                                     size_t maxDims) {
  mlir::SmallVector<std::pair<size_t, mlir::Value>> newOperands;
  auto operands = linalgOp->getOperands();
  auto loc = linalgOp.getLoc();

  for (size_t i = 0; i < operands.size(); i++) {
    auto operand = operands[i];
    auto type = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
    if (!type) {
      // Skip non-memref operands
      continue;
    }

    if (type.getShape().size() <= maxDims)
      continue;

    auto res = reduceMemrefDims(rewriter, loc, operand, maxDims);
    if (mlir::failed(res)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Can't squeeze memref to the desired number of dimensions");
    }

    auto flatSubview = res.value();
    newOperands.emplace_back(i, flatSubview);
  }

  if (newOperands.empty())
    return mlir::success();

  rewriter.modifyOpInPlace(linalgOp, [&] {
    for (auto [i, operand] : newOperands)
      linalgOp->setOperand(i, operand);
  });
  return mlir::success();
}

bool canSqueezeDims(llvm::ArrayRef<int64_t> shape, size_t maxDims) {
  if (shape.size() <= maxDims)
    return true;

  for (size_t i = 0; i < shape.size() - maxDims; i++)
    if (shape[i] != 1)
      return false;

  return true;
}

} // namespace imex
