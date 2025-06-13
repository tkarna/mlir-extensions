// FIXME add header

#include "imex/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h"
#include "imex/Utils/PassUtils.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <numeric>

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "xegpu-hoist-desc"

using namespace mlir;

class XeGPUTransformOps
    : public transform::TransformDialectExtension<XeGPUTransformOps> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XeGPUTransformOps)

  using Base::Base;

  void init();
};

void XeGPUTransformOps::init() {
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<gpu::GPUDialect>();
  declareGeneratedDialect<xegpu::XeGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include <imex/Dialect/XeGPU/TransformOps/XeGPUTransformOps.cpp.inc>
      >();
}

#define GET_OP_CLASSES
#include <imex/Dialect/XeGPU/TransformOps/XeGPUTransformOps.cpp.inc>

// Return vector type with specified VNNI shape.
static VectorType getVnniVector(ArrayRef<int64_t> shape, Type elementType,
                                int64_t vnniFactor = 2, int64_t vnniAxis = 0) {
  assert(shape.size() == 2 && "Expected plain 2D shape");
  SmallVector<int64_t> vecShape{shape};
  vecShape[vnniAxis] /= vnniFactor;
  vecShape.push_back(vnniFactor);
  return VectorType::get(vecShape, elementType);
}

// Folds defining memref.SubviewOp into the xegpu.CreateNDDescOp
// Only considers subview ops in the same region.
std::optional<xegpu::CreateNdDescOp>
foldSubview(transform::TransformRewriter &rewriter,
            xegpu::CreateNdDescOp descOp) {
  auto subViewOp = descOp.getSource().getDefiningOp<memref::SubViewOp>();
  if (!subViewOp) {
    LLVM_DEBUG(llvm::dbgs() << "No defining subview op.\n");
    return std::nullopt;
  }
  if (!subViewOp.hasUnitStride()) {
    LLVM_DEBUG(llvm::dbgs() << "Defining subview not unit stride.\n");
    return std::nullopt;
  }
  // restrict folding to the same region
  if (subViewOp.getOperation()->getParentRegion() !=
      descOp.getOperation()->getParentRegion()) {
    LLVM_DEBUG(llvm::dbgs() << "Defining subview in different region.\n");
    return std::nullopt;
  }

  rewriter.setInsertionPointAfter(descOp);

  SmallVector<Value> resolvedOffsets;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, descOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(),
      descOp.getMixedOffsets(), resolvedOffsets);

  auto newOp = rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descOp.getTensorDesc().getType(), subViewOp.getSource(),
      getAsOpFoldResult(resolvedOffsets));

  return cast<xegpu::CreateNdDescOp>(newOp);
}

xegpu::CreateNdDescOp
foldSubviewIntoDescOp(transform::TransformRewriter &rewriter,
                      xegpu::CreateNdDescOp descOp) {
  auto newOp = descOp;
  while (true) {
    auto maybeNewOp = foldSubview(rewriter, newOp);
    if (!maybeNewOp)
      break;
    newOp = *maybeNewOp;
  }
  return newOp;
}

// Recurse operands and collect all producer ops in the given region.
void collectProducerOps(Operation *op, Region &inRegion,
                        SmallVector<Operation *> &ops) {
  for (auto val : op->getOperands()) {
    if (const auto definingOp = val.getDefiningOp();
        definingOp && definingOp->getParentRegion() == &inRegion) {
      ops.push_back(definingOp);
      collectProducerOps(definingOp, inRegion, ops);
    }
  }
}

// Returns all producer ops in the given region
SmallVector<Operation *> getProducerOpsInRegion(Operation *op, Region &inRegion,
                                                bool includeOp = true) {
  SmallVector<Operation *> producerOps;
  if (includeOp) {
    producerOps.push_back(op);
  }
  collectProducerOps(op, inRegion, producerOps);
  return producerOps;
}

// Get user of type T in immediate users of the value.
template <typename T> static std::optional<T> getUserOfType(Value value) {
  auto users = value.getUsers();
  auto it = llvm::find_if(users, [&](Operation *op) { return isa<T>(op); });
  if (it != users.end()) {
    return cast<T>(*it);
  }
  return std::nullopt;
}

// Get defining op ot the given type
template <typename T> static std::optional<T> getDefiningOpOfType(Value value) {
  if (auto op = value.getDefiningOp()) {
    if (auto castedOp = dyn_cast<T>(op)) {
      return castedOp;
    }
  }
  return std::nullopt;
}

// Follow user chain in region and find the first user of type T.
template <typename T>
static std::optional<T> findUserInRegion(Value value, Region &region) {
  for (auto user : value.getUsers()) {
    if (user->getParentRegion() != &region) {
      continue; // skip users outside the region
    }
    if (auto op = dyn_cast<T>(user)) {
      return op;
    } else {
      for (auto val : user->getResults()) {
        if (auto op = findUserInRegion<T>(val, region)) {
          return op;
        }
      }
    }
  }
  return std::nullopt;
}

// Add offset update op after create desc op if tile is updated in the loop.
xegpu::CreateNdDescOp insertUpdateOp(transform::TransformRewriter &rewriter,
                                     scf::ForOp parentLoopOp,
                                     xegpu::CreateNdDescOp descOp) {

  // Clone producers and replace loop induction variable with lower bound
  rewriter.setInsertionPointAfter(descOp);
  auto loc = descOp.getLoc();
  IRMapping mapping;
  SmallVector<Operation *> clonedOps;
  auto producers = getProducerOpsInRegion(descOp.getOperation(),
                                          parentLoopOp.getRegion(), true);
  for (auto &op : llvm::reverse(producers)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cloning producer op: " << op->getName() << "\n");
    auto newOp = rewriter.clone(*op, mapping);
    clonedOps.push_back(newOp);
  }
  // Descriptor op offset should be a constant defined by loop lower bound
  rewriter.replaceUsesWithIf(parentLoopOp.getInductionVar(),
                             parentLoopOp.getLowerBound(), [&](OpOperand &use) {
                               return ::llvm::is_contained(clonedOps,
                                                           use.getOwner());
                             });
  auto newDescOp = cast<xegpu::CreateNdDescOp>(clonedOps.back());

  // Compute offset for update operation: original offset - constant offset
  llvm::SmallVector<Value> origDynamicOffsets, constDynamicOffsets,
      dynamicOffsets;
  llvm::SmallVector<int64_t> origStaticOffsets, constStaticOffsets,
      staticOffsets;
  dispatchIndexOpFoldResults(descOp.getMixedOffsets(), origDynamicOffsets,
                             origStaticOffsets);
  dispatchIndexOpFoldResults(newDescOp.getMixedOffsets(), constDynamicOffsets,
                             constStaticOffsets);
  // Deduce correct offsets for update offset op
  int64_t dynIndex = 0;
  for (auto [i, origStaticOffset] : llvm::enumerate(origStaticOffsets)) {
    if (origStaticOffset == ShapedType::kDynamic) {
      auto origDynOffset = origDynamicOffsets[dynIndex];
      auto cstDynOffset = constDynamicOffsets[dynIndex];
      if (true) { // FIXME check if this operand depends on loop variable
        auto subOp = rewriter.create<arith::SubIOp>(
            loc, origDynOffset.getType(), origDynOffset, cstDynOffset);
        dynamicOffsets.push_back(subOp.getResult());
        staticOffsets.push_back(ShapedType::kDynamic);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "Dynamic offset does not depend on induction var, "
                      "skipping "
                   << i << "\n");
        staticOffsets.push_back(0);
      }
      dynIndex++;
    } else {
      staticOffsets.push_back(0);
    }
  }

  // add an offset update op after the create desc op
  if (!dynamicOffsets.empty()) {
    auto tile = newDescOp.getResult();
    auto offsetOp = rewriter.create<xegpu::UpdateNdOffsetOp>(
        loc, tile.getType(), tile, dynamicOffsets, staticOffsets);
    // replace subsequent uses of the descriptor with the offset descriptor
    rewriter.replaceUsesWithIf(
        descOp.getResult(), offsetOp.getResult(), [&](OpOperand &use) {
          return use.getOwner() != offsetOp.getOperation();
        });
  }
  rewriter.replaceOp(descOp, newDescOp);
  return newDescOp;
}

LogicalResult insertOffsetUpdateOps(transform::TransformRewriter &rewriter,
                                    scf::ForOp loopOp) {
  // Find all create desc operations in the loop body
  SmallVector<Operation *> createDescOps;
  for (auto &op : loopOp.getBody()->getOperations()) {
    if (isa<xegpu::CreateNdDescOp>(op)) {
      createDescOps.push_back(&op);
    }
  }
  if (createDescOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No xegpu.create_nd_desc ops found in the loop body");
    return failure();
  }
  // canonicalize
  for (auto &op : createDescOps) {
    auto descOp = cast<xegpu::CreateNdDescOp>(op);
    descOp = foldSubviewIntoDescOp(rewriter, descOp);
    insertUpdateOp(rewriter, loopOp, descOp);
  }
  return success();
}

// Hoist create desc ops out of the loop.
// If offset update ops exist, add values to loop iter_args and yield
FailureOr<scf::ForOp> hoistDescOps(transform::TransformRewriter &rewriter,
                                   scf::ForOp loopOp) {
  SmallVector<xegpu::CreateNdDescOp> descOps;
  for (auto &op : loopOp.getBody()->getOperations()) {
    if (auto descOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
      // Assume that desc ops can be hoisted
      descOps.push_back(descOp);
    }
  }
  if (descOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No create desc ops found, returning.\n");
    return loopOp;
  }

  SmallVector<Value> initValues, yieldValues;
  for (auto &descOp : descOps) {
    // hoist desc op
    LLVM_DEBUG(llvm::dbgs() << "Hoisting desc op: " << descOp.getLoc() << "\n");
    auto producers =
        getProducerOpsInRegion(descOp.getOperation(), loopOp.getRegion(), true);
    for (auto &op : llvm::reverse(producers)) {
      rewriter.moveOpBefore(op, loopOp);
    }

    // Find the correct loop init and yield values
    auto maybeOffsetOp =
        getUserOfType<xegpu::UpdateNdOffsetOp>(descOp.getResult());
    if (!maybeOffsetOp) {
      // skip if no offset update op
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "  Found offset update op\n");
    // replace offset update with loop step size
    auto offsetOp = *maybeOffsetOp;
    auto offsetProducerOps =
        getProducerOpsInRegion(offsetOp.getOperation(), loopOp.getRegion());
    rewriter.replaceUsesWithIf(
        loopOp.getInductionVar(), loopOp.getStep(), [&](OpOperand &use) {
          return llvm::is_contained(offsetProducerOps, use.getOwner());
        });
    // offset now points to next tile, desc users must use current tile
    rewriter.replaceAllUsesWith(offsetOp.getResult(), offsetOp.getTensorDesc());
    initValues.push_back(descOp.getResult());
    yieldValues.push_back(offsetOp.getResult());
  }
  // rewrite loop with new init/yield values
  NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                 llvm::ArrayRef<BlockArgument> newBBArgs) {
    return yieldValues;
  };
  auto maybeNewLoop = loopOp.replaceWithAdditionalYields(
      rewriter, initValues,
      /*replaceInitOperandUsesInLoop=*/true, yieldFn);
  if (failed(maybeNewLoop)) {
    LLVM_DEBUG(llvm::dbgs() << "Creating new loop failed\n");
    return failure();
  }
  return cast<scf::ForOp>(*maybeNewLoop);
}

// Hoist loop independent load/store ops out of the loop.
//
// Detect load/update/store patterns that act on a loop-invariant tile and
// moves the load/store ops before/after the loop. If there are multiple such
// patterns acting on the same tile, chains the update ops correctly inside the
// loop.
FailureOr<scf::ForOp> hoistLoadStoreOps(transform::TransformRewriter &rewriter,
                                        scf::ForOp loopOp) {
  SmallVector<Operation *> opsToRemove;
  SmallVector<Operation *> opsToHoist;
  llvm::DenseMap<Value, int64_t> tileToYieldIndexMap;
  llvm::DenseMap<Value, Value> tileToInitValueMap;
  llvm::DenseMap<Value, Value> tileToYieldValueMap;
  int64_t nbYields = loopOp.getNumResults();
  for (auto &op : loopOp.getBody()->getOperations()) {
    // find loop-invariant load and store ops
    auto loadOp = dyn_cast<xegpu::LoadNdOp>(op);
    if (!loadOp) {
      continue;
    }
    auto tile = loadOp.getTensorDesc();
    auto defOp = tile.getDefiningOp();
    if (!defOp || defOp->getParentRegion() == &loopOp.getRegion()) {
      LLVM_DEBUG(llvm::dbgs() << "Load op tile defined inside loop, skipping. "
                              << loadOp.getLoc() << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "Found load op: " << loadOp.getLoc() << "\n");
    auto maybeStoreOp = findUserInRegion<xegpu::StoreNdOp>(loadOp.getResult(),
                                                           loopOp.getRegion());
    if (!maybeStoreOp) {
      LLVM_DEBUG(llvm::dbgs() << "No store op found for load op, skipping.\n");
      continue;
    }
    auto storeOp = *maybeStoreOp;
    if (storeOp.getTensorDesc() != tile) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Store op destination tile differs, skipping.\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "Found store op: " << storeOp.getLoc() << "\n");
    // define loop init and yield values
    Value loadedVect = loadOp.getResult();
    Value yieldValue = storeOp.getValue();
    if (!tileToYieldIndexMap.contains(tile)) {
      // case 1: tile has not been seen yet
      // hoist load/cast op and its producers
      auto maybeCastOp = getUserOfType<arith::ExtFOp>(loadOp.getValue());
      if (maybeCastOp) {
        // if the load op is a cast, hoist the cast as well
        opsToHoist.push_back((*maybeCastOp).getOperation());
        loadedVect = (*maybeCastOp).getResult();
      } else {
        opsToHoist.push_back(loadOp.getOperation());
      }
      tileToInitValueMap[tile] = loadedVect;
      int64_t yieldIndex = nbYields + tileToInitValueMap.size() - 1;
      tileToYieldIndexMap[tile] = yieldIndex;
    } else {
      // case 2: tile has been seen before
      // update the producer-consumer chain
      auto maybeCastOp = getUserOfType<arith::ExtFOp>(loadedVect);
      if (maybeCastOp) {
        loadedVect = (*maybeCastOp).getResult();
        opsToRemove.push_back(maybeCastOp->getOperation());
      }
      auto prevYieldValue = tileToYieldValueMap[tile];
      rewriter.replaceAllUsesWith(loadedVect, prevYieldValue);
      opsToRemove.push_back(loadOp.getOperation());
    }
    // mark store op and its cast as ops to remove
    opsToRemove.push_back(storeOp.getOperation());
    auto maybeReCastOp =
        getDefiningOpOfType<arith::TruncFOp>(storeOp.getValue());
    if (maybeReCastOp) {
      yieldValue = (*maybeReCastOp).getOperand();
      opsToRemove.push_back(maybeReCastOp->getOperation());
    }
    // update yield value
    tileToYieldValueMap[tile] = yieldValue;
  }

  if (opsToHoist.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No load/store ops to hoist, returning.\n");
    return loopOp;
  }

  // hoist producer ops before the loop
  for (auto &op : llvm::reverse(opsToHoist)) {
    auto producers = getProducerOpsInRegion(op, loopOp.getRegion(), true);
    for (auto &prod : llvm::reverse(producers)) {
      rewriter.moveOpBefore(prod, loopOp);
    }
  }

  // rewrite loop with new init/yield values
  SmallVector<Value> initValues, yieldValues;
  for (auto &[tile, initValue] : tileToInitValueMap) {
    auto yieldValue = tileToYieldValueMap[tile]; // TODO check if exists
    initValues.push_back(initValue);
    yieldValues.push_back(yieldValue);
  }
  NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                 llvm::ArrayRef<BlockArgument> newBBArgs) {
    return yieldValues;
  };
  auto maybeNewLoop = loopOp.replaceWithAdditionalYields(
      rewriter, initValues,
      /*replaceInitOperandUsesInLoop=*/true, yieldFn);
  if (failed(maybeNewLoop)) {
    LLVM_DEBUG(llvm::dbgs() << "Creating new loop failed\n");
    return failure();
  }
  auto newLoopOp = cast<scf::ForOp>(*maybeNewLoop);

  // create store ops after the loop
  rewriter.setInsertionPointAfter(newLoopOp);
  auto ctx = rewriter.getContext();
  auto writeCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::WRITE_BACK);
  for (auto &[tile, yieldIndex] : tileToYieldIndexMap) {
    auto tileElemType = cast<ShapedType>(tile.getType()).getElementType();
    Value storeValue = newLoopOp.getResult(yieldIndex);
    if (cast<ShapedType>(storeValue.getType()).getElementType() !=
        tileElemType) {
      auto dstType = VectorType::get(
          cast<ShapedType>(storeValue.getType()).getShape(), tileElemType);
      auto reCastOp = rewriter.create<arith::TruncFOp>(newLoopOp.getLoc(),
                                                       dstType, storeValue);
      storeValue = reCastOp.getResult();
    }
    rewriter.create<xegpu::StoreNdOp>(newLoopOp.getLoc(), storeValue, tile,
                                      writeCacheHint, writeCacheHint,
                                      writeCacheHint);
  }
  // remove deprecated ops
  for (auto &op : opsToRemove) {
    rewriter.eraseOp(op);
  }
  return newLoopOp;
}

void foldRedundantLoadOps(transform::TransformRewriter &rewriter,
                          scf::ForOp loopOp) {
  llvm::DenseMap<Value, SmallVector<Operation *>> tileToLoadOps;
  for (auto &op : loopOp.getBody()->getOperations()) {
    if (auto loadOp = dyn_cast<xegpu::LoadNdOp>(op)) {
      if (!loadOp->hasOneUse()) {
        continue; // assume that load is only used once in dpas op
      }
      // FIXME more generic memory effect check
      auto maybeDpasOp = findUserInRegion<xegpu::DpasOp>(loadOp.getResult(),
                                                         loopOp.getRegion());
      if (maybeDpasOp) {
        auto dpasOp = *maybeDpasOp;
        if (dpasOp.getAcc() == loadOp.getResult()) {
          continue; // used as DPAS accumulator, write effect, do not fold
        }
      }

      auto tile = loadOp.getTensorDesc();
      tileToLoadOps[tile].push_back(&op);
    }
  }
  for (auto &[tile, loadOps] : tileToLoadOps) {
    if (loadOps.size() < 2) {
      continue; // nothing to collapse
    }
    bool firstOp = true;
    for (auto &loadOp : loadOps) {
      if (firstOp) {
        // keep the first op
        firstOp = false;
        continue;
      }
      // replace all uses of the tile with the first load op result
      rewriter.replaceAllUsesWith(loadOp->getResult(0),
                                  loadOps.front()->getResult(0));
      rewriter.eraseOp(loadOp);
    }
  }
}

void castDpasAccumulatorType(transform::TransformRewriter &rewriter,
                             scf::ForOp loopOp) {
  for (auto &op : loopOp.getBody()->getOperations()) {
    // FIXME convert this to a helper function
    // find loop-invariant load and store ops
    auto loadOp = dyn_cast<xegpu::LoadNdOp>(op);
    if (!loadOp) {
      continue;
    }
    auto tile = loadOp.getTensorDesc();
    auto loadedVect = loadOp.getResult();
    auto defOp = tile.getDefiningOp();
    if (!defOp || defOp->getParentRegion() == &loopOp.getRegion()) {
      LLVM_DEBUG(llvm::dbgs() << "Load op tile defined inside loop, skipping. "
                              << loadOp.getLoc() << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "Found load op: " << loadOp.getLoc() << "\n");
    auto maybeStoreOp = findUserInRegion<xegpu::StoreNdOp>(loadOp.getResult(),
                                                           loopOp.getRegion());
    if (!maybeStoreOp) {
      LLVM_DEBUG(llvm::dbgs() << "No store op found for load op, skipping.\n");
      continue;
    }
    auto storeOp = *maybeStoreOp;
    if (storeOp.getTensorDesc() != tile) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Store op destination tile differs, skipping.\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "Found store op: " << storeOp.getLoc() << "\n");
    // Dpas specific
    auto maybeDpasOp = getUserOfType<xegpu::DpasOp>(loadOp.getValue());
    if (!maybeDpasOp) {
      LLVM_DEBUG(llvm::dbgs() << "No dpas op found for load op, skipping.\n");
      continue;
    }
    // TODO check that the vector is the accumulator operand
    auto dpasOp = *maybeDpasOp;
    rewriter.setInsertionPointAfter(loadOp);
    // DPAS only works with f32 accumulators
    auto cTileShape = cast<ShapedType>(loadOp.getResult().getType());
    auto dpasResType =
        VectorType::get(cTileShape.getShape(), rewriter.getF32Type());
    // cast C tile to f32
    auto castedLoad = rewriter.create<arith::ExtFOp>(loadOp.getLoc(),
                                                     dpasResType, loadedVect);
    // replace dpas op
    rewriter.setInsertionPointAfter(dpasOp);
    auto newDpasOp = rewriter.create<xegpu::DpasOp>(
        dpasOp.getLoc(), dpasResType,
        ValueRange{dpasOp.getLhs(), dpasOp.getRhs(), castedLoad});
    // add a reverse cast to the dpas op result
    auto castedDpasRes = rewriter.create<arith::TruncFOp>(
        dpasOp.getLoc(), loadOp.getResult().getType(), newDpasOp.getResult());
    storeOp.getValueMutable().assign(castedDpasRes);
    rewriter.replaceOp(dpasOp, newDpasOp);
  }
}

LogicalResult insertThreadSync(transform::TransformRewriter &rewriter,
                               scf::ForOp loopOp) {
  rewriter.setInsertionPointToStart(loopOp.getBody());
  auto maybeKDimSize = getConstantIntValue(loopOp.getUpperBound());
  if (!maybeKDimSize) {
    LLVM_DEBUG(llvm::dbgs() << "K loop upper bound not a constant\n");
    return failure();
  }
  int kDimSize = *maybeKDimSize;
  auto loc = loopOp.getLoc();
  // FIXME compute sync step based on tile size.
  int syncFreq = 4;
  int maxSyncStep = 1024;
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  int syncStep =
      std::min(std::max(kDimSize / syncFreq, maxSyncStep), maxSyncStep);
  auto syncStepConst = rewriter.create<arith::ConstantIndexOp>(loc, syncStep);
  auto loopStepMod = rewriter.create<arith::RemUIOp>(
      loc, loopOp.getInductionVar(), syncStepConst);
  auto syncBlockCond = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, loopStepMod, zero);
  rewriter.create<scf::IfOp>(
      loc, syncBlockCond,
      /*thenBuilder=*/
      [](OpBuilder &b, Location loc) {
        b.create<gpu::BarrierOp>(loc);
        b.create<scf::YieldOp>(loc);
      },
      /*elseBuilder=*/nullptr);
  return success();
}

DiagnosedSilenceableFailure transform::XeGPUHoistDescOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto loopOp = dyn_cast<scf::ForOp>(target);
  if (!loopOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a scf.for op, but got: " << target->getName();
  }

  if (failed(insertOffsetUpdateOps(rewriter, loopOp))) {
    return emitSilenceableFailure(getLoc())
           << "No desc ops found in the loop body " << target->getName();
  }
  auto newLoopOp = hoistDescOps(rewriter, loopOp);
  if (failed(newLoopOp)) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Failed to hoist xegpu.create_nd_desc ops";
    diag.attachNote(loopOp.getLoc()) << "loop op";
    return diag;
  }
  loopOp = *newLoopOp;
  foldRedundantLoadOps(rewriter, *newLoopOp);
  results.push_back(loopOp.getOperation());
  return DiagnosedSilenceableFailure::success();
}

void transform::XeGPUHoistDescOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLoopMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform::XeGPUHoistLoadStoreOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto loopOp = dyn_cast<scf::ForOp>(target);
  if (!loopOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a scf.for op, but got: " << target->getName();
  }

  castDpasAccumulatorType(rewriter, loopOp);
  auto newLoopOp = hoistLoadStoreOps(rewriter, loopOp);
  if (failed(newLoopOp)) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Failed to hoist load/store ops";
    diag.attachNote(loopOp.getLoc()) << "loop op";
    return diag;
  }
  foldRedundantLoadOps(rewriter, *newLoopOp);
  loopOp = *newLoopOp;
  if (failed(insertThreadSync(rewriter, loopOp))) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Failed to add thread sync ops";
    diag.attachNote(loopOp.getLoc()) << "loop op";
    return diag;
  }
  results.push_back(loopOp.getOperation());
  return DiagnosedSilenceableFailure::success();
}

void transform::XeGPUHoistLoadStoreOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLoopMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

std::optional<Value> getIthSubtile(transform::TransformRewriter &rewriter,
                                   Value &source, Value &index,
                                   Value &upperBound,
                                   llvm::ArrayRef<int64_t> &tileSize) {
  auto defOp = source.getDefiningOp();
  if (!defOp) {
    LLVM_DEBUG(llvm::dbgs() << "No defining op.\n");
    return std::nullopt;
  }
  rewriter.setInsertionPointAfter(defOp);
  auto loc = defOp->getLoc();

  llvm::ArrayRef<int64_t> srcShape =
      cast<ShapedType>(source.getType()).getShape();
  if (ShapedType::isDynamicShape(srcShape)) {
    LLVM_DEBUG(llvm::dbgs() << "Expecting memref with static shape.\n");
    return std::nullopt;
  }
  if (srcShape[0] % tileSize[0] != 0 || srcShape[1] % tileSize[1] != 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "Source shape is not divisible by tile tileSize.\n");
    return std::nullopt;
  }
  SmallVector<int64_t, 2> grid{srcShape[0] / tileSize[0],
                               srcShape[1] / tileSize[1]};
  auto nGrid = grid[0] * grid[1];
  auto maybeUpperBound = getConstantIntValue(upperBound);
  if (!maybeUpperBound) {
    LLVM_DEBUG(llvm::dbgs() << "Upper bound is not a constant.\n");
    return std::nullopt;
  }
  if (*maybeUpperBound != nGrid) {
    LLVM_DEBUG(llvm::dbgs()
               << "Loop iteration count is not equal to number of subtiles: "
               << *maybeUpperBound << " != " << nGrid << "\n");
    return std::nullopt;
  }
  // linear to 2d tile index
  auto nColTiles =
      rewriter.create<arith::ConstantIndexOp>(loc, grid[1]).getResult();
  auto rowIndex = rewriter.create<arith::DivSIOp>(loc, index, nColTiles);
  auto colIndex = rewriter.create<arith::RemSIOp>(loc, index, nColTiles);
  // calculate tile offset
  auto tileRowsCst =
      rewriter.create<arith::ConstantIndexOp>(loc, tileSize[0]).getResult();
  auto tileColsCst =
      rewriter.create<arith::ConstantIndexOp>(loc, tileSize[1]).getResult();
  auto row_offset = rewriter.create<arith::MulIOp>(loc, tileRowsCst, rowIndex);
  auto col_offset = rewriter.create<arith::MulIOp>(loc, tileColsCst, colIndex);
  // create a subview with the calculated offset and size
  auto offsets = getMixedValues({ShapedType::kDynamic, ShapedType::kDynamic},
                                {row_offset, col_offset}, rewriter);
  auto sizes = getMixedValues({tileSize[0], tileSize[1]}, {}, rewriter);
  auto strides = getMixedValues({1, 1}, {}, rewriter);
  auto subview =
      rewriter.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
  return subview.getResult();
}

DiagnosedSilenceableFailure transform::XeGPUInsertPrefetchOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto matmulOp = dyn_cast<linalg::MatmulOp>(target);
  if (!matmulOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a linalg.matmul op, but got: " << target->getName();
  }

  auto loopOp = matmulOp->getParentOfType<scf::ForOp>();
  if (!loopOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a scf.for op as parent of the matmul op";
    diag.attachNote(loopOp.getLoc()) << "parent op";
    return diag;
  }

  // defines which matmul operand to prefetch
  int64_t tileIndex = getTileIndex();
  if (tileIndex < 0 || tileIndex >= 2) {
    return emitSilenceableFailure(getLoc())
           << "Invalid tile index: " << tileIndex
           << ", expected 0 or 1 for A or B operand";
  }
  // prefetch tile size
  llvm::ArrayRef<int64_t> tileSize = getTileSize();
  if (tileSize.size() != 2) {
    return emitSilenceableFailure(getLoc()) << "Expected 2d tile size";
  }
  // clone k loop with only A tile subview op
  rewriter.setInsertionPoint(loopOp);
  auto cloned = rewriter.clone(*loopOp.getOperation());
  auto newLoopOp = cast<scf::ForOp>(cloned);

  // get cloned matmul and its operand tile
  auto maybeMatmulOp =
      llvm::find_if(newLoopOp.getBody()->getOperations(),
                    [](Operation &op) { return isa<linalg::MatmulOp>(op); });
  if (maybeMatmulOp == newLoopOp.getBody()->getOperations().end()) {
    return emitSilenceableFailure(getLoc())
           << "No linalg.matmul op found in the loop body";
  }
  linalg::MatmulOp clonedMatmulOp = cast<linalg::MatmulOp>(*maybeMatmulOp);
  auto aTile = clonedMatmulOp.getInputs()[tileIndex];

  auto tileSubviewOp = aTile.getDefiningOp<memref::SubViewOp>();
  if (!tileSubviewOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected operand tile to be a associated with a memref.subview "
              "op, but got: "
           << aTile.getDefiningOp()->getName();
  }
  // create a subview for cooperative prefetching
  rewriter.setInsertionPoint(newLoopOp);
  auto parentLoop = newLoopOp.getOperation()->getParentOfType<scf::ForallOp>();
  if (!parentLoop) {
    return emitSilenceableFailure(getLoc())
           << "Expecting scf.forall op as parent of the loop";
  }
  auto subGroupIndVars = parentLoop.getInductionVars();
  auto subGroupUpperBound = parentLoop.getUpperBound(rewriter);
  if (subGroupIndVars.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expecting a two induction variables in subgroup loop.";
  }
  // A tile is reused by all threads defined by the 2nd induction variable
  int64_t indVarIndex = tileIndex == 0 ? 1 : 0;
  auto maybePrefetchTile =
      getIthSubtile(rewriter, aTile, subGroupIndVars[indVarIndex],
                    subGroupUpperBound[indVarIndex], tileSize);
  if (!maybePrefetchTile) {
    return emitSilenceableFailure(getLoc())
           << "Failed to generate prefetch subtile";
  }
  auto prefetchTile = *maybePrefetchTile;

  // add xegpu desc op for the tile
  auto aTileType = cast<ShapedType>(prefetchTile.getType());
  if (ShapedType::isDynamicShape(aTileType.getShape())) {
    return emitSilenceableFailure(getLoc())
           << "Prefetch tile must have static shape";
  }
  SmallVector<int64_t> descShape(aTileType.getShape());
  auto descType = xegpu::TensorDescType::get(
      descShape, aTileType.getElementType(), /*array_length=*/1,
      /*boundary_check=*/true, xegpu::MemorySpace::Global);
  auto loc = prefetchTile.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto zeroOffset = getAsOpFoldResult({zero, zero});
  auto descOp = rewriter.create<xegpu::CreateNdDescOp>(
      loc, descType, dyn_cast<TypedValue<MemRefType>>(prefetchTile),
      zeroOffset);
  // add prefetch op
  auto ctx = rewriter.getContext();
  auto readCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);
  rewriter.create<xegpu::PrefetchNdOp>(loc, descOp.getResult(), readCacheHint,
                                       readCacheHint, readCacheHint);

  // clean up matmul and unused subviews
  rewriter.eraseOp(clonedMatmulOp);
  SmallVector<Operation *> toRemoveOps;
  for (auto &op : newLoopOp.getBody()->getOperations()) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      if (subviewOp.use_empty()) {
        toRemoveOps.push_back(&op);
      }
    }
  }
  for (auto &op : toRemoveOps) {
    rewriter.eraseOp(op);
  }
  if (failed(insertOffsetUpdateOps(rewriter, newLoopOp))) {
    return emitSilenceableFailure(getLoc())
           << "No desc ops found in the loop body " << target->getName();
  }
  auto maybeNewLoopOp = hoistDescOps(rewriter, newLoopOp);
  if (failed(maybeNewLoopOp)) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Failed to hoist xegpu.create_nd_desc ops";
    diag.attachNote(newLoopOp.getLoc()) << "loop op";
    return diag;
  }
  newLoopOp = *maybeNewLoopOp;

  // peel first iteration of the loop
  scf::ForOp firstLoopOp;
  if (failed(
          scf::peelForLoopFirstIteration(rewriter, newLoopOp, firstLoopOp))) {
    auto diag = emitSilenceableFailure(getLoc()) << "Failed to peel the loop";
  }

  // reset lower bound back to original value
  newLoopOp.setLowerBound(loopOp.getLowerBound());

  results.push_back(newLoopOp);
  return DiagnosedSilenceableFailure::success();
}

void transform::XeGPUInsertPrefetchOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getMatmulMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform::XeGPUSetLoadTileOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto loadTileShape = getTileSize();
  if (loadTileShape.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected tile sizes to be a 2D vector";
  }

  // defines DPAS operand to be converted
  int64_t tileIndex = getTileIndex();
  if (tileIndex < 0 || tileIndex >= 2) {
    return emitSilenceableFailure(getLoc())
           << "Invalid tile index: " << tileIndex
           << ", expected 0 or 1 for A or B operand";
  }

  auto forOp = dyn_cast<scf::ForOp>(target);
  if (!forOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a scf.for op, but got: " << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  foldRedundantLoadOps(rewriter, forOp);

  // find all dpas ops and the desc op associated with their operand i
  SmallVector<xegpu::CreateNdDescOp> descOps;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(op)) {
      auto defOp = dpasOp.getOperand(tileIndex).getDefiningOp();
      if (!defOp) {
        return emitSilenceableFailure(getLoc())
               << "DPAS op operand does not have a defining op";
      }
      auto producers = getProducerOpsInRegion(defOp, forOp.getRegion(), true);
      auto maybeDescOp = llvm::find_if(producers, [&](Operation *op) {
        return isa<xegpu::CreateNdDescOp>(op);
      });
      if (maybeDescOp == producers.end()) {
        return emitSilenceableFailure(getLoc())
               << "DPAS op operand does not have a desc op in the loop body";
      }
      auto descOp = cast<xegpu::CreateNdDescOp>(*maybeDescOp);
      if (!llvm::is_contained(descOps, descOp)) {
        descOps.push_back(descOp);
      }
    }
  }
  if (descOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No create desc ops found, returning.\n");
    return DiagnosedSilenceableFailure::success();
  }
  llvm::DenseMap<std::tuple<Value, int64_t, int64_t>, Value> createdLoadTiles;
  for (auto &descOp : descOps) {
    LLVM_DEBUG(llvm::dbgs()
               << "Processing desc op: " << descOp.getLoc() << "\n");

    // fold descriptor to parent subview; this is SG_k tile to DPAS tile view
    auto maybeFolded = foldSubview(rewriter, descOp);
    if (!maybeFolded) {
      return emitSilenceableFailure(getLoc())
             << "Failed to fold subview into the descriptor op";
    }
    auto foldedDescOp = *maybeFolded;

    auto parentTileShape =
        cast<ShapedType>(foldedDescOp.getSource().getType()).getShape();
    if (ShapedType::isDynamicShape(parentTileShape)) {
      LLVM_DEBUG(llvm::dbgs() << "Loaded vector has dynamic shape.\n");
    }
    if (parentTileShape.size() != 2) {
      return emitSilenceableFailure(getLoc())
             << "Expected memref.subview op to have 2D shape.";
    }
    LLVM_DEBUG(llvm::dbgs() << "  Parent tile size:  [" << parentTileShape[0]
                            << ", " << parentTileShape[1] << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "  Load tile size:    [" << loadTileShape[0]
                            << ", " << loadTileShape[1] << "]\n");

    auto targetTileShapeType =
        cast<ShapedType>(foldedDescOp.getResult().getType());
    auto targetTileShape = targetTileShapeType.getShape();
    if (ShapedType::isDynamicShape(targetTileShape)) {
      LLVM_DEBUG(llvm::dbgs() << "Loaded vector has dynamic shape.\n");
    }
    if (targetTileShape.size() != 2) {
      return emitSilenceableFailure(getLoc())
             << "Expected memref.subview op to have 2D shape.";
    }
    LLVM_DEBUG(llvm::dbgs() << "  Target tile shape: [" << targetTileShape[0]
                            << ", " << targetTileShape[1] << "]\n");

    // get target tile offsets from the desc op
    SmallVector<Value> targetDynOffsets;
    SmallVector<int64_t> targetStaOffsets;
    dispatchIndexOpFoldResults(foldedDescOp.getMixedOffsets(), targetDynOffsets,
                               targetStaOffsets);
    if (ShapedType::isDynamicShape(targetStaOffsets)) {
      return emitSilenceableFailure(getLoc())
             << "Expecting fully static offsets in desc op.";
    }

    if (loadTileShape[0] > parentTileShape[0] ||
        loadTileShape[1] > parentTileShape[1]) {
      return emitSilenceableFailure(getLoc())
             << "Load tile shape is larger than parent tile shape: "
             << "[" << loadTileShape[0] << ", " << loadTileShape[1] << "] vs ["
             << parentTileShape[0] << ", " << parentTileShape[1] << "]";
    }
    if (targetTileShape[0] > loadTileShape[0] ||
        targetTileShape[1] > loadTileShape[1]) {
      return emitSilenceableFailure(getLoc())
             << "Target tile shape is larger than load tile shape: "
             << "[" << targetTileShape[0] << ", " << targetTileShape[1]
             << "] vs [" << loadTileShape[0] << ", " << loadTileShape[1] << "]";
    }

    // offset for target tile in the larger load tile
    SmallVector<int64_t> targetNestedOffsets{
        targetStaOffsets[0] % loadTileShape[0],
        targetStaOffsets[1] % loadTileShape[1]};
    // offset for the load tile in the parent tile
    SmallVector<int64_t> loadTileOffsets{
        targetStaOffsets[0] - targetNestedOffsets[0],
        targetStaOffsets[1] - targetNestedOffsets[1]};

    auto loc = descOp.getLoc();
    bool useVnni = tileIndex == 1;
    VectorType loadVecType =
        VectorType::get(loadTileShape, targetTileShapeType.getElementType());
    if (useVnni) {
      loadVecType =
          getVnniVector(loadTileShape, targetTileShapeType.getElementType());
    }
    auto maybeLoadOp = getUserOfType<xegpu::LoadNdOp>(foldedDescOp.getResult());
    if (!maybeLoadOp) {
      return emitSilenceableFailure(getLoc())
             << "xegpu.create_nd_tdesc op without a xegpu.load_nd op.";
    }
    auto oldLoadOp = *maybeLoadOp;

    // create new desc/load op, or use cached one
    std::tuple<Value, int64_t, int64_t> loadTileKey{
        foldedDescOp.getSource(), loadTileOffsets[0], loadTileOffsets[1]};
    Value loadedTileValue;
    if (llvm::is_contained(createdLoadTiles, loadTileKey)) {
      loadedTileValue = createdLoadTiles[loadTileKey];
    } else {
      // create a new descriptor for the parent subview tile
      auto ctx = rewriter.getContext();
      auto descType = xegpu::TensorDescType::get(
          loadTileShape, targetTileShapeType.getElementType(),
          /*array_length=*/1,
          /*boundary_check=*/true, xegpu::MemorySpace::Global);
      auto newDescOp = rewriter.create<xegpu::CreateNdDescOp>(
          loc, descType,
          dyn_cast<TypedValue<MemRefType>>(foldedDescOp.getSource()),
          getAsIndexOpFoldResult(ctx, loadTileOffsets));

      // create new load op
      // use packed attribute for B tile
      UnitAttr packedAttr =
          useVnni ? UnitAttr::get(rewriter.getContext()) : nullptr;
      auto readCacheHint =
          xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);
      auto loadOp = rewriter.create<xegpu::LoadNdOp>(
          loc, loadVecType, newDescOp.getResult(), packedAttr,
          /*transpose=*/nullptr,
          /*transpose_bit_width=*/nullptr, readCacheHint, readCacheHint,
          readCacheHint);
      // add to cache
      loadedTileValue = loadOp.getResult();
      createdLoadTiles[loadTileKey] = loadedTileValue;
    }

    // extract the correct target tile from the loaded vector
    // flatten vector
    int loadVecSize = std::accumulate(loadVecType.getShape().begin(),
                                      loadVecType.getShape().end(), 1,
                                      std::multiplies<int64_t>());
    auto loadVecFlatType =
        VectorType::get(loadVecSize, loadVecType.getElementType());
    auto castFlat = rewriter.create<vector::ShapeCastOp>(loc, loadVecFlatType,
                                                         loadedTileValue);
    // target tile size and offset for flattened vector
    const int targetFlatSize = targetTileShape[0] * targetTileShape[1];
    const int targetFlatOffset =
        targetNestedOffsets[0] * loadTileShape[1] + targetNestedOffsets[1];
    // extract target slice from the flattened vector
    auto slice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, castFlat, /*offsets=*/ArrayRef<int64_t>{targetFlatOffset},
        /*sizes=*/ArrayRef<int64_t>{targetFlatSize},
        /*strides=*/ArrayRef<int64_t>{1});
    // reshape to target shape
    auto targetVecType =
        VectorType::get(targetTileShape, targetTileShapeType.getElementType());
    if (useVnni) {
      targetVecType =
          getVnniVector(targetTileShape, targetTileShapeType.getElementType());
    }
    auto castTile =
        rewriter.create<vector::ShapeCastOp>(loc, targetVecType, slice);

    // replace old load op with the new tile
    rewriter.replaceOp(oldLoadOp, castTile.getResult());
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::XeGPUSetLoadTileOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getLoopMutable(), effects);
  // consumesHandle(getDescMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void registerXeGPUTransformOps(DialectRegistry &registry) {
  registry.addExtensions<XeGPUTransformOps>();
}
