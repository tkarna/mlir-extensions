#ifndef XEGPU_EXTENSION
#define XEGPU_EXTENSION

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XeGPUHoistDescOp : Op<Transform_Dialect, "xegpu.hoist_desc_ops",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Hoists xegpu tile descriptor ops outside the containing loop";
  let description = [{
    FIXME
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$loop);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$loop attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def XeGPUHoistLoadStoreOp : Op<Transform_Dialect, "xegpu.hoist_load_store_ops",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Hoists xegpu load/store pattern outside the containing loop";
  let description = [{
    FIXME
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$loop);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$loop attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def XeGPUInsertPrefetchOp : Op<Transform_Dialect, "xegpu.insert_prefetch",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Adds xegpu prefetch ops to matmul operand tiles.";
  let description = [{
    FIXME
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$matmul,
    I64Attr:$tileIndex,
    DenseI64ArrayAttr:$tileSize
    );

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$matmul $tileIndex $tileSize attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def XeGPUSetLoadTileOp : Op<Transform_Dialect, "xegpu.set_load_tile",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Converts given xegpu desc op into a larger load tile size and extracts the correct vector slice.";
  let description = [{
    FIXME
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$loop,
    I64Attr:$tileIndex,
    DenseI64ArrayAttr:$tileSize
  );

  let results = (outs);

  let assemblyFormat = "$loop $tileIndex $tileSize attr-dict `:` type($loop)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // XEGPU_EXTENSION
