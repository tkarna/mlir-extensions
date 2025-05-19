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
    Hoists `xepu.create_nd_tdesc` out of the loop. If the descriptor's offset is loop dependent, a `xegpu.update_nd_offset` op is inserted in the loop to update the offset.
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
    Hoists `xegpu.load_nd` and `xegpu.store_nd` ops out of the loop if they are loop invariant. If multiple load/store pairs exists in the loop, updates
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
    Given a `linalg.matmul` operation residing in a `scf.for` loop, this transform inserts cooperative `xegpu.prefetch` operations for the A (index = 0) or B (index = 1) operand. The prefetch tile size is specified by the `tile_size` attribute. The cooperating threads are determined by inspecting the enclosing `scf.forall` operation's iteration space.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$matmul,
    I64Attr:$tileIndex,
    DenseI64ArrayAttr:$tileSize
    );

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$matmul `index` `=` $tileIndex `tile_size` `=` $tileSize attr-dict `:` functional-type(operands, results)";

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
    Finds `xegpu.dpas` ops in the given loop and combines the individual desc and load operations with a larger one, defined by the `tile_size` attribute. The `tile_index` attribute specifies which DPAS operand tile (0 for A, 1 for B) to modify.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$loop,
    I64Attr:$tileIndex,
    DenseI64ArrayAttr:$tileSize
  );

  let results = (outs);

  let assemblyFormat = "$loop `index` `=` $tileIndex `tile_size` `=` $tileSize attr-dict `:` type($loop)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def XeGPUInsertThreadSyncOp : Op<Transform_Dialect, "xegpu.insert_thread_sync",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Inserts a gpu.barrier operation to loop body.";
  let description = [{
    Inserts a `gpu.barrier` operation to the beginning of the loop body to synchronize threads.
    The attribute `split` defines how many times threads are synced during the loop execution. The maximum sync interval can be capped by `max_step`.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$loop,
    I64Attr:$syncSplit,
    I64Attr:$maxStep
  );

  let results = (outs);

  let assemblyFormat = "$loop `split` `=` $syncSplit `max_step` `=` $maxStep attr-dict `:` type($loop)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // XEGPU_EXTENSION
