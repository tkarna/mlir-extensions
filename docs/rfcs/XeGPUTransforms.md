# RFC for XeGPU Transform ops

## Summary

The XeGPU dialect capabilities need to be extended upstream to facilitate more generic lowering of high-level operations. Specifically we need to support lowering of `linalg` dialect operations, such as `linalg.matmul`, to the XeGPU dialect.

Existing upstream transform dialect operations can be used to prepare a matmul operation for XeGPU lowering, for example, by applying appropriate tiling or unrolling transforms, but many XeGPU specific transform patterns are still missing.

This proposal outlines the following new operations to fill the gaps. These operators reside in the XeGPU namespace of the transform dialect:

* `transform.xegpu.hoist_desc_ops`: hoists `xegpu.create_nd_tdesc` operations out the containing loop if possible.
* `transform.xegpu.hoist_load_store_ops`: hoists `xegpu.load_nd` and `xegpu.store_nd` operations out of the containing loop if the tile is loop invariant.
* `transform.xegpu.insert_prefetch`: Inserts XeGPU cooperative prefetch operations to `linalg.matmul` operand tile A or B.
* `transform.xegpu.set_load_tile`: Replaces load operations of a `xegpu.dpas` operand A or B with a larger load tile.
* `transform.xegpu.insert_thread_sync`: Adds thread sync ops to reduction loop.

These operations are sufficient for lowering a `linalg.matmul` operation to XeGPU dialect and provide necessary transforms to obtain good performance on Intel GPUs (PVC, 4k matmul benchmark).

The proposed transform operations and their APIs are subject to change as we add support for more payload operations. The operations themselves are composed of several more granular patterns and could be split to different transform ops or patterns, for example.

## Example: 4k matrix multiplication payload

Consider the following 4k `linalg.matmul` payload function.

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>,
               %arg2: memref<4096x4096xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<4096x4096xf16>, memref<4096x4096xf16>)
                outs(%arg2 : memref<4096x4096xf16>)
  return
}
```

This function can be mapped to GPU work groups (WG) and subgroups (SG) using the following upstream transform operations:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // WG tiling
    %1, %loop_wg = transform.structured.tile_using_forall %0 tile_sizes [256,256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // SG tiling
    %2, %loop_sg = transform.structured.tile_using_forall %1 tile_sizes [32,64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // tile k-dimension
    %sg_matmul, %loop_k = transform.structured.tile_using_for %2 tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // tile inner matmul op to DPAS size, change loop order to k,n,m
    %dpas_matmul, %dpas_loops:3 = transform.structured.tile_using_for %sg_matmul tile_sizes [8, 16, 16] interchange = [2, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

After the transformations (`imex-opt -transform-interpreter -lower-affine -canonicalize`) the payload function is correctly tiled to parallel WG and SG loops, followed by a sequential reduction loop and 3 DPAS loops.

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  scf.forall (%arg3, %arg4) in (16, 16) {
    %0 = arith.muli %arg3, %c256 overflow<nsw> : index
    %1 = arith.muli %arg4, %c256 overflow<nsw> : index
    %subview = memref.subview %arg0[%0, 0] [256, 4096] [1, 1] : memref<4096x4096xf16> to memref<256x4096xf16, strided<[4096, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[0, %1] [4096, 256] [1, 1] : memref<4096x4096xf16> to memref<4096x256xf16, strided<[4096, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%0, %1] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
    scf.forall (%arg5, %arg6) in (8, 4) {
      %2 = arith.muli %arg5, %c32 overflow<nsw> : index
      %3 = arith.muli %arg6, %c64 overflow<nsw> : index
      %subview_2 = memref.subview %subview[%2, 0] [32, 4096] [1, 1] : memref<256x4096xf16, strided<[4096, 1], offset: ?>> to memref<32x4096xf16, strided<[4096, 1], offset: ?>>
      %subview_3 = memref.subview %subview_0[0, %3] [4096, 64] [1, 1] : memref<4096x256xf16, strided<[4096, 1], offset: ?>> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
      %subview_4 = memref.subview %subview_1[%2, %3] [32, 64] [1, 1] : memref<256x256xf16, strided<[4096, 1], offset: ?>> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
      scf.for %arg7 = %c0 to %c4096 step %c32 {
        %subview_5 = memref.subview %subview_2[0, %arg7] [32, 32] [1, 1] : memref<32x4096xf16, strided<[4096, 1], offset: ?>> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
        %subview_6 = memref.subview %subview_3[%arg7, 0] [32, 64] [1, 1] : memref<4096x64xf16, strided<[4096, 1], offset: ?>> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
        scf.for %arg8 = %c0 to %c32 step %c16 {
          scf.for %arg9 = %c0 to %c32 step %c8 {
            scf.for %arg10 = %c0 to %c64 step %c16 {
              %subview_7 = memref.subview %subview_5[%arg9, %arg8] [8, 16] [1, 1] : memref<32x32xf16, strided<[4096, 1], offset: ?>> to memref<8x16xf16, strided<[4096, 1], offset: ?>>
              %subview_8 = memref.subview %subview_6[%arg8, %arg10] [16, 16] [1, 1] : memref<32x64xf16, strided<[4096, 1], offset: ?>> to memref<16x16xf16, strided<[4096, 1], offset: ?>>
              %subview_9 = memref.subview %subview_4[%arg9, %arg10] [8, 16] [1, 1] : memref<32x64xf16, strided<[4096, 1], offset: ?>> to memref<8x16xf16, strided<[4096, 1], offset: ?>>
              linalg.matmul ins(%subview_7, %subview_8 : memref<8x16xf16, strided<[4096, 1], offset: ?>>, memref<16x16xf16, strided<[4096, 1], offset: ?>>) outs(%subview_9 : memref<8x16xf16, strided<[4096, 1], offset: ?>>)
            }
          }
        }
      }
    }
  }
  return
}
```

We can now lower the `linalg.matmul` operations to the vector dialect:

```mlir
    // Vectorize linalg.matmul
    transform.structured.vectorize %dpas_matmul : !transform.any_op
    // Convert to vector.contract
    transform.apply_patterns to %dpas_loops#2 {
      transform.apply_patterns.vector.reduction_to_contract
      // Reduce the rank of xfer ops. This transforms vector.contract to be
      // more matmul-like and to enable the lowering to outer product Ops.
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op
```

After unrolling the DPAS loops,

```mlir
    // unroll DPAS loops, factor=size/tile_size must be computed
    transform.loop.unroll %dpas_loops#2 {factor = 4} : !transform.any_op
    transform.loop.unroll %dpas_loops#1 {factor = 4} : !transform.any_op
    transform.loop.unroll %dpas_loops#0 {factor = 2} : !transform.any_op
```

and applying the `convert-vector-to-xegpu` pass, the reduction loop becomes:

```mlir
        ...
        scf.for %arg7 = %c0 to %c4096 step %c32 {
          %subview_5 = memref.subview %subview_2[0, %arg7] [32, 32] [1, 1] : memref<32x4096xf16, strided<[4096, 1], offset: ?>> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
          %subview_6 = memref.subview %subview_3[%arg7, 0] [32, 64] [1, 1] : memref<4096x64xf16, strided<[4096, 1], offset: ?>> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
          %subview_7 = memref.subview %subview_5[0, 0] [8, 16] [1, 1] : memref<32x32xf16, strided<[4096, 1], offset: ?>> to memref<8x16xf16, strided<[4096, 1], offset: ?>>
          %subview_8 = memref.subview %subview_6[0, 0] [16, 16] [1, 1] : memref<32x64xf16, strided<[4096, 1], offset: ?>> to memref<16x16xf16, strided<[4096, 1], offset: ?>>
          %subview_9 = memref.subview %subview_4[0, 0] [8, 16] [1, 1] : memref<32x64xf16, strided<[4096, 1], offset: ?>> to memref<8x16xf16, strided<[4096, 1], offset: ?>>
          %4 = xegpu.create_nd_tdesc %subview_7[0, 0] : memref<8x16xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
          %6 = xegpu.create_nd_tdesc %subview_8[0, 0] : memref<16x16xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %7 = xegpu.load_nd %6  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<16x16xf16>
          %8 = xegpu.create_nd_tdesc %subview_9[0, 0] : memref<8x16xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %9 = xegpu.load_nd %8  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
          %10 = xegpu.dpas %5, %7, %9 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf16> -> vector<8x16xf16>
          xegpu.store_nd %10, %8  : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          ...
        }
```

That is, each DPAS op has separate XeGPU descriptor and load/store ops for all the operand tiles A, B, and C.

Several transformations are needed to convert the reduction loop into the desired form:

* A. Hoist descriptor ops of A and B tiles out of the loop, and add `update_nd_offset` to increment the offset.
* B. Hoist the descriptor, load, and store ops of the loop-invariant C tile out of the loop.
* C. Cast the accumulator C tile element type from `f16` to `f32` to match DPAS op specification.
* D. Insert a `gpu.barrier` op to the loop to periodically syncronize the threads.
* E. Add cooperative prefetching ops to A and B tiles.
* F. Reduce the number of load operations by loading A and B in a larger tile than the DPAS tile size (8x16 or 16x16).
* G. Apply correct VNNI layout for the DPAS ops.

These transformations are achieved with the following XeGPU transform operations.

## `transform.xegpu.hoist_desc_ops` Operation

Hoists descriptor ops of loop-dependent tiles out of the loop and inserts `xegpu.update_nd_offset` in the loop. Addresses transformation A.

Let `%loop_k2` be a handle to the naive xegpu reduction `scf.for` loop listed above. We apply CSE and `transform.xegpu.hoist_desc_ops` to it:

```mlir
    transform.apply_cse to %loop_k2 : !transform.any_op
    %loop_k3 = transform.xegpu.hoist_desc_ops %loop_k2 : (!transform.any_op) -> !transform.any_op
```

This creates a loop-invariant descriptor op for A and B tiles, hoists them out of the loop:

```mlir
        ...
        %4 = xegpu.create_nd_tdesc %subview_2[0, %c0] : memref<32x4096xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
        %5 = xegpu.create_nd_tdesc %subview_3[%c0, 0] : memref<4096x64xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
        %6 = xegpu.create_nd_tdesc %subview_4[0, 0] : memref<32x64xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
        ...
        %36:16 = scf.for %arg7 = %c0 to %c4096 step %c32 iter_args(%arg8 = %4, %arg9 = %5, ...) -> (!xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>, ...) {
          %37 = xegpu.update_nd_offset %arg8, [0, %c32] : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %38 = xegpu.load_nd %arg8  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
          %39 = xegpu.update_nd_offset %arg9, [%c32, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %40 = xegpu.load_nd %arg9  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<16x16xf16>
          %41 = xegpu.load_nd %6  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
          %42 = xegpu.dpas %38, %40, %41 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf16> -> vector<8x16xf16>
          xegpu.store_nd %42, %6  : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          ...
          scf.yield %37, %39, ... : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>, ...
        }
```

The descriptor is added in the loops `iter_args`, updating the offset by the k tile size on every iteration.

Note that the load and store ops of the loop-independent C tile are not changed.

## `transform.xegpu.hoist_load_store_ops` Operation

To handle the C tile load and store patterns we apply the `transform.xegpu.hoist_load_store_ops`:

```mlir
%loop_k4 = transform.xegpu.hoist_load_store_ops %loop_k3 : (!transform.any_op) -> !transform.any_op
```

which results in reduction loop:

```mlir
        ...
        %66 = xegpu.load_nd %6  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
        %67 = arith.extf %66 : vector<8x16xf16> to vector<8x16xf32> // f16 -> f32
        %68:32 = scf.for %arg7 = %c0 to %c4096 step %c32 iter_args(..., %arg24 = %67, ...) -> (..., vector<8x16xf32>, ...) {
          %85 = xegpu.update_nd_offset %arg8, [0, %c32] : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %86 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
          %87 = xegpu.update_nd_offset %arg9, [%c32, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %88 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<16x16xf16>
          %89 = xegpu.dpas %86, %88, %arg24 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          ...
          %121 = xegpu.dpas %118, %120, %89 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          ...
          scf.yield ..., %121, ... : ..., vector<8x16xf32>, ...
        }
        ...
        %69 = arith.truncf %68#16 : vector<8x16xf32> to vector<8x16xf16> // f32 -> f16
        xegpu.store_nd %69, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>        ...
```

This is a DPAS specific pattern. Addresses transformations B and C:

* Hosts the load ops before the loop. Vector is added in loop's `iter_args`.
* Adds float16 to float32 conversion to the loaded vector.
* Chains dependent DPAS ops correctly in the loop (`%91` -> `%123`)
* Pushes the final store op after the loop, with the appropriate float32 to float16 conversion.

## `transform.xegpu.insert_prefetch` Operation

To add cooperative prefetching for SG level A and B tiles, we go back to the SG level loop structure, before applying DPAS tiling. On this level, the `linalg.matmul` op has the A and B operand tiles that must be prefetched:

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  scf.forall (%arg3, %arg4) in (16, 16) {
    %0 = arith.muli %arg3, %c256 overflow<nsw> : index
    %1 = arith.muli %arg4, %c256 overflow<nsw> : index
    %subview = memref.subview %arg0[%0, 0] [256, 4096] [1, 1] : memref<4096x4096xf16> to memref<256x4096xf16, strided<[4096, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[0, %1] [4096, 256] [1, 1] : memref<4096x4096xf16> to memref<4096x256xf16, strided<[4096, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%0, %1] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
    scf.forall (%arg5, %arg6) in (8, 4) {  // SG parallel loop
      %2 = arith.muli %arg5, %c32 overflow<nsw> : index
      %3 = arith.muli %arg6, %c64 overflow<nsw> : index
      %subview_2 = memref.subview %subview[%2, 0] [32, 4096] [1, 1] : memref<256x4096xf16, strided<[4096, 1], offset: ?>> to memref<32x4096xf16, strided<[4096, 1], offset: ?>>
      %subview_3 = memref.subview %subview_0[0, %3] [4096, 64] [1, 1] : memref<4096x256xf16, strided<[4096, 1], offset: ?>> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
      %subview_4 = memref.subview %subview_1[%2, %3] [32, 64] [1, 1] : memref<256x256xf16, strided<[4096, 1], offset: ?>> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
      scf.for %arg7 = %c0 to %c4096 step %c32 {
        %subview_5 = memref.subview %subview_2[0, %arg7] [32, 32] [1, 1] : memref<32x4096xf16, strided<[4096, 1], offset: ?>> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
        %subview_6 = memref.subview %subview_3[%arg7, 0] [32, 64] [1, 1] : memref<4096x64xf16, strided<[4096, 1], offset: ?>> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
        linalg.matmul ins(%subview_5, %subview_6 : memref<32x32xf16, strided<[4096, 1], offset: ?>>, memref<32x64xf16, strided<[4096, 1], offset: ?>>) outs(%subview_4 : memref<32x64xf16, strided<[4096, 1], offset: ?>>)
      }
    }
  }
  return
}
```

Moreover, the parent SG parallel loop indicates how the subgroups are organized: in a 8x4 grid in this case. We can therefore identify the 4 threads that share the same A tile, and the 8 threads that share the same B tile. The respective tiles are dividend among these threads for cooperative prefetching.

Let `%sg_matmul` be the above `linalg.matmul` op. To emit xegpu prefetch operations for tile A, we apply `transform.xegpu.insert_prefetch` transform op to it:

```mlir
%prefetch_loop_a = transform.xegpu.insert_prefetch %sg_matmul index = 0 tile_size = [8, 32] : (!transform.any_op) -> !transform.any_op
```

The index attribute defines the matmul operand tile (0 for A, 1 for B) to prefetch using the speficied prefect tile size. The tile size must be compatible with the SG A tile and the number of sharing threads (resulting in a 4-way row decompostion in this case).

The transformation generates a new loop before the reduction loop, that only contains the prefetch operations.

```mlir
      scf.forall (%arg5, %arg6) in (8, 4) {
        ...
        %c8 = arith.constant 8 : index
        %4 = arith.muli %arg6, %c8 : index
        %5 = xegpu.create_nd_tdesc %subview_2[%4, %c0] : memref<32x4096xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        %6 = xegpu.update_nd_offset %5, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.prefetch_nd %5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        %7 = scf.for %arg7 = %c0 to %c4096 step %c32 iter_args(%arg8 = %6) -> (!xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>) {
          %8 = xegpu.update_nd_offset %arg8, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
          xegpu.prefetch_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
          scf.yield %8 : !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        }
        scf.for %arg7 = %c0 to %c4096 step %c32 {
          ...
        }
        ...
      }
```

As seen above, it creates the descriptor for the thread's subtile, prefetches it, and prefetches the next tile on every iteration. The new `%prefetch_loop_a` can be fused to the reduction loop with `transform.loop.fuse_sibling` operation.

Prefetcing B tile can be handled analogously.

Addresses transformation E.

## `transform.xegpu.set_load_tile` Operation

In the above schedule, we end up with many small load operations for DPAS A and B tiles (8x16 and 16x16,respectively) in the reduction loop, which hampers performance. Before applying `transform.xegpu.hoist_desc_ops` the reduction loop is:

```mlir
        scf.for %arg7 = %c0 to %c4096 step %c32 {
          ...
          %4 = xegpu.create_nd_tdesc %subview_7[0, 0] : memref<8x16xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<8x16xf16>
          %6 = xegpu.create_nd_tdesc %subview_8[0, 0] : memref<16x16xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>>
          %7 = xegpu.load_nd %6  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = false>> -> vector<16x16xf16>
          ...
          %10 = xegpu.dpas %5, %7, %9 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf16> -> vector<8x16xf16>
          ...
        }
```

We can combine the descriptor and load ops to a larger load op with the `transform.xegpu.set_load_tile` transform. For example, the following operation loads the A tiles using a 32x32 shape:

```mlir
transform.xegpu.set_load_tile %loop_k2 index = 0 tile_size = [32, 16] : !transform.any_op
```

The reduction loop becomes:

```mlir
        scf.for %arg7 = %c0 to %c4096 step %c32 {
          %subview_5 = memref.subview %subview_2[0, %arg7] [32, 32] [1, 1] : memref<32x4096xf16, strided<[4096, 1], offset: ?>> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
          ...
          %4 = xegpu.create_nd_tdesc %subview_5[0, 0] : memref<32x32xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
          %5 = xegpu.load_nd %4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
          %6 = vector.shape_cast %5 : vector<32x16xf16> to vector<512xf16>
          %7 = vector.extract_strided_slice %6 {offsets = [0], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
          %8 = vector.shape_cast %7 : vector<128xf16> to vector<8x16xf16>
          ...
          %13 = xegpu.dpas %8, %10, %12 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf16> -> vector<8x16xf16>
          ...
        }
```

That is, it creates a descriptor for the larger load tile, loads it to a vector, and uses `vector` dialect ops to get the appropriate subtile from it. (`vector.extract_strided_slice` only support static offsets so this transformation can only be done on the fully unrolled DPAS loop nest). Same pattern is repeated for all the load ops, reusing the existing load ops when appropriate.

It also applies VNNI layout for the B tile (using VNNI factor 2 on dimension 0):

```mlir
transform.xegpu.set_load_tile %loop_k2 index = 1 tile_size = [32, 16] : !transform.any_op
```

results in

```mlir
        scf.for %arg7 = %c0 to %c4096 step %c32 {
          ...
          %subview_6 = memref.subview %subview_3[%arg7, 0] [32, 64] [1, 1] : memref<4096x64xf16, strided<[4096, 1], offset: ?>> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
          ...
          %9 = xegpu.create_nd_tdesc %subview_6[0, 0] : memref<32x64xf16, strided<[4096, 1], offset: ?>> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
          %10 = xegpu.load_nd %9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
          %11 = vector.shape_cast %10 : vector<16x16x2xf16> to vector<512xf16>
          %12 = vector.extract_strided_slice %11 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
          %13 = vector.shape_cast %12 : vector<256xf16> to vector<8x16x2xf16>
          ...
          %16 = xegpu.dpas %8, %13, %15 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf16> -> vector<8x16xf16>
          ...
        }
```

The descriptor ops can now be hoisted with the `transform.xegpu.hoist_desc_ops` transform as detailed above. Addresses transformations F and G.

## `transform.xegpu.insert_thread_sync` Operation

Finally we can insert gpu thread sync operations to the beginning of the reduction loop:

```mlir
// insert thread sync to reduction loop
transform.xegpu.insert_thread_sync %loop_k2 split = 4 max_step = 1024 : !transform.any_op
```

The above command issues at least 4 syncs during the execution of the reduction loop, with maximum step size of 1024.
The reduction loop becomes:

```mlir
        scf.for %arg7 = %c0 to %c4096 step %c32 {
          %4 = arith.remui %arg7, %c1024 : index
          %5 = arith.cmpi eq, %4, %c0 : index
          scf.if %5 {
            gpu.barrier
          }
          ...
        }
```

## Full lowering schedule

Combining the above transformations we can now write the full lowering schedule for the matmul operation:

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>,
                  %arg2: memref<4096x4096xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<4096x4096xf16>, memref<4096x4096xf16>)
                outs(%arg2 : memref<4096x4096xf16>)
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // WG tiling
    %1, %loop_wg = transform.structured.tile_using_forall %0 tile_sizes [256,256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // SG tiling
    %2, %loop_sg = transform.structured.tile_using_forall %1 tile_sizes [32,64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // tile k-dimension
    %sg_matmul, %loop_k = transform.structured.tile_using_for %2 tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // generate prefetch k loops for sg level A and B tiles with cooperative prefetching
    %prefetch_loop_a = transform.xegpu.insert_prefetch %sg_matmul index = 0 tile_size = [8, 32] : (!transform.any_op) -> !transform.any_op
    %prefetch_loop_b = transform.xegpu.insert_prefetch %sg_matmul index = 1 tile_size = [8, 32] : (!transform.any_op) -> !transform.any_op

    // tile inner matmul op to DPAS size, change loop order to k,n,m
    %dpas_matmul, %dpas_loops:3 = transform.structured.tile_using_for %sg_matmul tile_sizes [8, 16, 16] interchange = [2, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Vectorize linalg.matmul
    transform.structured.vectorize %dpas_matmul : !transform.any_op
    // Convert to vector.contract
    transform.apply_patterns to %dpas_loops#2 {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op

    // unroll DPAS loops, factor=size/tile_size must be computed
    transform.loop.unroll %dpas_loops#2 {factor = 4} : !transform.any_op
    transform.loop.unroll %dpas_loops#1 {factor = 4} : !transform.any_op
    transform.loop.unroll %dpas_loops#0 {factor = 2} : !transform.any_op

    // apply vector-to-xegpu pass
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %func2 = transform.apply_registered_pass "convert-vector-to-xegpu" to %func : (!transform.any_op) -> !transform.any_op

    // canonicalize
    transform.apply_cse to %func2 : !transform.any_op
    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // match for loops
    %match_forloop = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
    %prefetch_loop_a2, %prefetch_loop_b2, %loop_k2 = transform.split_handle %match_forloop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // set load tile size for DPAS A and B tiles
    transform.xegpu.set_load_tile %loop_k2 index = 0 tile_size = [32, 16] : !transform.any_op
    transform.xegpu.set_load_tile %loop_k2 index = 1 tile_size = [32, 16] : !transform.any_op

    // clean up
    transform.apply_cse to %loop_k2 : !transform.any_op
    transform.apply_patterns to %loop_k2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // hoist xegpu tile descriptor ops
    %loop_k3 = transform.xegpu.hoist_desc_ops %loop_k2 : (!transform.any_op) -> !transform.any_op
    %loop_k4 = transform.xegpu.hoist_load_store_ops %loop_k3 : (!transform.any_op) -> !transform.any_op

    // fuse prefetch loops into k loop
    %fused_loop = transform.loop.fuse_sibling %prefetch_loop_b2 into %loop_k4 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    %fused_loop2 = transform.loop.fuse_sibling %prefetch_loop_a2 into %fused_loop : (!transform.any_op, !transform.any_op) -> !transform.any_op

    // add thread sync to reduction loop
    transform.xegpu.insert_thread_sync %fused_loop2 split = 4 max_step = 1024 : !transform.any_op

    transform.yield
  }
}
```

The above schedule exposes the following paremeters:

* WG tile size: [256, 256]
* SG tile size: [32, 64]
* K tile size: 32
* DPAS tile sizes [8, 16, 16]
* Prefetch tile sizes for A and B: [8, 32], [8, 32]
* Load tile sizes for A and B: [16, 32], [16, 32]
* Thread sync step: split=4, max_step=1024

## Performance

The above schedule yields >200 TFLOPS/s performance on a single PVC tile.

## Future work

Bugs/Missing features:

* `convert-vector-to-xegpu` pass does not currently support `(f16, f16) -> f32` matrix multiplication (`vector.contract`) operations.
* `imex-xegpu-apply-vnni-transformation` does convert DPAS operand layout correctly if the operands have been loaded directly as DPAS size vectors. However, if a larger tile is loaded and sliced to DPAS size (see `transform.xegpu.set_load_tile` op) the pass produces incorrect code (functional but spills registers).
* Add parameter for thread sync frequency.
* Support more VNNI layouts (?).
* Support XeGPU tensor descriptors with `array_length != 1` (?).
* `convert-vector-to-xegpu` pass should be available as a transform operation or a pattern.
* `xegpu-fold-alias-ops` pass should be available as a transform operation or a pattern.
* Add `xegpu` dialect canonicalization patterns, e.g., constant folding.
* Pattern to fold redundant xegpu load ops if only read memory effects exist.
