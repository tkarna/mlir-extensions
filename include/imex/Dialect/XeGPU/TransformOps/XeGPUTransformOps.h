// Transform dialect extension
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include <imex/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h.inc>

// Register extension
void registerXeGPUTransformOps(::mlir::DialectRegistry &registry);
