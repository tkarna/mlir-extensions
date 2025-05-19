# XeGPU transform dialect extension
add_mlir_library(
  IMEXXeGPUTransformOps

  XeGPUTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Dialect/XeGPU/TransformOps/

  DEPENDS
  XeGPUTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)
