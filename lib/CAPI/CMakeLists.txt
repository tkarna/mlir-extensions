set(IMEX_ALL_LIBS
  IMEXTransforms
  IMEXXeTileTransforms
  IMEXXeTileDialect
  IMEXNDArrayTransforms
  IMEXNDArrayDialect
  IMEXGPUXDialect
  IMEXDistRuntimeTransforms
  IMEXDistRuntimeDialect
  IMEXRegionTransforms
  IMEXRegionDialect
  IMEXGPUToGPUX
  IMEXGPUToSPIRV
  IMEXArithToVC
  IMEXXeGPUToVC
  IMEXDropRegions
  IMEXSCFToGPU
  IMEXGPUXToLLVM
  IMEXXeTileToXeGPU
  IMEXNDArrayToLinalg
  IMEXMathToVC
  IMEXUtil
  imex_runner_utils
)

if(IMEX_ENABLE_L0_RUNTIME)
  list(APPEND IMEX_ALL_LIBS level-zero-runtime)
endif()

if(IMEX_ENABLE_SYCL_RUNTIME)
  list(APPEND IMEX_ALL_LIBS sycl-runtime)
endif()

add_mlir_public_c_api_library(ImexCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  IMEXRegionDialect
  ${IMEX_ALL_LIBS}
)
target_link_libraries(obj.ImexCAPI PUBLIC ImexInterface)
set_property(GLOBAL APPEND PROPERTY IMEX_MLIR_LIBS ImexCAPI)
