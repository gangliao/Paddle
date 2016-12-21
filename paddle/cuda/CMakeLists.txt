# header srcs
file(GLOB_RECURSE CUDA_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include" "*.h")

# cuda srcs
file(GLOB_RECURSE CUDA_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/src" "*.cu")

# cpu naive srcs
file(GLOB CPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/x86/*.cc)

# cpu avx srcs
file(GLOB AVX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/x86/avx/*.cc)
set_source_files_properties(${AVX_SOURCES}
                            PROPERTIES COMPILE_FLAGS ${AVX_FLAG})

# cuda dynamic load wrapper srcs
set(CUDA_WRAPPER_SOURCES
    src/hl_dso_loader.cc
    src/hl_cudart_wrap.cc
    src/hl_cuda_cublas.cc
    src/hl_cuda_cudnn.cc
    src/hl_cuda_device.cc
)
set_source_files_properties(${CUDA_WRAPPER_SOURCES}
                            PROPERTIES COMPILE_FLAGS "-D__NVCC__")

# external plugin srcs: warp-ctc
set(WRAP_CTC_SOURCES src/hl_warpctc_wrap.cc)

if(WITH_GPU)
    cuda_add_library(
        paddle_cuda
        ${CPU_SOURCES}
        ${AVX_SOURCES}
        ${CUDA_SOURCES}
        ${WRAP_CTC_SOURCES}
        ${CUDA_WRAPPER_SOURCES})
else()
    add_library(
        paddle_cuda
        ${CPU_SOURCES}
        ${AVX_SOURCES}
        ${WRAP_CTC_SOURCES})
endif()

add_style_check_target(
        paddle_cuda
        ${CPU_SOURCES}
        ${AVX_SOURCES}
        ${WRAP_CTC_SOURCES}
        ${CUDA_WRAPPER_SOURCES})
