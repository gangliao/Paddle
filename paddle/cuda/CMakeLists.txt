file(GLOB_RECURSE CUDA_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include" "*.h")
file(GLOB_RECURSE CUDA_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/src" "*.cu")

file(GLOB CPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/x86/*.cc")
file(GLOB AVX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/x86/avx/*.cc")
set_source_files_properties(${AVX_SOURCES} PROPERTIES COMPILE_FLAGS ${AVX_FLAG})

file(GLOB CUDA_WRAP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

set(WRAP_CTC_SOURCES src/hl_warpctc_wrap.cc)

if(WITH_GPU)
    set_source_files_properties(${CPU_SOURCES}
                                ${AVX_SOURCES}
                                ${WRAP_CTC_SOURCES}
                                ${CUDA_WRAP_SOURCES}
                                PROPERTIES COMPILE_FLAGS "-D__NVCC__")
    cuda_add_library(paddle_cuda
                    ${CPU_SOURCES}
                    ${AVX_SOURCES}
                    ${CUDA_SOURCES}
                    ${WRAP_CTC_SOURCES}
                    ${CUDA_WRAP_SOURCES})
else(WITH_GPU)
    add_library(paddle_cuda
                ${CPU_SOURCES}
                ${AVX_SOURCES}
                ${WRAP_CTC_SOURCES}
                src/hl_dso_loader.cc)
endif(WITH_GPU)

add_style_check_target(paddle_cuda
                       ${CPU_SOURCES}
                       ${AVX_SOURCES}
                       ${WRAP_CTC_SOURCES}
                       ${CUDA_WRAPPER_SOURCES})
