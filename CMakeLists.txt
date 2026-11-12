cmake_minimum_required(VERSION 3.20)
project(bitino VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "aarch64 builds need clang: the NEON kernels' inline asm exceeds GCC's 30-operand limit "
                      "(cmake -DCMAKE_CXX_COMPILER=clang++ ...)")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BITINO_BUILD_TESTS "Build C++ tests" ON)
option(BITINO_SANITIZE "Build with ASan+UBSan (host debug)" OFF)
option(BITINO_NATIVE "Use -march=native on host builds" OFF)
option(BITINO_STATIC "Link executables statically (Android: lets qemu-aarch64 user mode run them)" OFF)

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
# Bit-exact agreement between kernels (and host vs device) requires that the
# compiler never fuses a*b+c into FMA behind our back.
add_compile_options(-ffp-contract=off)
if(BITINO_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()
if(BITINO_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options(-march=native)
endif()
if(BITINO_STATIC)
  add_link_options(-static)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # Baseline: ARMv8.2-A + dotprod is what the SM8450 (and most arm64 phones since
  # 2018, Neoverse N1/V1 servers, Raspberry Pi 5) report; kernels that use
  # extensions beyond the baseline gate on runtime CPU feature detection.
  add_compile_options(-march=armv8.2-a+dotprod)
endif()

set(BITINO_SOURCES
  runtime/src/json.cc
  runtime/src/tensor.cc
  runtime/src/util.cc
  runtime/src/cpu_features.cc
  runtime/src/thread_pool.cc
  runtime/src/model.cc
  runtime/src/session.cc
  runtime/src/c_api.cc
  runtime/ops/op_defs.cc
  runtime/ops/registry.cc
  runtime/ops/op_defs_torch_cnn.cc
  runtime/kernels/reference/torch_cnn_reference.cc
  runtime/ops/op_defs_larq.cc
  runtime/kernels/reference/reference_kernels.cc
  runtime/kernels/reference/larq_reference.cc
  runtime/kernels/binary_scalar/bconv_scalar.cc
  runtime/kernels/binary_neon/bconv_neon.cc
  runtime/kernels/binary_neon/bconv_neon_opt.cc
  runtime/kernels/int8_reference/int8_kernels.cc
  runtime/kernels/int8_neon/int8_neon_kernels.cc
  runtime/kernels/int8_neon/int8_gemm_sdot.cc
  runtime/kernels/int8_neon/int8_gemm_i8mm.cc
  runtime/kernels/float_opt/float_opt.cc
  runtime/kernels/float_opt/conv_gemm.cc
  runtime/kernels/float_opt/depthwise.cc
  runtime/kernels/float_opt/elementwise.cc
  runtime/ops/transformer_op_defs.cc
  runtime/kernels/reference/transformer_reference.cc
  runtime/kernels/ternary_scalar/ternary_kernels.cc
  runtime/ops/op_defs_bhvit.cc
  runtime/kernels/reference/bhvit_reference.cc
  runtime/kernels/binary_avx2/binary_avx2_kernels.cc
  runtime/kernels/binary_avx2/binary_avx2_simd.cc
  runtime/src/graph_rewrite.cc
  runtime/ops/op_defs_fused.cc
  runtime/kernels/reference/fused_reference.cc
  runtime/kernels/float_opt/fused_elementwise.cc
  runtime/kernels/float_opt/transformer_opt.cc
  runtime/kernels/binary_scalar/bmatmul_bitplane.cc
  runtime/kernels/int8_reference/int8_matmul.cc
  runtime/ops/op_defs_int8_full.cc
  runtime/kernels/int8_full/int8_full_reference.cc
  runtime/kernels/int8_full/int8_full_neon.cc
  runtime/kernels/int8_full/int8_full_gemm_i8mm.cc
  runtime/kernels/int8_full/int8_full_portable.cc
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # binary_avx2 tier: only the SIMD TU gets AVX2 code generation; it is called
  # only when cpu_info() reports avx2 + popcnt (runtime/kernels/binary_avx2/binary_avx2.h).
  set_source_files_properties(runtime/kernels/binary_avx2/binary_avx2_simd.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mpopcnt")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # INT8 GEMM micro-kernels: one TU per ISA extension, selected only when
  # cpu_info() reports the feature (see runtime/kernels/int8_neon/int8_gemm.h).
  set_source_files_properties(runtime/kernels/int8_neon/int8_gemm_sdot.cc PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
  set_source_files_properties(runtime/kernels/int8_neon/int8_gemm_i8mm.cc PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+i8mm")
  set_source_files_properties(runtime/kernels/int8_full/int8_full_gemm_i8mm.cc PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+i8mm")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # x86-64 float_opt AVX2+FMA leaf routines: the only TU built with these flags; its
  # kernels are selected only when cpu_info() reports avx2+fma, so the default host
  # binary still runs on any x86-64 (see runtime/kernels/float_opt/float_opt_avx2.h).
  list(APPEND BITINO_SOURCES runtime/kernels/float_opt/float_opt_avx2.cc)
  set_source_files_properties(runtime/kernels/float_opt/float_opt_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  # int8_avx2 tier: baseline driver + one TU per ISA level (runtime/kernels/int8_avx2/int8_avx2.h).
  # No -mfma here: the int8 epilogue must round the multiply and the add separately.
  list(APPEND BITINO_SOURCES runtime/kernels/int8_avx2/int8_avx2_kernels.cc
       runtime/kernels/int8_avx2/int8_gemm_avx2.cc runtime/kernels/int8_avx2/int8_gemm_avxvnni.cc)
  set_source_files_properties(runtime/kernels/int8_avx2/int8_gemm_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2")
  set_source_files_properties(runtime/kernels/int8_avx2/int8_gemm_avxvnni.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
else()
  # register_int8_avx2_kernels() is a no-op off x86-64.
  list(APPEND BITINO_SOURCES runtime/kernels/int8_avx2/int8_avx2_kernels.cc)
endif()

add_library(bitino STATIC ${BITINO_SOURCES})
target_include_directories(bitino PUBLIC runtime/include PRIVATE runtime)
find_package(Threads REQUIRED)
target_link_libraries(bitino PUBLIC Threads::Threads)

add_executable(bitino_bench runtime/tools/bitino_bench.cc)
target_link_libraries(bitino_bench PRIVATE bitino)
target_include_directories(bitino_bench PRIVATE runtime)

add_executable(bitino_microbench runtime/tools/bitino_microbench.cc)
target_link_libraries(bitino_microbench PRIVATE bitino)
target_include_directories(bitino_microbench PRIVATE runtime)

if(ANDROID)
  option(BITINO_BUILD_JNI "Build the JNI example library" ON)
  if(BITINO_BUILD_JNI AND NOT BITINO_STATIC)
    add_library(bitino_jni SHARED examples/android_jni/bitino_jni.cc)
    target_link_libraries(bitino_jni PRIVATE bitino)
  endif()
endif()

add_executable(bitino_int8_quantbench runtime/tools/bitino_int8_quantbench.cc)
target_link_libraries(bitino_int8_quantbench PRIVATE bitino)
target_include_directories(bitino_int8_quantbench PRIVATE runtime)

if(BITINO_BUILD_TESTS)
  enable_testing()
  add_executable(bitino_tests tests/cpp/test_main.cc tests/cpp/test_core.cc tests/cpp/test_binary_kernels.cc
    tests/cpp/test_int8_kernels.cc tests/cpp/test_float_kernels.cc tests/cpp/test_transformer_ops.cc
    tests/cpp/test_bhvit_ops.cc tests/cpp/test_brevitas_ops.cc tests/cpp/test_larq_ops.cc
    tests/cpp/test_torch_cnn.cc tests/cpp/test_int8_full.cc)
  target_sources(bitino_tests PRIVATE tests/cpp/test_binary_avx2.cc)
  target_sources(bitino_tests PRIVATE tests/cpp/test_transformer_opt.cc)
  target_sources(bitino_tests PRIVATE tests/cpp/test_bit_outputs.cc)
  target_sources(bitino_tests PRIVATE tests/cpp/test_neon_a32.cc)
  target_link_libraries(bitino_tests PRIVATE bitino)
  target_include_directories(bitino_tests PRIVATE runtime tests/cpp)
  add_test(NAME bitino_tests COMMAND bitino_tests)
endif()

# Record build configuration inside benchmark reports.
get_directory_property(_bf_opts COMPILE_OPTIONS)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bf_bt)
string(REPLACE ";" " " _bf_opts_str "${_bf_opts}")
target_compile_definitions(bitino_bench PRIVATE
  BITINO_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bf_bt}} ${_bf_opts_str}"
  BITINO_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
