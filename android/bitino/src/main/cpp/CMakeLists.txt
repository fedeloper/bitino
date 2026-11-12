# Native part of the Bitino Android library: the runtime (built by the repository's top-level CMakeLists.txt, which
# carries the per-ABI kernel flags) plus the JNI bridge, as one shared library libbitino_jni.so per ABI.
cmake_minimum_required(VERSION 3.22)
project(bitino_android LANGUAGES CXX)

get_filename_component(BITINO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
if(NOT EXISTS "${BITINO_ROOT}/runtime/include/bitino/bitino_c.h")
  message(FATAL_ERROR "runtime sources not found at ${BITINO_ROOT} (the library builds from the repository checkout)")
endif()

# Only the runtime library: no tests, tools or the plain JNI example.
set(BITINO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BITINO_BUILD_JNI OFF CACHE BOOL "" FORCE)

# Optional cap on parallel compile jobs (gradle.properties bitino.nativeJobs).
if(BITINO_NATIVE_JOBS GREATER 0)
  set_property(GLOBAL PROPERTY JOB_POOLS bitino_compile=${BITINO_NATIVE_JOBS})
  set(CMAKE_JOB_POOL_COMPILE bitino_compile)
endif()

add_subdirectory("${BITINO_ROOT}" runtime EXCLUDE_FROM_ALL)

add_library(bitino_jni SHARED bitino_jni.cc)
target_link_libraries(bitino_jni PRIVATE bitino log)
target_compile_options(bitino_jni PRIVATE -Wall -Wextra -fvisibility=hidden)
target_link_options(bitino_jni PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL)
