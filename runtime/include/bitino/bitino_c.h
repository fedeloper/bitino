/* Bitino stable C API (for JNI, Python ctypes and other FFI users).
 *
 * All functions are thread-compatible: distinct handles may be used from
 * different threads; one session handle must not be run concurrently.
 * Errors: functions return 0 on success and non-zero on failure; the message
 * of the last failure on the calling thread is available via bf_last_error().
 *
 * Version 1 (bf_model_load ... bf_session_get_output) is unchanged. Version 2 adds
 * session options (stage, affinity), multi-input models, tensor descriptions, direct
 * access to the session's input/output buffers and an in-library timing loop; it is
 * what the Python package binds (python/bitino/_native.py). Strings returned by the
 * library stay valid until the handle they came from is freed (bf_last_error: until
 * the next failing call on the same thread).
 */
#ifndef BITINO_C_H_
#define BITINO_C_H_

#include <stddef.h>
#include <stdint.h>

/* Symbols of the shared library (libbitino_c): exported; everything else in it is hidden. */
#if defined(_WIN32)
#if defined(BITINO_C_BUILD_DLL)
#define BF_API __declspec(dllexport)
#else
#define BF_API
#endif
#elif defined(__GNUC__)
#define BF_API __attribute__((visibility("default")))
#else
#define BF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BF_C_API_VERSION 2

typedef struct bf_model bf_model;
typedef struct bf_session bf_session;

BF_API const char* bf_version(void);
BF_API const char* bf_cpu_features(void);
BF_API const char* bf_last_error(void);

/* Loads a .bitino; verify_hash != 0 checks the data-section SHA-256. */
BF_API int bf_model_load(const char* path, int verify_hash, bf_model** out);
/* Parses an in-memory .bitino image (copied: the caller's buffer may be freed after the call), e.g. an Android
 * asset. */
BF_API int bf_model_load_bytes(const void* data, size_t size, int verify_hash, bf_model** out);
BF_API void bf_model_free(bf_model* model);
/* The model's manifest (provenance, input preprocessing, ...) as a JSON object; owned by the model. */
BF_API const char* bf_model_manifest_json(bf_model* model);

/* num_threads >= 1. stage: NULL or "" = all tiers, otherwise a bitino_bench stage name. */
BF_API int bf_session_create(bf_model* model, int num_threads, const char* stage, bf_session** out);
BF_API void bf_session_free(bf_session* session);

BF_API size_t bf_session_input_size(bf_session* session, int index);   /* elements (f32) */
BF_API size_t bf_session_output_size(bf_session* session, int index);  /* elements (f32) */
/* Shape of input `index`: writes min(rank, max_dims) dims, returns the rank (0 for a bad index). */
BF_API int bf_session_input_shape(bf_session* session, int index, int64_t* dims, int max_dims);
BF_API int bf_session_num_outputs(bf_session* session);

/* Copies `count` floats into input 0 and runs. */
BF_API int bf_session_run(bf_session* session, const float* input, size_t count);
/* Copies output `out_index` (`count` floats) into dst. */
BF_API int bf_session_get_output(bf_session* session, int out_index, float* dst, size_t count);

/* ------------------------------------------------------------------ version 2 */

BF_API int bf_c_api_version(void); /* BF_C_API_VERSION of the library */

/* Element types reported in bf_tensor_info.dtype (BF_DTYPE_BIT: bit-packed along the last dimension). */
enum { BF_DTYPE_F32 = 0, BF_DTYPE_I32 = 1, BF_DTYPE_I8 = 2, BF_DTYPE_U8 = 3, BF_DTYPE_BIT = 4 };
#define BF_MAX_RANK 8

typedef struct bf_tensor_info {
  const char* name; /* graph tensor name; valid while the session lives */
  int32_t dtype;    /* BF_DTYPE_* */
  int32_t rank;
  int64_t shape[BF_MAX_RANK];
  uint64_t bytes; /* storage size of the buffer */
} bf_tensor_info;

/* JSON object with the model's provenance manifest (as written by the converter). */
BF_API const char* bf_model_manifest(bf_model* model);

/* Like bf_session_create, plus an optional CPU affinity: the caller thread is pinned to affinity[0] and
 * worker i to affinity[i % num_affinity] (Linux/Android; elsewhere pinning is not supported and is
 * counted as an affinity error). affinity may be NULL when num_affinity == 0. */
BF_API int bf_session_create_ex(bf_model* model, int num_threads, const char* stage, const int* affinity,
                                int num_affinity, bf_session** out);

BF_API int bf_session_num_inputs(bf_session* session);
BF_API int bf_session_input_info(bf_session* session, int index, bf_tensor_info* out);
BF_API int bf_session_output_info(bf_session* session, int index, bf_tensor_info* out);

/* The session's own buffers (no copies): write inputs, call bf_session_invoke, read outputs. Re-query the
 * output pointer after each invoke; both stay valid until the session is freed. NULL on a bad index. */
BF_API void* bf_session_input_data(bf_session* session, int index);
BF_API const void* bf_session_output_data(bf_session* session, int index);

/* Copies `bytes` bytes (must equal the input's storage size) into input `index`. */
BF_API int bf_session_set_input(bf_session* session, int index, const void* data, size_t bytes);

/* Runs the graph on the current contents of the input buffers. */
BF_API int bf_session_invoke(bf_session* session);

/* Timing loop inside the library, on the current inputs: `warmup` untimed runs, then `iterations` runs,
 * each timed alone (steady clock, milliseconds) into samples_ms[i]. Times Session::run() only. */
BF_API int bf_session_benchmark(bf_session* session, int warmup, int iterations, double* samples_ms);

/* JSON object describing the prepared session: stage, threads, kernel counts, plan rewrites,
 * prepare time, activation/prepacked bytes, affinity errors. */
BF_API const char* bf_session_describe(bf_session* session);

#ifdef __cplusplus
}
#endif

#endif /* BITINO_C_H_ */
