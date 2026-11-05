/* Bitino stable C API (for JNI and other FFI users).
 *
 * All functions are thread-compatible: distinct handles may be used from
 * different threads; one session handle must not be run concurrently.
 * Errors: functions return 0 on success and non-zero on failure; the message
 * of the last failure on the calling thread is available via bf_last_error().
 */
#ifndef BITINO_C_H_
#define BITINO_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_model bf_model;
typedef struct bf_session bf_session;

const char* bf_version(void);
const char* bf_cpu_features(void);
const char* bf_last_error(void);

/* Loads a .bitino; verify_hash != 0 checks the data-section SHA-256. */
int bf_model_load(const char* path, int verify_hash, bf_model** out);
void bf_model_free(bf_model* model);

/* num_threads >= 1. stage: NULL or "" = all tiers, otherwise a bitino_bench stage name. */
int bf_session_create(bf_model* model, int num_threads, const char* stage, bf_session** out);
void bf_session_free(bf_session* session);

size_t bf_session_input_size(bf_session* session, int index);   /* elements (f32) */
size_t bf_session_output_size(bf_session* session, int index);  /* elements (f32) */
int bf_session_num_outputs(bf_session* session);

/* Copies `count` floats into input `index`, runs, and copies output `out_index`. */
int bf_session_run(bf_session* session, const float* input, size_t count);
int bf_session_get_output(bf_session* session, int out_index, float* dst, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* BITINO_C_H_ */
