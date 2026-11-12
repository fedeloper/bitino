// JNI bridge of the Bitino Android library (io.github.fedeloper.bitino.NativeLib) over the stable C API
// (runtime/include/bitino/bitino_c.h). Handles are bf_model* / bf_session* cast to jlong; every failure becomes
// a Java exception carrying bf_last_error().
#include <jni.h>

#include <cstdint>
#include <vector>

#include "bitino/bitino_c.h"

namespace {

void throw_java(JNIEnv* env, const char* cls_name, const char* msg) {
  jclass cls = env->FindClass(cls_name);
  if (cls) env->ThrowNew(cls, msg);
}

void throw_runtime(JNIEnv* env, const char* msg) { throw_java(env, "java/lang/IllegalStateException", msg); }

bf_session* session_of(jlong h) { return reinterpret_cast<bf_session*>(h); }

// Per-thread staging buffers: the C API copies from / into plain float pointers.
thread_local std::vector<float> g_in, g_out;

}  // namespace

#define BITINO_JNI(ret, name) extern "C" JNIEXPORT ret JNICALL Java_io_github_fedeloper_bitino_NativeLib_##name

BITINO_JNI(jstring, version)(JNIEnv* env, jclass) { return env->NewStringUTF(bf_version()); }

BITINO_JNI(jstring, cpuFeatures)(JNIEnv* env, jclass) { return env->NewStringUTF(bf_cpu_features()); }

BITINO_JNI(jlong, loadFile)(JNIEnv* env, jclass, jstring path, jboolean verify) {
  const char* p = env->GetStringUTFChars(path, nullptr);
  bf_model* m = nullptr;
  int rc = bf_model_load(p, verify ? 1 : 0, &m);
  env->ReleaseStringUTFChars(path, p);
  if (rc) {
    throw_java(env, "java/io/IOException", bf_last_error());
    return 0;
  }
  return reinterpret_cast<jlong>(m);
}

BITINO_JNI(jlong, loadBytes)(JNIEnv* env, jclass, jbyteArray data, jboolean verify) {
  jsize n = env->GetArrayLength(data);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  env->GetByteArrayRegion(data, 0, n, reinterpret_cast<jbyte*>(buf.data()));
  bf_model* m = nullptr;
  if (bf_model_load_bytes(buf.data(), buf.size(), verify ? 1 : 0, &m)) {
    throw_java(env, "java/io/IOException", bf_last_error());
    return 0;
  }
  return reinterpret_cast<jlong>(m);
}

BITINO_JNI(jstring, manifest)(JNIEnv* env, jclass, jlong model) {
  return env->NewStringUTF(bf_model_manifest_json(reinterpret_cast<bf_model*>(model)));
}

BITINO_JNI(void, freeModel)(JNIEnv*, jclass, jlong model) { bf_model_free(reinterpret_cast<bf_model*>(model)); }

BITINO_JNI(jlong, createSession)(JNIEnv* env, jclass, jlong model, jint threads) {
  bf_session* s = nullptr;
  if (bf_session_create(reinterpret_cast<bf_model*>(model), threads, "all", &s)) {
    throw_java(env, "java/lang/IllegalArgumentException", bf_last_error());
    return 0;
  }
  return reinterpret_cast<jlong>(s);
}

BITINO_JNI(void, freeSession)(JNIEnv*, jclass, jlong s) { bf_session_free(session_of(s)); }

BITINO_JNI(jlongArray, inputShape)(JNIEnv* env, jclass, jlong s) {
  int64_t dims[8];
  int rank = bf_session_input_shape(session_of(s), 0, dims, 8);
  if (rank > 8) rank = 8;
  jlongArray out = env->NewLongArray(rank);
  static_assert(sizeof(jlong) == sizeof(int64_t), "jlong");
  env->SetLongArrayRegion(out, 0, rank, reinterpret_cast<const jlong*>(dims));
  return out;
}

BITINO_JNI(jint, numOutputs)(JNIEnv*, jclass, jlong s) { return bf_session_num_outputs(session_of(s)); }

BITINO_JNI(jint, outputSize)(JNIEnv*, jclass, jlong s, jint index) {
  return static_cast<jint>(bf_session_output_size(session_of(s), index));
}

// Runs one inference on `input` (the session's input 0, f32) and copies output 0 into `output`.
BITINO_JNI(void, run)(JNIEnv* env, jclass, jlong s, jfloatArray input, jfloatArray output) {
  jsize n = env->GetArrayLength(input);
  g_in.resize(static_cast<size_t>(n));
  env->GetFloatArrayRegion(input, 0, n, g_in.data());
  if (bf_session_run(session_of(s), g_in.data(), g_in.size())) {
    throw_java(env, "java/lang/IllegalArgumentException", bf_last_error());
    return;
  }
  if (output == nullptr) return;
  jsize m = env->GetArrayLength(output);
  g_out.resize(static_cast<size_t>(m));
  if (bf_session_get_output(session_of(s), 0, g_out.data(), g_out.size())) {
    throw_runtime(env, bf_last_error());
    return;
  }
  env->SetFloatArrayRegion(output, 0, m, g_out.data());
}

// Copies output `index` of the last run into `output` (its length must equal the output size).
BITINO_JNI(void, getOutput)(JNIEnv* env, jclass, jlong s, jint index, jfloatArray output) {
  jsize m = env->GetArrayLength(output);
  g_out.resize(static_cast<size_t>(m));
  if (bf_session_get_output(session_of(s), index, g_out.data(), g_out.size())) {
    throw_runtime(env, bf_last_error());
    return;
  }
  env->SetFloatArrayRegion(output, 0, m, g_out.data());
}
