// Minimal JNI bridge over the Bitino C API (see Bitino.java).
#include <jni.h>

#include <vector>

#include "bitino/bitino_c.h"

namespace {

void throw_java(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls) env->ThrowNew(cls, msg);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_bitino_Bitino_nativeCreate(JNIEnv* env, jclass, jstring path, jint threads) {
  const char* p = env->GetStringUTFChars(path, nullptr);
  bf_model* model = nullptr;
  int rc = bf_model_load(p, 1, &model);
  env->ReleaseStringUTFChars(path, p);
  if (rc) {
    throw_java(env, bf_last_error());
    return 0;
  }
  bf_session* session = nullptr;
  if (bf_session_create(model, threads, "all", &session)) {
    bf_model_free(model);  // the session keeps its own reference only on success
    throw_java(env, bf_last_error());
    return 0;
  }
  bf_model_free(model);  // safe: the session holds a shared reference to the model
  return reinterpret_cast<jlong>(session);
}

JNIEXPORT jfloatArray JNICALL Java_org_bitino_Bitino_nativeRun(JNIEnv* env, jclass, jlong handle, jfloatArray input) {
  auto* s = reinterpret_cast<bf_session*>(handle);
  jsize n = env->GetArrayLength(input);
  std::vector<float> in(static_cast<size_t>(n));
  env->GetFloatArrayRegion(input, 0, n, in.data());
  if (bf_session_run(s, in.data(), in.size())) {
    throw_java(env, bf_last_error());
    return nullptr;
  }
  size_t out_n = bf_session_output_size(s, 0);
  std::vector<float> out(out_n);
  if (bf_session_get_output(s, 0, out.data(), out_n)) {
    throw_java(env, bf_last_error());
    return nullptr;
  }
  jfloatArray result = env->NewFloatArray(static_cast<jsize>(out_n));
  env->SetFloatArrayRegion(result, 0, static_cast<jsize>(out_n), out.data());
  return result;
}

JNIEXPORT void JNICALL Java_org_bitino_Bitino_nativeDestroy(JNIEnv*, jclass, jlong handle) {
  bf_session_free(reinterpret_cast<bf_session*>(handle));
}

}  // extern "C"
