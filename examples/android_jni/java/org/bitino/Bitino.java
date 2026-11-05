package org.bitino;

/** Minimal Java wrapper around the Bitino native runtime (libbitino_jni.so). */
public final class Bitino implements AutoCloseable {
  static {
    System.loadLibrary("bitino_jni");
  }

  private long handle;

  /** Loads a .bitino (SHA-256 verified) and creates a session with {@code threads} CPU workers. */
  public Bitino(String modelPath, int threads) {
    handle = nativeCreate(modelPath, threads);
  }

  /** Runs one inference on a preprocessed NHWC float input; returns output 0 (e.g. logits). */
  public float[] run(float[] input) {
    if (handle == 0) throw new IllegalStateException("closed");
    return nativeRun(handle, input);
  }

  @Override
  public void close() {
    if (handle != 0) {
      nativeDestroy(handle);
      handle = 0;
    }
  }

  private static native long nativeCreate(String modelPath, int threads);
  private static native float[] nativeRun(long handle, float[] input);
  private static native void nativeDestroy(long handle);
}
