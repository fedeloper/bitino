package io.github.fedeloper.bitino

/** Pixel-array helpers (pure Kotlin, no Android dependency). */
public object Pixels {
    /**
     * Rotates a row-major [width] x [height] pixel array clockwise by [degrees] (0, 90, 180, 270). The result is
     * [height] x [width] for 90 / 270. [dst]: optional output buffer (reused when large enough).
     */
    @JvmStatic
    @JvmOverloads
    public fun rotate(src: IntArray, width: Int, height: Int, degrees: Int, dst: IntArray? = null): IntArray {
        val d = ((degrees % 360) + 360) % 360
        require(d % 90 == 0) { "rotation must be a multiple of 90 degrees" }
        val out = if (dst != null && dst.size >= width * height) dst else IntArray(width * height)
        if (d == 0) {
            System.arraycopy(src, 0, out, 0, width * height)
            return out
        }
        when (d) {
            90 -> for (y in 0 until height) for (x in 0 until width) out[x * height + (height - 1 - y)] = src[y * width + x]
            180 -> for (i in 0 until width * height) out[width * height - 1 - i] = src[i]
            270 -> for (y in 0 until height) for (x in 0 until width) out[(width - 1 - x) * height + y] = src[y * width + x]
        }
        return out
    }
}
