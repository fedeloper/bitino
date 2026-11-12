package io.github.fedeloper.bitino

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * The Kotlin preprocessing must equal the Python pipelines that produced the models' evaluation tensors, bit for bit.
 * Fixtures: android/tools/make_preprocess_fixtures.py (PIL 12.2 / torchvision 0.21). Optional real-image check:
 * BITINO_EVAL_RGB_DIR (android/tools/dump_eval_rgb.py).
 */
class PreprocessingTest {
    // manifest definitions, verbatim from converted models (ab-bnn-imagenet-*, omz-*, IR-Net family CIFAR-10,
    // bitorch-*, bhvit-*, larq-zoo-*, bnext-*)
    private val tvDef = "torchvision 0.21.0+cu124 / PIL 12.2.0: PIL.Image.open(f).convert('RGB') -> transforms.Resize(256) " +
        "[InterpolationMode.BILINEAR, antialias=True, short side 256] -> transforms.CenterCrop(224) -> transforms.ToTensor() " +
        "-> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)); CHW -> HWC (AB-BNN Inference/ImageNet.ipynb transform_test)"
    private val omzDef = "OMZ accuracy-check.yml (resnet50-binary-0001 / resnet18-xnor-binary-onnx-0001), Pillow 12.2.0: " +
        "PIL.Image.open(f).convert('RGB') -> resize short side to 256 with Accuracy Checker scale_greater (int truncation), " +
        "Image.BILINEAR -> center crop 224 (offsets int(round((H-224)/2)), Python round) -> RGB->BGR; values 0..255 as " +
        "float32, NHWC. Mean subtraction (103.5,116.3,123.7 BGR) is inside the IR."
    private val cifarDef = "torchvision ToTensor() then Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201]) " +
        "in float32, CHW->HWC (ShuaiZ1037/bnn-xnor-bireal train.py:117-122)"
    private val bitorchDef = "torchvision 0.21.0+cu124 / PIL 12.2.0 / torch 2.6.0+cu124: PIL.Image.open(f).convert('RGB') -> " +
        "transforms.Resize(256) [InterpolationMode.BILINEAR, antialias=True] -> transforms.CenterCrop(224) -> " +
        "transforms.ToTensor() -> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)); CHW -> HWC"
    private val bhvitDef = "torchvision 0.20.1+cu121 / PIL 12.2.0 / timm 1.0.27: IMRL/BHViT datasets.build_transform(is_train=False, " +
        "input_size=224) = Compose(     Resize(size=256, interpolation=bicubic, max_size=None, antialias=True)     " +
        "CenterCrop(size=(224, 224))     ToTensor()     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) ) " +
        "applied to PIL.Image.open(f).convert('RGB'); CHW -> HWC"
    private val larqDef = "larq_zoo.training.data.preprocess_image_bytes(is_training=False)"
    private val bnextDef = "torchvision 0.21.0+cu124 / PIL 12.2.0 / torch 2.6.0+cu124: PIL.Image.open(f).convert('RGB') -> " +
        "transforms.Resize(235) [short side, InterpolationMode.BICUBIC] -> transforms.CenterCrop(224) -> uint8 -> float32 " +
        "(x - 255*mean)/(255*std), timm IMAGENET_DEFAULT mean/std; CHW -> HWC."

    private fun specs(): Map<String, Preprocessing> = mapOf(
        "torchvision_imagenet" to Preprocessing.parse(tvDef)!!,
        "omz_bgr" to Preprocessing.parse(omzDef)!!,
        "torchvision_bicubic" to Preprocessing.parse(bhvitDef)!!,
        "cifar_norm" to Preprocessing.parse(cifarDef, 32)!!,
    )

    /** Same generator as android/tools/make_preprocess_fixtures.py synth(). */
    private fun synth(w: Int, h: Int, seed: Int): IntArray {
        var s = (seed.toLong() * 0x9E3779B9L + 1L).toInt()
        val px = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            s = s xor (s shl 13)
            s = s xor (s ushr 17)
            s = s xor (s shl 5)
            val r = (x * 255 / maxOf(w - 1, 1)) xor (s and 0x0F)
            val g = (y * 255 / maxOf(h - 1, 1)) xor ((s ushr 4) and 0x0F)
            val b = ((x / 8 + y / 8) % 2) * 200 + ((s ushr 24) and 0x37)
            px[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return px
    }

    private fun sha256(v: FloatArray): String {
        val bb = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(v)
        return MessageDigest.getInstance("SHA-256").digest(bb.array()).joinToString("") { "%02x".format(it) }
    }

    @Test
    fun parsesManifestDefinitions() {
        val tv = Preprocessing.parse(tvDef)!!
        assertEquals(256, tv.resizeShortSide); assertEquals(224, tv.cropSize)
        assertEquals(Preprocessing.Filter.BILINEAR, tv.filter); assertEquals(255f, tv.divisor)
        assertArrayEquals(floatArrayOf(0.485f, 0.456f, 0.406f), tv.mean, 0f)
        assertArrayEquals(floatArrayOf(0.229f, 0.224f, 0.225f), tv.std, 0f)
        val omz = Preprocessing.parse(omzDef)!!
        assertTrue(omz.bgr); assertEquals(1f, omz.divisor); assertEquals(256, omz.resizeShortSide)
        assertArrayEquals(floatArrayOf(0f, 0f, 0f), omz.mean, 0f)
        val bitorch = Preprocessing.parse(bitorchDef)!!
        assertArrayEquals(floatArrayOf(0.229f, 0.224f, 0.255f), bitorch.std, 0f)   // bitorch's std, as published
        val bhvit = Preprocessing.parse(bhvitDef)!!
        assertEquals(Preprocessing.Filter.BICUBIC, bhvit.filter); assertEquals(224, bhvit.cropSize)
        val cifar = Preprocessing.parse(cifarDef, 32)!!
        assertEquals(0, cifar.resizeShortSide); assertEquals(32, cifar.cropSize)
        assertNull(Preprocessing.parse(larqDef))
        assertNull(Preprocessing.parse(bnextDef))
    }

    @Test
    fun matchesPythonFixtures() {
        val lines = javaClass.classLoader!!.getResourceAsStream("preprocess_fixtures.txt")!!.bufferedReader().readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
        assertTrue(lines.size >= 30)
        val specs = specs()
        val bad = mutableListOf<String>()
        for (line in lines) {
            val f = line.split(' ')
            val p = specs.getValue(f[0])
            val w = f[1].toInt(); val h = f[2].toInt(); val seed = f[3].toInt()
            val out = FloatArray(p.outputSize)
            p.apply(synth(w, h, seed), w, h, out)
            val sum = out.fold(0.0) { a, v -> a + v }
            if (sha256(out) != f[4]) bad += "${f[0]} ${w}x$h seed $seed: sum ${"%.6f".format(sum)} vs ${f[5]}"
        }
        assertTrue("mismatches:\n" + bad.joinToString("\n"), bad.isEmpty())
    }

    @Test
    fun rotationThenPreprocessEqualsPreprocessOfRotatedImage() {
        val w = 37; val h = 23
        val px = synth(w, h, 3)
        val r90 = Pixels.rotate(px, w, h, 90)
        // pixel (x, y) of the source lands at (h - 1 - y, x) in the h x w result
        assertEquals(px[5 * w + 7], r90[7 * h + (h - 1 - 5)])
        val back = Pixels.rotate(Pixels.rotate(Pixels.rotate(r90, h, w, 90), w, h, 90), h, w, 90)
        assertArrayEquals(px, back)
        assertArrayEquals(Pixels.rotate(px, w, h, 270), Pixels.rotate(Pixels.rotate(r90, h, w, 90), w, h, 90))
    }

    @Test
    fun topK() {
        val s = floatArrayOf(0.1f, 3f, -2f, 3f, 7f, 0f)
        assertEquals(listOf(4 to 7f, 1 to 3f, 3 to 3f), Bitino.topK(s, 3))
        assertEquals(6, Bitino.topK(s, 10).size)
        val p = Bitino.softmax(floatArrayOf(1f, 2f, 3f))
        assertEquals(1.0, p.fold(0.0) { a, v -> a + v }, 1e-6)
    }

    /** Real images (ImageNet eval128): Kotlin(PIL-decoded RGB) == the eval set's tensors, every byte. */
    @Test
    fun matchesEvalSetTensors() {
        val dir = System.getenv("BITINO_EVAL_RGB_DIR")
        assumeTrue("BITINO_EVAL_RGB_DIR not set", dir != null)
        val images = File(dir, "images.txt").readLines().filter { it.isNotBlank() }.map { it.split(' ') }
        for (set in File(dir, "sets.txt").readLines().filter { it.isNotBlank() }) {
            val (key, tensorFile, definition) = set.split('\t')
            val p = Preprocessing.parse(definition)
            assertNotNull("$key: definition not parsed", p)
            val ref = ByteBuffer.wrap(File(tensorFile).readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            assertEquals(images.size * p!!.outputSize, ref.remaining())
            val out = FloatArray(p.outputSize)
            val expect = FloatArray(p.outputSize)
            var mismatched = 0
            for ((i, im) in images.withIndex()) {
                val w = im[0].toInt(); val h = im[1].toInt()
                val rgb = File(dir, im[2]).readBytes()
                val px = IntArray(w * h) { k ->
                    (0xFF shl 24) or ((rgb[3 * k].toInt() and 0xFF) shl 16) or ((rgb[3 * k + 1].toInt() and 0xFF) shl 8) or
                        (rgb[3 * k + 2].toInt() and 0xFF)
                }
                p.apply(px, w, h, out)
                ref.position(i * p.outputSize); ref.get(expect)
                if (!out.contentEquals(expect)) {
                    mismatched++
                    val n = out.indices.count { out[it] != expect[it] }
                    println("$key sample $i (${w}x$h): $n of ${out.size} values differ")
                }
            }
            println("$key: ${images.size - mismatched}/${images.size} samples bit-identical to $tensorFile")
            assertEquals("$key: samples differing from the eval tensors", 0, mismatched)
        }
    }
}
