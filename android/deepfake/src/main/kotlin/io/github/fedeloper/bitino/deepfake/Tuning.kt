package io.github.fedeloper.bitino.deepfake

/**
 * Recommended P(fake) threshold and frames in a row per detector, from tools/screen_threshold_study.py: the app's
 * screen pipeline simulated on the OnePlus Open, 200 COCOFake val2014 real/fake pairs, threshold and frame count chosen
 * on one half for the most fakes caught at <= 5 % false alarms, measured on the other half
 * (reports/screen_threshold/<model>.md). Detectors without data of their kind keep the untuned default.
 */
internal object Tuning {
    /** [resize]: the images' preprocessing, Resize(resize) + CenterCrop 224 (252 = the authors'). */
    class Rec(val threshold: Float, val frames: Int, val note: String, val resize: Int = 252)

    private val table = mapOf(
        // full-resolution capture, Resize 224: test half 100 % of fakes caught, 2.5 % false alarms (AUC 0.998-1.000 for
        // images shown 360-1800 px wide)
        "ftl-deepfake-bnext-t-cocofake-unfrozen" to Rec(0.36f, 2, "tuned on COCOFake", resize = 224),
        // full-resolution capture, the authors' Resize 252: weak detector (AUC ~0.8), test half 23 % caught, 0.3 % false alarms
        "ftl-deepfake-bnext-t-cocofake" to Rec(0.85f, 2, "tuned on COCOFake (weak detector)"),
    )
    val DEFAULT = Rec(0.70f, 3, "not tuned")

    /** [name]: the model file name without extension (e.g. "ftl-deepfake-bnext-t-cocofake"). */
    fun forModel(name: String): Rec = table[name] ?: DEFAULT
}
