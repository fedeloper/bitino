# Image-feature operators

Hand-crafted input channels computed inside the model, so the whole method (not only the network) runs in the
runtime on the device. First user: the "Faster Than Lies" deepfake detector (Lanzino, Fontana, Diko, Marini,
Cinque, CVPRW 2024, arXiv:2406.04932; `python/bitino/adapters/torch_ftl_deepfake.py`), whose BNext backbone
reads RGB plus an FFT-magnitude channel and a local-binary-pattern channel through a 5 → 3 adapter convolution.

| op | input | output | attributes |
|---|---|---|---|
| `ImageGrayU8` | f32 `[N,H,W,3]` | f32 `[N,H,W,1]`, integers 0..255 | `coeffs` (3 ints), `shift`, `scale` (255) |
| `FftLogMagnitude` | f32 `[N,H,W,1]` | f32 `[N,H,W,1]` | `eps` (1e-9), `mul` (20), `div` (255), `shift` (1) |
| `LocalBinaryPattern` | f32 `[N,H,W,1]` | f32 `[N,H,W,1]` | `rp`, `cp` (sample offsets), `method` (`uniform` / `default`), `div` |

Implementations: OpDefs `runtime/ops/op_defs_image_features.cc`; reference kernels
`runtime/kernels/reference/image_features_reference.cc`; optimized kernels `runtime/kernels/image_features_opt.cc`
(tier `neon` on AArch64 with 2-lane float64 NEON, tier `binary_scalar` elsewhere with the same code on a portable
2-lane struct); shared arithmetic `runtime/kernels/image_features_common.h`; numpy executor
`python/bitino/image_ops.py`. **All three are bit-identical by construction** (same IEEE double operations in the
same order, `-ffp-contract=off`, no libm transcendental), so stage `neon` and every other stage reproduce the numpy
executor exactly; there is no float32 approximation and no tolerance.

## ImageGrayU8

`u_c = trunc(clamp(f32(x_c · scale), 0, 255))` (NaN → 0), `y = (Σ_c u_c · coeffs[c] + 2^(shift−1)) >> shift`.

The source computes `cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)` on RGB data. `image * 255` is a
float32 product and `astype(uint8)` truncates; for inputs `k · fl32(1/255)` (torchvision `ToDtype(scale=True)`) the
truncation returns `k` for all 256 values, in float32 and in float64. OpenCV 4.x's 8-bit BGR2GRAY is the 15-bit
fixed-point formula with coefficients (B, G, R) = (3735, 19235, 9798) applied to channels (0, 1, 2), so on RGB data
the red channel gets the blue weight: `coeffs = [3735, 19235, 9798], shift = 15`. Checked against `cv2.cvtColor` on
all 2^24 pixel values with OpenCV 4.1.2 and 4.12.0 (the older 14-bit coefficients (1868, 9617, 4899) differ on 43,864
of them).

## FftLogMagnitude

`y[r, c] = f32((mul · log(|X[k1, k2]| + eps)) / div)` with `X = FFT2(x)` (forward, `exp(−2πi·kn/N)`), stored at
`(r, c) = ((k1 + H/2) mod H, (k2 + W/2) mod W)` when `shift = 1` (`np.fft.fftshift`), `|X| = sqrt(re² + im²)`.
The source computes `20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9)`, then `/ 255` in float64 and
`.float()`.

Algorithm (the contract, since floating-point FFTs differ in rounding):

* 1-D transforms: mixed-radix Stockham decimation in frequency, radix order 4, 4, …, then one 2, then odd primes
  ascending (224 = 4·4·2·7). Stage with radix `p`, current length `n`, stride `s`, `m = n/p`:
  `a_q = x[k + s(j + mq)]`, `b_r = DFT_p(a)_r`, `y[k + s(pj + r)] = b_r` if `j = 0` or `r = 0`, else `b_r · W_n^{jr}`.
  Radix 2 / 4 butterflies are the usual sums (the radix-4 `−i` rotation is exact); odd radices sum
  `a_0 + Σ_{q≥1} a_q · W_p^{qr}` in `q` order (no multiply when the exponent is 0). Complex products are
  `(ac − bd, ad + bc)`. Prime factors > 7 are rejected at plan time.
* Twiddles `W_N^t = (cos, −sin)(2πt/N)` from a fixed polynomial: exact integer octant reduction of `8t/N`, angle
  `φ = (π/4 · r)/N ∈ [0, π/4]`, Taylor polynomials to degree 17 / 18 by Horner (error ≤ 1.3e-16 absolute against the
  exact values for N = 224; no libm, so every platform builds the same table).
* 2-D: every column along H (real input, imaginary part 0), then the rows `k1 = 0..H/2` along W; rows
  `k1 > H/2` are taken from the Hermitian mirror `|X[k1, k2]| = |X[H−k1, (W−k2) mod W]|` (real input).
* `log` is a fixed polynomial for positive normal doubles (`m·2^e`, `m ∈ [√½, √2)`, `s = (m−1)/(m+1)`,
  `2s + 2s·R(s²)` with the coefficients 1/(2k+1), fdlibm's split ln2): ≤ 2 ulp from glibc's `log`.

Against the source (numpy's pocketfft + `np.log`): identical float32 channel values on natural images (8 + 10 checked
images, 0 differing values of 401,408 / 501,760), differences only where `|X|` is at rounding level (a constant
image: 12 values with `|X| ≤ 4e-10`, max 2.1e-2 after the log); per-model counts are in each
`conversion_report.json` (`feature_check`).

The optimized kernel transforms blocks of 8 columns (pass 1) / 8 rows (pass 2) at once with the lanes across
independent transforms, keeps the half spectrum transposed, and evaluates magnitude, log and the division on
2-lane vectors; the mirrored half is a store of the same float.

## LocalBinaryPattern

scikit-image `local_binary_pattern(image, P, R, method)` (0.25, `_texture.pyx`) with
`bilinear_interpolation(mode 'C', cval 0)`, in double: sample `i` at `(r + rp[i], c + cp[i])`,
`minr = floor`, `maxr = ceil`, `dr = rr − minr` (same for columns),
`top = (1−dc)·v(minr,minc) + dc·v(minr,maxc)`, `bottom = (1−dc)·v(maxr,minc) + dc·v(maxr,maxc)`,
`t = (1−dr)·top + dr·bottom` (0 outside the image), `bit_i = (t − x[r,c] ≥ 0)`.
`uniform`: `changes = Σ_{i<P−1} [bit_i ≠ bit_{i+1}]` (not circular, as scikit-image), `lbp = Σ bit_i` if
`changes ≤ 2` else `P + 1`; `default`: `Σ bit_i · 2^i`. Output `f32(lbp / div)`.

`rp`, `cp` are stored in the node exactly as scikit-image computes them (`np.round(∓R·sin / R·cos(2πi/P), 5)`;
`image_ops.lbp_offsets`), so the runtime needs no trigonometry: for P = 3, R = 6 they are `rp = [−0, −5.19615,
5.19615]`, `cp = [6, −3, −3]`. Identical to scikit-image on every checked image (0 mismatches).

The optimized kernel copies the image into a zero-padded double buffer so that the neighbours of consecutive
columns are consecutive in memory (requires `floor(c + cp) = c + floor(cp)` and the same for `ceil` for every
column, checked in `supports()`; true for all offsets that are not within an ulp of an integer) and evaluates the
same formula on 2-lane vectors, row by row (rows in parallel).

## Running the detector (demo)

```bash
# phone / host: binary PPM frames (e.g. ffmpeg -i clip.mp4 -vf fps=10 frames/%05d.ppm), a directory of them, or a
# raw float32 N x 224 x 224 x 3 file preprocessed like the authors' test loader
bitino_deepfake --model model.bitino --affinity 7 --warmup 10 frames/
# host, from JPEG / PNG files (exact torchvision preprocessing when torchvision is importable)
./bitino deepfake artifacts/ftl-deepfake-bnext-t-cocofake-unfrozen/model.bitino photo.jpg frames/
```

It prints, per frame, the logit, P(fake) = 1 / (1 + e^logit), the label, and the time of the host-side resize (PPM
only: short side 252 bilinear without antialiasing + center crop, an approximation of torchvision's resize), of the
`features.*` nodes and of the network (session per-node profile), then the median per-frame time and fps. Phone timing
of the stages: `tools/ftl_deepfake_submodels.py` splits the model into `model_features.bitino` and
`model_network.bitino` for `bitino_bench`; numbers in the development log (not published).

## Tests

* `tests/python/test_image_features.py`: polynomial sin/cos / log vs libm, the FFT vs `numpy.fft` (lengths 1–224,
  radices 2–7), numpy executor vs C++ bit-exact at stages `reference`, `neon` (1 and 3 threads) and `all` for six
  image sizes / LBP settings including NaN and out-of-range pixels, and (when OpenCV / scikit-image are importable)
  equality with `cv2.cvtColor`, `skimage.feature.local_binary_pattern` and the numpy-FFT channel.
* `tests/cpp/test_image_features.cc`: gray formula, a long-double direct DFT, a hand-computed LBP row, and the
  optimized kernels bit-identical to the reference (host, emulated arm64, OnePlus Open, Moto e13).
