// Image-feature operators (hand-crafted input channels computed inside the model, e.g. the FFT-magnitude and
// local-binary-pattern channels of the "Faster Than Lies" deepfake detector). Contracts:
// docs/operators_image_features.md; numpy counterparts: python/bitino/image_ops.py.
//
//   ImageGrayU8(x[N,H,W,3] f32)        -> [N,H,W,1] f32 integers 0..255:
//       u_c = trunc(clamp(f32(x_c * scale), 0, 255)); y = (sum_c u_c * coeffs[c] + 2^(shift-1)) >> shift
//   FftLogMagnitude(x[N,H,W,1] f32)    -> [N,H,W,1] f32: f32((mul * log(|FFT2(x)| + eps)) / div), fftshift-ed
//   LocalBinaryPattern(x[N,H,W,1] f32) -> [N,H,W,1] f32: f32(lbp / div), scikit-image semantics ('default', 'uniform')
#include "kernels/image_features_common.h"
#include "ops/op_defs.h"
#include "ops/ops_common.h"

namespace bitino {
namespace {

using Ins = std::vector<const TensorInfo*>;
using Outs = std::vector<TensorInfo*>;

const TensorInfo& image_input(const Ins& ins, const char* op, int64_t channels) {
  BF_CHECK(!ins.empty() && ins[0], op, ": missing input");
  const auto& x = *ins[0];
  BF_CHECK(x.dtype == DType::kF32 && x.shape.size() == 4 && x.shape[3] == channels, op, " expects f32 [N,H,W,", channels,
           "], got ", shape_str(x.shape));
  BF_CHECK(x.shape[1] > 0 && x.shape[2] > 0, op, ": empty image");
  return x;
}

void set_single_channel(const TensorInfo& x, Outs& outs) {
  BF_CHECK(!outs.empty(), "missing output");
  outs[0]->dtype = DType::kF32;
  outs[0]->shape = {x.shape[0], x.shape[1], x.shape[2], 1};
}

void infer_gray(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = image_input(ins, "ImageGrayU8", 3);
  auto c = n.attrs.get_ints("coeffs", {});
  BF_CHECK(c.size() == 3, "ImageGrayU8: coeffs must have 3 integers");
  int64_t shift = n.attrs.get_int("shift", -1);
  BF_CHECK(shift >= 1 && shift <= 24, "ImageGrayU8: shift must be in [1, 24]");
  for (auto v : c) BF_CHECK(v >= 0 && v < (int64_t{1} << 24), "ImageGrayU8: coefficients must be in [0, 2^24)");
  n.attrs.get_number("scale", 255.0);
  set_single_channel(x, outs);
}

void infer_fft(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = image_input(ins, "FftLogMagnitude", 1);
  for (int64_t len : {x.shape[1], x.shape[2]})
    for (int p : imgf::fft_factors(len))
      BF_CHECK(p <= imgf::kMaxRadix, "FftLogMagnitude: length ", len, " has the prime factor ", p, " (> ", imgf::kMaxRadix, ")");
  BF_CHECK(n.attrs.get_number("eps", 1e-9) > 0, "FftLogMagnitude: eps must be > 0");
  n.attrs.get_number("mul", 20.0);
  BF_CHECK(n.attrs.get_number("div", 255.0) != 0, "FftLogMagnitude: div must be non-zero");
  int64_t shift = n.attrs.get_int("shift", 1);
  BF_CHECK(shift == 0 || shift == 1, "FftLogMagnitude: shift must be 0 or 1");
  set_single_channel(x, outs);
}

void infer_lbp(const Node& n, const Ins& ins, Outs& outs) {
  const auto& x = image_input(ins, "LocalBinaryPattern", 1);
  BF_CHECK(n.attrs.contains("rp") && n.attrs.contains("cp"), "LocalBinaryPattern needs rp and cp");
  const auto& rp = n.attrs["rp"];
  const auto& cp = n.attrs["cp"];
  BF_CHECK(rp.is_array() && cp.is_array() && rp.size() == cp.size() && rp.size() >= 1 && rp.size() <= 24,
           "LocalBinaryPattern: rp / cp must be arrays of the same length (1..24)");
  for (size_t i = 0; i < rp.size(); ++i)
    BF_CHECK(std::fabs(rp[i].as_number()) < 1e6 && std::fabs(cp[i].as_number()) < 1e6, "LocalBinaryPattern: offset out of range");
  std::string m = n.attrs.get_string("method", "uniform");
  BF_CHECK(m == "uniform" || m == "default", "LocalBinaryPattern: method must be 'uniform' or 'default'");
  BF_CHECK(n.attrs.get_number("div", 1.0) != 0, "LocalBinaryPattern: div must be non-zero");
  set_single_channel(x, outs);
}

}  // namespace

void register_image_feature_ops(OpRegistry& reg) {
  reg.add_op({"ImageGrayU8", infer_gray});
  reg.add_op({"FftLogMagnitude", infer_fft});
  reg.add_op({"LocalBinaryPattern", infer_lbp});
}

}  // namespace bitino
