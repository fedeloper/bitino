// Bitino public C++ API.
//
//   auto model = bitino::Model::load("model.bitino");
//   bitino::SessionOptions opts;
//   opts.num_threads = 4;
//   bitino::Session session(model, opts);
//   std::memcpy(session.input(0).raw(), image, session.input(0).bytes());
//   session.run();
//   const float* logits = session.output(0).data<float>();
#pragma once

#include "bitino/cpu_features.h"
#include "bitino/error.h"
#include "bitino/model.h"
#include "bitino/session.h"
#include "bitino/tensor.h"

#define BITINO_VERSION "0.1.0"
