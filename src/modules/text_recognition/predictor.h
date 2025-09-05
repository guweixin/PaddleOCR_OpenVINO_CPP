// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "processors.h"
#include "src/base/base_batch_sampler.h"
#include "src/base/base_cv_result.h"
#include "src/base/base_predictor.h"
#include "src/common/processors.h"

struct TextRecPredictorResult {
  std::string input_path = "";
  cv::Mat input_image;
  std::string rec_text = "";
  float rec_score = 0.0;
  std::string vis_font = "";
};

struct TextRecPredictorParams {
  std::optional<std::string> model_name = std::nullopt;
  std::optional<std::string> model_dir = std::nullopt;
  std::optional<std::string> lang = std::nullopt;
  std::optional<std::string> ocr_version = std::nullopt;
  std::optional<std::string> vis_font_dir = std::nullopt;
  std::optional<std::string> device = std::nullopt;
  std::string precision = "fp32";
  int cpu_threads = 8;
  int batch_size = 1;
  std::optional<std::vector<int>> input_shape = std::nullopt;
};

class TextRecPredictor : public BasePredictor {
public:
  TextRecPredictor(const TextRecPredictorParams &params);

  std::vector<TextRecPredictorResult> PredictorResult() const {
    return predictor_result_vec_;
  };

  void ResetResult() override { predictor_result_vec_.clear(); };

  Status Build();

  std::vector<std::unique_ptr<BaseCVResult>>
  Process(std::vector<cv::Mat> &batch_data) override;

  Status CheckRecModelParams();

private:
  std::unordered_map<std::string, std::unique_ptr<CTCLabelDecode>> post_op_;
  std::vector<TextRecPredictorResult> predictor_result_vec_;
  std::unique_ptr<OpenVinoInfer> infer_ptr_;
  TextRecPredictorParams params_;
  int input_index_ = 0;
};

