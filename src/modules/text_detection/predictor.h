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

struct TextDetPredictorResult {
  std::string input_path = "";
  cv::Mat input_image;
  std::vector<std::vector<cv::Point2f>> dt_polys = {};
  std::vector<float> dt_scores = {};
};

struct TextDetPredictorParams {
  std::optional<std::string> model_name = std::nullopt;
  std::optional<std::string> model_dir = std::nullopt;
  std::optional<std::string> device = std::nullopt;
  std::string precision = "fp32";
  int cpu_threads = 8;
  int batch_size = 1;
  std::optional<int> limit_side_len = std::nullopt;
  std::optional<std::string> limit_type = std::nullopt;
  std::optional<int> max_side_limit = std::nullopt;
  std::optional<float> thresh = std::nullopt;
  std::optional<float> box_thresh = std::nullopt;
  std::optional<float> unclip_ratio = std::nullopt;
  std::optional<std::vector<int>> input_shape = std::nullopt;
};

class TextDetPredictor : public BasePredictor {
public:
  TextDetPredictor(const TextDetPredictorParams &params);

  std::vector<TextDetPredictorResult> PredictorResult() const {
    return predictor_result_vec_;
  };

  void ResetResult() override { predictor_result_vec_.clear(); };

  Status Build();

  std::vector<std::unique_ptr<BaseCVResult>>
  Process(std::vector<cv::Mat> &batch_data) override;

private:
  TextDetPredictorParams params_;
  std::unordered_map<std::string, std::unique_ptr<DBPostProcess>> post_op_;
  std::vector<TextDetPredictorResult> predictor_result_vec_;
  std::unique_ptr<PaddleInfer> infer_ptr_;
  int input_index_ = 0;
};

