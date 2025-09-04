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

#include "src/pipelines/ocr/pipeline.h"

struct PaddleOCRParams {
  std::optional<std::string> text_detection_model_name = std::nullopt;
  std::optional<std::string> text_detection_model_dir = std::nullopt;
  std::optional<std::string> text_recognition_model_name = std::nullopt;
  std::optional<std::string> text_recognition_model_dir = std::nullopt;
  std::optional<int> text_recognition_batch_size = std::nullopt;
  std::optional<int> text_det_limit_side_len = std::nullopt;
  std::optional<std::string> text_det_limit_type = std::nullopt;
  std::optional<float> text_det_thresh = std::nullopt;
  std::optional<float> text_det_box_thresh = std::nullopt;
  std::optional<float> text_det_unclip_ratio = std::nullopt;
  std::optional<std::vector<int>> text_det_input_shape = std::nullopt;
  std::optional<float> text_rec_score_thresh = std::nullopt;
  std::optional<std::vector<int>> text_rec_input_shape = std::nullopt;
  std::optional<std::string> lang = std::nullopt;
  std::optional<std::string> ocr_version = std::nullopt;
  std::optional<std::string> vis_font_dir = std::nullopt;
  std::optional<std::string> device = std::nullopt;
  std::string precision = "fp32";
  int cpu_threads = 8;
  int thread_num = 1;
  std::optional<Utility::PaddleXConfigVariant> paddlex_config = std::nullopt;
};

class PaddleOCR {
public:
  PaddleOCR(const PaddleOCRParams &params = PaddleOCRParams());

  std::vector<std::unique_ptr<BaseCVResult>> Predict(const std::string &input) {
    std::vector<std::string> inputs = {input};
    return Predict(inputs);
  };
  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input);

  void CreatePipeline();
  Status CheckParams();
  static OCRPipelineParams ToOCRPipelineParams(const PaddleOCRParams &from);

private:
  PaddleOCRParams params_;
  std::unique_ptr<BasePipeline> pipeline_infer_;
};

