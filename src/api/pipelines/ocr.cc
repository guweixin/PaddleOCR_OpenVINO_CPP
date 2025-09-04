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

#include "ocr.h"

#include "src/utils/args.h"
#include "src/utils/yaml_config.h"

#define COPY_PARAMS(field) to.field = from.field;

PaddleOCR::PaddleOCR(const PaddleOCRParams &params) : params_(params) {
  auto status = CheckParams();
  if (!status.ok()) {
    INFOE("Init paddleOCR fail : %s", status.ToString().c_str());
    exit(-1);
  }
  CreatePipeline();
};
std::vector<std::unique_ptr<BaseCVResult>>
PaddleOCR::Predict(const std::vector<std::string> &input) {
  return pipeline_infer_->Predict(input);
}
void PaddleOCR::CreatePipeline() {
  pipeline_infer_ = std::unique_ptr<BasePipeline>(
      new OCRPipeline(ToOCRPipelineParams(params_)));
}

Status PaddleOCR::CheckParams() {
  if (!params_.text_detection_model_dir.has_value()) {
    return Status::NotFoundError("Require text detection model dir.");
  }
  if (!params_.text_recognition_model_dir.has_value()) {
    return Status::NotFoundError("Require text recognition model_dir.");
  }
  return Status::OK();
}

OCRPipelineParams PaddleOCR::ToOCRPipelineParams(const PaddleOCRParams &from) {
  OCRPipelineParams to;
  COPY_PARAMS(text_detection_model_name)
  COPY_PARAMS(text_detection_model_dir)
  COPY_PARAMS(text_recognition_model_name)
  COPY_PARAMS(text_recognition_model_dir)
  COPY_PARAMS(text_recognition_batch_size)
  COPY_PARAMS(text_det_limit_side_len)
  COPY_PARAMS(text_det_limit_type)
  COPY_PARAMS(text_det_thresh)
  COPY_PARAMS(text_det_box_thresh)
  COPY_PARAMS(text_det_unclip_ratio)
  COPY_PARAMS(text_det_input_shape)
  COPY_PARAMS(text_rec_score_thresh)
  COPY_PARAMS(text_rec_input_shape)
  COPY_PARAMS(lang)
  COPY_PARAMS(ocr_version)
  COPY_PARAMS(vis_font_dir)
  COPY_PARAMS(device)
  COPY_PARAMS(precision)
  COPY_PARAMS(cpu_threads)
  COPY_PARAMS(thread_num)
  COPY_PARAMS(paddlex_config)
  return to;
}

