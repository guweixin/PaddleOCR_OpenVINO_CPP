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

#include "src/api/pipelines/ocr.h"
#include "src/utils/args.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

static const std::unordered_set<std::string> SUPPORT_MODE_PIPELINE = {
    "ocr",
};

void PrintErrorInfo(const std::string &msg, const std::string &main_mode = "") {
  auto join_modes =
      [](const std::unordered_set<std::string> &modes) -> std::string {
    std::string result;
    for (const auto &mode : modes) {
      result += mode + ", ";
    }
    if (!result.empty()) {
      result.pop_back();
      result.pop_back();
    }
    return result;
  };

  std::string pipeline_modes = join_modes(SUPPORT_MODE_PIPELINE);

  INFOE("%s%s", msg.c_str(),
        main_mode.empty() ? "" : (": \"" + main_mode + "\"").c_str());
  INFO("==========================================");
  INFO("Supported pipeline : [%s]", pipeline_modes.c_str());
  INFO("==========================================");
}

PaddleOCRParams GetPipelineParams() {
  PaddleOCRParams ocr_params;
  if (!text_detection_model_name.empty()) {
    ocr_params.text_detection_model_name = text_detection_model_name;
  }
  if (!text_detection_model_dir.empty()) {
    ocr_params.text_detection_model_dir = text_detection_model_dir;
  }
  if (!text_recognition_model_name.empty()) {
    ocr_params.text_recognition_model_name = text_recognition_model_name;
  }
  if (!text_recognition_model_dir.empty()) {
    ocr_params.text_recognition_model_dir = text_recognition_model_dir;
  }
  if (!text_recognition_batch_size.empty()) {
    ocr_params.text_recognition_batch_size =
        std::stoi(text_recognition_batch_size);
  }
  if (!text_det_limit_side_len.empty()) {
    ocr_params.text_det_limit_side_len =
        std::stoi(text_det_limit_side_len);
  }
  if (!text_det_limit_type.empty()) {
    ocr_params.text_det_limit_type = text_det_limit_type;
  }
  if (!text_det_thresh.empty()) {
    ocr_params.text_det_thresh = std::stof(text_det_thresh);
  }
  if (!text_det_box_thresh.empty()) {
    ocr_params.text_det_box_thresh = std::stof(text_det_box_thresh);
  }
  if (!text_det_unclip_ratio.empty()) {
    ocr_params.text_det_unclip_ratio = std::stof(text_det_unclip_ratio);
  }
  if (!text_det_input_shape.empty()) {
    ocr_params.text_det_input_shape =
        YamlConfig::SmartParseVector(text_det_input_shape).vec_int;
  }
  if (!text_rec_score_thresh.empty()) {
    ocr_params.text_rec_score_thresh = std::stof(text_rec_score_thresh);
  }
  if (!text_rec_input_shape.empty()) {
    ocr_params.text_rec_input_shape =
        YamlConfig::SmartParseVector(text_rec_input_shape).vec_int;
  }
  if (!device.empty()) {
    ocr_params.device = device;
  }
  if (!precision.empty()) {
    ocr_params.precision = precision;
  }
  if (!cpu_threads.empty()) {
    ocr_params.cpu_threads = std::stoi(cpu_threads);
  }
  if (!thread_num.empty()) {
    ocr_params.thread_num = std::stoi(thread_num);
  }
  return ocr_params;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (input.empty()) {
    INFOE("Require input, such as ./build/ppocr ocr --input "
          "your_image_path [--param1] [--param2] [...]");
    exit(-1);
  }
  std::string main_mode = "";
  if (argc > 1) {
    main_mode = argv[1];
    if (SUPPORT_MODE_PIPELINE.count(main_mode) == 0) {
      PrintErrorInfo("ERROR: Unsupported pipeline", main_mode);
      exit(-1);
    }
  } else {
    PrintErrorInfo(
        "Must provide pipeline name, such as ./build/ppocr "
        "ocr [--param1] [--param2] [...]");
    exit(-1);
  }
  
  auto params = GetPipelineParams();
  auto outputs = PaddleOCR(params).Predict(input);
  
  for (auto &output : outputs) {
    output->Print();
    output->SaveToImg(save_path);
    output->SaveToJson(save_path);
  }
  return 0;
}
