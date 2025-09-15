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

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

#include "src/utils/status.h"
#include "openvino/openvino.hpp"
#include "src/utils/ilogger.h"
#include "src/utils/openvino_option.h"

// NPU recognition model sizes (used by OpenVinoInfer)
enum class NPURecModelSize { TINY = 0, SMALL = 1, MEDIUM = 2, BIG = 3, LARGE = 4 };

class OpenVinoInfer {
public:
  explicit OpenVinoInfer(const std::string &model_name,
                         const std::string &model_dir,
                         const std::string &model_file_prefix,
                         const OpenVinoOption &option);
  ~OpenVinoInfer() = default;
  
  StatusOr<std::vector<cv::Mat>>
  Apply(const std::vector<cv::Mat> &x);

  // Get device type for conditional logic
  std::string GetDeviceType() const { return option_.DeviceType(); }

  // Get the last NPU mapping ratios for coordinate restoration
  std::vector<std::pair<float, float>> GetLastNpuRatios() const { return last_npu_ratios_; }

  // Get NPU model input sizes (height, width) for all loaded models
  std::vector<std::pair<int, int>> GetNPURecInputSizes() const;

  private:
  std::string model_dir_;
  std::string model_file_prefix_;
  std::string model_name_;
  OpenVinoOption option_;
  
  ov::Core core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  
  // NPU-specific compiled models and infer requests
  std::unordered_map<NPURecModelSize, ov::CompiledModel> npu_compiled_models_;
  std::unordered_map<NPURecModelSize, ov::InferRequest> npu_infer_requests_;

  // NPU detection model (single model)
  ov::CompiledModel npu_detection_compiled_model_;
  ov::InferRequest npu_detection_infer_request_;

  Status Create();
  Status CheckRunMode();
  
private:
  // Storage for NPU coordinate mapping ratios (ratio_h, ratio_w) for each image in the last batch
  std::vector<std::pair<float, float>> last_npu_ratios_;
  
  // Model type identification
  bool is_detector_;
  bool is_recognizer_;
};

