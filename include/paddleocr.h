// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <include/utility.h>

namespace PaddleOCR
{
  // Global variable to store current image path for debugging
  extern thread_local std::string g_current_image_path;

  // Utility function to get the correct model path for different frameworks
  std::string getModelPath(const std::string &model_dir, const std::string &model_type);

  class PPOCR
  {
  public:
    explicit PPOCR() noexcept;
    virtual ~PPOCR();

    std::vector<std::vector<OCRPredictResult>>
    ocr(const std::vector<cv::Mat> &img_list, bool det = true, bool rec = true) noexcept;
    std::vector<OCRPredictResult> ocr(const cv::Mat &img, bool det = true,
                                      bool rec = true) noexcept;

    void reset_timer() noexcept;
    void benchmark_log(int img_num) noexcept;
    void detailed_benchmark_log(int img_num) noexcept;

  protected:
    // Detailed timing information for detection
    // [resize, normalize, permute, inference, threshold, dilation, boxes_from_bitmap, filter_tag]
    std::vector<double> time_info_det_detailed = {0, 0, 0, 0, 0, 0, 0, 0};
    // Summary timing for detection [preprocess, inference, postprocess]
    std::vector<double> time_info_det = {0, 0, 0};
    
    // Detailed timing information for recognition
    // [resize, normalize, permute, inference, postprocess_decode]
    std::vector<double> time_info_rec_detailed = {0, 0, 0, 0, 0};
    // Summary timing for recognition [preprocess, inference, postprocess]
    std::vector<double> time_info_rec = {0, 0, 0};

    void det(const cv::Mat &img,
             std::vector<OCRPredictResult> &ocr_results) noexcept;
    void rec(const std::vector<cv::Mat> &img_list,
             std::vector<OCRPredictResult> &ocr_results) noexcept;

  private:
    struct PPOCR_PRIVATE;
    PPOCR_PRIVATE *pri_;
  };

} // namespace PaddleOCR
