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

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace PaddleOCR
{

    // Abstract base class for detection interfaces
    class DetectorInterface
    {
    public:
        virtual ~DetectorInterface() = default;

        virtual void Run(const cv::Mat &img,
                         std::vector<std::vector<std::vector<int>>> &boxes,
                         std::vector<double> &times) noexcept = 0;
    };

    // Abstract base class for recognition interfaces
    class RecognizerInterface
    {
    public:
        virtual ~RecognizerInterface() = default;

        virtual void Run(const std::vector<cv::Mat> &img_list,
                         std::vector<std::string> &rec_texts,
                         std::vector<float> &rec_text_scores,
                         std::vector<double> &times) noexcept = 0;
    };

    // Factory class to create appropriate detector based on framework
    class DetectorFactory
    {
    public:
        static std::unique_ptr<DetectorInterface> CreateDetector(
            const std::string &framework,
            const std::string &model_dir,
            const bool &use_gpu,
            const int &gpu_id,
            const int &gpu_mem,
            const int &cpu_math_library_num_threads,
            const bool &use_mkldnn,
            const std::string &limit_type,
            const int &limit_side_len,
            const double &det_db_thresh,
            const double &det_db_box_thresh,
            const double &det_db_unclip_ratio,
            const std::string &det_db_score_mode,
            const bool &use_dilation,
            const bool &use_tensorrt,
            const std::string &precision,
            const std::string &device = "CPU");
    };

    // Factory class to create appropriate recognizer based on framework
    class RecognizerFactory
    {
    public:
        static std::unique_ptr<RecognizerInterface> CreateRecognizer(
            const std::string &framework,
            const std::string &model_dir,
            const bool &use_gpu,
            const int &gpu_id,
            const int &gpu_mem,
            const int &cpu_math_library_num_threads,
            const bool &use_mkldnn,
            const std::string &label_path,
            const bool &use_tensorrt,
            const std::string &precision,
            const int &rec_batch_num,
            const int &rec_img_h,
            const int &rec_img_w,
            const std::string &device = "CPU");
    };

} // namespace PaddleOCR
