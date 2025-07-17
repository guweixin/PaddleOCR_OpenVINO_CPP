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

#include <include/postprocess_op.h>
#include <include/preprocess_op.h>
#include <openvino/openvino.hpp>

#include <fstream>
#include <iostream>
#include <memory>

namespace PaddleOCR
{

    class CRNNRecognizerOpenVINO
    {
    public:
        explicit CRNNRecognizerOpenVINO(const std::string &model_dir, const std::string &device,
                                        const std::string &label_path,
                                        const int &rec_batch_num, const int &rec_img_h,
                                        const int &rec_img_w) noexcept
        {
            this->device_ = device;
            this->rec_batch_num_ = rec_batch_num;
            this->rec_img_h_ = rec_img_h;
            this->rec_img_w_ = rec_img_w;

            this->mean_ = {0.5, 0.5, 0.5};
            this->scale_ = {1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5};
            this->is_scale_ = true;

            LoadModel(model_dir);
            LoadLabelList(label_path);
        }

        // Load OpenVINO model
        void LoadModel(const std::string &model_path) noexcept;

        // Load label list
        void LoadLabelList(const std::string &label_path) noexcept;

        // Run inference on image batch
        void Run(const std::vector<cv::Mat> &img_list,
                 std::vector<std::string> &rec_texts,
                 std::vector<float> &rec_text_scores,
                 std::vector<double> &times) noexcept;

    private:
        ov::Core core_;
        ov::CompiledModel compiled_model_;
        ov::InferRequest infer_request_;

        std::string device_;
        int rec_batch_num_;
        int rec_img_h_;
        int rec_img_w_;

        std::vector<float> mean_;
        std::vector<float> scale_;
        bool is_scale_;

        std::vector<std::string> label_list_;

        // pre-process
        CrnnResizeImg resize_op_;
        Normalize normalize_op_;
        PermuteBatch permute_op_;
    };

} // namespace PaddleOCR
