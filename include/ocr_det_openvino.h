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

#include <fstream>
#include <include/postprocess_op.h>
#include <include/preprocess_op.h>
#include <iostream>
#include <memory>
#include <openvino/openvino.hpp>

namespace PaddleOCR
{

    class DBDetectorOpenVINO
    {
    public:
        explicit DBDetectorOpenVINO(const std::string &model_dir, const std::string &device,
                                    const std::string &limit_type,
                                    const int &limit_side_len, const double &det_db_thresh,
                                    const double &det_db_box_thresh,
                                    const double &det_db_unclip_ratio,
                                    const std::string &det_db_score_mode,
                                    const bool &use_dilation) noexcept
        {
            this->device_ = device;
            this->limit_type_ = limit_type;
            this->limit_side_len_ = limit_side_len;
            this->det_db_thresh_ = det_db_thresh;
            this->det_db_box_thresh_ = det_db_box_thresh;
            this->det_db_unclip_ratio_ = det_db_unclip_ratio;
            this->det_db_score_mode_ = det_db_score_mode;
            this->use_dilation_ = use_dilation;
            this->use_limit_side_len_ = (limit_type == "limit_max" || limit_type == "limit_min");

            this->mean_ = {0.485, 0.456, 0.406};
            this->scale_ = {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};
            this->is_scale_ = true;

            LoadModel(model_dir);
        }

        // Load OpenVINO model
        void LoadModel(const std::string &model_path) noexcept;

        // Run inference on image
        void Run(cv::Mat &img,
                 std::vector<std::vector<std::vector<int>>> &boxes,
                 std::vector<double> &times);

        // Print final detection memory transfer statistics
        void PrintFinalDetectionStats();

    private:
        ov::Core core_;
        ov::CompiledModel compiled_model_;
        ov::InferRequest infer_request_;

        std::string device_;
        std::string limit_type_;
        int limit_side_len_;
        double det_db_thresh_;
        double det_db_box_thresh_;
        double det_db_unclip_ratio_;
        std::string det_db_score_mode_;
        bool use_dilation_;
        bool use_limit_side_len_;

        std::vector<float> mean_;
        std::vector<float> scale_;
        bool is_scale_;

        // GPU optimization members
        bool use_gpu_buffers_;
        void* ocl_context_;
        void* ocl_queue_;

        // pre-process
        ResizeImgType0 resize_op_;
        Normalize normalize_op_;
        Permute permute_op_;

        // post-process
        DBPostProcessor post_processor_;
    };

} // namespace PaddleOCR
