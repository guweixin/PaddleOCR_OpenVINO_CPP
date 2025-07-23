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

#include <include/ocr_det_openvino.h>
#include <include/utility.h>
#include <include/args.h>
#include <include/data_saver.h>

#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace PaddleOCR
{

    void DBDetectorOpenVINO::LoadModel(const std::string &model_path) noexcept
    {
        try
        {
            std::cout << "[OpenVINO] Loading detection model from: " << model_path << std::endl;
            std::cout << "[OpenVINO] Target device: " << device_ << std::endl;

            // Check if model file exists
            std::ifstream file(model_path);
            if (!file.good())
            {
                std::cerr << "[ERROR] Model file not found: " << model_path << std::endl;
                std::cerr << "[ERROR] For OpenVINO, you need .xml model file" << std::endl;
                exit(1);
            }
            file.close();

            // Load the model
            auto model = core_.read_model(model_path);
            std::cout << "[OpenVINO] Model file loaded successfully" << std::endl;

            // Configure device-specific settings
            // ov::AnyMap config;
            ov::AnyMap config = {{"CACHE_DIR", "./cache"},
                                 {"PERFORMANCE_HINT", "LATENCY"}};
            if (device_ == "CPU")
            {
                // CPU-specific configurations
                config["CPU_RUNTIME_CACHE_CAPACITY"] = "10";
            }
            // Compile the model for the specified device
            std::cout << "[OpenVINO] Compiling model for device: " << device_ << std::endl;
            compiled_model_ = core_.compile_model(model, device_, config);

            // Create inference request
            infer_request_ = compiled_model_.create_infer_request();

            std::cout << "[OpenVINO] Model loaded successfully for device: " << device_ << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Failed to load OpenVINO model: " << e.what() << std::endl;
            exit(1);
        }
    }

    void DBDetectorOpenVINO::Run(const cv::Mat &img,
                                 std::vector<std::vector<std::vector<int>>> &boxes,
                                 std::vector<double> &times) noexcept
    {
        try
        {
            float ratio_h{};
            float ratio_w{};

            cv::Mat srcimg;
            cv::Mat resize_img;
            img.copyTo(srcimg);

            auto preprocess_start = std::chrono::steady_clock::now();

            // Preprocessing - special handling for NPU device
            if (device_ == "NPU")
            {
                // NPU specific preprocessing: resize to 960x960 with aspect ratio preserved and white padding
                int target_size = 960;
                int original_h = img.rows;
                int original_w = img.cols;

                // Calculate scale to fit the longer side to 960
                float scale = static_cast<float>(target_size) / std::max(original_h, original_w);

                int new_h = static_cast<int>(original_h * scale);
                int new_w = static_cast<int>(original_w * scale);

                // Resize while preserving aspect ratio
                cv::Mat scaled_img;
                cv::resize(img, scaled_img, cv::Size(new_w, new_h));

                // Create 960x960 canvas with white background
                resize_img = cv::Mat(target_size, target_size, CV_8UC3, cv::Scalar(255, 255, 255));

                // Calculate position to center the scaled image
                int start_x = (target_size - new_w) / 2;
                int start_y = (target_size - new_h) / 2;

                // Copy scaled image to the center of the canvas
                cv::Rect roi(start_x, start_y, new_w, new_h);
                scaled_img.copyTo(resize_img(roi));

                // Calculate ratios for postprocessing
                ratio_h = static_cast<float>(target_size) / static_cast<float>(original_h);
                ratio_w = static_cast<float>(target_size) / static_cast<float>(original_w);
            }
            else
            {
                // Standard preprocessing for CPU/GPU
                this->resize_op_.Run(img, resize_img, this->limit_type_,
                                     this->limit_side_len_, ratio_h, ratio_w, false);
            }

            this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);

            std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);

            this->permute_op_.Run(resize_img, input.data());

            auto preprocess_end = std::chrono::steady_clock::now();

            // Inference with OpenVINO
            auto inference_start = std::chrono::steady_clock::now();

            // Get input tensor
            auto input_tensor = infer_request_.get_input_tensor();

            // Set input shape and copy data
            input_tensor.set_shape({1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols)});

            float *input_data = input_tensor.data<float>();
            std::memcpy(input_data, input.data(), input.size() * sizeof(float));

            // Run inference
            infer_request_.infer();

            // Get output tensor
            auto output_tensor = infer_request_.get_output_tensor();
            auto output_shape = output_tensor.get_shape();

            int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
            std::vector<float> out_data(out_num);

            float *output_data = output_tensor.data<float>();
            std::memcpy(out_data.data(), output_data, out_num * sizeof(float));

            // Save debug data if enabled
            if (FLAGS_save_debug_data)
            {
                static int det_inference_counter = 0;

                // Save input tensor data
                std::vector<size_t> input_shape = {1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols)};
                std::vector<size_t> output_shape_vec(output_shape.begin(), output_shape.end());

                DataSaver::SaveDetectionData(input, input_shape, out_data, output_shape_vec, det_inference_counter);

                // // Save preprocessed image
                // std::string prep_img_filename = "../../debug_data/cpp_preprocessed_img_" + std::to_string(det_inference_counter) + ".npy";
                // DataSaver::SaveMatAsNpy(resize_img, prep_img_filename);

                det_inference_counter++;
            }

            auto inference_end = std::chrono::steady_clock::now();

            // Postprocessing
            auto postprocess_start = std::chrono::steady_clock::now();

            int n2 = static_cast<int>(output_shape[2]);
            int n3 = static_cast<int>(output_shape[3]);
            int n = n2 * n3;

            std::cout << "[DEBUG] Original image size: " << img.cols << "x" << img.rows << std::endl;
            std::cout << "[DEBUG] Resized image size: " << resize_img.cols << "x" << resize_img.rows << std::endl;
            std::cout << "[DEBUG] Model output size: " << n3 << "x" << n2 << std::endl;
            std::cout << "[DEBUG] srcimg size passed to BoxesFromBitmap: " << srcimg.cols << "x" << srcimg.rows << std::endl;

            std::vector<float> pred(n, 0.0);
            std::vector<unsigned char> cbuf(n, ' ');

            for (int i = 0; i < n; ++i)
            {
                pred[i] = out_data[i];
                cbuf[i] = (unsigned char)((out_data[i]) * 255);
            }

            cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
            cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

            const double threshold = this->det_db_thresh_ * 255;
            const double maxvalue = 255;
            cv::Mat bit_map;
            cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

            if (this->use_dilation_)
            {
                cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
                cv::dilate(bit_map, bit_map, dila_ele);
            }

            boxes = std::move(post_processor_.BoxesFromBitmap(
                pred_map, bit_map, srcimg.cols, srcimg.rows));

            // Print boxes for debugging
            std::cout << "[DEBUG] BoxesFromBitmap returned " << boxes.size() << " boxes:" << std::endl;
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                std::cout << "  Box " << i << ": ";
                for (size_t j = 0; j < boxes[i].size(); ++j)
                {
                    if (j > 0)
                        std::cout << " -> ";
                    std::cout << "(" << boxes[i][j][0] << "," << boxes[i][j][1] << ")";
                }
                std::cout << std::endl;
            }

            post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
            
            // Print boxes after FilterTagDetRes for debugging
            std::cout << "[DEBUG] After FilterTagDetRes, final boxes:" << std::endl;
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                std::cout << "  Final Box " << i << ": ";
                for (size_t j = 0; j < boxes[i].size(); ++j)
                {
                    if (j > 0)
                        std::cout << " -> ";
                    std::cout << "(" << boxes[i][j][0] << "," << boxes[i][j][1] << ")";
                }
                std::cout << std::endl;
            }

            // Save image with text boxes if debug mode is enabled
            if (FLAGS_save_debug_data)
            {
                static int debug_image_counter = 0;
                DataSaver::SaveImageWithTextBoxes(srcimg, boxes, debug_image_counter);
                debug_image_counter++;
            }
            
            auto postprocess_end = std::chrono::steady_clock::now();

            // Calculate timing
            std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
            times.emplace_back(preprocess_diff.count() * 1000);
            std::chrono::duration<float> inference_diff = inference_end - inference_start;
            times.emplace_back(inference_diff.count() * 1000);
            std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
            times.emplace_back(postprocess_diff.count() * 1000);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Exception in DBDetectorOpenVINO::Run: " << e.what() << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(3, 0.0);
            // Exit to avoid further processing errors
            exit(1);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Unknown exception in DBDetectorOpenVINO::Run" << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(3, 0.0);
            // Exit to avoid further processing errors
            exit(1);
        }
    }

} // namespace PaddleOCR
