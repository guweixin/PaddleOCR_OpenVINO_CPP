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

#include <include/ocr_rec_openvino.h>
#include <include/utility.h>
#include <include/args.h>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace PaddleOCR
{

    void CRNNRecognizerOpenVINO::LoadModels(const std::string &model_dir) noexcept
    {
        try
        {
            std::cout << "[OpenVINO] Loading recognition models from: " << model_dir << std::endl;
            std::cout << "[OpenVINO] Target device: " << device_ << std::endl;

            // Configure device-specific settings
            ov::AnyMap config = {{"PERFORMANCE_HINT", "LATENCY"}};
            if (device_ == "CPU")
            {
                // CPU-specific configurations
                config["CPU_RUNTIME_CACHE_CAPACITY"] = "0";
            }
            else if (device_ == "GPU" || device_ == "NPU")
            {
                config["INFERENCE_PRECISION_HINT"] = "f16";
                config["CACHE_DIR"] = "./openvino_cache";
            }

            if (device_ == "NPU")
            {
                // NPU: Load dual models
                std::string model_small_path, model_medium_path, model_big_path;

                // model_dir is a directory path, ensure it ends with a path separator
                std::string dir = model_dir;
                if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
                {
                    dir += "/";
                }
                model_small_path = dir + "inference_480_bs1.xml";
                model_medium_path = dir + "inference_800_bs1.xml";
                model_big_path = dir + "inference_1280_bs1.xml";

                // Load small model
                std::ifstream file_small(model_small_path);
                if (!file_small.good())
                {
                    std::cerr << "[ERROR] small model file not found: " << model_small_path << std::endl;
                    exit(1);
                }
                file_small.close();

                // Load medium model
                std::ifstream file_medium(model_medium_path);
                if (!file_medium.good())
                {
                    std::cerr << "[ERROR] medium model file not found: " << model_medium_path << std::endl;
                    exit(1);
                }
                file_medium.close();

                // Load big model
                std::ifstream file_big(model_big_path);
                if (!file_big.good())
                {
                    std::cerr << "[ERROR] big model file not found: " << model_big_path << std::endl;
                    exit(1);
                }
                file_big.close();

                // Load both models
                auto model_small = core_.read_model(model_small_path);
                auto model_medium = core_.read_model(model_medium_path);
                auto model_big = core_.read_model(model_big_path);
                std::cout << "[OpenVINO] Both NPU recognition model files loaded successfully" << std::endl;

                // Compile both models for NPU
                compiled_model_small_ = core_.compile_model(model_small, device_, config);
                std::cout << "[OpenVINO] model_small successfully" << std::endl;
                compiled_model_medium_ = core_.compile_model(model_medium, device_, config);
                std::cout << "[OpenVINO] model_medium successfully" << std::endl;
                compiled_model_big_ = core_.compile_model(model_big, device_, config);
                std::cout << "[OpenVINO] model_big successfully" << std::endl;
                // Create inference requests
                infer_request_small_ = compiled_model_small_.create_infer_request();
                infer_request_medium_ = compiled_model_medium_.create_infer_request();
                infer_request_big_ = compiled_model_big_.create_infer_request();

                std::cout << "[OpenVINO] Both NPU recognition models compiled successfully" << std::endl;
            }
            else
            {
                // CPU/GPU: Load single model (inference.xml)
                std::string model_path;

                // Check if model_dir ends with .xml (indicating it's a file path)
                if (model_dir.size() > 4 && model_dir.substr(model_dir.size() - 4) == ".xml")
                {
                    model_path = model_dir;
                }
                else
                {
                    // model_dir is a directory path, ensure it ends with a path separator
                    std::string dir = model_dir;
                    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
                    {
                        dir += "/";
                    }
                    model_path = dir + "inference.xml";
                }

                // Load single model
                std::ifstream file_check(model_path);
                if (!file_check.good())
                {
                    std::cerr << "[ERROR] Model file not found: " << model_path << std::endl;
                    exit(1);
                }
                file_check.close();

                auto model = core_.read_model(model_path);
                std::cout << "[OpenVINO] " << device_ << " recognition model file loaded successfully" << std::endl;

                // Compile model for CPU/GPU (use small slot for single model)
                compiled_model_small_ = core_.compile_model(model, device_, config);

                // Create inference request
                infer_request_small_ = compiled_model_small_.create_infer_request();

                std::cout << "[OpenVINO] " << device_ << " recognition model loaded successfully" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Failed to load OpenVINO recognition models: " << e.what() << std::endl;
            exit(1);
        }
    }

    void CRNNRecognizerOpenVINO::LoadLabelList(const std::string &label_path) noexcept
    {
        this->label_list_ = Utility::ReadDict(label_path);
        this->label_list_.emplace(this->label_list_.begin(), "#"); // blank char for ctc
        this->label_list_.emplace_back(" ");

        if (this->label_list_.empty())
        {
            std::cerr << "[ERROR] no label in " << label_path << std::endl;
            exit(1);
        }
    }

    void CRNNRecognizerOpenVINO::Run(const std::vector<cv::Mat> &img_list,
                                     std::vector<std::string> &rec_texts,
                                     std::vector<float> &rec_text_scores,
                                     std::vector<double> &times) noexcept
    {
        try
        {
            // Clear output vectors to avoid accumulation
            rec_texts.clear();
            rec_text_scores.clear();

            if (device_ == "NPU")
            {
                // NPU: Use dual-model processing with aspect ratio-based selection
                runNPUProcessing(img_list, rec_texts, rec_text_scores, times);
            }
            else
            {
                // CPU/GPU: Use original processing logic with single model
                runCPUGPUProcessing(img_list, rec_texts, rec_text_scores, times);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Exception in CRNNRecognizerOpenVINO::Run: " << e.what() << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(8, 0.0);
            // Exit to avoid further processing errors
            exit(1);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Unknown exception in CRNNRecognizerOpenVINO::Run" << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(8, 0.0);
            // Exit to avoid further processing errors
            exit(1);
        }
    }

    void CRNNRecognizerOpenVINO::runNPUProcessing(const std::vector<cv::Mat> &img_list,
                                                  std::vector<std::string> &rec_texts,
                                                  std::vector<float> &rec_text_scores,
                                                  std::vector<double> &times) noexcept
    {
        auto total_start = std::chrono::steady_clock::now();

        // Initialize timing variables
        double sort_time = 0.0;
        double resize_time = 0.0;
        double normalize_time = 0.0;
        double permute_time = 0.0;
        double inference_time = 0.0;
        double postprocess_time = 0.0;

        int img_num = static_cast<int>(img_list.size());

        // Create temporary vectors to store results in sorted order
        std::vector<std::string> temp_rec_texts;
        std::vector<float> temp_rec_text_scores;
        temp_rec_texts.reserve(img_num);
        temp_rec_text_scores.reserve(img_num);

        // Get input dimensions from both models
        auto small_input_shape = compiled_model_small_.input().get_shape();
        auto medium_input_shape = compiled_model_medium_.input().get_shape();
        auto big_input_shape = compiled_model_big_.input().get_shape();
        int small_model_width = static_cast<int>(small_input_shape[small_input_shape.size() - 1]);
        int medium_model_width = static_cast<int>(medium_input_shape[medium_input_shape.size() - 1]);
        int big_model_width = static_cast<int>(big_input_shape[big_input_shape.size() - 1]);
        int target_h = static_cast<int>(big_input_shape[big_input_shape.size() - 2]);

        // std::cout << "[NPU] Small model input width: " << small_model_width << std::endl;
        // std::cout << "[NPU] Medium model input width: " << medium_model_width << std::endl;
        // std::cout << "[NPU] Big model input width: " << big_model_width << std::endl;
        auto sort_start = std::chrono::steady_clock::now();
        // Calculate aspect ratio and sort indices for optimization
        std::vector<float> width_list;
        for (int i = 0; i < img_num; ++i)
        {
            width_list.emplace_back(static_cast<float>(img_list[i].cols) / static_cast<float>(img_list[i].rows));
        }
        std::vector<size_t> indices = Utility::argsort(width_list);
        auto sort_end = std::chrono::steady_clock::now();
        sort_time = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();

        // NPU uses batch size 1
        int batch_num = 1;

        // Calculate batches
        for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num)
        {
            int end_img_no = std::min(img_num, beg_img_no + batch_num);
            int batch_size = end_img_no - beg_img_no;

            // For NPU: group images by aspect ratio to determine model and preprocessing
            std::vector<cv::Mat> norm_img_batch;
            norm_img_batch.reserve(batch_size);

            // std::cout << "[NPU] Model input height: " << target_h << std::endl;
            // Determine which model to use based on aspect ratio
            int model_type = 0; // 0 for small; 1 for medium; 2 for big
            int target_w = small_model_width;

            // Check the aspect ratio of current batch to determine model
            float max_aspect_ratio = 0.0f;
            for (int idx = beg_img_no; idx < end_img_no; ++idx)
            {
                float aspect_ratio = width_list[indices[idx]];
                max_aspect_ratio = std::max(max_aspect_ratio, aspect_ratio);
            }

            // Select model based on maximum aspect ratio in the batch
            float small_threshold = static_cast<float>(small_model_width) / static_cast<float>(target_h);
            float medium_threshold = static_cast<float>(medium_model_width) / static_cast<float>(target_h);

            if (max_aspect_ratio <= small_threshold)
            {
                model_type = 0;
                target_w = small_model_width;
            }
            else if (max_aspect_ratio <= medium_threshold)
            {
                model_type = 1;
                target_w = medium_model_width;
            }
            else
            {
                model_type = 2;
                target_w = big_model_width;
            }

            // std::cout << "[NPU] Batch " << beg_img_no / batch_num + 1 << ": Using model_type " << model_type
            //           << " (target_w=" << target_w << ") for max_aspect_ratio=" << max_aspect_ratio << std::endl;

            auto resize_start = std::chrono::steady_clock::now();

            // NPU specific preprocessing with dynamic model selection
            for (int idx = beg_img_no; idx < end_img_no; ++idx)
            {
                cv::Mat resize_img;
                int original_h = img_list[indices[idx]].rows;
                int original_w = img_list[indices[idx]].cols;
                float aspect_ratio = static_cast<float>(original_w) / static_cast<float>(original_h);

                // Calculate scale to fit within target size while preserving aspect ratio
                float scale_h = static_cast<float>(target_h) / original_h;
                float scale_w = static_cast<float>(target_w) / original_w;
                float scale = std::min(scale_h, scale_w);

                int new_h = static_cast<int>(original_h * scale);
                int new_w = static_cast<int>(original_w * scale);

                // Resize while preserving aspect ratio
                cv::Mat scaled_img;
                cv::resize(img_list[indices[idx]], scaled_img, cv::Size(new_w, new_h));

                // Create canvas with white background
                resize_img = cv::Mat(target_h, target_w, CV_8UC3, cv::Scalar(255, 255, 255));

                int start_x, start_y;

                // Model-specific image placement strategy
                if (model_type == 2) // Big model
                {
                    // For big model: center both horizontally and vertically
                    start_x = (target_w - new_w) / 2;
                    start_y = (target_h - new_h) / 2;
                }
                else // Small or medium model
                {
                    // For small/medium model: left-align horizontally, center vertically
                    start_x = 0;
                    start_y = (target_h - new_h) / 2;
                }

                // Ensure we don't go out of bounds
                start_x = std::max(0, std::min(start_x, target_w - new_w));
                start_y = std::max(0, std::min(start_y, target_h - new_h));

                // Copy scaled image to the canvas
                cv::Rect roi(start_x, start_y, new_w, new_h);
                scaled_img.copyTo(resize_img(roi));

                // // ! just resize
                // cv::resize(img_list[indices[idx]], resize_img, cv::Size(target_w, target_h));

                auto normalize_start = std::chrono::steady_clock::now();
                this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);
                auto normalize_end = std::chrono::steady_clock::now();
                normalize_time += std::chrono::duration<double, std::milli>(normalize_end - normalize_start).count();

                norm_img_batch.push_back(resize_img);
            }

            auto resize_end = std::chrono::steady_clock::now();
            resize_time += std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

            // Prepare input data
            auto permute_start = std::chrono::steady_clock::now();
            std::vector<float> input(batch_size * 3 * target_h * target_w, 0.0f);
            this->permute_op_.Run(norm_img_batch, input.data());
            auto permute_end = std::chrono::steady_clock::now();
            permute_time += std::chrono::duration<double, std::milli>(permute_end - permute_start).count();

            // Inference with OpenVINO - choose appropriate model for NPU
            auto inference_start = std::chrono::steady_clock::now();
            ov::InferRequest *current_infer_request;
            if (model_type == 0)
            {
                current_infer_request = &infer_request_small_;
            }
            else if (model_type == 1)
            {
                current_infer_request = &infer_request_medium_;
            }
            else
            {
                current_infer_request = &infer_request_big_;
            }

            // ov::InferRequest *current_infer_request = use_small_model ? &infer_request_small_ : &infer_request_big_;

            // Get input tensor
            auto input_tensor = current_infer_request->get_input_tensor();
            input_tensor.set_shape({static_cast<size_t>(batch_size), 3,
                                    static_cast<size_t>(target_h),
                                    static_cast<size_t>(target_w)});

            // Copy input data and run inference
            float *input_data = input_tensor.data<float>();
            std::memcpy(input_data, input.data(), input.size() * sizeof(float));
            current_infer_request->infer();

            // Get output tensor
            auto output_tensor = current_infer_request->get_output_tensor();
            auto output_shape = output_tensor.get_shape();
            size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());

            std::vector<float> predict_batch(out_num);
            float *output_data = output_tensor.data<float>();
            std::memcpy(predict_batch.data(), output_data, out_num * sizeof(float));

            auto inference_end = std::chrono::steady_clock::now();
            inference_time += std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

            auto postprocess_start = std::chrono::steady_clock::now();

            // Process results for current batch
            for (int m = 0; m < batch_size; m++)
            {
                int start_idx = m * static_cast<int>(output_shape[1]) * static_cast<int>(output_shape[2]);
                int end_idx = (m + 1) * static_cast<int>(output_shape[1]) * static_cast<int>(output_shape[2]);

                std::vector<float> single_predict(predict_batch.begin() + start_idx,
                                                  predict_batch.begin() + end_idx);

                // CTC decode - based on original ocr_rec.cpp implementation
                std::string str_res;
                int argmax_idx;
                int last_index = 0;
                float score = 0.f;
                int count = 0;
                float max_value = 0.0f;

                for (int n = 0; n < static_cast<int>(output_shape[1]); ++n)
                {
                    // get idx
                    argmax_idx = int(Utility::argmax(
                        &single_predict[n * static_cast<int>(output_shape[2])],
                        &single_predict[(n + 1) * static_cast<int>(output_shape[2])]));

                    // get score
                    max_value = float(*std::max_element(
                        &single_predict[n * static_cast<int>(output_shape[2])],
                        &single_predict[(n + 1) * static_cast<int>(output_shape[2])]));

                    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
                    {
                        score += max_value;
                        count += 1;
                        str_res += this->label_list_[argmax_idx];
                    }
                    last_index = argmax_idx;
                }

                if (count > 0)
                {
                    score /= count;
                }
                else
                {
                    score = 0.0f;
                }

                if (!std::isnan(score) && !str_res.empty())
                {
                    temp_rec_texts.push_back(str_res);
                    temp_rec_text_scores.push_back(score);
                }
                else
                {
                    temp_rec_texts.push_back("");
                    temp_rec_text_scores.push_back(0.0f);
                }
            }

            auto postprocess_end = std::chrono::steady_clock::now();
            postprocess_time += std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        }

        // Reorder results back to original sequence
        rec_texts.resize(img_num);
        rec_text_scores.resize(img_num);
        for (int i = 0; i < img_num; ++i)
        {
            rec_texts[indices[i]] = temp_rec_texts[i];
            rec_text_scores[indices[i]] = temp_rec_text_scores[i];
        }

        auto total_end = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Store detailed timings (in milliseconds)
        times.clear();
        times.resize(8);
        times[0] = sort_time;                   // 排序时间
        times[1] = resize_time;                 // 图像resize时间
        times[2] = normalize_time;              // 图像归一化时间
        times[3] = permute_time;                // 维度变换时间
        times[4] = inference_time;              // 神经网络推理时间
        times[5] = postprocess_time;            // CTC解码时间
        times[6] = total_time - inference_time; // 除推理外的总时间
        times[7] = total_time;                  // 总时间
    }

    void CRNNRecognizerOpenVINO::runCPUGPUProcessing(const std::vector<cv::Mat> &img_list,
                                                     std::vector<std::string> &rec_texts,
                                                     std::vector<float> &rec_text_scores,
                                                     std::vector<double> &times) noexcept
    {
        auto total_start = std::chrono::steady_clock::now();

        // Initialize timing variables
        double sort_time = 0.0;
        double resize_time = 0.0;
        double normalize_time = 0.0;
        double permute_time = 0.0;
        double inference_time = 0.0;
        double postprocess_time = 0.0;

        int img_num = static_cast<int>(img_list.size());

        // Create temporary vectors to store results in sorted order
        std::vector<std::string> temp_rec_texts;
        std::vector<float> temp_rec_text_scores;
        temp_rec_texts.reserve(img_num);
        temp_rec_text_scores.reserve(img_num);

        auto sort_start = std::chrono::steady_clock::now();
        // Calculate aspect ratio and sort indices for optimization
        std::vector<float> width_list;
        for (int i = 0; i < img_num; ++i)
        {
            width_list.emplace_back(static_cast<float>(img_list[i].cols) / static_cast<float>(img_list[i].rows));
        }
        std::vector<size_t> indices = Utility::argsort(width_list);
        auto sort_end = std::chrono::steady_clock::now();
        sort_time = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();

        // CPU/GPU use configurable batch size
        int batch_num = this->rec_batch_num_;

        // Calculate batches
        for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num)
        {
            int end_img_no = std::min(img_num, beg_img_no + batch_num);
            int batch_size = end_img_no - beg_img_no;

            std::vector<cv::Mat> norm_img_batch;
            norm_img_batch.reserve(batch_size);

            auto resize_start = std::chrono::steady_clock::now();

            // Standard preprocessing for CPU/GPU (dynamic width calculation)
            float max_wh_ratio = static_cast<float>(this->rec_img_w_) / static_cast<float>(this->rec_img_h_);
            for (int idx = beg_img_no; idx < end_img_no; ++idx)
            {
                int h = img_list[indices[idx]].rows;
                int w = img_list[indices[idx]].cols;
                float wh_ratio = w * 1.0f / h;
                max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
            }

            // Normalize images
            for (int idx = beg_img_no; idx < end_img_no; ++idx)
            {
                cv::Mat resize_img;
                std::vector<int> rec_image_shape = {3, this->rec_img_h_, this->rec_img_w_};
                this->resize_op_.Run(img_list[indices[idx]], resize_img, max_wh_ratio, false, rec_image_shape);

                auto normalize_start = std::chrono::steady_clock::now();
                this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);
                auto normalize_end = std::chrono::steady_clock::now();
                normalize_time += std::chrono::duration<double, std::milli>(normalize_end - normalize_start).count();

                norm_img_batch.push_back(resize_img);
            }

            // Use the calculated width for CPU/GPU
            int target_w = int(this->rec_img_h_ * max_wh_ratio);
            int target_h = this->rec_img_h_;

            auto resize_end = std::chrono::steady_clock::now();
            resize_time += std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

            // Prepare input data
            auto permute_start = std::chrono::steady_clock::now();
            std::vector<float> input(batch_size * 3 * target_h * target_w, 0.0f);
            this->permute_op_.Run(norm_img_batch, input.data());
            auto permute_end = std::chrono::steady_clock::now();
            permute_time += std::chrono::duration<double, std::milli>(permute_end - permute_start).count();

            // Inference with OpenVINO - use single model for CPU/GPU
            auto inference_start = std::chrono::steady_clock::now();

            // Get input tensor
            auto input_tensor = infer_request_small_.get_input_tensor();
            input_tensor.set_shape({static_cast<size_t>(batch_size), 3,
                                    static_cast<size_t>(target_h),
                                    static_cast<size_t>(target_w)});

            // Copy input data and run inference
            float *input_data = input_tensor.data<float>();
            std::memcpy(input_data, input.data(), input.size() * sizeof(float));
            infer_request_small_.infer();

            // Get output tensor
            auto output_tensor = infer_request_small_.get_output_tensor();
            auto output_shape = output_tensor.get_shape();
            size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());

            std::vector<float> predict_batch(out_num);
            float *output_data = output_tensor.data<float>();
            std::memcpy(predict_batch.data(), output_data, out_num * sizeof(float));

            auto inference_end = std::chrono::steady_clock::now();
            inference_time += std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

            auto postprocess_start = std::chrono::steady_clock::now();

            // Process results for current batch
            for (int m = 0; m < batch_size; m++)
            {
                int start_idx = m * static_cast<int>(output_shape[1]) * static_cast<int>(output_shape[2]);
                int end_idx = (m + 1) * static_cast<int>(output_shape[1]) * static_cast<int>(output_shape[2]);

                std::vector<float> single_predict(predict_batch.begin() + start_idx,
                                                  predict_batch.begin() + end_idx);

                // CTC decode - based on original ocr_rec.cpp implementation
                std::string str_res;
                int argmax_idx;
                int last_index = 0;
                float score = 0.f;
                int count = 0;
                float max_value = 0.0f;

                for (int n = 0; n < static_cast<int>(output_shape[1]); ++n)
                {
                    // get idx
                    argmax_idx = int(Utility::argmax(
                        &single_predict[n * static_cast<int>(output_shape[2])],
                        &single_predict[(n + 1) * static_cast<int>(output_shape[2])]));

                    // get score
                    max_value = float(*std::max_element(
                        &single_predict[n * static_cast<int>(output_shape[2])],
                        &single_predict[(n + 1) * static_cast<int>(output_shape[2])]));

                    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
                    {
                        score += max_value;
                        count += 1;
                        str_res += this->label_list_[argmax_idx];
                    }
                    last_index = argmax_idx;
                }

                if (count > 0)
                {
                    score /= count;
                }
                else
                {
                    score = 0.0f;
                }

                if (!std::isnan(score) && !str_res.empty())
                {
                    temp_rec_texts.push_back(str_res);
                    temp_rec_text_scores.push_back(score);
                }
                else
                {
                    temp_rec_texts.push_back("");
                    temp_rec_text_scores.push_back(0.0f);
                }
            }

            auto postprocess_end = std::chrono::steady_clock::now();
            postprocess_time += std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        }

        // Reorder results back to original sequence
        rec_texts.resize(img_num);
        rec_text_scores.resize(img_num);
        for (int i = 0; i < img_num; ++i)
        {
            rec_texts[indices[i]] = temp_rec_texts[i];
            rec_text_scores[indices[i]] = temp_rec_text_scores[i];
        }

        auto total_end = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Store detailed timings (in milliseconds)
        times.clear();
        times.resize(8);
        times[0] = sort_time;                   // 排序时间
        times[1] = resize_time;                 // 图像resize时间
        times[2] = normalize_time;              // 图像归一化时间
        times[3] = permute_time;                // 维度变换时间
        times[4] = inference_time;              // 神经网络推理时间
        times[5] = postprocess_time;            // CTC解码时间
        times[6] = total_time - inference_time; // 除推理外的总时间
        times[7] = total_time;                  // 总时间
    }

} // namespace PaddleOCR
