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

// 移除了OpenCL头文件以避免编译错误

#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace PaddleOCR
{

    void CRNNRecognizerOpenVINO::LoadModel(const std::string &model_path) noexcept
    {
        try
        {
            std::cout << "[OpenVINO] Loading recognition model from: " << model_path << std::endl;
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
            std::cout << "[OpenVINO] Recognition model file loaded successfully" << std::endl;

            // Configure device-specific settings
            ov::AnyMap config = {{"PERFORMANCE_HINT", "LATENCY"}, {"CACHE_DIR", "./openvino_cache"}};
            if (device_ == "CPU")
            {
                // CPU-specific configurations
                config["CPU_RUNTIME_CACHE_CAPACITY"] = "0";
            }
            else if (device_ == "GPU" || device_ == "NPU")
            {
                config["INFERENCE_PRECISION_HINT"] = "f16";
            }
            // Compile the model for the specified device
            compiled_model_ = core_.compile_model(model, device_, config);

            // Create inference request
            infer_request_ = compiled_model_.create_infer_request();

            std::cout << "[OpenVINO] Recognition model loaded successfully for device: " << device_ << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Failed to load OpenVINO recognition model: " << e.what() << std::endl;
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
            auto total_start = std::chrono::steady_clock::now();
            auto preprocess_start = std::chrono::steady_clock::now();

            // Initialize timing variables
            double sort_time = 0.0;
            double resize_time = 0.0;
            double normalize_time = 0.0;
            double permute_time = 0.0;
            double inference_time = 0.0;
            double postprocess_time = 0.0;

            // Clear output vectors to avoid accumulation
            rec_texts.clear();
            rec_text_scores.clear();

            int img_num = static_cast<int>(img_list.size());

            // Create temporary vectors to store results in sorted order
            std::vector<std::string> temp_rec_texts;
            std::vector<float> temp_rec_text_scores;
            temp_rec_texts.reserve(img_num);
            temp_rec_text_scores.reserve(img_num);

            auto sort_start = std::chrono::steady_clock::now();
            // Calculate aspect ratio and sort indices for optimization (same as Python version)
            std::vector<float> width_list;
            for (int i = 0; i < img_num; ++i)
            {
                width_list.emplace_back(static_cast<float>(img_list[i].cols) / static_cast<float>(img_list[i].rows));
            }
            std::vector<size_t> indices = Utility::argsort(width_list);
            auto sort_end = std::chrono::steady_clock::now();
            sort_time = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();

            // Set batch size based on device type
            int batch_num;
            if (device_ == "NPU")
            {
                batch_num = 1; // NPU uses batch size 1
            }
            else
            {
                batch_num = this->rec_batch_num_; // CPU/GPU use configurable batch size
            }

            // Calculate batches
            int predict_batch_num = 0;

            for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num)
            {
                int end_img_no = std::min(img_num, beg_img_no + batch_num);
                int batch_size = end_img_no - beg_img_no;

                // NPU specific preprocessing or standard preprocessing
                std::vector<cv::Mat> norm_img_batch;
                norm_img_batch.reserve(batch_size);
                int batch_width;

                auto resize_start = std::chrono::steady_clock::now();

                if (device_ == "NPU")
                {
                    // NPU specific preprocessing: fixed size [48,480] with aspect ratio preserved and white padding
                    int target_h = 48;
                    int target_w = 480;
                    batch_width = target_w;

                    for (int idx = beg_img_no; idx < end_img_no; ++idx)
                    {
                        cv::Mat resize_img;
                        // int original_h = img_list[indices[idx]].rows;
                        // int original_w = img_list[indices[idx]].cols;

                        // // Calculate scale to fit within [48,480] while preserving aspect ratio
                        // float scale_h = static_cast<float>(target_h) / original_h;
                        // float scale_w = static_cast<float>(target_w) / original_w;
                        // float scale = std::min(scale_h, scale_w);

                        // int new_h = static_cast<int>(original_h * scale);
                        // int new_w = static_cast<int>(original_w * scale);

                        // // Resize while preserving aspect ratio
                        // cv::Mat scaled_img;
                        // cv::resize(img_list[indices[idx]], scaled_img, cv::Size(new_w, new_h));

                        // // Create [48,480] canvas with white background
                        // resize_img = cv::Mat(target_h, target_w, CV_8UC3, cv::Scalar(255, 255, 255));

                        // // Left-align the scaled image (same as Python version)
                        // int start_x = 0; // Left-align instead of center
                        // int start_y = 0; // Top-align instead of center

                        // // Copy scaled image to the left-top of the canvas
                        // cv::Rect roi(start_x, start_y, new_w, new_h);
                        // scaled_img.copyTo(resize_img(roi));

                        // ! resize image
                        cv::resize(img_list[indices[idx]], resize_img, cv::Size(target_w, target_h));

                        auto normalize_start = std::chrono::steady_clock::now();
                        this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);
                        auto normalize_end = std::chrono::steady_clock::now();
                        normalize_time += std::chrono::duration<double, std::milli>(normalize_end - normalize_start).count();

                        norm_img_batch.push_back(resize_img);
                    }
                }
                else
                {
                    // Standard preprocessing for CPU/GPU (dynamic width calculation)
                    // Calculate max_wh_ratio for current batch (Python version logic)
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

                    // Calculate actual batch width based on max_wh_ratio (Python version logic)
                    batch_width = int(this->rec_img_h_ * max_wh_ratio);
                }

                auto resize_end = std::chrono::steady_clock::now();
                resize_time += std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

                // Prepare input data
                auto permute_start = std::chrono::steady_clock::now();
                std::vector<float> input(batch_size * 3 * this->rec_img_h_ * batch_width, 0.0f);
                this->permute_op_.Run(norm_img_batch, input.data());
                auto permute_end = std::chrono::steady_clock::now();
                permute_time += std::chrono::duration<double, std::milli>(permute_end - permute_start).count();

                auto preprocess_end = std::chrono::steady_clock::now();

                // Inference with OpenVINO
                auto inference_start = std::chrono::steady_clock::now();

                // Get input tensor and set shape
                auto input_tensor = infer_request_.get_input_tensor();
                input_tensor.set_shape({static_cast<size_t>(batch_size), 3,
                                        static_cast<size_t>(this->rec_img_h_),
                                        static_cast<size_t>(batch_width)});

                // Standard path for inference
                float *input_data = input_tensor.data<float>();
                std::memcpy(input_data, input.data(), input.size() * sizeof(float));
                infer_request_.infer();

                // Get output
                auto output_tensor = infer_request_.get_output_tensor();
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

            // Reorder results back to original sequence (same as Python version)
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

} // namespace PaddleOCR
