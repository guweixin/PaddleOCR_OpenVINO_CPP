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
#include <include/data_saver.h>

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
            ov::AnyMap config = {{"PERFORMANCE_HINT", "LATENCY"}};
            if (device_ == "CPU")
            {
                // CPU-specific configurations
                config["CPU_RUNTIME_CACHE_CAPACITY"] = "0";
            }
            else if (device_ == "GPU")
            {
                // GPU-specific optimizations
                config["GPU_ENABLE_LOOP_UNROLLING"] = "YES";
                config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "NO";
                config["INFERENCE_PRECISION_HINT"] = "f16"; // Use FP16 for better GPU performance
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
            auto preprocess_start = std::chrono::steady_clock::now();

            // Initialize timing info - provide detailed timing values
            times.clear();
            times.resize(8, 0.0); // 5 detailed + 3 summary

            // Clear output vectors to avoid accumulation
            rec_texts.clear();
            rec_text_scores.clear();

            // Detailed timing variables
            auto total_resize_time = 0.0;
            auto total_normalize_time = 0.0;
            auto total_permute_time = 0.0;
            auto total_inference_time = 0.0;
            auto total_postprocess_time = 0.0;

            // Summary timing variables
            auto total_preprocess_time = 0.0;
            auto total_summary_inference_time = 0.0;
            auto total_summary_postprocess_time = 0.0;

            // Memory transfer timing variables
            auto total_shape_time = 0.0;
            auto total_cpu_to_gpu_time = 0.0;
            auto total_pure_inference_time = 0.0;
            auto total_gpu_to_cpu_time = 0.0;
            auto total_input_size_mb = 0.0;
            auto total_output_size_mb = 0.0;
            int total_batches = 0;

            // Detailed overhead timing variables
            auto total_batch_setup_overhead = 0.0;
            auto total_memory_allocation_overhead = 0.0;
            auto total_openvino_framework_overhead = 0.0;
            auto total_data_transfer_overhead = 0.0;
            auto total_shape_setting_overhead = 0.0;
            auto total_system_call_overhead = 0.0;
            
            // Additional detailed overhead measurements
            auto total_preprocessing_overhead = 0.0;
            auto total_loop_overhead = 0.0;
            auto total_vector_operations_overhead = 0.0;
            auto total_openvino_api_overhead = 0.0;
            auto total_ctc_decode_overhead = 0.0;
            auto total_result_handling_overhead = 0.0;

            int img_num = static_cast<int>(img_list.size());

            // Create temporary vectors to store results in sorted order
            std::vector<std::string> temp_rec_texts;
            std::vector<float> temp_rec_text_scores;
            temp_rec_texts.reserve(img_num);
            temp_rec_text_scores.reserve(img_num);

            // Calculate aspect ratio and sort indices for optimization (same as Python version)
            std::vector<float> width_list;
            for (int i = 0; i < img_num; ++i)
            {
                width_list.emplace_back(static_cast<float>(img_list[i].cols) / static_cast<float>(img_list[i].rows));
            }
            std::vector<size_t> indices = Utility::argsort(width_list);

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
            
            // Additional timing variables for detailed analysis
            double total_batch_setup_time = 0.0;
            double total_memory_allocation_time = 0.0;
            double total_image_processing_time = 0.0;
            double total_tensor_creation_time = 0.0;
            double total_postprocess_overhead_time = 0.0;
            
            for (int idx = 0; idx < img_num; idx += batch_num)
            {
                auto batch_start_time = std::chrono::steady_clock::now();
                
                predict_batch_num++;
            }

            for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num)
            {
                // === 测量整个批处理循环的开始 ===
                auto batch_loop_start = std::chrono::steady_clock::now();
                
                // === 1. 批处理设置开销测量 ===
                auto batch_setup_start = std::chrono::steady_clock::now();
                int end_img_no = std::min(img_num, beg_img_no + batch_num);
                int batch_size = end_img_no - beg_img_no;

                auto batch_setup_end = std::chrono::steady_clock::now();
                total_batch_setup_overhead += std::chrono::duration<double, std::milli>(batch_setup_end - batch_setup_start).count();

                // === 2. 内存分配开销测量 ===
                auto memory_alloc_start = std::chrono::steady_clock::now();

                // NPU specific preprocessing or standard preprocessing
                std::vector<cv::Mat> norm_img_batch;
                norm_img_batch.reserve(batch_size);  // 预分配内存
                int batch_width;

                auto memory_alloc_end = std::chrono::steady_clock::now();
                total_memory_allocation_overhead += std::chrono::duration<double, std::milli>(memory_alloc_end - memory_alloc_start).count();

                if (device_ == "NPU")
                {
                    // NPU specific preprocessing: fixed size [48,480] with aspect ratio preserved and white padding
                    int target_h = 48;
                    int target_w = 480;
                    batch_width = target_w;

                    for (int idx = beg_img_no; idx < end_img_no; ++idx)
                    {
                        // === 预处理间隔开销测量 ===
                        auto preprocess_gap_start = std::chrono::steady_clock::now();
                        
                        auto resize_start = std::chrono::steady_clock::now();
                        cv::Mat resize_img;
                        int original_h = img_list[indices[idx]].rows;
                        int original_w = img_list[indices[idx]].cols;

                        // Calculate scale to fit within [48,480] while preserving aspect ratio
                        float scale_h = static_cast<float>(target_h) / original_h;
                        float scale_w = static_cast<float>(target_w) / original_w;
                        float scale = std::min(scale_h, scale_w);

                        int new_h = static_cast<int>(original_h * scale);
                        int new_w = static_cast<int>(original_w * scale);

                        // Resize while preserving aspect ratio
                        cv::Mat scaled_img;
                        cv::resize(img_list[indices[idx]], scaled_img, cv::Size(new_w, new_h));

                        // Create [48,480] canvas with white background
                        resize_img = cv::Mat(target_h, target_w, CV_8UC3, cv::Scalar(255, 255, 255));

                        // Left-align the scaled image (same as Python version)
                        int start_x = 0; // Left-align instead of center
                        int start_y = 0; // Top-align instead of center

                        // Copy scaled image to the left-top of the canvas
                        cv::Rect roi(start_x, start_y, new_w, new_h);
                        scaled_img.copyTo(resize_img(roi));
                        auto resize_end = std::chrono::steady_clock::now();

                        auto normalize_start = std::chrono::steady_clock::now();
                        this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);
                        auto normalize_end = std::chrono::steady_clock::now();

                        // === 向量操作开销测量 ===
                        auto vector_op_start = std::chrono::steady_clock::now();
                        norm_img_batch.push_back(resize_img);
                        auto vector_op_end = std::chrono::steady_clock::now();
                        total_vector_operations_overhead += std::chrono::duration<double, std::milli>(vector_op_end - vector_op_start).count();

                        auto preprocess_gap_end = std::chrono::steady_clock::now();
                        double gap_time = std::chrono::duration<double, std::milli>(preprocess_gap_end - preprocess_gap_start).count();
                        double resize_time = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();
                        double normalize_time = std::chrono::duration<double, std::milli>(normalize_end - normalize_start).count();
                        total_preprocessing_overhead += gap_time - resize_time - normalize_time;

                        // Accumulate detailed timing
                        total_resize_time += std::chrono::duration<float>(resize_end - resize_start).count() * 1000;
                        total_normalize_time += std::chrono::duration<float>(normalize_end - normalize_start).count() * 1000;
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
                        // === 预处理间隔开销测量 ===
                        auto preprocess_gap_start = std::chrono::steady_clock::now();
                        
                        auto resize_start = std::chrono::steady_clock::now();
                        cv::Mat resize_img;
                        std::vector<int> rec_image_shape = {3, this->rec_img_h_, this->rec_img_w_};
                        this->resize_op_.Run(img_list[indices[idx]], resize_img, max_wh_ratio, false, rec_image_shape);
                        auto resize_end = std::chrono::steady_clock::now();

                        auto normalize_start = std::chrono::steady_clock::now();
                        this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);
                        auto normalize_end = std::chrono::steady_clock::now();

                        // === 向量操作开销测量 ===
                        auto vector_op_start = std::chrono::steady_clock::now();
                        norm_img_batch.push_back(resize_img);
                        auto vector_op_end = std::chrono::steady_clock::now();
                        total_vector_operations_overhead += std::chrono::duration<double, std::milli>(vector_op_end - vector_op_start).count();

                        auto preprocess_gap_end = std::chrono::steady_clock::now();
                        double gap_time = std::chrono::duration<double, std::milli>(preprocess_gap_end - preprocess_gap_start).count();
                        double resize_time = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();
                        double normalize_time = std::chrono::duration<double, std::milli>(normalize_end - normalize_start).count();
                        total_preprocessing_overhead += gap_time - resize_time - normalize_time;

                        // Accumulate detailed timing
                        total_resize_time += std::chrono::duration<float>(resize_end - resize_start).count() * 1000;
                        total_normalize_time += std::chrono::duration<float>(normalize_end - normalize_start).count() * 1000;
                    }

                    // Calculate actual batch width based on max_wh_ratio (Python version logic)
                    batch_width = int(this->rec_img_h_ * max_wh_ratio);
                }

                // === 3. 数据准备和张量创建开销测量 ===
                auto tensor_creation_start = std::chrono::steady_clock::now();

                // Prepare input data - 测量内存分配开销
                auto vector_alloc_start = std::chrono::steady_clock::now();
                std::vector<float> input(batch_size * 3 * this->rec_img_h_ * batch_width, 0.0f);
                auto vector_alloc_end = std::chrono::steady_clock::now();
                total_memory_allocation_overhead += std::chrono::duration<double, std::milli>(vector_alloc_end - vector_alloc_start).count();

                auto permute_start = std::chrono::steady_clock::now();
                this->permute_op_.Run(norm_img_batch, input.data());
                auto permute_end = std::chrono::steady_clock::now();

                auto tensor_creation_end = std::chrono::steady_clock::now();
                total_system_call_overhead += std::chrono::duration<double, std::milli>(tensor_creation_end - tensor_creation_start).count();

                // Accumulate permute timing
                total_permute_time += std::chrono::duration<float>(permute_end - permute_start).count() * 1000;

                auto preprocess_end = std::chrono::steady_clock::now();

                // === 4. OpenVINO API开销测量 ===
                auto openvino_api_start = std::chrono::steady_clock::now();

                // Inference with OpenVINO - detailed memory transfer timing
                auto inference_start = std::chrono::steady_clock::now();

                // Get input tensor and measure shape setup time
                auto shape_start = std::chrono::steady_clock::now();
                auto input_tensor = infer_request_.get_input_tensor();
                auto openvino_api_intermediate1 = std::chrono::steady_clock::now();
                total_openvino_api_overhead += std::chrono::duration<double, std::milli>(openvino_api_intermediate1 - openvino_api_start).count();

                auto openvino_overhead_start = std::chrono::steady_clock::now();
                auto openvino_overhead_intermediate = std::chrono::steady_clock::now();
                total_openvino_framework_overhead += std::chrono::duration<double, std::milli>(openvino_overhead_intermediate - openvino_overhead_start).count();

                // === 5. 形状设置时间测量 ===
                // Set input shape and copy data
                input_tensor.set_shape({static_cast<size_t>(batch_size), 3,
                                        static_cast<size_t>(this->rec_img_h_),
                                        static_cast<size_t>(batch_width)});
                auto shape_end = std::chrono::steady_clock::now();
                total_shape_setting_overhead += std::chrono::duration<double, std::milli>(shape_end - shape_start).count();

                // === 6. 数据传输时间测量 ===
                // CPU to GPU data transfer timing
                auto cpu_to_gpu_start = std::chrono::steady_clock::now();
                float *input_data = input_tensor.data<float>();
                std::memcpy(input_data, input.data(), input.size() * sizeof(float));
                auto cpu_to_gpu_end = std::chrono::steady_clock::now();
                total_data_transfer_overhead += std::chrono::duration<double, std::milli>(cpu_to_gpu_end - cpu_to_gpu_start).count();

                // Pure inference timing
                auto pure_inference_start = std::chrono::steady_clock::now();
                infer_request_.infer();
                auto pure_inference_end = std::chrono::steady_clock::now();

                // GPU to CPU data transfer timing
                auto gpu_to_cpu_start = std::chrono::steady_clock::now();
                
                // === OpenVINO API开销测量 ===
                auto openvino_api_start2 = std::chrono::steady_clock::now();
                auto output_tensor = infer_request_.get_output_tensor();
                auto output_shape = output_tensor.get_shape();
                auto openvino_api_end2 = std::chrono::steady_clock::now();
                total_openvino_api_overhead += std::chrono::duration<double, std::milli>(openvino_api_end2 - openvino_api_start2).count();

                size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
                
                // === 向量内存分配开销 ===
                auto vector_alloc_start2 = std::chrono::steady_clock::now();
                std::vector<float> predict_batch;
                predict_batch.reserve(out_num);
                predict_batch.resize(out_num);
                auto vector_alloc_end2 = std::chrono::steady_clock::now();
                total_memory_allocation_overhead += std::chrono::duration<double, std::milli>(vector_alloc_end2 - vector_alloc_start2).count();

                float *output_data = output_tensor.data<float>();
                std::memcpy(predict_batch.data(), output_data, out_num * sizeof(float));
                auto gpu_to_cpu_end = std::chrono::steady_clock::now();
                
                // 继续累计数据传输开销
                total_data_transfer_overhead += std::chrono::duration<double, std::milli>(gpu_to_cpu_end - gpu_to_cpu_start).count();

                // Calculate detailed memory transfer timings
                double shape_time = std::chrono::duration<double, std::milli>(shape_end - shape_start).count();
                double cpu_to_gpu_time = std::chrono::duration<double, std::milli>(cpu_to_gpu_end - cpu_to_gpu_start).count();
                double pure_inference_time = std::chrono::duration<double, std::milli>(pure_inference_end - pure_inference_start).count();
                double gpu_to_cpu_time = std::chrono::duration<double, std::milli>(gpu_to_cpu_end - gpu_to_cpu_start).count();
                
                // Calculate data sizes for context
                size_t input_size_mb = (input.size() * sizeof(float)) / (1024 * 1024);
                size_t output_size_mb = (out_num * sizeof(float)) / (1024 * 1024);

                // Accumulate memory transfer timing statistics
                total_shape_time += shape_time;
                total_cpu_to_gpu_time += cpu_to_gpu_time;
                total_pure_inference_time += pure_inference_time;
                total_gpu_to_cpu_time += gpu_to_cpu_time;
                total_input_size_mb += input_size_mb;
                total_output_size_mb += output_size_mb;
                total_batches++;

                // Save debug data if enabled
                if (FLAGS_save_debug_data)
                {
                    static int rec_batch_counter = 0;

                    std::vector<size_t> input_shape_vec = {static_cast<size_t>(batch_size), 3,
                                                           static_cast<size_t>(this->rec_img_h_),
                                                           static_cast<size_t>(batch_width)};
                    std::vector<size_t> output_shape_vec(output_shape.begin(), output_shape.end());

                    // Save recognition model input and output data
                    DataSaver::SaveRecognitionData(input, input_shape_vec, predict_batch, output_shape_vec,
                                                   rec_batch_counter, beg_img_no);

                    // // Save individual preprocessed images for debugging
                    // for (int i = 0; i < batch_size && i < norm_img_batch.size(); ++i)
                    // {
                    //     std::string img_filename = "../../debug_data/cpp_preprocessed_img_batch" +
                    //                                std::to_string(rec_batch_counter) + "_" + std::to_string(beg_img_no + i) + ".npy";
                    //     DataSaver::SaveMatAsNpy(norm_img_batch[i], img_filename);
                    // }

                    rec_batch_counter++;
                }

                auto inference_end = std::chrono::steady_clock::now(); // Postprocessing
                auto postprocess_start = std::chrono::steady_clock::now();

                // === CTC解码开销测量 ===
                auto ctc_decode_start = std::chrono::steady_clock::now();

                // Process results for current batch
                for (int m = 0; m < batch_size; m++)
                {
                    // === 结果处理开销测量 ===
                    auto result_handling_start = std::chrono::steady_clock::now();
                    
                    int start_idx = m * static_cast<int>(output_shape[1]) * static_cast<int>(output_shape[2]);
                    int end_idx = (m + 1) * static_cast<int>(output_shape[1]) * static_cast<int>(output_shape[2]);

                    std::vector<float> single_predict(predict_batch.begin() + start_idx,
                                                      predict_batch.begin() + end_idx);

                    auto result_handling_intermediate = std::chrono::steady_clock::now();
                    total_result_handling_overhead += std::chrono::duration<double, std::milli>(result_handling_intermediate - result_handling_start).count();

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

                    // === 向量操作开销 ===
                    auto vector_op_start = std::chrono::steady_clock::now();
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
                    auto vector_op_end = std::chrono::steady_clock::now();
                    total_vector_operations_overhead += std::chrono::duration<double, std::milli>(vector_op_end - vector_op_start).count();
                }

                auto ctc_decode_end = std::chrono::steady_clock::now();
                total_ctc_decode_overhead += std::chrono::duration<double, std::milli>(ctc_decode_end - ctc_decode_start).count();

                auto postprocess_end = std::chrono::steady_clock::now();

                // === 批处理循环结束开销测量 ===
                auto batch_loop_end = std::chrono::steady_clock::now();
                double total_batch_time = std::chrono::duration<double, std::milli>(batch_loop_end - batch_loop_start).count();
                double measured_components_time = total_batch_setup_overhead + total_memory_allocation_overhead + 
                                                 total_preprocessing_overhead + total_vector_operations_overhead +
                                                 std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count() +
                                                 std::chrono::duration<double, std::milli>(inference_end - inference_start).count() +
                                                 total_ctc_decode_overhead + total_result_handling_overhead;
                total_loop_overhead += std::max(0.0, total_batch_time - measured_components_time);

                // Calculate timing for all batches
                std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
                std::chrono::duration<float> inference_diff = inference_end - inference_start;
                std::chrono::duration<float> postprocess_diff = postprocess_end - inference_end;

                // Accumulate summary timing
                total_preprocess_time += preprocess_diff.count() * 1000;
                total_summary_inference_time += inference_diff.count() * 1000;
                total_summary_postprocess_time += postprocess_diff.count() * 1000;

                // Accumulate detailed inference and postprocess timing
                total_inference_time += inference_diff.count() * 1000;
                total_postprocess_time += postprocess_diff.count() * 1000;
            }

            // Reorder results back to original sequence (same as Python version)
            rec_texts.resize(img_num);
            rec_text_scores.resize(img_num);
            for (int i = 0; i < img_num; ++i)
            {
                rec_texts[indices[i]] = temp_rec_texts[i];
                rec_text_scores[indices[i]] = temp_rec_text_scores[i];
            }

            printf("\n=== Recognition Memory Transfer Analysis ===\n");
            printf("Total batches processed: %d\n", total_batches);
            printf("Total images processed: %d\n", img_num);
            printf("Batch size used: %d\n", this->rec_batch_num_);
            printf("\nPer-batch Average Memory Transfer Timing:\n");
            if (total_batches > 0) {
                printf("  Shape setup:     %.2f ms\n", total_shape_time / total_batches);
                printf("  CPU->GPU copy:   %.2f ms (avg %.2f MB)\n", total_cpu_to_gpu_time / total_batches, total_input_size_mb / total_batches);
                printf("  Pure inference:  %.2f ms\n", total_pure_inference_time / total_batches);
                printf("  GPU->CPU copy:   %.2f ms (avg %.2f MB)\n", total_gpu_to_cpu_time / total_batches, total_output_size_mb / total_batches);
                printf("  Total transfer:  %.2f ms (%.1f%% of total inference)\n", 
                       (total_cpu_to_gpu_time + total_gpu_to_cpu_time) / total_batches,
                       ((total_cpu_to_gpu_time + total_gpu_to_cpu_time) / (total_shape_time + total_cpu_to_gpu_time + total_pure_inference_time + total_gpu_to_cpu_time)) * 100);
            }
            printf("\nPer-image Average Memory Transfer Timing:\n");
            if (img_num > 0) {
                printf("  Shape setup:     %.2f ms per image\n", total_shape_time / img_num);
                printf("  CPU->GPU copy:   %.2f ms per image (avg %.2f MB)\n", total_cpu_to_gpu_time / img_num, total_input_size_mb / img_num);
                printf("  Pure inference:  %.2f ms per image\n", total_pure_inference_time / img_num);
                printf("  GPU->CPU copy:   %.2f ms per image (avg %.2f MB)\n", total_gpu_to_cpu_time / img_num, total_output_size_mb / img_num);
                printf("  Total transfer:  %.2f ms per image\n", (total_cpu_to_gpu_time + total_gpu_to_cpu_time) / img_num);
                printf("  Memory transfer overhead: %.1f%% of pure inference time\n", 
                       ((total_cpu_to_gpu_time + total_gpu_to_cpu_time) / total_pure_inference_time) * 100);
            }
            printf("============================================\n\n");

            // === 详细开销时间分析 ===
            printf("=== Recognition Detailed Overhead Analysis ===\n");
            printf("Total processing time breakdown:\n");
            if (img_num > 0) {
                double total_overhead = total_batch_setup_overhead + total_memory_allocation_overhead + 
                                      total_openvino_framework_overhead + total_data_transfer_overhead + 
                                      total_shape_setting_overhead + total_system_call_overhead +
                                      total_preprocessing_overhead + total_loop_overhead + 
                                      total_vector_operations_overhead + total_openvino_api_overhead + 
                                      total_ctc_decode_overhead + total_result_handling_overhead;
                
                double total_time = total_preprocess_time + total_summary_inference_time + total_summary_postprocess_time;
                
                printf("=== 基础开销 ===\n");
                printf("1. 批处理设置开销:      %.2f ms (%.1f%% of total)\n", 
                       total_batch_setup_overhead / img_num, 
                       (total_batch_setup_overhead / total_time) * 100);
                       
                printf("2. 内存分配/释放开销:   %.2f ms (%.1f%% of total)\n", 
                       total_memory_allocation_overhead / img_num,
                       (total_memory_allocation_overhead / total_time) * 100);
                       
                printf("3. OpenVINO框架开销:    %.2f ms (%.1f%% of total)\n", 
                       total_openvino_framework_overhead / img_num,
                       (total_openvino_framework_overhead / total_time) * 100);
                       
                printf("4. 数据传输开销:        %.2f ms (%.1f%% of total)\n", 
                       total_data_transfer_overhead / img_num,
                       (total_data_transfer_overhead / total_time) * 100);
                       
                printf("5. 形状设置开销:        %.2f ms (%.1f%% of total)\n", 
                       total_shape_setting_overhead / img_num,
                       (total_shape_setting_overhead / total_time) * 100);
                       
                printf("6. 系统调用开销:        %.2f ms (%.1f%% of total)\n", 
                       total_system_call_overhead / img_num,
                       (total_system_call_overhead / total_time) * 100);
                       
                printf("=== 新增开销分析 ===\n");
                printf("7. 预处理间隔开销:      %.2f ms (%.1f%% of total)\n", 
                       total_preprocessing_overhead / img_num,
                       (total_preprocessing_overhead / total_time) * 100);
                       
                printf("8. 批处理循环开销:      %.2f ms (%.1f%% of total)\n", 
                       total_loop_overhead / img_num,
                       (total_loop_overhead / total_time) * 100);
                       
                printf("9. 向量操作开销:        %.2f ms (%.1f%% of total)\n", 
                       total_vector_operations_overhead / img_num,
                       (total_vector_operations_overhead / total_time) * 100);
                       
                printf("10. OpenVINO API开销:   %.2f ms (%.1f%% of total)\n", 
                       total_openvino_api_overhead / img_num,
                       (total_openvino_api_overhead / total_time) * 100);
                       
                printf("11. CTC解码开销:        %.2f ms (%.1f%% of total)\n", 
                       total_ctc_decode_overhead / img_num,
                       (total_ctc_decode_overhead / total_time) * 100);
                       
                printf("12. 结果处理开销:       %.2f ms (%.1f%% of total)\n", 
                       total_result_handling_overhead / img_num,
                       (total_result_handling_overhead / total_time) * 100);
                       
                printf("-------------------------------------------\n");
                printf("测量到的总开销:         %.2f ms (%.1f%% of total)\n", 
                       total_overhead / img_num,
                       (total_overhead / total_time) * 100);
                       
                printf("纯计算时间:             %.2f ms (%.1f%% of total)\n", 
                       (total_resize_time + total_normalize_time + total_permute_time + total_pure_inference_time + total_postprocess_time) / img_num,
                       ((total_resize_time + total_normalize_time + total_permute_time + total_pure_inference_time + total_postprocess_time) / total_time) * 100);
                       
                printf("===== 时间差值分析 =====\n");
                double time_diff = total_time - 
                                  (total_resize_time + total_normalize_time + total_permute_time + total_inference_time + total_postprocess_time);
                double measured_overhead = total_overhead;
                double unmeasured_time = time_diff - measured_overhead;
                
                printf("时间差值总计:           %.2f ms (%.1f ms per image)\n", 
                       time_diff, time_diff / img_num);
                printf("已测量开销:             %.2f ms (%.1f%% of 差值)\n", 
                       measured_overhead, (measured_overhead / time_diff) * 100);
                printf("未测量时间:             %.2f ms (%.1f%% of 差值)\n", 
                       unmeasured_time, (unmeasured_time / time_diff) * 100);
                       
                if (unmeasured_time > 0) {
                    printf("*** 警告: 仍有 %.1f ms (%.1f%%) 的时间无法解释! ***\n",
                           unmeasured_time / img_num, (unmeasured_time / time_diff) * 100);
                }
            }
            printf("===============================================\n\n");

            // Set final timing results
            // Detailed timings: [resize, normalize, permute, inference, postprocess]
            times[0] = total_resize_time;      // Detailed: resize
            times[1] = total_normalize_time;   // Detailed: normalize
            times[2] = total_permute_time;     // Detailed: permute
            times[3] = total_inference_time;   // Detailed: inference
            times[4] = total_postprocess_time; // Detailed: postprocess

            // Summary timings: [preprocess, inference, postprocess]
            times[5] = total_preprocess_time;          // Summary: preprocess
            times[6] = total_summary_inference_time;   // Summary: inference
            times[7] = total_summary_postprocess_time; // Summary: postprocess
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Exception in CRNNRecognizerOpenVINO::Run: " << e.what() << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(8, 0.0); // 5 detailed + 3 summary
            // Exit to avoid further processing errors
            exit(1);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Unknown exception in CRNNRecognizerOpenVINO::Run" << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(8, 0.0); // 5 detailed + 3 summary
            // Exit to avoid further processing errors
            exit(1);
        }
    }

} // namespace PaddleOCR
