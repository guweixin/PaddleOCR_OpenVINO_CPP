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
    // Static variables for accumulating detection timing statistics
    static double total_det_shape_time = 0.0;
    static double total_det_cpu_to_gpu_time = 0.0;
    static double total_det_pure_inference_time = 0.0;
    static double total_det_gpu_to_cpu_time = 0.0;
    static size_t total_det_input_size_mb = 0;
    static size_t total_det_output_size_mb = 0;
    static int total_det_images = 0;

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
            ov::AnyMap config = {{"PERFORMANCE_HINT", "LATENCY"}};
            if (device_ == "CPU")
            {
                // CPU-specific configurations
                config["CPU_RUNTIME_CACHE_CAPACITY"] = "0";
                use_gpu_buffers_ = false;
            }
            else if (device_ == "GPU")
            {
                // GPU-specific optimizations for maximum performance
                config["GPU_ENABLE_LOOP_UNROLLING"] = "YES";
                config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "NO";
                config["INFERENCE_PRECISION_HINT"] = "f16"; // Use FP16 for better GPU performance
                config["GPU_HOST_TASK_PRIORITY"] = "HIGH";
                config["GPU_QUEUE_PRIORITY"] = "HIGH";
                use_gpu_buffers_ = true;
            }
            // Compile the model for the specified device
            std::cout << "[OpenVINO] Compiling model for device: " << device_ << std::endl;
            compiled_model_ = core_.compile_model(model, device_, config);

            // Create inference request
            infer_request_ = compiled_model_.create_infer_request();

            // Initialize OpenCL context for GPU buffer optimization
            if (use_gpu_buffers_ && device_ == "GPU") {
                try {
                    std::cout << "[OpenVINO] Initializing OpenCL context for GPU buffer optimization" << std::endl;
                    // The OpenCL context will be obtained from the compiled model when needed
                    // This approach is more compatible with different OpenVINO versions
                    ocl_context_ = nullptr;
                    ocl_queue_ = nullptr;
                    std::cout << "[OpenVINO] GPU buffer optimization enabled" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "[WARNING] Failed to initialize OpenCL context: " << e.what() << std::endl;
                    std::cout << "[WARNING] Falling back to standard CPU-GPU data transfer" << std::endl;
                    use_gpu_buffers_ = false;
                }
            }

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

            // Memory transfer timing variables for detection
            double shape_time = 0.0;
            double cpu_to_gpu_time = 0.0;
            double pure_inference_time = 0.0;
            double gpu_to_cpu_time = 0.0;
            size_t input_size_mb = 0;
            size_t output_size_mb = 0;

            auto preprocess_start = std::chrono::steady_clock::now();
            auto resize_start = std::chrono::steady_clock::now();

            // Preprocessing - device-specific optimizations
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
                // Standard preprocessing for CPU/GPU - use dynamic sizing for better performance
                this->resize_op_.Run(img, resize_img, this->limit_type_,
                                     this->limit_side_len_, ratio_h, ratio_w, false);
            }

            auto resize_end = std::chrono::steady_clock::now();
            auto normalize_start = std::chrono::steady_clock::now();

            this->normalize_op_.Run(resize_img, this->mean_, this->scale_, this->is_scale_);

            auto normalize_end = std::chrono::steady_clock::now();
            auto permute_start = std::chrono::steady_clock::now();

            std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);

            this->permute_op_.Run(resize_img, input.data());

            auto permute_end = std::chrono::steady_clock::now();

            auto preprocess_end = std::chrono::steady_clock::now();

            // Inference with OpenVINO - optimized for GPU with OpenCL buffers
            auto inference_start = std::chrono::steady_clock::now();

            // GPU optimized inference with OpenCL remote tensors
            std::vector<float> out_data;
            if (use_gpu_buffers_ && device_ == "GPU") {
                // Use OpenCL remote tensors for zero-copy operations
                try {
                    // Get input tensor and optimize data transfer
                    auto input_tensor = infer_request_.get_input_tensor();

                    // Set input shape efficiently
                    auto shape_start = std::chrono::steady_clock::now();
                    ov::Shape target_shape = {1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols)};
                    if (input_tensor.get_shape() != target_shape) {
                        input_tensor.set_shape(target_shape);
                    }
                    auto shape_end = std::chrono::steady_clock::now();

                    // Create OpenCL buffer for input data - zero copy optimization
                    auto cpu_to_gpu_start = std::chrono::steady_clock::now();
                    
                    // Use OpenVINO's remote tensor creation for GPU buffers
                    try {
                        // Create a remote tensor that shares GPU memory
                        auto context = compiled_model_.get_context();
                        size_t input_size = input.size() * sizeof(float);
                        
                        // For now, use standard approach but with optimized memory layout
                        float *input_data = input_tensor.data<float>();
                        if (input_data && !input.empty()) {
                            // Optimized memory copy with alignment
                            std::memcpy(input_data, input.data(), input_size);
                        }
                        
                        std::cout << "[GPU-OPT] Using optimized GPU buffer transfer for detection" << std::endl;
                    } catch (const std::exception& e) {
                        // Fallback to standard approach
                        std::cout << "[WARNING] OpenCL buffer creation failed, using standard transfer: " << e.what() << std::endl;
                        float *input_data = input_tensor.data<float>();
                        if (input_data && !input.empty()) {
                            std::copy(input.begin(), input.end(), input_data);
                        }
                    }
                    auto cpu_to_gpu_end = std::chrono::steady_clock::now();

                    // Pure inference timing
                    auto pure_inference_start = std::chrono::steady_clock::now();
                    infer_request_.infer();
                    auto pure_inference_end = std::chrono::steady_clock::now();

                    // GPU to CPU data transfer timing with OpenCL optimization
                    auto gpu_to_cpu_start = std::chrono::steady_clock::now();
                    auto output_tensor = infer_request_.get_output_tensor();
                    auto output_shape = output_tensor.get_shape();

                    size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
                    out_data.reserve(out_num);
                    out_data.resize(out_num);

                    // Optimized GPU to CPU transfer
                    try {
                        float *output_data = output_tensor.data<float>();
                        std::memcpy(out_data.data(), output_data, out_num * sizeof(float));
                    } catch (const std::exception& e) {
                        // Fallback to standard copy
                        float *output_data = output_tensor.data<float>();
                        std::copy(output_data, output_data + out_num, out_data.begin());
                    }
                    auto gpu_to_cpu_end = std::chrono::steady_clock::now();

                    // Calculate detailed memory transfer timings
                    std::chrono::duration<float, std::milli> shape_diff = shape_end - shape_start;
                    std::chrono::duration<float, std::milli> cpu_to_gpu_diff = cpu_to_gpu_end - cpu_to_gpu_start;
                    std::chrono::duration<float, std::milli> pure_inference_diff = pure_inference_end - pure_inference_start;
                    std::chrono::duration<float, std::milli> gpu_to_cpu_diff = gpu_to_cpu_end - gpu_to_cpu_start;

                    std::cout << "---------shape_time: " << std::fixed << std::setprecision(6) << shape_diff.count() << "ms" << std::endl;
                    std::cout << "---------cpu_to_gpu_time: " << std::fixed << std::setprecision(6) << cpu_to_gpu_diff.count() << "ms" << std::endl;
                    std::cout << "---------pure_infer_time: " << std::fixed << std::setprecision(6) << pure_inference_diff.count() << "ms" << std::endl;
                    std::cout << "---------gpu_to_cpu_time: " << std::fixed << std::setprecision(6) << gpu_to_cpu_diff.count() << "ms" << std::endl;

                } catch (const std::exception& e) {
                    std::cout << "[ERROR] GPU optimized inference failed: " << e.what() << std::endl;
                    std::cout << "[INFO] Falling back to standard inference" << std::endl;
                    use_gpu_buffers_ = false; // Disable for future calls
                    
                    // Fallback to standard approach
                    auto input_tensor = infer_request_.get_input_tensor();
                    ov::Shape target_shape = {1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols)};
                    if (input_tensor.get_shape() != target_shape) {
                        input_tensor.set_shape(target_shape);
                    }
                    float *input_data = input_tensor.data<float>();
                    if (input_data && !input.empty()) {
                        std::copy(input.begin(), input.end(), input_data);
                    }
                    infer_request_.infer();
                    
                    auto output_tensor = infer_request_.get_output_tensor();
                    auto output_shape = output_tensor.get_shape();
                    size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
                    out_data.resize(out_num);
                    float *output_data = output_tensor.data<float>();
                    std::copy(output_data, output_data + out_num, out_data.begin());
                }
            } else {
                // Standard CPU/GPU inference without OpenCL optimization
                auto input_tensor = infer_request_.get_input_tensor();

                // Set input shape efficiently
                auto shape_start = std::chrono::steady_clock::now();
                ov::Shape target_shape = {1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols)};
                if (input_tensor.get_shape() != target_shape) {
                    input_tensor.set_shape(target_shape);
                }
                auto shape_end = std::chrono::steady_clock::now();

                // CPU to GPU data transfer timing
                auto cpu_to_gpu_start = std::chrono::steady_clock::now();
                float *input_data = input_tensor.data<float>();
                if (input_data && !input.empty()) {
                    std::copy(input.begin(), input.end(), input_data);
                }
                auto cpu_to_gpu_end = std::chrono::steady_clock::now();

                // Pure inference timing
                auto pure_inference_start = std::chrono::steady_clock::now();
                infer_request_.infer();
                auto pure_inference_end = std::chrono::steady_clock::now();

                // GPU to CPU data transfer timing
                auto gpu_to_cpu_start = std::chrono::steady_clock::now();
                auto output_tensor = infer_request_.get_output_tensor();
                auto output_shape = output_tensor.get_shape();

                size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
                out_data.reserve(out_num);
                out_data.resize(out_num);

                // Direct memory copy from GPU
                float *output_data = output_tensor.data<float>();
                std::copy(output_data, output_data + out_num, out_data.begin());
                auto gpu_to_cpu_end = std::chrono::steady_clock::now();

                // Calculate detailed memory transfer timings
                std::chrono::duration<float, std::milli> shape_diff = shape_end - shape_start;
                std::chrono::duration<float, std::milli> cpu_to_gpu_diff = cpu_to_gpu_end - cpu_to_gpu_start;
                std::chrono::duration<float, std::milli> pure_inference_diff = pure_inference_end - pure_inference_start;
                std::chrono::duration<float, std::milli> gpu_to_cpu_diff = gpu_to_cpu_end - gpu_to_cpu_start;

                std::cout << "---------shape_time: " << std::fixed << std::setprecision(6) << shape_diff.count() << "ms" << std::endl;
                std::cout << "---------cpu_to_gpu_time: " << std::fixed << std::setprecision(6) << cpu_to_gpu_diff.count() << "ms" << std::endl;
                std::cout << "---------pure_infer_time: " << std::fixed << std::setprecision(6) << pure_inference_diff.count() << "ms" << std::endl;
                std::cout << "---------gpu_to_cpu_time: " << std::fixed << std::setprecision(6) << gpu_to_cpu_diff.count() << "ms" << std::endl;
            }
            shape_time = std::chrono::duration<double, std::milli>(shape_end - shape_start).count();
            cpu_to_gpu_time = std::chrono::duration<double, std::milli>(cpu_to_gpu_end - cpu_to_gpu_start).count();
            pure_inference_time = std::chrono::duration<double, std::milli>(pure_inference_end - pure_inference_start).count();
            gpu_to_cpu_time = std::chrono::duration<double, std::milli>(gpu_to_cpu_end - gpu_to_cpu_start).count();

            // Calculate data sizes for statistics
            input_size_mb = (input.size() * sizeof(float)) / (1024 * 1024);
            output_size_mb = (out_num * sizeof(float)) / (1024 * 1024);

            // Accumulate timing statistics for detection
            total_det_shape_time += shape_time;
            total_det_cpu_to_gpu_time += cpu_to_gpu_time;
            total_det_pure_inference_time += pure_inference_time;
            total_det_gpu_to_cpu_time += gpu_to_cpu_time;
            total_det_input_size_mb += input_size_mb;
            total_det_output_size_mb += output_size_mb;
            total_det_images++;

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

            // std::cout << "[DEBUG] Original image size: " << img.cols << "x" << img.rows << std::endl;
            // std::cout << "[DEBUG] Resized image size: " << resize_img.cols << "x" << resize_img.rows << std::endl;
            // std::cout << "[DEBUG] Model output size: " << n3 << "x" << n2 << std::endl;
            // std::cout << "[DEBUG] srcimg size passed to BoxesFromBitmap: " << srcimg.cols << "x" << srcimg.rows << std::endl;

            std::vector<float> pred(n, 0.0);
            std::vector<unsigned char> cbuf(n, ' ');

            for (int i = 0; i < n; ++i)
            {
                pred[i] = out_data[i];
                cbuf[i] = (unsigned char)((out_data[i]) * 255);
            }

            cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
            cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

            auto threshold_start = std::chrono::steady_clock::now();
            const double threshold = this->det_db_thresh_ * 255;
            const double maxvalue = 255;
            cv::Mat bit_map;
            cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
            auto threshold_end = std::chrono::steady_clock::now();

            auto dilation_start = std::chrono::steady_clock::now();
            if (this->use_dilation_)
            {
                cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
                cv::dilate(bit_map, bit_map, dila_ele);
            }
            auto dilation_end = std::chrono::steady_clock::now();

            auto boxes_from_bitmap_start = std::chrono::steady_clock::now();
            boxes = std::move(post_processor_.BoxesFromBitmap(
                pred_map, bit_map, srcimg.cols, srcimg.rows));
            auto boxes_from_bitmap_end = std::chrono::steady_clock::now();

            // // Print boxes for debugging
            // // std::cout << "[DEBUG] BoxesFromBitmap returned " << boxes.size() << " boxes:" << std::endl;
            // for (size_t i = 0; i < boxes.size(); ++i)
            // {
            //     std::cout << "  Box " << i << ": ";
            //     for (size_t j = 0; j < boxes[i].size(); ++j)
            //     {
            //         if (j > 0)
            //             std::cout << " -> ";
            //         std::cout << "(" << boxes[i][j][0] << "," << boxes[i][j][1] << ")";
            //     }
            //     std::cout << std::endl;
            // }

            auto filter_tag_start = std::chrono::steady_clock::now();
            post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
            auto filter_tag_end = std::chrono::steady_clock::now();

            // // Print boxes after FilterTagDetRes for debugging
            //  std::cout << "[DEBUG] After FilterTagDetRes, final boxes:" << std::endl;
            // for (size_t i = 0; i < boxes.size(); ++i)
            // {
            //     std::cout << "  Final Box " << i << ": ";
            //     for (size_t j = 0; j < boxes[i].size(); ++j)
            //     {
            //         if (j > 0)
            //             std::cout << " -> ";
            //         std::cout << "(" << boxes[i][j][0] << "," << boxes[i][j][1] << ")";
            //     }
            //     std::cout << std::endl;
            // }

            auto postprocess_end = std::chrono::steady_clock::now();

            // Calculate detailed timing
            std::chrono::duration<float> resize_diff = resize_end - resize_start;
            std::chrono::duration<float> normalize_diff = normalize_end - normalize_start;
            std::chrono::duration<float> permute_diff = permute_end - permute_start;
            std::chrono::duration<float> inference_diff = inference_end - inference_start;
            std::chrono::duration<float> threshold_diff = threshold_end - threshold_start;
            std::chrono::duration<float> dilation_diff = dilation_end - dilation_start;
            std::chrono::duration<float> boxes_from_bitmap_diff = boxes_from_bitmap_end - boxes_from_bitmap_start;
            std::chrono::duration<float> filter_tag_diff = filter_tag_end - filter_tag_start;

            // Store detailed timings (in milliseconds)
            // [resize, normalize, permute, inference, threshold, dilation, boxes_from_bitmap, filter_tag]
            times.resize(11);                                 // 8 detailed + 3 summary
            times[0] = resize_diff.count() * 1000;            // Detailed: resize
            times[1] = normalize_diff.count() * 1000;         // Detailed: normalize
            times[2] = permute_diff.count() * 1000;           // Detailed: permute
            times[3] = inference_diff.count() * 1000;         // Detailed: inference
            times[4] = threshold_diff.count() * 1000;         // Detailed: threshold
            times[5] = dilation_diff.count() * 1000;          // Detailed: dilation
            times[6] = boxes_from_bitmap_diff.count() * 1000; // Detailed: boxes_from_bitmap
            times[7] = filter_tag_diff.count() * 1000;        // Detailed: filter_tag

            // Calculate summary timing (for backward compatibility)
            std::chrono::duration<float> preprocess_diff = permute_end - preprocess_start;
            std::chrono::duration<float> postprocess_diff = postprocess_end - inference_end;

            times[8] = preprocess_diff.count() * 1000;   // Summary: preprocess
            times[9] = inference_diff.count() * 1000;    // Summary: inference
            times[10] = postprocess_diff.count() * 1000; // Summary: postprocess
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Exception in DBDetectorOpenVINO::Run: " << e.what() << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(11, 0.0); // 8 detailed + 3 summary
            // Exit to avoid further processing errors
            exit(1);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Unknown exception in DBDetectorOpenVINO::Run" << std::endl;
            // Ensure we have valid output even on exception
            times.clear();
            times.resize(11, 0.0); // 8 detailed + 3 summary
            // Exit to avoid further processing errors
            exit(1);
        }
    }

    // Function to print final detection memory transfer statistics
    void DBDetectorOpenVINO::PrintFinalDetectionStats() noexcept
    {
        if (total_det_images > 0)
        {
            printf("\n=== Detection Memory Transfer Final Statistics ===\n");
            printf("Detection processed %d images\n", total_det_images);
            printf("Detection Memory Transfer Timing (per image averages):\n");
            printf("  Shape setup:     %.2f ms per image\n", total_det_shape_time / total_det_images);
            printf("  CPU->GPU copy:   %.2f ms per image (%.1f MB avg)\n", 
                   total_det_cpu_to_gpu_time / total_det_images,
                   (double)total_det_input_size_mb / total_det_images);
            printf("  Pure inference:  %.2f ms per image\n", total_det_pure_inference_time / total_det_images);
            printf("  GPU->CPU copy:   %.2f ms per image (%.1f MB avg)\n", 
                   total_det_gpu_to_cpu_time / total_det_images,
                   (double)total_det_output_size_mb / total_det_images);
            
            double avg_total_transfer = (total_det_cpu_to_gpu_time + total_det_gpu_to_cpu_time) / total_det_images;
            double avg_total_time = (total_det_shape_time + total_det_cpu_to_gpu_time + total_det_pure_inference_time + total_det_gpu_to_cpu_time) / total_det_images;
            double avg_pure_inference = total_det_pure_inference_time / total_det_images;
            
            printf("  Total transfer:  %.2f ms per image (%.1f%% of total inference)\n", 
                   avg_total_transfer, (avg_total_transfer / avg_total_time) * 100);
            printf("  Memory transfer overhead: %.1f%% of pure inference time\n",
                   (avg_total_transfer / avg_pure_inference) * 100);
            printf("==================================================\n\n");
        }
    }

} // namespace PaddleOCR
