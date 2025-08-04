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

    void DBDetectorOpenVINO::Run(cv::Mat &img,
                                 std::vector<std::vector<std::vector<int>>> &boxes,
                                 std::vector<double> &times)
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
                // NPU specific preprocessing: conditional resize with white padding
                // Get target dimensions from detection model input shape
                auto input_tensor = infer_request_.get_input_tensor();
                auto input_shape = input_tensor.get_shape();
                int target_h = static_cast<int>(input_shape[2]); // Height dimension
                int target_w = static_cast<int>(input_shape[3]); // Width dimension

                int original_h = img.rows;
                int original_w = img.cols;

                // Calculate scale to fit within target dimensions while preserving aspect ratio
                float scale_h = static_cast<float>(target_h) / static_cast<float>(original_h);
                float scale_w = static_cast<float>(target_w) / static_cast<float>(original_w);
                float scale = std::min(scale_h, scale_w);

                cv::Mat processed_img;
                float actual_scale = 1.0f;
                int start_x = 0, start_y = 0;

                if (scale < 1.0f)
                {
                    // Scale down if image is larger than target dimensions
                    actual_scale = scale;
                    int new_h = static_cast<int>(original_h * actual_scale);
                    int new_w = static_cast<int>(original_w * actual_scale);
                    cv::resize(img, processed_img, cv::Size(new_w, new_h));
                }
                else
                {
                    // Keep original size if image fits within target dimensions
                    processed_img = img.clone();
                    actual_scale = 1.0f;
                    scale = actual_scale;
                }

                // Create canvas with white background using model input dimensions
                resize_img = cv::Mat(target_h, target_w, CV_8UC3, cv::Scalar(255, 255, 255));

                // Place processed image at top-left corner (0, 0) instead of center
                start_x = 0;
                start_y = 0;

                // Copy processed image to the top-left corner of the canvas
                cv::Rect roi(start_x, start_y, processed_img.cols, processed_img.rows);
                processed_img.copyTo(resize_img(roi));

                // Calculate ratio values for NPU coordinate mapping
                if (scale < 1.0f)
                {
                    // Image was scaled down - need to map coordinates back
                    ratio_h = 1.0f / actual_scale;
                    ratio_w = 1.0f / actual_scale;
                }
                else
                {
                    // Image was only padded - coordinates don't need mapping
                    ratio_h = 1.0f;
                    ratio_w = 1.0f;
                }
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

            // Inference with OpenVINO - standard approach
            auto inference_start = std::chrono::steady_clock::now();

            // // Declare timing variables outside of conditional blocks
            // std::chrono::steady_clock::time_point shape_start, shape_end;
            // std::chrono::steady_clock::time_point cpu_to_gpu_start, cpu_to_gpu_end;
            // std::chrono::steady_clock::time_point pure_inference_start, pure_inference_end;
            // std::chrono::steady_clock::time_point gpu_to_cpu_start, gpu_to_cpu_end;
            ov::Shape output_shape;
            size_t out_num = 0;

            std::vector<float> out_data;

            auto input_tensor = infer_request_.get_input_tensor();

            // Set input shape efficiently
            // shape_start = std::chrono::steady_clock::now();
            ov::Shape target_shape = {1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols)};
            if (input_tensor.get_shape() != target_shape)
            {
                input_tensor.set_shape(target_shape);
            }
            // shape_end = std::chrono::steady_clock::now();

            // CPU to GPU data transfer timing
            // cpu_to_gpu_start = std::chrono::steady_clock::now();
            float *input_data = input_tensor.data<float>();
            if (input_data && !input.empty())
            {
                std::copy(input.begin(), input.end(), input_data);
            }
            // cpu_to_gpu_end = std::chrono::steady_clock::now();

            // Pure inference timing
            // pure_inference_start = std::chrono::steady_clock::now();
            infer_request_.infer();
            // pure_inference_end = std::chrono::steady_clock::now();

            // GPU to CPU data transfer timing
            // gpu_to_cpu_start = std::chrono::steady_clock::now();
            auto output_tensor = infer_request_.get_output_tensor();
            output_shape = output_tensor.get_shape();

            out_num = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
            out_data.reserve(out_num);
            out_data.resize(out_num);

            // Direct memory copy from GPU
            float *output_data = output_tensor.data<float>();
            std::copy(output_data, output_data + out_num, out_data.begin());
            // gpu_to_cpu_end = std::chrono::steady_clock::now();

            // // Calculate detailed memory transfer timings
            // std::chrono::duration<float, std::milli> shape_diff = shape_end - shape_start;
            // std::chrono::duration<float, std::milli> cpu_to_gpu_diff = cpu_to_gpu_end - cpu_to_gpu_start;
            // std::chrono::duration<float, std::milli> pure_inference_diff = pure_inference_end - pure_inference_start;
            // std::chrono::duration<float, std::milli> gpu_to_cpu_diff = gpu_to_cpu_end - gpu_to_cpu_start;

            // std::cout << "---------shape_time: " << std::fixed << std::setprecision(6) << shape_diff.count() << "ms" << std::endl;
            // std::cout << "---------cpu_to_gpu_time: " << std::fixed << std::setprecision(6) << cpu_to_gpu_diff.count() << "ms" << std::endl;
            // std::cout << "---------pure_infer_time: " << std::fixed << std::setprecision(6) << pure_inference_diff.count() << "ms" << std::endl;
            // std::cout << "---------gpu_to_cpu_time: " << std::fixed << std::setprecision(6) << gpu_to_cpu_diff.count() << "ms" << std::endl;
            // shape_time = std::chrono::duration<double, std::milli>(shape_end - shape_start).count();
            // cpu_to_gpu_time = std::chrono::duration<double, std::milli>(cpu_to_gpu_end - cpu_to_gpu_start).count();
            // pure_inference_time = std::chrono::duration<double, std::milli>(pure_inference_end - pure_inference_start).count();
            // gpu_to_cpu_time = std::chrono::duration<double, std::milli>(gpu_to_cpu_end - gpu_to_cpu_start).count();

            // // Calculate data sizes for statistics
            // input_size_mb = (input.size() * sizeof(float)) / (1024 * 1024);
            // output_size_mb = (out_num * sizeof(float)) / (1024 * 1024);

            // // Accumulate timing statistics for detection
            // total_det_shape_time += shape_time;
            // total_det_cpu_to_gpu_time += cpu_to_gpu_time;
            // total_det_pure_inference_time += pure_inference_time;
            // total_det_gpu_to_cpu_time += gpu_to_cpu_time;
            // total_det_input_size_mb += input_size_mb;
            // total_det_output_size_mb += output_size_mb;
            // total_det_images++;

            // auto inference_end = std::chrono::steady_clock::now();

            // // Postprocessing
            // auto postprocess_start = std::chrono::steady_clock::now();

            int n2 = static_cast<int>(output_shape[2]);
            int n3 = static_cast<int>(output_shape[3]);
            int n = n2 * n3;

            std::vector<float> pred(n, 0.0);
            std::vector<unsigned char> cbuf(n, ' ');

            for (int i = 0; i < n; ++i)
            {
                pred[i] = out_data[i];
                cbuf[i] = (unsigned char)((out_data[i]) * 255);
            }

            cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
            cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

            // auto threshold_start = std::chrono::steady_clock::now();
            const double threshold = this->det_db_thresh_ * 255;
            const double maxvalue = 255;
            cv::Mat bit_map;
            cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
            // auto threshold_end = std::chrono::steady_clock::now();

            // auto dilation_start = std::chrono::steady_clock::now();
            if (this->use_dilation_)
            {
                cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
                cv::dilate(bit_map, bit_map, dila_ele);
            }
            // auto dilation_end = std::chrono::steady_clock::now();

            // auto boxes_from_bitmap_start = std::chrono::steady_clock::now();
            if (device_ == "NPU")
            {
                if (ratio_h != 1.0f || ratio_w != 1.0f)
                {
                    // For NPU, apply ratio mapping to the detected boxes
                    boxes = std::move(post_processor_.BoxesFromBitmap(
                        pred_map, bit_map, srcimg.cols, srcimg.rows));
                }
                else
                {
                    auto input_tensor = infer_request_.get_input_tensor();
                    auto input_shape = input_tensor.get_shape();
                    int target_h = static_cast<int>(input_shape[2]); // Height dimension
                    int target_w = static_cast<int>(input_shape[3]); // Width dimension
                    // For NPU, use fixed dimensions for 960 model
                    boxes = std::move(post_processor_.BoxesFromBitmap(
                        pred_map, bit_map, target_h, target_w));
                }
            }
            else
            {
                // For CPU/GPU, use original image dimensions
                boxes = std::move(post_processor_.BoxesFromBitmap(
                    pred_map, bit_map, srcimg.cols, srcimg.rows));
            }
            // auto boxes_from_bitmap_end = std::chrono::steady_clock::now();

            // auto filter_tag_start = std::chrono::steady_clock::now();
            post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
            // auto filter_tag_end = std::chrono::steady_clock::now();

            // auto postprocess_end = std::chrono::steady_clock::now();

            // // Calculate detailed timing
            // std::chrono::duration<float> resize_diff = resize_end - resize_start;
            // std::chrono::duration<float> normalize_diff = normalize_end - normalize_start;
            // std::chrono::duration<float> permute_diff = permute_end - permute_start;
            // std::chrono::duration<float> inference_diff = inference_end - inference_start;
            // std::chrono::duration<float> threshold_diff = threshold_end - threshold_start;
            // std::chrono::duration<float> dilation_diff = dilation_end - dilation_start;
            // std::chrono::duration<float> boxes_from_bitmap_diff = boxes_from_bitmap_end - boxes_from_bitmap_start;
            // std::chrono::duration<float> filter_tag_diff = filter_tag_end - filter_tag_start;

            // // Store detailed timings (in milliseconds)
            // // [resize, normalize, permute, inference, threshold, dilation, boxes_from_bitmap, filter_tag]
            // times.resize(11);                                 // 8 detailed + 3 summary
            // times[0] = resize_diff.count() * 1000;            // Detailed: resize
            // times[1] = normalize_diff.count() * 1000;         // Detailed: normalize
            // times[2] = permute_diff.count() * 1000;           // Detailed: permute
            // times[3] = inference_diff.count() * 1000;         // Detailed: inference
            // times[4] = threshold_diff.count() * 1000;         // Detailed: threshold
            // times[5] = dilation_diff.count() * 1000;          // Detailed: dilation
            // times[6] = boxes_from_bitmap_diff.count() * 1000; // Detailed: boxes_from_bitmap
            // times[7] = filter_tag_diff.count() * 1000;        // Detailed: filter_tag

            // // Calculate summary timing (for backward compatibility)
            // std::chrono::duration<float> preprocess_diff = permute_end - preprocess_start;
            // std::chrono::duration<float> postprocess_diff = postprocess_end - inference_end;

            // times[8] = preprocess_diff.count() * 1000;   // Summary: preprocess
            // times[9] = inference_diff.count() * 1000;    // Summary: inference
            // times[10] = postprocess_diff.count() * 1000; // Summary: postprocess
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Exception in DBDetectorOpenVINO::Run: " << e.what() << std::endl;
            // Ensure we have valid output even on exception
            // times.clear();
            // times.resize(11, 0.0); // 8 detailed + 3 summary
            // Exit to avoid further processing errors
            exit(1);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Unknown exception in DBDetectorOpenVINO::Run" << std::endl;
            // Ensure we have valid output even on exception
            // times.clear();
            // times.resize(11, 0.0); // 8 detailed + 3 summary
            // Exit to avoid further processing errors
            exit(1);
        }
    }

    // Function to print final detection memory transfer statistics
    void DBDetectorOpenVINO::PrintFinalDetectionStats()
    {
        // if (total_det_images > 0)
        // {
        // printf("\n=== Detection Memory Transfer Final Statistics ===\n");
        // printf("Detection processed %d images\n", total_det_images);
        // printf("Detection Memory Transfer Timing (per image averages):\n");
        // printf("  Shape setup:     %.2f ms per image\n", total_det_shape_time / total_det_images);
        // printf("  CPU->GPU copy:   %.2f ms per image (%.1f MB avg)\n",
        //        total_det_cpu_to_gpu_time / total_det_images,
        //        (double)total_det_input_size_mb / total_det_images);
        // printf("  Pure inference:  %.2f ms per image\n", total_det_pure_inference_time / total_det_images);
        // printf("  GPU->CPU copy:   %.2f ms per image (%.1f MB avg)\n",
        //        total_det_gpu_to_cpu_time / total_det_images,
        //        (double)total_det_output_size_mb / total_det_images);

        // double avg_total_transfer = (total_det_cpu_to_gpu_time + total_det_gpu_to_cpu_time) / total_det_images;
        // double avg_total_time = (total_det_shape_time + total_det_cpu_to_gpu_time + total_det_pure_inference_time + total_det_gpu_to_cpu_time) / total_det_images;
        // double avg_pure_inference = total_det_pure_inference_time / total_det_images;

        // printf("  Total transfer:  %.2f ms per image (%.1f%% of total inference)\n",
        //        avg_total_transfer, (avg_total_transfer / avg_total_time) * 100);
        // printf("  Memory transfer overhead: %.1f%% of pure inference time\n",
        //        (avg_total_transfer / avg_pure_inference) * 100);
        // printf("==================================================\n\n");
        // }
    }

} // namespace PaddleOCR
