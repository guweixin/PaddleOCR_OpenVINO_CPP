// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "openvino_infer.h"
#include "src/utils/utility.h"
#include <iostream>

OpenVinoInfer::OpenVinoInfer(const std::string &model_name,
                             const std::string &model_dir,
                             const std::string &model_file_prefix,
                             const OpenVinoOption &option)
    : model_name_(model_name), model_dir_(model_dir), 
      model_file_prefix_(model_file_prefix), option_(option),
      is_detector_(false), is_recognizer_(false) {
  std::cout << "[INFO] OpenVinoInfer initialized for model: " << model_name_ << std::endl;
  
  // Initialize the model immediately
  auto status = Create();
  if (!status.ok()) {
    std::cout << "[ERROR] Failed to create OpenVINO model: " << status.ToString() << std::endl;
    // For now, we'll continue but the Apply method will fail
  }
}

// Scale and pad helper: equal-proportion scale to fit within target, then zero-pad
StatusOr<std::tuple<cv::Mat, float, float>> OpenVinoInfer::ScaleAndPad(const cv::Mat &image, int target_h, int target_w) const {
  if (image.empty()) return Status::InvalidArgumentError("Empty image for ScaleAndPad");
  if (target_h <= 0 || target_w <= 0) return Status::InvalidArgumentError("Invalid target dimensions for ScaleAndPad");

  int src_h = image.rows;
  int src_w = image.cols;
  if (src_h <= 0 || src_w <= 0) return Status::InvalidArgumentError("Invalid source image dimensions");

  float scale_h = static_cast<float>(target_h) / static_cast<float>(src_h);
  float scale_w = static_cast<float>(target_w) / static_cast<float>(src_w);
  float scale = std::min(scale_h, scale_w);

  int new_h = static_cast<int>(std::round(src_h * scale));
  int new_w = static_cast<int>(std::round(src_w * scale));
  if (new_h > target_h) new_h = target_h;
  if (new_w > target_w) new_w = target_w;

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat result = cv::Mat::zeros(target_h, target_w, image.type());
  int offset_y = (target_h - new_h) / 2;
  int offset_x = (target_w - new_w) / 2;
  cv::Rect roi(offset_x, offset_y, new_w, new_h);
  resized.copyTo(result(roi));

  float ratio_h = 1.0f / scale;
  float ratio_w = 1.0f / scale;
  return std::make_tuple(result, ratio_h, ratio_w);
}

Status OpenVinoInfer::Create() {
  try {
    // Find OpenVINO format files (.xml and .bin)
    is_detector_ = model_name_.find("det") != std::string::npos || model_name_.find("Det") != std::string::npos;
    is_recognizer_ = model_name_.find("rec") != std::string::npos || model_name_.find("Rec") != std::string::npos;
    
    std::string model_path;
    std::string weights_path;

    // Set device
    std::string device;
    if (option_.DeviceType() == "gpu"){ device = "GPU";}
    else if (option_.DeviceType() == "npu") {device = "NPU"; }// OpenVINO NPU plugin
    else{device = "CPU";}
    
    // Configure compilation options
    ov::AnyMap config;
    if (device == "CPU") {
      config["NUM_STREAMS"] = std::to_string(option_.CpuThreads());
    } else{
      config["INFERENCE_PRECISION_HINT"] = "f16";
      config["CACHE_DIR"] = "./openvino_cache";
    }

    if (device == "NPU" && is_detector_){
      model_path = model_dir_ + "/inference_960.xml";
      weights_path = model_dir_ + "/inference_960.bin";
    }else if(device == "NPU" && is_recognizer_){
        struct RecSpec { NPURecModelSize size; const char* file; } specs[] = {
          {NPURecModelSize::SMALL, "inference_480"},
          {NPURecModelSize::MEDIUM, "inference_800"},
          {NPURecModelSize::LARGE, "inference_1280"}
        };

        for (const auto &s : specs) {
          std::string rec_path = model_dir_ + "/" + s.file + ".xml";
          std::string rec_weights = model_dir_ + "/" + s.file + ".bin";
          if (!Utility::FileExists(rec_path).ok() || !Utility::FileExists(rec_weights).ok()) {
            std::cout << "[WARNING] NPU recognition model files not found: " << rec_path << std::endl;
            continue;
          }
          try {
            auto rec_model = core_.read_model(rec_path, rec_weights);
            auto compiled = core_.compile_model(rec_model, device);
            npu_compiled_models_[s.size] = compiled;
            npu_infer_requests_[s.size] = compiled.create_infer_request();
            std::cout << "[INFO] NPU recognition model loaded: " << rec_path << std::endl;
            
            // Set the first loaded model as the main model for compatibility
            if (!model_) {
              model_ = rec_model;
              compiled_model_ = compiled;
              infer_request_ = compiled.create_infer_request();
            }
          } catch (const std::exception &e) {
            std::cout << "[WARNING] Failed to load NPU recognition model " << rec_path << ": " << e.what() << std::endl;
          }      
        }
        
        // Ensure at least one model was loaded
        if (npu_compiled_models_.empty()) {
          return Status::InternalError("No NPU recognition models could be loaded");
        }
        
        // Get input/output names from the main model    
        for (const auto& input : model_->inputs()) {
          std::string input_name = input.get_any_name();
          input_names_.push_back(input_name);
        }
        for (const auto& output : model_->outputs()) {
          std::string output_name = output.get_any_name();
          output_names_.push_back(output_name);
        }
        
        if (input_names_.empty()) {
          std::cout << "[ERROR] No input tensors found in the NPU recognition model!" << std::endl;
          return Status::InternalError("No input tensors found in the NPU recognition model");
        }
        
        std::cout << "[INFO] NPU recognition models loaded successfully. Device: " << device 
                  << ", Models: " << npu_compiled_models_.size() 
                  << ", Inputs: " << input_names_.size() << ", Outputs: " << output_names_.size() << std::endl;
                  
        return Status::OK();
    }
    else{
      model_path = model_dir_ + "/" + model_file_prefix_ + ".xml";
      weights_path = model_dir_ + "/" + model_file_prefix_ + ".bin";
    }
    
    // Check if OpenVINO files exist
    auto model_status = Utility::FileExists(model_path);
    if (!model_status.ok()) {
      std::cout << "[ERROR] OpenVINO model file not found: " << model_path << std::endl;
      return Status::NotFoundError("OpenVINO model file not found: " + model_path);
    }
    
    auto weights_status = Utility::FileExists(weights_path);
    if (!weights_status.ok()) {
      std::cout << "[ERROR] OpenVINO weights file not found: " << weights_path << std::endl;
      return Status::NotFoundError("OpenVINO weights file not found: " + weights_path);
    }
    
    // Load the model
    model_ = core_.read_model(model_path, weights_path);
    if (!model_) {
      std::cout << "[ERROR] Failed to read OpenVINO model" << std::endl;
      return Status::InternalError("Failed to read OpenVINO model");
    }
    
    // Compile the model
    compiled_model_ = core_.compile_model(model_, device, config);
    infer_request_ = compiled_model_.create_infer_request();
    
    // Get input/output names    
    for (const auto& input : model_->inputs()) {
      std::string input_name = input.get_any_name();
      input_names_.push_back(input_name);
    }
    for (const auto& output : model_->outputs()) {
      std::string output_name = output.get_any_name();
      output_names_.push_back(output_name);
    }
    
    if (input_names_.empty()) {
      std::cout << "[ERROR] No input tensors found in the model!" << std::endl;
      return Status::InternalError("No input tensors found in the model");
    }
    
    std::cout << "[INFO] OpenVINO model loaded successfully. Device: " << device 
              << ", Inputs: " << input_names_.size() << ", Outputs: " << output_names_.size() << std::endl;
          
    return Status::OK();
    
  } catch (const std::exception& e) {
    return Status::InternalError("OpenVINO error: " + std::string(e.what()));
  }
}

Status OpenVinoInfer::CheckRunMode() {
  std::cout << "[INFO] Using OpenVINO inference engine" << std::endl;
  return Status::OK();
}

StatusOr<std::vector<cv::Mat>>
OpenVinoInfer::Apply(const std::vector<cv::Mat> &input_mats) {
  if (input_mats.empty()) {
    return Status::InvalidArgumentError("Input matrices are empty");
  }
  
  std::cout << "[DEBUG] Apply called with " << input_mats.size() << " input images" << std::endl;
  std::cout << "[DEBUG] Device type: " << option_.DeviceType() << ", is_recognizer: " << is_recognizer_ << ", is_detector: " << is_detector_ << std::endl;
  
  // Debug first input image
  if (!input_mats.empty()) {
    const auto& first = input_mats[0];
    std::cout << "[DEBUG] First input image: dims=" << first.dims 
              << ", rows=" << first.rows << ", cols=" << first.cols 
              << ", channels=" << first.channels() << ", type=" << first.type() 
              << ", empty=" << first.empty() << std::endl;
  }
  
  try {
    // Prepare input tensor
    if (input_names_.empty()) {
      return Status::InternalError("No input tensors found");
    }
    
    size_t batch_size = input_mats.size();
    
    // Get the first input for shape analysis
    auto input_partial_shape = model_->input().get_partial_shape();
    
    // NPU recognition: Force batch size = 1, process each image individually
    if (option_.DeviceType() == "npu" && is_recognizer_) {
      std::vector<cv::Mat> all_outputs;
      last_npu_ratios_.clear();
      
      std::cout << "[DEBUG] NPU recognition: Force batch size=1, processing " << input_mats.size() << " images individually" << std::endl;
      
      // Process each image with batch size = 1
      for (size_t img_idx = 0; img_idx < input_mats.size(); ++img_idx) {
        const auto &img = input_mats[img_idx];
        
        // Validate input image
        if (img.empty()) {
          std::cout << "[ERROR] Empty input image " << img_idx << std::endl;
          return Status::InvalidArgumentError("Empty input image");
        }
        
        // Extract source dimensions for aspect ratio calculation
        int src_h = 0, src_w = 0;
        if (img.dims == 3) {
          if (img.size[0] == 3 || img.size[0] == 1) {
            // CHW format: [C, H, W]
            src_h = static_cast<int>(img.size[1]);
            src_w = static_cast<int>(img.size[2]);
          } else {
            // HWC format: [H, W, C]
            src_h = static_cast<int>(img.size[0]);
            src_w = static_cast<int>(img.size[1]);
          }
        } else if (img.dims == 2) {
          src_h = img.rows;
          src_w = img.cols;
        } else {
          return Status::InvalidArgumentError("Unsupported image format for NPU recognition");
        }
        
        if (src_h <= 0 || src_w <= 0) {
          std::cout << "[ERROR] Invalid source dimensions: " << src_w << "x" << src_h << std::endl;
          return Status::InvalidArgumentError("Invalid source image dimensions");
        }
        
        // Calculate aspect ratio for model selection
        float aspect_ratio = static_cast<float>(src_w) / static_cast<float>(src_h);
        
        // Model specifications and thresholds
        const int target_height = 48;  // Standard OCR recognition height
        const int small_width = 480;   // Small model width
        const int medium_width = 800;  // Medium model width
        const int large_width = 1280;  // Large model width
        
        // Calculate thresholds: model_width / target_height
        float small_threshold = static_cast<float>(small_width) / static_cast<float>(target_height);    // 480/48 = 10.0
        float medium_threshold = static_cast<float>(medium_width) / static_cast<float>(target_height);  // 800/48 = 16.67
        
        // Select model type based on aspect ratio comparison
        NPURecModelSize selected_model_type;
        int target_w = 0;
        
        if (aspect_ratio <= small_threshold) {
          selected_model_type = NPURecModelSize::SMALL;  // model_type = 0
          target_w = small_width;
        } else if (aspect_ratio <= medium_threshold) {
          selected_model_type = NPURecModelSize::MEDIUM; // model_type = 1
          target_w = medium_width;
        } else {
          selected_model_type = NPURecModelSize::LARGE;  // model_type = 2
          target_w = large_width;
        }
        
        std::cout << "[DEBUG] NPU image " << img_idx << ": src(" << src_w << "x" << src_h 
                  << "), aspect_ratio=" << aspect_ratio 
                  << ", thresholds(small=" << small_threshold << ", medium=" << medium_threshold << ")"
                  << ", selected_model=" << static_cast<int>(selected_model_type) 
                  << ", target_size(" << target_w << "x" << target_height << ")" << std::endl;
        
        // Check if the selected model is available
        if (npu_compiled_models_.find(selected_model_type) == npu_compiled_models_.end() ||
            npu_infer_requests_.find(selected_model_type) == npu_infer_requests_.end()) {
          return Status::InternalError("Selected NPU recognition model not available");
        }
        
        auto& selected_infer_request = npu_infer_requests_[selected_model_type];
        
        // NPU recognition preprocessing: direct normalization (no reshape, no resize)
        // Record mapping ratios (no scaling applied)
        float ratio_h = 1.0f;
        float ratio_w = 1.0f;
        last_npu_ratios_.push_back({ratio_h, ratio_w});

        // Standard recognition preprocessing (same as CPU)
        // 1. Convert to float32 and normalize: (pixel / 255.0 - 0.5) / 0.5
        cv::Mat normalized_img;
        img.convertTo(normalized_img, CV_32F, 1.0/255.0);
        normalized_img = (normalized_img - cv::Scalar(0.5, 0.5, 0.5)) / cv::Scalar(0.5, 0.5, 0.5);
        
        // Debug output for image dimensions
        std::string size_info;
        if (img.dims == 2) {
          size_info = std::to_string(img.cols) + "x" + std::to_string(img.rows);
        } else if (img.dims == 3) {
          size_info = "[" + std::to_string(img.size[0]) + "," + 
                      std::to_string(img.size[1]) + "," + 
                      std::to_string(img.size[2]) + "]";
        } else {
          size_info = "dims=" + std::to_string(img.dims);
        }
        
        std::cout << "[DEBUG] NPU preprocessing " << img_idx << ": img shape=" << size_info
                  << ", type=" << img.type()
                  << ", normalized range=[" << *std::min_element(reinterpret_cast<const float*>(normalized_img.data), 
                     reinterpret_cast<const float*>(normalized_img.data) + normalized_img.total() * normalized_img.channels())
                  << ", " << *std::max_element(reinterpret_cast<const float*>(normalized_img.data), 
                     reinterpret_cast<const float*>(normalized_img.data) + normalized_img.total() * normalized_img.channels()) << "]" << std::endl;
        
        // 2. Handle CHW/HWC format for channel splitting
        std::vector<cv::Mat> channels;
        
        if (normalized_img.dims == 3) {
          // 3D tensor - check format and split accordingly
          if (normalized_img.size[0] == 3 || normalized_img.size[0] == 1) {
            // CHW format: already in CHW, extract channels differently
            int num_channels = normalized_img.size[0];
            int height = normalized_img.size[1];
            int width = normalized_img.size[2];
            
            for (int c = 0; c < num_channels; ++c) {
              cv::Range ranges[3] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
              cv::Mat channel = normalized_img(ranges).clone();
              channel = channel.reshape(1, height);  // Convert to 2D [H, W]
              channels.push_back(channel);
            }
            
            std::cout << "[DEBUG] NPU CHW tensor split into " << channels.size() << " channels" << std::endl;
          } else {
            // HWC format: use normal split
            cv::split(normalized_img, channels);
            std::cout << "[DEBUG] NPU HWC tensor split into " << channels.size() << " channels" << std::endl;
          }
        } else {
          // 2D image: use normal split
          cv::split(normalized_img, channels);
          std::cout << "[DEBUG] NPU 2D image split into " << channels.size() << " channels" << std::endl;
        }
        
        // Validate channel count
        if (channels.size() != 3) {
          std::cout << "[ERROR] Expected 3 channels, got " << channels.size() << std::endl;
          return Status::InvalidArgumentError("Invalid channel count for NPU recognition");
        }
        
        // 3. Create CHW tensor: [C, H, W] using original image dimensions
        int original_height, original_width;
        
        // Get correct dimensions based on normalized_img format
        if (normalized_img.dims == 3) {
          if (normalized_img.size[0] == 3 || normalized_img.size[0] == 1) {
            // CHW format: [C, H, W]
            original_height = static_cast<int>(normalized_img.size[1]);
            original_width = static_cast<int>(normalized_img.size[2]);
          } else {
            // HWC format: [H, W, C]
            original_height = static_cast<int>(normalized_img.size[0]);
            original_width = static_cast<int>(normalized_img.size[1]);
          }
        } else if (normalized_img.dims == 2) {
          // 2D format: [H, W]
          original_height = normalized_img.rows;
          original_width = normalized_img.cols;
        } else {
          std::cout << "[ERROR] Unsupported normalized_img format: dims=" << normalized_img.dims << std::endl;
          return Status::InvalidArgumentError("Unsupported normalized image format");
        }
        
        if (original_height <= 0 || original_width <= 0) {
          std::cout << "[ERROR] Invalid original dimensions: " << original_width << "x" << original_height << std::endl;
          return Status::InvalidArgumentError("Invalid original image dimensions");
        }
        
        int total_elements = original_height * original_width;
        
        // Create CHW data in proper order: RRRRR...GGGGG...BBBBB...
        std::vector<float> chw_data(3 * total_elements);
        
        for (int ch = 0; ch < 3; ++ch) {
          const float* channel_data = reinterpret_cast<const float*>(channels[ch].data);
          std::memcpy(&chw_data[ch * total_elements], channel_data, total_elements * sizeof(float));
        }
        
        std::cout << "[DEBUG] NPU CHW conversion " << img_idx << ": original_size(" << original_width << "x" << original_height 
                  << "), created CHW data, size=" << chw_data.size() << ", elements_per_channel=" << total_elements << std::endl;
        
        // 4. Create OpenVINO input tensor with original dimensions (batch size = 1)
        ov::Shape tensor_shape = {1, 3, static_cast<size_t>(original_height), static_cast<size_t>(original_width)};
        ov::Tensor input_tensor(ov::element::f32, tensor_shape);
        
        float* tensor_data = input_tensor.data<float>();
        std::memcpy(tensor_data, chw_data.data(), chw_data.size() * sizeof(float));
        
        // 5. Set input tensor and perform inference (batch size = 1)
        selected_infer_request.set_input_tensor(input_tensor);
        
        std::cout << "[DEBUG] NPU inference " << img_idx << ": input tensor shape[" 
                  << tensor_shape[0] << "," << tensor_shape[1] << "," << tensor_shape[2] << "," << tensor_shape[3] << "]"
                  << ", model=" << static_cast<int>(selected_model_type) << std::endl;
        
        try {
          selected_infer_request.infer();
          std::cout << "[DEBUG] NPU inference " << img_idx << ": completed successfully" << std::endl;
        } catch (const std::exception& e) {
          std::cout << "[ERROR] NPU inference " << img_idx << " failed: " << e.what() << std::endl;
          return Status::InternalError("NPU recognition inference failed: " + std::string(e.what()));
        }
        
        // 6. Get output tensor
        ov::Tensor output_tensor = selected_infer_request.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        const float* output_data = output_tensor.data<float>();
        
        std::cout << "[DEBUG] NPU output " << img_idx << ": shape[";
        for (size_t i = 0; i < output_shape.size(); ++i) {
          std::cout << output_shape[i];
          if (i < output_shape.size() - 1) std::cout << ",";
        }
        std::cout << "], elements=" << output_tensor.get_size() << std::endl;
        
        // Check output data validity
        if (output_tensor.get_size() > 0) {
          float min_val = *std::min_element(output_data, output_data + output_tensor.get_size());
          float max_val = *std::max_element(output_data, output_data + output_tensor.get_size());
          std::cout << "[DEBUG] NPU output " << img_idx << ": data_range=[" << min_val << ", " << max_val << "]" << std::endl;
        }
        
        // 7. Convert output to cv::Mat for further processing
        if (output_shape.size() == 3) {
          // Expected shape: [batch=1, sequence_length, num_classes]
          int seq_len = static_cast<int>(output_shape[1]);
          int num_classes = static_cast<int>(output_shape[2]);
          
          if (output_shape[0] != 1) {
            std::cout << "[WARNING] NPU output batch size != 1: " << output_shape[0] << std::endl;
          }
          
          cv::Mat output_mat(seq_len, num_classes, CV_32F);
          std::memcpy(output_mat.data, output_data, seq_len * num_classes * sizeof(float));
          
          all_outputs.push_back(output_mat);
          
          std::cout << "[DEBUG] NPU recognition output " << img_idx 
                    << ": batch_size=" << output_shape[0] << ", shape[" << seq_len << "x" << num_classes << "]"
                    << ", output_mat created successfully" << std::endl;
        } else {
          return Status::InternalError("Unsupported NPU recognition output shape");
        }
      }
      
      return all_outputs;
    }
    
    // Standard preprocessing for CPU/GPU or NPU detection
    std::vector<cv::Mat> processed_mats = input_mats;
    last_npu_ratios_.clear();
    
    std::cout << "[DEBUG] Apply called with " << input_mats.size() << " input images" << std::endl;
    std::cout << "[DEBUG] Device type: " << option_.DeviceType() 
              << ", is_recognizer: " << is_recognizer_ << ", is_detector: " << is_detector_ << std::endl;
    
    if (!input_mats.empty()) {
      const cv::Mat& first_input = input_mats[0];
      std::cout << "[DEBUG] First input image: dims=" << first_input.dims 
                << ", rows=" << first_input.rows << ", cols=" << first_input.cols 
                << ", channels=" << first_input.channels() << ", type=" << first_input.type() 
                << ", empty=" << first_input.empty() << std::endl;
    }
    
    if (option_.DeviceType() == "npu" && is_detector_) {
      // For NPU detection, check if input is already preprocessed (4D tensor)
      const cv::Mat& first_input = input_mats[0];
      
      if (first_input.dims == 4) {
        // Input is already preprocessed as 4D tensor, no need for NPU scale/pad
        std::cout << "[DEBUG] NPU detection: Input already preprocessed as 4D tensor, skipping NPU scale/pad" << std::endl;
        processed_mats = input_mats; // Use input as-is
        // For 4D tensors, we don't apply NPU preprocessing, but we still need to initialize ratios
        for (size_t i = 0; i < input_mats.size(); ++i) {
          last_npu_ratios_.push_back({1.0f, 1.0f}); // No scaling applied
        }
      } else if (first_input.dims == 2 && first_input.rows > 0 && first_input.cols > 0) {
        // Input is 2D image, apply NPU preprocessing
        std::cout << "[DEBUG] NPU detection: Input is 2D image, applying NPU scale/pad" << std::endl;
        
        auto input_port = model_->input(0);
        auto shape = input_port.get_shape();
        int target_h = 0, target_w = 0;
        
        if (shape.size() == 4) {
          target_h = static_cast<int>(shape[2]);
          target_w = static_cast<int>(shape[3]);
        } else if (shape.size() == 3) {
          target_h = static_cast<int>(shape[1]);
          target_w = static_cast<int>(shape[2]);
        } else if (shape.size() == 2) {
          target_h = static_cast<int>(shape[0]);
          target_w = static_cast<int>(shape[1]);
        } else {
          return Status::InvalidArgumentError("Unsupported model input shape for NPU detection");
        }
        
        processed_mats.clear();
        for (const auto &img : input_mats) {
          auto result = ScaleAndPad(img, target_h, target_w);
          if (!result.ok()) {
            return result.status();
          }
          
          cv::Mat padded_image;
          float ratio_h, ratio_w;
          std::tie(padded_image, ratio_h, ratio_w) = result.value();
          
          processed_mats.push_back(padded_image);
          last_npu_ratios_.push_back({ratio_h, ratio_w});
        }
      } else {
        std::cout << "[DEBUG] NPU detection: Unsupported input format, using as-is" << std::endl;
        // Initialize ratios for unknown format
        for (size_t i = 0; i < input_mats.size(); ++i) {
          last_npu_ratios_.push_back({1.0f, 1.0f});
        }
      }
    }
    
    // Analyze the input data format (use processed_mats after NPU preprocessing)
    const cv::Mat& first_mat = processed_mats[0];
    
    // Check if the Mat contains valid data
    // For multi-dimensional Mat (dims >= 3), don't check rows/cols as they may be -1
    bool valid_data = false;
    if (first_mat.dims >= 3) {
      // For multi-dimensional Mat, check if all dimensions are positive
      valid_data = first_mat.data && first_mat.total() > 0;
      for (int i = 0; i < first_mat.dims; ++i) {
        if (first_mat.size[i] <= 0) {
          valid_data = false;
          break;
        }
      }
    } else {
      // For 2D Mat, use traditional validation
      valid_data = first_mat.data && first_mat.rows > 0 && first_mat.cols > 0;
    }
    
    if (!valid_data) {
      std::cout << "[ERROR] Input Mat contains invalid data!" << std::endl;
      return Status::InvalidArgumentError("Input Mat contains invalid data");
    }
    
    // For dynamic shapes or when input size changes, we need to reshape the model
    ov::Shape input_shape;


    // Determine the input shape based on the actual data layout
    if (first_mat.dims == 4) {
      // Perfect! We have 4D NCHW format from ToBatch
      input_shape = {
        static_cast<size_t>(first_mat.size[0]),  // N (batch)
        static_cast<size_t>(first_mat.size[1]),  // C (channels)
        static_cast<size_t>(first_mat.size[2]),  // H (height)
        static_cast<size_t>(first_mat.size[3])   // W (width)
      };
    } else if (first_mat.dims == 3) {
      // 3D CHW format, add batch dimension
      input_shape = {
        batch_size,                               // N (batch)
        static_cast<size_t>(first_mat.size[0]),  // C (channels)
        static_cast<size_t>(first_mat.size[1]),  // H (height)
        static_cast<size_t>(first_mat.size[2])   // W (width)
      };
    } else if (first_mat.channels() == 3 && first_mat.rows > 0 && first_mat.cols > 0) {
      // Standard 2D HWC format: convert to NCHW
      input_shape = {batch_size, 3, static_cast<size_t>(first_mat.rows), static_cast<size_t>(first_mat.cols)};
    } else if (first_mat.channels() == 1 && first_mat.total() > 0) {
      // Flattened CHW data - need to determine dimensions
      size_t total_elements = first_mat.total();
      
      // For most OCR models, we expect CHW format with 3 channels
      if (total_elements % 3 == 0) {
        size_t hw_size = total_elements / 3;
        // Try to find reasonable height and width
        size_t height, width;
        
        // Common OCR input sizes
        if (hw_size == 64 * 64) {
          height = width = 64;
        } else if (hw_size == 32 * 32) {
          height = width = 32;
        } else if (hw_size == 48 * 192) {
          height = 48; width = 192;
        } else if (hw_size == 32 * 128) {
          height = 32; width = 128;
        } else {
          // Fallback: try to make it square-ish
          height = static_cast<size_t>(std::sqrt(hw_size));
          width = hw_size / height;
        }
        
        input_shape = {batch_size, 3, height, width};
      } else {
        std::cout << "[ERROR] Cannot determine shape from flattened data with " << total_elements << " elements" << std::endl;
        return Status::InvalidArgumentError("Cannot determine input shape from data");
      }
    } else {
      std::cout << "[ERROR] Unsupported input data format" << std::endl;
      return Status::InvalidArgumentError("Unsupported input data format");
    }
    
    // Check if we need to reshape (either dynamic model or shape mismatch)
    bool need_reshape = input_partial_shape.is_dynamic();
    if (!need_reshape) {
      // For static models, check if current tensor shape matches required shape
      auto current_tensor = infer_request_.get_input_tensor();
      auto current_shape = current_tensor.get_shape();
      need_reshape = (current_shape != input_shape);
    }
    
    if (need_reshape) {
      try {
        model_->reshape({{input_names_[0], input_shape}});
        compiled_model_ = core_.compile_model(model_, (option_.DeviceType() == "gpu") ? "GPU" : "CPU");
        infer_request_ = compiled_model_.create_infer_request();
      } catch (const std::exception& e) {
        std::cout << "[ERROR] Failed to reshape model: " << e.what() << std::endl;
        return Status::InternalError("Failed to reshape model: " + std::string(e.what()));
      }
    }
    
    // Create input tensor
    ov::Tensor input_tensor = infer_request_.get_input_tensor();
    auto tensor_shape = input_tensor.get_shape();
    
    size_t tensor_size = input_tensor.get_size();
    
    // Copy input data
    float* input_data = input_tensor.data<float>();
    
    // For batch data (dims == 4), we expect only one Mat containing all batches
    if (batch_size == 1 && first_mat.dims == 4) {
      // Single Mat contains all batch data - copy directly
      size_t total_elements = first_mat.total();
      
      if (total_elements != tensor_size) {
        std::cout << "[ERROR] Data size mismatch: Mat=" << total_elements 
                  << ", Tensor=" << tensor_size << std::endl;
        return Status::InternalError("Data size mismatch between Mat and tensor");
      }
      
      // Copy data directly
      if (first_mat.type() == CV_32F) {
        try {
          std::memcpy(input_data, first_mat.data, total_elements * sizeof(float));
        } catch (const std::exception& e) {
          std::cout << "[ERROR] Memory copy failed: " << e.what() << std::endl;
          return Status::InternalError("Memory copy failed: " + std::string(e.what()));
        }
      } else {
        cv::Mat float_mat;
        first_mat.convertTo(float_mat, CV_32F);
        try {
          std::memcpy(input_data, float_mat.data, total_elements * sizeof(float));
        } catch (const std::exception& e) {
          std::cout << "[ERROR] Memory copy (convert) failed: " << e.what() << std::endl;
          return Status::InternalError("Memory copy (convert) failed: " + std::string(e.what()));
        }
      }
    } else {
      // Multiple Mats or single 3D Mat - copy batch by batch
      // Calculate elements per batch correctly based on tensor shape
      size_t elements_per_batch = 1;
      if (tensor_shape.size() >= 4) {
        // NCHW format: skip batch dimension (index 0)
        for (size_t i = 1; i < tensor_shape.size(); ++i) {
          elements_per_batch *= tensor_shape[i];
        }
      } else {
        // Fallback: total size divided by batch size
        elements_per_batch = tensor_size / batch_size;
      }
      
      for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const cv::Mat& mat = processed_mats[batch_idx];
        
        // Calculate elements per sample based on Mat dimensions
        size_t mat_elements;
        if (mat.dims >= 3) {
          // For multi-dimensional Mat, calculate total elements directly
          mat_elements = mat.total();
        } else {
          // For 2D Mat, include channels
          mat_elements = mat.total() * mat.channels();
        }
        
        if (mat_elements != elements_per_batch) {
          std::cout << "[WARNING] Mat elements (" << mat_elements 
                    << ") != expected elements per batch (" << elements_per_batch << ")" << std::endl;
        }
        
        // Check memory bounds
        size_t total_offset = batch_idx * elements_per_batch;
        size_t required_memory = total_offset + elements_per_batch;
        
        if (required_memory > tensor_size) {
          std::cout << "[ERROR] Memory overflow! Required: " << required_memory 
                    << ", Available: " << tensor_size << std::endl;
          return Status::InternalError("Memory overflow in tensor copy");
        }
        
        // Copy data directly
        if (mat.type() == CV_32F) {
          try {
            std::memcpy(input_data + total_offset,
                        mat.data, elements_per_batch * sizeof(float));
          } catch (const std::exception& e) {
            std::cout << "[ERROR] Memory copy failed: " << e.what() << std::endl;
            return Status::InternalError("Memory copy failed: " + std::string(e.what()));
          }
        } else {
          cv::Mat float_mat;
          mat.convertTo(float_mat, CV_32F);
          try {
            std::memcpy(input_data + total_offset,
                        float_mat.data, elements_per_batch * sizeof(float));
          } catch (const std::exception& e) {
            std::cout << "[ERROR] Memory copy (convert) failed: " << e.what() << std::endl;
            return Status::InternalError("Memory copy (convert) failed: " + std::string(e.what()));
          }
        }
      }
    }
    
    // Run inference
    try {
      infer_request_.infer();
    } catch (const std::exception& e) {
      std::cout << "[ERROR] Inference failed: " << e.what() << std::endl;
      return Status::InternalError("Inference failed: " + std::string(e.what()));
    }
    
    std::vector<cv::Mat> output_mats;
    
    for (size_t output_idx = 0; output_idx < output_names_.size(); ++output_idx) {
      auto output_tensor = infer_request_.get_output_tensor(output_idx);
      auto output_shape = output_tensor.get_shape();
      
      // Create output cv::Mat
      if (output_shape.size() == 4) {
        // 4D tensor: [batch, channels, height, width]
        size_t out_batch = output_shape[0];
        size_t out_channels = output_shape[1];
        size_t out_height = output_shape[2];
        size_t out_width = output_shape[3];
        
        for (size_t b = 0; b < out_batch; ++b) {
          cv::Mat output_mat(static_cast<int>(out_height), static_cast<int>(out_width), CV_32FC(static_cast<int>(out_channels)));
          
          float* output_data = output_tensor.data<float>();
          
          std::memcpy(output_mat.data, 
                      output_data + b * out_channels * out_height * out_width,
                      out_channels * out_height * out_width * sizeof(float));
          
          output_mats.push_back(output_mat);
        }
      } else if (output_shape.size() == 2) {
        // 2D tensor: [batch, features]
        size_t out_batch = output_shape[0];
        size_t out_features = output_shape[1];
        
        for (size_t b = 0; b < out_batch; ++b) {
          cv::Mat output_mat(1, static_cast<int>(out_features), CV_32F);
          
          float* output_data = output_tensor.data<float>();
          
          std::memcpy(output_mat.data, 
                      output_data + b * out_features,
                      out_features * sizeof(float));
          
          output_mats.push_back(output_mat);
        }
      } else if (output_shape.size() == 3) {
        // 3D tensor: [batch, sequence, classes] - for text recognition
        size_t out_batch = output_shape[0];
        size_t out_sequence = output_shape[1];
        size_t out_classes = output_shape[2];
        
        // For 3D output, we create a 3D cv::Mat
        int sizes[] = {static_cast<int>(out_batch), static_cast<int>(out_sequence), static_cast<int>(out_classes)};
        cv::Mat output_mat(3, sizes, CV_32F);
        
        float* output_data = output_tensor.data<float>();
        
        size_t total_elements = out_batch * out_sequence * out_classes;
        std::memcpy(output_mat.data, output_data, total_elements * sizeof(float));
        
        output_mats.push_back(output_mat);
      } else {
        std::cout << "[ERROR] Unsupported output tensor shape with " << output_shape.size() << " dimensions" << std::endl;
        return Status::InternalError("Unsupported output tensor shape");
      }
    }
    
    return output_mats;
    
  } catch (const std::exception& e) {
    return Status::InternalError("OpenVINO inference error: " + std::string(e.what()));
  }
}
