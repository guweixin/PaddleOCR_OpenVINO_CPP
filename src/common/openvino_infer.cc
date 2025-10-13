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
    ov::AnyMap config = {{"PERFORMANCE_HINT", "LATENCY"}};
    if (device == "CPU") {
      config["CPU_RUNTIME_CACHE_CAPACITY"] = "0";
    } else{
      config["INFERENCE_PRECISION_HINT"] = "f16";
      config["CACHE_DIR"] = "./openvino_cache";
    }

    if (device == "NPU" && is_detector_){
      model_path = model_dir_ + "/inference_960.xml";
      weights_path = model_dir_ + "/inference_960.bin";
    }else if(device == "NPU" && is_recognizer_){
        struct RecSpec { NPURecModelSize size; const char* file; } specs[] = {
          {NPURecModelSize::TINY, "inference_240"},
          {NPURecModelSize::SMALL, "inference_480"},
          // {NPURecModelSize::MEDIUM, "inference_440"},
          // {NPURecModelSize::BIG, "inference_560"},
          // {NPURecModelSize::LARGE, "inference_680"},
          // {NPURecModelSize::HUGE, "inference_800"},
          {NPURecModelSize::UNKNOWN, "inference_800"} 
        };

        for (const auto &s : specs) {
          std::string rec_path = model_dir_ + "/" + s.file + ".xml";
          std::string rec_weights = model_dir_ + "/" + "inference_static.bin";
          // std::string rec_weights = model_dir_ + "/" + s.file + ".bin";
          if (!Utility::FileExists(rec_path).ok() || !Utility::FileExists(rec_weights).ok()) {
            std::cout << "[WARNING] NPU recognition model files not found: " << rec_path << std::endl;
            continue;
          }
          try {
            auto rec_model = core_.read_model(rec_path, rec_weights);
            auto compiled = core_.compile_model(rec_model, device, config);
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
  
  try {
    // Prepare input tensor
    if (input_names_.empty()) {
      return Status::InternalError("No input tensors found");
    }
    
    size_t batch_size = input_mats.size();
    
    // Get the first input for shape analysis
    auto input_partial_shape = model_->input().get_partial_shape();
    
    // Standard preprocessing for CPU/GPU or NPU detection
    std::vector<cv::Mat> processed_mats = input_mats;
    last_npu_ratios_.clear();
    
    // std::cout << "[DEBUG] Device type: " << option_.DeviceType() 
    //           << ", is_recognizer: " << is_recognizer_ << ", is_detector: " << is_detector_ << std::endl;
          
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
    
    // If running on NPU and this is a recognizer, choose the precompiled model
    // that best matches the input width to avoid input tensor size mismatch.
    if (option_.DeviceType() == "npu" && is_recognizer_) {
      size_t in_w = 0;
      if (input_shape.size() >= 4) {
        in_w = input_shape[3];
      } else if (first_mat.dims == 4) {
        in_w = static_cast<size_t>(first_mat.size[3]);
      } else if (first_mat.dims == 3) {
        in_w = static_cast<size_t>(first_mat.size[2]);
      } else {
        in_w = static_cast<size_t>(first_mat.cols);
      }

      // choose npu rec model
      NPURecModelSize choose = NPURecModelSize::TINY;
      int tiny_model_width;
      int small_model_width;
      // int medium_model_width;
      // int big_model_width;
      // int large_model_width;
      // int huge_model_width;

      auto it_tiny = npu_compiled_models_.find(NPURecModelSize::TINY);
      if (it_tiny != npu_compiled_models_.end()) {
        auto shape = it_tiny->second.input(0).get_shape();
        if (shape.size() >= 4) tiny_model_width = static_cast<int>(shape[3]);  
      }
      auto it_small = npu_compiled_models_.find(NPURecModelSize::SMALL);
      if (it_small != npu_compiled_models_.end()) {
        auto shape = it_small->second.input(0).get_shape();
        if (shape.size() >= 4) small_model_width = static_cast<int>(shape[3]);  
      }
      // auto it_med = npu_compiled_models_.find(NPURecModelSize::MEDIUM);
      // if (it_med != npu_compiled_models_.end()) {
      //   auto shape = it_med->second.input(0).get_shape();
      //   if (shape.size() >= 4) medium_model_width = static_cast<int>(shape[3]);
      // }
      // auto it_big = npu_compiled_models_.find(NPURecModelSize::BIG);
      // if (it_big != npu_compiled_models_.end()) {
      //   auto shape = it_big->second.input(0).get_shape();
      //   if (shape.size() >= 4) big_model_width = static_cast<int>(shape[3]);  
      // }
      // auto it_large = npu_compiled_models_.find(NPURecModelSize::LARGE);
      // if (it_large != npu_compiled_models_.end()) {
      //   auto shape = it_large->second.input(0).get_shape();
      //   if (shape.size() >= 4) large_model_width = static_cast<int>(shape[3]);  
      // }
      // auto it_huge = npu_compiled_models_.find(NPURecModelSize::HUGE);
      // if (it_huge != npu_compiled_models_.end()) {
      //   auto shape = it_huge->second.input(0).get_shape();
      //   if (shape.size() >= 4) huge_model_width = static_cast<int>(shape[3]);  
      // }

      if (in_w <= tiny_model_width) choose = NPURecModelSize::TINY;
      else if (in_w <= small_model_width) choose = NPURecModelSize::SMALL;
      // else if (in_w <= medium_model_width) choose = NPURecModelSize::MEDIUM;
      // else if (in_w <= big_model_width) choose = NPURecModelSize::BIG;
      // else if (in_w <= large_model_width) choose = NPURecModelSize::LARGE;
      // else if (in_w <= huge_model_width) choose = NPURecModelSize::HUGE;
      else  choose = NPURecModelSize::UNKNOWN;
      auto it_model = npu_compiled_models_.find(choose);
      auto it_req = npu_infer_requests_.find(choose);

      //
      if (it_model != npu_compiled_models_.end() && it_req != npu_infer_requests_.end()) {
        // Switch to the chosen compiled model and infer request
        compiled_model_ = it_model->second;
        infer_request_ = it_req->second;
      } else {
        std::cout << "[WARNING] No compiled NPU model available for chosen size, using default compiled_model_." << std::endl;
      }
    }

    // For dynamic models, create tensor directly based on input data shape
    ov::Tensor input_tensor(ov::element::f32, input_shape);
    infer_request_.set_input_tensor(input_tensor);
    
    size_t tensor_size = input_tensor.get_size();
    // std::cout << "[DEBUG] Created tensor with shape: [";
    // for (size_t i = 0; i < input_shape.size(); ++i) {
    //   std::cout << input_shape[i];
    //   if (i < input_shape.size() - 1) std::cout << ",";
    // }
    // std::cout << "], size: " << tensor_size << std::endl;
    
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
      auto tensor_shape = input_tensor.get_shape();
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

std::vector<std::pair<int, int>> OpenVinoInfer::GetNPURecInputSizes() const {
  std::vector<std::pair<int, int>> sizes;
  
  if (option_.DeviceType() == "npu" && is_recognizer_) {
    // 按模型顺序获取输入尺寸
    for (const auto& model_pair : npu_compiled_models_) {
      try {
        auto input_shape = model_pair.second.input(0).get_shape();
        if (input_shape.size() >= 4) {
          int height = static_cast<int>(input_shape[2]);
          int width = static_cast<int>(input_shape[3]);
          sizes.push_back({height, width});
        }
      } catch (const std::exception& e) {
        std::cout << "[WARN] Failed to get NPU model input shape: " << e.what() << std::endl;
      }
    }
  }
  
  return sizes;
}