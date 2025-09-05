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
      model_file_prefix_(model_file_prefix), option_(option) {
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
    std::string model_path = model_dir_ + "/" + model_file_prefix_ + ".xml";
    std::string weights_path = model_dir_ + "/" + model_file_prefix_ + ".bin";
    
    std::cout << "[DEBUG] Looking for model files:" << std::endl;
    std::cout << "[DEBUG] Model path: " << model_path << std::endl;
    std::cout << "[DEBUG] Weights path: " << weights_path << std::endl;
    
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
    
    std::cout << "[DEBUG] Model files found, loading model..." << std::endl;
    
    // Load the model
    model_ = core_.read_model(model_path, weights_path);
    if (!model_) {
      std::cout << "[ERROR] Failed to read OpenVINO model" << std::endl;
      return Status::InternalError("Failed to read OpenVINO model");
    }
    
    std::cout << "[DEBUG] Model loaded successfully" << std::endl;
    
    // Set device
    std::string device = (option_.DeviceType() == "gpu") ? "GPU" : "CPU";
    
    // Configure compilation options
    ov::AnyMap compile_options;
    if (device == "CPU") {
      compile_options["NUM_STREAMS"] = std::to_string(option_.CpuThreads());
    }
    
    // Compile the model
    compiled_model_ = core_.compile_model(model_, device, compile_options);
    infer_request_ = compiled_model_.create_infer_request();
    
    // Get input/output names
    std::cout << "[DEBUG] Getting model inputs and outputs..." << std::endl;
    std::cout << "[DEBUG] Model has " << model_->inputs().size() << " inputs" << std::endl;
    std::cout << "[DEBUG] Model has " << model_->outputs().size() << " outputs" << std::endl;
    
    for (const auto& input : model_->inputs()) {
      std::string input_name = input.get_any_name();
      input_names_.push_back(input_name);
      std::cout << "[DEBUG] Input: " << input_name << ", Shape: " << input.get_partial_shape().to_string() << std::endl;
    }
    for (const auto& output : model_->outputs()) {
      std::string output_name = output.get_any_name();
      output_names_.push_back(output_name);
      std::cout << "[DEBUG] Output: " << output_name << ", Shape: " << output.get_partial_shape().to_string() << std::endl;
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
    std::cout << "[DEBUG] Apply() called with " << batch_size << " input matrices" << std::endl;
    
    // Get the first input for shape analysis
    auto input_partial_shape = model_->input().get_partial_shape();
    std::cout << "[DEBUG] Original input shape: " << input_partial_shape.to_string() << std::endl;
    
    // Analyze the input data format
    const cv::Mat& first_mat = input_mats[0];
    std::cout << "[DEBUG] Input Mat info: dims=" << first_mat.dims;
    if (first_mat.dims >= 3) {
      std::cout << ", size=[";
      for (int i = 0; i < first_mat.dims; ++i) {
        std::cout << first_mat.size[i];
        if (i < first_mat.dims - 1) std::cout << ",";
      }
      std::cout << "]";
    } else {
      std::cout << ", rows=" << first_mat.rows << ", cols=" << first_mat.cols;
    }
    std::cout << ", channels=" << first_mat.channels() << ", type=" << first_mat.type() 
              << ", total=" << first_mat.total() << ", isContinuous=" << first_mat.isContinuous() << std::endl;
    
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
      std::cout << "[DEBUG] Multi-dimensional Mat validation: " << (valid_data ? "PASSED" : "FAILED") << std::endl;
    } else {
      // For 2D Mat, use traditional validation
      valid_data = first_mat.data && first_mat.rows > 0 && first_mat.cols > 0;
      std::cout << "[DEBUG] 2D Mat validation: " << (valid_data ? "PASSED" : "FAILED") << std::endl;
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
      std::cout << "[DEBUG] Using 4D NCHW format, shape: [" << input_shape[0] << ", " << input_shape[1] 
                << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
    } else if (first_mat.dims == 3) {
      // 3D CHW format, add batch dimension
      input_shape = {
        batch_size,                               // N (batch)
        static_cast<size_t>(first_mat.size[0]),  // C (channels)
        static_cast<size_t>(first_mat.size[1]),  // H (height)
        static_cast<size_t>(first_mat.size[2])   // W (width)
      };
      std::cout << "[DEBUG] Using 3D CHW format, adding batch dimension, shape: [" << input_shape[0] << ", " << input_shape[1] 
                << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
    } else if (first_mat.channels() == 3 && first_mat.rows > 0 && first_mat.cols > 0) {
      // Standard 2D HWC format: convert to NCHW
      input_shape = {batch_size, 3, static_cast<size_t>(first_mat.rows), static_cast<size_t>(first_mat.cols)};
      std::cout << "[DEBUG] Detected 2D HWC format, using shape: [" << input_shape[0] << ", " << input_shape[1] 
                << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
    } else if (first_mat.channels() == 1 && first_mat.total() > 0) {
      // Flattened CHW data - need to determine dimensions
      size_t total_elements = first_mat.total();
      std::cout << "[DEBUG] Flattened data with " << total_elements << " elements" << std::endl;
      
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
        std::cout << "[DEBUG] Inferred CHW shape from flattened data: [" << input_shape[0] 
                  << ", " << input_shape[1] << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
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
      
      std::cout << "[DEBUG] Static model shape check:" << std::endl;
      std::cout << "[DEBUG] Current: [";
      for (size_t i = 0; i < current_shape.size(); ++i) {
        std::cout << current_shape[i];
        if (i < current_shape.size() - 1) std::cout << ",";
      }
      std::cout << "]" << std::endl;
      std::cout << "[DEBUG] Required: [";
      for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ",";
      }
      std::cout << "]" << std::endl;
      std::cout << "[DEBUG] Need reshape: " << (need_reshape ? "YES" : "NO") << std::endl;
    }
    
    if (need_reshape) {
      std::cout << "[DEBUG] Reshaping model to input shape: [" << input_shape[0] << ", " << input_shape[1] 
                << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
      
      try {
        model_->reshape({{input_names_[0], input_shape}});
        compiled_model_ = core_.compile_model(model_, (option_.DeviceType() == "gpu") ? "GPU" : "CPU");
        infer_request_ = compiled_model_.create_infer_request();
        std::cout << "[DEBUG] Model reshaped and recompiled successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "[ERROR] Failed to reshape model: " << e.what() << std::endl;
        return Status::InternalError("Failed to reshape model: " + std::string(e.what()));
      }
    } else {
      std::cout << "[DEBUG] Using existing model shape, no reshaping needed" << std::endl;
    }
    
    // Create input tensor
    ov::Tensor input_tensor = infer_request_.get_input_tensor();
    auto tensor_shape = input_tensor.get_shape();
    
    std::cout << "[DEBUG] Input tensor shape: [";
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
      std::cout << tensor_shape[i];
      if (i < tensor_shape.size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
    
    size_t tensor_size = input_tensor.get_size();
    std::cout << "[DEBUG] Input tensor total size: " << tensor_size << " elements" << std::endl;
    
    // Copy input data
    float* input_data = input_tensor.data<float>();
    std::cout << "[DEBUG] Input tensor data pointer: " << static_cast<void*>(input_data) << std::endl;
    
    // For batch data (dims == 4), we expect only one Mat containing all batches
    if (batch_size == 1 && first_mat.dims == 4) {
      // Single Mat contains all batch data - copy directly
      size_t total_elements = first_mat.total();
      std::cout << "[DEBUG] Copying batch data: " << total_elements << " elements" << std::endl;
      
      if (total_elements != tensor_size) {
        std::cout << "[ERROR] Data size mismatch: Mat=" << total_elements 
                  << ", Tensor=" << tensor_size << std::endl;
        return Status::InternalError("Data size mismatch between Mat and tensor");
      }
      
      // Copy data directly
      if (first_mat.type() == CV_32F) {
        std::cout << "[DEBUG] Copying " << total_elements << " float elements directly..." << std::endl;
        try {
          std::cout << "[DEBUG] Source pointer: " << static_cast<void*>(first_mat.data) << std::endl;
          std::cout << "[DEBUG] Destination pointer: " << static_cast<void*>(input_data) << std::endl;
          std::cout << "[DEBUG] Bytes to copy: " << total_elements * sizeof(float) << std::endl;
          
          std::memcpy(input_data, first_mat.data, total_elements * sizeof(float));
          std::cout << "[DEBUG] Batch data copied successfully" << std::endl;
        } catch (const std::exception& e) {
          std::cout << "[ERROR] Memory copy failed: " << e.what() << std::endl;
          return Status::InternalError("Memory copy failed: " + std::string(e.what()));
        }
      } else {
        std::cout << "[DEBUG] Converting to float and copying..." << std::endl;
        cv::Mat float_mat;
        first_mat.convertTo(float_mat, CV_32F);
        try {
          std::cout << "[DEBUG] Source pointer (converted): " << static_cast<void*>(float_mat.data) << std::endl;
          std::cout << "[DEBUG] Destination pointer: " << static_cast<void*>(input_data) << std::endl;
          std::cout << "[DEBUG] Bytes to copy: " << total_elements * sizeof(float) << std::endl;
          
          std::memcpy(input_data, float_mat.data, total_elements * sizeof(float));
          std::cout << "[DEBUG] Batch data converted and copied successfully" << std::endl;
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
      
      std::cout << "[DEBUG] Calculated elements_per_batch: " << elements_per_batch << std::endl;
      
      for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const cv::Mat& mat = input_mats[batch_idx];
        
        std::cout << "[DEBUG] ========== Processing batch " << batch_idx << " ==========" << std::endl;
        
        // Calculate elements per sample based on Mat dimensions
        size_t mat_elements;
        if (mat.dims >= 3) {
          // For multi-dimensional Mat, calculate total elements directly
          mat_elements = mat.total();
        } else {
          // For 2D Mat, include channels
          mat_elements = mat.total() * mat.channels();
        }
        
        std::cout << "[DEBUG] Mat elements: " << mat_elements << std::endl;
        std::cout << "[DEBUG] Expected elements per batch: " << elements_per_batch << std::endl;
        
        if (mat_elements != elements_per_batch) {
          std::cout << "[WARNING] Mat elements (" << mat_elements 
                    << ") != expected elements per batch (" << elements_per_batch << ")" << std::endl;
        }
        
        // Check memory bounds
        size_t total_offset = batch_idx * elements_per_batch;
        size_t required_memory = total_offset + elements_per_batch;
        std::cout << "[DEBUG] Memory calculation:" << std::endl;
        std::cout << "[DEBUG]   - Batch index: " << batch_idx << std::endl;
        std::cout << "[DEBUG]   - Elements per batch: " << elements_per_batch << std::endl;
        std::cout << "[DEBUG]   - Total offset: " << total_offset << std::endl;
        std::cout << "[DEBUG]   - Required memory: " << required_memory << std::endl;
        std::cout << "[DEBUG]   - Tensor capacity: " << tensor_size << std::endl;
        
        if (required_memory > tensor_size) {
          std::cout << "[ERROR] Memory overflow! Required: " << required_memory 
                    << ", Available: " << tensor_size << std::endl;
          return Status::InternalError("Memory overflow in tensor copy");
        }
        
        // Copy data directly
        if (mat.type() == CV_32F) {
          std::cout << "[DEBUG] Copying " << elements_per_batch << " float elements directly..." << std::endl;
          try {
            std::cout << "[DEBUG] Source pointer: " << static_cast<void*>(mat.data) << std::endl;
            std::cout << "[DEBUG] Destination pointer: " << static_cast<void*>(input_data + total_offset) << std::endl;
            std::cout << "[DEBUG] Bytes to copy: " << elements_per_batch * sizeof(float) << std::endl;
            
            std::memcpy(input_data + total_offset,
                        mat.data, elements_per_batch * sizeof(float));
            std::cout << "[DEBUG] Copied " << elements_per_batch << " float elements directly - SUCCESS" << std::endl;
          } catch (const std::exception& e) {
            std::cout << "[ERROR] Memory copy failed: " << e.what() << std::endl;
            return Status::InternalError("Memory copy failed: " + std::string(e.what()));
          }
        } else {
          std::cout << "[DEBUG] Converting to float and copying..." << std::endl;
          cv::Mat float_mat;
          mat.convertTo(float_mat, CV_32F);
          try {
            std::cout << "[DEBUG] Source pointer (converted): " << static_cast<void*>(float_mat.data) << std::endl;
            std::cout << "[DEBUG] Destination pointer: " << static_cast<void*>(input_data + total_offset) << std::endl;
            std::cout << "[DEBUG] Bytes to copy: " << elements_per_batch * sizeof(float) << std::endl;
            
            std::memcpy(input_data + total_offset,
                        float_mat.data, elements_per_batch * sizeof(float));
            std::cout << "[DEBUG] Converted and copied " << elements_per_batch << " elements - SUCCESS" << std::endl;
          } catch (const std::exception& e) {
            std::cout << "[ERROR] Memory copy (convert) failed: " << e.what() << std::endl;
            return Status::InternalError("Memory copy (convert) failed: " + std::string(e.what()));
          }
        }
        
        std::cout << "[DEBUG] ========== Batch " << batch_idx << " completed ==========" << std::endl;
      }
    }
    
    std::cout << "[DEBUG] All input data copied, starting inference..." << std::endl;
    
    // Run inference
    try {
      std::cout << "[DEBUG] Calling infer_request_.infer()..." << std::endl;
      infer_request_.infer();
      std::cout << "[DEBUG] Inference completed successfully" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[ERROR] Inference failed: " << e.what() << std::endl;
      return Status::InternalError("Inference failed: " + std::string(e.what()));
    }
    
    std::cout << "[DEBUG] Getting output tensors..." << std::endl;
    std::vector<cv::Mat> output_mats;
    
    std::cout << "[DEBUG] Number of outputs: " << output_names_.size() << std::endl;
    
    for (size_t output_idx = 0; output_idx < output_names_.size(); ++output_idx) {
      std::cout << "[DEBUG] Processing output " << output_idx << std::endl;
      
      auto output_tensor = infer_request_.get_output_tensor(output_idx);
      auto output_shape = output_tensor.get_shape();
      
      std::cout << "[DEBUG] Output " << output_idx << " shape: [";
      for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << ",";
      }
      std::cout << "]" << std::endl;
      
      // Create output cv::Mat
      if (output_shape.size() == 4) {
        // 4D tensor: [batch, channels, height, width]
        size_t out_batch = output_shape[0];
        size_t out_channels = output_shape[1];
        size_t out_height = output_shape[2];
        size_t out_width = output_shape[3];
        
        std::cout << "[DEBUG] Creating 4D output Mat: [" << out_batch << "," << out_channels 
                  << "," << out_height << "," << out_width << "]" << std::endl;
        
        for (size_t b = 0; b < out_batch; ++b) {
          std::cout << "[DEBUG] Creating output Mat for batch " << b << std::endl;
          cv::Mat output_mat(static_cast<int>(out_height), static_cast<int>(out_width), CV_32FC(static_cast<int>(out_channels)));
          
          std::cout << "[DEBUG] Getting output tensor data..." << std::endl;
          float* output_data = output_tensor.data<float>();
          
          std::cout << "[DEBUG] Copying output data, size: " << (out_channels * out_height * out_width * sizeof(float)) << " bytes" << std::endl;
          std::memcpy(output_mat.data, 
                      output_data + b * out_channels * out_height * out_width,
                      out_channels * out_height * out_width * sizeof(float));
          
          std::cout << "[DEBUG] Output Mat created successfully, adding to results" << std::endl;
          output_mats.push_back(output_mat);
        }
      } else if (output_shape.size() == 2) {
        // 2D tensor: [batch, features]
        size_t out_batch = output_shape[0];
        size_t out_features = output_shape[1];
        
        std::cout << "[DEBUG] Creating 2D output Mat: [" << out_batch << "," << out_features << "]" << std::endl;
        
        for (size_t b = 0; b < out_batch; ++b) {
          std::cout << "[DEBUG] Creating 2D output Mat for batch " << b << std::endl;
          cv::Mat output_mat(1, static_cast<int>(out_features), CV_32F);
          
          std::cout << "[DEBUG] Getting 2D output tensor data..." << std::endl;
          float* output_data = output_tensor.data<float>();
          
          std::cout << "[DEBUG] Copying 2D output data, size: " << (out_features * sizeof(float)) << " bytes" << std::endl;
          std::memcpy(output_mat.data, 
                      output_data + b * out_features,
                      out_features * sizeof(float));
          
          std::cout << "[DEBUG] 2D Output Mat created successfully" << std::endl;
          output_mats.push_back(output_mat);
        }
      } else if (output_shape.size() == 3) {
        // 3D tensor: [batch, sequence, classes] - for text recognition
        size_t out_batch = output_shape[0];
        size_t out_sequence = output_shape[1];
        size_t out_classes = output_shape[2];
        
        std::cout << "[DEBUG] Creating 3D output Mat: [" << out_batch << "," << out_sequence 
                  << "," << out_classes << "]" << std::endl;
        
        // For 3D output, we create a 3D cv::Mat
        int sizes[] = {static_cast<int>(out_batch), static_cast<int>(out_sequence), static_cast<int>(out_classes)};
        cv::Mat output_mat(3, sizes, CV_32F);
        
        std::cout << "[DEBUG] Getting 3D output tensor data..." << std::endl;
        float* output_data = output_tensor.data<float>();
        
        size_t total_elements = out_batch * out_sequence * out_classes;
        std::cout << "[DEBUG] Copying 3D output data, size: " << (total_elements * sizeof(float)) << " bytes" << std::endl;
        std::memcpy(output_mat.data, output_data, total_elements * sizeof(float));
        
        std::cout << "[DEBUG] 3D Output Mat created successfully" << std::endl;
        output_mats.push_back(output_mat);
      } else {
        std::cout << "[ERROR] Unsupported output tensor shape with " << output_shape.size() << " dimensions" << std::endl;
        return Status::InternalError("Unsupported output tensor shape");
      }
      
      std::cout << "[DEBUG] Finished processing output " << output_idx << std::endl;
    }
    
    std::cout << "[DEBUG] All outputs processed successfully, returning " << output_mats.size() << " output matrices" << std::endl;
    return output_mats;
    
  } catch (const std::exception& e) {
    return Status::InternalError("OpenVINO inference error: " + std::string(e.what()));
  }
}
