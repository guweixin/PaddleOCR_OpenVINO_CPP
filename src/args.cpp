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

#include <gflags/gflags.h>

// common args
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU memory size (MB) to use.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");
DEFINE_string(output, "./output/", "Save results to output directory.");
DEFINE_string(image_dir, "", "Dir of input image.");
DEFINE_string(
    type, "ocr",
    "Perform ocr or structure, the value is selected in ['ocr','structure'].");
// OpenVINO related
DEFINE_string(inference_framework, "paddle", "Inference framework: 'paddle' or 'ov' (OpenVINO)");
DEFINE_string(inference_device, "CPU", "Inference device for OpenVINO: 'CPU', 'GPU', or 'NPU'");
// detection related
DEFINE_string(det_model_dir, "", "Path of det inference model.");
DEFINE_string(limit_type, "max", "limit_type of input image.");
DEFINE_int32(limit_side_len, 960, "limit_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.6, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.5, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_dilation, false, "Whether use the dilation on output map.");
DEFINE_string(det_db_score_mode, "slow", "Whether use polygon score.");
DEFINE_bool(visualize, true, "Whether show the detection results.");
// recognition related
DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
DEFINE_int32(rec_batch_num, 6, "rec_batch_num.");
DEFINE_string(rec_char_dict_path, "./utils/ppocr_keys_v1.txt",
              "Path of dictionary.");
DEFINE_int32(rec_img_h, 48, "rec image height");
DEFINE_int32(rec_img_w, 320, "rec image width");

// Character recognition configuration
DEFINE_bool(use_space_char, true, "Whether to add space character to the recognition vocabulary for space prediction.");

// Smart space processing
DEFINE_bool(enable_smart_spaces, true, "Whether to add smart spaces between words in OCR results.");

// ocr forward related
DEFINE_bool(det, true, "Whether use det in forward.");
DEFINE_bool(rec, true, "Whether use rec in forward.");
