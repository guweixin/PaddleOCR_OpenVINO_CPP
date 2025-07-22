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

#pragma once

#include <gflags/gflags.h>

// common args
DECLARE_bool(use_tensorrt);
DECLARE_int32(gpu_id);
DECLARE_int32(gpu_mem);
DECLARE_int32(cpu_threads);
DECLARE_bool(enable_mkldnn);
DECLARE_string(precision);
DECLARE_string(output);
DECLARE_string(image_dir);
DECLARE_string(type);
// OpenVINO related
DECLARE_string(inference_framework);
DECLARE_string(inference_device);
// detection related
DECLARE_string(det_model_dir);
DECLARE_string(limit_type);
DECLARE_int32(limit_side_len);
DECLARE_double(det_db_thresh);
DECLARE_double(det_db_box_thresh);
DECLARE_double(det_db_unclip_ratio);
DECLARE_bool(use_dilation);
DECLARE_string(det_db_score_mode);
DECLARE_bool(visualize);
// recognition related
DECLARE_string(rec_model_dir);
DECLARE_int32(rec_batch_num);
DECLARE_string(rec_char_dict_path);
DECLARE_int32(rec_img_h);
DECLARE_int32(rec_img_w);
// Character recognition configuration
DECLARE_bool(use_space_char);
// forward related
DECLARE_bool(det);
DECLARE_bool(rec);
// evaluation related
// Language parameter removed - batch processing mode is now default
DECLARE_string(image_dir);
// debug data saving
DECLARE_bool(save_debug_data);

// Auto-detected GPU usage flag (external variable)
extern bool auto_use_gpu;
