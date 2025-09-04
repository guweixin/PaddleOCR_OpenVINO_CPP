// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License";
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

#pragma once

// #include <gflags/gflags.h>
#ifndef ARGS_H
#define ARGS_H

#include <string>

// 声明所有全局变量（extern）
// common args
extern std::string input;
extern std::string save_path;
extern std::string text_detection_model_name;
extern std::string text_detection_model_dir;
extern std::string text_recognition_model_name;
extern std::string text_recognition_model_dir;
extern std::string text_recognition_batch_size;

// detection related
extern std::string text_det_limit_side_len;
extern std::string text_det_limit_type;
extern std::string text_det_thresh;
extern std::string text_det_box_thresh;
extern std::string text_det_unclip_ratio;
extern std::string text_det_input_shape;

// recognition related
extern std::string text_rec_score_thresh;
extern std::string text_rec_input_shape;

extern std::string lang;
extern std::string ocr_version;
extern std::string device;
extern std::string vis_font_dir;
extern std::string precision;
extern std::string cpu_threads;
extern std::string thread_num;
extern std::string paddlex_config;


// 函数声明
void print_help();
bool parse_args(int argc, char *argv[]);

#endif // ARGS_H
