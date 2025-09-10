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

#include "args.h"
#include <iostream>
#include <cstring>

// 定义所有全局变量
// common args
std::string input = "";
std::string save_path = "./output/";
std::string text_detection_model_name = "PP-OCRv4_mobile_det";
std::string text_detection_model_dir = "";
std::string text_recognition_model_name = "PP-OCRv4_mobile_rec";
std::string text_recognition_model_dir = "";
std::string text_recognition_batch_size = "6";

// detection related
std::string text_det_limit_side_len = "64";
std::string text_det_limit_type = "min";
std::string text_det_thresh = "0.3";
std::string text_det_box_thresh = "0.6";
std::string text_det_unclip_ratio = "1.5";
std::string text_det_input_shape = "";

// recognition related
std::string text_rec_score_thresh = "0";
std::string text_rec_input_shape = "";

std::string lang = "";
std::string ocr_version = "";
#ifdef WITH_GPU
std::string device = "gpu:0";
#else
std::string device = "cpu";
#endif
std::string vis_font_dir = "";
std::string precision = "fp32";
std::string cpu_threads = "8";
std::string thread_num = "1";
std::string paddlex_config = "";

void print_help()
{
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "Common args:\n"
              << "  --input <path>                     Data to be predicted, required. Local path of an image file\n"
              << "  --save_path <path>                 Path to save inference result files (default: ./output/)\n"
              << "  --text_detection_model_name <name> Name of the text detection model (default: PP-OCRv4_mobile_det)\n"
              << "  --text_detection_model_dir <path>  Path to the text detection model directory\n"
              << "  --text_recognition_model_name <name> Name of the text recognition model (default: PP-OCRv4_mobile_rec)\n"
              << "  --text_recognition_model_dir <path> Path to the text recognition model directory\n"
              << "  --text_recognition_batch_size <num> Batch size for the text recognition model (default: 6)\n"
              << "\n"
              << "Detection related:\n"
              << "  --text_det_limit_side_len <len>    Limit on the side length of input image (default: 64)\n"
              << "  --text_det_limit_type <type>       How the side length limit is applied (default: min)\n"
              << "  --text_det_thresh <val>            Detection pixel threshold (default: 0.3)\n"
              << "  --text_det_box_thresh <val>        Detection box threshold (default: 0.6)\n"
              << "  --text_det_unclip_ratio <val>      Text detection expansion coefficient (default: 1.5)\n"
              << "  --text_det_input_shape <shape>     Input shape of text detection model (e.g., C,H,W)\n"
              << "\n"
              << "Recognition related:\n"
              << "  --text_rec_score_thresh <val>      Text recognition threshold (default: 0)\n"
              << "  --text_rec_input_shape <shape>     Input shape of text recognition model (e.g., C,H,W)\n"
              << "\n"
              << "Other options:\n"
              << "  --lang <language>                  Language in the input image for OCR processing\n"
              << "  --ocr_version <version>            PP-OCR version to use\n"
              << "  --device <device>                  Device for inference (default: cpu)\n"
              << "  --vis_font_dir <path>              Path to the visualization font\n"
              << "  --precision <type>                 Computational precision: fp32/fp16 (default: fp32)\n"
              << "  --cpu_threads <num>                Number of CPU threads (default: 8)\n"
              << "  --thread_num <num>                 Number of pipeline threads (default: 1)\n"
              << "  --paddlex_config <path>            Path to PaddleX pipeline configuration file\n"
              << "\n"
              << "  --help, -h                         Show this help message\n";
}

bool parse_args(int argc, char *argv[])
{
    // Skip the first non-option argument (subcommand like "ocr")
    int start_index = 1;
    if (argc > 1 && argv[1][0] != '-') {
        start_index = 2; // Skip subcommand
    }
    
    for (int i = start_index; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_help();
            return false;
        }
        else if (arg.find("--input=") == 0)
        {
            input = arg.substr(strlen("--input="));
        }
        else if (arg.find("--save_path=") == 0)
        {
            save_path = arg.substr(strlen("--save_path="));
        }
        else if (arg.find("--text_detection_model_name=") == 0)
        {
            text_detection_model_name = arg.substr(strlen("--text_detection_model_name="));
        }
        else if (arg.find("--text_detection_model_dir=") == 0)
        {
            text_detection_model_dir = arg.substr(strlen("--text_detection_model_dir="));
        }
        else if (arg.find("--text_recognition_model_name=") == 0)
        {
            text_recognition_model_name = arg.substr(strlen("--text_recognition_model_name="));
        }
        else if (arg.find("--text_recognition_model_dir=") == 0)
        {
            text_recognition_model_dir = arg.substr(strlen("--text_recognition_model_dir="));
        }
        else if (arg.find("--text_recognition_batch_size=") == 0)
        {
            text_recognition_batch_size = arg.substr(strlen("--text_recognition_batch_size="));
        }
        else if (arg.find("--text_det_limit_side_len=") == 0)
        {
            text_det_limit_side_len = arg.substr(strlen("--text_det_limit_side_len="));
        }
        else if (arg.find("--text_det_limit_type=") == 0)
        {
            text_det_limit_type = arg.substr(strlen("--text_det_limit_type="));
        }
        else if (arg.find("--text_det_thresh=") == 0)
        {
            text_det_thresh = arg.substr(strlen("--text_det_thresh="));
        }
        else if (arg.find("--text_det_box_thresh=") == 0)
        {
            text_det_box_thresh = arg.substr(strlen("--text_det_box_thresh="));
        }
        else if (arg.find("--text_det_unclip_ratio=") == 0)
        {
            text_det_unclip_ratio = arg.substr(strlen("--text_det_unclip_ratio="));
        }
        else if (arg.find("--text_det_input_shape=") == 0)
        {
            text_det_input_shape = arg.substr(strlen("--text_det_input_shape="));
        }
        else if (arg.find("--text_rec_score_thresh=") == 0)
        {
            text_rec_score_thresh = arg.substr(strlen("--text_rec_score_thresh="));
        }
        else if (arg.find("--text_rec_input_shape=") == 0)
        {
            text_rec_input_shape = arg.substr(strlen("--text_rec_input_shape="));
        }
        else if (arg.find("--lang=") == 0)
        {
            lang = arg.substr(strlen("--lang="));
        }
        else if (arg.find("--ocr_version=") == 0)
        {
            ocr_version = arg.substr(strlen("--ocr_version="));
        }
        else if (arg.find("--device=") == 0)
        {
            device = arg.substr(strlen("--device="));
        }
        else if (arg.find("--vis_font_dir=") == 0)
        {
            vis_font_dir = arg.substr(strlen("--vis_font_dir="));
        }
        else if (arg.find("--precision=") == 0)
        {
            precision = arg.substr(strlen("--precision="));
        }
        else if (arg.find("--cpu_threads=") == 0)
        {
            cpu_threads = arg.substr(strlen("--cpu_threads="));
        }
        else if (arg.find("--thread_num=") == 0)
        {
            thread_num = arg.substr(strlen("--thread_num="));
        }
        else if (arg.find("--paddlex_config=") == 0)
        {
            paddlex_config = arg.substr(strlen("--paddlex_config="));
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_help();
            return false;
        }

        // if (device== "npu")
        //     text_recognition_batch_size = 1;
    }
    return true;
}

