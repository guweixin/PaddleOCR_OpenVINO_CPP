#include <include/args.h>
#include <iostream>
#include <string>
#include <cstring>

// common args

int FLAGS_cpu_threads = 10;
std::string FLAGS_precision = "fp32";
std::string FLAGS_output = "./output/";
std::string FLAGS_image_dir = "";

// OpenVINO related
std::string FLAGS_inference_device = "CPU";

// detection related
std::string FLAGS_det_model_dir = "";
std::string FLAGS_limit_type = "max";
int FLAGS_limit_side_len = 960;
double FLAGS_det_db_thresh = 0.3;
double FLAGS_det_db_box_thresh = 0.6;
double FLAGS_det_db_unclip_ratio = 1.5;
bool FLAGS_use_dilation = false;
std::string FLAGS_det_db_score_mode = "slow";
bool FLAGS_visualize = true;

// recognition related
std::string FLAGS_rec_model_dir = "";
int FLAGS_rec_batch_num = 6;
std::string FLAGS_rec_char_dict_path = "../../utils/ppocr_keys_v1.txt";
int FLAGS_rec_img_h = 48;
int FLAGS_rec_img_w = 320;

// Character recognition configuration
bool FLAGS_use_space_char = true;

// ocr forward related
bool FLAGS_det = true;
bool FLAGS_rec = true;

void print_help()
{
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "Common args:\n"
              << "  --cpu_threads <num>        Number of CPU threads (default: 10)\n"
              << "  --precision <type>         Precision: fp32/fp16/int8 (default: fp32)\n"
              << "  --output <path>            Output directory (default: ./output/)\n"
              << "  --image_dir <path>         Directory of input images\n"
              << "\n"
              << "OpenVINO related:\n"
              << "  --inference_device <dev>   OpenVINO device: CPU/GPU/NPU (default: CPU)\n"
              << "\n"
              << "Detection related:\n"
              << "  --det_model_dir <path>     Path of detection inference model\n"
              << "  --limit_type <type>        Limit type of input image (default: max)\n"
              << "  --limit_side_len <len>     Limit side length of input image (default: 960)\n"
              << "  --det_db_thresh <val>      DB threshold (default: 0.3)\n"
              << "  --det_db_box_thresh <val>  DB box threshold (default: 0.6)\n"
              << "  --det_db_unclip_ratio <val> DB unclip ratio (default: 1.5)\n"
              << "  --use_dilation             Use dilation on output map\n"
              << "  --det_db_score_mode <mode> Polygon score mode (default: slow)\n"
              << "  --visualize                Show detection results (default: true)\n"
              << "\n"
              << "Recognition related:\n"
              << "  --rec_model_dir <path>     Path of recognition inference model\n"
              << "  --rec_batch_num <num>      Recognition batch number (default: 6)\n"
              << "  --rec_char_dict_path <path> Path of character dictionary\n"
              << "  --rec_img_h <height>       Recognition image height (default: 48)\n"
              << "  --rec_img_w <width>        Recognition image width (default: 320)\n"
              << "  --use_space_char           Add space character to vocabulary\n"
              << "\n"
              << "OCR forward related:\n"
              << "  --det                      Use detection in forward (default: true)\n"
              << "  --rec                      Use recognition in forward (default: true)\n"
              << "\n"
              << "  --help, -h                 Show this help message\n";
}

bool parse_args(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_help();
            return false;
        }
        else if (arg.find("--det_model_dir=") == 0)
        {
            FLAGS_det_model_dir = arg.substr(strlen("--det_model_dir="));
        }
        else if (arg.find("--rec_model_dir=") == 0)
        {
            FLAGS_rec_model_dir = arg.substr(strlen("--rec_model_dir="));
        }
        else if (arg.find("--image_dir=") == 0)
        {
            FLAGS_image_dir = arg.substr(strlen("--image_dir="));
        }
        else if (arg.find("--output=") == 0)
        {
            FLAGS_output = arg.substr(strlen("--output="));
        }
        else if (arg.find("--inference_device=") == 0)
        {
            FLAGS_inference_device = arg.substr(strlen("--inference_device="));
        }
        else if (arg.find("--cpu_threads=") == 0)
        {
            FLAGS_cpu_threads = std::stoi(arg.substr(strlen("--cpu_threads=")));
        }
        else if (arg.find("--limit_side_len=") == 0)
        {
            FLAGS_limit_side_len = std::stoi(arg.substr(strlen("--limit_side_len=")));
        }
        else if (arg.find("--det_db_thresh=") == 0)
        {
            FLAGS_det_db_thresh = std::stod(arg.substr(strlen("--det_db_thresh=")));
        }
        else if (arg.find("--det_db_box_thresh=") == 0)
        {
            FLAGS_det_db_box_thresh = std::stod(arg.substr(strlen("--det_db_box_thresh=")));
        }
        else if (arg.find("--det_db_unclip_ratio=") == 0)
        {
            FLAGS_det_db_unclip_ratio = std::stod(arg.substr(strlen("--det_db_unclip_ratio=")));
        }
        else if (arg.find("--rec_batch_num=") == 0)
        {
            FLAGS_rec_batch_num = std::stoi(arg.substr(strlen("--rec_batch_num=")));
        }
        else if (arg.find("--rec_char_dict_path=") == 0)
        {
            FLAGS_rec_char_dict_path = arg.substr(strlen("--rec_char_dict_path="));
        }
        else if (arg.find("--rec_img_h=") == 0)
        {
            FLAGS_rec_img_h = std::stoi(arg.substr(strlen("--rec_img_h=")));
        }
        else if (arg.find("--rec_img_w=") == 0)
        {
            FLAGS_rec_img_w = std::stoi(arg.substr(strlen("--rec_img_w=")));
        }
        else if (arg.find("--precision=") == 0)
        {
            FLAGS_precision = arg.substr(strlen("--precision="));
        }
        else if (arg.find("--limit_type=") == 0)
        {
            FLAGS_limit_type = arg.substr(strlen("--limit_type="));
        }
        else if (arg.find("--det_db_score_mode=") == 0)
        {
            FLAGS_det_db_score_mode = arg.substr(strlen("--det_db_score_mode="));
        }
        else if (arg == "--use_dilation")
        {
            FLAGS_use_dilation = true;
        }
        else if (arg == "--visualize")
        {
            FLAGS_visualize = true;
        }
        else if (arg == "--use_space_char")
        {
            FLAGS_use_space_char = true;
        }
        else if (arg == "--det")
        {
            FLAGS_det = true;
        }
        else if (arg == "--rec")
        {
            FLAGS_rec = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_help();
            return false;
        }
    }
    return true;
}