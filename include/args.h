#ifndef ARGS_H
#define ARGS_H

#include <string>

// 声明所有全局变量（extern）
// common args
extern int FLAGS_cpu_threads;
extern std::string FLAGS_precision;
extern std::string FLAGS_output;
extern std::string FLAGS_image_dir;

// OpenVINO related
extern std::string FLAGS_inference_device;

// detection related
extern std::string FLAGS_det_model_dir;
extern std::string FLAGS_limit_type;
extern int FLAGS_limit_side_len;
extern double FLAGS_det_db_thresh;
extern double FLAGS_det_db_box_thresh;
extern double FLAGS_det_db_unclip_ratio;
extern bool FLAGS_use_dilation;
extern std::string FLAGS_det_db_score_mode;
extern bool FLAGS_visualize;

// recognition related
extern std::string FLAGS_rec_model_dir;
extern int FLAGS_rec_batch_num;
extern std::string FLAGS_rec_char_dict_path;
extern int FLAGS_rec_img_h;
extern int FLAGS_rec_img_w;

// Character recognition configuration
extern bool FLAGS_use_space_char;

// ocr forward related
extern bool FLAGS_det;
extern bool FLAGS_rec;

// 函数声明
void print_help();
bool parse_args(int argc, char *argv[]);

#endif // ARGS_H