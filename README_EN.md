English | [简体中文](README.md) 

# PaddleOCR_OpenVINO_CPP

This sample shows how to use the <font color=red> OpenVINO C++ API </font> to deploy <font color=red> Paddle PP-OCRv4 </font> model

- [PaddleOCR\_OpenVINO\_CPP](#paddleocr_openvino_cpp)
  - [1. Start running](#1-start-running)
    - [1.1 Prepare models](#11-prepare-models)
    - [1.2 Compile the PaddleOCR C++ prediction demo](#12-compile-the-paddleocr-c-prediction-demo)
    - [1.3 Runnig the demo](#13-runnig-the-demo)

This section describes how to deploy a PaddleOCR model in C++. C++ outperforms Python in computational performance, so it's often used in most CPU and GPU deployment scenarios. This section explains how to configure a C++ environment and deploy a PaddleOCR model in a Windows (CPU, GPU, or NPU) environment.

<a name="1"></a>

## 1. Start running

<a name="11"></a>

### 1.1 Prepare models

The directory structure is as follows:

```shell
model/
|-- ch_PP-OCRv4_det_infer
|   |--inference.xml
|   |--inference.bin
|   |--inference_960.xml
|   |--inference_960.bin
|-- ch_PP-OCRv4_rec_infer
|   |--inference.xml
|   |--inference.bin
|   |--inference_480.xml
|   |--inference_480.bin 
|   |--inference_800.xml
|   |--inference_800.bin 
|   |--inference_1280.xml
|   |--inference_1280.bin 
```

<a name="12"></a>

### 1.2 Compile the PaddleOCR C++ prediction demo

Refer to [Windows Compiling Documents](docs/windows_vs2019_build_en.md)。

<a name="13"></a>

### 1.3 Runnig the demo

This demo only supports the detection and recognition function.

How to run:

```shell
ppocr.exe ocr [--param1] [--param2] [...]
```

The specific commands are as follows:

```shell
ppocr.exe ocr --input=image_dir \
  --text_detection_model_name=PP-OCRv4_mobile_det --text_detection_model_dir=model_dir \
  --text_recognition_model_name=PP-OCRv4_mobile_rec --text_recognition_model_dir=model_dir \
  --save_path=save_dir \
  --device=npu \
  --text_recognition_batch_size=1
```

More supported adjustable parameters are explained as follows:

- General Parameters

|           Parameter           | Type  | Default | Required |                               Meaning                                |
| :--------------------------: | :---: | :------: | :------: | :---------------------------------------------------------------: |
| device | str | CPU | No | Inference device, supports CPU / GPU / NPU (can be e.g. gpu:0) |
| input | str | '' | Yes | Path to image file or image directory to run OCR on (required) |
| save_path | str | ./output | No | Directory to save results and visualization images |
| cpu_threads | int | 8 | No | Number of CPU threads for inference; higher can speed up on multi-core machines |
| thread_num | int | 1 | No | Number of threads to use at runtime (general control) |
| precision | str | fp32 | No | Inference precision (e.g. fp32 / fp16 / int8) |
| vis_font_dir | str | '' | No | Font directory used for visualization (for non-ASCII text) |
| paddlex_config | str | '' | No | PaddleX config file path (optional) |
| lang | str | '' | No | Specify recognition language (optional) |
| ocr_version | str | '' | No | OCR model version identifier (optional) |
| --help, -h | flag | - | No | Print help and exit |

- Detection model related

|           Parameter           | Type  | Default | Required |                               Meaning                                |
| :--------------------------: | :---: | :------: | :------: | :---------------------------------------------------------------: |
| text_detection_model_name | string | PP-OCRv4_mobile_det | Yes | Detection inference model name (select which detection model to use) |
| text_detection_model_dir | string | '' | Yes | Directory containing detection inference model files (must point to .xml/.bin or model files) |
| max_side_len | int | 960 | No | Resize input so the longest side equals this when larger |
| text_det_thresh | float | 0.3 | No | DB binarization threshold to filter probability map |
| det_db_box_thresh | float | 0.5 | No | Threshold to filter boxes after DB post-processing |
| det_db_unclip_ratio | float | 1.6 | No | Unclip ratio for expanding text boxes (smaller = tighter) |
| det_db_score_mode | string | slow | No | Score mode: slow (polygon-based, more accurate) / fast (rect-based, faster) |
| visualize | bool | true | No | Whether to output visualization images to save_path |

- Recognition model related

|           Parameter           | Type  | Default | Required |                               Meaning                                |
| :--------------------------: | :---: | :------: | :------: | :---------------------------------------------------------------: |
| text_recognition_model_name | string | PP-OCRv4_mobile_rec | Yes | Recognition inference model name (select which recognition model to use) |
| text_recognition_model_dir / rec_model_dir | string | '' | Yes | Directory containing recognition inference model files (must point to .xml/.bin or model files) |
| rec_char_dict_path | string | ../../ppocr/utils/ppocr_keys_v1.txt | No | Character dictionary path for recognition model (defines output charset) |
| text_recognition_batch_size / rec_batch_num | int | 6 | No | Recognition model batch size (affects throughput and memory) |
| rec_img_h | int | 48 | No | Recognition model input image height |
| rec_img_w | int | 320 | No | Recognition model input image width |
| text_rec_score_thresh | float | 0 | No | Recognition confidence threshold |
| text_rec_input_shape | string | '' | No | Optional input shape spec for recognition model |


* PaddleOCR also supports multi-language prediction. For more supported languages and models. If you want to perform multi-language prediction, just modify the `rec_char_dict_path` (dictionary file path) and `rec_model_dir` (inference model path) fields.

Finally, the average image processing time and memory usage will be output on the screen as follows.

- ocr

```bash
I0717 10:40:31.994189 10176 analysis_predictor.cc:2259] ======= ir optimization completed =======
I0717 10:40:31.995692 10176 naive_executor.cc:211] ---  skip [feed], feed -> x
I0717 10:40:31.997696 10176 naive_executor.cc:211] ---  skip [softmax_11.tmp_0], fetch -> fetch
[==================================================] 100.0% 1000/1000

======================== Processing Results ========================
Average inference time: 118.86 ms
Memory usage (increase only):
  Average increase: 794.16 MB
  Maximum increase: 805 MB
Results saved to: D:\output\cpp_paddleocr_paddle_gpu
=================================================================
```