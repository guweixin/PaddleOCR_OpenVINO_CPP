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
|   |--inference_480_bs1.xml
|   |--inference_480_bs1.bin 
|   |--inference_800_bs1.xml
|   |--inference_800_bs1.bin 
```

<a name="12"></a>

### 1.2 Compile the PaddleOCR C++ prediction demo

Refer to [Windows Compiling Documents](docs/windows_vs2019_build_en.md)。

<a name="13"></a>

### 1.3 Runnig the demo

This demo only supports the detection and recognition function.

How to run:

```shell
ppocr.exe [--param1] [--param2] [...]
```

The specific commands are as follows:

```shell
ppocr.exe --det_model_dir=path_to_models/ch_PP-OCRv4_det_infer \
    --rec_model_dir=path_to_models/ch_PP-OCRv4_rec_infer \
    --image_dir=path_to_imageset  \
    --output=path_to_output \
    --inference_device=GPU \
```

More supported adjustable parameters are explained as follows:

- General Parameters

|           Parameters name          | Type  | Default value |                Significance                   |
| :--------------------------: | :---: | :------: | :------------------------------------------------------: |
|       inference_device       |  str  |   CPU    |       Inference device, supports CPU/GPU/NPU             |
|          image_dir           |  str  |    ''    |             Path of images to be identified              |
|            output            |  str  | ./output |          Path of recognition results are saved           |
| cpu_math_library_num_threads |  int  |    10    | The number of threads during CPU prediction. If the machine has enough cores, the larger the value, the faster the prediction speed. |

- Function-related

|           Parameters name          | Type  | Default value |                Significance                   |
| :--------------------------: | :---: | :------: | :------------------------------------------------------: |
|   det    | bool  |   true   | Whether to perform text detection   |
|   rec    | bool  |   true   | Whether to perform text recognition |

- Detection model related

|           Parameters name          | Type  | Default value |                Significance                   |
| :--------------------------: | :---: | :------: | :------------------------------------------------------: |
|    det_model_dir    | string |    -     |             The path of the detection model                      |
|    max_side_len     |  int   |   960    | When the input image is larger than 960 in length and width, the image is scaled so that the longest side of the image is 960.     |
|    det_db_thresh    | float  |   0.3    | Used to filter the binary image predicted by DB. Setting it to 0.-0.3 has an unknown effect on the results. |
|  det_db_box_thresh  | float  |   0.5    | The threshold of the DB post-processing filter box. If there are missed boxes during detection, it can be reduced as appropriate. |
| det_db_unclip_ratio | float  |   1.6    | Indicates the compactness of the text box. The smaller the size, the closer the text box is to the text. |
|  det_db_score_mode  | string |   slow   | Slow: Use polygonal boxes to calculate bbox score, fast: Use rectangular boxes to calculate. Rectangular boxes are faster to calculate, and polygonal boxes are more accurate for curved text areas. |
|      visualize      |  bool  |   true   | Whether to visualize the results. When it is 1, the prediction results will be saved in the folder specified by the `output` field on an image with the same name as the input image. |

- Recognition model related

|           Parameters name          | Type  | Default value |                Significance                   |
| :--------------------------: | :---: | :------: | :------------------------------------------------------: |
|    rec_model_dir    | string |    -     |             The path of the recognition model                    |
| rec_char_dict_path | string | ../../ppocr/utils/ppocr_keys_v1.txt |           Dictionary file          |
|   rec_batch_num    |  int   |                  6                  |      text recognition modelbatchsize      |
|     rec_img_h      |  int   |                 48                  |    text recognition model input img height  |
|     rec_img_w      |  int   |                 320                 |    text recognition model input img width  |


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