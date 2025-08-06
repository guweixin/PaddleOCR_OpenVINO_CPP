[English](readme_en.md) | 简体中文


# PaddleOCR_OpenVINO_CPP
This sample shows how to use the <font color=red> OpenVINO C++ API </font> to deploy <font color=red> Paddle PP-OCRv4 </font> model

- [PaddleOCR\_OpenVINO\_CPP](#paddleocr_openvino_cpp)
  - [1. 开始运行](#1-开始运行)
    - [1.1 准备模型](#11-准备模型)
    - [1.2 编译PaddleOCR C++预测demo](#12-编译paddleocr-c预测demo)
    - [1.3 运行demo](#13-运行demo)
  - [3. FAQ](#3-faq)

本章节介绍PaddleOCR 模型的C++部署方法。C++在性能计算上优于Python，因此，在大多数CPU、GPU部署场景，多采用C++的部署方式，本节将介绍如何在Linux\Windows (CPU\GPU)环境下配置C++环境并完成PaddleOCR模型部署。


<a name="1"></a>

## 1. 开始运行

<a name="11"></a>

### 1.1 准备模型

目录结构如下。

```
model/
|-- ch_PP-OCRv4_det_infer
|   |--inference.pdiparams
|   |--inference.pdiparams.info
|   |--inference.pdmodel
|   |--inference.xml
|   |--inference.bin
|   |--inference_960.xml
|   |--inference_960.bin
|-- ch_PP-OCRv4_rec_infer
|   |--inference.pdiparams
|   |--inference.pdiparams.info
|   |--inference.pdmodel
|   |--inference.xml
|   |--inference.bin
|   |--inference_480_bs1.xml
|   |--inference_480_bs1.bin 
```


<a name="12"></a>

### 1.2 编译PaddleOCR C++预测demo

参考[Windows编译文档](docs/windows_vs2019_build.md)。

<a name="13"></a>

### 1.3 运行demo

本demo只支持使用检测+识别功能。


运行方式：
```shell
ppocr.exe [--param1] [--param2] [...]
```
具体命令如下：

```shell
ppocr.exe --det_model_dir=path_to_models/ch_PP-OCRv4_det_infer \
    --rec_model_dir=path_to_models/ch_PP-OCRv4_rec_infer \
    --image_dir=path_to_imageset  \
    --output=path_to_output \
    --inference_framework=paddle \
    --inference_device=GPU \
    --enable_mkldnn=true 
```


更多支持的可调节参数解释如下：

- 通用参数

|           参数名称           | 类型  | 默认参数 |                               意义                                |
| :--------------------------: | :---: | :------: | :---------------------------------------------------------------: |
|     inference_framework      |  str  |  paddle  |                      推理框架，paddle或者ov                       |
|       inference_device       |  str  |   CPU    |          推理设备，支持CPU/GPU/NPU(仅使用openvino时可用)          |
|          image_dir           |  str  |    ''    |                      需要识别图像保存的路径                       |
|            output            |  str  | ./output |                        识别结果保存的路径                         |
|            gpu_id            |  int  |    0     |                       GPU id，使用GPU时有效                       |
|           gpu_mem            |  int  |   4000   |                           申请的GPU内存                           |
| cpu_math_library_num_threads |  int  |    10    | CPU预测时的线程数，在机器核数充足的情况下，该值越大，预测速度越快 |
|        enable_mkldnn         | bool  |   true   |                         是否使用mkldnn库                          |

- 前向相关

| 参数名称 | 类型  | 默认参数 |         意义         |
| :------: | :---: | :------: | :------------------: |
|   det    | bool  |   true   | 前向是否执行文字检测 |
|   rec    | bool  |   true   | 前向是否执行文字识别 |

- 检测模型相关

|      参数名称       |  类型  | 默认参数 |                                                     意义                                                     |
| :-----------------: | :----: | :------: | :----------------------------------------------------------------------------------------------------------: |
|    det_model_dir    | string |    -     |                                         检测模型inference model地址                                          |
|    max_side_len     |  int   |   960    |                          输入图像长宽大于960时，等比例缩放图像，使得图像最长边为960                          |
|    det_db_thresh    | float  |   0.3    |                           用于过滤DB预测的二值化图像，设置为0.-0.3对结果影响不明显                           |
|  det_db_box_thresh  | float  |   0.5    |                           DB后处理过滤box的阈值，如果检测存在漏框情况，可酌情减小                            |
| det_db_unclip_ratio | float  |   1.6    |                                 表示文本框的紧致程度，越小则文本框更靠近文本                                 |
|  det_db_score_mode  | string |   slow   | slow:使用多边形框计算bbox score，fast:使用矩形框计算。矩形框计算速度更快，多边形框对弯曲文本区域计算更准确。 |
|      visualize      |  bool  |   true   |       是否对结果进行可视化，为1时，预测结果会保存在`output`字段指定的文件夹下和输入图像同名的图像上。        |

- 文字识别模型相关

|      参数名称      |  类型  |              默认参数               |              意义               |
| :----------------: | :----: | :---------------------------------: | :-----------------------------: |
|   rec_model_dir    | string |                  -                  | 文字识别模型inference model地址 |
| rec_char_dict_path | string | ../../ppocr/utils/ppocr_keys_v1.txt |            字典文件             |
|   rec_batch_num    |  int   |                  6                  |      文字识别模型batchsize      |
|     rec_img_h      |  int   |                 48                  |    文字识别模型输入图像高度     |
|     rec_img_w      |  int   |                 320                 |    文字识别模型输入图像宽度     |


* PaddleOCR也支持多语言的预测，更多支持的语言和模型可以参考[识别文档](../../doc/doc_ch/recognition.md)中的多语言字典与模型部分，如果希望进行多语言预测，只需将修改`rec_char_dict_path`（字典文件路径）以及`rec_model_dir`（inference模型路径）字段即可。

最终屏幕上会输出图像平均处理时间和占用内存如下。

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

<a name="3"></a>
## 3. FAQ

 1.  遇到报错 `unable to access 'https://github.com/LDOUBLEV/AutoLog.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.`， 将 `deploy/cpp_infer/external-cmake/auto-log.cmake` 中的github地址改为 https://gitee.com/Double_V/AutoLog 地址即可。
