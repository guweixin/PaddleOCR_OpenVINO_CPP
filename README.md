[English](README_EN.md) | 简体中文

# PaddleOCR_OpenVINO_CPP

This sample shows how to use the <font color=red> OpenVINO C++ API </font> to deploy <font color=red> Paddle PP-OCRv4 </font> model

- [PaddleOCR\_OpenVINO\_CPP](#paddleocr_openvino_cpp)
  - [1. 开始运行](#1-开始运行)
    - [1.1 准备模型](#11-准备模型)
    - [1.2 编译PaddleOCR C++预测demo](#12-编译paddleocr-c预测demo)
    - [1.3 运行demo](#13-运行demo)

本章节介绍PaddleOCR 模型的C++部署方法。C++在性能计算上优于Python，因此，在大多数CPU、GPU部署场景，多采用C++的部署方式，本节将介绍如何在Windows (CPU\GPU\NPU)环境下配置C++环境并完成PaddleOCR模型部署。

<a name="1"></a>

## 1. 开始运行

<a name="11"></a>

### 1.1 准备模型

目录结构如下：

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

### 1.2 编译PaddleOCR C++预测demo

参考[Windows编译文档](docs/windows_vs2019_build.md)。

<a name="13"></a>

### 1.3 运行demo

本demo只支持使用检测+识别功能。

运行方式：

```shell
ppocr.exe ocr [--param1] [--param2] [...]
```

具体命令如下：

```shell
ppocr.exe ocr --input=image_dir \
  --text_detection_model_name=PP-OCRv4_mobile_det --text_detection_model_dir=model_dir \
  --text_recognition_model_name=PP-OCRv4_mobile_rec --text_recognition_model_dir=model_dir \
  --save_path=save_dir \
  --device=npu \
  --text_recognition_batch_size=1
```

更多支持的可调节参数解释如下：

- 通用参数

|           参数名称           | 类型  | 默认参数 | 是否必需 |                               意义                                |
| :--------------------------: | :---: | :------: | :------: | :---------------------------------------------------------------: |
| device | str | CPU | 否 | 推理设备，支持 CPU / GPU / NPU（亦可用 gpu:0 等设备标识） |
| input | str | '' | 是 | 待识别的图像文件或图片目录路径（必填） |
| save_path | str | ./output | 否 | 识别结果与可视化图片保存目录 |
| cpu_threads | int | 8 | 否 | CPU 推理线程数，机器核数充足时设大可提升性能 |
| thread_num | int | 1 | 否 | 运行时使用的线程数（通用控制） |
| precision | str | fp32 | 否 | 推理精度（常见值：fp32 / fp16 / int8） |
| vis_font_dir | str | '' | 否 | 可视化时使用的字体目录（用于生成中文/特殊字符可读图片） |
| paddlex_config | str | '' | 否 | PaddleX 配置文件路径（可选） |
| lang | str | '' | 否 | 指定识别语言（可选，多语言需对应字典与模型） |
| ocr_version | str | '' | 否 | OCR 模型版本标识（可选） |
| --help, -h | flag | - | 否 | 打印帮助并退出 |

- 检测模型相关

|           参数名称           | 类型  | 默认参数 | 是否必需 |                               意义                                |
| :--------------------------: | :---: | :------: | :------: | :---------------------------------------------------------------: |
| text_detection_model_name | string | PP-OCRv4_mobile_det | 是 | 检测模型的 inference 名称（选择要使用的检测模型） |
| text_detection_model_dir | string | '' | 是 | 检测模型 inference 文件所在目录（必须指向包含 .xml/.bin 的目录或相应文件） |
| max_side_len | int | 960 | 否 | 若输入图像最长边大于该值，等比缩放使最长边等于该值以限制输入大小 |
| text_det_thresh | float | 0.3 | 否 | DB 二值化阈值，用于过滤概率图（影响分割/二值化结果） |
| det_db_box_thresh | float | 0.5 | 否 | DB 后处理过滤 box 的阈值，增大可减少误检，减小可减少漏检 |
| det_db_unclip_ratio | float | 1.6 | 否 | DB 后处理扩展文本框的比例，值越小框越贴近文本 |
| det_db_score_mode | string | slow | 否 | 评分模式：slow（多边形评分，准确）/ fast（矩形评分，速度快） |
| visualize | bool | true | 否 | 是否输出可视化图片到 save_path（true/false） |

- 文字识别模型相关

|           参数名称           | 类型  | 默认参数 | 是否必需 |                               意义                                |
| :--------------------------: | :---: | :------: | :------: | :---------------------------------------------------------------: |
| text_recognition_model_name | string | PP-OCRv4_mobile_rec | 是 | 识别模型的 inference 名称（选择要使用的识别模型） |
| text_recognition_model_dir / rec_model_dir | string | '' | 是 | 识别模型 inference 文件所在目录（必须指向包含 .xml/.bin 的目录或相应文件） |
| rec_char_dict_path | string | ../../ppocr/utils/ppocr_keys_v1.txt | 否 | 识别模型使用的字符字典路径，决定输出字符集 |
| text_recognition_batch_size / rec_batch_num | int | 6 | 否 | 识别模型的 batch size（并行识别文本数量，影响吞吐与显存/内存） |
| rec_img_h | int | 48 | 否 | 识别模型输入图像高度（取决于模型训练尺寸） |
| rec_img_w | int | 320 | 否 | 识别模型输入图像宽度（取决于模型训练尺寸） |
| text_rec_score_thresh | float | 0 | 否 | 识别结果置信度阈值（低于则可过滤） |
| text_rec_input_shape | string | '' | 否 | 可指定识别模型的输入形状（若模型支持可变尺寸） |

* PaddleOCR也支持多语言的预测，更多支持的语言和模型可以参考[识别文档](../../doc/doc_ch/recognition.md)中的多语言字典与模型部分，如果希望进行多语言预测，只需将修改`rec_char_dict_path`（字典文件路径）以及`rec_model_dir`（inference模型路径）字段即可。

最终屏幕上会输出图像平均处理时间如下。

- ocr

```bash
Models init time: 7745.79 ms
[==================================================] 100.0% 2000/2000
Models average inference time:: 251.7 ms
```