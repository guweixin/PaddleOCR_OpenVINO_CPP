cmake -G "Visual Studio 16 2019" -A x64 -DWITH_GPU=ON -DOPENCV_DIR="D:/Project_source_code/opencv/opencv-4.12.0/build" -DPADDLE_LIB="D:\Project_source_code\repo_OCR\Paddle\build_mkldnn\paddle_inference_install_dir" -DCUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"  -DCUDNN_LIB="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64" ..

cmake --build . --config Release

## test paddle infer
ppocr-paddle.exe ocr --input=..\ocr_data\fake_ocr_images  --text_detection_model_name=PP-OCRv4_mobile_det --text_detection_model_dir=..\models\PP-OCRv4_mobile_det_infer  --text_recognition_model_name=PP-OCRv4_mobile_rec --text_recognition_model_dir=..\models\PP-OCRv4_mobile_rec_infer

## test ov infer
ppocr.exe ocr --input=d:\My_Work_progress_2\HP_Xiaowei\20250626_paddleocr\ocr_data\hp_test\1.png  --text_detection_model_name=PP-OCRv4_mobile_det --text_detection_model_dir=..\..\..\models\PP-OCRv4_mobile_det_infer --text_recognition_model_name=PP-OCRv4_mobile_rec --text_recognition_model_dir=..\..\..\models\PP-OCRv4_mobile_rec_infer  --save_path=..\..\..\output\temp