cmake_minimum_required(VERSION 3.14)
project(ppocr CXX C)

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       OFF)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
option(WITH_TENSORRT "Compile demo with TensorRT."   OFF)
option(WITH_OPENVINO "Compile demo with OpenVINO support."   ON)

SET(PADDLE_LIB "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(CUDA_LIB "" CACHE PATH "Location of libraries")
SET(CUDNN_LIB "" CACHE PATH "Location of libraries")
SET(TENSORRT_DIR "" CACHE PATH "Compile demo with TensorRT")
SET(OPENVINO_DIR "" CACHE PATH "Location of OpenVINO installation")

set(DEMO_NAME "ppocr")


macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WITH_MKL)
    ADD_DEFINITIONS(-DUSE_MKL)
endif()

if (WITH_OPENVINO)
    ADD_DEFINITIONS(-DWITH_OPENVINO)
    if(NOT DEFINED OPENVINO_DIR)
        message(FATAL_ERROR "please set OPENVINO_DIR with -DOPENVINO_DIR=/path/openvino")
    endif()
    find_package(OpenVINO REQUIRED PATHS ${OPENVINO_DIR}/cmake NO_DEFAULT_PATH)
    message(STATUS "OpenVINO found: ${OpenVINO_FOUND}")
endif()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()


if (WIN32)
  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/lib")
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)

else ()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/lib")
endif ()
include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    if(WITH_MKL)
        set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
	add_definitions(-DYAML_CPP_STATIC_DEFINE)
    endif()
    message("cmake c debug flags " ${CMAKE_C_FLAGS_DEBUG})
    message("cmake c release flags " ${CMAKE_C_FLAGS_RELEASE})
    message("cmake cxx debug flags " ${CMAKE_CXX_FLAGS_DEBUG})
    message("cmake cxx release flags " ${CMAKE_CXX_FLAGS_RELEASE})
else()
    if(WITH_MKL)
        set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 ${FLAG_OPENMP} -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    message("cmake cxx flags" ${CMAKE_CXX_FLAGS})
endif()

if (WITH_GPU)
    if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
        message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
    endif()
    if (NOT WIN32)
        if (NOT DEFINED CUDNN_LIB)
            message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
        endif()
    endif(NOT WIN32)
endif()

include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/install/onnxruntime/include")
include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
include_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

include_directories("${CMAKE_SOURCE_DIR}/")

if (NOT WIN32)
  if (WITH_TENSORRT AND WITH_GPU)
     include_directories("${TENSORRT_DIR}/include")
     link_directories("${TENSORRT_DIR}/lib")
  endif()
endif(NOT WIN32)

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/onnxruntime/lib")
link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
link_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
link_directories("${PADDLE_LIB}/paddle/lib")


if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  if (WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib
            ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.lib)
  else ()
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    execute_process(COMMAND cp -r ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} /usr/lib)
  endif ()
  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/onednn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if (WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else ()
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libdnnl.so.3)
    endif ()
  endif()
else()
  if (WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  else ()
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif ()
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_inference.so/a
if(WITH_STATIC_LIB)
  if(WIN32)
    set(DEPS
        ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS
        ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
else()
  if(WIN32)
    set(DEPS
        ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS
        ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif(WITH_STATIC_LIB)

if (NOT WIN32)
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags protobuf z xxhash
        )
    if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
    if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
else()
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags_static libprotobuf xxhash)
    set(DEPS ${DEPS} libcmt shlwapi)
    if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
    if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
endif(NOT WIN32)

if (EXISTS "${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
  set(DEPS ${DEPS} yaml-cpp)
endif()

if(WITH_GPU)
  if(NOT WIN32)
    if (WITH_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()


if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

set(DEPS ${DEPS} ${OpenCV_LIBS})

# Add OpenVINO dependencies if enabled
if (WITH_OPENVINO)
    set(DEPS ${DEPS} openvino::runtime)
endif()

include(FetchContent)
include(external-cmake/auto-log.cmake)
include_directories(${FETCHCONTENT_BASE_DIR}/extern_autolog-src)

aux_source_directory(./src SRCS)
# Conditionally add OpenVINO source files
if (WITH_OPENVINO)
    list(APPEND SRCS "./src/ocr_det_openvino.cpp")
    list(APPEND SRCS "./src/ocr_rec_openvino.cpp")
endif()

list(APPEND SRCS "./src/ocr_interface.cpp")

add_executable(${DEMO_NAME} ${SRCS})
target_link_libraries(${DEMO_NAME} ${DEPS})

if (WIN32 AND WITH_MKL)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./release/mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./release/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./release/mkldnn.dll
    )
endif()
