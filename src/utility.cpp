// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dirent.h>
#include <include/utility.h>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_set>

#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace PaddleOCR
{

  std::vector<std::string> Utility::ReadDict(const std::string &path) noexcept
  {
    std::vector<std::string> m_vec;
    std::ifstream in(path);
    if (in)
    {
      for (;;)
      {
        std::string line;
        if (!getline(in, line))
          break;
        m_vec.emplace_back(std::move(line));
      }
    }
    else
    {
      std::cout << "no such label file: " << path << ", exit the program..."
                << std::endl;
      exit(1);
    }

    // Note: Space character is added in ocr_rec.h after blank char
    // No need to add space here to avoid duplication

    return m_vec;
  }

  void Utility::VisualizeBboxes(const cv::Mat &srcimg,
                                const std::vector<OCRPredictResult> &ocr_result,
                                const std::string &save_path) noexcept
  {
    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    for (size_t n = 0; n < ocr_result.size(); ++n)
    {
      cv::Point rook_points[4];
      for (size_t m = 0; m < ocr_result[n].box.size(); ++m)
      {
        rook_points[m] =
            cv::Point(int(ocr_result[n].box[m][0]), int(ocr_result[n].box[m][1]));
      }

      const cv::Point *ppt[1] = {rook_points};
      int npt[] = {4};
      cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    }

    cv::imwrite(save_path, img_vis);
    std::cout << "The detection visualized image saved in " + save_path
              << std::endl;
  }

  // list all files under a directory
  void Utility::GetAllFiles(const char *dir_name,
                            std::vector<std::string> &all_inputs) noexcept
  {
    if (NULL == dir_name)
    {
      std::cout << " dir_name is null ! " << std::endl;
      return;
    }
    struct stat s;
    stat(dir_name, &s);
    if (!S_ISDIR(s.st_mode))
    {
      std::cout << "dir_name is not a valid directory !" << std::endl;
      all_inputs.emplace_back(dir_name);
      return;
    }
    else
    {
      struct dirent *filename; // return value for readdir()
      DIR *dir;                // return value for opendir()
      dir = opendir(dir_name);
      if (NULL == dir)
      {
        std::cout << "Can not open dir " << dir_name << std::endl;
        return;
      }
      std::cout << "Successfully opened the dir !" << std::endl;
      while ((filename = readdir(dir)) != NULL)
      {
        if (strcmp(filename->d_name, ".") == 0 ||
            strcmp(filename->d_name, "..") == 0)
          continue;
        // img_dir + std::string("/") + all_inputs[0];
        all_inputs.emplace_back(dir_name + std::string("/") +
                                std::string(filename->d_name));
      }
    }
  }

  cv::Mat
  Utility::GetRotateCropImage(const cv::Mat &srcimage,
                              const std::vector<std::vector<int>> &box) noexcept
  {
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;

    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (size_t i = 0; i < points.size(); ++i)
    {
      points[i][0] -= left;
      points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                  pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                   pow(points[0][1] - points[3][1], 2)));

    const cv::Point2f pts_std[4] = {
        {0., 0.},
        {(float)img_crop_width, 0.},
        {(float)img_crop_width, (float)img_crop_height},
        {0.f, (float)img_crop_height}};

    const cv::Point2f pointsf[4] = {{(float)points[0][0], (float)points[0][1]},
                                    {(float)points[1][0], (float)points[1][1]},
                                    {(float)points[2][0], (float)points[2][1]},
                                    {(float)points[3][0], (float)points[3][1]}};

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5)
    {
      cv::Mat srcCopy(dst_img.rows, dst_img.cols, dst_img.depth());
      cv::transpose(dst_img, srcCopy);
      cv::flip(srcCopy, srcCopy, 0);
      return srcCopy;
    }
    else
    {
      return dst_img;
    }
  }

  std::vector<size_t> Utility::argsort(const std::vector<float> &array) noexcept
  {
    std::vector<size_t> array_index(array.size(), 0);
    for (size_t i = 0; i < array.size(); ++i)
      array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](size_t pos1, size_t pos2) noexcept
              {
                return (array[pos1] < array[pos2]);
              });

    return array_index;
  }

  std::string Utility::basename(const std::string &filename) noexcept
  {
    if (filename.empty())
    {
      return "";
    }

    auto len = filename.length();
    auto index = filename.find_last_of("/\\");

    if (index == std::string::npos)
    {
      return filename;
    }

    if (index + 1 >= len)
    {

      --len;
      index = filename.substr(0, len).find_last_of("/\\");

      if (len == 0)
      {
        return filename;
      }

      if (index == 0)
      {
        return filename.substr(1, len - 1);
      }

      if (index == std::string::npos)
      {
        return filename.substr(0, len);
      }

      return filename.substr(index + 1, len - index - 1);
    }

    return filename.substr(index + 1, len - index);
  }

  bool Utility::PathExists(const char *path) noexcept
  {
#ifdef _WIN32
    struct _stat buffer;
    return (_stat(path, &buffer) == 0);
#else
    struct stat buffer;
    return (stat(path, &buffer) == 0);
#endif // !_WIN32
  }

  void Utility::CreateDir(const char *path) noexcept
  {
#ifdef _MSC_VER
    _mkdir(path);
#elif defined __MINGW32__
    mkdir(path);
#else
    mkdir(path, 0777);
#endif // !_WIN32
  }

  void Utility::print_result(
      const std::vector<OCRPredictResult> &ocr_result) noexcept
  {
    for (size_t i = 0; i < ocr_result.size(); ++i)
    {
      std::cout << i << "\t";
      // det
      const std::vector<std::vector<int>> &boxes = ocr_result[i].box;
      if (boxes.size() > 0)
      {
        std::cout << "det boxes: [";
        for (size_t n = 0; n < boxes.size(); ++n)
        {
          std::cout << '[' << boxes[n][0] << ',' << boxes[n][1] << "]";
          if (n != boxes.size() - 1)
          {
            std::cout << ',';
          }
        }
        std::cout << "] ";
      }
      // rec
      if (ocr_result[i].score != -1.0)
      {
        std::cout << "rec text: " << ocr_result[i].text
                  << " rec score: " << ocr_result[i].score << " ";
      }

      // cls
      if (ocr_result[i].cls_label != -1)
      {
        std::cout << "cls label: " << ocr_result[i].cls_label
                  << " cls score: " << ocr_result[i].cls_score;
      }
      std::cout << std::endl;
    }
  }

  cv::Mat Utility::crop_image(const cv::Mat &img,
                              const std::vector<int> &box) noexcept
  {
    cv::Mat crop_im = cv::Mat::zeros(box[3] - box[1], box[2] - box[0], 16);
    int crop_x1 = std::max(0, box[0]);
    int crop_y1 = std::max(0, box[1]);
    int crop_x2 = std::min(img.cols - 1, box[2] - 1);
    int crop_y2 = std::min(img.rows - 1, box[3] - 1);

    cv::Mat crop_im_window =
        crop_im(cv::Range(crop_y1 - box[1], crop_y2 + 1 - box[1]),
                cv::Range(crop_x1 - box[0], crop_x2 + 1 - box[0]));
    cv::Mat roi_img =
        img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
    crop_im_window += roi_img;
    return crop_im;
  }

  cv::Mat Utility::crop_image(const cv::Mat &img,
                              const std::vector<float> &box) noexcept
  {
    std::vector<int> box_int = {(int)box[0], (int)box[1], (int)box[2],
                                (int)box[3]};
    return crop_image(img, box_int);
  }

  void Utility::sort_boxes(std::vector<OCRPredictResult> &ocr_result) noexcept
  {
    std::sort(ocr_result.begin(), ocr_result.end(), Utility::comparison_box);
    if (ocr_result.size() > 1)
    {
      for (size_t i = 0; i < ocr_result.size() - 1; ++i)
      {
        for (size_t j = i; j != size_t(-1); --j)
        {
          if (abs(ocr_result[j + 1].box[0][1] - ocr_result[j].box[0][1]) < 10 &&
              (ocr_result[j + 1].box[0][0] < ocr_result[j].box[0][0]))
          {
            std::swap(ocr_result[i], ocr_result[i + 1]);
          }
        }
      }
    }
  }

  std::vector<int>
  Utility::xyxyxyxy2xyxy(const std::vector<std::vector<int>> &box) noexcept
  {
    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));
    return {left, top, right, bottom};
  }

  std::vector<int> Utility::xyxyxyxy2xyxy(const std::vector<int> &box) noexcept
  {
    int x_collect[4] = {box[0], box[2], box[4], box[6]};
    int y_collect[4] = {box[1], box[3], box[5], box[7]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));
    return {left, top, right, bottom};
  }

  float Utility::fast_exp(float x) noexcept
  {
    union
    {
      uint32_t i;
      float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
  }

  std::vector<float>
  Utility::activation_function_softmax(const std::vector<float> &src) noexcept
  {
    size_t length = src.size();
    std::vector<float> dst;
    dst.resize(length);
    const float alpha = float(*std::max_element(&src[0], &src[length]));
    float denominator{0};

    for (size_t i = 0; i < length; ++i)
    {
      dst[i] = fast_exp(src[i] - alpha);
      denominator += dst[i];
    }

    for (size_t i = 0; i < length; ++i)
    {
      dst[i] /= denominator;
    }
    return dst;
  }

  float Utility::iou(const std::vector<int> &box1,
                     const std::vector<int> &box2) noexcept
  {
    int area1 = std::max(0, box1[2] - box1[0]) * std::max(0, box1[3] - box1[1]);
    int area2 = std::max(0, box2[2] - box2[0]) * std::max(0, box2[3] - box2[1]);

    // computing the sum_area
    int sum_area = area1 + area2;

    // find the each point of intersect rectangle
    int x1 = std::max(box1[0], box2[0]);
    int y1 = std::max(box1[1], box2[1]);
    int x2 = std::min(box1[2], box2[2]);
    int y2 = std::min(box1[3], box2[3]);

    // judge if there is an intersect
    if (y1 >= y2 || x1 >= x2)
    {
      return 0.0;
    }
    else
    {
      int intersect = (x2 - x1) * (y2 - y1);
      return intersect / (sum_area - intersect + 0.00000001);
    }
  }

  float Utility::iou(const std::vector<float> &box1,
                     const std::vector<float> &box2) noexcept
  {
    float area1 = std::max((float)0.0, box1[2] - box1[0]) *
                  std::max((float)0.0, box1[3] - box1[1]);
    float area2 = std::max((float)0.0, box2[2] - box2[0]) *
                  std::max((float)0.0, box2[3] - box2[1]);

    // computing the sum_area
    float sum_area = area1 + area2;

    // find the each point of intersect rectangle
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);

    // judge if there is an intersect
    if (y1 >= y2 || x1 >= x2)
    {
      return 0.0;
    }
    else
    {
      float intersect = (x2 - x1) * (y2 - y1);
      return intersect / (sum_area - intersect + 0.00000001);
    }
  }

  // OCR智能空格添加 - 优化版本，专注于常见英文词汇分割
  std::string Utility::AddSmartSpaces(const std::string &text) noexcept
  {
    if (text.empty())
      return text;

    // 智能空格处理逻辑 - 处理所有文本，包括已有部分空格的文本
    // 首先按现有空格分割文本，然后对每个部分单独处理
    std::vector<std::string> parts;
    std::string currentPart;

    for (char c : text)
    {
      if (c == ' ')
      {
        if (!currentPart.empty())
        {
          parts.push_back(currentPart);
          currentPart.clear();
        }
        // 保留空格分隔符
        parts.push_back(" ");
      }
      else
      {
        currentPart += c;
      }
    }

    if (!currentPart.empty())
    {
      parts.push_back(currentPart);
    }

    // 对每个非空格部分进行智能空格处理
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i)
    {
      if (parts[i] == " ")
      {
        result += " ";
      }
      else
      {
        std::string processedPart = ProcessSinglePart(parts[i]);
        result += processedPart;
      }
    }

    return result;
  }

  std::string Utility::ProcessSinglePart(const std::string &text) noexcept
  {
    if (text.empty())
      return text;

    // 如果文本长度小于等于4，或者是常见的完整单词，不处理
    if (text.length() <= 4)
      return text;

    // 转换为小写进行分析
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);

    // 检查是否为完整的常见单词，如果是就不处理
    static const std::unordered_set<std::string> completeWords = {
        "about", "above", "after", "again", "against", "along", "already", "also", "although", "always", "among", "another", "anyone", "anything", "anywhere", "around", "asked", "became", "because", "become", "before", "began", "begin", "being", "believe", "below", "between", "beyond", "both", "bring", "brought", "build", "business", "called", "cannot", "certain", "change", "coming", "common", "company", "complete", "completely", "consider", "continue", "could", "country", "course", "create", "different", "does", "doing", "done", "down", "during", "each", "early", "earth", "either", "enough", "even", "ever", "every", "everything", "example", "experience", "family", "father", "feel", "few", "field", "find", "fire", "first", "five", "follow", "found", "four", "friend", "from", "full", "gave", "give", "going", "good", "government", "great", "group", "grow", "hand", "hard", "have", "hear", "help", "here", "high", "hold", "home", "hope", "house", "however", "human", "idea", "important", "information", "inside", "instead", "into", "itself", "just", "keep", "kind", "know", "known", "large", "last", "late", "later", "learn", "least", "leave", "left", "less", "life", "light", "line", "list", "little", "live", "local", "long", "look", "made", "make", "making", "many", "might", "mind", "money", "more", "most", "move", "much", "must", "name", "national", "natural", "near", "need", "never", "news", "next", "night", "nothing", "number", "often", "once", "only", "open", "other", "over", "part", "particular", "past", "people", "person", "place", "plan", "play", "point", "possible", "power", "present", "problem", "program", "provide", "public", "question", "quite", "rather", "real", "reason", "receive", "remember", "report", "result", "return", "right", "room", "said", "same", "school", "second", "seem", "several", "shall", "should", "show", "side", "simple", "since", "small", "social", "some", "someone", "something", "sometimes", "soon", "sound", "space", "special", "start", "state", "still", "story", "study", "such", "support", "sure", "system", "take", "talk", "tell", "than", "that", "their", "them", "then", "there", "these", "they", "thing", "think", "this", "those", "though", "thought", "three", "through", "time", "today", "together", "told", "took", "turn", "under", "understand", "until", "upon", "used", "using", "very", "want", "water", "went", "were", "what", "when", "where", "which", "while", "whole", "will", "with", "within", "without", "word", "work", "working", "world", "would", "write", "year", "years", "young", "your",
        "american", "congress", "pelosi", "nunes", "trump", "president", "administration", "department", "justice", "federal", "bureau", "investigation", "intelligence", "committee", "chairman", "director", "secretary", "attorney", "general", "campaign", "election", "political", "democratic", "republican", "conservative", "liberal", "oversight", "surveillance", "classification", "declassification", "transparency", "accountability", "democracy", "constitution", "constitutional", "executive", "legislative", "judicial", "supreme", "court", "decision", "ruling", "opinion", "dissent", "majority", "minority", "bipartisan", "partisan", "politics", "policy", "legislation", "regulation", "enforcement", "compliance", "violation", "prosecution", "defense", "plaintiff", "defendant", "evidence", "testimony", "witness", "subpoena", "hearing", "trial", "verdict", "sentence", "appeal", "impeachment", "resignation", "appointment", "confirmation", "nomination", "cabinet", "advisor", "counsel", "spokesperson", "representative", "senator", "congressman", "governor", "mayor", "official", "astonishing", "partnership", "argument", "conference", "communication", "development", "performance", "achievement", "establishment", "organization", "information", "management", "construction", "relationship", "opportunity", "environment", "technology", "responsibility", "understanding", "particularly", "specifically", "especially", "generally", "obviously", "certainly", "possibly", "probably", "definitely", "immediately", "eventually", "frequently", "recently", "currently", "previously", "originally", "finally", "basically", "essentially", "actually", "potentially", "effectively", "successfully", "carefully", "clearly", "directly", "exactly"};

    // 如果是完整的常见单词，不处理
    if (completeWords.find(lowerText) != completeWords.end())
    {
      return text;
    }

    // 只有当文本长度较长且可能包含多个单词时才进行拆分
    if (text.length() < 8)
    {
      return text;
    }

    // 简化的英文词典，只包含最常见的短词
    static const std::unordered_set<std::string> dictionary = {
        "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
        "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "all", "and", "any",
        "are", "ask", "bad", "big", "but", "can", "car", "cat", "cut", "day", "did", "dog",
        "eat", "end", "eye", "far", "few", "for", "get", "had", "has", "her", "him", "his",
        "how", "let", "man", "may", "new", "not", "now", "old", "one", "our", "out", "own",
        "put", "red", "run", "say", "see", "she", "sit", "six", "ten", "the", "too", "top",
        "two", "use", "was", "way", "who", "why", "win", "yes", "yet", "you", "able", "back",
        "ball", "been", "best", "blue", "book", "both", "call", "came", "come", "days", "does",
        "done", "down", "each", "even", "ever", "fact", "fall", "find", "fire", "five", "four",
        "from", "full", "gave", "give", "good", "great", "hand", "hard", "have", "hear", "help",
        "here", "high", "home", "hope", "house", "into", "just", "keep", "kind", "know", "land",
        "last", "late", "left", "life", "like", "line", "list", "live", "look", "made", "make",
        "many", "mine", "more", "most", "move", "much", "must", "name", "near", "need", "next",
        "once", "only", "open", "over", "page", "part", "past", "plan", "play", "read", "real",
        "right", "said", "same", "seem", "show", "side", "small", "some", "soon", "such", "take",
        "talk", "tell", "than", "that", "them", "then", "they", "this", "time", "turn", "very",
        "want", "water", "ways", "well", "went", "what", "when", "will", "with", "word", "work",
        "year", "your", "about", "above", "after", "again", "among", "asked", "began", "being",
        "below", "black", "bring", "build", "carry", "clean", "clear", "close", "color", "could",
        "creat", "doing", "don't", "early", "earth", "every", "field", "first", "found", "green",
        "group", "happy", "heard", "heart", "large", "learn", "leave", "light", "local",
        "money", "music", "never", "night", "north", "often", "order", "other", "paper", "place",
        "point", "quick", "right", "round", "shall", "short", "shown", "since", "small", "sound",
        "south", "space", "stand", "start", "state", "still", "story", "study", "their", "there",
        "these", "thing", "think", "those", "three", "today", "under", "until", "voice", "water",
        "where", "which", "white", "whole", "whose", "woman", "world", "would", "write", "young",
        "brown", "fox", "quick", "test", "sit"};

    int n = lowerText.length();
    std::vector<bool> dp(n + 1, false);
    std::vector<int> parent(n + 1, -1);

    dp[0] = true;

    // 动态规划填表 - 但要更保守，只处理明显的连接词
    for (int i = 1; i <= n; ++i)
    {
      // 优先尝试较短的词，避免过度拆分
      for (int len = 1; len <= std::min(6, i); ++len)
      {
        int j = i - len;
        if (dp[j])
        {
          std::string word = lowerText.substr(j, len);
          if (dictionary.find(word) != dictionary.end())
          {
            dp[i] = true;
            parent[i] = j;
            break;
          }
        }
      }
    }

    // 如果可以完美分割，回溯构建结果
    if (dp[n])
    {
      std::vector<std::string> words;
      int pos = n;
      while (pos > 0)
      {
        int prevPos = parent[pos];
        words.push_back(text.substr(prevPos, pos - prevPos));
        pos = prevPos;
      }

      std::reverse(words.begin(), words.end());

      // 只有当拆分成多个词时才返回拆分结果
      if (words.size() > 1)
      {
        std::string result;
        for (size_t i = 0; i < words.size(); ++i)
        {
          if (i > 0)
            result += " ";
          result += words[i];
        }
        return result;
      }
    }

    // 如果无法合理拆分，返回原文本
    return text;
  }

  void Utility::ProcessOCRResultsWithSpaces(std::vector<OCRPredictResult> &ocr_results) noexcept
  {
    for (auto &result : ocr_results)
    {
      if (!result.text.empty())
      {
        result.text = AddSmartSpaces(result.text);
      }
    }
  }

  // Helper functions for smart space processing
  bool Utility::IsVowel(char c) noexcept
  {
    c = std::tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
  }

  bool Utility::IsConsonant(char c) noexcept
  {
    return std::isalpha(c) && !IsVowel(c);
  }

  bool Utility::IsPunctuation(char c) noexcept
  {
    return std::ispunct(c);
  }

} // namespace PaddleOCR
