#ifndef SIMPLE_LOGGER_H_
#define SIMPLE_LOGGER_H_

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <cstdarg>

// Simple logger implementation using standard C++ libraries
namespace SimpleLogger {

enum LogLevel {
    DEBUG_LEVEL = 0,
    INFO_LEVEL = 1, 
    WARNING_LEVEL = 2,
    ERROR_LEVEL = 3
};

inline std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char timeStr[100];
    
    // Use thread-safe localtime function based on platform
    #ifdef _WIN32
    struct tm tm_buf;
    localtime_s(&tm_buf, &now);
    std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm_buf);
    #else
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm_buf);
    #endif
    
    return std::string(timeStr);
}

inline const char* levelToString(LogLevel level) {
    switch (level) {
        case DEBUG_LEVEL: return "DEBUG";
        case INFO_LEVEL: return "INFO";
        case WARNING_LEVEL: return "WARN";
        case ERROR_LEVEL: return "ERROR";
        default: return "UNKNOWN";
    }
}

inline void log(LogLevel level, const char* file, int line, const char* format, ...) {
    // Extract filename from full path
    const char* filename = std::strrchr(file, '/');
    if (filename == nullptr) {
        filename = std::strrchr(file, '\\');
    }
    filename = (filename != nullptr) ? filename + 1 : file;
    
    // Format the message using va_list
    va_list args;
    va_start(args, format);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    // Print to stderr for warnings and errors, stdout for others
    std::ostream& stream = (level >= WARNING_LEVEL) ? std::cerr : std::cout;
    
    stream << "[" << getCurrentTime() << "] " 
           << "[" << levelToString(level) << "] "
           << "[" << filename << ":" << line << "] "
           << buffer << std::endl;
}

} // namespace SimpleLogger

// Define macros for backward compatibility
#define INFOD(fmt, ...) SimpleLogger::log(SimpleLogger::DEBUG_LEVEL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define INFOV(fmt, ...) SimpleLogger::log(SimpleLogger::DEBUG_LEVEL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define INFO(fmt, ...)  SimpleLogger::log(SimpleLogger::INFO_LEVEL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define INFOW(fmt, ...) SimpleLogger::log(SimpleLogger::WARNING_LEVEL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define INFOE(fmt, ...) SimpleLogger::log(SimpleLogger::ERROR_LEVEL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define INFOF(fmt, ...) SimpleLogger::log(SimpleLogger::ERROR_LEVEL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#endif // SIMPLE_LOGGER_H_
