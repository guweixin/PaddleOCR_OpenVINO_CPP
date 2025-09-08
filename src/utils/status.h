

#ifndef STATUS_H_
#define STATUS_H_

#include <string>
#include <optional>
#include <stdexcept>

// 用标准库替代 Status 和 StatusOr
class Status {
public:
    Status() : ok_(true) {}
    Status(const std::string& message) : ok_(false), message_(message) {}
    
    static Status OK() { return Status(); }
    static Status InvalidArgumentError(const std::string& message) {
        return Status("Invalid argument: " + message);
    }
    static Status NotFoundError(const std::string& message) {
        return Status("Not found: " + message);
    }
    static Status InternalError(const std::string& message) {
        return Status("Internal error: " + message);
    }
    static Status ErrnoToStatus(int error_code, const std::string& message) {
        return Status("Error " + std::to_string(error_code) + ": " + message);
    }
    
    bool ok() const { return ok_; }
    std::string ToString() const { return ok_ ? "OK" : message_; }
    std::string message() const { return message_; }
    
private:
    bool ok_;
    std::string message_;
};

template<typename T>
class StatusOr {
public:
    // 默认构造函数
    StatusOr() : has_value_(false), status_(Status::InternalError("Uninitialized StatusOr")) {}
    
    StatusOr(const T& value) : has_value_(true), value_(value), status_(Status::OK()) {}
    StatusOr(const Status& status) : has_value_(false), status_(status) {}
    
    bool ok() const { return has_value_; }
    const T& value() const { 
        if (!has_value_) throw std::runtime_error(status_.ToString());
        return value_; 
    }
    T& value() { 
        if (!has_value_) throw std::runtime_error(status_.ToString());
        return value_; 
    }
    const Status& status() const { return status_; }
    
    // 重载解引用操作符
    const T& operator*() const { return value(); }
    T& operator*() { return value(); }
    
    // 重载箭头操作符
    const T* operator->() const { return &value(); }
    T* operator->() { return &value(); }
    
private:
    bool has_value_;
    T value_;
    Status status_;
};

#endif // STATUS_H_

