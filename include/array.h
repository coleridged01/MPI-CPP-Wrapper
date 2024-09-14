#ifndef ARRAY_H
#define ARRAY_H

#include <vector>


namespace mpi {

template <typename T>
class array {
public:

    using value_type = T;

    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        explicit Iterator(T* ptr) : ptr_(ptr) {}

        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }

        Iterator& operator++() { ++ptr_; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++ptr_; return tmp; }

        Iterator& operator--() { --ptr_; return *this; }

        Iterator operator--(int) { Iterator tmp = *this; --ptr_; return tmp; }

        Iterator& operator+=(difference_type n) { ptr_ += n; return *this; }
        Iterator& operator-=(difference_type n) { ptr_ -= n; return *this; }

        Iterator operator+(difference_type n) const { return Iterator(ptr_ + n); }
        Iterator operator-(difference_type n) const { return Iterator(ptr_ - n); }

        difference_type operator-(const Iterator& other) const { return ptr_ - other.ptr_; }

        bool operator==(const Iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const Iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const Iterator& other) const { return ptr_ < other.ptr_; }
        bool operator<=(const Iterator& other) const { return ptr_ <= other.ptr_; }
        bool operator>(const Iterator& other) const { return ptr_ > other.ptr_; }
        bool operator>=(const Iterator& other) const { return ptr_ >= other.ptr_; }

    private:
        T* ptr_;
    };

    array() : size_(0), array_(nullptr) {}

    explicit array(const size_t size)
        : size_(size), array_(new T[size_]) {
        if (!array_) {
            throw std::bad_alloc();
        }
    }

    explicit array(const std::vector<T>& vector)
        : size_(vector.size()), array_(new T[size_]) {
        if (!array_) {
            throw std::bad_alloc();
        }
        std::copy(vector.begin(), vector.end(), this->begin());
    }

    explicit array(const std::vector<T>& vector, const size_t offset, const size_t size)
        : size_(size), array_(new T[size_]) {
        if (!array_) {
            throw std::bad_alloc();
        }
        const size_t diff = size_ - size;
        std::copy(vector.begin() + static_cast<long>(offset),
            vector.end() - static_cast<long>(diff), this->begin());
    }

    array(const array& other)
        : size_(other.size()), array_(new T[size_]) {
        if (!array_) {
            throw std::bad_alloc();
        }
        std::copy(other.begin(), other.end(), this->begin());
    }

    explicit array(const array& other, const size_t offset, const size_t size)
        : size_(size), array_(new T[size_]) {
        if (offset + size > other.size()) {
            throw std::out_of_range("array::array");
        }
        if (!array_) {
            throw std::bad_alloc();
        }
        const size_t diff = size_ - size;
        std::copy(other.begin() + static_cast<long>(offset),
                  other.end() - static_cast<long>(diff), this->begin());
    }

    array(array&& other) noexcept {
        array_ = other.array_;
        size_ = other.size_;
        other.array_ = nullptr;
        other.size_ = 0;
    }

    array(std::initializer_list<T> init_list)
        : size_(init_list.size()), array_(new T[size_]) {
        std::copy(init_list.begin(), init_list.end(), array_);
    }

    explicit operator std::vector<T>() {
        return std::vector<T>(std::make_move_iterator(this->begin()),
                          std::make_move_iterator(this->end()));
    }

    array& operator=(const array& other) {
        if (this != &other) {
            if (size_ < other.size_) {
                throw std::out_of_range("array::operator= (copy)");
            }
            std::copy(other.begin(), other.end(), this->begin());
        }
        return *this;
    }

    array& operator=(array&& other) noexcept {
        if (this != &other) {
            delete[] array_;
            array_ = other.array_;
            size_ = other.size_;
            other.array_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~array() {
        delete[] array_;
    }

    Iterator begin() const { return Iterator(array_); }

    Iterator end() const { return Iterator(array_ + size_); }

    [[nodiscard]] size_t size() const { return size_; }

    [[nodiscard]] bool empty() const { return size_ == 0; }

    [[nodiscard]] T* data() const { return array_; }

    void clear() const {
        std::memset(array_, 0, size_ * sizeof(T));
    }

    T& operator[](const size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("array::operator[]");
        }
        return array_[index];
    }

private:

    size_t size_;

    T* array_;

};

template<typename T>
array(T, size_t) -> array<T>;


}

#endif //ARRAY_H
