#ifndef LOCALPROCESS_H
#define LOCALPROCESS_H

#include <functional>
#include <Process.h>
#include <array.h>
#include <mpi.h>
#include <mpi_types.h>
#include <utility>
#include <functional> // For std::invoke



namespace mpi {

class LocalProcess final : public Process {
public:

    template <typename Predicate>
    struct ProcessFunctor {

        explicit ProcessFunctor(Predicate&& pred)
        : pred_(std::forward<Predicate>(pred)) {}

        template<typename Func>
        void operator()(Func&& func) const {
            if (pred_()) {
                std::forward<Func>(func)();
            }
        }

    private:

        Predicate pred_;

    };

    template <typename Predicate>
    ProcessFunctor(Predicate) -> ProcessFunctor<Predicate>;

    /// Datatype returned after using overloaded ops and needed to perform mpi::reduce(),
    /// mpi::allReduce(), mpi::scan(), mpi::reduceScatter()
    template<typename T>
    using arith_op_args = std::tuple<const LocalProcess&, array<T>, MPI_Op>;

    template<typename T>
    using in_op_args = std::tuple<const LocalProcess&, array<T>, size_t>;

    template<typename T>
    using out_op_args = std::tuple<const LocalProcess&, array<T>>;

    explicit LocalProcess(const int rank, const int commSize) : Process(rank, commSize) {}

    // Assigns new Root Process
    void operator()(const int newRoot) const {
        ROOT = newRoot;
    }

    /// Runs on root
    void operator()(const std::function<void()>& func) const {
        if (this->rank_ == ROOT) {
            func();
        }
    }

    auto operator~() const {
        return ProcessFunctor([this] { return rank_ != ROOT; });
    }

    auto operator[](const int runOn) const {
        if (runOn < commSize_) {
            return ProcessFunctor([this, runOn] { return rank_ == runOn; });
        }
        throw std::out_of_range("LocalProcess::operator[]");
    }

    template<typename T, typename Func, typename... Args>
    in_op_args<T> init(Func&& initData, size_t size, const Args&... args) const {
        size = roundup(size);

        array<T> data;
        if (this->rank_ == ROOT) {
            data = array<T>(size);
            initData(data, args...);

        }
        return {*this, std::move(data), size};
    }

    /// Initializes an array for scatter operation with 0
    template<typename T>
    in_op_args<T> init(size_t size) const {
        size = roundup(size);

        array<T> data;
        if (this->rank_ == ROOT) {
            data = array<T>(size);
            data.clear();
        }
        return {*this, std::move(data), size};
    }

    /// Binds chunk with LocalProcess
    template<typename T>
    out_op_args<T> forward(array<T>&& chunk) const {
        return {*this, std::move(chunk)};
    }

    template<typename T>
    [[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, arith_op_args<T>>
    max(array<T>&& data) const {
        return {*this, std::move(data), MPI_MAX};
    }

    template<typename T>
    [[nodiscard]]std::enable_if_t<is_mpi_type<T>::value, arith_op_args<T>>
    min(array<T> data) const {
        return {*this, std::move(data), MPI_MIN};
    }

    template<typename T>
    [[nodiscard]]std::enable_if_t<is_mpi_type<T>::value, arith_op_args<T>>
    operator+(array<T>&& data) const {
        return {*this, std::move(data), MPI_SUM};
    }

    template<typename T>
    [[nodiscard]]std::enable_if_t<is_mpi_type<T>::value, arith_op_args<T>>
    operator*(array<T>&& data) const {
        return {*this, std::move(data), MPI_PROD};
    }

    template<typename T>
    [[nodiscard]]std::enable_if_t<is_mpi_type<T>::value &&
    !std::is_same_v<T, float> && !std::is_same_v<T, double>,arith_op_args<T>>
    operator&&(array<T>&& data) const {
        return {*this, std::move(data) MPI_LAND};
    }

    template<typename T>
    [[nodiscard]]std::enable_if_t<!std::is_same_v<T, float> && !std::is_same_v<T, double>, arith_op_args<T>>
    operator&(array<T>&& data) const {
        return {*this, std::move(data), MPI_BAND};
    }

    template<typename T>
    [[nodiscard]] std::enable_if_t<is_mpi_type<T>::value &&
    !std::is_same_v<T, float> && !std::is_same_v<T, double>, arith_op_args<T>>
    operator||(array<T>&& data) const {
        return{*this, std::move(data), MPI_LOR};
    }

    template<typename T>
    [[nodiscard]] std::enable_if_t<!std::is_same_v<T, float> && !std::is_same_v<T, double>,arith_op_args<T>>
    operator|(array<T> data) const {
        return {*this, std::move(data), MPI_BOR};
    }

    template<typename T>
    [[nodiscard]] std::enable_if_t<is_mpi_type<T>::value &&
    !std::is_same_v<T, float> && !std::is_same_v<T, double>, arith_op_args<T>>
    operator!=(array<T>&& data) const {
        return {*this, std::move(data), MPI_LXOR};
    }

    template<typename T>
    [[nodiscard]] std::enable_if_t<!std::is_same_v<T, float> && !std::is_same_v<T, double>, arith_op_args<T>>
    operator^(array<T>&& data) const {
        return {*this, std::move(data), MPI_BXOR};
    }

private:

    [[nodiscard]] size_t roundup(const size_t size) const {
        const auto commSize = static_cast<size_t>(commSize_);
        return (size + commSize - 1) / commSize * commSize;
    }

};

}

#endif //LOCALPROCESS_H
