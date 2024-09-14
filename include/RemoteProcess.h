#ifndef REMOTEPROCESS_H
#define REMOTEPROCESS_H

#include <mpi.h>
#include <Process.h>



namespace mpi {

class RemoteProcess final : public Process {
public:

    class Awaitable {
    public:

        explicit Awaitable(std::unique_ptr<MPI_Request>&& request)
            : request_(std::move(request)) {}

        void operator()() const {
            MPI_Wait(request_.get(), MPI_STATUS_IGNORE);
        }

    private:

        std::unique_ptr<MPI_Request> request_ = nullptr;

    };

    class SyncFunctor {
    public:

        explicit SyncFunctor(const int rank) : rank_(rank) {}

        template<typename T>
        std::enable_if_t<!is_mpi_type<T>::value, void>
        operator<<(const T& data) {
            MPI_Send(&data, sizeof(T), MPI_BYTE, rank_, 0, MPI_COMM_WORLD);
        }

        template<typename T>
        std::enable_if_t<is_mpi_type<T>::value, void>
        operator<<(const T& data) {
            MPI_Send(&data, 1, get_mpi_type<T>(), rank_, 0,
                MPI_COMM_WORLD);
        }

        template <typename T>
        std::enable_if_t<!is_mpi_type<T>::value, void>
        operator>>(const T& data) {
            MPI_Recv(&data, sizeof(T), MPI_BYTE, rank_, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        template <typename T>
        std::enable_if_t<is_mpi_type<T>::value, void>
        operator>>(const T& data) {
            MPI_Recv(&data, 1, get_mpi_type<T>(), rank_, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        template<typename T>
        std::enable_if_t<!is_mpi_type<T>::value, void>
        operator<<(const array<T>& data) {
            MPI_Send(data.data(), static_cast<int>(data.size() * sizeof(T)),
                MPI_BYTE, rank_, 0, MPI_COMM_WORLD);
        }

        template<typename T>
        std::enable_if_t<is_mpi_type<T>::value, void>
        operator<<(const array<T>& data) {
            MPI_Send(data.data(), static_cast<int>(data.size()),
                get_mpi_type<T>(), rank_, 0, MPI_COMM_WORLD);
        }

        template <typename T>
        std::enable_if_t<!is_mpi_type<T>::value, void>
        operator>>(const array<T>& data) {
            MPI_Recv(data.data(), static_cast<int>(data.size()) * sizeof(T),
                MPI_BYTE, rank_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        template <typename T>
        std::enable_if_t<is_mpi_type<T>::value, void>
        operator>>(const array<T>& data) {
            MPI_Recv(data.data(), static_cast<int>(data.size()),
                get_mpi_type<T>(), rank_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    private:

        int rank_;

    };

    class AsyncFunctor {
    public:
        explicit AsyncFunctor(const int rank) : rank_(rank) {}

        template<typename T>
        std::enable_if_t<!is_mpi_type<T>::value, Awaitable>
        operator<<(const T& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Isend(&data, sizeof(T), MPI_BYTE, rank_, 0,
                MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template<typename T>
        std::enable_if_t<is_mpi_type<T>::value, Awaitable>
        operator<<(const T& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Isend(&data, 1, get_mpi_type<T>(), rank_, 0,
                MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template <typename T>
        std::enable_if_t<!is_mpi_type<T>::value, Awaitable>
        operator>>(const T& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Irecv(&data, sizeof(T), MPI_BYTE, rank_, 0,
                MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template <typename T>
        std::enable_if_t<is_mpi_type<T>::value, Awaitable>
        operator>>(const T& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Irecv(&data, 1, get_mpi_type<T>(), rank_, 0,
                MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template<typename T>
        std::enable_if_t<!is_mpi_type<T>::value, Awaitable>
        operator<<(const array<T>& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Isend(data.data(), static_cast<int>(data.size() * sizeof(T)),
                MPI_BYTE, rank_, 0, MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template<typename T>
        std::enable_if_t<is_mpi_type<T>::value, Awaitable>
        operator<<(const array<T>& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Isend(data.data(), static_cast<int>(data.size()),
                get_mpi_type<T>(), rank_, 0, MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template <typename T>
        std::enable_if_t<!is_mpi_type<T>::value, Awaitable>
        operator>>(const array<T>& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Irecv(data.data(), static_cast<int>(data.size()) * sizeof(T),
                MPI_BYTE, rank_, 0, MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

        template <typename T>
        std::enable_if_t<is_mpi_type<T>::value, Awaitable>
        operator>>(const array<T>& data) {
            auto request = std::make_unique<MPI_Request>();
            MPI_Irecv(data.data(), static_cast<int>(data.size()),
                get_mpi_type<T>(), rank_, 0, MPI_COMM_WORLD, request.get());
            return Awaitable(std::move(request));
        }

    private:

        int rank_;

    };

    explicit RemoteProcess(const int rank, const int commSize) : Process(rank, commSize) {}

    explicit RemoteProcess(const RemoteProcess& other) = delete;

    RemoteProcess(RemoteProcess&& other) = default;

    RemoteProcess& operator=(const RemoteProcess& other) = delete;

    RemoteProcess& operator=(RemoteProcess&& other) = default;

    [[nodiscard]] SyncFunctor sync() const {
        return SyncFunctor(rank());
    }

    [[nodiscard]] AsyncFunctor async() const {
        return AsyncFunctor(rank());
    }

};

}

#endif //REMOTEPROCESS_H
