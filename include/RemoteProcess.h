#ifndef REMOTEPROCESS_H
#define REMOTEPROCESS_H

#include <mpi.h>
#include <Process.h>



namespace mpi {

class RemoteProcess final : public Process {
public:

    explicit RemoteProcess(const int rank, const int commSize) : Process(rank, commSize) {}

    explicit RemoteProcess(const RemoteProcess& other) = delete;

    RemoteProcess(RemoteProcess&& other) = default;

    RemoteProcess& operator=(const RemoteProcess& other) = delete;

    RemoteProcess& operator=(RemoteProcess&& other) = default;

    template<typename T>
    RemoteProcess& operator<<(T&& data) {
        MPI_Send(&data, sizeof(T), MPI_BYTE, rank_, 0, MPI_COMM_WORLD);
        return *this;
    }

    template <typename T>
    RemoteProcess& operator>>(T& data) {
        MPI_Recv(&data, sizeof(T), MPI_BYTE, rank_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return *this;
    }
};

}

#endif //REMOTEPROCESS_H
