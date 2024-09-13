#include <MPIEnvironment.h>

#include <mpi.h>



namespace mpi {

MPIEnvironment::MPIEnvironment(int &argc, char **&argv) :
    commSize_([&argc, &argv] {
        if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
           throw std::runtime_error("MPI Initialization failed");
        }
        int commSize;
        MPI_Comm_size(MPI_COMM_WORLD, &commSize);
        return commSize;
    }()),
    local_process_([this] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return std::make_shared<LocalProcess>(rank, commSize_);
    }()) {
        for (int i = 0; i < commSize_; ++i) {
        if (i != local_process_->rank()) {
            remote_processes_->emplace_back(i, commSize_);
        }
    }
}

int MPIEnvironment::getPoolSize() const {
    return commSize_;
}

std::weak_ptr<LocalProcess> MPIEnvironment::getLocalProcess() const {
    return local_process_;
}

std::weak_ptr<std::vector<RemoteProcess>> MPIEnvironment::getRemoteProcesses() const {
    return remote_processes_;
}

MPIEnvironment::~MPIEnvironment() {
    MPI_Finalize();
}

}
