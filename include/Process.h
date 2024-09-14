#ifndef PROCESS_H
#define PROCESS_H



namespace mpi {

class Process {
public:

    static int ROOT;

    explicit Process(const int rank, const int commSize) : rank_(rank), commSize_(commSize) {}

    explicit Process(const Process& other) = delete;

    Process(Process&& other) noexcept = default;

    Process& operator=(const Process& other) = delete;

    Process& operator=(Process&& other) = default;

    [[nodiscard]] int rank() const;

    [[nodiscard]] int commSize() const;

protected:

    int rank_;  // MPI rank

    int commSize_;

};

}


#endif //PROCESS_H
