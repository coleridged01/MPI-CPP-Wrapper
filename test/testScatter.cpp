#include <iostream>
#include <MPIEnvironment.h>
#include <Operations.h>
#include <ostream>
#include <thread>



int main(int argc, char** argv) {

    const mpi::MPIEnvironment environment{argc, argv};

    const auto local = environment.getLocalProcess().lock();

    if (!local) {
        throw std::runtime_error("No MPI local process was provided");
    }

    mpi::array chunk = mpi::scatter(
    local->init<int>(
        [](const mpi::array<int>& data) {
            for (int i = 0; auto& val : data) {
                val = i++;
            }
        }, 16)
    );

    for (auto& val : chunk) {
        val = val * val;
    }

    const mpi::array result = mpi::gather<int>(local->forward(std::move(chunk)));

    if (!result.empty()) {
        for (const auto& val : result) {
            std::cout << val << std::endl;
        }
    }
}
