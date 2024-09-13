#include <MPIEnvironment.h>
#include <thread>

int main(int argc, char** argv) {
    const mpi::MPIEnvironment environment{argc, argv};

    const auto local = environment.getLocalProcess().lock();

    if (!local) {
        return EXIT_FAILURE;
    }

    const auto remotes = environment.getRemoteProcesses().lock();

    if (!remotes) {
        return EXIT_FAILURE;
    }

    local->execRoot([&remotes] {
        for (auto& remote : *remotes) {
            remote << remote.rank();
        }
        for (auto& remote : *remotes) {
            int rec;
            remote >> rec;
            if (rec != remote.rank()) {
                throw std::runtime_error("Received wrong rank");
            }
        }
    });

    local->execNonRoot([&remotes, &local] {
        const auto it = std::ranges::find_if(*remotes, [](auto& remote) {
            return remote.rank() == static_cast<int>(mpi::Process::Type::ROOT);
        });
        if (it == remotes->end()) {
            throw std::runtime_error("Received wrong rank");
        }
        int rec;
        *it >> rec;

        if (rec != local->rank()) {
            throw std::runtime_error("Received wrong rank");
        }
        *it << rec;
    });

    return EXIT_SUCCESS;
}
