#ifndef MPIENVIRONMENT_H
#define MPIENVIRONMENT_H

#include <LocalProcess.h>
#include <RemoteProcess.h>

#include <vector>


namespace mpi {

class MPIEnvironment {
public:

    MPIEnvironment(int &argc, char** &argv);

    [[nodiscard]] int getPoolSize() const;

    [[nodiscard]] std::weak_ptr<LocalProcess> getLocalProcess() const;

    [[nodiscard]] std::weak_ptr<std::vector<RemoteProcess>> getRemoteProcesses() const;

    ~MPIEnvironment();

private:

    int commSize_;

    std::shared_ptr<LocalProcess> local_process_;

    std::shared_ptr<std::vector<RemoteProcess>> remote_processes_ =
        std::make_shared<std::vector<RemoteProcess>>();

};

}

#endif //MPIENVIRONMENT_H
