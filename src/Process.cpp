#include <Process.h>



namespace mpi {

int Process::rank() const {
    return rank_;
}

int Process::commSize() const {
    return commSize_;
}

bool Process::operator==(const Type type) const {
    return rank_ == static_cast<int>(type);
}

}
