#include <Process.h>



namespace mpi {

int Process::ROOT = 0;

int Process::rank() const {
    return rank_;
}

int Process::commSize() const {
    return commSize_;
}

}
