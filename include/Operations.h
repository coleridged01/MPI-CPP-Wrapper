#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <mpi_types.h>
#include <LocalProcess.h>



namespace mpi {

template<typename T>
[[nodiscard]] std::enable_if_t<!is_mpi_type<T>::value, array<T>>
scatter(LocalProcess::in_op_args<T>&& args) {
    auto& [data, commSize, size] = args;
    const size_t chunkSize = size / commSize;
    array<T> chunk(chunkSize);
    const int read = static_cast<int>(sizeof(T) * chunkSize);
    MPI_Scatter(data.data(), read, MPI_BYTE,
        chunk.data(), read, MPI_BYTE,
        static_cast<int>(LocalProcess::Type::ROOT), MPI_COMM_WORLD);
    return chunk;
}

template<typename T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
scatter(LocalProcess::in_op_args<T>&& args) {
    auto& [local, data, size] = args;
    const size_t chunkSize = size / static_cast<size_t>(local.commSize());
    array<T> chunk(chunkSize);
    const int read =  static_cast<int>(chunkSize);
    MPI_Scatter(data.data(), read, get_mpi_type<T>(),
        chunk.data(), read, get_mpi_type<T>(),
        static_cast<int>(LocalProcess::Type::ROOT), MPI_COMM_WORLD);
    return chunk;
}

template<typename T>
[[nodiscard]]std::enable_if_t<!is_mpi_type<T>::value, array<T>>
broadcast(LocalProcess::in_op_args<T>&& args) {
    auto& [local, data, size] = args;
    if (local.rank() != static_cast<int>(LocalProcess::Type::ROOT)) {
        data = array<T>(size);
    }
    MPI_Bcast(data.data(), static_cast<int>(data.size) * sizeof(T), MPI_BYTE,
            static_cast<int>(LocalProcess::Type::ROOT), MPI_COMM_WORLD);
    return data;
}

template<typename T>
[[nodiscard]]std::enable_if_t<is_mpi_type<T>::value, array<T>>
broadcast(LocalProcess::in_op_args<T>&& args) {
    auto& [local, data, size] = args;
    if (local.rank() != static_cast<int>(LocalProcess::Type::ROOT)) {
        data = array<T>(size);
    }
    MPI_Bcast(data.data(), static_cast<int>(data.size()), get_mpi_type<T>(),
        static_cast<int>(LocalProcess::Type::ROOT), MPI_COMM_WORLD);
    return data;
}

template<typename T>
[[nodiscard]]std::enable_if_t<!is_mpi_type<T>::value, array<T>>
gather(LocalProcess::out_op_args<T>&& args) {
    auto& [local, chunk] = args;
    array<T> data;
    if (local.rank() == static_cast<int>(LocalProcess::Type::ROOT)) {
        data = array<T>(chunk.size() * local.commSize());
    }
    const int read = static_cast<int>(chunk.size()) * sizeof(T);
    MPI_Gather(chunk.data(), read, MPI_BYTE,
            data.data(), read, MPI_BYTE,
            static_cast<int>(LocalProcess::Type::ROOT), MPI_COMM_WORLD);
    return data;
}

template<typename T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
gather(LocalProcess::out_op_args<T>&& args) {
    auto& [local, chunk] = args;
    array<T> data;
    if (local.rank() == static_cast<int>(LocalProcess::Type::ROOT)) {
        data = array<T>(chunk.size() * static_cast<size_t>(local.commSize()));
    }
    const int read = static_cast<int>(chunk.size());
    MPI_Gather(chunk.data(), read, get_mpi_type<T>(),
            data.data(), read, get_mpi_type<T>(),
            static_cast<int>(LocalProcess::Type::ROOT), MPI_COMM_WORLD);
    return data;
}

template<typename T>
[[nodiscard]]std::enable_if_t<!is_mpi_type<T>::value, array<T>>
allGather(LocalProcess::out_op_args<T>&& args) {
    auto& [local, chunk] = args;
    array<T> data(chunk.size() * local.commSize());
    const int read = static_cast<int>(chunk.size() * sizeof(T));
    MPI_Allgather(chunk.data(), read, MPI_BYTE,
            data.data(), read, MPI_BYTE, MPI_COMM_WORLD);
    return data;
}

template<typename T>
[[nodiscard]]std::enable_if_t<is_mpi_type<T>::value, array<T>>
allGather(LocalProcess::out_op_args<T>&& args) {
    auto& [local, chunk] = args;
    array<T> data(chunk.size() * local.commSize());
    const int read = static_cast<int>(chunk.size());
    MPI_Allgather(chunk.data(), read, get_mpi_type<T>(),
            data.data(), read, get_mpi_type<T>(),
            MPI_COMM_WORLD);
   return data;
}

template<typename T>
[[nodiscard]] std::enable_if_t<!is_mpi_type<T>::value, array<T>>
allToAll(LocalProcess::out_op_args<T>&& args) {
    auto& [local, data] = args;
    array<T> ret(data.size() * local.commSize());
    const int read = static_cast<int>(data.size()) * sizeof(T);
    MPI_Alltoall(data.data(), read, MPI_BYTE,
           ret.data(), read, MPI_BYTE, MPI_COMM_WORLD);
    return ret;
}

template<typename T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
allToAll(LocalProcess::out_op_args<T>&& args) {
    auto& [local, data] = args;
    array<T> ret(data.size() * local.commSize());
    MPI_Alltoall(data.data(), data.size(), get_mpi_type<T>(),
            ret.data(), data.size(), get_mpi_type<T>(), MPI_COMM_WORLD);
    return ret;
}


template<class T>
[[nodiscard]] std::enable_if_t<!is_mpi_type<T>::value, array<T>>
reduce(LocalProcess::arith_op_args<T>&& op) {
        auto& [local, src, mop] = op;
        array<T> ret(src.size());
        MPI_Reduce(src.data(), ret.data(), static_cast<int>(src.size() * sizeof(T)),
            MPI_BYTE, mop, static_cast<int>(Process::Type::ROOT), MPI_COMM_WORLD);
        return ret;
}


template<class T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
reduce(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;
    array<T> ret;
    if (local.rank() == static_cast<int>(LocalProcess::Type::ROOT)) {
        ret = array<T>(src.size());
    }
    MPI_Reduce(src.data(), ret.data(), static_cast<int>(src.size()),
        get_mpi_type<T>(), mop, static_cast<int>(Process::Type::ROOT), MPI_COMM_WORLD);
    return ret;
}

template<class T>
[[nodiscard]] std::enable_if_t<!is_mpi_type<T>::value, array<T>>
allReduce(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;
    array<T> ret(src.size());
    MPI_Allreduce(src.data(), ret.data(), static_cast<int>(src.size() * sizeof(T)),
        MPI_BYTE, mop, MPI_COMM_WORLD);
    return ret;
}

template<class T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
allReduce(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;
    array<T> ret(src.size());
    MPI_Allreduce(src.data(), ret.data(), static_cast<int>(src.size()),
        get_mpi_type<T>(), mop, MPI_COMM_WORLD);
    return ret;
}

template<class T>
[[nodiscard]] std::enable_if_t<!is_mpi_type<T>::value, array<T>>
scan(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;
    array<T> ret(src.size());
    MPI_Scan(src.data(), src.data(), static_cast<int>(src.size() * sizeof(T)),
        MPI_BYTE, mop, MPI_COMM_WORLD);
    return ret;
}

template<class T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
scan(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;
    array<T> ret(src.size());
    MPI_Scan(src.data(), ret.data(), static_cast<int>(src.size()),
        get_mpi_type<T>(), mop, MPI_COMM_WORLD);
    return ret;
}

template<class T>
[[nodiscard]] std::enable_if_t<!is_mpi_type<T>::value, array<T>>
reduceScatter(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;
    const auto commSize = static_cast<size_t>(local.commSize());
    const array<int> count(commSize);
    count.clear();
    for (size_t i = 0; i < src.size(); i = (i + 1) % commSize) {
        count[i] += sizeof(T);
    }
    array<T> ret(static_cast<size_t>(count[static_cast<size_t>(local.rank())]));
    MPI_Reduce_scatter(src.data(), ret.data(),
        count.data(), MPI_BYTE, mop, MPI_COMM_WORLD);
    return ret;
}

template<class T>
[[nodiscard]] std::enable_if_t<is_mpi_type<T>::value, array<T>>
reduceScatter(LocalProcess::arith_op_args<T>&& op) {
    auto& [local, src, mop] = op;

    const auto commSize = static_cast<size_t>(local.commSize());
    const array<int> count(commSize);
    count.clear();
    for (size_t i = 0; i < src.size(); i = (i + 1) % commSize) {
        count[i] += 1;
    }
    array<T> ret(static_cast<size_t>(count[static_cast<size_t>(local.rank())]));
    MPI_Reduce_scatter(src.data(), ret.data(), count.data(), get_mpi_type<T>(),
        mop, MPI_COMM_WORLD);
    return ret;
}

}

#endif //OPERATIONS_H
