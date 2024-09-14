#ifndef HELPERMAPMPI_H
#define HELPERMAPMPI_H

#include <mpi.h>


template<typename>
struct is_mpi_type : std::false_type {};

template<> struct is_mpi_type<char> : std::true_type {};
template<> struct is_mpi_type<unsigned char> : std::true_type {};
template<> struct is_mpi_type<short> : std::true_type {};
template<> struct is_mpi_type<unsigned short> : std::true_type {};
template<> struct is_mpi_type<int> : std::true_type {};
template<> struct is_mpi_type<unsigned int> : std::true_type {};
template<> struct is_mpi_type<long> : std::true_type {};
template<> struct is_mpi_type<unsigned long> : std::true_type {};
template<> struct is_mpi_type<long long> : std::true_type {};
template<> struct is_mpi_type<float> : std::true_type {};
template<> struct is_mpi_type<double> : std::true_type {};

template<typename T>
MPI_Datatype get_mpi_type();

template<>
inline MPI_Datatype get_mpi_type<char>() { return MPI_CHAR; }

template<>
inline MPI_Datatype get_mpi_type<unsigned char>() { return MPI_UNSIGNED_CHAR; }

template<>
inline MPI_Datatype get_mpi_type<short>() { return MPI_SHORT; }

template<>
inline MPI_Datatype get_mpi_type<unsigned short>() { return MPI_UNSIGNED_SHORT; }

template<>
inline MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template<>
inline MPI_Datatype get_mpi_type<unsigned int>() { return MPI_UNSIGNED; }

template<>
inline MPI_Datatype get_mpi_type<long>() { return MPI_LONG; }

template<>
inline MPI_Datatype get_mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }

template<>
inline MPI_Datatype get_mpi_type<long long>() { return MPI_LONG_LONG; }

template<>
inline MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }

template<>
inline MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

#endif //HELPERMAPMPI_H
