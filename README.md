# MPI C++ Wrapper



## Introduction

This project introduces a simple C++ wrapper for MPI, originally implemented in C and Fortran.
By leveraging OpenMPI’s current implementation, the wrapper provides a more idiomatic and functional C++ interface.
The goal is to eliminate the need for raw pointers, manual resource management, and verbose syntax, 
allowing developers to focus on data processing and parallelization in a more declarative way.

Key C++ concepts such as RAII, templates, SFINAE, lambdas, and operator overloading are used to achieve this. 
While the wrapper does not aim to cover the entire MPI interface, 
it includes support for common operations like scatter, gather, reduce, all-reduce, and others.

## Features

- **Idiomatic C++ Interface**: Provides a clean and familiar interface for C++ developers, avoiding raw pointers and manual memory management.

- **RAII (Resource Acquisition Is Initialization)**: Automatically manages resources, ensuring proper cleanup of MPI objects and reducing the risk of memory leaks.

- **Templates and SFINAE**: Leverages C++ templates and SFINAE to allow for flexible and type-safe APIs that adapt to various data types and scenarios.

- **Lambda Support**: Enables the use of lambdas for defining custom behavior in parallel operations, simplifying the code for common MPI tasks with minimal to no overhead.

- **Sensible Operator Overloading**: Implements operator overloading to provide intuitive and readable syntax for performing MPI operations like communication and reduction.

- **Common MPI Operations**: Includes support for the most commonly used MPI operations such as:
    - `scatter`
    - `gather`
    - `reduce`
    - `all-reduce`
    - and more.

- **Error Handling with C++ Exceptions**: Improves upon traditional MPI error handling by integrating C++ exceptions, making it easier to detect and manage errors during runtime.

- **C++20 Compatibility**: Fully compatible with modern C++ standards, ensuring ease of use with the latest language features.

- **High Performance**: Designed to maintain the high performance of underlying MPI operations, with minimal overhead added by the C++ abstraction layer.

- **Extensible Design**: The wrapper is easily extendable, allowing users to add new features or customize existing MPI operations as needed.

- **Integration with STL and Other Libraries**: Seamlessly works with the standard C++ library and can be integrated with other common libraries and frameworks.

- **Unit Tests and Documentation**: Comes with unit tests and examples to help developers get started quickly.

## Examples

### Gaussian Elimination

## Dependencies

This library only depends on the OpenMPI library. It can be installed with:

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```
