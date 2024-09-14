#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <MPIEnvironment.h>
#include <Operations.h>
#include <thread>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>



std::unique_ptr<mpi::MPIEnvironment> mpi_env;

bool areEqual(const double a, const double b, const double e = 0.0001) {
    return std::abs(a - b) < e;
}

TEST_CASE("Root/Worker Execute") {

    auto local = mpi_env->getLocalProcess();

    if (auto l = local.lock()) {
        (*l)([l] {
            CHECK(l->rank() == mpi::Process::ROOT);
            std::cout << "Running from root process"  << std::endl;
        });
        (~*l)([l] {
            CHECK(l->rank() != mpi::Process::ROOT);
            std::cout << "Running from worker process"  << std::endl;
        });
        for (int i = 0; i < l->commSize(); i++) {
            (*l)[i]([l, i] {
                CHECK(l->rank() == i);
            });
        }
    } else {
        CHECK(false);
    }
}

TEST_CASE("Scatter&Gather") {
    const auto local = mpi_env ->getLocalProcess().lock();

    CHECK(local);

    constexpr size_t DATASIZE = 32;

    mpi::array chunk = mpi::scatter(
    local->init<int>(
        [](const mpi::array<int>& data) {
            CHECK(data.size() == 32);
            CHECK(data[0] == 0);
            for (int i = 0; auto& val : data) {
                val = i++;
            }
        }, DATASIZE)
    );

    CHECK(DATASIZE % chunk.size() == 0);

    for (auto& val : chunk) {
        val = val * val;
    }

    const mpi::array result = mpi::gather<int>(local->forward(std::move(chunk)));

    if (!result.empty()) {
        CHECK(result.size() == DATASIZE);
        for (int i = 0; const auto& val : result) {
            CHECK(val == i * i);
            i++;
        }
    }
}

TEST_CASE("Scatter&Reduce") {
    const auto local = mpi_env->getLocalProcess().lock();

    CHECK(local);

    constexpr size_t DATASIZE = 32;

    mpi::array chunk = mpi::scatter(
    local->init<int>(
        [](const mpi::array<int>& data) {
            CHECK(data.size() == 32);
            CHECK(data[0] == 0);
            for (int i = 0; auto& val : data) {
                val = i++;
            }
        }, DATASIZE)
    );

    CHECK(DATASIZE % chunk.size() == 0);

    for (auto& val : chunk) {
        val = val * val;
    }

    const mpi::array result = mpi::reduce<int>(*local + std::move(chunk));

    if (!result.empty()) {
        CHECK(result.size() == DATASIZE / static_cast<size_t>(mpi_env->getCommSize()));
        for (int i = 0; const auto& val : result) {
            int acc = 0;
            for (int j = 0; j < mpi_env->getCommSize(); j++) {
                acc += (j * static_cast<int>(result.size()) + i) * (j * static_cast<int>(result.size()) + i);
            }
            CHECK(val == acc);
            i++;
        }
        CHECK(local->rank() == 0);
    } else {
        CHECK(local->rank() != 0);
    }
}

TEST_CASE("Scatter&AllReduce") {
    const auto local = mpi_env->getLocalProcess().lock();

    CHECK(local);

    constexpr size_t DATASIZE = 32;

    mpi::array chunk = mpi::scatter(
    local->init<int>(
        [](const mpi::array<int>& data) {
            CHECK(data.size() == 32);
            CHECK(data[0] == 0);
            for (int i = 0; auto& val : data) {
                val = i++;
            }
        }, DATASIZE)
    );

    CHECK(chunk.size() == DATASIZE / static_cast<size_t>(mpi_env->getCommSize()));

    // Process Data in some way
    for (auto& val : chunk) {
        val = val * val;
    }

    size_t size = chunk.size();

    const mpi::array result = mpi::allReduce<int>(*local + std::move(chunk));

    CHECK(result.size() == size);

    CHECK(result.size() == DATASIZE / static_cast<size_t>(mpi_env->getCommSize()));
    for (int i = 0; const auto& val : result) {
        int acc = 0;
        for (int j = 0; j < mpi_env->getCommSize(); j++) {
            acc += (j * static_cast<int>(result.size()) + i) * (j * static_cast<int>(result.size()) + i);
        }
        CHECK(val == acc);
        i++;
    }
}

TEST_CASE("Broadcast&Gather") {

    const auto local = mpi_env->getLocalProcess().lock();

    CHECK(local);

    constexpr size_t DATASIZE = 32;

    mpi::array data = mpi::broadcast(
    local->init<int>(
        [](const mpi::array<int>& data) {
            CHECK(data.size() == 32);
            CHECK(data[0] == 0);
            for (int i = 0; auto& val : data) {
                val = i++;
            }
        }, DATASIZE)
    );

    CHECK(data.size() == DATASIZE);

    // Process Data in some way
    for (auto& val : data) {
        val = val * val;
    }

    const mpi::array result = mpi::gather<int>(
        local->forward(std::move(data))
    );

    if (!result.empty()) {
        CHECK(result.size() == DATASIZE * static_cast<size_t>(mpi_env->getCommSize()));
        for (size_t i = 0; const auto& val : result) {
            CHECK(val == i * i);
            i = (i + 1) % DATASIZE;
        }
        CHECK(local->rank() == 0);
    } else {
        CHECK(local->rank() != 0);
    }
}

TEST_CASE("GaussianElimination") {

    const std::vector solution = {
        1.0, -0.50, 0.0, 1.50,
        0.0, 1.00, 0.571429, -1.00,
        0.0, 0.0, 1.0, -2.33333,
        0.0, 0.0, 0.0, 1.0
    };

    constexpr int N = 4;
    constexpr int M = 4;
    auto local = mpi_env->getLocalProcess().lock();

    auto remote = mpi_env->getRemoteProcesses().lock();

    CHECK(local);
    CHECK(remote);

    auto chunk = mpi::scatter(local->init<double>(
        [](mpi::array<double>& data) {
            data = {
                2, -1, 0, 3,
                1, 3, 2, -2,
                0, 4, 1, -1,
                5, -2, 3, 4
            };
        }, N * M)
    );

    size_t rowsPerProcess = chunk.size() / M;

    for (size_t k = 0; k < N; ++k) {
        int mappedProcess = static_cast<int>(k / rowsPerProcess);
        (*local)(mappedProcess);
        (*local)(
        [k, rowsPerProcess, &chunk, remote, mappedProcess] {
            auto row = k % rowsPerProcess;
            auto pivot = chunk[row * M + k];
            for (size_t l = k; l < M; l++) {
                chunk[row * M + l] /= pivot;
            }

            std::vector<mpi::RemoteProcess::Awaitable> awaits;

            for (auto& r : *remote) {
                if (r.rank() > mappedProcess) {
                    awaits.push_back(r.async() << mpi::array(chunk, M * row, M));
                }
            }

            for (size_t elim = row + 1; elim < rowsPerProcess; ++elim) {
                const double scale = chunk[elim * M + k];

                for (size_t l = k; l < M; l++) {
                    chunk[elim * M + l] -= chunk[row * M + l] * scale;
                }
            }

            for (auto& await : awaits) {
                await();
            }
        });
        (~*local)([&remote, &chunk, rowsPerProcess, k, mappedProcess, &local] {
            const mpi::array<double> pivotRow(M);
            const auto it = std::ranges::find_if(*remote,
            [](const mpi::RemoteProcess& r){ return r.rank() == mpi::LocalProcess::ROOT; });

            if (local->rank() > mappedProcess) {
                it->sync() >> pivotRow;

                for (size_t elim = 0; elim < rowsPerProcess; ++elim) {
                    double scale = chunk[elim * M + k];

                    for (size_t l = k; l < M; l++) {
                        chunk[elim * M + l] -= pivotRow[l] * scale;
                    }
                }
            }
        });
    }

    (*local)(0);

    mpi::array<double> result = mpi::gather(
        local->forward(std::move(chunk))
    );

    (*local)([&result, &solution] {
        for (size_t i = 0; i < result.size(); ++i) {
            CHECK(areEqual(result[i], solution[i]));
        }
    });
}

int main(int argc, char** argv) {
    mpi_env = std::make_unique<mpi::MPIEnvironment>(argc, argv);

    doctest::Context context;
    context.applyCommandLine(argc, argv);

    return context.run();
}
