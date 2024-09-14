#define DOCTEST_CONFIG_IMPLEMENT
#include <MPIEnvironment.h>
#include <Operations.h>
#include <thread>
#include <doctest/doctest.h>



std::unique_ptr<mpi::MPIEnvironment> mpi_env;

TEST_CASE("Root/Worker Execute") {

    auto local = mpi_env->getLocalProcess();

    if (auto l = local.lock()) {
        (*l)([l] {
            CHECK(l->rank() == static_cast<int>(mpi::LocalProcess::Type::ROOT));
            std::cout << "Running from root process"  << std::endl;
        });
        (~*l)([l] {
            CHECK(l->rank() != static_cast<int>(mpi::LocalProcess::Type::ROOT));
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
        CHECK(result.size() == DATASIZE / mpi_env->getCommSize());
        for (int i = 0; const auto& val : result) {
            int acc = 0;
            for (int j = 0; j < mpi_env->getCommSize(); j++) {
                acc += (j * static_cast<int>(result.size()) + i) * (j * static_cast<int>(result.size()) + i);
            }
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

    CHECK(chunk.size() == DATASIZE / mpi_env->getCommSize());

    // Process Data in some way
    for (auto& val : chunk) {
        val = val * val;
    }

    size_t size = chunk.size();

    const mpi::array result = mpi::allReduce<int>(*local + std::move(chunk));

    CHECK(result.size() == size);

    CHECK(result.size() == DATASIZE / mpi_env->getCommSize());
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
        CHECK(result.size() == DATASIZE * mpi_env->getCommSize());
        for (size_t i = 0; const auto& val : result) {
            CHECK(val == i * i);
            i = (i + 1) % DATASIZE;
        }
        CHECK(local->rank() == 0);
    } else {
        CHECK(local->rank() != 0);
    }
}

int main(int argc, char** argv) {
    mpi_env = std::make_unique<mpi::MPIEnvironment>(argc, argv);

    doctest::Context context;
    context.applyCommandLine(argc, argv);

    return context.run();;
}
