#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <MPIEnvironment.h>
#include <Operations.h>
#include <thread>
#include <doctest/doctest.h>



TEST_CASE("Root/Worker Execute") {

    int argc = 1;
    std::vector<char*> argv;
    argv.push_back("testMPIWrapper");
    auto ptr = argv.data();

    const mpi::MPIEnvironment environment{argc, ptr};

    auto local = environment.getLocalProcess();

    if (auto l = local.lock()) {
        (*l)([l] {
            CHECK(l->rank() == static_cast<int>(mpi::LocalProcess::Type::ROOT));
            std::cout << "Running from root process"  << std::endl;
        });
        (~*l)([l] {
            CHECK(l->rank() != static_cast<int>(mpi::LocalProcess::Type::ROOT));
            std::cout << "Running from worker process"  << std::endl;
        });
    }
}

TEST_CASE("ScatterAndGather") {
    int argc = 1;
    std::vector<char*> argv;
    argv.push_back("testMPIWrapper");
    auto ptr = argv.data();

    const mpi::MPIEnvironment environment{argc, ptr};

    const auto local = environment.getLocalProcess().lock();

    CHECK(local);

    mpi::array chunk = mpi::scatter(
    local->init<int>(
        [](const mpi::array<int>& data) {
            CHECK(data.size() == 32);
            CHECK(data[0] == 0);
            for (int i = 0; auto& val : data) {
                val = i++;
            }
        }, 32)
    );

    CHECK(32 % chunk.size() == 0);

    for (auto& val : chunk) {
        val = val * val;
    }

    const mpi::array result = mpi::gather<int>(local->forward(std::move(chunk)));

    if (!result.empty()) {
        CHECK(result.size() == 32);
        for (int i = 0; const auto& val : result) {
            CHECK(val == i * i);
            i++;
        }
    }
}

