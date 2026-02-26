cmake_minimum_required(VERSION 3.22)
project(gpu_alignment LANGUAGES CXX CUDA)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── CUDA settings ────────────────────────────────────────────────────────────
# CMake 3.22 doesn't know CUDA C++20 dialect.  We pass --std=c++20 via flags
# and avoid setting CMAKE_CUDA_STANDARD to prevent the "CUDA20" dialect error.

# Default to sm_75 (Turing / T4).  Override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++20 --expt-relaxed-constexpr")

# ── Options ───────────────────────────────────────────────────────────────────
option(ENABLE_PROFILING "Enable ncu-compatible profiling markers (cudaProfilerStart/Stop)" OFF)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Sources ───────────────────────────────────────────────────────────────────
set(KERNEL_SOURCES
    src/kernels/attention_naive.cu
    src/kernels/attention_tiled.cu
    src/kernels/attention_tiled_db.cu
    src/kernels/ssm_scan.cu
)

set(MODELING_SOURCES
    src/modeling/analytical_model.cpp
    src/modeling/roofline.cpp
)

# ── Profiling sources (only compiled when ENABLE_PROFILING is ON) ────────────
set(PROFILING_SOURCES "")
if(ENABLE_PROFILING)
    set(PROFILING_SOURCES
        src/profiling/ncu_metrics.cpp
        src/profiling/validation_report.cpp
        src/profiling/plot_data.cpp
    )
endif()

# ── Benchmark executable ──────────────────────────────────────────────────────
add_executable(benchmark
    src/main.cpp
    ${KERNEL_SOURCES}
    ${MODELING_SOURCES}
    ${PROFILING_SOURCES}
)

target_link_libraries(benchmark PRIVATE CUDA::cudart)

if(ENABLE_PROFILING)
    target_compile_definitions(benchmark PRIVATE ENABLE_PROFILING)
endif()

# CMake 3.22 lacks the CUDA20 dialect.  Pin the target to CUDA17 so CMake
# doesn't try to propagate CXX_STANDARD 20; actual C++20 comes from --std=c++20
# in CMAKE_CUDA_FLAGS above.
set_target_properties(benchmark PROPERTIES CUDA_STANDARD 17)

# ── Compiler warnings ────────────────────────────────────────────────────────
target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
