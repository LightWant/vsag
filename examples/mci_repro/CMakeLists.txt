cmake_minimum_required(VERSION 3.18)
project(mci_hdf5_e2e LANGUAGES C CXX)

set(VSAG_ROOT "" CACHE PATH "VSAG source checkout or installed prefix")
set(VSAG_LIBRARY "" CACHE FILEPATH "Optional explicit path to libvsag.so/dylib")
find_path(VSAG_INCLUDE_DIR vsag/vsag.h HINTS "${VSAG_ROOT}/include" REQUIRED)
if(NOT VSAG_LIBRARY)
    # Prefer an already built shared library; no VSAG sources are compiled here.
    find_library(VSAG_SHARED_LIBRARY NAMES vsag
        HINTS "${VSAG_ROOT}/build-release/src" "${VSAG_ROOT}/build-release/lib"
              "${VSAG_ROOT}/lib" "${VSAG_ROOT}/lib64" REQUIRED)
    set(VSAG_LIBRARY "${VSAG_SHARED_LIBRARY}")
endif()
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(mci_hdf5_e2e mci_hdf5_e2e.cpp)
target_compile_features(mci_hdf5_e2e PRIVATE cxx_std_17)
target_compile_options(mci_hdf5_e2e PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(mci_hdf5_e2e SYSTEM PRIVATE "${VSAG_INCLUDE_DIR}" ${HDF5_INCLUDE_DIRS})
target_link_libraries(mci_hdf5_e2e PRIVATE "${VSAG_LIBRARY}" ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)
get_filename_component(VSAG_LIBRARY_DIR "${VSAG_LIBRARY}" DIRECTORY)
set_target_properties(mci_hdf5_e2e PROPERTIES BUILD_RPATH "${VSAG_LIBRARY_DIR}")
