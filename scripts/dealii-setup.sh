# remove old cmake files
rm -rf CMakeFiles/ CMakeCache.txt

# build configuration
cmake \
    -D CMAKE_CXX_FLAGS="-march=native -Wno-array-bounds -Wno-literal-suffix -pthread -std=c++17" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3 -funroll-all-loops" \
    -D DEAL_II_CXX_FLAGS_DEBUG="-O0 -g" \
    -D CMAKE_C_FLAGS="-march=native -Wno-array-bounds -Wfatal-errors" \
    -D DEAL_II_WITH_MPI:BOOL="ON" \
    -D DEAL_II_LINKER_FLAGS="-lpthread" \
    -D DEAL_II_WITH_TRILINOS:BOOL="OFF" \
    -D DEAL_II_WITH_PETSC:BOOL="OFF" \
    -D DEAL_II_FORCE_BUNDLED_BOOST="OFF" \
    -D DEAL_II_WITH_GSL="OFF" \
    -D DEAL_II_WITH_NETCDF="OFF" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D DEAL_II_WITH_THREADS="ON" \
    -D DEAL_II_COMPONENT_DOCUMENTATION="OFF" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    ..
