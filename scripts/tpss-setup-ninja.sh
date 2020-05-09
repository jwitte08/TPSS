cmake -DCMAKE_BUILD_TYPE="Release" \
      -DCMAKE_CXX_FLAGS="-march=native" \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -funroll-all-loops" \
      -GNinja \
      ..
