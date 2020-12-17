#!/bin/bash

rm -r CMakeCache.txt CMakeFiles

cmake                                                \
    -DTPL_ENABLE_MPI=ON                              \
    -DTPL_ENABLE_gtest=OFF                           \
    -DTrilinos_ENABLE_Amesos=ON                      \
    -DTrilinos_ENABLE_AztecOO=ON                     \
    -DTrilinos_ENABLE_Epetra=ON                      \
    -DTrilinos_ENABLE_EpetraExt=ON                   \
    -DTrilinos_ENABLE_Ifpack=ON                      \
    -DTrilinos_ENABLE_ML=ON                          \
    -DTrilinos_ENABLE_MueLu=ON                       \
    -DTrilinos_ENABLE_ROL=ON                         \
    -DTrilinos_ENABLE_Sacado=ON                      \
    -DTrilinos_ENABLE_Teuchos=ON                     \
    -DTrilinos_ENABLE_Tpetra=ON                      \
    -DTrilinos_ENABLE_Zoltan=ON                      \
    -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON              \
    -DTrilinos_ENABLE_COMPLEX_FLOAT=ON               \
    -DTrilinos_VERBOSE_CONFIGURE=OFF                 \
    -DTrilinos_ENABLE_TESTS=OFF                      \
    -DTrilinos_ENABLE_Gtest=OFF                      \
    -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON      \
    -DBUILD_SHARED_LIBS=ON                           \
    -DCMAKE_VERBOSE_MAKEFILE=OFF                     \
    -DCMAKE_BUILD_TYPE=RELEASE                       \
    -DCMAKE_INSTALL_PREFIX:PATH=../install           \
    ../
