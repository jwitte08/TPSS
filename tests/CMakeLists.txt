project(Tests
  LANGUAGES CXX
  DESCRIPTION "Collection of tests verifying the implementation of TPSS"
  )

message(STATUS "Configuring implementation tests within subdirectory 'tests'")

enable_testing()


## TODO use gtest
# set(testfiles
  # # Tpatch_info_01.cc
  # Tpatch_info_mpi.cc
  # # Ttensors_01.cc ## obsolete
  # # Tblock_matrix_diagonal_01.cc
  # Tlaplace_check_iterations.cc ## obsolete
  # Tpoisson_check_iterations.cc ## obsolete
  # # Tlaplace_geometry.cc
  # # Tlaplace_integrator_cp.cc
  # # Tlaplace_integrator_mf.cc
  # # Tlaplace_integrator_vp.cc
  # # Tmlaplace_integrator.cc
  # # Tmlaplace_check_iterations.cc
  # # Tlinelasticity_integrator_mf.cc
  # # Tlinelasticity_integrator_fd.cc
  # )

set (src_files
  apps_biharmonic_integrator.cc
  apps_coloring.cc
  apps_laplace_integrator.cc
  # apps_linelasticity.cc !!!
  apps_linelasticity_integrator.cc
  apps_poisson.cc
  apps_stokes_integrator.cc
  block_matrix.cc
  patch_transfer.cc
  tensors.cc
  tensor_product_matrix.cc
  kroneckersvd.cc
  block_matrix_base.cc
  )

add_custom_target(setup_tests)

foreach(test ${src_files})
  string(REPLACE ".cc" "" target ${test})
  add_executable(${target} EXCLUDE_FROM_ALL ${test})
  target_include_directories(${target} PUBLIC
    ${CMAKE_SOURCE_DIR}/apps
    ${CMAKE_BINARY_DIR}/include
    )
  target_compile_features(${target} PUBLIC cxx_std_17)
  ## Keywords PUBLIC|INTERFACE|PRIVATE aren't possible since DEAL_II_SETUP_TARGET doesn't use keywords.
  target_link_libraries(${target} tpss gtest gtest_main pthread)
  DEAL_II_SETUP_TARGET(${target})
  add_test(NAME ${target} COMMAND ${target})
  add_dependencies(setup_tests ${target})
endforeach(test)
