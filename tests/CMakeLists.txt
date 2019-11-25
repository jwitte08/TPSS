project(Tests
  LANGUAGES CXX
  DESCRIPTION "Collection of tests verifying the implementation of TPSS"
  )

message(STATUS "Configuring implementation tests within subdirectory 'tests'")

enable_testing()

set(testfiles
  # Tpatch_info_01.cc
  Tpatch_info_mpi.cc
  # Tpatch_transfer_01.cc
  # Tpatch_transfer_02.cc
  Ttensors_01.cc
  # Tblock_matrix_diagonal_01.cc
  Tlaplace_check_iterations.cc
  Tpoisson_check_iterations.cc
  # Tlaplace_geometry.cc
  # Tlaplace_integrator_cp.cc
  # Tlaplace_integrator_mf.cc
  # Tlaplace_integrator_vp.cc
  # Tmlaplace_integrator.cc
  # Tmlaplace_check_iterations.cc
  # Tlinelasticity_integrator_mf.cc
  # Tlinelasticity_integrator_fd.cc
  )

add_custom_target(setup_tests)

foreach(test ${testfiles})
  string(REPLACE ".cc" "" target ${test})
  add_executable(${target} EXCLUDE_FROM_ALL ${test})
  target_include_directories(${target} PUBLIC
    ${CMAKE_SOURCE_DIR}/apps
    ${CMAKE_BINARY_DIR}/include
    )
  target_link_libraries(${target} tpss)
  DEAL_II_SETUP_TARGET(${target})
  add_test(NAME ${target} COMMAND ${target})
  add_dependencies(setup_tests ${target})
endforeach(test)

set (src_files
  tensors.cc
  apps_poisson.cc
  )

add_custom_target(setup_mytests)

foreach(test ${src_files})
  string(REPLACE ".cc" "" target ${test})
  add_executable(${target} EXCLUDE_FROM_ALL ${test})
  target_include_directories(${target} PUBLIC
    ${CMAKE_SOURCE_DIR}/apps
    ${CMAKE_BINARY_DIR}/include
    )
  target_compile_features(${target} PUBLIC cxx_std_17)
  ## Keywords PUBLIC|INTERFACE|PRIVATE aren't possible since DEAL_II_SETUP_TARGET doesn't use keywords.
  target_link_libraries(${target} tpss gtest gtest_main)
  DEAL_II_SETUP_TARGET(${target})
  add_test(NAME ${target} COMMAND ${target})
  add_dependencies(setup_mytests ${target})
endforeach(test)


