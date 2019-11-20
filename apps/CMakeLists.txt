project(Apps
  LANGUAGES CXX
  DESCRIPTION "Numerical applications and model problems using fast tensor product Schwarz smoothers (TPSS)"
  )

message(STATUS "Configuring numerical applications within subdirectory 'apps'")

# configure_file(
#   ${CMAKE_SOURCE_DIR}/scripts/poisson_distorted.py.in
#   ${CMAKE_BINARY_DIR}/scripts/poisson_distorted.py
#   )

set(sourcefiles
  # mlaplace.cc # TODO
  # linelasticity.cc # TODO
  poisson.cc
  poisson_distorted.cc
  poisson_mpiscaling.cc
  )

foreach(src ${sourcefiles})
  # string replace: cut off .cc from file src  
  string(REPLACE ".cc" "" target ${src})
  add_executable(${target} ${src})
  # include headers constructed at compile time
  target_include_directories(${target} PUBLIC ${CMAKE_BINARY_DIR}/include)
  target_link_libraries(${target} tpss)
  DEAL_II_SETUP_TARGET(${target})
endforeach(src)

################################################################################

enable_testing()

set(testfiles
  # test_coloring.cc # TODO
  )

add_custom_target(setup_tests_apps)

foreach(test ${testfiles})
  string(REPLACE ".cc" "" target ${test})
  add_executable(${target} EXCLUDE_FROM_ALL ${test})
  # include headers constructed at compile time
  target_include_directories(${target} PUBLIC ${CMAKE_BINARY_DIR}/include)
  target_link_libraries(${target} tpss)
  DEAL_II_SETUP_TARGET(${target})
  add_test(NAME ${target} COMMAND ${target})
  add_dependencies(setup_tests_apps ${target})
endforeach(test)

