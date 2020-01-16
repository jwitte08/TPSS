project(Apps
  LANGUAGES CXX
  DESCRIPTION "Numerical applications and model problems using fast tensor product Schwarz smoothers (TPSS)"
  )

message(STATUS "Configuring numerical applications within subdirectory 'apps'")

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/linelasticity_sipg.py.in
  ${CMAKE_BINARY_DIR}/scripts/linelasticity_sipg.py
  )

set(sourcefiles
  # mlaplace.cc # TODO
  # linelasticity.cc # TODO
  linelasticity_diagonly.cc
  linelasticity_sipg.cc
  poisson.cc
  poisson_distorted.cc
  poisson_mpiscaling.cc
  )

foreach(src ${sourcefiles})
  # string replace: cut off .cc from file src  
  string(REPLACE ".cc" "" target ${src})
  add_executable(${target} ${src})
  # include headers constructed at compile time
  target_include_directories(${target} PUBLIC
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    )
  target_link_libraries(${target} tpss)
  target_compile_features(${target} PUBLIC cxx_std_17)
  DEAL_II_SETUP_TARGET(${target})
endforeach(src)
