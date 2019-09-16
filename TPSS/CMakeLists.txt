##
#  CMake script
##

PROJECT(TPSS)

MESSAGE("_________________________________________________________________________________")
MESSAGE("                                                                                 ")
MESSAGE("                                    TPSS                                         ")
MESSAGE("       numerical experiments and tests of tensor product Schwarz smoothers       ")
MESSAGE("_________________________________________________________________________________")

MESSAGE("${PROJECT_SOURCE_DIR}")

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  # extract the name of the current git branch
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

  # extract the abbr. hash of the last commit
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(GIT_BRANCH "")
  set(GIT_COMMIT_HASH "")
endif(EXISTS "${CMAKE_SOURCE_DIR}/.git")

message(STATUS "Git - current branch: ${GIT_BRANCH}")
message(STATUS "Git - commit hash:    ${GIT_COMMIT_HASH}")
message(STATUS "Git - store information in version.h")

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/git_version.h.in
  ${CMAKE_BINARY_DIR}/include/git_version.h
  )

#: generate python script (paper_poisson)
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/paper_poisson.py.in
  ${CMAKE_BINARY_DIR}/paper_poisson.py
  )

#: generate python script (ct_parameter)
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/ct_parameter.py.in
  ${CMAKE_BINARY_DIR}/ct_parameter.py
  )
#: find a python3 interpreter
find_package ( PythonInterp 3 REQUIRED )
#: path to the compile-time parameter file
set (CTPRM_FILE "${CMAKE_BINARY_DIR}/include/ct_parameter.h")
#: generating compile-time parameters
execute_process (
  COMMAND ${PYTHON_EXECUTABLE} ct_parameter.py -LOG ${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  )

SET( SOURCE_FILES
  # Tpatch_info_01.cc
  # Tpatch_info_mpi.cc
  # Tpatch_transfer_01.cc
  # Tpatch_transfer_02.cc
  # Ttensors_01.cc
  # Tblock_matrix_diagonal_01.cc
  Tlaplace_check_iterations.cc
  # Tlaplace_geometry.cc
  # Tlaplace_integrator_cp.cc
  # Tlaplace_integrator_mf.cc
  # Tlaplace_integrator_vp.cc
  # Tmlaplace_integrator.cc
  # Tmlaplace_check_iterations.cc
  # Tlinelasticity_integrator_mf.cc
  # Tlinelasticity_integrator_fd.cc
  # Plaplace_fdss_tbb.cc
  # Plaplace_fdss.cc
  # mlaplace.cc
  # linelasticity.cc
  # proceeding_elasticity_diagonly.cc
  paper_poisson.cc
  mpi_coloring.cc
  cell_coloring.cc
  poisson.cc
  )

foreach ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
endforeach ( sourcefile ${APP_SOURCES} )
