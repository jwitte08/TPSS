
/*
 * Numerical Experiments: Fast Diagonal Schwarz Smoothers (Poisson)
 *
 *  Created on: Aug 15, 2018
 *      Author: witte
 */

#ifndef PLAPLACE_FDSS_H
#define PLAPLACE_FDSS_H

#include <deal.II/base/revision.h>
#include <deal.II/base/utilities.h>

#include <cxxabi.h>

#include <ct_parameter.h>
#include "laplace_problem.h"

#include <git_version.h>

namespace GlobParam
{
enum class TestVariant
{
  NoTest,
  VmultTime,
  SmoothTime,
  SmoothTimeComplexity,
  SmoothTimeStrong,
  MGTimeSameDoFs,
  MGTime,
  Robust,
  SmoothingRange,
  LocalDamping
};

using value_type = double;
// 0: none, 1: partition_partition, 2: partition_color, 3: color
constexpr unsigned int mf_tasks_scheme_id = 0;

constexpr auto test_variant         = TestVariant::Robust;
constexpr auto patch_variant        = CT::PATCH_VARIANT_;    // TPSS::PatchVariant::cell;
constexpr auto smoother_variant     = CT::SMOOTHER_VARIANT_; // TPSS::SmootherVariant::additive;
constexpr bool overlapping          = true;
constexpr auto geometry_variant     = Laplace::Parameter::GeometryVariant::Cube; // CubeDist//Cube
constexpr bool symmetrize_smoothing = false;

constexpr unsigned n_dofs_minimal   = 1e+6;               // minimal amount of DoFs
constexpr unsigned n_dofs_maximal   = 3 * n_dofs_minimal; // maximal amount of DoFs
constexpr unsigned n_reps_allowed[] = {1, 3, 5};          // ,7
constexpr unsigned n_loops          = 1;                  // SmoothTime:50 MGTime:25
constexpr unsigned n_loops_warmup   = 0;

constexpr bool    use_filestream   = true;
const std::string output_directory = CT::LOG_DIR_;


using namespace dealii;
using namespace Laplace;

// ++++++++++++++++++++++++++++++   formatted printing   ++++++++++++++++++++++++++++++

void
print_generic_header(ConditionalOStream & cout, const MPI_Comm & mpi_comm)
{
  print_row_variable(cout, 45, "Git - deal.II version: ", DEAL_II_GIT_SHORTREV);
  print_row_variable(cout, 45, "Git - deal.II branch: ", DEAL_II_GIT_BRANCH);
  print_row_variable(cout, 45, "Git - ExaDG version: ", GIT_COMMIT_HASH);
  print_row_variable(cout, 45, "Git - ExaDG branch: ", GIT_BRANCH);
  cout << std::endl;

  int    status;
  char * demangled_name = abi::__cxa_demangle(typeid(value_type).name(), 0, 0, &status);
  print_row_variable(cout, 45, "Floating point precision:", demangled_name);
  constexpr auto n_lanes = VectorizedArray<value_type>::n_array_elements;
  print_row_variable(cout, 45, "Vectorization length:", n_lanes);
  print_row_variable(cout, 45, "MultithreadInfo::n_cores:", MultithreadInfo::n_cores());
  print_row_variable(cout, 45, "MultithreadInfo::n_threads:", MultithreadInfo::n_threads());
  const auto n_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);
  print_row_variable(cout, 45, "MPI - number of mpi processes:", n_mpi_processes);
  print_row_variable(cout, 45, "Number of experimental loops:", n_loops);
  print_row_variable(cout, 45, "Number of warmup loops:", n_loops_warmup);

  cout << std::endl;
}

std::string
str_append_fname(const std::string prefix = "", const std::string suffix = "")
{
  std::ostringstream ostrstream;
  ostrstream << "/" + prefix;
  ostrstream << TPSS::getstr_schwarz_variant(GlobParam::patch_variant,
                                             GlobParam::smoother_variant,
                                             GlobParam::overlapping);
  ostrstream << "_DIM" << CT::DIMENSION_;
  ostrstream << "_DEG" << CT::FE_DEGREE_;
  ostrstream << suffix;
  return ostrstream.str();
}

void
fprint_time_info(ConditionalOStream & cout, const TimeInfo & time_info)
{
  std::ostringstream ostrstream;
  ostrstream << std::scientific << std::setprecision(3) << time_info.time << " " << time_info.unit;
  auto && ftime = ostrstream.str();
  print_row_variable(cout, 2, "T", 43, time_info.description, ftime);
}

template<int dim, typename Number>
void
fprint_misc(ConditionalOStream &                                       cout,
            const std::shared_ptr<const MatrixFree<dim, Number>>       mf_storage,
            const std::shared_ptr<const SubdomainHandler<dim, Number>> patch_storage)
{
  const auto &   subdomain_data      = patch_storage->get_additional_data();
  const auto     level               = subdomain_data.level;
  const auto &   dof_handler         = mf_storage->get_dof_handler();
  const unsigned n_dofs              = dof_handler.n_dofs(level);
  const unsigned n_cells             = dof_handler.get_triangulation().n_cells(level);
  const unsigned n_macro_cells       = mf_storage->n_cell_batches();
  const unsigned n_inner_macro_faces = mf_storage->n_inner_face_batches();
  const unsigned n_bdry_macro_faces  = mf_storage->n_boundary_face_batches();
  const unsigned n_subdomains        = patch_storage->n_physical_subdomains();
  const unsigned n_macro_subdomains  = patch_storage->get_partition_data().n_subdomains();

  cout << "Miscellaneous information on level " << level << ":" << std::endl;
  print_row_variable(cout, 2, "", 43, "Number of degrees of freedom:", n_dofs);
  print_row_variable(cout, 2, "", 43, "Number of cells:", n_cells);
  print_row_variable(cout, 2, "", 43, "Number of macro cells:", n_macro_cells);
  print_row_variable(cout, 2, "", 43, "Number of interior macro faces:", n_inner_macro_faces);
  print_row_variable(cout, 2, "", 43, "Number of boundary macro faces:", n_bdry_macro_faces);
  print_row_variable(cout, 2, "", 43, "Number of physical subdomains:", n_subdomains);
  print_row_variable(cout, 2, "", 43, "Number of macro subdomains:", n_macro_subdomains);
  cout << std::endl;
}


// ++++++++++++++++++++++++++++++   look-up functions   ++++++++++++++++++++++++++++++

/**
 * computes the level and the number of repetitions of cells (n_rep)
 * on the coarse mesh which are required to obtain a DoF count thats
 * within the demanded range. consequently, the coarse mesh has
 * n_rep^dim cells.
 */
std::pair<unsigned, unsigned>
lookup_refine_pair(const int      dim,
                   const int      fe_degree,
                   const unsigned min_level     = 1,
                   const bool     print_details = false)
{
  std::pair<unsigned, unsigned> refine_pair{/*level*/ 0, /*n_rep*/ 0};
  const auto                    dofs_per_cell = Utilities::pow(fe_degree + 1, dim);
  unsigned                      n_dofs_min    = static_cast<unsigned>(-1);

  for(unsigned n_cells : n_reps_allowed)
  {
    unsigned n_coarse_cells = Utilities::pow(n_cells, dim);
    unsigned level          = 1;
    unsigned n_dofs =
      n_coarse_cells * dofs_per_cell * Utilities::pow(Utilities::pow(2, dim), level);
    while(n_dofs < n_dofs_minimal)
    {
      ++level;
      n_dofs = n_coarse_cells * dofs_per_cell * Utilities::pow(Utilities::pow(2, dim), level);
    }

    if(min_level <= level && n_dofs < n_dofs_min && n_dofs < n_dofs_maximal)
    {
      n_dofs_min  = n_dofs;
      refine_pair = std::make_pair(level, n_cells);
    }
  }

  if(print_details)
  {
    std::cout << "n_dofs_min for degree " << fe_degree << ": " << n_dofs_min << std::endl;
    std::cout << "n_refines:" << refine_pair.first << std::endl
              << "n_cell_repetitions:" << refine_pair.second << std::endl;
  }

  return refine_pair;
}

Parameter::SolverVariant
lookup_solver_variant_impl(const TPSS::SmootherVariant smoother_variant,
                           const bool                  symmetrize_smoothing = false)
{
  if(smoother_variant == TPSS::SmootherVariant::multiplicative)
    if(!symmetrize_smoothing)
      return Parameter::SolverVariant::GMRES;
  return Parameter::SolverVariant::CG;
}

Parameter::SolverVariant
lookup_solver_variant()
{
  // if(smoother_variant == TPSS::SmootherVariant::multiplicative)
  //   if(!symmetrize_smoothing)
  //     return Parameter::SolverVariant::GMRES;
  // return Parameter::SolverVariant::CG;
  return lookup_solver_variant_impl(CT::SMOOTHER_VARIANT_, symmetrize_smoothing);
}

double
lookup_chebyshev_reduction()
{
  double reduction{1.e-4};
  if(geometry_variant == Parameter::GeometryVariant::Cube)
    reduction = 1.e-7;
  if(geometry_variant == Parameter::GeometryVariant::Ball)
    reduction = 1.e-6;
  return reduction;
}

double
lookup_smoothing_range(const unsigned degree)
{
  double range{6.};
  if(geometry_variant == Parameter::GeometryVariant::Cube)
  {
    if(smoother_variant == TPSS::SmootherVariant::multiplicative)
      range = 4.;
  }
  else // if (geometry_variant == Parameter::GeometryVariant::CubeDistorted)
  {
    if(degree == 1)
      range = 6.;
    else if(degree == 2)
      range = 12.;
    else if(degree == 4)
      range = 18.;
  }
  return range;
}

double
lookup_local_damping()
{
  double factor{1.};
  if(geometry_variant == Parameter::GeometryVariant::CubeDistorted)
    factor = CT::DIMENSION_ == 3 ? 0.7 : 0.8;
  else if(geometry_variant == Parameter::GeometryVariant::Ball)
    factor = 0.8;
  return factor;
}

} // namespace GlobParam

#endif
