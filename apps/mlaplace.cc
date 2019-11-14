
/*
 * Multi - Laplace
 *
 *  Created on: Apr 30, 2019
 *      Author: witte
 */

#include "Plaplace_fdss.h"
#include "ct_parameter.h"
#include "mlaplace_problem.h"

using namespace dealii;
using namespace MultiLaplace;



int
main(int argc, char * argv[])
{
  deallog.depth_console(3);

  // *** init TBB and MPI
  const unsigned int max_threads = 1; // serial
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static TEST parameters
  using value_type                              = double;
  constexpr unsigned dim                        = CT::DIMENSION_;
  constexpr unsigned fe_degree                  = CT::FE_DEGREE_;
  constexpr int      n_patch_dofs_per_direction = -1;
  // TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  Laplace::Parameter parameters;
  parameters.coarse_level                           = 1;
  parameters.n_refines                              = parameters.coarse_level + 1;
  parameters.n_cycles                               = 3;
  parameters.schwarz_smoother_data.patch_variant    = CT::PATCH_VARIANT_;
  parameters.schwarz_smoother_data.smoother_variant = CT::SMOOTHER_VARIANT_;
  parameters.solver_max_iterations                  = 200;
  parameters.solver_variant       = GlobParam::lookup_solver_variant_impl(CT::SMOOTHER_VARIANT_);
  parameters.precondition_variant = Laplace::Parameter::PreconditionVariant::MG;

  using MultiLaplaceOperator = typename MultiLaplace::
    MatrixOperator<dim, fe_degree, dim, value_type, n_patch_dofs_per_direction>;
  MultiLaplaceOperator mlaplace_problem(pcout, parameters);
  mlaplace_problem.run(true);

  return 0;
}
