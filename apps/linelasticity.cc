
/*
 * Linear Elasticiity
 *
 *  Created on: Jun 25, 2019
 *      Author: witte
 */

#include "ct_parameter.h"
#include "linelasticity_problem.h"

using namespace dealii;
using namespace LinElasticity;



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
  constexpr unsigned dim                        = 2; // CT::DIMENSION_;
  constexpr unsigned fe_degree                  = CT::FE_DEGREE_;
  constexpr int      n_patch_dofs_per_direction = -1;
  // TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  Laplace::Parameter parameters;

  // *** PDE
  EquationData equation_data;
  equation_data.mu      = 1.;
  equation_data.lambda  = 1.;
  auto & n_subdivisions = parameters.n_subdivisions;
  n_subdivisions.resize(dim);
  std::fill(n_subdivisions.begin(), n_subdivisions.end(), 1);
  n_subdivisions[0]           = 2;
  parameters.geometry_variant = Laplace::Parameter::GeometryVariant::Cube; // SubdividedCubeoid;

  // *** DISCRETIZATION
  parameters.n_refines            = 2; // parameters.coarse_level + 1;
  parameters.allow_one_level_only = true;
  parameters.n_cycles             = 2; // std::max<int>(9 - fe_degree, 5);

  // *** SOLVER
  parameters.solver_max_iterations = 500;
  parameters.precondition_variant  = Laplace::Parameter::PreconditionVariant::MG;

  // *** MULTIGRID
  parameters.coarse_level                           = 1;
  parameters.schwarz_smoother_data.patch_variant    = CT::PATCH_VARIANT_;
  parameters.schwarz_smoother_data.smoother_variant = CT::SMOOTHER_VARIANT_;
  parameters.schwarz_smoother_data.damping_factor =
    TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, dim);
  parameters.mg_smoother_post_reversed = true;
  // parameters.mg_coarse_iterations = 100;
  // parameters.mg_coarse_grid_variant =
  //   Laplace::Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
  // parameters.mg_coarse_chebyshev_reduction = 1.e-6;

  // *** POSTPROCESS
  // parameters.write_fe_output = true;

  using LinElasticityOperator =
    typename LinElasticity::MatrixOperator<dim, fe_degree, value_type, n_patch_dofs_per_direction>;
  LinElasticityOperator linelasticity_problem(pcout, parameters, equation_data);
  // linelasticity_problem.assemble_matrix();
  linelasticity_problem.run(true);

  // // *** check analytical solution and manufactured force
  // Point<2> p11(1., 1.), paniso(0.2, 0.8), phalfhalf(0.5, 0.5);
  // pcout << "x: " << paniso << std::endl;
  // pcout << "u(x): " << linelasticity_problem.analytical_solution.value(paniso) << std::endl;
  // pcout << "f(x): " << linelasticity_problem.volume_force.value(paniso) << std::endl;
  // pcout << "x: " << phalfhalf << std::endl;
  // pcout << "u(x): " << linelasticity_problem.analytical_solution.value(phalfhalf) << std::endl;
  // pcout << "f(x): " << linelasticity_problem.volume_force.value(phalfhalf) << std::endl;
  // pcout << "x: " << p11 << std::endl;
  // pcout << "u(x): " << linelasticity_problem.analytical_solution.value(p11) << std::endl;
  // pcout << "f(x): " << linelasticity_problem.volume_force.value(p11) << std::endl;

  // Laplace::MatrixOperator<dim, fe_degree> laplace_problem(pcout, parameters);
  // laplace_problem.run();
  // laplace_problem.assemble_matrix();
  // laplace_problem.sparse_matrix.print_formatted(std::cout);

  return 0;
}
