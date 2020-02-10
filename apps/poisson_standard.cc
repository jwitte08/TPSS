
/*
 * sandbox poisson problem
 *
 *  Created on: Oct 30, 2019
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "ct_parameter.h"
#include "poisson_standard.h"

using namespace dealii;
using namespace Laplace;

struct TestParameter
{
  TPSS::PatchVariant                 patch_variant    = CT::PATCH_VARIANT_;
  TPSS::SmootherVariant              smoother_variant = CT::SMOOTHER_VARIANT_;
  std::string                        solver_variant   = "cg"; // see SolverSelector
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::IterativeAcc;
  double   coarse_grid_accuracy = 1.e-8;
  double   cg_reduction         = 1.e-8;
  unsigned n_refinements        = 2;
  unsigned n_repetitions        = 2;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("n_refinements:", n_refinements);
    oss << Util::parameter_to_fstring("n_repetitions:", n_repetitions);
    oss << Util::parameter_to_fstring(solver_variant + " solver reduction:", cg_reduction);
    return oss.str();
  }
};

std::string
write_ppdata_to_string(const PostProcessData & pp_data)
{
  std::ostringstream oss;
  ConvergenceTable   info_table;
  Assert(!pp_data.n_cells_global.empty(), ExcMessage("No cells to post process."));
  for(unsigned run = 0; run < pp_data.n_cells_global.size(); ++run)
  {
    // info_table.add_value("n_levels", pp_data.n_mg_levels.at(run));
    info_table.add_value("n_cells", pp_data.n_cells_global.at(run));
    info_table.add_value("n_dofs", pp_data.n_dofs_global.at(run));
    // info_table.add_value("n_colors", pp_data.n_colors_system.at(run));
    info_table.add_value("n_iter", pp_data.n_iterations_system.at(run));
    info_table.add_value("reduction", pp_data.average_reduction_system.at(run));
    info_table.add_value("L2_error", pp_data.L2_error.at(run));
  }
  info_table.set_scientific("reduction", true);
  info_table.set_precision("reduction", 3);
  info_table.set_scientific("L2_error", true);
  info_table.set_precision("L2_error", 3);
  info_table.evaluate_convergence_rates("L2_error", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("L2_error",
                                        "n_dofs",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);
  info_table.write_text(oss);
  return oss.str();
}

int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  constexpr int                    dim       = CT::DIMENSION_;
  constexpr int                    fe_degree = CT::FE_DEGREE_;
  constexpr int                    n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);
  using PoissonProblem =
    typename Poisson::Std::ModelProblem<dim, fe_degree, double, n_patch_dofs_per_direction>;

  TestParameter testprms;
  if(argc > 1)
    testprms.n_refinements = std::atoi(argv[1]);
  if(argc > 2)
    testprms.n_repetitions = std::atoi(argv[2]);
  RT::Parameter rt_parameters;

  //: discretization
  rt_parameters.n_cycles              = 3;
  rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  rt_parameters.mesh.n_refinements    = testprms.n_refinements;
  rt_parameters.mesh.n_repetitions    = testprms.n_repetitions;

  //: solver
  rt_parameters.solver.variant              = testprms.solver_variant;
  rt_parameters.solver.rel_tolerance        = testprms.cg_reduction;
  rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::None;
  rt_parameters.solver.n_iterations_max     = 500;

  // //: multigrid
  // const double damping_factor =
  //   TPSS::lookup_damping_factor(testprms.patch_variant, testprms.smoother_variant, dim);
  // rt_parameters.multigrid.coarse_level                 = 0;
  // rt_parameters.multigrid.coarse_grid.solver_variant   = testprms.coarse_grid_variant;
  // rt_parameters.multigrid.coarse_grid.iterative_solver = testprms.solver_variant;
  // rt_parameters.multigrid.coarse_grid.accuracy         = testprms.coarse_grid_accuracy;
  // rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
  // rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = testprms.patch_variant;
  // rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = testprms.smoother_variant;
  // rt_parameters.multigrid.pre_smoother.schwarz.manual_coloring      = true;
  // rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
  // rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(8, fe_degree + 1);
  // rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
  // rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

  PoissonProblem poisson_problem{rt_parameters};
  /// set reference solution with non-zero Dirichlet boundary
  poisson_problem.analytical_solution = std::make_shared<Laplace::Solution<dim>>();
  poisson_problem.load_function =
    std::make_shared<Laplace::ManufacturedLoad<dim>>(poisson_problem.analytical_solution);

  const bool is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  const auto pcout         = std::make_shared<ConditionalOStream>(std::cout, is_first_proc);
  poisson_problem.pcout    = pcout;
  *pcout << Util::generic_info_to_fstring() << std::endl;
  *pcout << testprms.to_string() << std::endl;

  poisson_problem.run();
  const auto pp_data = poisson_problem.pp_data;
  *pcout << write_ppdata_to_string(pp_data);

  return 0;
}
