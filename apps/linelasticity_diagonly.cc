
/*
 * Test Schwarz Smoothers based on the block diagonal of the
 * elasticity system.
 *
 *  Created on: Jul 31, 2019
 *      Author: witte
 */

#include "ct_parameter.h"
#include "linelasticity_problem.h"

using namespace dealii;
using namespace LinElasticity;

struct TestParameter
{
  TPSS::PatchVariant    patch_variant    = CT::PATCH_VARIANT_;
  TPSS::SmootherVariant smoother_variant = CT::SMOOTHER_VARIANT_;

  double                             cg_reduction         = 1.e-8;
  double                             coarse_grid_accuracy = 1.e-8;
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::IterativeAcc;
  types::global_dof_index dof_limit_min = 1e4;
  types::global_dof_index dof_limit_max = 1e6;
  EquationData            equation_data;
  double                  local_damping_factor = 1.;
  unsigned                n_smoothing_steps    = 2;
  std::string             solver_variant       = "cg";
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const TestParameter & prms = TestParameter{})
{
  using LinElasticityOperator = typename LinElasticity::
    ModelProblem<dim, fe_degree, value_type, /*n_patch_dofs_per_direction*/ -1>;

  RT::Parameter rt_parameters;
  //: discretization
  rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  rt_parameters.mesh.n_refinements    = 1;
  rt_parameters.mesh.n_repetitions    = 2;

  //: solver
  rt_parameters.solver.variant              = prms.solver_variant;
  rt_parameters.solver.rel_tolerance        = prms.cg_reduction;
  rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;

  //: multigrid
  const double damping_factor =
    TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
  rt_parameters.multigrid.coarse_level                 = 0;
  rt_parameters.multigrid.coarse_grid.solver_variant   = prms.coarse_grid_variant;
  rt_parameters.multigrid.coarse_grid.iterative_solver = prms.solver_variant;
  rt_parameters.multigrid.coarse_grid.accuracy         = prms.coarse_grid_accuracy;
  rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
  rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = prms.patch_variant;
  rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = prms.smoother_variant;
  rt_parameters.multigrid.pre_smoother.schwarz.manual_coloring      = true;
  rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
  rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = prms.n_smoothing_steps;
  rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(6, fe_degree + 1);
  rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
  rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

  //: misc
  rt_parameters.dof_limits = {prms.dof_limit_min, prms.dof_limit_max};
  rt_parameters.n_cycles   = 10;

  Laplace::Parameter parameters;
  // *** PDE
  const EquationData & equation_data = prms.equation_data;
  parameters.geometry_variant        = Laplace::Parameter::GeometryVariant::Cube;

  // *** DISCRETIZATION
  parameters.dof_limit_min        = prms.dof_limit_min;
  parameters.dof_limit_max        = prms.dof_limit_max;
  parameters.n_refines            = 2;
  parameters.allow_one_level_only = false;
  parameters.n_cycles             = 20;

  // *** SOLVER
  parameters.solver_reduction      = prms.cg_reduction;
  parameters.solver_max_iterations = 100;
  // parameters.solver_variant        =
  parameters.precondition_variant = Laplace::Parameter::PreconditionVariant::MG;

  // *** MULTIGRID
  parameters.coarse_level                                    = 1;
  parameters.schwarz_smoother_data.patch_variant             = CT::PATCH_VARIANT_;
  parameters.schwarz_smoother_data.smoother_variant          = CT::SMOOTHER_VARIANT_;
  parameters.schwarz_smoother_data.number_of_smoothing_steps = prms.n_smoothing_steps;
  parameters.schwarz_smoother_data.damping_factor =
    TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, dim);
  parameters.schwarz_smoother_data.local_damping_factor = prms.local_damping_factor;
  parameters.mg_smoother_post_reversed                  = true;
  // parameters.mg_coarse_iterations = 100;
  // parameters.mg_coarse_grid_variant =
  //   Laplace::Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
  // parameters.mg_coarse_chebyshev_reduction = 1.e-6;

  // *** POSTPROCESS
  // parameters.write_fe_output = true;

  // *** configure output filenames
  std::ostringstream oss;
  oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_) + "_";
  oss << dim << "D_" << fe_degree << "deg_";
  oss << prms.n_smoothing_steps << "steps_";
  oss << std::scientific << std::setprecision(3);
  oss << prms.local_damping_factor << "ldamp_";
  oss << prms.equation_data.mu << "mu_";
  oss << prms.equation_data.lambda << "lambda";
  const std::string filename = oss.str();

  std::fstream fstream_log;
  fstream_log.open(filename + ".log", std::ios_base::app);
  ConditionalOStream    pcout(fstream_log, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  LinElasticityOperator linelasticity_problem(pcout, parameters, rt_parameters, equation_data);
  linelasticity_problem.run(true);
  fstream_log.close();

  const auto       pp_data = linelasticity_problem.pp_data;
  ConvergenceTable convergence_table;

  for(const auto item : pp_data.n_mg_levels)
    convergence_table.add_value("n_levels", item);
  for(const auto item : pp_data.n_cells_global)
    convergence_table.add_value("n_cells", item);
  for(const auto item : pp_data.n_dofs_global)
    convergence_table.add_value("n_dofs", item);
  for(const auto item : pp_data.L2_error)
    convergence_table.add_value("error_L2", item);
  for(const auto item : pp_data.n_iterations_system)
    convergence_table.add_value("n_iter", item);
  for(const auto item : pp_data.average_reduction_system)
    convergence_table.add_value("reduction", item);
  for(const auto item : pp_data.solve_time)
    convergence_table.add_value("t_solve", item);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    convergence_table.set_scientific("error_L2", true);
    convergence_table.set_precision("error_L2", 3);
    convergence_table.evaluate_convergence_rates("error_L2",
                                                 "n_cells",
                                                 ConvergenceTable::reduction_rate_log2,
                                                 dim);
    convergence_table.set_scientific("reduction", true);
    convergence_table.set_precision("reduction", 3);
    convergence_table.set_scientific("t_solve", true);
    convergence_table.set_precision("t_solve", 3);

    std::fstream fstream_pp;
    fstream_pp.open(filename + ".txt", std::ios_base::app);
    convergence_table.write_text(fstream_pp);
    fstream_pp.close();

    std::cout << "written to: " << filename << std::endl << std::endl;
  }
}

int
main(int argc, char * argv[])
{
  deallog.depth_console(3);

  // *** init TBB and MPI
  constexpr unsigned int max_threads = 1; // serial
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static TEST parameters
  using value_type             = double;
  constexpr unsigned dim       = 2; // CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;
  // constexpr int      n_patch_dofs_per_direction = -1;
  // TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);

  TestParameter test_prms;
  if(argc > 1)
    test_prms.n_smoothing_steps = std::atoi(argv[1]);
  if(argc > 2)
    test_prms.local_damping_factor = std::atof(argv[2]);
  if(argc > 3)
    test_prms.equation_data.mu = std::atof(argv[3]);
  if(argc > 4)
    test_prms.equation_data.lambda = std::atof(argv[4]);

  test<dim, fe_degree, value_type>(test_prms);

  return 0;
}
