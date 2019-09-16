
/*
 * test poisson problem
 *
 *  Created on: Sep 12, 2019
 *      Author: witte
 */

#include "poisson.h"
#include "ct_parameter.h"
//#include "laplace_problem.h"

using namespace dealii;
using namespace Laplace;

struct TestParameter
{
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::multiplicative;
  double                cg_reduction     = 1.e-8;
  unsigned              n_refinements    = 1;
  unsigned              n_cycles         = 1;
};

template<int dim, int fe_degree>
void
test(const TestParameter & prm = TestParameter{})
{
  using PoissonProblem = typename Poisson::ModelProblem<dim, fe_degree>;
  Laplace::Parameter parameters;
  //: discretization
  parameters.n_cycles  = prm.n_cycles;
  parameters.n_refines = prm.n_refinements;
  //: iterative solver
  parameters.solver_reduction      = prm.cg_reduction;
  parameters.solver_max_iterations = 200;
  parameters.precondition_variant  = Parameter::PreconditionVariant::MG;
  //: multigrid
  parameters.coarse_level                           = 0;
  parameters.schwarz_smoother_data.patch_variant    = prm.patch_variant;
  parameters.schwarz_smoother_data.smoother_variant = prm.smoother_variant;
  parameters.schwarz_smoother_data.manual_coloring  = true;
  //   parameters.schwarz_smoother_data.number_of_smoothing_steps = prm.n_smoothing_steps;
  //   parameters.compute_damping_factor                          = false;
  //   parameters.schwarz_smoother_data.damping_factor            = outer_damping_factor;
  //   parameters.schwarz_smoother_data.local_damping_factor      = local_damping_factor;
  //   parameters.mg_smoother_post_reversed                       = true;
  //   // parameters.mg_coarse_iterations = 100;
  //   parameters.mg_coarse_grid_variant =
  //     Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
  //   parameters.mg_coarse_chebyshev_reduction = 1.e-8;
  //   //: post process
  parameters.sync_timings = true;
  //   parameters.compute_errors = true;
  //   // parameters.write_fe_output = true;

  PoissonProblem poisson_problem{parameters};
  poisson_problem.create_triangulation(parameters.n_refines);
  poisson_problem.distribute_dofs();
  poisson_problem.prepare_linear_system();
  const unsigned level             = poisson_problem.level;
  const auto     mf_storage        = poisson_problem.system_matrix.get_matrix_free();
  const auto     subdomain_handler = poisson_problem.build_patch_storage(level, mf_storage);
  poisson_problem.run();

  //   // *** configure output filenames
  //   std::ostringstream oss;
  //   oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
  //   oss << "_" << dim << "D"
  //       << "_" << fe_degree << "deg";
  //   oss << "_" << prm.n_smoothing_steps << "steps";
  //   oss << std::scientific << std::setprecision(3);
  //   oss << "_" << prm.damping_factor << "ldamp";
  //   const std::string filename = oss.str();

  //   std::fstream fstream_log;
  //   fstream_log.open(filename + ".log", std::ios_base::out); // std::ios_base::app);
  //   ConditionalOStream pcout(fstream_log, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  //   LaplaceOperator    laplace_problem(pcout, parameters, false);
  //   laplace_problem.run(true);
  //   fstream_log.close();

  const auto       pp_data = poisson_problem.pp_data;
  ConvergenceTable convergence_table;
  //   for(const auto item : pp_data.n_mg_levels)
  //     convergence_table.add_value("n_levels", item);
  for(const auto item : pp_data.n_cells_global)
    convergence_table.add_value("n_cells", item);
  for(const auto item : pp_data.n_dofs_global)
    convergence_table.add_value("n_dofs", item);
  // for(const auto item : pp_data.n_colors_system)
  //   convergence_table.add_value("n_colors", item);
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

    // std::fstream fstream_pp;
    // fstream_pp.open(filename + ".txt", std::ios_base::out);
    // print_row_variable(fstream_pp, 50, "outer damping factor:", outer_damping_factor);
    // print_row_variable(fstream_pp, 50, "local damping factor:", local_damping_factor);
    // print_row_variable(fstream_pp, 50, "total damping factor:", prm.damping_factor);
    // convergence_table.write_text(fstream_pp);
    // fstream_pp.close();
    // std::cout << "written to: " << filename << std::endl << std::endl;

    convergence_table.write_text(std::cout);
  }
}

int
main(int argc, char * argv[])
{
  // deallog.depth_console(3);

  // *** init TBB and MPI
  constexpr unsigned int max_threads = 1; // no multithreading
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static parameters
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;

  // *** run-time parameters
  TestParameter prms;
  if(argc > 1)
    prms.n_refinements = std::atoi(argv[1]);
  if(argc > 2)
    prms.n_cycles = std::atoi(argv[2]);

  test<dim, fe_degree>(prms);

  return 0;
}
