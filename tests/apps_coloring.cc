
/*
 * Implement MPI-parallel multiplicative Schwarz smoothers
 *
 *  Created on: Sep 06, 2019
 *      Author: witte
 */

#include "ct_parameter.h"
#include "laplace_problem.h"

using namespace dealii;
using namespace Laplace;

struct TestParameter
{
  unsigned                   dof_limit_min        = 1;
  unsigned                   dof_limit_max        = 1000;
  double                     cg_reduction         = 1.e-8;
  unsigned                   n_smoothing_steps    = 1;
  double                     damping_factor       = 1.;
  Parameter::GeometryVariant geometry_variant     = Parameter::GeometryVariant::Cube;
  bool                       manual_coloring      = true;
  unsigned                   n_refinements_coarse = 0;
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const TestParameter & prm = TestParameter{})
{
  using LaplaceOperator =
    typename Laplace::MatrixOperator<dim, fe_degree, value_type, /*n_patch_dofs_per_direction*/ -1>;

  Parameter parameters;
  // *** PDE
  parameters.geometry_variant = prm.geometry_variant;

  // *** DISCRETIZATION
  parameters.n_cycles      = 15;
  parameters.n_refines     = prm.n_refinements_coarse + 1;
  parameters.dof_limit_min = prm.dof_limit_min;
  parameters.dof_limit_max = prm.dof_limit_max;

  // *** MULTIGRID
  parameters.coarse_level                                    = prm.n_refinements_coarse;
  parameters.schwarz_smoother_data.patch_variant             = CT::PATCH_VARIANT_;
  parameters.schwarz_smoother_data.smoother_variant          = CT::SMOOTHER_VARIANT_;
  parameters.schwarz_smoother_data.manual_coloring           = prm.manual_coloring;
  parameters.schwarz_smoother_data.number_of_smoothing_steps = prm.n_smoothing_steps;
  parameters.compute_damping_factor                          = false;
  parameters.schwarz_smoother_data.damping_factor            = prm.damping_factor;
  parameters.mg_smoother_post_reversed                       = true;
  // parameters.mg_coarse_iterations = 100;
  parameters.mg_coarse_grid_variant =
    Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
  parameters.mg_coarse_chebyshev_reduction = 1.e-8;

  // *** SOLVER
  parameters.solver_reduction      = prm.cg_reduction;
  parameters.solver_max_iterations = 100;
  parameters.precondition_variant  = Parameter::PreconditionVariant::MG;

  // *** POSTPROCESS
  parameters.compute_errors                      = true;
  parameters.schwarz_smoother_data.print_details = true;
  // parameters.write_fe_output = true;

  // *** configure output filenames
  std::ostringstream oss;
  oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
  oss << "_" << dim << "D"
      << "_" << fe_degree << "deg";
  oss << "_" << prm.n_smoothing_steps << "steps";
  oss << std::scientific << std::setprecision(3);
  oss << "_" << prm.damping_factor << "ldamp";
  const std::string filename = oss.str();

  std::fstream fstream_log;
  fstream_log.open(filename + ".log", std::ios_base::out); // std::ios_base::app);
  ConditionalOStream pcout(fstream_log, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  LaplaceOperator    laplace_problem(pcout, parameters, false);
  laplace_problem.run(true);
  fstream_log.close();

  const auto       pp_data = laplace_problem.pp_data;
  ConvergenceTable convergence_table;

  for(const auto item : pp_data.n_mg_levels)
    convergence_table.add_value("n_levels", item);
  for(const auto item : pp_data.n_cells_global)
    convergence_table.add_value("n_cells", item);
  for(const auto item : pp_data.n_dofs_global)
    convergence_table.add_value("n_dofs", item);
  for(const auto item : pp_data.n_colors_system)
    convergence_table.add_value("n_colors", item);
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
    fstream_pp.open(filename + ".txt", std::ios_base::out);
    print_row_variable(fstream_pp, 50, "total damping factor:", prm.damping_factor);
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
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;
  // TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);

  TestParameter test_prms;
  test_prms.damping_factor =
    TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, CT::DIMENSION_);
  if(argc > 1)
    test_prms.n_smoothing_steps = std::atoi(argv[1]);
  if(argc > 2)
    test_prms.damping_factor = std::atof(argv[2]);
  // if(argc > 3)
  //   test_prms.geometry_variant = Parameter::GeometryVariant(std::atoi(argv[3]));

  test<dim, fe_degree, value_type>(test_prms);

  return 0;
}
