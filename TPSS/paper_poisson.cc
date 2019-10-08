
/*
 * Test Schwarz Smoothers based on the block diagonal of the
 * elasticity system.
 *
 *  Created on: Jul 31, 2019
 *      Author: witte
 */

#include "Plaplace_fdss.h"
#include "ct_parameter.h"
#include "laplace_problem.h"

using namespace dealii;
using namespace Laplace;

struct TestParameter
{
  unsigned                   dof_limit_min         = 1e4;
  unsigned                   dof_limit_max         = 1e6;
  double                     cg_reduction          = 1.e-8;
  unsigned                   n_smoothing_steps     = 1;
  double                     damping_factor        = 1.;
  Parameter::GeometryVariant geometry_variant      = Parameter::GeometryVariant::Cube;
  unsigned                   n_refinements_distort = (CT::DIMENSION_ == 2) ? 5 : 3;
  unsigned                   n_repetitions         = (CT::DIMENSION_ == 2) ? 32 : 8;
  double                     distortion_factor     = 0.25;
  unsigned                   n_mg_levels_distort   = 2;
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const TestParameter & prm = TestParameter{})
{
  using LaplaceOperator =
    typename Laplace::MatrixOperator<dim, fe_degree, value_type, /*n_patch_dofs_per_direction*/ -1>;

  Parameter parameters;
  // *** PDE
  parameters.geometry_variant = Parameter::GeometryVariant::Cube;

  // *** DISCRETIZATION
  const bool mesh_is_cartesian = (prm.geometry_variant == Parameter::GeometryVariant::Cube);
  const bool mesh_is_distorted =
    (prm.geometry_variant == Parameter::GeometryVariant::CubeDistorted);
  const bool mesh_is_circular   = (prm.geometry_variant == Parameter::GeometryVariant::Ball);
  IP::pre_factor                = !mesh_is_cartesian ? 4. : 1.;
  parameters.n_cycles           = mesh_is_distorted ? 1 : 15;
  parameters.n_refines          = mesh_is_distorted ? (prm.n_mg_levels_distort - 1) : 2;
  parameters.geometry_variant   = prm.geometry_variant;
  parameters.n_cell_repetitions = prm.n_repetitions;
  parameters.n_refines_distort  = mesh_is_distorted ? prm.n_refinements_distort : 0;
  parameters.distortion_factor  = prm.distortion_factor;
  parameters.dof_limit_min      = mesh_is_distorted ? 1 : prm.dof_limit_min;
  parameters.dof_limit_max      = mesh_is_distorted ? 1e10 : prm.dof_limit_max;

  // *** MULTIGRID
  const double outer_damping_factor =
    TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, dim);
  const double local_damping_factor = prm.damping_factor / outer_damping_factor;

  parameters.coarse_level                                    = mesh_is_distorted ? 0 : 1; // TODO distorted?
  parameters.schwarz_smoother_data.patch_variant             = CT::PATCH_VARIANT_;
  parameters.schwarz_smoother_data.smoother_variant          = CT::SMOOTHER_VARIANT_;
  parameters.schwarz_smoother_data.manual_coloring           = true;
  parameters.schwarz_smoother_data.number_of_smoothing_steps = prm.n_smoothing_steps;
  parameters.compute_damping_factor                          = false;
  parameters.schwarz_smoother_data.damping_factor            = outer_damping_factor;
  parameters.schwarz_smoother_data.local_damping_factor      = local_damping_factor;
  parameters.mg_smoother_post_reversed                       = true;
  // parameters.mg_coarse_iterations = 100;
  parameters.mg_coarse_grid_variant =
    Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
  parameters.mg_coarse_chebyshev_reduction = 1.e-8;

  // *** SOLVER>
  parameters.solver_reduction      = prm.cg_reduction;
  parameters.solver_max_iterations = 100;
  if (CT::PATCH_VARIANT_ == TPSS::PatchVariant::vertex)
    parameters.solver_max_iterations = 50;
  // parameters.solver_variant = Parameter::SolverVariant::GMRES;
  if(!parameters.mg_smoother_post_reversed)
    parameters.solver_variant = GlobParam::lookup_solver_variant_impl(CT::SMOOTHER_VARIANT_);
  parameters.precondition_variant = Parameter::PreconditionVariant::MG;

  // *** POSTPROCESS
  parameters.compute_errors = true;
  // parameters.write_fe_output = true;

  // *** configure output filenames
  std::ostringstream oss;
  oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
  oss << "_" << dim << "D"
      << "_" << fe_degree << "deg";
  oss << std::scientific << std::setprecision(3);
  oss << "_" << prm.distortion_factor << "dist";
  oss << "_" << prm.n_mg_levels_distort << "lvls";
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
    print_row_variable(fstream_pp, 50, "outer damping factor:", outer_damping_factor);
    print_row_variable(fstream_pp, 50, "local damping factor:", local_damping_factor);
    print_row_variable(fstream_pp, 50, "total damping factor:", prm.damping_factor);
    print_row_variable(fstream_pp, 50, "Distortion factor:", prm.distortion_factor);
    fstream_pp << std::endl;
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
  if(argc > 3)
    test_prms.geometry_variant = Parameter::GeometryVariant(std::atoi(argv[3]));
  if(argc > 4)
    test_prms.n_mg_levels_distort = std::atoi(argv[4]);
  if(argc > 5)
    test_prms.distortion_factor = std::atof(argv[5]);

  test<dim, fe_degree, value_type>(test_prms);

  return 0;
}
