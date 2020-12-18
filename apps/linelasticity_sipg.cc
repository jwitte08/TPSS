
/*
 * Compare Schwarz Smoothers:
 *
 *   (0) block diagonal only
 *   (1) exact
 *   (2) fast diagonalized Schur complement
 *
 * for the 2D linear elasticity problem with (inhomogeneous?) Dirichlet
 * boundary. Symmetric interior penalty discretization of the Lame equations
 * using standard tensor product polynomials.
 *
 *  Created on: Jan 15, 2020
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
  double                             coarse_grid_accuracy = 1.e-12;
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::DirectSVD;
  types::global_dof_index dof_limit_min = 1e4;
  types::global_dof_index dof_limit_max = 1e6;
  EquationData            equation_data;
  unsigned                n_smoothing_steps = 2;
  std::string             solver_variant    = "cg"; //"gmres";
  unsigned                test_variant      = 0;
  std::string             test_description  = "";
};

template<typename ModelProblem, int dim, int fe_degree, typename value_type = double>
// std::shared_ptr<ModelProblem>
void
debug_impl(const TestParameter & prms = TestParameter{})
{
  // std::shared_ptr<ModelProblem> model_problem;
  // return linelasticity_problem;

  RT::Parameter rt_parameters;
  //: discretization
  rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  rt_parameters.mesh.n_refinements    = 1;
  rt_parameters.mesh.n_repetitions    = 2;

  //: solver
  rt_parameters.solver.variant              = prms.solver_variant;
  rt_parameters.solver.rel_tolerance        = prms.cg_reduction;
  rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
  rt_parameters.solver.n_iterations_max     = 200;

  //: multigrid
  // const double damping_factor =
  //   TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
  rt_parameters.multigrid.coarse_level                 = 0;
  rt_parameters.multigrid.coarse_grid.solver_variant   = prms.coarse_grid_variant;
  rt_parameters.multigrid.coarse_grid.iterative_solver = prms.solver_variant;
  rt_parameters.multigrid.coarse_grid.accuracy         = prms.coarse_grid_accuracy;
  rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
  rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = prms.patch_variant;
  rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = prms.smoother_variant;
  rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
  rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = prms.n_smoothing_steps;
  rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
  rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;
  rt_parameters.reset_damping_factor(dim);
  // rt_parameters.reset_solver_variant();

  // //: misc
  // rt_parameters.dof_limits = {prms.dof_limit_min, prms.dof_limit_max};
  // rt_parameters.n_cycles   = 10;

  // *** configure output filenames
  std::ostringstream oss;
  oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_) + "_";
  oss << dim << "D_" << fe_degree << "deg_";
  oss << prms.test_description << "_";
  oss << prms.n_smoothing_steps << "steps_";
  oss << std::scientific << std::setprecision(3);
  oss << prms.equation_data.mu << "mu_";
  oss << prms.equation_data.lambda << "lambda";
  if(prms.equation_data.factor != 1.)
    oss << "_" << prms.equation_data.factor << "fctr";
  if(prms.equation_data.ip_factor > 1.)
    oss << "_" << prms.equation_data.ip_factor << "ip";
  const std::string filename = oss.str();

  // std::fstream fstream_log;
  // fstream_log.open(filename + ".log", std::ios_base::app);
  // ConditionalOStream pcout(fstream_log, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << filename << std::endl;
  const EquationData & equation_data = prms.equation_data;
  ModelProblem         linelasticity_problem(pcout, rt_parameters, equation_data);
  linelasticity_problem.run(true);
  linelasticity_problem.create_triangulation();
  linelasticity_problem.distribute_dofs();
  linelasticity_problem.prepare_system();
  linelasticity_problem.prepare_preconditioner_mg();

  const auto   precond        = linelasticity_problem.mg_schwarz_smoother_pre->get_preconditioner();
  const auto   solvers_shared = precond->get_local_solvers();
  const auto & local_solvers  = *solvers_shared;
  const auto   print_eigenvalues = [](const auto & mat, const std::string name) {
    const auto & eigenvalues = compute_eigenvalues(mat);
    std::cout << "Eigenvalues of " << name << std::endl;
    std::cout << vector_to_string(eigenvalues) << std::endl;
  };
  for(auto patch = 0U; patch < local_solvers.size(); ++patch)
  {
    std::cout << "patch = " << patch << std::endl;
    const auto & solver = local_solvers[patch];
    // AlignedVector<VectorizedArray<value_type>> dst(solver.m());
    // AlignedVector<VectorizedArray<value_type>> src(solver.n());
    // ArrayView<VectorizedArray<value_type>> dst_view(dst.begin(), dst.size());
    // ArrayView<const VectorizedArray<value_type>> src_view(src.begin(), src.size());
    // solver.apply_inverse(dst_view,src_view);
    const auto & vectorized_Aj_inv = solver.as_inverse_table();
    const auto   Aj_inv            = table_to_fullmatrix(vectorized_Aj_inv, 0);

    /// eigenvalues
    print_eigenvalues(Aj_inv, "Aj^{-1}");
  }

  linelasticity_problem.solve(*(linelasticity_problem.preconditioner_mg));

  // fstream_log.close();
}

template<typename ModelProblem, int dim, int fe_degree, typename value_type = double>
void
test_impl(const TestParameter & prms = TestParameter{})
{
  RT::Parameter rt_parameters;
  //: discretization
  rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  rt_parameters.mesh.n_refinements    = 1;
  rt_parameters.mesh.n_repetitions    = 2;

  //: solver
  rt_parameters.solver.variant              = prms.solver_variant;
  rt_parameters.solver.rel_tolerance        = prms.cg_reduction;
  rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
  rt_parameters.solver.n_iterations_max     = 200;

  //: multigrid
  // const double damping_factor =
  //   TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
  rt_parameters.multigrid.coarse_level                 = 0;
  rt_parameters.multigrid.coarse_grid.solver_variant   = prms.coarse_grid_variant;
  rt_parameters.multigrid.coarse_grid.iterative_solver = prms.solver_variant;
  rt_parameters.multigrid.coarse_grid.accuracy         = prms.coarse_grid_accuracy;
  rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
  rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = prms.patch_variant;
  rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = prms.smoother_variant;
  rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
  rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = prms.n_smoothing_steps;
  rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
  rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;
  rt_parameters.reset_damping_factor(dim);
  // rt_parameters.reset_solver_variant();

  //: misc
  rt_parameters.dof_limits = {prms.dof_limit_min, prms.dof_limit_max};
  rt_parameters.n_cycles   = 10;

  // *** configure output filenames
  std::ostringstream oss;
  oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_) + "_";
  oss << dim << "D_" << fe_degree << "deg_";
  oss << prms.test_description << "_";
  oss << prms.n_smoothing_steps << "steps_";
  oss << std::scientific << std::setprecision(3);
  oss << prms.equation_data.mu << "mu_";
  oss << prms.equation_data.lambda << "lambda";
  if(prms.equation_data.ip_factor > 1.)
    oss << prms.equation_data.ip_factor << "ip";
  const std::string filename = oss.str();

  std::fstream fstream_log;
  fstream_log.open(filename + ".log", std::ios_base::app);
  ConditionalOStream   pcout(fstream_log, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  const EquationData & equation_data = prms.equation_data;
  ModelProblem         linelasticity_problem(pcout, rt_parameters, equation_data);
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
    convergence_table.write_text(fstream_pp, TableHandler::TextOutputFormat::org_mode_table);
    fstream_pp.close();

    std::cout << "written to: " << filename << std::endl << std::endl;
  }
}

template<int dim, int fe_degree, typename value_type = double>
void
test(TestParameter prms = TestParameter{})
{
  using ElasticityDiagOnly = typename LinElasticity::ModelProblem<dim, fe_degree, value_type>;
  using BlockMatrixExact   = Tensors::BlockMatrix<dim, VectorizedArray<value_type>>;
  using ElasticityExact =
    typename LinElasticity::ModelProblem<dim, fe_degree, value_type, BlockMatrixExact>;
  using BlockMatrixFast = Tensors::BlockMatrix<dim, VectorizedArray<value_type>, /*fast*/ 0>;
  using ElasticityFast =
    typename LinElasticity::ModelProblem<dim, fe_degree, value_type, BlockMatrixFast>;

  if(prms.test_variant == 0)
  {
    prms.test_description = "diag";
    test_impl<ElasticityDiagOnly, dim, fe_degree, value_type>(prms);
  }
  else if(prms.test_variant == 1)
  {
    prms.test_description = "exact";
    test_impl<ElasticityExact, dim, fe_degree, value_type>(prms);
  }
  else if(prms.test_variant == 2)
  {
    prms.test_description = "fast";
    test_impl<ElasticityFast, dim, fe_degree, value_type>(prms);
  }
  else
    AssertThrow(false, ExcMessage("Invalid test variant."));
}
// template<int dim, int fe_degree, typename value_type = double>
// void
// test(TestParameter prms = TestParameter{})
// {
//   using ElasticityDiagOnly = typename LinElasticity::ModelProblem<dim, fe_degree, value_type>;
//   using BlockMatrixExact   = Tensors::BlockMatrix<dim, VectorizedArray<value_type>>;
//   using ElasticityExact =
//     typename LinElasticity::ModelProblem<dim, fe_degree, value_type, BlockMatrixExact>;
//   using BlockMatrixFast = Tensors::BlockMatrix<dim, VectorizedArray<value_type>, /*fast*/ 0>;
//   using ElasticityFast =
//     typename LinElasticity::ModelProblem<dim, fe_degree, value_type, BlockMatrixFast>;

//   if(prms.test_variant == 0)
//   {
//     prms.test_description = "diag";
//     debug_impl<ElasticityDiagOnly, dim, fe_degree, value_type>(prms);
//   }
//   else if(prms.test_variant == 1)
//   {
//     prms.test_description = "exact";
//     debug_impl<ElasticityExact, dim, fe_degree, value_type>(prms);
//   }
//   else if(prms.test_variant == 2)
//   {
//     prms.test_description = "fast";
//     debug_impl<ElasticityFast, dim, fe_degree, value_type>(prms);
//   }
//   else
//     AssertThrow(false, ExcMessage("Invalid test variant."));
// }

int
main(int argc, char * argv[])
{
  deallog.depth_console(3);

  // *** init TBB and MPI
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // *** static TEST parameters
  using value_type             = double;
  constexpr unsigned dim       = 2; // CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;

  TestParameter test_prms;
  if(argc > 1)
    test_prms.test_variant = std::atoi(argv[1]);
  if(argc > 2)
    test_prms.n_smoothing_steps = std::atoi(argv[2]);
  if(argc > 3)
    test_prms.equation_data.mu = std::atof(argv[3]);
  if(argc > 4)
    test_prms.equation_data.lambda = std::atof(argv[4]);
  if(argc > 5)
    test_prms.equation_data.ip_factor = std::atof(argv[5]);
  if(argc > 7)
    test_prms.equation_data.kronecker_rank = std::atoi(argv[7]);
  if(argc > 8)
    test_prms.equation_data.lambda_rank = std::atoi(argv[8]);
  if(argc > 9)
    test_prms.equation_data.factor = std::atof(argv[9]);

  test<dim, fe_degree, value_type>(test_prms);

  return 0;
}
