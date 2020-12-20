
/*
 * sandbox poisson problem
 *
 *  Created on: Oct 30, 2019
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "ct_parameter.h"
#include "poisson_problem.h"
#include "utilities.h"

using namespace dealii;
using namespace Laplace;



struct TestParameter
{
  TPSS::PatchVariant                 patch_variant    = CT::PATCH_VARIANT_;
  TPSS::SmootherVariant              smoother_variant = CT::SMOOTHER_VARIANT_;
  std::string                        solver_variant   = "gmres"; //!!! see SolverSelector
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::Iterative;
  double       coarse_grid_accuracy    = 1.e-12;
  double       cg_reduction            = 1.e-8; // -8 !!!
  unsigned     n_refinements           = 1;
  unsigned     n_repetitions           = 2;
  double       damping                 = 0.;
  bool         use_ras_weights         = false;
  bool         use_ras_boolean_weights = false;
  unsigned int n_smoothing_steps       = 1;
  EquationData equation_data;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("n_refinements (initial):", n_refinements);
    oss << Util::parameter_to_fstring("n_repetitions (initial):", n_repetitions);
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
    info_table.add_value("n_levels", pp_data.n_mg_levels.at(run));
    info_table.add_value("n_cells", pp_data.n_cells_global.at(run));
    info_table.add_value("n_dofs", pp_data.n_dofs_global.at(run));
    info_table.add_value("n_colors", pp_data.n_colors_system.at(run));
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

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, int n_patch_dofs_1d_static = -1>
struct Tester
{
  using PoissonProblem =
    typename Poisson::ModelProblem<dim, fe_degree, dof_layout, double, n_patch_dofs_1d_static>;

  Tester(const TestParameter & testprms_in)
    : testprms(testprms_in),
      is_first_proc(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    //: discretization
    rt_parameters.n_cycles              = 15;
    rt_parameters.dof_limits            = {1e1, 5e5}; // {1e5, 1e8};
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = testprms.n_refinements;
    rt_parameters.mesh.n_repetitions    = testprms.n_repetitions;
    rt_parameters.use_tbb               = MultithreadInfo::n_threads() > 1;

    //: solver
    rt_parameters.solver.variant              = testprms.solver_variant;
    rt_parameters.solver.rel_tolerance        = testprms.cg_reduction;
    rt_parameters.solver.abs_tolerance        = 1.e-16;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
    rt_parameters.solver.n_iterations_max     = 100;

    //: multigrid
    if(testprms.damping == 0.)
      testprms.damping =
        (testprms.use_ras_weights || testprms.use_ras_boolean_weights) ?
          1. :
          TPSS::lookup_damping_factor(testprms.patch_variant, testprms.smoother_variant, dim);
    rt_parameters.multigrid.coarse_level                 = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant   = testprms.coarse_grid_variant;
    rt_parameters.multigrid.coarse_grid.iterative_solver = testprms.solver_variant;
    rt_parameters.multigrid.coarse_grid.accuracy         = testprms.coarse_grid_accuracy;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = testprms.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = testprms.smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.use_ras_weights  = testprms.use_ras_weights; // !!!
    rt_parameters.multigrid.pre_smoother.schwarz.use_ras_boolean_weights =
      testprms.use_ras_boolean_weights; // !!!
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = testprms.damping;
    rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(5, fe_degree + 1);
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = testprms.n_smoothing_steps;
    // rt_parameters.multigrid.pre_smoother.use_doubling_of_steps = true;
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing =
      (testprms.solver_variant == "cg");

    //: initialize output file
    if(is_first_proc)
    {
      const auto filename = get_filename();
      fout.open(filename + ".log", std::ios_base::out);
      pcout = std::make_shared<ConditionalOStream>(fout, is_first_proc);
    }
  }

  ~Tester()
  {
    if(is_first_proc)
      fout.close();
  }

  std::string
  get_filename()
  {
    std::ostringstream oss;
    const auto         n_mpi_procs            = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto         n_threads_per_mpi_proc = MultithreadInfo::n_threads();
    const std::string  str_schwarz_variant =
      TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);

    oss << "poisson";
    oss << std::scientific << std::setprecision(2);
    oss << "_" << n_mpi_procs << "prcs";
    if(n_threads_per_mpi_proc > 1)
      oss << "_" << n_threads_per_mpi_proc << "tpp";
    oss << "_" << TPSS::str_dof_layout(dof_layout);
    oss << "_" << str_schwarz_variant;
    oss << "_" << dim << "D";
    oss << "_" << fe_degree << "deg";
    if(testprms.damping != 1.)
      oss << "_" << testprms.damping << "xdmp";
    if(testprms.use_ras_weights)
      oss << "_RASw";
    if(testprms.use_ras_boolean_weights)
      oss << "_RASb";
    return oss.str();
  }

  void
  run()
  {
    poisson_problem = std::make_shared<PoissonProblem>(rt_parameters, testprms.equation_data);

    //: write headers to logfile
    *pcout << Util::generic_info_to_fstring() << std::endl;
    *pcout << testprms.to_string() << std::endl;

    //: run
    poisson_problem->pcout = this->pcout;
    // poisson_problem->fe = std::make_shared<FE_DGQHermite<dim>>(fe_degree);
    poisson_problem->run();

    //: write post process data to logfile
    *pcout << "!TABLEPP\n";
    const auto pp_data = poisson_problem->pp_data;
    *pcout << write_ppdata_to_string(pp_data);
    *pcout << "TABLE!PP\n";

    //: write post process data to table
    if(is_first_proc)
    {
      std::fstream fout_table;
      const auto   filename = get_filename();
      fout_table.open(filename + ".tab", std::ios_base::out);
      fout_table << write_ppdata_to_string(pp_data);
      fout_table.close();
    }
  }

  std::shared_ptr<PoissonProblem>     poisson_problem;
  RT::Parameter                       rt_parameters;
  TestParameter                       testprms;
  const bool                          is_first_proc;
  std::fstream                        fout;
  std::shared_ptr<ConditionalOStream> pcout;
};



int
main(int argc, char * argv[])
{
  const auto atoi_if = [&](auto & prm, const int index) {
    Util::ConditionalAtoi(argc, argv)(prm, index);
  };
  const auto atof_if = [&](auto & prm, const int index) {
    Util::ConditionalAtof(argc, argv)(prm, index);
  };

  const unsigned int solver_index_max                     = 1;
  const std::string  solver_variant[solver_index_max + 1] = {"cg", "gmres"};

  constexpr int  dim        = CT::DIMENSION_;
  constexpr int  fe_degree  = CT::FE_DEGREE_;
  constexpr auto dof_layout = CT::DOF_LAYOUT_;
  constexpr int  n_patch_dofs_1d_static =
    TPSS::UniversalInfo<dim>::n_dofs_1d(CT::PATCH_VARIANT_, dof_layout, fe_degree);

  //: default
  TestParameter testprms;
  unsigned int  solver_index            = 0; // CG + GMG + Schwarz
  unsigned int  use_ras_weights         = 0;
  unsigned int  use_ras_boolean_weights = 0;
  int           n_threads_max           = 1;
  unsigned int  pde_index               = 1; // Gaussian Bells
  unsigned int  debug_depth             = 0;

  //: parse runtime arguments
  atoi_if(solver_index, 1);
  atoi_if(pde_index, 2);
  atoi_if(use_ras_weights, 3);
  atoi_if(use_ras_boolean_weights, 4);
  atoi_if(n_threads_max, 5);
  atoi_if(testprms.n_smoothing_steps, 6);
  atof_if(testprms.damping, 7);
  atoi_if(debug_depth, 8);

  deallog.depth_console(debug_depth);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                      argv,
                                                      n_threads_max == -1 ?
                                                        numbers::invalid_unsigned_int :
                                                        static_cast<unsigned int>(n_threads_max));

  AssertThrow(solver_index <= solver_index_max, ExcMessage("Invalid solver index."));
  AssertThrow(pde_index <= EquationData::n_variants, ExcMessage("Invalid PDE index."));
  AssertThrow(use_ras_weights <= 1U, ExcMessage("Invalid integer value for use_ras_weights."));
  AssertThrow(use_ras_boolean_weights <= 1U,
              ExcMessage("Invalid integer value for use_ras_boolean_weights."));
  AssertThrow(testprms.damping <= 1., ExcMessage("No over-relaxation allowed."));

  testprms.solver_variant          = solver_variant[solver_index];
  testprms.equation_data.variant   = static_cast<EquationData::Variant>(pde_index);
  testprms.use_ras_weights         = static_cast<bool>(use_ras_weights);
  testprms.use_ras_boolean_weights = static_cast<bool>(use_ras_boolean_weights);

  Tester<dim, fe_degree, dof_layout, n_patch_dofs_1d_static> tester(testprms);
  tester.run();

  return 0;
}
