
/*
 * sandbox poisson problem
 *
 *  Created on: Oct 30, 2019
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "ct_parameter.h"
#include "poisson_problem.h"

using namespace dealii;
using namespace Laplace;

struct TestParameter
{
  TPSS::PatchVariant                 patch_variant    = CT::PATCH_VARIANT_;
  TPSS::SmootherVariant              smoother_variant = CT::SMOOTHER_VARIANT_;
  std::string                        solver_variant   = "cg"; // see SolverSelector
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::IterativeAcc;
  double   coarse_grid_accuracy = 1.e-12;
  double   cg_reduction         = 1.e-8;
  unsigned n_refinements        = 1;
  unsigned n_repetitions        = 2;
  double   extra_damping        = 1.;

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
    rt_parameters.n_cycles              = 10;
    rt_parameters.dof_limits            = {1e5, 2e7};
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = testprms.n_refinements;
    rt_parameters.mesh.n_repetitions    = testprms.n_repetitions;

    //: solver
    rt_parameters.solver.variant              = testprms.solver_variant;
    rt_parameters.solver.rel_tolerance        = testprms.cg_reduction;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
    rt_parameters.solver.n_iterations_max     = 100;

    //: multigrid
    const double damping_factor =
      testprms.extra_damping *
      TPSS::lookup_damping_factor(testprms.patch_variant, testprms.smoother_variant, dim);
    rt_parameters.multigrid.coarse_level                 = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant   = testprms.coarse_grid_variant;
    rt_parameters.multigrid.coarse_grid.iterative_solver = testprms.solver_variant;
    rt_parameters.multigrid.coarse_grid.accuracy         = testprms.coarse_grid_accuracy;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = testprms.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = testprms.smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.manual_coloring      = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
    rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(5, fe_degree + 1);
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

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
    const auto         n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const std::string  str_schwarz_variant =
      TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);

    oss << "poisson";
    oss << std::scientific << std::setprecision(2);
    oss << "_" << n_mpi_procs << "prcs";
    oss << "_" << TPSS::str_dof_layout(dof_layout);
    oss << "_" << str_schwarz_variant;
    oss << "_" << dim << "D";
    oss << "_" << fe_degree << "deg";
    if(testprms.extra_damping != 1.)
      oss << "_" << testprms.extra_damping << "xdmp";
    return oss.str();
  }

  void
  run()
  {
    poisson_problem = std::make_shared<PoissonProblem>(rt_parameters);

    //: write headers to logfile
    *pcout << Util::generic_info_to_fstring() << std::endl;
    *pcout << testprms.to_string() << std::endl;

    //: run
    poisson_problem->pcout = this->pcout;
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
  const TestParameter &               testprms;
  const bool                          is_first_proc;
  std::fstream                        fout;
  std::shared_ptr<ConditionalOStream> pcout;
};



int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  constexpr int                    dim        = CT::DIMENSION_;
  constexpr int                    fe_degree  = CT::FE_DEGREE_;
  constexpr auto                   dof_layout = CT::DOF_LAYOUT_;
  constexpr int                    n_patch_dofs_1d_static =
    TPSS::UniversalInfo<dim>::n_dofs_1d(CT::PATCH_VARIANT_, dof_layout, fe_degree);

  TestParameter testprms;
  if(argc > 1)
    testprms.extra_damping = std::atof(argv[1]);

  Tester<dim, fe_degree, dof_layout, n_patch_dofs_1d_static> tester(testprms);
  tester.run();

  return 0;
}
