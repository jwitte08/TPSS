
/*
 * test poisson problem on distorted mesh
 *
 *  Created on: Oct 09, 2019
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "ct_parameter.h"
#include "poisson_problem.h"

using namespace dealii;

struct TestParameter
{
  std::map<int, int> dim_to_refinements = {{2, 3}, {3, 1}};

  unsigned n_refinements     = dim_to_refinements.at(CT::DIMENSION_);
  unsigned n_cycles          = 3;
  unsigned n_smoothing_steps = 1;
  double   distortion        = 0.1;
  double   damping           = 1.0;
  double   penalty           = 4.;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Number of run cycles:", n_cycles);
    oss << Util::parameter_to_fstring("Distortion in %:", 100. * distortion);
    oss << Util::parameter_to_fstring("Interior penalty pre-factor:", penalty);
    oss << Util::parameter_to_fstring("Damping factor (combined):", damping);
    return oss.str();
  }
};

template<int dim, int fe_degree>
struct Test
{
  static constexpr int n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);
  static constexpr auto dof_layout = TPSS::DoFLayout::DGQ;
  using PoissonProblem =
    typename Poisson::ModelProblem<dim, fe_degree, dof_layout, double, n_patch_dofs_per_direction>;
  using VECTOR           = typename PoissonProblem::VECTOR;
  using SCHWARZ_SMOOTHER = typename PoissonProblem::SCHWARZ_SMOOTHER;
  using SYSTEM_MATRIX    = typename PoissonProblem::SYSTEM_MATRIX;
  using LEVEL_MATRIX     = typename PoissonProblem::LEVEL_MATRIX;

  const TestParameter      prms;
  RT::Parameter            rt_parameters;
  Laplace::EquationData    equation_data;
  const double             outer_damping_factor;
  const std::map<int, int> dim_to_repetitions;

  Test(const TestParameter & prms_in = TestParameter{})
    : prms(prms_in),
      outer_damping_factor(
        TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, dim)),
      dim_to_repetitions({{2, 32}, {3, 8}})
  {
    //: discretization
    rt_parameters.n_cycles              = prms.n_cycles;
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::CubeDistorted;
    rt_parameters.mesh.n_refinements    = prms.n_refinements;
    rt_parameters.mesh.n_repetitions    = dim_to_repetitions.at(dim);
    rt_parameters.mesh.distortion       = prms.distortion;

    //: solver
    rt_parameters.solver.variant              = "cg";
    rt_parameters.solver.rel_tolerance        = 1.e-8;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;

    //: multigrid
    rt_parameters.multigrid.coarse_level = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant =
      CoarseGridParameter::SolverVariant::Iterative;
    rt_parameters.multigrid.coarse_grid.iterative_solver = "cg";
    rt_parameters.multigrid.coarse_grid.accuracy         = 1.e-8;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = CT::PATCH_VARIANT_;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = CT::SMOOTHER_VARIANT_;
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = outer_damping_factor;
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = prms.n_smoothing_steps;
    rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(8, fe_degree + 1);
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << write_header();
  }

  std::string
  write_header()
  {
    std::ostringstream oss;
    oss << Util::generic_info_to_fstring() << std::endl;
    oss << prms.to_string() << std::endl;
    return oss.str();
  }

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
    }

    info_table.set_scientific("reduction", true);
    info_table.set_precision("reduction", 3);
    info_table.write_text(oss);

    return oss.str();
  }

  //: determine filename from test parameters
  std::string
  get_filename()
  {
    std::ostringstream oss;
    const auto         n_mpi_procs    = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto         damping_factor = outer_damping_factor;

    oss << std::scientific << std::setprecision(2);
    oss << n_mpi_procs << "procs";
    oss << "_" << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
    oss << "_" << dim << "D";
    oss << "_" << fe_degree << "deg";
    oss << "_" << rt_parameters.mesh.distortion << "dist";
    oss << "_" << rt_parameters.multigrid.pre_smoother.n_smoothing_steps << "step";
    {
      oss << "_" << rt_parameters.mesh.n_refinements;
      if(rt_parameters.n_cycles > 1)
        oss << "to" << rt_parameters.mesh.n_refinements + rt_parameters.n_cycles - 1;
      oss << "L";
    }
    oss << "_" << damping_factor << "damp";
    return oss.str();
  }

  void
  operator()()
  {
    PoissonProblem poisson_problem{rt_parameters, equation_data};
    const bool     is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::fstream   fstream;
    poisson_problem.pcout      = std::make_shared<ConditionalOStream>(fstream, is_first_proc);
    const std::string filename = get_filename();
    if(is_first_proc)
      fstream.open("poisson_" + filename + ".log", std::ios_base::out);
    poisson_problem.run();

    const auto pp_data = poisson_problem.pp_data;
    fstream << write_ppdata_to_string(pp_data);
    if(is_first_proc)
      fstream.close();
  }
};


int
main(int argc, char * argv[])
{
  // *** init TBB and MPI
  constexpr unsigned int           max_threads = 1; // no multithreading !?
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** run-time options
  TestParameter prms;
  prms.damping =
    TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, CT::DIMENSION_);
  if(argc > 1)
    prms.distortion = std::atof(argv[1]);
  if(argc > 2)
    prms.damping = std::atof(argv[2]);
  if(argc > 3)
    prms.n_smoothing_steps = std::atoi(argv[3]);
  if(argc > 4)
    prms.n_refinements = std::atoi(argv[4]);
  if(argc > 5)
    prms.n_cycles = std::atoi(argv[5]);

  // *** run tests
  Timer      time(MPI_COMM_WORLD, true);
  const bool is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  const auto pcout         = ConditionalOStream(std::cout, is_first_proc);
  Test<CT::DIMENSION_, CT::FE_DEGREE_> tester(prms);

  time.restart();
  pcout << Util::parameter_to_fstring("Testing distorted meshes", "");
  tester();
  time.stop();
  time.print_last_lap_wall_time_data(pcout);
  pcout << std::endl;

  return 0;
}
