#include <deal.II/base/convergence_table.h>


#include "ct_parameter.h"
#include "stokes_problem.h"



std::string
write_ppdata_to_string(const PostProcessData & pp_data, const PostProcessData & pp_data_pressure)
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
    info_table.add_value("L2_error_u", pp_data.L2_error.at(run));
    info_table.add_value("L2_error_p", pp_data_pressure.L2_error.at(run));
    info_table.add_value("H1semi_error_u", pp_data.H1semi_error.at(run));
  }
  info_table.set_scientific("reduction", true);
  info_table.set_precision("reduction", 3);
  info_table.set_scientific("L2_error_u", true);
  info_table.set_precision("L2_error_u", 3);
  info_table.evaluate_convergence_rates("L2_error_u", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("L2_error_u",
                                        "n_dofs",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);
  info_table.set_scientific("H1semi_error_u", true);
  info_table.set_precision("H1semi_error_u", 3);
  info_table.evaluate_convergence_rates("H1semi_error_u", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("H1semi_error_u",
                                        "n_dofs",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);
  info_table.set_scientific("L2_error_p", true);
  info_table.set_precision("L2_error_p", 3);
  info_table.evaluate_convergence_rates("L2_error_p", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("L2_error_p",
                                        "n_dofs",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);

  info_table.write_text(oss);
  return oss.str();
}



int
main(int argc, char * argv[])
{
  try
  {
    using namespace Stokes;

    // deallog.depth_console(3);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int                        dim    = CT::DIMENSION_;
    const int                        degree = CT::FE_DEGREE_;

    // 0: direct solver (UMFPACK)
    // 1: flexible GMRES prec. by ILU (FGMRES_ILU)
    // 2: flexible GMRES prec. by Schur approx., GMG for velocity (FGMRES_GMG)
    constexpr unsigned int test_index_max = 2;
    const unsigned int     test_index     = argc > 2 ? std::atoi(argv[2]) : 0;
    AssertThrow(test_index <= test_index_max, ExcMessage("test_index is not valid"));

    RT::Parameter prms;
    {
      //: discretization
      prms.n_cycles              = 10;
      prms.dof_limits            = {1e3, 1e5};
      prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
      prms.mesh.n_refinements    = 1;
      prms.mesh.n_repetitions    = 2;

      //: solver
      const std::string str_solver_variant[test_index_max + 1] = {"UMFPACK",
                                                                  "FGMRES_ILU",
                                                                  "FGMRES_GMG"};
      prms.solver.variant                                      = str_solver_variant[test_index];
      prms.solver.rel_tolerance                                = 1.e-8;
      prms.solver.precondition_variant                         = test_index >= 2 ?
                                           SolverParameter::PreconditionVariant::GMG :
                                           SolverParameter::PreconditionVariant::None;
      prms.solver.n_iterations_max = 200;

      //: multigrid
      const double damping_factor =
        (argc > 1) ? std::atof(argv[1]) :
                     TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, dim);
      prms.multigrid.coarse_level                 = 0;
      prms.multigrid.coarse_grid.solver_variant   = CoarseGridParameter::SolverVariant::FullSVD;
      prms.multigrid.coarse_grid.iterative_solver = "cg";
      prms.multigrid.coarse_grid.accuracy         = 1.e-12;
      const SmootherParameter::SmootherVariant smoother_variant[test_index_max + 1] = {
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None};
      prms.multigrid.pre_smoother.variant                    = smoother_variant[test_index];
      prms.multigrid.pre_smoother.n_smoothing_steps          = 2;
      prms.multigrid.pre_smoother.schwarz.patch_variant      = CT::PATCH_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.smoother_variant   = CT::SMOOTHER_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.manual_coloring    = true;
      prms.multigrid.pre_smoother.schwarz.damping_factor     = damping_factor;
      prms.multigrid.post_smoother                           = prms.multigrid.pre_smoother;
      prms.multigrid.post_smoother.schwarz.reverse_smoothing = true;
    }

    EquationData              equation_data;
    ModelProblem<dim, degree> flow_problem(prms, equation_data);

    flow_problem.run();
    std::cout << std::endl
              << std::endl
              << write_ppdata_to_string(flow_problem.pp_data, flow_problem.pp_data_pressure);
  }

  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }

  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
