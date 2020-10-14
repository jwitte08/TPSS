#include <deal.II/base/convergence_table.h>


#include "app_utilities.h"
#include "biharmonic_problem.h"
#include "ct_parameter.h"



std::string
write_ppdata_to_string(const PostProcessData & pp_data,
                       const PostProcessData & pp_data_velocity = PostProcessData{},
                       const PostProcessData & pp_data_pressure = PostProcessData{})
{
  const bool has_velocity_data = !pp_data_velocity.L2_error.empty();
  const bool has_pressure_data = !pp_data_pressure.L2_error.empty();

  std::ostringstream oss;
  ConvergenceTable   info_table;
  Assert(!pp_data.n_cells_global.empty(), ExcMessage("No cells to post process."));
  if(has_pressure_data)
    AssertDimension(pp_data_pressure.n_dofs_global.size(), pp_data.n_dofs_global.size());
  for(unsigned run = 0; run < pp_data.n_cells_global.size(); ++run)
  {
    info_table.add_value("n_levels", pp_data.n_mg_levels.at(run));
    info_table.add_value("n_cells", pp_data.n_cells_global.at(run));
    info_table.add_value("n_dofs", pp_data.n_dofs_global.at(run));
    info_table.add_value("n_colors", pp_data.n_colors_system.at(run));
    info_table.add_value("n_iter", pp_data.n_iterations_system.at(run));
    info_table.add_value("reduction", pp_data.average_reduction_system.at(run));
    info_table.add_value("L2_error", pp_data.L2_error.at(run));
    info_table.add_value("H2semiO_error", pp_data.H2semi_error.at(run));
    if(has_velocity_data)
      info_table.add_value("L2_velocity_error", pp_data_velocity.L2_error.at(run));
    if(has_pressure_data)
    {
      info_table.add_value("n_dofs_p", pp_data_pressure.n_dofs_global.at(run));
      info_table.add_value("L2_pressure_error", pp_data_pressure.L2_error.at(run));
    }
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
  info_table.set_scientific("H2semiO_error", true);
  info_table.set_precision("H2semiO_error", 3);
  info_table.evaluate_convergence_rates("H2semiO_error", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("H2semiO_error",
                                        "n_dofs",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);

  if(has_velocity_data)
  {
    info_table.set_scientific("L2_velocity_error", true);
    info_table.set_precision("L2_velocity_error", 3);
    info_table.evaluate_convergence_rates("L2_velocity_error", ConvergenceTable::reduction_rate);
    info_table.evaluate_convergence_rates("L2_velocity_error",
                                          "n_dofs",
                                          ConvergenceTable::reduction_rate_log2,
                                          pp_data.n_dimensions);
  }

  if(has_pressure_data)
  {
    info_table.set_scientific("L2_pressure_error", true);
    info_table.set_precision("L2_pressure_error", 3);
    info_table.evaluate_convergence_rates("L2_pressure_error", ConvergenceTable::reduction_rate);
    info_table.evaluate_convergence_rates("L2_pressure_error",
                                          "n_dofs_p",
                                          ConvergenceTable::reduction_rate_log2,
                                          pp_data.n_dimensions);
  }

  info_table.write_text(oss);
  return oss.str();
}

std::string
get_filename(const RT::Parameter &            prms,
             const Biharmonic::EquationData & equation_data,
             const bool                       add_damping = false)
{
  std::ostringstream oss;
  const auto         n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const std::string  str_schwarz_variant =
    TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
  const auto damping = prms.multigrid.pre_smoother.schwarz.damping_factor *
                       prms.multigrid.pre_smoother.schwarz.local_damping_factor;
  const auto short_name = [](const std::string & str_in) {
    std::string sname = str_in.substr(0, 4);
    std::transform(sname.begin(), sname.end(), sname.begin(), [](auto c) {
      return std::tolower(c);
    });
    return sname;
  };

  oss << "biharmonic";
  oss << std::scientific << std::setprecision(2);
  oss << "_" << n_mpi_procs << "prcs";
  if(prms.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
  {
    oss << "_" << str_schwarz_variant;
    oss << "_" << short_name(equation_data.str_local_solver());
  }
  oss << "_" << CT::DIMENSION_ << "D";
  oss << "_" << CT::FE_DEGREE_ << "deg";
  if(add_damping)
    oss << "_" << damping << "damp";
  return oss.str();
}



int
main(int argc, char * argv[])
{
  try
  {
    using namespace dealii;
    using namespace Biharmonic;

    const auto atoi_if = [&](auto & prm, const int index) {
      Util::ConditionalAtoi(argc, argv)(prm, index);
    };
    const auto atof_if = [&](auto & prm, const int index) {
      Util::ConditionalAtof(argc, argv)(prm, index);
    };

    //: default
    unsigned int test_index  = 4;
    unsigned int debug_depth = 0;
    double       damping     = 0.;
    double       ip_factor   = 1.;
    unsigned int pde_index   = 0;


    //: parse arguments
    atoi_if(test_index, 1);
    atoi_if(pde_index, 2);
    atof_if(ip_factor, 3);
    atoi_if(debug_depth, 4);
    atof_if(damping, 5);

    deallog.depth_console(debug_depth);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    constexpr int  dim              = CT::DIMENSION_;
    constexpr int  fe_degree        = CT::FE_DEGREE_;
    constexpr auto patch_variant    = CT::PATCH_VARIANT_;
    constexpr auto smoother_variant = CT::SMOOTHER_VARIANT_;

    // 0: direct solver
    // 1: CG solver (no preconditioner)
    // 2: CG solver (GMG preconditioner without smoothing)
    // 3: CG solver (GMG preconditioner with symm. Gauss-Seidel smoothing)
    // 4: CG solver (GMG preconditioner with Schwarz smoothing)
    constexpr unsigned int test_index_max = 4;
    AssertThrow(test_index <= test_index_max, ExcMessage("test_index is not valid"));

    RT::Parameter prms;
    {
      //: discretization
      prms.n_cycles              = 10;
      prms.dof_limits            = {1e1, 1e5}; //{1e4, 1e6};
      prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
      prms.mesh.n_refinements    = 1;
      prms.mesh.n_repetitions    = 2;

      //: solver
      prms.solver.variant              = test_index == 0 ? "direct" : "cg";
      prms.solver.rel_tolerance        = 1.e-8;
      prms.solver.precondition_variant = test_index >= 2 ?
                                           SolverParameter::PreconditionVariant::GMG :
                                           SolverParameter::PreconditionVariant::None;
      prms.solver.n_iterations_max = 200;

      //: multigrid
      if(damping == 0.)
        damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
      prms.multigrid.coarse_level                 = 0;
      prms.multigrid.coarse_grid.solver_variant   = CoarseGridParameter::SolverVariant::FullSVD;
      prms.multigrid.coarse_grid.iterative_solver = "cg";
      prms.multigrid.coarse_grid.accuracy         = 1.e-12;
      const SmootherParameter::SmootherVariant smoother_variant[test_index_max + 1] = {
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::GaussSeidel,
        SmootherParameter::SmootherVariant::Schwarz};
      prms.multigrid.pre_smoother.variant                    = smoother_variant[test_index];
      prms.multigrid.pre_smoother.n_smoothing_steps          = 2;
      prms.multigrid.pre_smoother.schwarz.patch_variant      = CT::PATCH_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.smoother_variant   = CT::SMOOTHER_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.manual_coloring    = true;
      prms.multigrid.pre_smoother.schwarz.damping_factor     = damping;
      prms.multigrid.post_smoother                           = prms.multigrid.pre_smoother;
      prms.multigrid.post_smoother.schwarz.reverse_smoothing = true;
    }

    EquationData equation_data;
    AssertThrow(pde_index < EquationData::n_variants,
                ExcMessage("This equation is not implemented."));
    equation_data.variant              = static_cast<EquationData::Variant>(pde_index);
    equation_data.local_solver_variant = LocalSolverVariant::Exact;
    equation_data.ip_factor            = ip_factor;

    ModelProblem<dim, fe_degree> biharmonic_problem(prms, equation_data);

    std::fstream fout;
    const auto   filename = get_filename(prms, equation_data, argc > 1);
    fout.open(filename + ".log", std::ios_base::out);
    auto pcout               = std::make_shared<ConditionalOStream>(std::cout /*!!!fout*/, true);
    biharmonic_problem.pcout = pcout;

    biharmonic_problem.run();

    std::string pp_output_as_string;
    if(biharmonic_problem.equation_data.is_stream_function())
    {
      AssertThrow(biharmonic_problem.stokes_problem, ExcMessage("stokes_problem isnt initialized"));
      const auto & stokes = *biharmonic_problem.stokes_problem;
      pp_output_as_string =
        write_ppdata_to_string(biharmonic_problem.pp_data, stokes.pp_data, stokes.pp_data_pressure);
    }
    else
      pp_output_as_string = write_ppdata_to_string(biharmonic_problem.pp_data);

    *pcout << std::endl << std::endl << pp_output_as_string;
    fout.close();

    fout.open(filename + ".tab", std::ios_base::out);
    fout << pp_output_as_string;
    fout.close();
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
