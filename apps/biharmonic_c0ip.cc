#include <deal.II/base/convergence_table.h>


#include "biharmonic_problem.h"
#include "ct_parameter.h"
#include "utilities.h"



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
    info_table.add_value("energy_error", pp_data.H2semi_error.at(run));
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
  info_table.set_scientific("energy_error", true);
  info_table.set_precision("energy_error", 3);
  info_table.evaluate_convergence_rates("energy_error", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("energy_error",
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
get_filename(const RT::Parameter & prms, const Biharmonic::EquationData & equation_data)
{
  std::ostringstream oss;

  const auto        n_mpi_procs            = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const auto        n_threads_per_mpi_proc = MultithreadInfo::n_threads();
  const auto &      pre_schwarz            = prms.multigrid.pre_smoother.schwarz;
  const auto        damping                = pre_schwarz.damping_factor;
  const std::string str_schwarz_variant =
    TPSS::getstr_schwarz_variant(pre_schwarz.patch_variant, pre_schwarz.smoother_variant);

  oss << "biharm";
  oss << std::scientific << std::setprecision(2);
  oss << "_" << n_mpi_procs << "prcs";
  if(n_threads_per_mpi_proc > 1)
    oss << "_" << n_threads_per_mpi_proc << "tpp";
  if(prms.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
  {
    oss << "_" << str_schwarz_variant;
    oss << "_" << Util::short_name(equation_data.sstr_local_solver());
    if(equation_data.local_solver_variant == Biharmonic::LocalSolverVariant::KSVD)
      for(const auto index : equation_data.ksvd_tensor_indices)
        oss << index;
  }
  oss << "_" << CT::DIMENSION_ << "D";
  oss << "_" << CT::FE_DEGREE_ << "deg";
  if(damping != 1.)
    oss << "_" << Util::damping_to_fstring(damping) << "damp";
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
    unsigned int solver_index              = 4; // CG + unprec
    unsigned int debug_depth               = 0;
    double       damping                   = 0.;
    double       ip_factor                 = 1.;
    unsigned int pde_index                 = 1; // clamped Gaussian bells
    int          n_threads_max             = 1;
    unsigned int use_hierarchical_elements = false;
    unsigned int use_doubling_of_steps     = false;
    unsigned int n_smoothing_steps         = 2;
    unsigned int local_solver_index        = 0; // exact
    unsigned int ksvd_rank                 = 1;

    //: parse arguments
    atoi_if(solver_index, 1);
    atoi_if(pde_index, 2);
    atof_if(damping, 3);
    atof_if(local_solver_index, 4);
    atof_if(ksvd_rank, 5);
    atoi_if(n_smoothing_steps, 6);
    atof_if(ip_factor, 7);
    atoi_if(n_threads_max, 8);
    atoi_if(use_doubling_of_steps, 9);
    atoi_if(debug_depth, 10);
    atoi_if(use_hierarchical_elements, 11);

    deallog.depth_console(debug_depth);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                        argv,
                                                        n_threads_max == -1 ?
                                                          numbers::invalid_unsigned_int :
                                                          static_cast<unsigned int>(n_threads_max));
    const auto                       n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    constexpr int  dim                   = CT::DIMENSION_;
    constexpr int  fe_degree             = CT::FE_DEGREE_;
    constexpr auto patch_variant         = CT::PATCH_VARIANT_;
    constexpr auto tpss_smoother_variant = CT::SMOOTHER_VARIANT_;

    // 0: direct solver
    // 1: CG solver (no preconditioner)
    // 2: CG solver (GMG preconditioner without smoothing)
    // 3: CG solver (GMG preconditioner with symm. Gauss-Seidel smoothing)
    // 4: CG solver (GMG preconditioner with Schwarz smoothing)
    constexpr unsigned int solver_index_max = 4;
    AssertThrow(solver_index <= solver_index_max, ExcMessage("solver_index is not valid"));
    AssertThrow(damping <= 1., ExcMessage("Do you intend to over-relax?"));
    AssertThrow(ip_factor >= 1., ExcMessage("IP factor should be larger than one."));
    AssertThrow(n_threads_max == -1 || n_threads_max > 0,
                ExcMessage("Check the number of active threads."));
    AssertThrow(use_hierarchical_elements == 0 || use_hierarchical_elements == 1,
                ExcMessage("use_hierarchical_elements is treated as boolean"));
    AssertThrow(use_doubling_of_steps == 0 || use_doubling_of_steps == 1,
                ExcMessage("use_doubling_of_steps is treated as boolean"));
    AssertThrow(n_smoothing_steps < 10, ExcMessage("Check n_smoothing_steps!"));
    AssertThrow(local_solver_index < 3U, ExcMessage("invalid local solver"));

    RT::Parameter prms;
    {
      prms.use_tbb = MultithreadInfo::n_threads() > 1;

      //: discretization
      prms.n_cycles              = 12;
      prms.dof_limits            = {1e1, 2e5}; //{1e5, 1e7};
      prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
      prms.mesh.n_refinements    = 1;
      prms.mesh.n_repetitions    = 2;

      //: solver
      prms.solver.variant              = solver_index == 0 ? "direct" : "cg";
      prms.solver.n_iterations_max     = 100;
      prms.solver.control_variant      = SolverParameter::ControlVariant::relative;
      prms.solver.abs_tolerance        = 1.e-14;
      prms.solver.rel_tolerance        = 1.e-08;
      prms.solver.precondition_variant = solver_index >= 2 ?
                                           SolverParameter::PreconditionVariant::GMG :
                                           SolverParameter::PreconditionVariant::None;

      //: multigrid
      prms.multigrid.coarse_level               = 0;
      prms.multigrid.coarse_grid.solver_variant = CoarseGridParameter::SolverVariant::DirectSVD;
      const SmootherParameter::SmootherVariant smoother_variant[solver_index_max + 1] = {
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::GaussSeidel,
        SmootherParameter::SmootherVariant::Schwarz};
      prms.multigrid.pre_smoother.variant                      = smoother_variant[solver_index];
      prms.multigrid.pre_smoother.n_smoothing_steps            = n_smoothing_steps;
      prms.multigrid.pre_smoother.schwarz.patch_variant        = patch_variant;
      prms.multigrid.pre_smoother.schwarz.smoother_variant     = tpss_smoother_variant;
      prms.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
      prms.multigrid.pre_smoother.schwarz.damping_factor       = damping;
      if(prms.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::GaussSeidel)
        prms.multigrid.pre_smoother.damping_factor =
          damping == 0. ? (n_mpi_procs == 1 ? 1. : 0.7) : damping;
      prms.multigrid.pre_smoother.use_doubling_of_steps      = use_doubling_of_steps;
      prms.multigrid.post_smoother                           = prms.multigrid.pre_smoother;
      prms.multigrid.post_smoother.schwarz.reverse_smoothing = prms.solver.variant == "cg";
      if(damping == 0.)
        prms.reset_damping_factor(dim);
    }

    EquationData equation_data;
    AssertThrow(pde_index < EquationData::n_variants,
                ExcMessage("This equation is not implemented."));
    equation_data.variant              = static_cast<EquationData::Variant>(pde_index);
    equation_data.local_solver_variant = (LocalSolverVariant)local_solver_index;
    equation_data.ksvd_tensor_indices  = [&]() -> std::set<unsigned int> {
      if(ksvd_rank == 1U)
        return {0U};
      else if(ksvd_rank == 2U)
        return {0U, 1U};
      else if(ksvd_rank == 12U)
        return {0U, 2U};
      else if(ksvd_rank == 3U)
        return {0U, 1U, 2U};
      else if(ksvd_rank == 4U)
        return {0U, 1U, 2U, 3U};
      else
        AssertThrow(false, ExcMessage("KSVD rank isn't supported."));
      return {};
    }();
    if(equation_data.ksvd_tensor_indices == std::set<unsigned int>{0U, 1U})
      equation_data.addition_to_min_eigenvalue = 0.025;
    equation_data.ip_factor            = ip_factor;
    equation_data.n_lanczos_iterations = 4 + 1;

    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0U;

    ModelProblem<dim, fe_degree> biharmonic_problem(prms, equation_data);

    std::fstream fout;
    const auto   filename = get_filename(prms, equation_data);
    fout.open(filename + ".log", std::ios_base::out);
    auto pcout               = std::make_shared<ConditionalOStream>(fout, is_first_proc);
    biharmonic_problem.pcout = pcout;

    if(use_hierarchical_elements)
      biharmonic_problem.finite_element = std::make_shared<FE_Q_Hierarchical<dim>>(fe_degree);
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
