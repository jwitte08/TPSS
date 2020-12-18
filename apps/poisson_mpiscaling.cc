
/*
 * test poisson problem
 *
 *  Created on: Sep 12, 2019
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include <ct_parameter.h>
#include "poisson_problem.h"

using namespace dealii;
using namespace Laplace;

struct Timings
{
  std::vector<Utilities::MPI::MinMaxAvg> setup;
  std::vector<Utilities::MPI::MinMaxAvg> apply;
};

struct TestParameter
{
  TPSS::PatchVariant                 patch_variant    = CT::PATCH_VARIANT_;
  TPSS::SmootherVariant              smoother_variant = CT::SMOOTHER_VARIANT_;
  std::string                        solver_variant   = "cg"; // see SolverSelector
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::Iterative;
  double   cg_reduction         = 1.e-8;
  double   coarse_grid_accuracy = 1.e-12;
  unsigned n_refinements        = 1;
  unsigned n_smoothing_steps    = 1;
  unsigned n_samples            = 10;
  unsigned n_subsamples_vmult   = 100;
  unsigned n_subsamples_smooth  = 20;
  unsigned n_subsamples_mg      = 10;
  unsigned test_variants        = 0;

  /*
   * variant_no: 0 = vmult
   *             1 = smooth
   *             2 = mg
   *             3 = solve
   */
  bool
  do_test_variant(unsigned variant_no) const
  {
    /// integer characterizes 4 test variants bitwise
    AssertThrow(test_variants < (1 << 4), ExcMessage("No valid choice of test_variants"));
    AssertIndexRange(variant_no, 4);
    std::bitset<4> flags(test_variants);
    // for (auto bit : flags)
    //   std::cout << bool_to_str(bit) << std::endl;
    return flags[variant_no];
  }

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("n_samples:", n_samples);
    oss << Util::parameter_to_fstring("n_subsamples_vmult:", n_subsamples_vmult);
    oss << Util::parameter_to_fstring("n_subsamples_smooth:", n_subsamples_smooth);
    oss << Util::parameter_to_fstring("n_subsamples_mg:", n_subsamples_mg);
    oss << Util::parameter_to_fstring(solver_variant + " solver reduction:", cg_reduction);
    return oss.str();
  }
};

template<int             dim,
         int             fe_degree,
         int             n_patch_dofs_1d = -1,
         TPSS::DoFLayout dof_layout      = TPSS::DoFLayout::DGQ>
struct Test
{
  using PoissonProblem =
    typename Poisson::ModelProblem<dim, fe_degree, dof_layout, double, n_patch_dofs_1d>;
  using VECTOR           = typename PoissonProblem::VECTOR;
  using SCHWARZ_SMOOTHER = typename PoissonProblem::SCHWARZ_SMOOTHER;
  using SYSTEM_MATRIX    = typename PoissonProblem::SYSTEM_MATRIX;
  using LEVEL_MATRIX     = typename PoissonProblem::LEVEL_MATRIX;

  const TestParameter   prms;
  RT::Parameter         rt_parameters;
  Laplace::EquationData equation_data;

  Test(const TestParameter & prms_in = TestParameter{}) : prms(prms_in)
  {
    //: discretization
    rt_parameters.n_cycles              = 1;
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = prms.n_refinements;
    rt_parameters.mesh.n_repetitions    = 2;

    //: solver
    rt_parameters.solver.variant              = prms.solver_variant;
    rt_parameters.solver.rel_tolerance        = prms.cg_reduction;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;

    //: multigrid
    const double damping_factor =
      TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
    rt_parameters.multigrid.coarse_level                 = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant   = prms.coarse_grid_variant;
    rt_parameters.multigrid.coarse_grid.iterative_solver = prms.solver_variant;
    rt_parameters.multigrid.coarse_grid.accuracy         = prms.coarse_grid_accuracy;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = prms.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = prms.smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = prms.n_smoothing_steps;
    rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(5, fe_degree + 1);
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

    // //: misc
    // // rt_parameters.compressed = prms.compressed;

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

  std::string
  write_timings_to_string(const Timings & timings, const PostProcessData & pp_data)
  {
    std::ostringstream oss;
    ConvergenceTable   timings_table;
    for(unsigned n = 0; n < timings.apply.size(); ++n)
    {
      timings_table.add_value("sample", n + 1);
      timings_table.add_value("procs", Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
      timings_table.add_value("dofs", pp_data.n_dofs_global.at(0));
      timings_table.add_value("setup (max)", timings.setup[n].max);
      timings_table.add_value("setup (min)", timings.setup[n].min);
      timings_table.add_value("setup (avg)", timings.setup[n].avg);
      timings_table.add_value("apply (max)", timings.apply[n].max);
      timings_table.add_value("apply (min)", timings.apply[n].min);
      timings_table.add_value("apply (avg)", timings.apply[n].avg);
    }

    timings_table.set_scientific("setup (max)", true);
    timings_table.set_scientific("setup (min)", true);
    timings_table.set_scientific("setup (avg)", true);
    timings_table.set_scientific("apply (max)", true);
    timings_table.set_scientific("apply (min)", true);
    timings_table.set_scientific("apply (avg)", true);
    timings_table.write_text(oss, TableHandler::TextOutputFormat::org_mode_table);

    return oss.str();
  }

  //: determine filename from test parameters
  std::string
  get_filename(const types::global_dof_index n_dofs_global)
  {
    std::ostringstream oss;
    const auto         n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const std::string  str_schwarz_variant =
      TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);

    oss << std::scientific << std::setprecision(2);
    oss << n_mpi_procs << "prcs";
    oss << "_" << TPSS::str_dof_layout(dof_layout);
    oss << "_" << str_schwarz_variant;
    oss << "_" << dim << "D";
    oss << "_" << fe_degree << "deg";
    oss << "_" << Util::si_metric_prefix(n_dofs_global) << "DoFs";
    return oss.str();
  }

  /*
   * The memory stats in KByte. VmPeak and VmSize determines the peak and
   * current virtual memory size (RAM + swap), whereas VmHWM and VmRSS
   * determines the peak and current resident memory size (RAM).
   */
  std::string
  str_memory_stats(std::string tag = "TBA")
  {
    std::ostringstream             oss;
    Utilities::System::MemoryStats memory_stats;
    Utilities::System::get_memory_stats(memory_stats);
    unsigned long long VmPeak_max =
      Utilities::MPI::max<unsigned long long>(memory_stats.VmPeak, MPI_COMM_WORLD);
    unsigned long long VmSize_max =
      Utilities::MPI::max<unsigned long long>(memory_stats.VmSize, MPI_COMM_WORLD);
    unsigned long long VmHWM_max =
      Utilities::MPI::max<unsigned long long>(memory_stats.VmHWM, MPI_COMM_WORLD);
    unsigned long long VmRSS_max =
      Utilities::MPI::max<unsigned long long>(memory_stats.VmRSS, MPI_COMM_WORLD);
    print_row(oss, 20, VmPeak_max, VmSize_max, VmHWM_max, VmRSS_max, "tag=" + tag);
    return oss.str();
  }

  /**
   * Test vmult, smooth, mg or solve depending on the @p test_variants flag in
   * TestParameters.
   */
  void
  partial()
  {
    if(prms.test_variants == 0) // nothing to test
      return;

    PoissonProblem poisson_problem{rt_parameters, equation_data};
    poisson_problem.print_informations();
    poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
    poisson_problem.distribute_dofs();
    Timer      time(MPI_COMM_WORLD, true);
    const bool is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    const auto pcout         = ConditionalOStream(std::cout, is_first_proc);

    if(prms.do_test_variant(0))
    {
      pcout << Util::parameter_to_fstring("Testing vmult()", "");
      time.start();
      vmult_impl(poisson_problem);
      time.stop();
      time.print_last_lap_wall_time_data(pcout);
      pcout << std::endl;
    }

    if(prms.do_test_variant(1))
    {
      pcout << Util::parameter_to_fstring("Testing smooth()", "");
      time.start();
      smooth_impl(poisson_problem);
      time.stop();
      time.print_last_lap_wall_time_data(pcout);
      pcout << std::endl;
    }

    if(!prms.do_test_variant(2) && !prms.do_test_variant(3))
      return;
    poisson_problem.prepare_linear_system();

    if(prms.do_test_variant(2))
    {
      pcout << Util::parameter_to_fstring("Testing mg()", "");
      time.start();
      mg_impl(poisson_problem);
      time.stop();
      time.print_last_lap_wall_time_data(pcout);
      pcout << std::endl;
    }

    if(prms.do_test_variant(3))
    {
      pcout << Util::parameter_to_fstring("Testing solve()", "");
      time.start();
      solve_impl(poisson_problem);
      time.stop();
      time.print_last_lap_wall_time_data(pcout);
      pcout << std::endl;
    }
  }

  // // DEBUG
  // void
  // vmult_raw()
  // {
  //   PoissonProblem poisson_problem{rt_parameters};
  //   auto &         pcout = *(poisson_problem.pcout);
  //   poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
  //   poisson_problem.distribute_dofs();
  //   for(unsigned sample = 0; sample < prms.n_samples; ++sample)
  //   {
  //     const auto    mf_storage = poisson_problem.template build_mf_storage<double>();
  //     SYSTEM_MATRIX system_matrix;
  //     system_matrix.initialize(mf_storage);
  //     //: vmult
  //     VECTOR tmp;
  //     mf_storage->initialize_dof_vector(tmp);
  //     fill_with_random_values(tmp);
  //     VECTOR dst = tmp;
  //     system_matrix.vmult(dst, tmp);
  //     print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
  //     pcout << str_memory_stats("vmult_raw");
  //   }
  // }

  void
  vmult_impl(const PoissonProblem & poisson_problem)
  {
    Timings                 timings_vmult;
    auto &                  pcout         = *(poisson_problem.pcout);
    types::global_dof_index n_dofs_global = 0;
    for(unsigned sample = 0; sample < prms.n_samples; ++sample)
    {
      Timer time(MPI_COMM_WORLD, true);
      //:setup
      time.restart();
      const auto    mf_storage = poisson_problem.template build_mf_storage<double>();
      SYSTEM_MATRIX system_matrix;
      system_matrix.initialize(mf_storage, poisson_problem.equation_data);
      n_dofs_global = system_matrix.m();
      time.stop();
      timings_vmult.setup.push_back(time.get_last_lap_wall_time_data());

      //: vmult
      VECTOR tmp;
      mf_storage->initialize_dof_vector(tmp);
      fill_with_random_values(tmp);
      VECTOR dst = tmp;
      time.restart();
      for(unsigned subsample = 0; subsample < prms.n_subsamples_vmult; ++subsample)
        system_matrix.vmult(dst, tmp);
      print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
      pcout << str_memory_stats("vmult");
      time.stop();
      Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
      t_apply                           = t_apply / prms.n_subsamples_vmult;
      timings_vmult.apply.push_back(t_apply);
    }

    //: write performance timings
    const std::string filename = get_filename(n_dofs_global);
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::fstream      fstream;
      PostProcessData & pp_data = poisson_problem.pp_data;
      pp_data.n_dofs_global.push_back(n_dofs_global);

      fstream.open("vmult_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_vmult, pp_data);
      fstream.close();
    }
  }

  void
  vmult()
  {
    PoissonProblem poisson_problem{rt_parameters, equation_data};
    poisson_problem.print_informations();
    poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
    poisson_problem.distribute_dofs();
    vmult_impl(poisson_problem);
  }

  // // DEBUG
  // void
  // smooth_raw()
  // {
  //   PoissonProblem poisson_problem{rt_parameters};
  //   auto &         pcout = *(poisson_problem.pcout);
  //   poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
  //   poisson_problem.distribute_dofs();
  //   for(unsigned sample = 0; sample < prms.n_samples; ++sample)
  //   {
  //     //: setup
  //     const unsigned fine_level = poisson_problem.triangulation.n_global_levels() - 1;
  //     const auto     mf_storage = poisson_problem.template build_mf_storage<double>(fine_level);
  //     LEVEL_MATRIX   level_matrix;
  //     level_matrix.initialize(mf_storage);
  //     const auto   subdomain_handler = poisson_problem.build_patch_storage(fine_level,
  //     mf_storage); const auto & schwarz_data      =
  //     poisson_problem.rt_parameters.multigrid.pre_smoother.schwarz; const auto
  //     schwarz_preconditioner =
  //       poisson_problem.build_schwarz_preconditioner(subdomain_handler, level_matrix,
  //       schwarz_data);
  //     typename SCHWARZ_SMOOTHER::AdditionalData smoother_data;
  //     SCHWARZ_SMOOTHER schwarz_smoother;
  //     schwarz_smoother.initialize(level_matrix, schwarz_preconditioner, smoother_data);

  //     //: smooth
  //     VECTOR tmp;
  //     mf_storage->initialize_dof_vector(tmp);
  //     fill_with_random_values(tmp);
  //     VECTOR dst = tmp;
  //     schwarz_smoother.step(dst, tmp);
  //     print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
  //     pcout << str_memory_stats("smooth");
  //   }
  // }

  void
  smooth_impl(PoissonProblem & poisson_problem)
  {
    Timings                 timings_smooth;
    Timer                   time(MPI_COMM_WORLD, true);
    auto &                  pcout         = *(poisson_problem.pcout);
    types::global_dof_index n_dofs_global = 0;
    for(unsigned sample = 0; sample < prms.n_samples; ++sample)
    {
      //: setup
      time.restart();
      const unsigned fine_level = poisson_problem.triangulation.n_global_levels() - 1;
      const auto     mf_storage = poisson_problem.template build_mf_storage<double>(fine_level);
      LEVEL_MATRIX   level_matrix;
      level_matrix.initialize(mf_storage,
                              *(poisson_problem.mg_constrained_dofs),
                              poisson_problem.equation_data);
      n_dofs_global                  = level_matrix.m();
      const auto   subdomain_handler = poisson_problem.build_patch_storage(fine_level, mf_storage);
      const auto & schwarz_data      = poisson_problem.rt_parameters.multigrid.pre_smoother.schwarz;
      const auto   schwarz_preconditioner =
        poisson_problem.build_schwarz_preconditioner(subdomain_handler, level_matrix, schwarz_data);
      typename SCHWARZ_SMOOTHER::AdditionalData smoother_data;
      SCHWARZ_SMOOTHER                          schwarz_smoother;
      schwarz_smoother.initialize(level_matrix, schwarz_preconditioner, smoother_data);
      time.stop();
      timings_smooth.setup.push_back(time.get_last_lap_wall_time_data());

      //: smooth
      VECTOR tmp;
      mf_storage->initialize_dof_vector(tmp);
      fill_with_random_values(tmp);
      VECTOR dst = tmp;
      time.restart();
      for(unsigned subsample = 0; subsample < prms.n_subsamples_smooth; ++subsample)
        schwarz_smoother.step(dst, tmp);
      time.stop();

      //: post process
      Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
      t_apply                           = t_apply / prms.n_subsamples_smooth;
      timings_smooth.apply.push_back(t_apply);
      print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
      pcout << str_memory_stats("smooth");
      const auto time_data = schwarz_preconditioner->get_time_data();
      for(const auto & time_info : time_data)
        pcout << Util::parameter_to_fstring(time_info.description,
                                            Utilities::MPI::max(time_info.time, MPI_COMM_WORLD));
    }

    //: write performance timings
    const std::string filename = get_filename(n_dofs_global);
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::fstream      fstream;
      PostProcessData & pp_data = poisson_problem.pp_data;
      pp_data.n_dofs_global.push_back(n_dofs_global);

      fstream.open("smooth_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_smooth, pp_data);
      fstream.close();
    }
  }

  void
  smooth()
  {
    PoissonProblem poisson_problem{rt_parameters, equation_data};
    poisson_problem.print_informations();
    poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
    poisson_problem.distribute_dofs();
    smooth_impl(poisson_problem);
  }

  // // DEBUG
  // void
  // mg_raw()
  // {
  //   PoissonProblem poisson_problem{rt_parameters};
  //   auto &         pcout = *(poisson_problem.pcout);
  //   poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
  //   poisson_problem.distribute_dofs();
  //   poisson_problem.prepare_linear_system();
  //   for(unsigned sample = 0; sample < prms.n_samples; ++sample)
  //   {
  //     //: setup
  //     const auto & gmg_preconditioner = poisson_problem.prepare_preconditioner_mg();

  //     //: V-cycle
  //     VECTOR tmp;
  //     poisson_problem.system_matrix.get_matrix_free()->initialize_dof_vector(tmp);
  //     fill_with_random_values(tmp);
  //     VECTOR dst = tmp;
  //     gmg_preconditioner.vmult(dst, tmp);
  //     print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
  //     pcout << str_memory_stats("mg");
  //   }
  // }

  void
  mg_impl(PoissonProblem & poisson_problem)
  {
    Timings timings_mg;
    Timer   time(MPI_COMM_WORLD, true);
    auto &  pcout = *(poisson_problem.pcout);

    for(unsigned sample = 0; sample < prms.n_samples; ++sample)
    {
      //: setup
      time.restart();
      const auto & gmg_preconditioner = poisson_problem.prepare_preconditioner_mg();
      time.stop();
      timings_mg.setup.push_back(time.get_last_lap_wall_time_data());

      //: V-cycle
      VECTOR tmp;
      poisson_problem.system_matrix.get_matrix_free()->initialize_dof_vector(tmp);
      fill_with_random_values(tmp);
      VECTOR dst = tmp;
      time.restart();
      for(unsigned subsample = 0; subsample < prms.n_subsamples_mg; ++subsample)
        gmg_preconditioner.vmult(dst, tmp);
      time.stop();
      Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
      t_apply                           = t_apply / prms.n_subsamples_mg;
      timings_mg.apply.push_back(t_apply);
      print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
      pcout << str_memory_stats("mg");
    }

    //: write performance timings
    const types::global_dof_index n_dofs_global = poisson_problem.pp_data.n_dofs_global.front();
    const std::string             filename      = get_filename(n_dofs_global);
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::fstream            fstream;
      const PostProcessData & pp_data = poisson_problem.pp_data;
      fstream.open("mg_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_mg, pp_data);
      fstream.close();
    }
  }

  void
  mg()
  {
    PoissonProblem poisson_problem{rt_parameters, equation_data};
    poisson_problem.print_informations();
    poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
    poisson_problem.distribute_dofs();
    poisson_problem.prepare_linear_system();
    mg_impl(poisson_problem);
  }

  // // DEBUG
  // void
  // poisson_run_raw()
  // {
  //   PoissonProblem poisson_problem{rt_parameters};

  //   for(unsigned sample = 0; sample < prms.n_samples; ++sample)
  //   {
  //     poisson_problem.run();
  //     auto & pcout = *(poisson_problem.pcout);
  //     print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
  //     pcout << str_memory_stats("run");
  //   }
  // }


  void
  poisson_run(const bool once = true)
  {
    PoissonProblem     poisson_problem{rt_parameters, equation_data};
    std::ostringstream oss;
    const bool         is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    poisson_problem.pcout            = std::make_shared<ConditionalOStream>(oss, is_first_proc);
    oss << write_header();

    ConditionalOStream pcout(std::cout, is_first_proc);
    poisson_problem.run();
    print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
    pcout << str_memory_stats("run");
    const auto pp_data = poisson_problem.pp_data;
    oss << write_ppdata_to_string(pp_data);

    if(is_first_proc)
    {
      const types::global_dof_index n_dofs_global = poisson_problem.pp_data.n_dofs_global.front();
      const std::string             filename      = get_filename(n_dofs_global);
      std::fstream                  fstream;
      fstream.open("poisson_" + filename + ".log", std::ios_base::out);
      fstream << oss.str();
      fstream.close();
    }

    if(!once)
      for(unsigned sample = 1; sample < prms.n_samples; ++sample)
      {
        poisson_problem.run();
        print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
        pcout << str_memory_stats("run");
      }
  }

  void
  solve_impl(PoissonProblem & poisson_problem)
  {
    Timer                   time(MPI_COMM_WORLD, true);
    Timings                 timings_solve;
    auto &                  pcout              = *(poisson_problem.pcout);
    types::global_dof_index n_dofs_global      = 0;
    const auto &            gmg_preconditioner = poisson_problem.prepare_preconditioner_mg();
    for(unsigned sample = 0; sample < prms.n_samples; ++sample)
    {
      //: setup
      n_dofs_global = poisson_problem.system_matrix.m();
      time.restart(); // DUMMY
      time.stop();
      Utilities::MPI::MinMaxAvg t_setup = time.get_last_lap_wall_time_data();
      timings_solve.setup.push_back(t_setup);

      //: solve
      poisson_problem.prepare_linear_system(); // reinits solution and rhs vector
      time.restart();
      poisson_problem.solve(gmg_preconditioner);
      print_row(pcout, 20, "VmPeak", "VmSize", "VmHWM", "VmRSS");
      pcout << str_memory_stats("solve");
      time.stop();
      Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
      timings_solve.apply.push_back(t_apply);
    }

    //: write performance timings
    const std::string filename = get_filename(n_dofs_global);
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::fstream      fstream;
      PostProcessData & pp_data = poisson_problem.pp_data;
      pp_data.n_dofs_global.push_back(n_dofs_global);

      fstream.open("solve_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_solve, pp_data);
      fstream.close();
    }
  }

  void
  solve()
  {
    PoissonProblem poisson_problem{rt_parameters, equation_data};
    poisson_problem.print_informations();
    poisson_problem.create_triangulation(rt_parameters.mesh.n_refinements);
    poisson_problem.distribute_dofs();
    poisson_problem.prepare_linear_system();
    solve_impl(poisson_problem);
  }
};


int
main(int argc, char * argv[])
{
  // deallog.depth_console(3);

  // *** init TBB and MPI
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // *** run-time options
  TestParameter prms;
  if(argc > 1)
    prms.n_refinements = std::atoi(argv[1]);
  if(argc > 2)
    prms.test_variants = std::atoi(argv[2]);
  if(argc > 3)
    prms.n_samples = std::atoi(argv[3]);

  // *** run tests
  constexpr int  dim        = CT::DIMENSION_;
  constexpr int  fe_degree  = CT::FE_DEGREE_;
  constexpr auto dof_layout = CT::DOF_LAYOUT_;
  constexpr int  n_patch_dofs_1d_static =
    TPSS::UniversalInfo<dim>::n_dofs_1d(CT::PATCH_VARIANT_, dof_layout, fe_degree);
  const bool is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  const auto pcout         = ConditionalOStream(std::cout, is_first_proc);
  Test<dim, fe_degree, n_patch_dofs_1d_static, dof_layout> tester(prms);

  Timer time(MPI_COMM_WORLD, true);
  time.start();
  tester.partial();
  time.stop();

  // if(prms.do_test_variant(0)) // vmult
  // {
  //   // time.start(); // checked: no leak!
  //   // tester.vmult_raw();
  //   // time.stop();

  //   time.start();
  //   tester.vmult();
  //   time.stop();
  //   pcout << Util::parameter_to_fstring("Testing vmult()", "");
  //   time.print_last_lap_wall_time_data(pcout);
  //   pcout << std::endl;
  // }

  // if(prms.do_test_variant(1)) // smooth
  // {
  //   // time.start(); // checked: no leak!
  //   // tester.smooth_raw();
  //   // time.stop();

  //   time.start();
  //   tester.smooth();
  //   time.stop();
  //   pcout << Util::parameter_to_fstring("Testing smooth()", "");
  //   time.print_last_lap_wall_time_data(pcout);
  //   pcout << std::endl;
  // }

  // if(prms.do_test_variant(2)) // mg
  // {
  //   // time.start();
  //   // tester.mg_raw();
  //   // time.stop();

  //   time.start();
  //   tester.mg(); // checked: leaks!
  //   time.stop();
  //   pcout << Util::parameter_to_fstring("Testing mg()", "");
  //   time.print_last_lap_wall_time_data(pcout);
  //   pcout << std::endl;
  // }

  // if(prms.do_test_variant(3)) // solve
  // {
  //   // time.start(); // checked: leaks!
  //   // tester.poisson_run_raw();
  //   // time.stop();

  //   time.start();
  //   tester.poisson_run(/*compute once? (otherwise N samples)*/ false); // checked: leaks!
  //   time.stop();
  //   pcout << Util::parameter_to_fstring("Testing PoissonProblem::run()", "");
  //   time.print_last_lap_wall_time_data(pcout);
  //   pcout << std::endl;
  // }

  pcout << Util::parameter_to_fstring("Total wall time elapsed", "");
  time.print_accumulated_wall_time_data(pcout);
  pcout << std::endl;

  return 0;
}
