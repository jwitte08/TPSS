
/*
 * Numerical Experiments: Fast Diagonal Schwarz Smoothers (Poisson)
 *                        using shared-memory parallelism (TBB)
 *
 *  Created on: Jan 24, 2019
 *      Author: witte
 */

#include "Plaplace_fdss.h"

using namespace dealii;
using namespace Laplace;

template<int dim, int fe_degree, typename Number = double, int n_dofs_per_patch_1d = -1>
struct Test
{
  Test()
    : pcout_owned(GlobParam::use_filestream ?
                    std::make_shared<ConditionalOStream>(fstream,
                                                         Utilities::MPI::this_mpi_process(
                                                           MPI_COMM_WORLD) == 0) :
                    std::make_shared<ConditionalOStream>(std::cout,
                                                         Utilities::MPI::this_mpi_process(
                                                           MPI_COMM_WORLD) == 0)),
      pcout(*pcout_owned)
  {
    // *** set common parameters
    // smoother
    parameters.schwarz_smoother_data.patch_variant    = GlobParam::patch_variant;
    parameters.schwarz_smoother_data.smoother_variant = GlobParam::smoother_variant;
    parameters.schwarz_smoother_data.manual_coloring  = true;
    parameters.n_threads                              = MultithreadInfo::n_threads();
    // mesh hierarchy + discretization
    const auto refine_pair        = GlobParam::lookup_refine_pair(dim, fe_degree);
    parameters.n_refines          = refine_pair.first;
    parameters.n_cell_repetitions = refine_pair.second;
    IP::pre_factor                = 1.;
    // multigrid
    parameters.mg_coarse_grid_variant        = Parameter::CoarseGridVariant::ChebyshevAccurate;
    parameters.mg_coarse_chebyshev_reduction = 1.e-4;
    // solver
    parameters.n_cycles           = 1;
    parameters.solver_variant     = GlobParam::lookup_solver_variant();
    parameters.mf_tasks_scheme_id = GlobParam::mf_tasks_scheme_id;
    // write output
    parameters.write_fe_output    = false;
    parameters.write_timer_output = false;

    // *** check setup
    AssertThrow(parameters.n_cell_repetitions % 2 == 1, ExcMessage("Cell repetition is not odd!"));
  }

  void
  print_generic_footer()
  {
    pcout << "============================================================" << std::endl;
    fstream.close();
  }

  void
  fprintf_times_of_sample(const std::string                          filename,
                          const std::vector<std::vector<TimeInfo>> & timings)
  {
    std::fstream fstream_sample;
    fstream_sample.open(filename, std::ios_base::app);
    const bool         pcondition = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    ConditionalOStream pcout_sample(fstream_sample, pcondition);
    TimeStat::fprint_sample(pcout_sample, timings);
    pcout_sample << std::endl;
    fstream_sample.close();
  }

  void
  multigrid_timings()
  {
    constexpr unsigned n_loops = GlobParam::n_loops_warmup + GlobParam::n_loops;
    if(parameters.n_cell_repetitions == 1)
      parameters.n_cell_repetitions = 2;

    // *** pre-processing
    std::string str_prefix = "TBB_mg_";
    std::string str_suffix =
      "_MDOF" + Utilities::to_string(static_cast<unsigned int>(GlobParam::n_dofs_minimal / 1e+6)) +
      ".txt";
    const std::string filename =
      GlobParam::output_directory + "/tbb/" + GlobParam::str_append_fname(str_prefix) + str_suffix;
    fstream.open(filename, std::ios_base::app);
    if(GlobParam::use_filestream && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "\n\n ... writing output to " << filename << "\n\n";
    GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);
    const std::string str_scheme[]  = {"none", "partition_partition", "partition_color", "color"};
    const std::string actual_scheme = str_scheme[GlobParam::mf_tasks_scheme_id];
    print_row_variable(pcout, 2, "", 43, "Matrix-free parallel scheme:", actual_scheme);
    pcout << std::endl;

    pcout << "Starting test run ... \n\n";
    MatrixOperator<dim, fe_degree, Number, n_dofs_per_patch_1d> laplace_problem{pcout, parameters};

    std::vector<std::vector<TimeInfo>> timings; // time section : loop
    std::vector<unsigned int>          timings_partition;
    Timer                              timer;
    timer.restart();
    for(unsigned loop_id = 0; loop_id < n_loops; ++loop_id)
    {
      laplace_problem.pcout.set_condition(loop_id == 0); // print only on first run
      // *** run the laplace problem
      const auto timings_run_vec = laplace_problem.run(); // returns TimeInfo collection per cycle
      AssertThrow(timings_run_vec.size() == 1, ExcMessage("Do only 1 cycle."));
      // *** store time information
      const auto timings_run = timings_run_vec.back();
      if(timings.empty())
      {
        timings.resize(timings_run.size());
        timings_partition = laplace_problem.get_timings_partition();
      }
      AssertThrow(timings_run.size() == timings.size(),
                  ExcMessage("Mismatching number of timings."));
      auto input = timings_run.cbegin();
      for(auto tvec = timings.begin(); tvec != timings.end(); ++tvec, ++input)
        tvec->emplace_back(*input);
    }
    pcout << " ... successful completion within " << timer.wall_time() << "[s] wall time. \n\n";

    // *** post process
    for(auto & timings_section : timings)
      timings_section.erase(timings_section.begin(),
                            std::next(timings_section.begin(), GlobParam::n_loops_warmup));
    pcout.set_condition(true);
    std::vector<TimeInfo> timings_mean;
    for(const auto & timings_section : timings)
      timings_mean.emplace_back(TimeStat::mean(timings_section));

    const std::string partition_labels[] = {"Average time (setup):", "Average time (solve):"};
    const auto        n_partitions       = timings_partition.size() - 1;
    for(unsigned part = 0; part < n_partitions; ++part)
    {
      pcout << partition_labels[part] << std::endl;
      for(unsigned sect = timings_partition[part]; sect < timings_partition[part + 1]; ++sect)
        GlobParam::fprint_time_info(pcout, timings_mean[sect]);
      pcout << std::endl;
    }

    // *** post process - all samples
    if(GlobParam::use_filestream)
      fprintf_times_of_sample(filename + ".sample", timings);

    print_generic_footer();
  }

  void
  smoother_timings(const unsigned int n_refines = -1, const unsigned int n_repetitions = -1)
  {
    using VECTOR       = LinearAlgebra::distributed::Vector<Number>;
    using LEVEL_MATRIX = CombinedOperator<dim, fe_degree, Number>;
    using PATCH_MATRIX =
      TensorProductMatrixSymmetricSum<dim, VectorizedArray<Number>, n_dofs_per_patch_1d>;
    using SCHWARZ_PRECONDITIONER = SchwarzPreconditioner<dim, LEVEL_MATRIX, VECTOR, PATCH_MATRIX>;
    using SMOOTHER = SchwarzSmoother<dim, LEVEL_MATRIX, SCHWARZ_PRECONDITIONER, VECTOR>;

    Timer total_time;
    total_time.restart();

    // *** pre process
    constexpr unsigned n_loops = GlobParam::n_loops_warmup + GlobParam::n_loops;
    if(n_refines != static_cast<unsigned int>(-1))
      parameters.n_refines = n_refines;
    if(n_repetitions != static_cast<unsigned int>(-1))
      parameters.n_cell_repetitions = n_repetitions;
    parameters.allow_one_level_only = true;
    const auto patch_variant        = parameters.schwarz_smoother_data.patch_variant;
    const auto smoother_variant     = parameters.schwarz_smoother_data.smoother_variant;
    parameters.schwarz_smoother_data.damping_factor =
      TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);

    MatrixOperator<dim, fe_degree, Number, n_dofs_per_patch_1d> laplace_problem{pcout, parameters};
    laplace_problem.distribute_dofs();

    // *** print information
    std::string str_prefix = "TBB_smooth_";
    std::string str_suffix =
      "_MDOF" + Utilities::to_string(static_cast<unsigned int>(GlobParam::n_dofs_minimal / 1e+6)) +
      ".txt";
    if(GlobParam::test_variant == GlobParam::TestVariant::SmoothTimeComplexity)
    {
      str_prefix = "TBBcomplexity_smooth_";
      str_suffix = ".txt";
    }
    else if(GlobParam::test_variant == GlobParam::TestVariant::SmoothTimeStrong)
    {
      str_prefix             = "TBBstrong_smooth_";
      const auto n_dofs_fine = laplace_problem.dof_handler.n_dofs(laplace_problem.global_level);
      str_suffix =
        "_MDOF" + Utilities::to_string(static_cast<unsigned int>(n_dofs_fine / 1e+6)) + ".txt";
    }
    const std::string filename =
      GlobParam::output_directory + "/tbb/" + GlobParam::str_append_fname(str_prefix) + str_suffix;
    fstream.open(filename, std::ios_base::app);
    if(GlobParam::use_filestream && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "\n\n ... writing output to " << filename << "\n\n";
    GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);
    const std::string str_scheme[]  = {"none", "partition_partition", "partition_color", "color"};
    const std::string actual_scheme = str_scheme[GlobParam::mf_tasks_scheme_id];
    print_row_variable(pcout, 2, "", 43, "Matrix-free parallel scheme:", actual_scheme);
    pcout << std::endl;
    parameters.schwarz_smoother_data.print(pcout);
    pcout << std::endl;
    laplace_problem.print_mesh_info();
    laplace_problem.print_dof_info();

    pcout << "Starting test run ... \n\n";
    const unsigned            level = laplace_problem.global_level;
    std::vector<unsigned int> timings_partition;
    timings_partition.push_back(0);
    std::vector<std::vector<TimeInfo>> time_infos; // time_section : loop
    time_infos.resize(16);
    for(unsigned loop_id = 0; loop_id < n_loops; ++loop_id)
    {
      Timer timer;
      auto  time_section = time_infos.begin();

      // *** setup matrix-free operator
      timer.restart();
      const auto   mf_storage_level = laplace_problem.template build_mf_storage<Number>(level);
      LEVEL_MATRIX level_matrix;
      level_matrix.initialize(mf_storage_level);
      time_section->emplace_back(timer.wall_time(), "Setup matrix-free operator:", "[s]");
      ++time_section;

      // *** setup Schwarz smoother (compute inverses)
      timer.restart();
      const auto patch_storage =
        laplace_problem.template build_patch_storage<Number>(level, level_matrix.get_matrix_free());
      typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
      precondition_data.relaxation      = parameters.schwarz_smoother_data.damping_factor;
      const auto schwarz_preconditioner = std::make_shared<SCHWARZ_PRECONDITIONER>();
      schwarz_preconditioner->initialize(patch_storage, level_matrix, precondition_data);
      typename SMOOTHER::AdditionalData smoother_data;
      smoother_data.number_of_smoothing_steps =
        parameters.schwarz_smoother_data.number_of_smoothing_steps;
      SMOOTHER smoother;
      smoother.initialize(level_matrix, schwarz_preconditioner, smoother_data);
      time_section->emplace_back(timer.wall_time(), "Setup Schwarz smoother:", "[s]");
      ++time_section;
      const auto preconditioner_timings_setup = schwarz_preconditioner->get_time_data();
      for(const auto & info : preconditioner_timings_setup)
        if(info.is_active())
          (time_section++)->emplace_back(info);
      if(loop_id == 0) // END setup section
        timings_partition.push_back(std::distance(time_infos.begin(), time_section));

      // *** initialize global vectors
      LinearAlgebra::distributed::Vector<Number> dst;
      LinearAlgebra::distributed::Vector<Number> src_memory;
      level_matrix.get_matrix_free()->initialize_dof_vector(dst);
      level_matrix.get_matrix_free()->initialize_dof_vector(src_memory);
      for(auto it = dst.begin(); it != dst.end(); ++it)
        *it = (Number)rand() / RAND_MAX;
      const LinearAlgebra::distributed::Vector<Number> & src{src_memory};

      // *** read-write operation (read dst and write non-trivial expression into src_memory)
      timer.restart();
      auto     it_src_memory = src_memory.begin();
      unsigned temp_integer  = 1;
      for(auto it = dst.begin(); it != dst.end(); ++it, ++it_src_memory)
        *it_src_memory = *it * 1.234 * static_cast<Number>(temp_integer);
      time_section->emplace_back(timer.wall_time(), "Read-write operation:", "[s]");
      ++time_section;

      // *** apply matrix-free operator
      timer.restart();
      level_matrix.vmult(dst, src);
      time_section->emplace_back(timer.wall_time(), "Apply matrix-free operator", "[s]");
      ++time_section;

      // *** apply tensor product Schwarz smoother
      timer.restart();
      smoother.step(dst, src); // or vmult performs additional dst = 0.
      time_section->emplace_back(timer.wall_time(), "Apply smoother operator", "[s]");
      ++time_section;
      const auto preconditioner_timings_apply = schwarz_preconditioner->get_time_data();
      auto       timings_before               = preconditioner_timings_setup.begin();
      for(const auto & info : preconditioner_timings_apply)
        if(!(timings_before++)->is_active())
          if(info.is_active())
            (time_section++)->emplace_back(info);
      if(loop_id == 0) // END apply section
        timings_partition.push_back(std::distance(time_infos.begin(), time_section));
    }
    pcout << " ... successful completion within " << total_time.wall_time()
          << "[s] wall time. \n\n";

    // *** post process
    time_infos.resize(timings_partition.back()); // cut empty sections
    for(auto & timings_section : time_infos)
      timings_section.erase(timings_section.cbegin(),
                            timings_section.cbegin() + GlobParam::n_loops_warmup);
    if(time_infos.front().empty())
      return;
    std::vector<TimeInfo> timings_mean;
    for(const auto & timings_section : time_infos)
      if(!timings_section.empty())
        timings_mean.emplace_back(TimeStat::mean(timings_section));
    const std::string partition_labels[] = {"Average time (setup):", "Average time (apply):"};
    const auto        n_partitions       = timings_partition.size() - 1;
    for(unsigned part = 0; part < n_partitions; ++part)
    {
      pcout << partition_labels[part] << std::endl;
      for(unsigned sect = timings_partition[part]; sect < timings_partition[part + 1]; ++sect)
        GlobParam::fprint_time_info(pcout, timings_mean[sect]);
      pcout << std::endl;
    }

    const auto mf_storage = laplace_problem.template build_mf_storage<Number>(level);
    const auto patch_storage =
      laplace_problem.template build_patch_storage<Number>(level, mf_storage);
    GlobParam::template fprint_misc<dim, Number>(pcout, mf_storage, patch_storage);

    // *** post process - all sample
    if(GlobParam::use_filestream)
      fprintf_times_of_sample(filename + ".sample", time_infos);

    print_generic_footer();
  }

  std::shared_ptr<ConditionalOStream> pcout_owned;
  std::fstream                        fstream;
  ConditionalOStream &                pcout;
  Parameter                           parameters;
};


int
main(int argc, char * argv[])
{
  deallog.depth_console(0);

  // *** init TBB and MPI
  const unsigned int arg_max_threads =
    argc > 1 ? std::atoi(argv[1]) : numbers::invalid_unsigned_int;
  const unsigned int max_threads =
    arg_max_threads > 0 ? arg_max_threads : numbers::invalid_unsigned_int;
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static TEST parameters
  using value_type             = GlobParam::value_type;
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;
  constexpr int      n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(GlobParam::patch_variant) * (fe_degree + 1);

  // *** choose a TEST
  Test<dim, fe_degree, value_type, n_patch_dofs_per_direction> laplace_test;
  switch(GlobParam::test_variant)
  {
    case GlobParam::TestVariant::SmoothTime:
    {
      laplace_test.smoother_timings();
      break;
    }
    case GlobParam::TestVariant::SmoothTimeStrong:
    {
      const unsigned int n_refines     = argc > 2 ? std::atoi(argv[2]) : -1;
      const unsigned int n_repetitions = argc > 3 ? std::atoi(argv[3]) : -1;
      laplace_test.smoother_timings(n_refines, n_repetitions);
      break;
    }
    case GlobParam::TestVariant::SmoothTimeComplexity:
    {
      const auto refine_pair     = GlobParam::lookup_refine_pair(dim, fe_degree);
      const auto n_refines_start = refine_pair.first >= 4 ? refine_pair.first - 4 : 0;
      for(unsigned int n_refines = n_refines_start; n_refines <= refine_pair.first; ++n_refines)
        laplace_test.smoother_timings(n_refines);
      break;
    }
    case GlobParam::TestVariant::MGTimeSameDoFs:
    {
      laplace_test.multigrid_timings();
      break;
    }
    default: // case GlobParam::TestVariant::NoTest:
    {
      std::cout << "Please choose a (valid) test." << std::endl;
      break;
    }
  }

  return 0;
}
