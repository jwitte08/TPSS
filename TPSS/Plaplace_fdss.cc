
/*
 * Numerical Experiments: Fast Diagonal Schwarz Smoothers (Poisson)
 *
 *  Created on: Aug 15, 2018
 *      Author: witte
 */

#include "Plaplace_fdss.h"

using namespace dealii;
using namespace Laplace;

template<int dim, int fe_degree, typename Number = double, int n_dofs = -1>
struct Test
{
  enum class SmoothRoutine
  {
    MG_SMOOTHER,
    PREC_VMULT
  };

  Test(ConditionalOStream & pcout_in, const Parameter & parameter_in)
    : parameters(parameter_in),
      pcout(pcout_in),
      time_details(pcout.get_stream(), Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      setup_time(0.),
      apply_time(0.),
      setup_time_misc(0.),
      apply_time_misc(0.),
      read_write_time(0.),
      n_dofs_level(0)
  {
  }

  void
  fprint_time_details(const std::string & description,
                      const double        time_value,
                      const std::string   unit = "[s]")
  {
    std::ostringstream ostream;
    ostream << std::scientific << std::setprecision(3) << time_value << " " << unit;
    auto && ftime = ostream.str();
    print_row_variable(time_details, 2, "T", 43, description, ftime);
  }

  void
  fprint_misc(const std::shared_ptr<const MatrixFree<dim, Number>>       mf_storage,
              const std::shared_ptr<const SubdomainHandler<dim, Number>> patch_storage,
              const unsigned                                             level)
  {
    const auto &   dof_handler        = mf_storage->get_dof_handler();
    const unsigned n_subdomains       = patch_storage->n_physical_subdomains();
    const unsigned n_macro_subdomains = patch_storage->get_partition_data().n_subdomains();

    time_details << "Miscellaneous:" << std::endl;
    print_row_variable(
      time_details, 2, "", 43, "Number of degrees of freedom:", dof_handler.n_dofs(level));
    print_row_variable(
      time_details, 2, "", 43, "Number of cells:", dof_handler.get_triangulation().n_cells(level));

    print_row_variable(time_details, 2, "", 43, "Number of physical subdomains:", n_subdomains);
    print_row_variable(time_details, 2, "", 43, "Number of macro subdomains:", n_macro_subdomains);
    print_row_variable(
      time_details, 2, "", 43, "Number of macro cells:", mf_storage->n_cell_batches());
    print_row_variable(time_details,
                       2,
                       "",
                       43,
                       "Number of interior macro faces:",
                       mf_storage->n_inner_face_batches());
    print_row_variable(time_details,
                       2,
                       "",
                       43,
                       "Number of boundary macro faces:",
                       mf_storage->n_boundary_face_batches());
    time_details << std::endl;
  }

  void
  test_matrix_free_vmult(const unsigned n_loops,
                         const unsigned n_refines   = 2,
                         const unsigned n_cell_reps = 1)
  {
    using SYSTEM_MATRIX = MF::Operator<dim, fe_degree, Number>;

    Timer total_time;
    total_time.restart();

    // *** pre process
    AssertThrow(n_cell_reps % 2 == 1, ExcMessage("Cell repetition is not odd!"));
    parameters.n_refines          = n_refines;   // initial fine level
    parameters.n_cell_repetitions = n_cell_reps; // # of coarse cells per dimension
    MatrixOperator<dim, fe_degree, Number, n_dofs> laplace_problem{pcout, parameters};
    laplace_problem.distribute_dofs();
    setup_time = 0.;
    apply_time = 0.;

    laplace_problem.print_dof_info();
    print_row_variable(pcout, 45, "Tested routine:", "MatrixFreeOperator::vmult");
    print_row_variable(pcout, 45, "Finite element:", laplace_problem.fe->get_name());
    pcout << std::endl;
    pcout << "Starting loop with " << n_loops << " runs ... \n\n";

    for(unsigned loop_id = 0; loop_id < n_loops; ++loop_id)
    {
      Timer time;
      time.restart();

      // *** initialize matrix-free system
      SYSTEM_MATRIX                                        system_matrix;
      const std::shared_ptr<const MatrixFree<dim, Number>> mf_storage =
        laplace_problem.template build_mf_storage<Number>();
      system_matrix.initialize(mf_storage);

      setup_time += time.wall_time();

      // *** initialize global vectors
      LinearAlgebra::distributed::Vector<Number> dst;
      mf_storage->initialize_dof_vector(dst);
      for(auto it = dst.begin(); it != dst.end(); ++it)
        *it = (Number)rand() / RAND_MAX;
      const LinearAlgebra::distributed::Vector<Number> src{dst};
      dst *= 0.;

      time.restart();

      // *** matrix-free vmult()
      system_matrix.vmult(dst, src);

      apply_time += time.wall_time();
    }

    // *** post process
    fprint_time_details("Setup matrix-free storage (total):", setup_time);
    fprint_time_details("Matrix-free vmult() (total):", apply_time);
    time_details << std::endl;

    fprint_time_details("Setup matrix-free storage (average):", setup_time / n_loops);
    fprint_time_details("Matrix-free vmult() (average):", apply_time / n_loops);
    time_details << std::endl;

    const double n_dofs_normalized = 1.e-6 * laplace_problem.dof_handler.n_dofs();
    fprint_time_details(
      "Throughput vmult():", n_dofs_normalized / (apply_time / n_loops), "[MDoFs/s]");
    time_details << std::endl;

    pcout << "... finished within " << total_time.wall_time() << " [s] run-time!\n\n";
  }

  template<SmoothRoutine routine>
  std::vector<TimeInfo>
  test_tensor_product_smooth(const unsigned n_loops,
                             const unsigned n_refines   = 2,
                             const unsigned n_cell_reps = 1)
  {
    using LEVEL_MATRIX = CombinedOperator<dim, fe_degree, Number>;
    using PATCH_MATRIX = TensorProductMatrixSymmetricSum<dim, VectorizedArray<Number>, n_dofs>;
    using VECTOR       = LinearAlgebra::distributed::Vector<Number>;
    using SCHWARZ_PRECONDITIONER = SchwarzPreconditioner<dim, LEVEL_MATRIX, VECTOR, PATCH_MATRIX>;
    using SMOOTHER = SchwarzSmoother<dim, LEVEL_MATRIX, SCHWARZ_PRECONDITIONER, VECTOR>;

    Timer total_time;
    total_time.restart();

    // *** pre process
    AssertThrow(n_cell_reps % 2 == 1, ExcMessage("Cell repetition is not odd!"));
    parameters.n_refines            = n_refines;   // initial fine level
    parameters.n_cell_repetitions   = n_cell_reps; // # of coarse cells per dimension
    parameters.allow_one_level_only = true;
    const auto patch_variant        = parameters.schwarz_smoother_data.patch_variant;
    const auto smoother_variant     = parameters.schwarz_smoother_data.smoother_variant;
    parameters.schwarz_smoother_data.damping_factor =
      TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    MatrixOperator<dim, fe_degree, Number, n_dofs> laplace_problem{pcout, parameters};
    laplace_problem.distribute_dofs();
    const unsigned mg_level_min = laplace_problem.global_level;
    const unsigned mg_level_max = laplace_problem.global_level;
    setup_time                  = 0.;
    apply_time                  = 0.;

    // *** print test header
    laplace_problem.print_dof_info();
    std::string str_smooth_routine[] = {"MGSmootherRelaxation::apply()",
                                        "SchwarzPreconditioner::vmult()"};
    print_row_variable(pcout, 45, "Tested routine:", str_smooth_routine[(int)routine]);
    print_row_variable(pcout, 45, "Finite element:", laplace_problem.fe->get_name());
    pcout << std::endl;
    parameters.schwarz_smoother_data.print(pcout);
    pcout << std::endl;
    pcout << "Starting loop with " << n_loops << " runs ... \n\n";

    std::vector<TimeInfo>       time_infos;
    MGLevelObject<LEVEL_MATRIX> mg_matrices;
    MGSmootherRelaxation<LEVEL_MATRIX, SMOOTHER, LinearAlgebra::distributed::Vector<Number>>
      mg_smoother;
    for(unsigned loop_id = 0; loop_id < n_loops; ++loop_id)
    {
      mg_matrices.clear_elements();
      mg_smoother.clear();

      Timer time;
      time.restart();

      // setup level matrix (matrix-free)
      mg_matrices.resize(mg_level_min, mg_level_max);
      for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
      {
        const auto mf_storage_level = laplace_problem.template build_mf_storage<Number>(level);
        mg_matrices[level].initialize(mf_storage_level);
      }

      setup_time_misc += time.wall_time();

      time.restart();

      // *** setup Schwarz smoother (compute inverses)
      typename SMOOTHER::AdditionalData dummy_data;
      mg_smoother.initialize(mg_matrices, dummy_data); // set mg_matrices in MGSmootherRelaxation
      for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
      {
        const auto mf_storage_on_level = mg_matrices[level].get_matrix_free();
        const auto patch_storage =
          laplace_problem.template build_patch_storage<Number>(level, mf_storage_on_level);

        // *** setup Schwarz preconditioner
        typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
        precondition_data.relaxation      = parameters.schwarz_smoother_data.damping_factor;
        const auto schwarz_preconditioner = std::make_shared<SCHWARZ_PRECONDITIONER>();
        schwarz_preconditioner->initialize(patch_storage, mg_matrices[level], precondition_data);

        // *** setup smoother
        typename SMOOTHER::AdditionalData mg_smoother_data;
        mg_smoother_data.number_of_smoothing_steps =
          parameters.schwarz_smoother_data.number_of_smoothing_steps;
        mg_smoother.smoothers[level].initialize(
          mg_matrices[level],
          schwarz_preconditioner,
          mg_smoother_data); // actual initialization of Schwarz smoother within
                             // MGSmootherRelaxation
      }

      setup_time += time.wall_time();

      // *** initialize global vectors
      LinearAlgebra::distributed::Vector<Number> dst;
      LinearAlgebra::distributed::Vector<Number> src_memory;
      mg_matrices[mg_level_max].get_matrix_free()->initialize_dof_vector(dst);
      mg_matrices[mg_level_max].get_matrix_free()->initialize_dof_vector(src_memory);
      for(auto it = dst.begin(); it != dst.end(); ++it)
        *it = (Number)rand() / RAND_MAX;
      const LinearAlgebra::distributed::Vector<Number> & src{src_memory};

      time.restart();

      // *** read write operation
      {
        auto     it_src_memory = src_memory.begin();
        unsigned i             = 1;
        for(auto it = dst.begin(); it != dst.end(); ++it, ++it_src_memory)
          *it_src_memory = *it * 1.234 * static_cast<Number>(i);
      }

      read_write_time += time.wall_time();
      time.restart();

      mg_matrices[mg_level_max].vmult(dst, src);

      apply_time_misc += time.wall_time();
      time.restart();

      // *** apply fast diagonal Schwarz smoother/preconditioner
      const auto & schwarz_preconditioner =
        mg_smoother.smoothers[mg_level_max].get_preconditioner();
      switch(routine)
      {
        case SmoothRoutine::MG_SMOOTHER:
        {
          mg_smoother.apply(mg_level_max, dst, src);
          break;
        }
        case SmoothRoutine::PREC_VMULT:
        {
          schwarz_preconditioner.vmult(dst, src);
          break;
        }
        default:
          break;
      }

      apply_time += time.wall_time();

      const auto & data = schwarz_preconditioner.get_time_data();
      if(time_infos.empty())
        for(const auto & info : data)
          time_infos.emplace_back(info);
      else
      {
        AssertThrow(data.size() == time_infos.size(), ExcMessage("Time data mismatch!"));
        for(std::size_t id = 0; id < data.size(); ++id)
        {
          time_infos[id].time += data[id].time;
          time_infos[id].n_calls += data[id].n_calls;
        }
      }
    }

    // *** post process
    time_details << "Total time:" << std::endl;
    fprint_time_details("Setup matrix-free storage:", setup_time_misc);
    fprint_time_details("Setup smoother:", setup_time);
    fprint_time_details("Matrix-free vmult:", apply_time_misc);
    fprint_time_details("Fast diagonal smooth:", apply_time);
    time_details << std::endl;

    time_details << "Average time:" << std::endl;
    fprint_time_details("Setup matrix-free storage:", setup_time_misc / n_loops);
    fprint_time_details("Setup smoother:", setup_time / n_loops);
    fprint_time_details("Matrix-free vmult:", apply_time_misc / n_loops);
    fprint_time_details("Fast diagonal smooth:", apply_time / n_loops);
    fprint_time_details("Read-Write operation:", read_write_time / n_loops);
    for(const auto & info : time_infos)
    {
      fprint_time_details(info.description, info.time / n_loops, info.unit);
    }
    time_details << std::endl;

    time_details << "Throughput:" << std::endl;
    n_dofs_level                   = laplace_problem.dof_handler.n_dofs(mg_level_max);
    const double n_dofs_normalized = static_cast<double>(n_dofs_level) / 1.e+6;
    fprint_time_details(
      "Matrix-free vmult:", n_dofs_normalized / (apply_time_misc / n_loops), "[MDoFs/s]");
    fprint_time_details(
      "Fast diagonal smooth:", n_dofs_normalized / (apply_time / n_loops), "[MDoFs/s]");
    fprint_time_details(
      "Read-Write operation:", n_dofs_normalized / (read_write_time / n_loops), "[MDoFs/s]");
    for(const auto & info : time_infos)
    {
      fprint_time_details(info.description, n_dofs_normalized / (info.time / n_loops), "[MDoFs/s]");
    }
    time_details << std::endl;

    time_details << "Ingredients (number of calls):" << std::endl;
    for(const auto & info : time_infos)
    {
      print_row_variable(time_details, 2, "", 43, info.description, info.n_calls / n_loops);
    }
    time_details << std::endl;

    const auto   mf_storage       = mg_matrices[mg_level_max].get_matrix_free();
    const auto & schwarz_smoother = mg_smoother.smoothers[mg_level_max];
    const auto   patch_storage    = schwarz_smoother.get_preconditioner().get_subdomain_handler();
    fprint_misc(mf_storage, patch_storage, mg_level_max);

    pcout << "... finished within " << total_time.wall_time() << " [s] run-time!\n\n";

    return time_infos;
  }

  void
  test_dg_laplace(const unsigned n_loops, unsigned n_refines = 2, unsigned n_cell_reps = 2)
  {
    Timer total_time;
    total_time.restart();

    // *** pre process
    parameters.write_timer_output   = false;
    parameters.allow_one_level_only = false;
    if(n_cell_reps == 1) // guarantee a vertex patch on coarsest level
    {
      Assert(n_refines > 2, ExcMessage("Not enough MG levels to test properly."));
      ++n_cell_reps;
      --n_refines;
    }
    parameters.coarse_level       = 0;
    parameters.n_refines          = n_refines;   // initial fine level
    parameters.n_cell_repetitions = n_cell_reps; // # of coarse cells per dimension
    MatrixOperator<dim, fe_degree, Number, n_dofs> laplace_problem{pcout, parameters};

    setup_time = 0.;
    apply_time = 0.;

    pcout << "Starting loop with " << n_loops << " runs ... \n\n";

    Timer time;
    time.restart();

    std::vector<TimeInfo> time_data;
    for(unsigned loop_id = 0; loop_id < n_loops; ++loop_id)
    {
      laplace_problem.pcout.set_condition(loop_id == 0); // print only on first run
      const auto run_times = laplace_problem.run();

      // *** submit timings
      if(time_data.empty())
        time_data.resize(run_times.back().size());
      for(std::size_t t = 0; t < time_data.size(); ++t)
        time_data[t] += run_times.back()[t];
    }

    std::pair<unsigned, unsigned> setup_range{0, laplace_problem.n_timings_setup};
    std::pair<unsigned, unsigned> solve_range{setup_range.second,
                                              setup_range.second + laplace_problem.n_timings_solve};
    time_details << "Total time (setup): \n";
    for(unsigned t = setup_range.first; t < setup_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time, time_data[t].unit);
    time_details << std::endl;

    time_details << "Total time (solve): \n";
    for(unsigned t = solve_range.first; t < solve_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time, time_data[t].unit);
    time_details << std::endl;

    time_details << "Average time (setup): \n";
    for(unsigned t = setup_range.first; t < setup_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time / n_loops, time_data[t].unit);
    time_details << std::endl;

    time_details << "Average time (solve): \n";
    for(unsigned t = solve_range.first; t < solve_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time / n_loops, time_data[t].unit);
    time_details << std::endl;

    const unsigned mg_level_max = laplace_problem.mg_matrices.max_level();
    const auto     mf_storage   = laplace_problem.mg_matrices[mg_level_max].get_matrix_free();
    const auto     schwarz_preconditioner = laplace_problem.mg_schwarz_precondition[mg_level_max];
    const auto     patch_storage          = schwarz_preconditioner->get_subdomain_handler();
    fprint_misc(mf_storage, patch_storage, mg_level_max);

    pcout << "... finished within " << total_time.wall_time() << " [s] run-time!\n\n";
  }

  void
  test_multigrid_timings(const unsigned                      n_loops,
                         const std::pair<unsigned, unsigned> refinement_pair = {0, 0})
  {
    Timer total_time;
    total_time.restart();

    // *** pre process
    parameters.write_timer_output   = false;
    parameters.allow_one_level_only = false;
    parameters.n_cycles             = 1;

    // *** fix test parameters dependent on the underlying geometry
    switch(parameters.geometry_variant)
    {
      case Parameter::GeometryVariant::Cube:
      {
        parameters.coarse_level = 1;
        parameters.n_refines    = dim == 3 ? 6 : 10;
        if(refinement_pair != std::pair<unsigned, unsigned>(0, 0)) // use refinement pair
        {
          parameters.coarse_level       = 0;
          parameters.n_cell_repetitions = refinement_pair.second;
          parameters.n_refines          = refinement_pair.first;
        }
        break;
      }
      case Parameter::GeometryVariant::CubeDistorted:
      {
        parameters.coarse_level      = dim == 3 ? 3 : 5;
        parameters.n_refines         = dim == 3 ? 6 : 10;
        IP::pre_factor               = 4.;
        parameters.n_refines_distort = parameters.coarse_level;
        parameters.distortion_factor = 0.25;
        break;
      }
      case Parameter::GeometryVariant::Ball:
      {
        AssertThrow(dim == 2, ExcMessage("dim==2 required."));
        IP::pre_factor          = 4.;
        parameters.coarse_level = 0;
        parameters.n_refines    = dim == 3 ? 5 : 9;
        AssertThrow(false, ExcMessage("Not implemented."));
        AssertThrow(parameters.schwarz_smoother_data.patch_variant == TPSS::PatchVariant::cell,
                    ExcMessage("Not implemented"));
        break;
      }
    }

    MatrixOperator<dim, fe_degree, Number, n_dofs> laplace_problem{pcout, parameters};

    pcout << "Starting loop with " << n_loops << " runs ... \n\n";

    Timer time;
    time.restart();

    std::vector<TimeInfo> time_data;
    for(unsigned loop_id = 0; loop_id < n_loops; ++loop_id)
    {
      laplace_problem.pcout.set_condition(loop_id == 0); // print only on first run
      const auto run_times = laplace_problem.run();      // TimeInfo collections per cycle

      // *** submit timings
      if(time_data.empty())
        time_data.resize(run_times.back().size());
      for(std::size_t t = 0; t < time_data.size(); ++t)
        time_data[t] += run_times.back()[t];
    }

    // *** post process
    laplace_problem.pcout.set_condition(true); // enable printing
    std::pair<unsigned, unsigned> setup_range{0, laplace_problem.n_timings_setup};
    std::pair<unsigned, unsigned> solve_range{setup_range.second,
                                              setup_range.second + laplace_problem.n_timings_solve};
    time_details << "Total time (setup): \n";
    for(unsigned t = setup_range.first; t < setup_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time, time_data[t].unit);
    time_details << std::endl;

    time_details << "Total time (solve): \n";
    for(unsigned t = solve_range.first; t < solve_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time, time_data[t].unit);
    time_details << std::endl;

    time_details << "Average time (setup): \n";
    for(unsigned t = setup_range.first; t < setup_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time / n_loops, time_data[t].unit);
    time_details << std::endl;

    time_details << "Average time (solve): \n";
    for(unsigned t = solve_range.first; t < solve_range.second; ++t)
      fprint_time_details(time_data[t].description, time_data[t].time / n_loops, time_data[t].unit);
    time_details << std::endl;

    const unsigned mg_level_max = laplace_problem.mg_matrices.max_level();
    const auto     mf_storage   = laplace_problem.mg_matrices[mg_level_max].get_matrix_free();
    const auto     schwarz_preconditioner = laplace_problem.mg_schwarz_precondition[mg_level_max];
    const auto     patch_storage          = schwarz_preconditioner->get_subdomain_handler();
    fprint_misc(mf_storage, patch_storage, mg_level_max);

    pcout << "... finished within " << total_time.wall_time() << " [s] run-time!\n\n";
  }

  void
  test_solver_iterations(const int coarse_level = -1,
                         const int n_refines    = -1,
                         const int n_cycles     = -1)
  {
    // *** pre process
    parameters.write_timer_output   = true;
    parameters.allow_one_level_only = false;

    // *** fix test parameters dependent on the underlying geometry
    switch(parameters.geometry_variant)
    {
      case Parameter::GeometryVariant::Cube:
      {
        parameters.coarse_level = 1;
        parameters.n_refines    = parameters.coarse_level + 1;
        parameters.n_cycles     = dim == 3 ? 5 : 9;
        // if(fe_degree > 4)
        //   --parameters.n_cycles;
        break;
      }
      case Parameter::GeometryVariant::CubeDistorted:
      {
        IP::pre_factor               = 4.;
        parameters.coarse_level      = dim == 3 ? 3 : 5;
        parameters.n_refines         = parameters.coarse_level + 1;
        parameters.n_cycles          = dim == 3 ? 3 : 5;
        parameters.n_refines_distort = parameters.coarse_level;
        parameters.distortion_factor = 0.25;
        break;
      }
      case Parameter::GeometryVariant::Ball:
      {
        AssertThrow(dim == 2, ExcMessage("dim==2 required."));
        IP::pre_factor          = 4.;
        parameters.coarse_level = 0;
        parameters.n_refines    = parameters.coarse_level + 1;
        parameters.n_cycles     = dim == 3 ? 5 : 9;
        AssertThrow(parameters.schwarz_smoother_data.patch_variant == TPSS::PatchVariant::cell,
                    ExcMessage("Not implemented"));
        break;
      }
    }

    // *** dynamic parameters
    if(coarse_level > -1)
      parameters.coarse_level = coarse_level;
    if(n_refines > 0)
      parameters.n_refines = n_refines;
    if(n_cycles > 0)
      parameters.n_cycles = n_cycles;

    // *** print test header and generic information
    print_row_variable(
      pcout, 15, "HEADER:", "Obtain the iteration counts of by means of GMG (Schwarz smoothers)");
    print_row_variable(pcout,
                       15,
                       "       ",
                       "preconditioned Krylov solvers of the DG Laplace problem");
    print_row_variable(pcout, 15, "STATUS:", 30, "IP pre-factor", IP::pre_factor);
    print_row_variable(pcout, 15, "STATUS:", 30, "Coarse level", parameters.coarse_level);
    print_row_variable(pcout, 15, "STATUS:", 30, "Number of refines", parameters.n_refines);
    print_row_variable(pcout, 15, "STATUS:", 30, "Number of cycles", parameters.n_cycles);
    pcout << std::endl << std::endl;

    const unsigned int nc = parameters.n_cycles;
    parameters.n_cycles   = 1;

    for(unsigned cycle = 0; cycle < nc; ++cycle)
    {
      MatrixOperator<dim, fe_degree, Number, n_dofs> laplace_problem{pcout, parameters};
      const bool                                     is_last_cycle  = (cycle + 1) == nc;
      const bool                                     is_first_cycle = cycle == 0;
      const auto run_times = laplace_problem.run(is_last_cycle || is_first_cycle);
      ++parameters.n_refines;
    }
  }

  Parameter parameters;

  ConditionalOStream & pcout;
  ConditionalOStream   time_details;

  double       setup_time;
  double       apply_time;
  double       setup_time_misc;
  double       apply_time_misc;
  double       read_write_time;
  unsigned int n_dofs_level;
};

int
main(int argc, char * argv[])
{
  // *** init MPI (1 proc only!)
  const auto max_threads = 1;
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** test parameters
  using value_type             = GlobParam::value_type;
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;
  constexpr int      n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(GlobParam::patch_variant) * (fe_degree + 1);

  // // *** print the possible refinements for the given thresholds
  // for (int fe_degree = 1; fe_degree < 20; ++fe_degree)
  //   GlobParam::lookup_refine_pair (dim, fe_degree, /*minimal level allowed*/2, true);
  // std::cout << std::endl;

  // *** common test parameters
  using TEST = Test<dim, fe_degree, value_type, n_patch_dofs_per_direction>;
  deallog.depth_console(0);

  Parameter parameters;
  parameters.schwarz_smoother_data.patch_variant    = GlobParam::patch_variant;
  parameters.schwarz_smoother_data.smoother_variant = GlobParam::smoother_variant;
  parameters.schwarz_smoother_data.manual_coloring  = true;
  parameters.write_fe_output                        = false;
  parameters.solver_variant                         = GlobParam::lookup_solver_variant();
  parameters.mf_tasks_scheme_id                     = GlobParam::mf_tasks_scheme_id;

  // *** set output stream
  std::fstream fstream;
  const auto   pcout_factory = [&]() {
    std::shared_ptr<ConditionalOStream> out_ptr;
    if(GlobParam::use_filestream)
      out_ptr =
        std::make_shared<ConditionalOStream>(fstream,
                                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    else
      out_ptr =
        std::make_shared<ConditionalOStream>(std::cout,
                                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    return out_ptr;
  };
  const auto           pcout_owned = pcout_factory();
  ConditionalOStream & pcout       = *pcout_owned;

  // *** append filename w.r.t. the underlying smoother
  const auto str_append_smoother = [](const std::string prefix = "") {
    std::ostringstream ostrstream;
    std::string        str_patch_variant[] = {"", "/cp/", "/vp/"};
    ostrstream << str_patch_variant[(int)GlobParam::patch_variant];
    ostrstream << prefix;
    std::string str_smoother_variant[] = {"invalid_", "ADD", "MUL"};
    ostrstream << str_smoother_variant[(int)GlobParam::smoother_variant];
    ostrstream << "_DIM" << dim;
    ostrstream << "_DEG" << fe_degree;
    return ostrstream.str();
  };

  switch(GlobParam::test_variant)
  {
    case GlobParam::TestVariant::NoTest:
    {
      std::cout << "Please choose a test." << std::endl;
      break;
    }

    case GlobParam::TestVariant::VmultTime:
    { // TEST1: Matrix-free finite element operation (Poisson)
      std::ostringstream ostrstream;
      ostrstream << "poisson_vmult";
      ostrstream << "_DIM" << dim;
      ostrstream << "_DEG" << fe_degree;
      ostrstream << "_MPI" << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      parameters.filename = GlobParam::output_directory + ostrstream.str();
      fstream.open(parameters.filename + ".txt", std::ios_base::app);
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);
      // *** refinement pair: (level, number of cells per dimension)
      const std::pair<unsigned, unsigned> refinement_pair{
        GlobParam::lookup_refine_pair(dim, fe_degree)};
      TEST{pcout, parameters}.test_matrix_free_vmult(GlobParam::n_loops,
                                                     refinement_pair.first,
                                                     refinement_pair.second);
      break;
    }

    case GlobParam::TestVariant::SmoothTime:
    { // TEST2: Matrix-free operation (level-based) and smoother operation on given refinement
      parameters.filename = GlobParam::output_directory + str_append_smoother("smooth_");
      const std::string str_suffix =
        "." + Utilities::to_string(GlobParam::n_dofs_minimal / 1e+6) + ".txt";
      std::cout << parameters.filename << std::endl;
      fstream.open(parameters.filename + str_suffix, std::ios_base::app);
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);
      // *** refinement pair: (level, number of cells per dimension)
      const std::pair<unsigned, unsigned> refinement_pair{
        GlobParam::lookup_refine_pair(dim, fe_degree)};
      constexpr auto routine = TEST::SmoothRoutine::MG_SMOOTHER;
      TEST{pcout, parameters}.template test_tensor_product_smooth<routine>(GlobParam::n_loops,
                                                                           refinement_pair.first,
                                                                           refinement_pair.second);
      break;
    }

    case GlobParam::TestVariant::MGTimeSameDoFs:
    { // TEST3: Complete linear system solve preconditioned by GMG
      parameters.geometry_variant = GlobParam::geometry_variant;
      const std::pair<unsigned, unsigned> refinement_pair{
        GlobParam::lookup_refine_pair(dim, fe_degree, 2)};

      parameters.mg_coarse_grid_variant =
        Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
      parameters.mg_coarse_chebyshev_reduction =
        GlobParam::lookup_chebyshev_reduction(); // 0.05;//1.e-3;

      parameters.mg_smoother_variant = Parameter::MGSmootherVariant::Schwarz; // ChebyshevSchwarz
      parameters.schwarz_smoother_data.symmetrize_smoothing = GlobParam::symmetrize_smoothing;
      const double additional_damping_factor                = GlobParam::lookup_local_damping();
      parameters.schwarz_smoother_data.local_damping_factor = additional_damping_factor;
      parameters.n_cg_steps_chebyshev                       = 10;
      parameters.chebyshev_degree                           = argc > 1 ? std::atoi(argv[1]) : 1;
      parameters.smoothing_range = GlobParam::lookup_smoothing_range(parameters.chebyshev_degree);
      if(parameters.mg_smoother_variant != Parameter::MGSmootherVariant::ChebyshevSchwarz)
        parameters.schwarz_smoother_data.number_of_smoothing_steps =
          argc > 1 ? std::atoi(argv[1]) : 1;

      // *** configure output stream
      const auto str_append_output = [&]() {
        std::ostringstream ostrstream;
        ostrstream << str_append_smoother("mgtimingsame_");
        ostrstream << "_GEO" << Parameter::str_geometry_variant[(int)parameters.geometry_variant];
        if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::ChebyshevSchwarz)
        {
          ostrstream << "_CHD" << parameters.chebyshev_degree;
          ostrstream << "_SRA" << parameters.smoothing_range;
        }
        if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::Schwarz)
          ostrstream << "_SST" << parameters.schwarz_smoother_data.number_of_smoothing_steps;
        if(GlobParam::smoother_variant == TPSS::SmootherVariant::multiplicative &&
           GlobParam::symmetrize_smoothing)
          ostrstream << "_sym";
        return ostrstream.str();
      };
      const std::string str_suffix =
        "." + Utilities::to_string(GlobParam::n_dofs_minimal / 1e+6) + ".txt";
      std::cout << parameters.filename << std::endl;
      parameters.filename = GlobParam::output_directory + str_append_output();
      fstream.open(parameters.filename + str_suffix, std::ios_base::app);
      std::cout << "filename: " << parameters.filename + str_suffix << std::endl;
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);

      TEST{pcout, parameters}.test_multigrid_timings(GlobParam::n_loops, refinement_pair);
      break;
    }

    case GlobParam::TestVariant::MGTime:
    { // TEST3.5: Complete linear system solve preconditioned by GMG
      parameters.geometry_variant = GlobParam::geometry_variant;

      parameters.mg_coarse_grid_variant =
        Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
      parameters.mg_coarse_chebyshev_reduction =
        GlobParam::lookup_chebyshev_reduction(); // 0.05;//1.e-3;

      parameters.mg_smoother_variant = Parameter::MGSmootherVariant::Schwarz; // ChebyshevSchwarz
      parameters.schwarz_smoother_data.symmetrize_smoothing = GlobParam::symmetrize_smoothing;
      const double additional_damping_factor                = GlobParam::lookup_local_damping();
      parameters.schwarz_smoother_data.local_damping_factor = additional_damping_factor;
      parameters.n_cg_steps_chebyshev                       = 10;
      parameters.chebyshev_degree                           = argc > 1 ? std::atoi(argv[1]) : 1;
      parameters.smoothing_range = GlobParam::lookup_smoothing_range(parameters.chebyshev_degree);
      if(parameters.mg_smoother_variant != Parameter::MGSmootherVariant::ChebyshevSchwarz)
        parameters.schwarz_smoother_data.number_of_smoothing_steps =
          argc > 1 ? std::atoi(argv[1]) : 1;

      // *** configure output stream
      const auto str_append_output = [&]() {
        std::ostringstream ostrstream;
        ostrstream << str_append_smoother("mgtiming_");
        ostrstream << "_GEO" << Parameter::str_geometry_variant[(int)parameters.geometry_variant];
        if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::ChebyshevSchwarz)
        {
          ostrstream << "_CHD" << parameters.chebyshev_degree;
          ostrstream << "_SRA" << parameters.smoothing_range;
        }
        if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::Schwarz)
          ostrstream << "_SST" << parameters.schwarz_smoother_data.number_of_smoothing_steps;
        if(GlobParam::smoother_variant == TPSS::SmootherVariant::multiplicative &&
           GlobParam::symmetrize_smoothing)
          ostrstream << "_sym";
        return ostrstream.str();
      };
      parameters.filename = GlobParam::output_directory + str_append_output();
      fstream.open(parameters.filename + ".C.txt", std::ios_base::app);
      std::cout << "filename: " << parameters.filename << std::endl;
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);

      TEST{pcout, parameters}.test_multigrid_timings(GlobParam::n_loops);
      break;
    }

    case GlobParam::TestVariant::Robust:
    { // TEST4: Check the robustness of the underlying multigrid smoother
      parameters.geometry_variant = GlobParam::geometry_variant;

      parameters.mg_coarse_grid_variant        = Parameter::CoarseGridVariant::ChebyshevAccurate;
      parameters.mg_coarse_chebyshev_reduction = 1.e-12;
      // GlobParam::lookup_chebyshev_reduction(); // 0.05;//1.e-3;

      parameters.mg_smoother_variant =
        Parameter::MGSmootherVariant::Schwarz; // Schwarz//ChebyshevSchwarz
      parameters.schwarz_smoother_data.symmetrize_smoothing = GlobParam::symmetrize_smoothing;
      const double additional_damping_factor                = 1.00;
      parameters.schwarz_smoother_data.damping_factor       = additional_damping_factor;
      parameters.chebyshev_degree                           = 1;
      parameters.n_cg_steps_chebyshev                       = 10;
      parameters.smoothing_range = GlobParam::lookup_smoothing_range(parameters.chebyshev_degree);
      parameters.schwarz_smoother_data.number_of_smoothing_steps =
        argc > 1 ? std::atoi(argv[1]) : 1;

      // *** configure output stream
      const std::string str_prefix = "robust_";
      const auto        str_suffix = [&]() {
        std::ostringstream ostrstream;
        ostrstream << str_append_smoother("robust_");
        ostrstream << "_GEO" << Parameter::str_geometry_variant[(int)parameters.geometry_variant];
        if(parameters.mg_smoother_variant != Parameter::MGSmootherVariant::ChebyshevSchwarz)
          ostrstream << "_SST" << parameters.schwarz_smoother_data.number_of_smoothing_steps;
        if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::ChebyshevSchwarz)
          ostrstream << "_CHD" << parameters.chebyshev_degree << "_SRA"
                     << parameters.smoothing_range;
        if(GlobParam::symmetrize_smoothing)
          ostrstream << "_sym";
        return ostrstream.str() + ".txt";
      };
      parameters.filename = CT::LOG_DIR_ + GlobParam::str_append_fname(str_prefix, str_suffix());
      std::cout << "file path (output): " << parameters.filename << std::endl;
      fstream.open(parameters.filename, std::ios_base::app);
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);

      TEST{pcout, parameters}.test_solver_iterations(/*coarse_level*/ -1,
                                                     /*n_refines*/ -1,
                                                     /*n_cycles*/ -1);
      break;
    }

    case GlobParam::TestVariant::SmoothingRange:
    { // TEST5: Check the robustness w.r.t. to different smoothing ranges and Chebyshev degrees
      parameters.geometry_variant = GlobParam::geometry_variant;

      parameters.mg_coarse_grid_variant = Parameter::CoarseGridVariant::ChebyshevAccurate;
      parameters.mg_coarse_chebyshev_reduction =
        GlobParam::lookup_chebyshev_reduction(); // 0.05;//1.e-3;

      parameters.mg_smoother_variant = Parameter::MGSmootherVariant::ChebyshevSchwarz;
      parameters.schwarz_smoother_data.symmetrize_smoothing = GlobParam::symmetrize_smoothing;
      const double additional_damping_factor                = argc > 3 ? std::atof(argv[3]) : 1.00;
      parameters.schwarz_smoother_data.local_damping_factor = additional_damping_factor;
      parameters.n_cg_steps_chebyshev                       = 10;
      parameters.chebyshev_degree                           = argc > 2 ? std::atof(argv[2]) : 1;
      parameters.smoothing_range                            = argc > 1 ?
                                     std::atof(argv[1]) :
                                     GlobParam::lookup_smoothing_range(parameters.chebyshev_degree);
      parameters.schwarz_smoother_data.number_of_smoothing_steps = 1;

      // *** configure output stream
      const auto str_append_output = [&]() {
        std::ostringstream ostrstream;
        ostrstream << str_append_smoother("srange_");
        ostrstream << "_GEO" << Parameter::str_geometry_variant[(int)parameters.geometry_variant];
        if(additional_damping_factor < 1.00)
          ostrstream << "_LDF" << std::setprecision(3) << additional_damping_factor;
        ostrstream << "_CHD" << parameters.chebyshev_degree;
        ostrstream << "_SRA" << parameters.smoothing_range;
        if(!GlobParam::symmetrize_smoothing)
          ostrstream << "_nonsym";
        return ostrstream.str();
      };
      parameters.filename = GlobParam::output_directory + str_append_output();
      fstream.open(parameters.filename + ".B.txt", std::ios_base::app);
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);

      TEST{pcout, parameters}.test_solver_iterations(/*coarse_level*/ -1,
                                                     /*n_refines*/ -1,
                                                     /*n_cycles*/ -1);
      break;
    }

    case GlobParam::TestVariant::LocalDamping:
    { // TEST6: Check the robustness on non-Cartesian meshes w.r.t. to (local) additional dampings
      parameters.geometry_variant = GlobParam::geometry_variant;
      AssertThrow(GlobParam::geometry_variant != Parameter::GeometryVariant::Cube,
                  ExcMessage("Wrong geometry."));

      parameters.mg_coarse_grid_variant =
        Parameter::CoarseGridVariant::ChebyshevAccurate; // IterativeFixed;
      parameters.mg_coarse_chebyshev_reduction =
        GlobParam::lookup_chebyshev_reduction(); // 0.05;//1.e-3;

      parameters.mg_smoother_variant = Parameter::MGSmootherVariant::Schwarz; // ChebyshevSchwarz;
      parameters.schwarz_smoother_data.symmetrize_smoothing = GlobParam::symmetrize_smoothing;
      const double additional_damping_factor                = argc > 1 ? std::atof(argv[1]) : 1.;
      parameters.schwarz_smoother_data.local_damping_factor = additional_damping_factor;
      parameters.n_cg_steps_chebyshev                       = 10;
      parameters.chebyshev_degree                           = argc > 3 ? std::atoi(argv[3]) : 1;
      parameters.smoothing_range =
        /*argc>2 ? std::atof(argv[2]) : */ GlobParam::lookup_smoothing_range(
          parameters.chebyshev_degree);
      parameters.schwarz_smoother_data.number_of_smoothing_steps =
        argc > 2 ? std::atoi(argv[2]) : 1;
      parameters.schwarz_smoother_data.print_details = true;

      // *** configure output stream
      const auto str_append_output = [&]() {
        std::ostringstream ostrstream;
        ostrstream << str_append_smoother("locdamp_");
        ostrstream << "_GEO" << Parameter::str_geometry_variant[(int)parameters.geometry_variant];
        ostrstream << "_LDF" << std::setprecision(3) << additional_damping_factor;
        if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::ChebyshevSchwarz)
        {
          ostrstream << "_CHD" << parameters.chebyshev_degree;
          ostrstream << "_SRA" << parameters.smoothing_range;
        }
        else
          ostrstream << "_SST" << parameters.schwarz_smoother_data.number_of_smoothing_steps;
        if(GlobParam::smoother_variant == TPSS::SmootherVariant::multiplicative &&
           GlobParam::symmetrize_smoothing)
          ostrstream << "_sym";
        return ostrstream.str();
      };
      parameters.filename = GlobParam::output_directory + str_append_output();
      fstream.open(parameters.filename + ".C.txt", std::ios_base::app);
      std::cout << "filename: " << parameters.filename << std::endl;
      GlobParam::print_generic_header(pcout, MPI_COMM_WORLD);

      TEST{pcout, parameters}.test_solver_iterations(/*coarse_level*/ -1,
                                                     /*n_refines*/ -1,
                                                     /*n_cycles*/ -1);
      break;
    }
  }

  // { // SANDBOX SANDBOX SANDBOX
  //   parameters.geometry_variant = Parameter::GeometryVariant::CubeDistorted;

  //   // *** define multigrid parameters
  //   const auto GlobParam::lookup_chebyshev_reduction = [&]()
  //     {
  // 	double reduction{0.05};
  // 	if (patch_variant == TPSS::PatchVariant::vertex
  // 	    && smoother_variant == TPSS::SmootherVariant::multiplicative)
  // 	  reduction = 1.e-3;
  // 	return reduction;
  //     };
  //   const auto GlobParam::lookup_smoothing_range = [&]()
  //     {
  // 	double range{15.};//{dim == 3 ? 15. : 20.};
  // 	return range;
  //     };

  //   // parameters.mg_coarse_grid_variant =
  //   Parameter::CoarseGridVariant::IterativeFixed;//ChebyshevAccurate;
  //   // parameters.mg_coarse_iterations = 1000 * Utilities::pow
  //   (fe_degree+1,dim);//GlobParam::lookup_chebyshev_reduction();//0.05;//1.e-3;
  //   parameters.mg_coarse_grid_variant =
  //   Parameter::CoarseGridVariant::ChebyshevAccurate;//IterativeFixed;
  //   parameters.mg_coarse_chebyshev_reduction =
  //   GlobParam::lookup_chebyshev_reduction();//0.05;//1.e-3;

  //   parameters.mg_smoother_variant =
  //   Parameter::MGSmootherVariant::ChebyshevSchwarz;//Schwarz;//ChebyshevSchwarz
  //   parameters.schwarz_smoother_data.symmetrize_smoothing = GlobParam::symmetrize_smoothing;
  //   const double additional_damping_factor = 0.8;
  //   //parameters.schwarz_smoother_data.damping_factor = additional_damping_factor;
  //   parameters.schwarz_smoother_data.local_damping_factor = additional_damping_factor;
  //   parameters.n_cg_steps_chebyshev = 10;
  //   parameters.smoothing_range = argc>1 ? std::atof(argv[1]) :
  //   GlobParam::lookup_smoothing_range(); parameters.chebyshev_degree = argc>2 ?
  //   std::atof(argv[2]) : 1; parameters.schwarz_smoother_data.number_of_smoothing_steps = 1;//argc
  //   > 1 ? std::atoi(argv[1]) : 1; parameters.schwarz_smoother_data.print_details = true;

  //   TEST {pcout, parameters}.test_solver_iterations
  //   (/*coarse_level*/-1,/*n_refines*/-1,/*n_cycles*/3);
  // }

  pcout << "============================================================" << std::endl;
  fstream.close();

  return 0;
}
