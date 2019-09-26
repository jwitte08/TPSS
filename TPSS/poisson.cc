
/*
 * test poisson problem
 *
 *  Created on: Sep 12, 2019
 *      Author: witte
 */

#include "poisson.h"
#include "ct_parameter.h"
//#include "laplace_problem.h"

using namespace dealii;
using namespace Laplace;

struct Timings
{
  std::vector<Utilities::MPI::MinMaxAvg> setup;
  std::vector<Utilities::MPI::MinMaxAvg> apply;
};

struct TestParameter
{
  TPSS::PatchVariant    patch_variant = CT::PATCH_VARIANT_; // TPSS::PatchVariant::cell;
  TPSS::SmootherVariant smoother_variant =
    CT::SMOOTHER_VARIANT_; // TPSS::SmootherVariant::multiplicative;
  double   cg_reduction        = 1.e-8;
  unsigned n_refinements       = 1;
  unsigned n_cycles            = 1;
  unsigned n_smoothing_steps   = 1;
  unsigned n_samples           = 10;
  unsigned n_subsamples_vmult  = 4;
  unsigned n_subsamples_smooth = 4;
  unsigned n_subsamples_mg     = 2;
};

std::string
write_table_to_string(const Laplace::PostProcessData & pp_data, const Timings & timings)
{
  std::ostringstream oss;

  {
    ConvergenceTable info_table;
    for(const auto item : pp_data.n_mg_levels)
      info_table.add_value("n_levels", item);
    for(const auto item : pp_data.n_cells_global)
      info_table.add_value("n_cells", item);
    for(const auto item : pp_data.n_dofs_global)
      info_table.add_value("n_dofs", item);
    for(const auto item : pp_data.n_colors_system)
      info_table.add_value("n_colors", item);
    for(const auto item : pp_data.n_iterations_system)
      info_table.add_value("n_iter", item);
    for(const auto item : pp_data.average_reduction_system)
      info_table.add_value("reduction", item);

    info_table.set_scientific("reduction", true);
    info_table.set_precision("reduction", 3);
    info_table.write_text(oss);
  }

  oss << std::endl << std::endl;

  {
    ConvergenceTable timings_table;

    for(unsigned n = 0; n < timings.apply.size(); ++n)
    {
      timings_table.add_value("sample", n);
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
    timings_table.write_text(oss);
  }

  return oss.str();
}

template<int dim, int fe_degree>
void
test(const TestParameter & prm = TestParameter{})
{
  constexpr int n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);
  using PoissonProblem =
    typename Poisson::ModelProblem<dim, fe_degree, double, n_patch_dofs_per_direction>;
  using VECTOR           = typename PoissonProblem::VECTOR;
  using SCHWARZ_SMOOTHER = typename PoissonProblem::SCHWARZ_SMOOTHER;
  using SYSTEM_MATRIX    = typename PoissonProblem::SYSTEM_MATRIX;
  using LEVEL_MATRIX     = typename PoissonProblem::LEVEL_MATRIX;

  // *** configure output filenames
  const auto         n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  std::ostringstream oss;
  oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
  oss << "_" << dim << "D";
  oss << "_" << fe_degree << "deg";
  oss << "_" << n_mpi_procs << "procs";
  oss << "_" << prm.n_smoothing_steps << "steps";
  // oss << std::scientific << std::setprecision(3);
  // oss << "_" << prm.damping_factor << "ldamp";
  const std::string filename = oss.str();

  Laplace::Parameter parameters;
  //: discretization
  parameters.n_cycles  = prm.n_cycles;
  parameters.n_refines = prm.n_refinements;
  //: iterative solver
  parameters.solver_reduction      = prm.cg_reduction;
  parameters.solver_max_iterations = 100;
  parameters.precondition_variant  = Parameter::PreconditionVariant::MG;
  //: multigrid
  parameters.coarse_level                           = 1;
  parameters.schwarz_smoother_data.patch_variant    = prm.patch_variant;
  parameters.schwarz_smoother_data.smoother_variant = prm.smoother_variant;
  parameters.schwarz_smoother_data.manual_coloring  = true;
  const double damping_factor =
    TPSS::lookup_damping_factor(prm.patch_variant, prm.smoother_variant, dim);
  parameters.schwarz_smoother_data.damping_factor            = damping_factor;
  parameters.schwarz_smoother_data.number_of_smoothing_steps = prm.n_smoothing_steps;
  parameters.mg_smoother_post_reversed                       = true;

  Timings                  timings_vmult, timings_smooth, timings_mg, timings_total;
  Laplace::PostProcessData pp_data;
  PoissonProblem poisson_problem{parameters};
  poisson_problem.create_triangulation(parameters.n_refines);
  poisson_problem.distribute_dofs();

  for(unsigned sample = 0; sample < prm.n_samples; ++sample)
  {
    Timer          time(MPI_COMM_WORLD, true);

    //: setup (total)
    time.restart();
    poisson_problem.prepare_linear_system();
    const auto & gmg_preconditioner = poisson_problem.prepare_preconditioner_mg();
    time.stop();
    timings_total.setup.push_back(time.get_last_lap_wall_time_data());

    {
      Timer time(MPI_COMM_WORLD, true);

      //: setup
      time.restart();
      const auto    mf_storage = poisson_problem.template build_mf_storage<double>();
      SYSTEM_MATRIX system_matrix;
      system_matrix.initialize(mf_storage);
      time.stop();
      timings_vmult.setup.push_back(time.get_last_lap_wall_time_data());

      //: vmult
      VECTOR tmp;
      mf_storage->initialize_dof_vector(tmp);
      fill_with_random_values(tmp);
      VECTOR dst = tmp;
      time.restart();
      for(unsigned subsample = 0; subsample < prm.n_subsamples_vmult; ++subsample)
        system_matrix.vmult(dst, tmp);
      time.stop();
      timings_vmult.apply.push_back(time.get_last_lap_wall_time_data());
    }

    {
      Timer time(MPI_COMM_WORLD, true);

      //: setup
      time.restart();
      const unsigned fine_level = poisson_problem.triangulation.n_global_levels() - 1;
      const auto     mf_storage = poisson_problem.template build_mf_storage<double>(fine_level);
      LEVEL_MATRIX   level_matrix;
      level_matrix.initialize(mf_storage);
      const auto   subdomain_handler = poisson_problem.build_patch_storage(fine_level, mf_storage);
      const auto & schwarz_data      = poisson_problem.parameters.schwarz_smoother_data;
      const auto   schwarz_preconditioner =
        poisson_problem.build_schwarz_preconditioner(subdomain_handler, level_matrix, schwarz_data);
      typename SCHWARZ_SMOOTHER::AdditionalData smoother_data;
      smoother_data.number_of_smoothing_steps = schwarz_data.number_of_smoothing_steps;
      SCHWARZ_SMOOTHER schwarz_smoother;
      schwarz_smoother.initialize(level_matrix, schwarz_preconditioner, smoother_data);
      time.stop();
      timings_smooth.setup.push_back(time.get_last_lap_wall_time_data());

      //: smooth
      VECTOR tmp;
      mf_storage->initialize_dof_vector(tmp);
      fill_with_random_values(tmp);
      VECTOR dst = tmp;
      time.restart();
      for(unsigned subsample = 0; subsample < prm.n_subsamples_smooth; ++subsample)
        schwarz_smoother.step(dst, tmp);
      time.stop();
      timings_smooth.apply.push_back(time.get_last_lap_wall_time_data());
    }

    time.restart();
    poisson_problem.solve(gmg_preconditioner);
    time.stop();
    timings_total.apply.push_back(time.get_last_lap_wall_time_data());

    pp_data = poisson_problem.pp_data;
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::fstream fstream_log;
    fstream_log.open("vmult_" + filename + ".time", std::ios_base::out); // std::ios_base::app);
    fstream_log << write_table_to_string(pp_data, timings_vmult);
    fstream_log.close();

    fstream_log.open("smooth_" + filename + ".time", std::ios_base::out); // std::ios_base::app);
    fstream_log << write_table_to_string(pp_data, timings_smooth);
    fstream_log.close();

    fstream_log.open("solve_" + filename + ".time", std::ios_base::out); // std::ios_base::app);
    fstream_log << write_table_to_string(pp_data, timings_total);
    fstream_log.close();
  }
}

int
main(int argc, char * argv[])
{
  // deallog.depth_console(3);

  // *** init TBB and MPI
  constexpr unsigned int           max_threads = 1; // no multithreading !?
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static parameters
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;

  // *** run-time parameters
  TestParameter prms;
  if(argc > 1)
    prms.n_refinements = std::atoi(argv[1]);
  if(argc > 2)
    prms.n_cycles = std::atoi(argv[2]);

  test<dim, fe_degree>(prms);

  return 0;
}
