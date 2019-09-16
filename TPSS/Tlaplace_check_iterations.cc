/*
 * Checks the iteration counts of cell & vertex patches on a Cartesian mesh.
 *
 *  Created on: Aug 20, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

template<int dim, int fe_degree>
struct Test
{
  Test(ConditionalOStream & pcout_in) : pcout(pcout_in)
  {
  }

  void
  operator()(const unsigned              fine_level,
             const unsigned              coarse_level,
             const TPSS::PatchVariant    patch_variant,
             const TPSS::SmootherVariant smoother_variant,
             const unsigned              bound,
             const unsigned              n_smoothing_steps            = 1,
             const bool                  manual_coloring              = false,
             const bool                  non_overlapping_vertex_patch = false)
  {
    Laplace::Parameter parameters;
    parameters.write_post_process_output              = true;
    parameters.n_refines                              = fine_level;
    parameters.n_cycles                               = 6 - dim;
    parameters.coarse_level                           = coarse_level;
    parameters.schwarz_smoother_data.patch_variant    = patch_variant;
    parameters.schwarz_smoother_data.smoother_variant = smoother_variant;
    if(smoother_variant != TPSS::SmootherVariant::additive)
      parameters.solver_variant = Laplace::Parameter::SolverVariant::GMRES;
    parameters.schwarz_smoother_data.number_of_smoothing_steps = n_smoothing_steps;
    parameters.schwarz_smoother_data.manual_coloring           = manual_coloring;
    if(patch_variant == TPSS::PatchVariant::vertex)
      parameters.non_overlapping = non_overlapping_vertex_patch;
    parameters.mg_coarse_grid_variant = Laplace::Parameter::CoarseGridVariant::ChebyshevAccurate;
    parameters.mg_coarse_chebyshev_reduction = /*0.05;*/ 1.e-8;

    Laplace::MatrixOperator<dim, fe_degree> laplace_problem{pcout, parameters};
    laplace_problem.run();

    // *** check if iteration count is below bound
    const auto & post_process_data = laplace_problem.pp_data;
    for(const auto count : post_process_data.n_iterations_system)
      AssertThrow(count < static_cast<typename std::decay<decltype(count)>::type>(bound),
                  ExcMessage("Iteration count exceeds the valid bound!"));
  }

  ConditionalOStream & pcout;
};

int
main(int argc, char * argv[])
{
  using namespace dealii;

  // *** init MPI
  const unsigned                   max_threads = 1; // bounds are for serial runs
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  print_row_variable(
    pcout, 2, "", 43, "MultithreadInfo - # of threads:", MultithreadInfo::n_threads());
  pcout << std::endl << std::endl;

  // *** test CASE:   cell patch, additive
  {
    pcout << "TEST: cell patch, additive: \n\n\n";
    const unsigned fine_level   = 3;
    const unsigned coarse_level = 0;
    const unsigned upper_bound  = 15;
    Test<2, 3>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::cell,
                      TPSS::SmootherVariant::additive,
                      upper_bound);
  }
  {
    pcout << "TEST: cell patch, additive: \n\n\n";
    const unsigned fine_level   = 2;
    const unsigned coarse_level = 0;
    const unsigned upper_bound  = 16;
    Test<3, 2>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::cell,
                      TPSS::SmootherVariant::additive,
                      upper_bound);
  }
  {
    pcout << "TEST: cell patch, additive: \n\n\n";
    const unsigned fine_level   = 2;
    const unsigned coarse_level = 1;
    const unsigned upper_bound  = 22;
    Test<3, 7>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::cell,
                      TPSS::SmootherVariant::additive,
                      upper_bound);
  }

  // *** test CASE:   cell patch, multiplicative
  {
    pcout << "TEST: cell patch, multiplicative: \n\n\n";
    const unsigned fine_level      = 3; // initial fine level
    const unsigned coarse_level    = 0;
    const unsigned upper_bound     = 5;
    const unsigned smoothing_steps = 3;
    Test<2, 3>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::cell,
                      TPSS::SmootherVariant::multiplicative,
                      upper_bound,
                      smoothing_steps);
  }
  {
    pcout << "TEST: cell patch, multiplicative: \n\n\n";
    const unsigned fine_level   = 2;
    const unsigned coarse_level = 0;
    const unsigned upper_bound  = 8;
    Test<3, 2>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::cell,
                      TPSS::SmootherVariant::multiplicative,
                      upper_bound);
  }

  // *** test CASE:   vertex patch, additive
  {
    pcout << "TEST: non-overlapping, vertex patch, additive: \n\n\n";
    const unsigned fine_level      = 3;
    const unsigned coarse_level    = 1;
    const unsigned upper_bound     = 8;
    const unsigned smoothing_steps = 2;
    Test<2, 3>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::additive,
                      upper_bound,
                      smoothing_steps,
                      /*manual_coloring*/ false,
                      /*non_overlapping_vpatch*/ true);
  }
  {
    pcout << "TEST: non-overlapping, vertex patch, additive: \n\n\n";
    const unsigned fine_level      = 2;
    const unsigned coarse_level    = 1;
    const unsigned upper_bound     = 8;
    const unsigned smoothing_steps = 2;
    Test<3, 2>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::additive,
                      upper_bound,
                      smoothing_steps,
                      /*manual_coloring*/ false,
                      /*non_overlapping_vpatch*/ true);
  }

  // *** test CASE:   vertex patch, additive
  {
    pcout << "TEST: vertex patch, additive: \n\n\n";
    const unsigned fine_level   = 3;
    const unsigned coarse_level = 1;
    const unsigned upper_bound  = 19;
    Test<2, 3>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::additive,
                      upper_bound);
  }
  {
    pcout << "TEST: vertex patch, additive: \n\n\n";
    const unsigned fine_level      = 2;
    const unsigned coarse_level    = 1;
    const unsigned upper_bound     = 17;
    const unsigned smoothing_steps = 2;
    Test<3, 2>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::additive,
                      upper_bound,
                      smoothing_steps);
  }

  // *** test CASE:   vertex patch, multiplicative (make_graph_coloring)
  {
    pcout << "TEST: graph-colored, vertex patch, multiplicative: \n\n\n";
    const unsigned fine_level   = 3; // initial fine level
    const unsigned coarse_level = 1;
    const unsigned upper_bound  = 3;
    Test<2, 3>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::multiplicative,
                      upper_bound);
  }
  {
    pcout << "TEST: graph-colored, vertex patch, multiplicative: \n\n\n";
    const unsigned fine_level   = 2;
    const unsigned coarse_level = 1;
    const unsigned upper_bound  = 3;
    Test<3, 2>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::multiplicative,
                      upper_bound);
  }

  // *** test CASE:   vertex patch, multiplicative (manual coloring)
  {
    pcout << "TEST: manually-colored, vertex patch, multiplicative: \n\n\n";
    const unsigned fine_level   = 3; // initial fine level
    const unsigned coarse_level = 1;
    const unsigned upper_bound  = 3;
    Test<2, 3>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::multiplicative,
                      upper_bound,
                      /*smoothing steps*/ 1,
                      /*manual_coloring*/ true);
  }
  {
    pcout << "TEST: manually-colored, vertex patch, multiplicative: \n\n\n";
    const unsigned fine_level   = 2;
    const unsigned coarse_level = 1;
    const unsigned upper_bound  = 3;
    Test<3, 2>{pcout}(fine_level,
                      coarse_level,
                      TPSS::PatchVariant::vertex,
                      TPSS::SmootherVariant::multiplicative,
                      upper_bound,
                      /*smoothing steps*/ 1,
                      /*manual_coloring*/ true);
  }

  return 0;
}
