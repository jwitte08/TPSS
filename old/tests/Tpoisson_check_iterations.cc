/*
 * Tests all variants of Schwarz-type smoothers by checking iteration counts of
 * the Poisson problem (poisson.h)
 *
 *  Created on: Nov 22, 2019
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "poisson.h"

struct TestParameter
{
  TPSS::PatchVariant                 patch_variant    = TPSS::PatchVariant::cell;
  TPSS::SmootherVariant              smoother_variant = TPSS::SmootherVariant::additive;
  std::string                        solver_variant   = "cg"; // see SolverSelector
  CoarseGridParameter::SolverVariant coarse_grid_variant =
    CoarseGridParameter::SolverVariant::IterativeAcc;
  int                            coarse_level         = 0;
  double                         coarse_grid_accuracy = 1.e-12;
  double                         cg_reduction         = 1.e-8;
  MeshParameter::GeometryVariant geometry_variant     = MeshParameter::GeometryVariant::Cube;
  unsigned                       n_cycles             = 2;
  unsigned                       n_refinements        = 2;
  unsigned                       n_repetitions        = 2;
  unsigned                       n_smoothing_steps    = 1;
  bool                           reverse_smoothing    = true;
  bool                           symmetric_smoothing  = false;
  bool                           user_coloring        = true;

  // std::string
  // to_string() const
  // {
  //   std::ostringstream oss;
  //   oss << Util::parameter_to_fstring("n_refinements:", n_refinements);
  //   oss << Util::parameter_to_fstring("n_repetitions:", n_repetitions);
  //   oss << Util::parameter_to_fstring(solver_variant + " solver reduction:", cg_reduction);
  //   return oss.str();
  // }
};


template<int dim, int fe_degree>
struct Test
{
  Test(ConditionalOStream & pcout_in) : pcout(pcout_in)
  {
  }

  void
  run(const TestParameter & prms, const unsigned bound)
  {
    RT::Parameter rt_parameters;

    //: discretization
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = prms.n_refinements;
    rt_parameters.mesh.n_repetitions    = prms.n_repetitions;

    //: solver
    rt_parameters.solver.variant              = prms.solver_variant;
    rt_parameters.solver.rel_tolerance        = prms.cg_reduction;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;

    //: multigrid
    const double damping_factor =
      TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
    rt_parameters.multigrid.coarse_level                 = prms.coarse_level;
    rt_parameters.multigrid.coarse_grid.solver_variant   = prms.coarse_grid_variant;
    rt_parameters.multigrid.coarse_grid.iterative_solver = prms.solver_variant;
    rt_parameters.multigrid.coarse_grid.accuracy         = prms.coarse_grid_accuracy;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = prms.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = prms.smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.manual_coloring      = prms.user_coloring;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = prms.n_smoothing_steps;
    rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(6, fe_degree + 1);
    rt_parameters.multigrid.pre_smoother.schwarz.symmetrize_smoothing = prms.symmetric_smoothing;
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = prms.reverse_smoothing;
    rt_parameters.reset_solver_variant(); // <--- test

    //: misc
    rt_parameters.compressed = false;
    rt_parameters.n_cycles   = prms.n_cycles;

    Timer time;
    time.restart();
    using PoissonProblem = typename Poisson::ModelProblem<dim, fe_degree, double>;
    PoissonProblem poisson_problem{rt_parameters};
    poisson_problem.run();

    // *** check if iteration count is below bound
    const auto & pp_data = poisson_problem.pp_data;
    for(const auto count : pp_data.n_iterations_system)
    {
      const bool success = count < static_cast<typename std::decay<decltype(count)>::type>(bound);
      std::ostringstream oss;
      oss << "Number of iterations " << count << " exceeds bound " << bound << "!";
      // pcout << oss.str() << " time: " << time.wall_time() << std::endl;
      AssertThrow(success, ExcMessage(oss.str()));
    }
  }

  ConditionalOStream & pcout;
};

int
main(int argc, char * argv[])
{
  using namespace dealii;

  const unsigned                   max_threads = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);
  const bool         is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream pcout(std::cout, is_first_proc);

  {
    TestParameter prms_basic;
    prms_basic.patch_variant    = TPSS::PatchVariant::cell;
    prms_basic.smoother_variant = TPSS::SmootherVariant::additive;
    {
      TestParameter prms = prms_basic;
      Test<2, 3>    tester2D(pcout);
      prms.n_cycles              = 3;
      const unsigned upper_bound = 15;
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<2, 2>    tester2D(pcout);
      prms.n_cycles              = 1;
      prms.n_repetitions         = 1;
      prms.n_refinements         = 0;
      prms.coarse_level          = 0;
      const unsigned upper_bound = 1; // <--- test coarse solver
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<3, 2>    tester3D(pcout);
      prms.n_cycles              = 2;
      prms.n_repetitions         = 1;
      const unsigned upper_bound = 16;
      tester3D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<2, 6>    tester2D(pcout); // <--- higher order
      prms.n_cycles              = 3;
      prms.n_repetitions         = 1;
      prms.n_smoothing_steps     = 3; // <---
      const unsigned upper_bound = 8;
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<3, 5>    tester3D(pcout); // <--- higher order
      prms.n_cycles              = 2;
      prms.n_smoothing_steps     = 1;
      prms.n_repetitions         = 1; // <---
      prms.coarse_level          = 1; // <---
      const unsigned upper_bound = 21;
      tester3D.run(prms, upper_bound);
    }
  }

  {
    TestParameter prms_basic;
    prms_basic.patch_variant    = TPSS::PatchVariant::cell;
    prms_basic.smoother_variant = TPSS::SmootherVariant::multiplicative;
    {
      TestParameter prms = prms_basic;
      Test<2, 3>    tester2D(pcout);
      prms.n_cycles              = 3;
      const unsigned upper_bound = 11;
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<3, 5>    tester3D(pcout);
      prms.n_cycles              = 2;
      prms.n_refinements         = 1;
      const unsigned upper_bound = 15;
      tester3D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<2, 5>    tester2D(pcout);
      prms.n_cycles              = 4;
      prms.n_repetitions         = 1;
      prms.n_smoothing_steps     = 2;
      prms.reverse_smoothing     = false; // <--- GMRES
      const unsigned upper_bound = 7;
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<2, 4>    tester2D(pcout);
      prms.n_cycles              = 4;
      prms.n_repetitions         = 1;
      prms.reverse_smoothing     = false;
      prms.symmetric_smoothing   = true; // <--- CG
      const unsigned upper_bound = 8;
      tester2D.run(prms, upper_bound);
    }
  }

  // TODO nonoverlapping AVP
  {
    TestParameter prms_basic;
    prms_basic.patch_variant    = TPSS::PatchVariant::vertex;
    prms_basic.smoother_variant = TPSS::SmootherVariant::additive;
    {
      TestParameter prms = prms_basic;
      Test<2, 5>    tester2D(pcout);
      prms.n_cycles              = 3;
      const unsigned upper_bound = 21;
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<3, 4>    tester3D(pcout);
      prms.n_cycles              = 2;
      prms.n_refinements         = 1;
      prms.n_smoothing_steps     = 3;
      const unsigned upper_bound = 13;
      tester3D.run(prms, upper_bound);
    }
  }

  {
    TestParameter prms_basic;
    prms_basic.patch_variant    = TPSS::PatchVariant::vertex;
    prms_basic.smoother_variant = TPSS::SmootherVariant::multiplicative;
    {
      TestParameter prms = prms_basic;
      Test<2, 5>    tester2D(pcout);
      prms.n_cycles              = 4;
      const unsigned upper_bound = 4;
      tester2D.run(prms, upper_bound);
      if(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
      {
        prms.user_coloring = false; // <--- graph coloring
        tester2D.run(prms, upper_bound);
      }
    }
    {
      TestParameter prms = prms_basic;
      Test<3, 3>    tester3D(pcout);
      prms.n_cycles              = 3;
      prms.n_refinements         = 1;
      const unsigned upper_bound = 4;
      tester3D.run(prms, upper_bound);
      if(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
      {
        prms.user_coloring = false; // <--- graph coloring
        tester3D.run(prms, upper_bound);
      }
    }
    {
      TestParameter prms = prms_basic;
      Test<2, 7>    tester2D(pcout);
      prms.n_cycles              = 2;
      prms.reverse_smoothing     = false;
      prms.symmetric_smoothing   = true; // <--- CG
      const unsigned upper_bound = 2;
      tester2D.run(prms, upper_bound);
    }
    {
      TestParameter prms = prms_basic;
      Test<3, 3>    tester3D(pcout);
      prms.n_cycles              = 2;
      prms.n_refinements         = 1;
      prms.n_smoothing_steps     = 3;
      prms.reverse_smoothing     = false; // <--- GMRES
      const unsigned upper_bound = 2;
      tester3D.run(prms, upper_bound);
    }
  }

  return 0;
}
