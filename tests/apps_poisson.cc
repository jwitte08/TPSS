/*
 * Tests all variants of Schwarz-type smoothers by checking iteration counts of
 * the Poisson problem (poisson_problem.h)
 *
 *  Created on: Nov 22, 2019
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include "poisson_problem.h"

#include "test_utilities.h"

template<int dim, int fe_degree>
class TestPoissonBase : public testing::Test
{
protected:
  static constexpr auto dof_layout = TPSS::DoFLayout::DGQ;
  using PoissonProblem             = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;

  struct Params
  {
    TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;
    double                bound            = 0;
  };

  virtual void
  SetUp() override
  {
    //: misc
    rt_parameters.n_cycles = 3;

    //: discretization
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = 4 - dim;
    rt_parameters.mesh.n_repetitions    = 2;

    //: multigrid
    const double damping_factor =
      TPSS::lookup_damping_factor(params.patch_variant, params.smoother_variant, dim);
    rt_parameters.multigrid.coarse_level = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant =
      CoarseGridParameter::SolverVariant::Iterative;
    rt_parameters.multigrid.coarse_grid.iterative_solver = "cg";
    rt_parameters.multigrid.coarse_grid.accuracy         = 1.e-12;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = params.smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = 1;
    rt_parameters.multigrid.pre_smoother.schwarz.symmetrize_smoothing = false;
    rt_parameters.multigrid.pre_smoother.schwarz.reverse_smoothing    = false;
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

    //: solver
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
    rt_parameters.solver.rel_tolerance        = 1.e-6;
    rt_parameters.reset_solver_variant();
  }

  void
  run(const Params & params_in)
  {
    params = params_in;
    SetUp();

    PoissonProblem poisson_problem{rt_parameters};
    std::ofstream  ofs("apps_poisson.log");
    const bool     is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    poisson_problem.pcout        = std::make_shared<ConditionalOStream>(ofs, is_first_proc);
    poisson_problem.run();

    const auto & pp_data = poisson_problem.pp_data;
    for(const auto n_iter : pp_data.n_iterations_system)
      EXPECT_LT(n_iter, params.bound) << "Number of iterations n_iter exceed the bound.";
  }

  void
  verify_iteration_counts()
  {
    /// ACP
    {
      const std::map<std::pair<int, int>, double> bounds          = {{std::make_pair(2, 4), 11.0},
                                                            {std::make_pair(2, 7), 13.0},
                                                            {std::make_pair(3, 2), 12.0},
                                                            {std::make_pair(3, 5), 14.0}};
      const auto                                  bound_of_n_iter = bounds.at({dim, fe_degree});
      Params params = {TPSS::PatchVariant::cell, TPSS::SmootherVariant::additive, bound_of_n_iter};
      this->run(params);
    }

    /// MCP
    {
      const std::map<std::pair<int, int>, double> bounds          = {{std::make_pair(2, 4), 8.0},
                                                            {std::make_pair(2, 7), 10.0},
                                                            {std::make_pair(3, 2), 8.0},
                                                            {std::make_pair(3, 5), 11.0}};
      const auto                                  bound_of_n_iter = bounds.at({dim, fe_degree});
      Params                                      params          = {TPSS::PatchVariant::cell,
                       TPSS::SmootherVariant::multiplicative,
                       bound_of_n_iter};
      this->run(params);
    }

    /// MVP
    {
      const std::map<std::pair<int, int>, double> bounds          = {{std::make_pair(2, 4), 3.0},
                                                            {std::make_pair(2, 7), 3.0},
                                                            {std::make_pair(3, 2), 3.0},
                                                            {std::make_pair(3, 5), 3.0}};
      const auto                                  bound_of_n_iter = bounds.at({dim, fe_degree});
      Params                                      params          = {TPSS::PatchVariant::vertex,
                       TPSS::SmootherVariant::multiplicative,
                       bound_of_n_iter};
      this->run(params);
    }

    /// AVP
    {
      const std::map<std::pair<int, int>, double> bounds          = {{std::make_pair(2, 4), 15.0},
                                                            {std::make_pair(2, 7), 17.0},
                                                            {std::make_pair(3, 2), 21.0},
                                                            {std::make_pair(3, 5), 24.0}};
      const auto                                  bound_of_n_iter = bounds.at({dim, fe_degree});
      Params                                      params          = {TPSS::PatchVariant::vertex,
                       TPSS::SmootherVariant::additive,
                       bound_of_n_iter};
      this->run(params);
    }
  }

  RT::Parameter rt_parameters;
  Params        params;
};



template<typename T>
class TestPoissonRun : public TestPoissonBase<T::template value<0>(), T::template value<1>()>
{
protected:
  using Base = TestPoissonBase<T::template value<0>(), T::template value<1>()>;
};

TYPED_TEST_SUITE_P(TestPoissonRun);
TYPED_TEST_P(TestPoissonRun, VaryDimAndDegree)
{
  using Base = TestPoissonRun<TypeParam>;
  this->verify_iteration_counts();
}

TYPED_TEST_P(TestPoissonRun, DoFsOnTheFly)
{
  using Base = TestPoissonRun<TypeParam>;
  Base::rt_parameters.multigrid.pre_smoother.schwarz.caching_strategy =
    TPSS::CachingStrategy::OnTheFly;
  Base::rt_parameters.multigrid.post_smoother.schwarz.caching_strategy =
    TPSS::CachingStrategy::OnTheFly;
  Base::rt_parameters.n_cycles = 1;
  this->verify_iteration_counts();
}

REGISTER_TYPED_TEST_SUITE_P(TestPoissonRun, VaryDimAndDegree, DoFsOnTheFly);

using TestParams2D = testing::Types<Util::NonTypeParams<2, 4>, Util::NonTypeParams<2, 7>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensions, TestPoissonRun, TestParams2D);

using TestParams3D = testing::Types<Util::NonTypeParams<3, 2>, Util::NonTypeParams<3, 5>>;
INSTANTIATE_TYPED_TEST_SUITE_P(ThreeDimensions, TestPoissonRun, TestParams3D);


int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
