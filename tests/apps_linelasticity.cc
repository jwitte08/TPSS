/*
 * Test the linear elasticity model problem in linelasticity_problem.h.
 *
 *  Created on: Nov 26, 2019
 *      Author: witte
 */

#include <gtest/gtest.h>

#include "apps_linelasticity.h"
#include "test_utilities.h"

template<typename T>
class TestLinElasticityDiagOnly : public testing::Test
{
protected:
  struct Params
  {
    TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;
    double                bound            = 0;
  };

  static constexpr int dim    = T::template value<0>();
  static constexpr int degree = T::template value<1>();
  using ModelProblem          = TestLinElasticity<dim, degree, double>;

  void
  check(const Params & params)
  {
    ModelProblem linelasticity_problem;
    linelasticity_problem.initialize();
    auto & rt_parameters = linelasticity_problem.rt_parameters;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant     = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant  = params.smoother_variant;
    rt_parameters.multigrid.post_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.post_smoother.schwarz.smoother_variant = params.smoother_variant;
    rt_parameters.reset_damping_factor(dim);

    linelasticity_problem.run();
    const auto & pp_data = linelasticity_problem.pp_data;

    for(const auto n_iter : pp_data.n_iterations_system)
      EXPECT_LT(n_iter, params.bound) << "Number of iterations n_iter exceed the bound.\n"
                                      << linelasticity_problem.rt_parameters.to_string();
  }
};

TYPED_TEST_SUITE_P(TestLinElasticityDiagOnly);
TYPED_TEST_P(TestLinElasticityDiagOnly, ACP)
{
  using Fixture                                      = TestLinElasticityDiagOnly<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 28.0},
                                                        {std::make_pair(2, 3), 25.0},
                                                        {std::make_pair(2, 7), 32.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params                    = {TPSS::PatchVariant::cell,
                                     TPSS::SmootherVariant::additive,
                                     bound_of_n_iter};
  Fixture::check(params);
}


TYPED_TEST_P(TestLinElasticityDiagOnly, MCP)
{
  using Fixture                                      = TestLinElasticityDiagOnly<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 20.0},
                                                        {std::make_pair(2, 3), 19.0},
                                                        {std::make_pair(2, 7), 25.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params                    = {TPSS::PatchVariant::cell,
                                     TPSS::SmootherVariant::multiplicative,
                                     bound_of_n_iter};
  Fixture::check(params);
}


TYPED_TEST_P(TestLinElasticityDiagOnly, AVP)
{
  using Fixture                                      = TestLinElasticityDiagOnly<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 36.0},
                                                        {std::make_pair(2, 3), 29.0},
                                                        {std::make_pair(2, 7), 32.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params                    = {TPSS::PatchVariant::vertex,
                                     TPSS::SmootherVariant::additive,
                                     bound_of_n_iter};
  Fixture::check(params);
}


TYPED_TEST_P(TestLinElasticityDiagOnly, MVP)
{
  using Fixture                                      = TestLinElasticityDiagOnly<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 9.0},
                                                        {std::make_pair(2, 3), 7.0},
                                                        {std::make_pair(2, 7), 7.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params                    = {TPSS::PatchVariant::vertex,
                                     TPSS::SmootherVariant::multiplicative,
                                     bound_of_n_iter};
  Fixture::check(params);
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityDiagOnly, ACP, MCP, AVP, MVP);

using TestParamsLinear      = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsHigherOrder = testing::Types<Util::NonTypeParams<2, 3>, Util::NonTypeParams<2, 7>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLinElasticityDiagOnly, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestLinElasticityDiagOnly, TestParamsHigherOrder);



template<typename T>
class TestLinElasticityExact : public testing::Test
{
protected:
  struct Params
  {
    TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;
    double                bound            = 0;
  };

  static constexpr int dim    = T::template value<0>();
  static constexpr int degree = T::template value<1>();
  using ModelProblem =
    TestLinElasticity<dim, degree, double, Tensors::BlockMatrix<dim, VectorizedArray<double>>>;

  void
  check(const Params & params)
  {
    ModelProblem linelasticity_problem;
    linelasticity_problem.initialize();
    auto & rt_parameters = linelasticity_problem.rt_parameters;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant     = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant  = params.smoother_variant;
    rt_parameters.multigrid.post_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.post_smoother.schwarz.smoother_variant = params.smoother_variant;
    rt_parameters.reset_damping_factor(dim);

    linelasticity_problem.run();
    const auto & pp_data = linelasticity_problem.pp_data;

    for(const auto n_iter : pp_data.n_iterations_system)
      EXPECT_LT(n_iter, params.bound) << "Number of iterations n_iter exceed the bound.\n"
                                      << linelasticity_problem.rt_parameters.to_string();
  }
};

TYPED_TEST_SUITE_P(TestLinElasticityExact);
TYPED_TEST_P(TestLinElasticityExact, ACP)
{
  using Fixture                                      = TestLinElasticityExact<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 27.0},
                                                        {std::make_pair(2, 3), 25.0},
                                                        {std::make_pair(2, 7), 30.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params;
  params.patch_variant    = TPSS::PatchVariant::cell;
  params.smoother_variant = TPSS::SmootherVariant::additive;
  params.bound            = bound_of_n_iter;
  Fixture::check(params);
}


TYPED_TEST_P(TestLinElasticityExact, MCP)
{
  using Fixture                                      = TestLinElasticityExact<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 15.0},
                                                        {std::make_pair(2, 3), 18.0},
                                                        {std::make_pair(2, 7), 23.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params;
  params.patch_variant    = TPSS::PatchVariant::cell;
  params.smoother_variant = TPSS::SmootherVariant::multiplicative;
  params.bound            = bound_of_n_iter;
  Fixture::check(params);
}


TYPED_TEST_P(TestLinElasticityExact, MVP)
{
  using Fixture                                      = TestLinElasticityExact<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 7.0},
                                                        {std::make_pair(2, 3), 5.0},
                                                        {std::make_pair(2, 7), 4.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params;
  params.patch_variant    = TPSS::PatchVariant::vertex;
  params.smoother_variant = TPSS::SmootherVariant::multiplicative;
  params.bound            = bound_of_n_iter;
  Fixture::check(params);
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityExact, ACP, MCP, MVP);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLinElasticityExact, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestLinElasticityExact, TestParamsHigherOrder);



template<typename T>
class TestLinElasticityFast : public testing::Test
{
protected:
  struct Params
  {
    TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;
    double                bound            = 0;
  };

  static constexpr int dim    = T::template value<0>();
  static constexpr int degree = T::template value<1>();
  using ModelProblem =
    TestLinElasticity<dim,
                      degree,
                      double,
                      Tensors::BlockMatrix<dim, VectorizedArray<double>, /*fast*/ 0>>;

  void
  check(const Params & params)
  {
    ModelProblem linelasticity_problem;
    linelasticity_problem.initialize();
    auto & rt_parameters = linelasticity_problem.rt_parameters;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant     = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant  = params.smoother_variant;
    rt_parameters.multigrid.post_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.post_smoother.schwarz.smoother_variant = params.smoother_variant;
    rt_parameters.reset_damping_factor(dim);

    linelasticity_problem.run();
    const auto & pp_data = linelasticity_problem.pp_data;

    for(const auto n_iter : pp_data.n_iterations_system)
      EXPECT_LT(n_iter, params.bound) << "Number of iterations n_iter exceed the bound.\n"
                                      << linelasticity_problem.rt_parameters.to_string();
  }
};

TYPED_TEST_SUITE_P(TestLinElasticityFast);
TYPED_TEST_P(TestLinElasticityFast, ACP)
{
  using Fixture                                      = TestLinElasticityFast<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 33.0},
                                                        {std::make_pair(2, 3), 25.0},
                                                        {std::make_pair(2, 7), 31.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params;
  params.patch_variant    = TPSS::PatchVariant::cell;
  params.smoother_variant = TPSS::SmootherVariant::additive;
  params.bound            = bound_of_n_iter;
  Fixture::check(params);
}

TYPED_TEST_P(TestLinElasticityFast, MCP)
{
  using Fixture                                      = TestLinElasticityFast<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 16.0},
                                                        {std::make_pair(2, 3), 18.0},
                                                        {std::make_pair(2, 7), 23.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params;
  params.patch_variant    = TPSS::PatchVariant::cell;
  params.smoother_variant = TPSS::SmootherVariant::multiplicative;
  params.bound            = bound_of_n_iter;
  Fixture::check(params);
}


TYPED_TEST_P(TestLinElasticityFast, MVP)
{
  using Fixture                                      = TestLinElasticityFast<TypeParam>;
  const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 1), 7.0},
                                                        {std::make_pair(2, 3), 5.0},
                                                        {std::make_pair(2, 7), 5.0}};
  const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
  typename Fixture::Params params;
  params.patch_variant    = TPSS::PatchVariant::vertex;
  params.smoother_variant = TPSS::SmootherVariant::multiplicative;
  params.bound            = bound_of_n_iter;
  Fixture::check(params);
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityFast, ACP, MCP, MVP);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLinElasticityFast, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestLinElasticityFast, TestParamsHigherOrder);



int
main(int argc, char ** argv)
{
  std::ofstream ofs("apps_linelasticity.log", std::ios_base::out);
  ofs.close();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
