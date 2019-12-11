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
    linelasticity_problem.params.patch_variant    = params.patch_variant;
    linelasticity_problem.params.smoother_variant = params.smoother_variant;
    linelasticity_problem.initialize();
    linelasticity_problem.run();
    const auto & pp_data = linelasticity_problem.pp_data;

    for(const auto n_iter : pp_data.n_iterations_system)
      EXPECT_LT(n_iter, params.bound) << "Number of iterations n_iter exceed the bound.\n"
                                      << linelasticity_problem.rt_parameters.to_string();
    // std::cout << pp_data.n_iterations_system.size() << std::endl;
  }
};

TYPED_TEST_SUITE_P(TestLinElasticityDiagOnly);
TYPED_TEST_P(TestLinElasticityDiagOnly, VaryDimAndDegree)
{
  using Fixture = TestLinElasticityDiagOnly<TypeParam>;

  { // ACP
    const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 4), 25.0},
                                                          {std::make_pair(2, 7), 32.0}};
    const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
    typename Fixture::Params params                    = {TPSS::PatchVariant::cell,
                                       TPSS::SmootherVariant::additive,
                                       bound_of_n_iter};
    Fixture::check(params);
  }

  { // MCP
    const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 4), 20.0},
                                                          {std::make_pair(2, 7), 25.0}};
    const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
    typename Fixture::Params params                    = {TPSS::PatchVariant::cell,
                                       TPSS::SmootherVariant::multiplicative,
                                       bound_of_n_iter};
    Fixture::check(params);
  }

  { // AVP
    const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 4), 29.0},
                                                          {std::make_pair(2, 7), 32.0}};
    const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
    typename Fixture::Params params                    = {TPSS::PatchVariant::vertex,
                                       TPSS::SmootherVariant::additive,
                                       bound_of_n_iter};
    Fixture::check(params);
  }

  { // MVP
    const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 4), 7.0},
                                                          {std::make_pair(2, 7), 7.0}};
    const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
    typename Fixture::Params params                    = {TPSS::PatchVariant::vertex,
                                       TPSS::SmootherVariant::multiplicative,
                                       bound_of_n_iter};
    Fixture::check(params);
  }
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityDiagOnly, VaryDimAndDegree);

using TestParamsDiagOnly = testing::Types<Util::NonTypeParams<2, 4>, Util::NonTypeParams<2, 7>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensions, TestLinElasticityDiagOnly, TestParamsDiagOnly);



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
    linelasticity_problem.params.patch_variant    = params.patch_variant;
    linelasticity_problem.params.smoother_variant = params.smoother_variant;
    linelasticity_problem.initialize();
    linelasticity_problem.run();
    const auto & pp_data = linelasticity_problem.pp_data;

    for(const auto n_iter : pp_data.n_iterations_system)
      EXPECT_LT(n_iter, params.bound) << "Number of iterations n_iter exceed the bound.\n"
                                      << linelasticity_problem.rt_parameters.to_string();
  }
};

TYPED_TEST_SUITE_P(TestLinElasticityExact);
TYPED_TEST_P(TestLinElasticityExact, VaryDimAndDegree)
{
  using Fixture = TestLinElasticityExact<TypeParam>;

  { // ACP
    const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 4), 24.0},
                                                          {std::make_pair(2, 7), 32.0}};
    const auto               bound_of_n_iter           = bounds.at({Fixture::dim, Fixture::degree});
    typename Fixture::Params params;
    params.patch_variant    = TPSS::PatchVariant::cell;
    params.smoother_variant = TPSS::SmootherVariant::additive;
    params.bound            = bound_of_n_iter;
    Fixture::check(params);
  }

  /// NOTE mcp requires more iterations than acp ??
  // { // MCP
  //   const std::map<std::pair<int, int>, double> bounds = {{std::make_pair(2, 4), 20.0},
  //                                                         {std::make_pair(2, 7), 25.0}};
  //   const auto               bound_of_n_iter           = bounds.at({Fixture::dim,
  //   Fixture::degree}); typename Fixture::Params params                    =
  //   {TPSS::PatchVariant::cell,
  //                                      TPSS::SmootherVariant::multiplicative,
  //                                      bound_of_n_iter};
  //   Fixture::check(params);
  // }
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityExact, VaryDimAndDegree);

using TestParamsExact = testing::Types<Util::NonTypeParams<2, 4> /*, Util::NonTypeParams<2, 7>*/>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensions, TestLinElasticityExact, TestParamsExact);



int
main(int argc, char ** argv)
{
  std::ofstream ofs("apps_linelasticity.log", std::ios_base::out);
  ofs.close();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
