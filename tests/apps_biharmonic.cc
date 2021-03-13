
/*
 * Test functionality of Biharmonic::ModelProblem.
 *
 *  Created on: Mar 13, 2021
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "biharmonic_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace Biharmonic;



using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
using TestParamsCubic     = testing::Types<Util::NonTypeParams<2, 3>>;



template<typename T>
class TestModelProblem : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();


  void
  SetUp() override
  {
    ofs.open("apps_biharmonic.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const bool is_quadratic  = fe_degree == 2;
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, is_quadratic && is_first_proc);

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::vertex;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  test_compute_nondivfree_shape_functions()
  {
    equation_data.variant = EquationData::Variant::ClampedStreamPoiseuilleNoSlip;

    Biharmonic::ModelProblem<dim, fe_degree> biharmonic_problem(rt_parameters, equation_data);
    biharmonic_problem.pcout = pcout_owned;
    biharmonic_problem.make_grid();
    biharmonic_problem.setup_system();

    biharmonic_problem.stokes_problem->pcout         = pcout_owned;
    biharmonic_problem.stokes_problem->triangulation = biharmonic_problem.triangulation;
    biharmonic_problem.stokes_problem->setup_system();

    const auto [trafomatrix_rt_to_gradp, trafomatrix_rt_to_constp] =
      biharmonic_problem.compute_nondivfree_shape_functions();
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  RT::Parameter rt_parameters;
  EquationData  equation_data;
};



TYPED_TEST_SUITE_P(TestModelProblem);

TYPED_TEST_P(TestModelProblem, compute_nondivfree_shape_functions)
{
  using Fixture = TestModelProblem<TypeParam>;

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::test_compute_nondivfree_shape_functions();

  // Fixture::rt_parameters.mesh.n_refinements = 1;
  // Fixture::test_compute_nondivfree_shape_functions();
}

REGISTER_TYPED_TEST_SUITE_P(TestModelProblem, compute_nondivfree_shape_functions);

INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestModelProblem, TestParamsQuadratic);
INSTANTIATE_TYPED_TEST_SUITE_P(Cubic2D, TestModelProblem, TestParamsCubic);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_biharmonic.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
