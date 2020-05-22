
/*
 * Test Stokes integrators.
 *
 *  Created on: May 21, 2020
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "linelasticity_problem.h"
#include "stokes_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace Stokes;



template<typename T>
class TestStokesIntegrator : public testing::Test
{
protected:
  static constexpr int dim                 = T::template value<0>();
  static constexpr int fe_degree_p         = T::template value<1>();
  using StokesProblem                      = ModelProblem<dim, fe_degree_p>;
  static constexpr unsigned int macro_size = VectorizedArray<double>::size();


  void
  SetUp() override
  {
    ofs.open("apps_stokes_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout_owned              = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::vertex;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  check_system_matrix_velocity()
  {
    *pcout_owned << "//////////   STOKES PROBLEM" << std::endl;
    EquationData equation_data;
    auto         new_problem = std::make_shared<StokesProblem>(rt_parameters, equation_data);
    new_problem->fe_velocity = std::make_shared<FESystem<dim>>(FE_DGQ<dim>(fe_degree_p + 1), dim);
    new_problem->pcout       = pcout_owned;
    new_problem->make_grid();
    new_problem->setup_system_velocity();
    new_problem->assemble_system_velocity();
    stokes_problem = new_problem;
    stokes_problem->print_informations();

    *pcout_owned << "//////////   LINEAR ELASTICITY PROBLEM" << std::endl;
    LinElasticity::EquationData equation_data_elasticity;
    equation_data_elasticity.integrator_variant =
      LinElasticity::EquationData::IntegratorVariant::strain;
    equation_data_elasticity.ip_factor = 1.;
    using ElasticityProblem            = LinElasticity::ModelProblem<dim, fe_degree_p + 1>;
    const auto elasticity_problem =
      std::make_shared<ElasticityProblem>(*pcout_owned, rt_parameters, equation_data_elasticity);
    elasticity_problem->create_triangulation();
    elasticity_problem->assemble_matrix();
    elasticity_problem->print_informations();

    FullMatrix<double> A(stokes_problem->system_matrix_velocity.m());
    FullMatrix<double> Aref(elasticity_problem->sparse_matrix.m());
    ASSERT_EQ(A.m(), Aref.m());
    A.copy_from(stokes_problem->system_matrix_velocity);
    Aref.copy_from(elasticity_problem->sparse_matrix);
    compare_matrix(A, Aref);
  }

  void
  compare_matrix(const FullMatrix<double> & patch_matrix_full,
                 const FullMatrix<double> & other) const
  {
    Util::compare_matrix(patch_matrix_full, other, *pcout_owned);
  }


  void
  compare_inverse_matrix(const FullMatrix<double> & inverse_patch_matrix,
                         const FullMatrix<double> & other) const
  {
    Util::compare_inverse_matrix(inverse_patch_matrix, other, *pcout_owned);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  RT::Parameter                                  rt_parameters;
  std::shared_ptr<const StokesProblem>           stokes_problem;
  std::shared_ptr<const MatrixFree<dim, double>> mf_storage;
  std::shared_ptr<SubdomainHandler<dim, double>> subdomain_handler;
};



TYPED_TEST_SUITE_P(TestStokesIntegrator);

TYPED_TEST_P(TestStokesIntegrator, CheckSystemMatrixVelocity)
{
  using Fixture                                = TestStokesIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_system_matrix_velocity();
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_system_matrix_velocity();
}

REGISTER_TYPED_TEST_SUITE_P(TestStokesIntegrator, CheckSystemMatrixVelocity);

// using TestParamsConstant = testing::Types<Util::NonTypeParams<2, 0>>;
using TestParamsLinear    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
using TestParamsHighOrder = testing::Types<Util::NonTypeParams<2, 5>>;

// INSTANTIATE_TYPED_TEST_SUITE_P(Constant2D, TestStokesIntegrator, TestParamsConstant);
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestStokesIntegrator, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestStokesIntegrator, TestParamsQuadratic);
INSTANTIATE_TYPED_TEST_SUITE_P(HighOrder2D, TestStokesIntegrator, TestParamsHighOrder);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_stokes_integrator.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
