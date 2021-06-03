/*
 * Test Stokes integrators.
 *
 *  Created on: May 21, 2020
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "stokes.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace Stokes;



template<typename T>
class TestBase : public testing::Test
{
protected:
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::RT;
  static constexpr int             dim          = T::template value<0>();
  static constexpr int             fe_degree_p  = T::template value<1>();
  static constexpr int             fe_degree_v  = fe_degree_p;


  void
  SetUp() override
  {
    ofs.open("apps_stokes_raviartthomas_all.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    // const bool is_higher_order = fe_degree_v > 2;
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, /*!is_higher_order &&*/ is_first_proc);

    const auto patch_variant    = TPSS::PatchVariant::vertex;
    const auto smoother_variant = TPSS::SmootherVariant::additive;
    const auto damping          = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    options.setup(/*GMG_CG*/ 5, damping, patch_variant, smoother_variant);
  }


  void
  TearDown() override
  {
    ofs.close();
  }



  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  StokesFlow<dim, fe_degree_p> options;
  EquationData                 equation_data;
};



template<typename T>
class TestSimplifiedRaviartThomas : public TestBase<T>
{
protected:
  static constexpr bool is_simplified = true;

  using Base = TestBase<T>;
  using Base::dim;
  using Base::fe_degree_p;
  using StokesProblem = ModelProblem<dim, fe_degree_p, Method::RaviartThomas, is_simplified>;

  void
  check_localsvd()
  {
    using MatrixIntegrator  = VelocityPressure::LMW::MatrixIntegrator<dim,
                                                                     fe_degree_p,
                                                                     double,
                                                                     StokesProblem::dof_layout_v,
                                                                     StokesProblem::fe_degree_v,
                                                                     is_simplified>;
    using local_matrix_type = typename MatrixIntegrator::matrix_type;

    stokes_problem = std::make_shared<StokesProblem>(Base::options.prms, Base::equation_data);
    stokes_problem->pcout = Base::pcout_owned;
    stokes_problem->make_grid();
    stokes_problem->setup_system();
    const auto mgc = stokes_problem->make_multigrid_velocity_pressure();
    // stokes_problem->print_informations();

    ASSERT_TRUE(mgc->mg_schwarz_smoother_pre) << "Is not initialized.";

    const auto   level             = stokes_problem->max_level();
    const auto   mgss              = mgc->mg_schwarz_smoother_pre;
    const auto   subdomain_handler = mgss->get_subdomain_handler(level);
    const auto & partition_data    = subdomain_handler->get_partition_data();
    const TrilinosWrappers::BlockSparseMatrix & level_matrix = mgc->mg_matrices[level];

    const auto                     n_subdomains = partition_data.n_subdomains();
    std::vector<local_matrix_type> local_matrices(n_subdomains);

    MatrixIntegrator integrator;
    integrator.initialize(Base::equation_data);
    integrator.assemble_subspace_inverses(*subdomain_handler,
                                          local_matrices,
                                          level_matrix,
                                          partition_data.get_patch_range());

    TPSS::PatchDoFWorker<dim, double> patch_dof_worker_v(subdomain_handler->get_dof_info());

    for(auto patch_index = 0U; patch_index < local_matrices.size(); ++patch_index)
    {
      const auto basic_inverse = local_matrices[patch_index].share_basic_inverse();
      ASSERT_TRUE(basic_inverse) << "inverse was not computed";

      *Base::pcout_owned << "patch: " << patch_index << std::endl;

      for(auto lane = 0U; lane < patch_dof_worker_v.n_lanes_filled(patch_index); ++lane)
      {
        const auto & inverse_svd = basic_inverse->inverses->at(lane);
        const auto   n           = inverse_svd.n();

        *Base::pcout_owned << "singular values: " << inverse_svd.singular_value(0) << " "
                           << inverse_svd.singular_value(1) << " " << inverse_svd.singular_value(2)
                           << " ... " << inverse_svd.singular_value(n - 3) << " "
                           << inverse_svd.singular_value(n - 2) << " "
                           << inverse_svd.singular_value(n - 1) << std::endl;
      }
    }
  }

  std::shared_ptr<StokesProblem> stokes_problem;
};



TYPED_TEST_SUITE_P(TestSimplifiedRaviartThomas);



TYPED_TEST_P(TestSimplifiedRaviartThomas, localsvd)
{
  using Fixture                               = TestSimplifiedRaviartThomas<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;

  /// single vertex-patch
  Fixture::options.prms.mesh.n_refinements = 0;
  double h                                 = 2.;
  for(auto l = 0U; l < 10; ++l)
  {
    *Fixture::pcout_owned << std::endl;
    h *= 0.5;
    Fixture::options.prms.mesh.domain_1d = {0., h};
    Fixture::check_localsvd();
    if(l == 0U)
      Fixture::stokes_problem->print_informations();
  }

  /// vertex-patches with all kinds of boundaries
  Fixture::options.prms.mesh.n_refinements = 1;
  h                                        = 4.;
  for(auto l = 0U; l < 10; ++l)
  {
    *Fixture::pcout_owned << std::endl;
    h *= 0.5;
    Fixture::options.prms.mesh.domain_1d = {0., h};
    Fixture::check_localsvd();
    if(l == 0U)
      Fixture::stokes_problem->print_informations();
  }
}



REGISTER_TYPED_TEST_SUITE_P(TestSimplifiedRaviartThomas, localsvd);



using TestParamsLinear    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
using TestParamsSeptic    = testing::Types<Util::NonTypeParams<2, 7>>;



INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestSimplifiedRaviartThomas, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestSimplifiedRaviartThomas, TestParamsQuadratic);
INSTANTIATE_TYPED_TEST_SUITE_P(Septic2D, TestSimplifiedRaviartThomas, TestParamsSeptic);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_stokes_raviartthomas_all.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
