
/*
 * Test Laplace integrators.
 *
 *  Created on: Apr 07, 2020
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



template<typename T>
class TestBiharmonicIntegrator : public testing::Test
{
protected:
  static constexpr int dim                 = T::template value<0>();
  static constexpr int fe_degree           = T::template value<1>();
  using BiharmonicProblem                  = ModelProblem<dim>;
  static constexpr unsigned int macro_size = VectorizedArray<double>::size();


  void
  SetUp() override
  {
    ofs.open("apps_biharmonic_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout_owned              = std::make_shared<ConditionalOStream>(ofs, is_first_proc);
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  initialize()
  {
    const auto initialize_problem = [&](auto & new_problem) {
      new_problem->pcout = pcout_owned;
      new_problem->make_grid();
      new_problem->setup_system();
      new_problem->assemble_system();
    };

    biharmonic_problem.reset();

    const auto new_problem = std::make_shared<BiharmonicProblem>(fe_degree);
    initialize_problem(new_problem);
    biharmonic_problem = new_problem;
  }


  void
  check_local_matrices()
  {
    initialize();

    /// matrix-free dummy
    const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.mg_level = biharmonic_problem->max_level();
      QGauss<1> quadrature(fe_degree + 1);
      mf_storage->reinit(biharmonic_problem->mapping,
                         biharmonic_problem->dof_handler,
                         biharmonic_problem->constraints,
                         quadrature,
                         additional_data);
    }

    /// distribute subdomains
    const auto subdomain_handler = std::make_shared<SubdomainHandler<dim, double>>();
    {
      typename SubdomainHandler<dim, double>::AdditionalData additional_data;
      additional_data.level            = biharmonic_problem->max_level();
      additional_data.smoother_variant = TPSS::SmootherVariant::additive;
      additional_data.patch_variant    = TPSS::PatchVariant::vertex;
      subdomain_handler->reinit(mf_storage, additional_data);
    }

    /// compare local matrices
    TPSS::PatchTransfer<dim, double, fe_degree> patch_transfer(*subdomain_handler);
    const auto & patch_worker = patch_transfer.get_patch_dof_worker();
    for(auto patch = 0U; patch < patch_worker.get_partition_data().n_subdomains(); ++patch)
    {
      patch_transfer.reinit(patch);
      for(auto lane = 0U; lane < patch_worker.n_lanes_filled(patch); ++lane)
      {
        std::vector<types::global_dof_index> dof_indices_on_patch;
        {
          const auto view = patch_transfer.get_dof_indices(lane);
          std::copy(view.cbegin(), view.cend(), std::back_inserter(dof_indices_on_patch));
        }
        FullMatrix<double> local_matrix(dof_indices_on_patch.size());
        local_matrix.extract_submatrix_from(biharmonic_problem->system_matrix,
                                            dof_indices_on_patch,
                                            dof_indices_on_patch);
        local_matrix.print_formatted(std::cout);
      }
    }

    // biharmonic_problem->system_matrix.print_formatted(std::cout);
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

  RT::Parameter                            rt_parameters;
  std::shared_ptr<const BiharmonicProblem> biharmonic_problem;
};



TYPED_TEST_SUITE_P(TestBiharmonicIntegrator);

TYPED_TEST_P(TestBiharmonicIntegrator, CheckVertexPatchMatrix)
{
  using Fixture = TestBiharmonicIntegrator<TypeParam>;
  Fixture::check_local_matrices();
}

REGISTER_TYPED_TEST_SUITE_P(TestBiharmonicIntegrator, CheckVertexPatchMatrix);

using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestBiharmonicIntegrator, TestParamsQuadratic);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_biharmonic_integrator.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
