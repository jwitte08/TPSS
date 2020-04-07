
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


  struct Params
  {
    unsigned int       n_refinements = 0;
    TPSS::PatchVariant patch_variant = TPSS::PatchVariant::vertex;
  };


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


  // FullMatrix<double>
  // assemble_system_matrix() const
  // {
  //   AssertThrow(biharmonic_problem, ExcMessage("Not initialized."));
  //   const auto & system_matrix_mf = biharmonic_problem->system_matrix;
  //   const auto   table            = Tensors::matrix_to_table(system_matrix_mf);
  //   return table_to_fullmatrix(table);
  // }


  // template<typename PatchTransfer, typename VectorType>
  // std::vector<unsigned int>
  // extract_dof_indices_per_patch(const unsigned int patch,
  //                               PatchTransfer &    patch_transfer,
  //                               VectorType &       vector,
  //                               const unsigned int lane = 0)
  // {
  //   patch_transfer.reinit(patch);
  //   for(auto i = 0U; i < vector.size(); ++i)
  //     vector(i) = static_cast<double>(i) + 0.1;
  //   const auto                indices      = patch_transfer.gather(vector);
  //   const auto                indices_view = make_array_view(indices.begin(), indices.end());
  //   const auto                indices_lane = array_view_to_vector(indices_view, lane);
  //   std::vector<unsigned int> indices_per_patch;
  //   std::copy(indices_lane.begin(), indices_lane.end(), std::back_inserter(indices_per_patch));
  //   return indices_per_patch;
  // }


  void
  check_local_matrices()
  {
    initialize();

    biharmonic_problem->system_matrix.print_formatted(std::cout);
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

  Params                                   params;
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
