
/*
 * Test Laplace integrators.
 *
 *  Created on: Feb 06, 2020
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "poisson_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;



enum class TestVariant
{
  matrix,
  inverse
};

template<typename T>
class TestLaplaceIntegrator : public testing::Test
{
protected:
  static constexpr int  dim        = T::template value<0>();
  static constexpr int  fe_degree  = T::template value<1>();
  static constexpr auto dof_layout = TPSS::DoFLayout::Q;
  using PoissonProblem             = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;
  using SystemMatrix               = typename PoissonProblem::SYSTEM_MATRIX;
  using LevelMatrix                = typename PoissonProblem::LEVEL_MATRIX;
  using vector_type                = typename PoissonProblem::VECTOR;
  static constexpr unsigned int macro_size = VectorizedArray<double>::size();

  struct Params
  {
    unsigned int       n_refinements = 0;
    TPSS::PatchVariant patch_variant = TPSS::PatchVariant::vertex;
  };

  void
  SetUp() override
  {
    ofs.open("apps_laplace_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout_owned              = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_repetitions    = 2U;

    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
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
      new_problem->create_triangulation();
      new_problem->distribute_dofs();
      new_problem->prepare_linear_system(/*compute_rhs?*/ true);
      new_problem->prepare_multigrid(); // do not clear MGConstrainedDoFs
    };

    poisson_problem.reset();
    rt_parameters.mesh.n_refinements                           = params.n_refinements;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = params.patch_variant;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;

    const auto new_problem = std::make_shared<PoissonProblem>(rt_parameters);
    initialize_problem(new_problem);
    poisson_problem = new_problem;
  }

  FullMatrix<double>
  assemble_system_matrix() const
  {
    AssertThrow(poisson_problem, ExcMessage("Not initialized."));
    const auto & system_matrix_mf = poisson_problem->system_matrix;
    const auto   table            = Tensors::matrix_to_table(system_matrix_mf);
    return table_to_fullmatrix(table);
  }

  FullMatrix<double>
  assemble_level_matrix(const unsigned int level) const
  {
    AssertThrow(poisson_problem, ExcMessage("Not initialized."));
    const auto & level_matrix_mf = poisson_problem->mg_matrices[level];
    const auto   table           = Tensors::matrix_to_table(level_matrix_mf);
    return table_to_fullmatrix(table);
  }

  template<typename PatchTransfer, typename VectorType>
  std::vector<unsigned int>
  extract_dof_indices_per_patch(const unsigned int patch,
                                PatchTransfer &    patch_transfer,
                                VectorType &       vector,
                                const unsigned int lane = 0)
  {
    patch_transfer.reinit(patch);
    for(auto i = 0U; i < vector.size(); ++i)
      vector(i) = static_cast<double>(i) + 0.1;
    const auto                indices      = patch_transfer.gather(vector);
    const auto                indices_view = make_array_view(indices.begin(), indices.end());
    const auto                indices_lane = array_view_to_vector(indices_view, lane);
    std::vector<unsigned int> indices_per_patch;
    std::copy(indices_lane.begin(), indices_lane.end(), std::back_inserter(indices_per_patch));
    return indices_per_patch;
  }

  void
  test(TestVariant test_variant = TestVariant::matrix)
  {
    initialize();

    /// compare system matrix and level matrix on finest level
    const auto & system_matrix = assemble_system_matrix();
    const auto   global_level  = poisson_problem->level;
    const auto & level_matrix  = assemble_level_matrix(global_level);
    *pcout_owned << "compare system and level matrix:" << std::endl;
    compare_matrix(system_matrix, level_matrix);

    const auto mf_storage_level = poisson_problem->mg_matrices[global_level].get_matrix_free();
    const auto schwarz_preconditioner =
      poisson_problem->mg_schwarz_smoother_pre->get_preconditioner();
    const auto patch_storage_level = schwarz_preconditioner->get_subdomain_handler();
    TPSS::PatchTransfer<dim, double, fe_degree> patch_transfer(*patch_storage_level);
    const auto & patch_worker   = patch_transfer.get_patch_dof_worker();
    const auto & partition_data = patch_worker.get_partition_data();

    vector_type tmp_vector;
    mf_storage_level->initialize_dof_vector(tmp_vector);
    const auto & local_solvers = *(schwarz_preconditioner->get_local_solvers());
    for(auto patch = 0U; patch < partition_data.n_subdomains(); ++patch)
      for(auto lane = 0U; lane < macro_size; ++lane)
      {
        /// extract patch matrix from level matrix
        const auto dof_indices =
          extract_dof_indices_per_patch(patch, patch_transfer, tmp_vector, lane);
        FullMatrix<double> patch_matrix_reference(dof_indices.size());
        patch_matrix_reference.extract_submatrix_from(level_matrix, dof_indices, dof_indices);

        const auto & local_matrix = local_solvers[patch];

        /// transform local solver to FullMatrix type and fill constrained
        /// diagonal entries with ones (in analogy to the matrix-free level
        /// matrix)
        if(test_variant == TestVariant::matrix)
        {
          auto patch_matrix_full =
            table_to_fullmatrix(Tensors::matrix_to_table(local_matrix), lane);
          *pcout_owned << "compare local matrix:" << std::endl;
          compare_matrix(patch_matrix_full, patch_matrix_reference);
        }

        /// same as before for the inverse operation
        if(test_variant == TestVariant::inverse)
        {
          *pcout_owned << "compare inverse of local matrix:" << std::endl;
          auto patch_matrix_inverse =
            table_to_fullmatrix(Tensors::inverse_matrix_to_table(local_matrix), lane);
          compare_inverse_matrix(patch_matrix_inverse, patch_matrix_reference);
        }
      }
  }

  void
  test_matrix_free_patch(TestVariant test_variant = TestVariant::matrix)
  {
    initialize();

    const auto global_level = poisson_problem->level;
    const auto schwarz_preconditioner =
      poisson_problem->mg_schwarz_smoother_pre->get_preconditioner();
    const auto patch_storage_level = schwarz_preconditioner->get_subdomain_handler();
    auto &     mf_level_matrix     = poisson_problem->mg_matrices[global_level];
    mf_level_matrix.reinit_patch_evaluator(*patch_storage_level);
    vector_type tmp_vector;
    mf_level_matrix.get_matrix_free()->initialize_dof_vector(tmp_vector);
    const auto & level_matrix = assemble_level_matrix(global_level);

    TPSS::PatchTransfer<dim, double, fe_degree> patch_transfer(*patch_storage_level);
    const auto & patch_worker   = patch_transfer.get_patch_dof_worker();
    const auto & partition_data = patch_worker.get_partition_data();
    for(auto patch = 0U; patch < partition_data.n_subdomains(); ++patch)
    {
      PatchMatrix<LevelMatrix> patch_matrix;
      patch_matrix.reinit(&mf_level_matrix, patch);
      for(auto lane = 0U; lane < macro_size; ++lane)
      {
        /// extract patch matrix from level matrix
        const auto dof_indices =
          extract_dof_indices_per_patch(patch, patch_transfer, tmp_vector, lane);
        FullMatrix<double> patch_matrix_reference(dof_indices.size());
        patch_matrix_reference.extract_submatrix_from(level_matrix, dof_indices, dof_indices);

        if(test_variant == TestVariant::matrix)
        {
          auto patch_matrix_full = patch_matrix.as_fullmatrix(lane);
          *pcout_owned << "compare local matrix (matrix-free patch operator):" << std::endl;
          compare_matrix(patch_matrix_full, patch_matrix_reference);
        }
      }
    }
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

  Params                                params;
  RT::Parameter                         rt_parameters;
  std::shared_ptr<const PoissonProblem> poisson_problem;
};


TYPED_TEST_SUITE_P(TestLaplaceIntegrator);

TYPED_TEST_P(TestLaplaceIntegrator, FDAssemblyVertexPatch)
{
  using Fixture = TestLaplaceIntegrator<TypeParam>;

  Fixture::params.n_refinements = 0U;
  Fixture::test();

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 3U;
  Fixture::params.n_refinements                = 0U;
  Fixture::test();
}

TYPED_TEST_P(TestLaplaceIntegrator, FDInverseVertexPatch)
{
  using Fixture = TestLaplaceIntegrator<TypeParam>;

  Fixture::params.n_refinements = 0U;
  Fixture::test(TestVariant::inverse);
}

TYPED_TEST_P(TestLaplaceIntegrator, MFPatchAssemblyVertexPatch)
{
  using Fixture = TestLaplaceIntegrator<TypeParam>;

  Fixture::params.n_refinements = 0U;
  Fixture::test_matrix_free_patch();

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 3U;
  Fixture::params.n_refinements                = 0U;
  Fixture::test_matrix_free_patch();
}

REGISTER_TYPED_TEST_SUITE_P(TestLaplaceIntegrator,
                            FDAssemblyVertexPatch,
                            FDInverseVertexPatch,
                            MFPatchAssemblyVertexPatch);

using TestParamsLinear = testing::Types<Util::NonTypeParams<2, 1>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLaplaceIntegrator, TestParamsLinear);
using TestParamsHigherOrder = testing::Types<Util::NonTypeParams<2, 2>, Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestLaplaceIntegrator, TestParamsHigherOrder);
using TestParamsLinear3D = testing::Types<Util::NonTypeParams<3, 1>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestLaplaceIntegrator, TestParamsLinear3D);
using TestParamsHigherOrder3D =
  testing::Types<Util::NonTypeParams<3, 2>, Util::NonTypeParams<3, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D, TestLaplaceIntegrator, TestParamsHigherOrder3D);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_laplace_integrator.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
