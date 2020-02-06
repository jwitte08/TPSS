
/*
 * Test Laplace integrators.
 *
 *  Created on: Feb 06, 2020
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "poisson_standard.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;



template<typename T>
class TestLaplaceIntegrator : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();
  using PoissonProblem           = typename Poisson::Std::ModelProblem<dim, fe_degree>;
  using SystemMatrix             = typename PoissonProblem::SYSTEM_MATRIX;
  // using LevelMatrix              = typename PoissonProblem::LEVEL_MATRIX;
  // using PatchTransfer                      = typename LevelMatrix::transfer_type;
  // using VectorizedMatrixType               = Table<2, VectorizedArray<double>>;
  // static constexpr unsigned int fe_order   = fe_degree + 1;
  static constexpr unsigned int macro_size = VectorizedArray<double>::n_array_elements;

  struct Params
  {
    unsigned int n_refinements = 0;
    // EquationData       equation_data;
    TPSS::PatchVariant patch_variant = TPSS::PatchVariant::cell;
  };

  void
  SetUp() override
  {
    ofs.open("apps_laplace_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout_owned              = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;

    // rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = params.patch_variant;
    // rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    // TPSS::SmootherVariant::additive; rt_parameters.multigrid.post_smoother.schwarz =
    // rt_parameters.multigrid.pre_smoother.schwarz;
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
      new_problem->prepare_multigrid();
    };

    poisson_problem.reset();
    rt_parameters.mesh.n_refinements = params.n_refinements;

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

  void
  test()
  {
    initialize();
    const auto & system_matrix = assemble_system_matrix();
    const auto   global_level  = poisson_problem->level;
    const auto & level_matrix  = assemble_level_matrix(global_level);
    compare_matrix(system_matrix, level_matrix);
  }

  void
  compare_matrix(const FullMatrix<double> & patch_matrix_full,
                 const FullMatrix<double> & other) const
  {
    Util::compare_matrix(patch_matrix_full, other, *pcout_owned);
  }

  // void
  // compare_matrix(const FullMatrix<double> & other) const
  // {
  //   const auto patch_matrix_full = table_to_fullmatrix(patch_matrix.as_table());
  //   compare_matrix(patch_matrix_full, other);
  // }

  // void
  // compare_inverse_matrix(const FullMatrix<double> & inverse_patch_matrix,
  //                        const FullMatrix<double> & other) const
  // {
  //   Util::compare_inverse_matrix(inverse_patch_matrix, other, *pcout_owned);
  // }

  // void
  // compare_inverse_matrix(const FullMatrix<double> & other) const
  // {
  //   const auto inverse_patch_matrix = table_to_fullmatrix(patch_matrix.as_inverse_table());
  //   compare_inverse_matrix(inverse_patch_matrix, other);
  // }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  Params                                params;
  RT::Parameter                         rt_parameters;
  std::shared_ptr<const PoissonProblem> poisson_problem;
};


TYPED_TEST_SUITE_P(TestLaplaceIntegrator);

TYPED_TEST_P(TestLaplaceIntegrator, AssembleSystemMatrix)
{
  using Fixture = TestLaplaceIntegrator<TypeParam>;

  Fixture::params.n_refinements = 0U;
  Fixture::test();

  // Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  // Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  // Fixture::params.n_refinements                = 1U;
  // Fixture::params.equation_data.lambda         = 1.234;
  // Fixture::params.equation_data.mu             = 9.876;
  // Fixture::manual_assembly();
}

REGISTER_TYPED_TEST_SUITE_P(TestLaplaceIntegrator, AssembleSystemMatrix);

using TestParamsLinear = testing::Types<Util::NonTypeParams<2, 1>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLaplaceIntegrator, TestParamsLinear);

using TestParamsHigherOrder = testing::Types<Util::NonTypeParams<2, 2>, Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestLaplaceIntegrator, TestParamsHigherOrder);



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
