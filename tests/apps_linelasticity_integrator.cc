
/*
 * Test linear elasticity integrators.
 *
 * 1) Comparison of MatrixFree and MeshWorker.
 *
 *  Created on: Jun 27, 2019
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "linelasticity_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace LinElasticity;

template<typename T>
class TestLinElasticityIntegratorMF : public testing::Test
{
protected:
  struct Params
  {
    unsigned int n_refinements = 0;
    EquationData equation_data;
    bool         print_details = true;
  };

  static constexpr int dim    = T::template value<0>();
  static constexpr int degree = T::template value<1>();
  using LinElasticityProblem  = typename LinElasticity::ModelProblem<dim, degree, double>;

  void
  SetUp() override
  {
    ofs.open("apps_linelasticity_integrator.log", std::ios_base::app);
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, params.print_details);
  }

  void
  TearDown() override
  {
    ofs.close();
  }

  void
  initialize()
  {
    linelasticity_problem.reset();

    rt_parameters.mesh.n_refinements    = params.n_refinements;
    rt_parameters.mesh.n_repetitions    = 1;
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;

    const auto new_problem =
      std::make_shared<LinElasticityProblem>(*pcout_owned, rt_parameters, params.equation_data);
    new_problem->create_triangulation();
    new_problem->distribute_dofs();
    new_problem->prepare_system(false, /*compute_rhs?*/ false);
    new_problem->assemble_matrix();

    linelasticity_problem = new_problem;
  }

  void
  test()
  {
    initialize();
    ASSERT_NE(linelasticity_problem, nullptr);

    const auto & sparse_matrix = linelasticity_problem->sparse_matrix;
    const auto & mf_operator   = linelasticity_problem->system_matrix;
    auto &       pcout         = linelasticity_problem->pcout;
    if(pcout.is_active())
    {
      pcout << "SPARSE MATRIX\n\n";
      sparse_matrix.print_formatted(pcout.get_stream());
      pcout << std::endl;
    }

    const auto     n_cells          = linelasticity_problem->triangulation.n_active_cells();
    const unsigned n_blocks         = dim;
    const unsigned n_components     = dim;
    const auto     n_dofs           = linelasticity_problem->system_u.size();
    const auto     n_dofs_per_block = linelasticity_problem->system_u.block(0).size();
    AssertDimension(1, linelasticity_problem->fe->n_base_elements());
    const auto n_dofs_per_cell_scalar = linelasticity_problem->fe->n_dofs_per_cell();
    const auto n_dofs_per_cell        = n_dofs_per_cell_scalar * n_components;
    // *** map block-wise indices to component-wise indices
    const auto block_to_comp_indices = [&](const unsigned int i) {
      AssertIndexRange(i, n_dofs);
      const unsigned block                = i / n_dofs_per_block;
      const unsigned i_within_block       = i % n_dofs_per_block;
      const unsigned cell                 = i_within_block / n_dofs_per_cell_scalar;
      const unsigned i_within_cell_scalar = i_within_block % n_dofs_per_cell_scalar;

      const unsigned component = block;
      AssertIndexRange(block, n_blocks);
      AssertIndexRange(component, n_components);
      AssertIndexRange(cell, n_cells);
      const unsigned j_within_cell = component * n_dofs_per_cell_scalar + i_within_cell_scalar;
      const unsigned j             = cell * n_dofs_per_cell + j_within_cell;
      AssertIndexRange(j, n_dofs);
      return j;
    };

    // *** fill the matrix associated to the matrix-free operator
    FullMatrix<double> mf_matrix_unsorted{sparse_matrix.m(), sparse_matrix.n()};
    for(unsigned j = 0; j < sparse_matrix.m(); ++j)
    {
      auto temp{linelasticity_problem->system_u};
      temp *= 0.;
      temp[j] = 1.;
      const auto basis_j{temp};
      mf_operator.vmult(temp, basis_j);
      Vector<double> dst(temp.begin(), temp.end());
      for(unsigned i = 0; i < sparse_matrix.n(); ++i)
        mf_matrix_unsorted(i, j) = dst[i];
    }

    // *** sort the degrees of freedom to match the order of the sparse matrix
    FullMatrix<double> mf_matrix{sparse_matrix.m(), sparse_matrix.n()};
    for(unsigned j = 0; j < sparse_matrix.m(); ++j)
      for(unsigned i = 0; i < sparse_matrix.m(); ++i)
        mf_matrix(block_to_comp_indices(i), block_to_comp_indices(j)) = mf_matrix_unsorted(i, j);

    if(pcout.is_active())
    {
      pcout << "MATRIX-FREE MATRIX (sorted) \n\n";
      mf_matrix.print_formatted(pcout.get_stream());
      pcout << std::endl;
    }

    { // TEST: comparison of vmults
      Vector<double> src(linelasticity_problem->system_u.size());
      const auto     src_view = make_array_view(src);
      fill_with_random_values(src_view);
      const Vector<double> src_copy(src_view.cbegin(), src_view.cend());

      Vector<double> dst(src_copy.size());
      mf_matrix.vmult(dst, src_copy);

      Vector<double> dst_sparse(src_copy.size());
      sparse_matrix.vmult(dst_sparse, src_copy);

      Vector<double> diff(dst.begin(), dst.end());
      diff -= dst_sparse;
      EXPECT_PRED_FORMAT2(::testing::DoubleLE,
                          diff.l2_norm(),
                          std::numeric_limits<double>::epsilon() *
                            std::max(1.e+4, dst_sparse.l2_norm()));
    } // end TEST
  }

  Params                                      params;
  RT::Parameter                               rt_parameters;
  std::shared_ptr<const LinElasticityProblem> linelasticity_problem;
  std::ofstream                               ofs;
  std::shared_ptr<ConditionalOStream>         pcout_owned;
};


TYPED_TEST_SUITE_P(TestLinElasticityIntegratorMF);
TYPED_TEST_P(TestLinElasticityIntegratorMF, VaryDimAndDegree)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1)
    << "Testing against serial sparse matrices, consequently only ONE mpi rank is allowed.";

  using Fixture = TestLinElasticityIntegratorMF<TypeParam>;

  Fixture::params.equation_data.lambda = 1.234;
  Fixture::params.equation_data.mu     = 9.876;
  Fixture::test();
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityIntegratorMF, VaryDimAndDegree);

using TestParams2D = testing::Types<Util::NonTypeParams<2, 1>, Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensions, TestLinElasticityIntegratorMF, TestParams2D);

using TestParams3D = testing::Types<Util::NonTypeParams<3, 1>, Util::NonTypeParams<3, 2>>;
INSTANTIATE_TYPED_TEST_SUITE_P(ThreeDimensions, TestLinElasticityIntegratorMF, TestParams3D);



template<typename T>
class TestLinElasticityIntegratorFD : public testing::Test
{
protected:
  static constexpr int dim    = T::template value<0>();
  static constexpr int degree = T::template value<1>();
  using LinElasticityProblem  = typename LinElasticity::ModelProblem<dim, degree, double>;

  struct Params
  {
    unsigned int n_refinements = 0;
    EquationData equation_data;
    bool         print_details = true;
  };

  struct Extractor
  {
    using BlockVector                        = LinearAlgebra::distributed::BlockVector<double>;
    using SystemMatrix                       = typename LinElasticityProblem::SYSTEM_MATRIX;
    static constexpr unsigned int macro_size = VectorizedArray<double>::n_array_elements;

    void
    initialize_dof_vector(BlockVector & vec) const
    {
      AssertThrow(data, ExcMessage("Matrix-free storage is uninitialized."));
      vec.reinit(dim);
      for(unsigned int comp = 0; comp < dim; ++comp)
        data->initialize_dof_vector(vec.block(comp), comp);
      vec.collect_sizes();
    }

    void
    extract_cell_indices(const MatrixFree<dim, double> & data,
                         BlockVector &                   dst,
                         const BlockVector & /*src*/,
                         const std::pair<unsigned int, unsigned int> & cell_range) const
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
      {
        const auto u =
          std::make_shared<FEEvaluation<dim, degree, degree + 1, 1, double>>(data, comp);
        for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          u->reinit(cell);
          auto cell_index = make_vectorized_array(-1.);
          for(unsigned int lane = 0; lane < data.n_components_filled(cell); ++lane)
          {
            const auto & cell_it = data.get_cell_iterator(cell, lane, comp);
            cell_index[lane]     = cell_it->index();
          }
          for(unsigned int i = 0; i < u->dofs_per_cell; ++i)
            u->submit_dof_value(cell_index, i);
          u->set_dof_values(dst.block(comp));
        }
      }
    }

    void
    operator()()
    {
      initialize_dof_vector(cell_indices);
      data->cell_loop(&Extractor::extract_cell_indices, this, cell_indices, cell_indices);
    }

    void
    assemble_full_matrix()
    {
      Assert(system_matrix, ExcMessage("System matrix is uninitialized."));
      BlockVector e_j;
      initialize_dof_vector(e_j);
      BlockVector col_j;
      initialize_dof_vector(col_j);
      const unsigned int n_dofs = system_matrix->m();
      const auto         mat    = std::make_shared<FullMatrix<double>>(n_dofs, n_dofs);

      for(unsigned j = 0; j < mat->n(); ++j)
      {
        e_j *= 0.;
        e_j[j] = 1.;
        col_j *= 0.;
        system_matrix->vmult(col_j, e_j);
        // Vector<double> dst(col_j.begin(), col_j.end());
        for(unsigned i = 0; i < mat->m(); ++i)
          (*mat)(i, j) = col_j[i];
      }

      full_matrix = mat;
    }

    const SystemMatrix *                           system_matrix;
    std::shared_ptr<const MatrixFree<dim, double>> data;
    std::shared_ptr<const FullMatrix<double>>      full_matrix;
    BlockVector                                    cell_indices;
  };

  void
  SetUp() override
  {
    ofs.open("apps_linelasticity_integrator.log", std::ios_base::app);
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, params.print_details);
  }

  void
  TearDown() override
  {
    ofs.close();
  }

  void
  initialize()
  {
    linelasticity_problem.reset();

    rt_parameters.mesh.n_refinements = params.n_refinements;
    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;

    const auto new_problem =
      std::make_shared<LinElasticityProblem>(*pcout_owned, rt_parameters, params.equation_data);
    new_problem->create_triangulation();
    new_problem->distribute_dofs();
    new_problem->prepare_system(false, /*compute_rhs?*/ false);
    // new_problem->assemble_matrix(); ???
    linelasticity_problem = new_problem;

    ex.system_matrix = &linelasticity_problem->system_matrix;
    ex.data          = ex.system_matrix->get_matrix_free();
  }

  void
  test()
  {
    params.n_refinements = 0;
    initialize();

    ex.assemble_full_matrix();
    ex.full_matrix->print_formatted(pcout_owned->get_stream());
    ex();
    for(auto i = 0; i < ex.cell_indices.size(); ++i)
      std::cout << ex.cell_indices[i] << std::endl;
  }

  void
  test_block_vector()
  {
    // initialize();
    // auto blockvec = linelasticity_problem->system_u;
    // for (auto b = 0; b < blockvec.n_blocks(); ++b)
    //   {
    // 	auto view = make_array_view(blockvec.block(b).begin(), blockvec.block(b).end());
    // 	fill_with_random_values(view);
    //   }
    // for (auto i = 0; i < blockvec.size(); ++i)
    //   *pcout_owned << blockvec[i] << std::endl;
    // for (auto b = 0; b < blockvec.n_blocks(); ++b)
    //   for (auto i = 0; i < blockvec.block(b).size(); ++i)
    // 	*pcout_owned << blockvec.block(b)[i] << std::endl;
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  Params                                      params;
  RT::Parameter                               rt_parameters;
  std::shared_ptr<const LinElasticityProblem> linelasticity_problem;
  Extractor                                   ex;
};

TYPED_TEST_SUITE_P(TestLinElasticityIntegratorFD);
TYPED_TEST_P(TestLinElasticityIntegratorFD, VaryDimAndDegree)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1)
    << "Testing against serial sparse matrices, consequently only ONE mpi rank is allowed.";

  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::params.equation_data.lambda = 1.234;
  Fixture::params.equation_data.mu     = 9.876;

  // Fixture::test_block_vector();
  Fixture::test();
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityIntegratorFD, VaryDimAndDegree);

using TestParams2DFD = testing::Types<Util::NonTypeParams<2, 1>>; //, Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensions, TestLinElasticityIntegratorFD, TestParams2DFD);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_linelasticity_integrator.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
