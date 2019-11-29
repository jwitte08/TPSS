
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

    const auto & sparse_matrix  = linelasticity_problem->sparse_matrix;
    const auto & combi_operator = linelasticity_problem->system_matrix;
    auto &       pcout          = linelasticity_problem->pcout;
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
      combi_operator.vmult(temp, basis_j);
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
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();
  using LinElasticityProblem     = typename LinElasticity::ModelProblem<dim, fe_degree, double>;
  using BlockVector              = LinearAlgebra::distributed::BlockVector<double>;
  using LevelMatrix              = typename LinElasticityProblem::LEVEL_MATRIX;
  static constexpr unsigned int fe_order   = fe_degree + 1;
  static constexpr unsigned int macro_size = VectorizedArray<double>::n_array_elements;

  struct Params
  {
    unsigned int n_refinements = 0;
    EquationData equation_data;
    bool         print_details = true;
  };

  struct Extractor
  {
    void
    extract_cell_indices_impl(const MatrixFree<dim, double> & data,
                              BlockVector &                   dst,
                              const BlockVector & /*src*/,
                              const std::pair<unsigned int, unsigned int> & cell_range) const
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
      {
        const auto u =
          std::make_shared<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>>(data, comp);
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

    std::vector<unsigned int>
    map_dof_to_cell_index() const
    {
      BlockVector cell_indices;
      combi_operator->initialize_dof_vector(cell_indices);
      data->cell_loop(&Extractor::extract_cell_indices_impl, this, cell_indices, cell_indices);
      std::vector<unsigned> dof_to_cell;
      std::transform(cell_indices.begin(),
                     cell_indices.end(),
                     std::back_inserter(dof_to_cell),
                     [](const double c) { return static_cast<unsigned int>(c + 0.5); });
      return dof_to_cell;
    }

    std::vector<std::pair<unsigned, unsigned>>
    map_cell_to_patch_index() const
    {
      const auto &                               patch_info = patch_storage->get_patch_info();
      TPSS::PatchWorker<dim, double>             worker(patch_info);
      const auto                                 n_cells = data->n_physical_cells();
      std::vector<std::pair<unsigned, unsigned>> cell_to_patch(n_cells);
      const auto &                               partition_data = worker.get_partition_data();
      Assert(n_cells == worker.n_physical_subdomains(), ExcMessage("TODO Only cell patches."));
      for(unsigned p = 0; p < partition_data.n_subdomains(); ++p)
      {
        const auto & collection = worker.get_cell_collection(p);
        for(unsigned lane = 0; lane < worker.n_lanes_filled(p); ++lane)
        {
          const auto & cell = collection.front()[lane];
          std::cout << cell->index() << std::endl;
          cell_to_patch[cell->index()] = {p, lane};
        }
      }
      return cell_to_patch;
    }

    const LevelMatrix *                                  combi_operator;
    std::shared_ptr<const MatrixFree<dim, double>>       data;
    std::shared_ptr<const SubdomainHandler<dim, double>> patch_storage;
  };

  struct Assembler
  {
    using VectorizedMatrixType = Table<2, VectorizedArray<double>>;
    using FDMatrixIntegrator   = typename FD::MatrixIntegrator<dim, fe_degree, double>;
    using EvaluatorType        = typename FDMatrixIntegrator::EvaluatorType;
    using CellMass             = typename FDMatrixIntegrator::CellMass;
    using CellStrain           = typename FDMatrixIntegrator::template CellStrain<EvaluatorType>;
    using CellGradMixed        = typename FDMatrixIntegrator::template CellGradMixed<EvaluatorType>;
    using CellGradDiv          = typename FDMatrixIntegrator::template CellGradDiv<EvaluatorType>;
    using NitscheStrain        = typename FDMatrixIntegrator::template NitscheStrain<EvaluatorType>;
    using NitscheGradDiv = typename FDMatrixIntegrator::template NitscheGradDiv<EvaluatorType>;


    std::shared_ptr<const FullMatrix<double>>
    assemble_level_matrix() const
    {
      Assert(combi_operator, ExcMessage("System matrix is uninitialized."));
      BlockVector e_j;
      combi_operator->initialize_dof_vector(e_j);
      BlockVector col_j;
      combi_operator->initialize_dof_vector(col_j);
      const unsigned int n_dofs = combi_operator->m();
      const auto         mat    = std::make_shared<FullMatrix<double>>(n_dofs, n_dofs);

      for(unsigned j = 0; j < mat->n(); ++j)
      {
        e_j *= 0.;
        e_j[j] = 1.;
        col_j *= 0.;
        combi_operator->vmult(col_j, e_j);
        // Vector<double> dst(col_j.begin(), col_j.end());
        for(unsigned i = 0; i < mat->m(); ++i)
          (*mat)(i, j) = col_j[i];
      }

      return mat;
    }


    std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>
    assemble_mass_matrices() const
    {
      const auto & assembler_mass_matrices =
        [](const SubdomainHandler<dim, double> &                                 data,
           std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
           const bool & /*dummy*/,
           const std::pair<unsigned int, unsigned int> & subdomain_range) {
          std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
          for(unsigned int comp = 0; comp < dim; ++comp)
            fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

          /// initialize CellMass operation for each component
          std::vector<CellMass> cell_mass_operations;
          for(unsigned int comp = 0; comp < dim; ++comp)
          {
            auto &               fd_eval = *(fd_evals[comp]);
            VectorizedMatrixType cell_mass_unit(fe_order, fe_order);
            fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
            cell_mass_operations.emplace_back(cell_mass_unit);
          }

          /// scales 1D unit mass with appropriate h for each component
          for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
            matrices[id] =
              FDMatrixIntegrator::assemble_mass_tensors(fd_evals, cell_mass_operations, id);
        };

      const auto & partition_data = patch_storage->get_partition_data();
      std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> mass_matrices;
      mass_matrices.resize(partition_data.n_subdomains());
      for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
        patch_storage
          ->template loop<bool,
                          std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
            assembler_mass_matrices, mass_matrices, /*dummy*/ false, color);

      return mass_matrices;
    }


    std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>
    assemble_elasticity_matrices() const
    {
      /// LAMBDA assemble strain : strain
      const auto & assembler_strain_matrices =
        [](const SubdomainHandler<dim, double> &                                 data,
           std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
           const EquationData &                                                  equation_data,
           const std::pair<unsigned int, unsigned int> &                         subdomain_range) {
          std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
          for(unsigned int comp = 0; comp < dim; ++comp)
            fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

          std::vector<CellStrain> cell_strain_operations;
          for(unsigned int comp = 0; comp < dim; ++comp)
            cell_strain_operations.emplace_back(equation_data, comp);
          std::vector<NitscheStrain> nitsche_strain_operations;
          for(unsigned int comp = 0; comp < dim; ++comp)
            nitsche_strain_operations.emplace_back(equation_data, comp);

          for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
            matrices[id] = FDMatrixIntegrator::assemble_strain_tensors(fd_evals,
                                                                       cell_strain_operations,
                                                                       nitsche_strain_operations,
                                                                       id);
        };

      /// LAMBDA assemble grad-div
      const auto & assembler_graddiv_matrices =
        [](const SubdomainHandler<dim, double> &                                 data,
           std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
           const EquationData &                                                  equation_data,
           const std::pair<unsigned int, unsigned int> &                         subdomain_range) {
          std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
          for(unsigned int comp = 0; comp < dim; ++comp)
            fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

          std::vector<CellGradDiv> cell_graddiv_operations;
          for(unsigned int comp = 0; comp < dim; ++comp)
            cell_graddiv_operations.emplace_back(equation_data, comp);
          std::vector<NitscheGradDiv> nitsche_graddiv_operations;
          for(unsigned int comp = 0; comp < dim; ++comp)
            nitsche_graddiv_operations.emplace_back(equation_data, comp);

          for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
            FDMatrixIntegrator::assemble_graddiv_tensors(
              matrices[id], fd_evals, cell_graddiv_operations, nitsche_graddiv_operations, id);
        };

      const auto & partition_data = patch_storage->get_partition_data();
      std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> elasticity_matrices;
      elasticity_matrices.resize(partition_data.n_subdomains());
      const auto & equation_data = combi_operator->get_equation_data();

      for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
      {
        /// assemble first strain-strain and then grad-div as
        /// 'assemble_strain_matrices' initializes new matrices
        patch_storage
          ->template loop<EquationData,
                          std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
            assembler_strain_matrices, elasticity_matrices, equation_data, color);
        patch_storage
          ->template loop<EquationData,
                          std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
            assembler_graddiv_matrices, elasticity_matrices, equation_data, color);
      }

      return elasticity_matrices;
    }

    const LevelMatrix *                                  combi_operator;
    std::shared_ptr<const MatrixFree<dim, double>>       data;
    std::shared_ptr<const SubdomainHandler<dim, double>> patch_storage;
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

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::cell;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;

    const auto new_problem =
      std::make_shared<LinElasticityProblem>(*pcout_owned, rt_parameters, params.equation_data);
    new_problem->create_triangulation();
    new_problem->distribute_dofs();
    new_problem->prepare_system(false, /*compute_rhs?*/ false);
    new_problem->prepare_multigrid();
    // new_problem->assemble_matrix(); ???
    linelasticity_problem = new_problem;

    const auto level  = linelasticity_problem->mg_matrices.max_level();
    ex.combi_operator = &(linelasticity_problem->mg_matrices[level]);
    ex.data           = ex.combi_operator->get_matrix_free();
    ex.patch_storage  = linelasticity_problem->get_subdomain_handler(level);

    ass.combi_operator = ex.combi_operator;
    ass.data           = ex.data;
    ass.patch_storage  = ex.patch_storage;
  }

  void
  test()
  {
    params.n_refinements = 0;
    initialize();

    const auto level_matrix = ass.assemble_level_matrix();
    level_matrix->print_formatted(pcout_owned->get_stream());
    const auto dof_to_cell_index = ex.map_dof_to_cell_index();
    for(auto i = 0; i < dof_to_cell_index.size(); ++i)
      std::cout << dof_to_cell_index[i] << std::endl;
    const auto cell_to_patch_index = ex.map_cell_to_patch_index();
    for(auto i = 0; i < cell_to_patch_index.size(); ++i)
      std::cout << cell_to_patch_index[i] << std::endl;

    const auto mass_matrices       = ass.assemble_mass_matrices();
    const auto elasticity_matrices = ass.assemble_elasticity_matrices();
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
  Assembler                                   ass;
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
