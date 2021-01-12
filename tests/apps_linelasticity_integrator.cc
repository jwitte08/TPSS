
/*
 * Test linear elasticity integrators.
 *
 * 1) Comparison of MatrixFree and MeshWorker.
 * 2) Comparison of MatrixFree and fast diagonal integrators
 *    (a) block diagonal only
 *    (b) exact inversion
 *    (c) fast diagonalized inverse Schur complement WIP
 *
 *  Created on: Jun 27, 2019
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "linelasticity_problem.h"

/// tests/
#include "apps_linelasticity.h"
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
    (void)n_cells, (void)n_blocks, (void)n_dofs;
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

      Util::compare_vector(dst, dst_sparse, pcout);
    }
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

  Fixture::params.equation_data.lambda    = 1.234;
  Fixture::params.equation_data.mu        = 9.876;
  Fixture::params.equation_data.ip_factor = 1.99;
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
  enum class TestVariant
  {
    matrix,
    inverse
  };

  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();
  using LinElasticityProblem     = typename LinElasticity::
    ModelProblem<dim, fe_degree, double, Tensors::BlockMatrix<dim, VectorizedArray<double>>>;
  using LinElasticityProblemFast = typename LinElasticity::
    ModelProblem<dim, fe_degree, double, Tensors::BlockMatrix<dim, VectorizedArray<double>, 0>>;
  using BlockVector                        = LinearAlgebra::distributed::BlockVector<double>;
  using LevelMatrix                        = typename LinElasticityProblem::LEVEL_MATRIX;
  using PatchTransfer                      = typename LevelMatrix::transfer_type;
  using VectorizedMatrixType               = Table<2, VectorizedArray<double>>;
  static constexpr unsigned int fe_order   = fe_degree + 1;
  static constexpr unsigned int macro_size = VectorizedArray<double>::size();

  struct Params
  {
    unsigned int       n_refinements = 0;
    EquationData       equation_data;
    TPSS::PatchVariant patch_variant = TPSS::PatchVariant::cell;
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
    extract_dof_indices_per_patch(const unsigned patch, const unsigned lane = 0)
    {
      Assert(patch_transfer, ExcMessage("Patch transfer not initialized."));
      patch_transfer->reinit(patch);

      BlockVector global_indices;
      combi_operator->initialize_dof_vector(global_indices);
      for(auto i = 0U; i < global_indices.size(); ++i)
        global_indices(i) = static_cast<double>(i) + 0.1;

      const auto                indices      = patch_transfer->gather(global_indices);
      const auto                indices_view = make_array_view(indices.begin(), indices.end());
      const auto                indices_lane = array_view_to_vector(indices_view, lane);
      std::vector<unsigned int> indices_per_patch;
      std::copy(indices_lane.begin(), indices_lane.end(), std::back_inserter(indices_per_patch));
      return indices_per_patch;
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

    std::vector<std::vector<unsigned int>>
    map_cell_to_dof_indices() const
    {
      const auto                             dof_to_cell      = map_dof_to_cell_index();
      const auto                             n_physical_cells = data->n_physical_cells();
      std::vector<std::vector<unsigned int>> cell_to_dofs(n_physical_cells);
      for(auto dof_index = 0U; dof_index < dof_to_cell.size(); ++dof_index)
        cell_to_dofs[dof_to_cell[dof_index]].push_back(dof_index);
      // // redundant
      // for (auto & dof_indices : cell_to_dofs)
      // 	{
      // 	  std::sort(dof_indices.begin(), dof_indices.end());
      // 	  const auto new_end = std::unique(dof_indices.begin(), dof_indices.end());
      // 	  dof_indices.erase(new_end, dof_indices.end());
      // 	}
      return cell_to_dofs;
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
          const auto & cell            = collection.front()[lane];
          cell_to_patch[cell->index()] = {p, lane};
        }
      }
      return cell_to_patch;
    }

    const LevelMatrix *                                  combi_operator;
    std::shared_ptr<const MatrixFree<dim, double>>       data;
    std::shared_ptr<const SubdomainHandler<dim, double>> patch_storage;
    std::shared_ptr<PatchTransfer>                       patch_transfer;
  };

  struct Assembler
  {
    using FDMatrixIntegrator = typename LinElasticity::FD::MatrixIntegrator<dim, fe_degree, double>;
    using EvaluatorType      = typename FDMatrixIntegrator::EvaluatorType;
    // using CellMass           = typename FDMatrixIntegrator::CellMass;
    // using CellStrain         = typename FDMatrixIntegrator::template CellStrain<EvaluatorType>;
    // using CellDerivative     = typename FDMatrixIntegrator::template
    // CellDerivative<EvaluatorType>; using CellGradDiv        = typename
    // FDMatrixIntegrator::template CellGradDiv<EvaluatorType>; using NitscheStrain      = typename
    // FDMatrixIntegrator::template NitscheStrain<EvaluatorType>; using NitscheGradDiv     =
    // typename FDMatrixIntegrator::template NitscheGradDiv<EvaluatorType>;


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


    // std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>
    // assemble_mass_matrices() const
    // {
    //   const auto & assembler_mass_matrices =
    //     [](const SubdomainHandler<dim, double> &                                 data,
    //        std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
    //        const bool & /*dummy*/,
    //        const std::pair<unsigned int, unsigned int> & subdomain_range) {
    //       std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    //       /// initialize CellMass operation for each component
    //       std::vector<CellMass> cell_mass_operations;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //       {
    //         auto &               fd_eval = *(fd_evals[comp]);
    //         VectorizedMatrixType cell_mass_unit(fe_order, fe_order);
    //         fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
    //         cell_mass_operations.emplace_back(cell_mass_unit);
    //       }

    //       /// scales 1D unit mass with appropriate h for each component
    //       for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    //         matrices[id] =
    //           FDMatrixIntegrator::assemble_mass_tensors(fd_evals, cell_mass_operations, id);
    //     };

    //   const auto & partition_data = patch_storage->get_partition_data();
    //   std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> mass_matrices;
    //   mass_matrices.resize(partition_data.n_subdomains());
    //   for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    //     patch_storage
    //       ->template loop<bool,
    //                       std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
    //         assembler_mass_matrices, mass_matrices, /*dummy*/ false, color);

    //   return mass_matrices;
    // }


    // std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>
    // assemble_elasticity_matrices() const
    // {
    //   /// LAMBDA assemble strain : strain
    //   const auto & assembler_strain_matrices =
    //     [](const SubdomainHandler<dim, double> &                                 data,
    //        std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
    //        const EquationData &                                                  equation_data,
    //        const std::pair<unsigned int, unsigned int> &                         subdomain_range)
    //        {
    //       std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    //       std::vector<CellStrain> cell_strain_operations;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         cell_strain_operations.emplace_back(equation_data, comp);
    //       std::vector<NitscheStrain> nitsche_strain_operations;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         nitsche_strain_operations.emplace_back(equation_data, comp);

    //       for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    //         matrices[id] = FDMatrixIntegrator::assemble_strain_tensors(fd_evals,
    //                                                                    cell_strain_operations,
    //                                                                    nitsche_strain_operations,
    //                                                                    id);
    //     };

    //   /// LAMBDA assemble grad-div
    //   const auto & assembler_graddiv_matrices =
    //     [](const SubdomainHandler<dim, double> &                                 data,
    //        std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
    //        const EquationData &                                                  equation_data,
    //        const std::pair<unsigned int, unsigned int> &                         subdomain_range)
    //        {
    //       std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    //       std::vector<CellGradDiv> cell_graddiv_operations;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         cell_graddiv_operations.emplace_back(equation_data, comp);
    //       std::vector<NitscheGradDiv> nitsche_graddiv_operations;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         nitsche_graddiv_operations.emplace_back(equation_data, comp);

    //       for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    //         FDMatrixIntegrator::assemble_graddiv_tensors(
    //           matrices[id], fd_evals, cell_graddiv_operations, nitsche_graddiv_operations, id);
    //     };

    //   const auto & partition_data = patch_storage->get_partition_data();
    //   std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> elasticity_matrices;
    //   elasticity_matrices.resize(partition_data.n_subdomains());
    //   const auto & equation_data = combi_operator->get_equation_data();

    //   /// assemble first strain-strain and then grad-div because
    //   /// 'assemble_strain_matrices' initializes new matrices
    //   for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    //   {
    //     patch_storage
    //       ->template loop<EquationData,
    //                       std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
    //         assembler_strain_matrices, elasticity_matrices, equation_data, color);
    //     patch_storage
    //       ->template loop<EquationData,
    //                       std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
    //         assembler_graddiv_matrices, elasticity_matrices, equation_data, color);
    //   }

    //   return elasticity_matrices;
    // }

    // // std::vector<std::pair<std::vector<VectorizedMatrixType>,
    // std::vector<VectorizedMatrixType>>> std::vector<std::vector<std::array<VectorizedMatrixType,
    // dim>>> assemble_block10() const
    // {
    //   /// LAMBDA assembles the univariate matrices of the (1,0)-block for each subdomain
    //   const auto & assembler_block10 =
    //     [&](const SubdomainHandler<dim, double> &                             data,
    //         std::vector<std::vector<std::array<VectorizedMatrixType, dim>>> & tensors,
    //         const EquationData &                                              equation_data,
    //         const std::pair<unsigned int, unsigned int> &                     subdomain_range) {
    //       std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
    //       for(unsigned int comp = 0; comp < dim; ++comp)
    //         fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    //       for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    //       {
    //         tensors[id] = FDMatrixIntegrator::assemble_mixed_block(
    //           fd_evals, equation_data, /*component_v*/ 1U, /*component_u*/ 0U, id);
    //       }
    //     };

    //   const auto & partition_data = patch_storage->get_partition_data();
    //   std::vector<std::vector<std::array<VectorizedMatrixType, dim>>> blocks;
    //   blocks.resize(partition_data.n_subdomains());
    //   const auto & equation_data = combi_operator->get_equation_data();
    //   for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    //   {
    //     patch_storage
    //       ->template loop<EquationData,
    //                       std::vector<std::vector<std::array<VectorizedMatrixType, dim>>>>(
    //         assembler_block10, blocks, equation_data, color);
    //   }

    //   return blocks;
    // }

    const LevelMatrix *                                  combi_operator;
    std::shared_ptr<const MatrixFree<dim, double>>       data;
    std::shared_ptr<const SubdomainHandler<dim, double>> patch_storage;
  };

  void
  SetUp() override
  {
    ofs.open("apps_linelasticity_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout_owned              = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;
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

    const auto initialize_problem = [](auto & new_problem) {
      new_problem->create_triangulation();
      new_problem->distribute_dofs();
      new_problem->prepare_system(false, /*compute_rhs?*/ false);
      new_problem->prepare_multigrid();
    };
    *pcout_owned << params.equation_data.to_string() << std::endl;
    const auto new_problem =
      std::make_shared<LinElasticityProblem>(*pcout_owned, rt_parameters, params.equation_data);
    initialize_problem(new_problem);
    linelasticity_problem = new_problem;

    const auto level  = linelasticity_problem->mg_matrices.max_level();
    ex.combi_operator = &(linelasticity_problem->mg_matrices[level]);
    ex.data           = ex.combi_operator->get_matrix_free();
    ex.patch_storage  = linelasticity_problem->get_subdomain_handler(level);
    ex.patch_transfer = std::make_shared<PatchTransfer>(*(ex.patch_storage));

    ass.combi_operator = ex.combi_operator;
    ass.data           = ex.data;
    ass.patch_storage  = ex.patch_storage;
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

  void
  compare_inverse_matrix(const FullMatrix<double> & inverse_patch_matrix,
                         const FullMatrix<double> & other) const
  {
    Util::compare_inverse_matrix(inverse_patch_matrix, other, *pcout_owned);
  }

  // void
  // compare_inverse_matrix(const FullMatrix<double> & other) const
  // {
  //   const auto inverse_patch_matrix = table_to_fullmatrix(patch_matrix.as_inverse_table());
  //   compare_inverse_matrix(inverse_patch_matrix, other);
  // }

  template<typename OtherNumber>
  std::vector<std::array<Table<2, OtherNumber>, dim>>
  zero_tensor(const unsigned int m)
  {
    std::vector<std::array<Table<2, OtherNumber>, dim>> zeros;
    const auto & zero_matrix_factory = [](const unsigned int m) {
      Table<2, OtherNumber> zero(m, m);
      zero.fill(0.);
      std::array<Table<2, OtherNumber>, dim> zero_tensor;
      std::fill(zero_tensor.begin(), zero_tensor.end(), zero);
      return zero_tensor;
    };
    zeros.emplace_back(zero_matrix_factory(m));
    return zeros;
  }

  // void
  // userdefined_assembly(const TestVariant test_variant = TestVariant::matrix)
  // {
  //   using State = typename Tensors::TensorProductMatrix<dim, double>::State;
  //   initialize();

  //   const std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> mass_matrices =
  //     ass.assemble_mass_matrices();
  //   const std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> elasticity_matrices
  //   =
  //     ass.assemble_elasticity_matrices();
  //   const std::vector<std::vector<std::array<VectorizedMatrixType, dim>>> mixed_tensors10 =
  //     ass.assemble_block10();

  //   const auto level_matrix        = ass.assemble_level_matrix();
  //   const auto cell_to_patch_index = ex.map_cell_to_patch_index();
  //   const auto cell_to_dofs        = ex.map_cell_to_dof_indices();
  //   // patch_matrix.resize(dim, dim);
  //   for(auto cell = 0U; cell < cell_to_patch_index.size(); ++cell)
  //   {
  //     patch_matrix.resize(dim, dim);
  //     auto [patch, lane] = cell_to_patch_index[cell];

  //     /// block diagonal
  //     for(auto comp = 0U; comp < dim; ++comp)
  //     {
  //       const auto & mass_tensor_comp = mass_matrices[patch][comp];
  //       const auto & elas_tensor_comp = elasticity_matrices[patch][comp];
  //       std::vector<std::array<Table<2, double>, dim>> tensors_comp;
  //       std::array<Table<2, double>, dim>              masses;
  //       std::transform(mass_tensor_comp.cbegin(),
  //                      mass_tensor_comp.cend(),
  //                      masses.begin(),
  //                      [lane](const auto & table) { return table_to_fullmatrix(table, lane); });
  //       std::array<Table<2, double>, dim> elases;
  //       std::transform(elas_tensor_comp.cbegin(),
  //                      elas_tensor_comp.cend(),
  //                      elases.begin(),
  //                      [lane](const auto & table) { return table_to_fullmatrix(table, lane); });
  //       tensors_comp.emplace_back(masses);
  //       tensors_comp.emplace_back(elases);
  //       patch_matrix.get_block(comp, comp).reinit(tensors_comp, State::separable);
  //     }

  //     /// block off-diagonals
  //     {
  //       const auto &                                   macro_tensors10 = mixed_tensors10[patch];
  //       std::vector<std::array<Table<2, double>, dim>> tensors10;
  //       std::transform(macro_tensors10.cbegin(),
  //                      macro_tensors10.cend(),
  //                      std::back_inserter(tensors10),
  //                      [lane](const auto & t) -> std::array<Table<2, double>, dim> {
  //                        return {table_to_fullmatrix(t[0], lane), table_to_fullmatrix(t[1],
  //                        lane)};
  //                      });
  //       patch_matrix.get_block(1U, 0U).reinit(tensors10);

  //       std::vector<std::array<Table<2, double>, dim>> tensors01;
  //       std::transform(macro_tensors10.cbegin(),
  //                      macro_tensors10.cend(),
  //                      std::back_inserter(tensors01),
  //                      [lane](const auto & t) -> std::array<Table<2, double>, dim> {
  //                        return {table_to_fullmatrix(LinAlg::transpose(t[0]), lane),
  //                                table_to_fullmatrix(LinAlg::transpose(t[1]), lane)};
  //                      });
  //       patch_matrix.get_block(0U, 1U).reinit(tensors01);
  //     }

  //     const auto & level_dof_indices = cell_to_dofs[cell];
  //     AssertDimension(level_dof_indices.size(), patch_matrix.m());
  //     AssertDimension(level_dof_indices.size(), patch_matrix.n());
  //     FullMatrix<double> extracted_matrix(patch_matrix.m(), patch_matrix.n());
  //     extracted_matrix.extract_submatrix_from(*level_matrix, level_dof_indices,
  //     level_dof_indices);

  //     if(test_variant == TestVariant::matrix)
  //       compare_matrix(extracted_matrix);
  //     else if(test_variant == TestVariant::inverse)
  //       compare_inverse_matrix(extracted_matrix);
  //   }
  // }

  void
  tpss_assembly(const TestVariant test_variant = TestVariant::matrix)
  {
    initialize();

    const auto level_matrix = ass.assemble_level_matrix();
    const auto schwarz_preconditioner =
      linelasticity_problem->mg_schwarz_smoother_pre->get_preconditioner();
    const auto & patch_matrices = *(schwarz_preconditioner->get_local_solvers());
    const auto   subdomain_handler =
      linelasticity_problem->mg_schwarz_smoother_pre->get_subdomain_handler();
    const auto &                   patch_info = subdomain_handler->get_patch_info();
    TPSS::PatchWorker<dim, double> worker(patch_info);
    const auto &                   partition_data = subdomain_handler->get_partition_data();

    for(auto patch = 0U; patch < partition_data.n_subdomains(); ++patch)
    {
      for(auto lane = 0U; lane < worker.n_lanes_filled(patch); ++lane)
      {
        /// extract reference matrix from level matrix
        const auto         dof_indices = ex.extract_dof_indices_per_patch(patch, lane);
        FullMatrix<double> patch_matrix_reference(dof_indices.size());
        patch_matrix_reference.extract_submatrix_from(*level_matrix, dof_indices, dof_indices);

        /// compare patch matrix
        auto patch_matrix_vectorized = patch_matrices[patch];
        if(test_variant == TestVariant::matrix)
        {
          const auto patch_matrix_full =
            table_to_fullmatrix(patch_matrix_vectorized.as_table(), lane);
          *pcout_owned << "Compare patch matrix " << patch << "@ lane " << lane << ":\n";
          compare_matrix(patch_matrix_full, patch_matrix_reference);
        }

        /// compare inverse patch matrix
        if(test_variant == TestVariant::inverse)
        {
          const auto patch_matrix_inverse =
            table_to_fullmatrix(patch_matrix_vectorized.as_inverse_table(), lane);
          *pcout_owned << "Compare inverse patch matrix " << patch << "@ lane " << lane << ":\n";
          compare_inverse_matrix(patch_matrix_inverse, patch_matrix_reference);
        }
      }
    }
  }

  void
  tpss_assembly_fast()
  {
    initialize();

    const auto level_matrix = ass.assemble_level_matrix();
    const auto schwarz_preconditioner =
      linelasticity_problem->mg_schwarz_smoother_pre->get_preconditioner();
    const auto & patch_matrices = *(schwarz_preconditioner->get_local_solvers());
    const auto   subdomain_handler =
      linelasticity_problem->mg_schwarz_smoother_pre->get_subdomain_handler();
    const auto &                   patch_info = subdomain_handler->get_patch_info();
    TPSS::PatchWorker<dim, double> worker(patch_info);
    const auto &                   partition_data = subdomain_handler->get_partition_data();

    for(auto patch = 0U; patch < partition_data.n_subdomains(); ++patch)
    {
      for(auto lane = 0U; lane < worker.n_lanes_filled(patch); ++lane)
      {
        /// extract reference matrix from level matrix
        const auto         dof_indices = ex.extract_dof_indices_per_patch(patch, lane);
        FullMatrix<double> patch_matrix_reference(dof_indices.size());
        patch_matrix_reference.extract_submatrix_from(*level_matrix, dof_indices, dof_indices);

        using TensorProductMatrixType =
          typename Tensors::TensorProductMatrix<dim, VectorizedArray<double>>;
        using SchurType = typename Tensors::SchurComplementFast<dim, VectorizedArray<double>>;
        auto patch_matrix_vectorized = patch_matrices[patch];
        Tensors::BlockGaussianInverse<TensorProductMatrixType, SchurType> Atilde_inv(
          patch_matrix_vectorized.get_block(0, 0),
          patch_matrix_vectorized.get_block(0, 1),
          patch_matrix_vectorized.get_block(1, 0),
          patch_matrix_vectorized.get_block(1, 1));
        const auto Atilde_inv_full = table_to_fullmatrix(Atilde_inv.as_table(), lane);

        FullMatrix<double> ID(IdentityMatrix(Atilde_inv.m())), Diff(Atilde_inv.m(), Atilde_inv.n());
        Atilde_inv_full.mmult(Diff, patch_matrix_reference);
        Diff.add(-1., ID);
        const auto deviation_frob = Diff.frobenius_norm() / (Diff.m() * Diff.n());
        *pcout_owned << "||Atilde^{-1} A - ID||_frob / m*n : " << deviation_frob << std::endl;
        EXPECT_PRED_FORMAT2(::testing::DoubleLE, deviation_frob, 1.e-2);

        // /// DEBUG
        // *pcout_owned << "Compare inverse patch matrix (fast) " << patch << "@ lane " << lane
        //              << ":\n";
        // const auto patch_matrix_inverse =
        //   table_to_fullmatrix(patch_matrix_vectorized.as_inverse_table(), lane);
        // compare_matrix(Atilde_inv_full, patch_matrix_inverse);
      }
    }
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  Params                                      params;
  RT::Parameter                               rt_parameters;
  std::shared_ptr<const LinElasticityProblem> linelasticity_problem;
  Extractor                                   ex;
  Assembler                                   ass;
  // Tensors::BlockMatrix<dim, double>           patch_matrix;
};


TYPED_TEST_SUITE_P(TestLinElasticityIntegratorFD);

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSPenaltyAssemblyCellPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::params.equation_data.ip_variant = LinElasticity::EquationData::PenaltyVariant::tensor;

  Fixture::params.n_refinements        = 0U;
  Fixture::params.equation_data.lambda = 1.;
  Fixture::params.equation_data.mu     = 1.;
  Fixture::tpss_assembly();

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 1U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 1.99;
  Fixture::tpss_assembly();
}

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSPenaltyInvertCellPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::params.equation_data.ip_variant = LinElasticity::EquationData::PenaltyVariant::tensor;

  Fixture::params.n_refinements        = 0U;
  Fixture::params.equation_data.lambda = 1.;
  Fixture::params.equation_data.mu     = 1.;
  Fixture::tpss_assembly(Fixture::TestVariant::inverse);

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 2U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 2.99;
  Fixture::tpss_assembly(Fixture::TestVariant::inverse);
}

/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Test the fast diagonal matrix integrator (cell patch) directly by means of
/// SchwarzPreconditioner

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSAssemblyCellPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::params.n_refinements        = 0U;
  Fixture::params.equation_data.lambda = 1.;
  Fixture::params.equation_data.mu     = 1.;
  Fixture::tpss_assembly();

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 1U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 4.99;
  Fixture::tpss_assembly();
}

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSInvertCellPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::params.n_refinements        = 0U;
  Fixture::params.equation_data.lambda = 1.;
  Fixture::params.equation_data.mu     = 1.;
  Fixture::tpss_assembly(Fixture::TestVariant::inverse);

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 2U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 5.99;
  Fixture::tpss_assembly(Fixture::TestVariant::inverse);
}

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSAssemblyVertexPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant  = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;

  /// TODO vertex patches are not build for subdivided mesh ?!
  // Fixture::rt_parameters.mesh.n_subdivisions.resize(Fixture::dim, 2);
  // Fixture::rt_parameters.mesh.n_subdivisions.at(0) = 3;
  // Fixture::params.n_refinements        = 2U;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 0U;
  Fixture::params.equation_data.lambda         = 1.;
  Fixture::params.equation_data.mu             = 1.;
  Fixture::tpss_assembly();

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 3U;
  Fixture::params.n_refinements                = 1U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 6.99;
  Fixture::tpss_assembly();
}

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSInvertVertexPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant  = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 0U;
  Fixture::params.equation_data.lambda         = 1.;
  Fixture::params.equation_data.mu             = 1.;
  Fixture::tpss_assembly(Fixture::TestVariant::inverse);

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 3U;
  Fixture::params.n_refinements                = 1U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 7.99;
  Fixture::tpss_assembly(Fixture::TestVariant::inverse);
}

TYPED_TEST_P(TestLinElasticityIntegratorFD, TPSSFastInvertCellPatch)
{
  using Fixture = TestLinElasticityIntegratorFD<TypeParam>;

  Fixture::params.n_refinements        = 0U;
  Fixture::params.equation_data.lambda = 1.;
  Fixture::params.equation_data.mu     = 1.;
  Fixture::tpss_assembly_fast();

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2U;
  Fixture::params.n_refinements                = 2U;
  Fixture::params.equation_data.lambda         = 1.234;
  Fixture::params.equation_data.mu             = 9.876;
  Fixture::params.equation_data.ip_factor      = 8.99;
  Fixture::tpss_assembly_fast();
}

REGISTER_TYPED_TEST_SUITE_P(TestLinElasticityIntegratorFD,
                            TPSSPenaltyAssemblyCellPatch,
                            TPSSPenaltyInvertCellPatch,
                            TPSSAssemblyCellPatch,
                            TPSSInvertCellPatch,
                            TPSSAssemblyVertexPatch,
                            TPSSInvertVertexPatch,
                            TPSSFastInvertCellPatch);

using TestParamsLinear = testing::Types<Util::NonTypeParams<2, 1>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLinElasticityIntegratorFD, TestParamsLinear);

using TestParamsHigherOrder = testing::Types<Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestLinElasticityIntegratorFD, TestParamsHigherOrder);



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
