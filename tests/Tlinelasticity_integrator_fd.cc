
/*
 * Linear Elasticity (Test MatrixFree Integrators)
 *
 *  Created on: Jul 23, 2019
 *      Author: witte
 */

// #include "Plaplace_fdss.h"
#include "ct_parameter.h"
#include "linelasticity_problem.h"

using namespace dealii;
using namespace LinElasticity;



struct TestParameter
{
  unsigned int                   n_refinements = 0;
  bool                           print_details = false;
  EquationData                   equation_data;
  TPSS::PatchVariant             patch_variant         = TPSS::PatchVariant::cell;
  MeshParameter::GeometryVariant geometry_variant      = MeshParameter::GeometryVariant::Cube;
  bool                           test_diagonal_blocks  = true;
  bool                           do_offdiagonal_blocks = true;
};

template<typename MatrixType, typename Number = double>
FullMatrix<Number>
extract_block(const MatrixType & matrix, std::array<unsigned, 2> start, std::array<unsigned, 2> n)
{
  FullMatrix<Number> block(n[0], n[1]);
  for(unsigned int i = 0; i < n[0]; ++i)
    for(unsigned int j = 0; j < n[1]; ++j)
      block(i, j) = matrix(start[0] + i, start[1] + j);
  return block;
}

template<int dim, int fe_degree, typename value_type = double>
struct Test
{
  static constexpr unsigned fe_order   = fe_degree + 1;
  static constexpr unsigned macro_size = VectorizedArray<value_type>::n_array_elements;

  using LinElasticityOperator = typename LinElasticity::ModelProblem<dim, fe_degree, value_type>;
  using VectorizedMatrixType  = Table<2, VectorizedArray<value_type>>;
  using FDMatrixIntegrator    = typename FD::MatrixIntegrator<dim, fe_degree, value_type>;
  using EvaluatorType         = typename FDMatrixIntegrator::EvaluatorType;
  using CellMass              = typename FDMatrixIntegrator::CellMass;
  using CellStrain            = typename FDMatrixIntegrator::template CellStrain<EvaluatorType>;
  using CellGradMixed         = typename FDMatrixIntegrator::template CellGradMixed<EvaluatorType>;
  using CellGradDiv           = typename FDMatrixIntegrator::template CellGradDiv<EvaluatorType>;
  using NitscheStrain         = typename FDMatrixIntegrator::template NitscheStrain<EvaluatorType>;
  using NitscheGradDiv        = typename FDMatrixIntegrator::template NitscheGradDiv<EvaluatorType>;

  std::shared_ptr<LinElasticityOperator>                              linelasticity_problem;
  SparseMatrix<value_type> *                                          sparse_matrix;
  std::shared_ptr<const SubdomainHandler<dim, value_type>>            patch_storage;
  std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> basic_matrices;
  std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> action_matrices;
  std::vector<VectorizedMatrixType>                                   elasticity_blocks10;

  Test(const TestParameter params)
  {
    const bool         is_mpi_process0 = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const bool         print_details   = is_mpi_process0 && params.print_details;
    ConditionalOStream pcout(std::cout, print_details);
    RT::Parameter      rt_parameters;
    rt_parameters.mesh.n_refinements = params.n_refinements;
    rt_parameters.mesh.n_repetitions = 1;
    rt_parameters.mesh.geometry_variant =
      params.geometry_variant; // Laplace::Parameter::GeometryVariant::SubdividedCubeoid;
    // auto & n_subdivisions = rt_parameters.mesh.n_subdivisions;
    // n_subdivisions.resize(dim);
    // std::fill(n_subdivisions.begin(), n_subdivisions.end(), 1);
    // n_subdivisions[0]                              = 2;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;

    linelasticity_problem =
      std::make_shared<LinElasticityOperator>(pcout, rt_parameters, params.equation_data);
    const bool mesh_is_valid = linelasticity_problem->create_triangulation();
    AssertThrow(mesh_is_valid, ExcMessage("Mesh is not initialized."));
    linelasticity_problem->distribute_dofs();
    linelasticity_problem->prepare_system(false, /*compute_rhs?*/ false);

    linelasticity_problem->assemble_matrix();
    sparse_matrix = &linelasticity_problem->sparse_matrix;
    // const auto & mf_operator   = linelasticity_problem->system_matrix;
    if(pcout.is_active())
    {
      pcout << "SPARSE MATRIX\n\n";
      sparse_matrix->print_formatted(std::cout /*, 1, true*/);
      pcout << std::endl;
    }

    const unsigned int level = linelasticity_problem->level;
    const auto         mf_storage_level =
      linelasticity_problem->template build_mf_storage<value_type>(level);
    patch_storage =
      linelasticity_problem->template build_patch_storage<value_type>(level, mf_storage_level);
    const auto & partition_data = patch_storage->get_patch_info().subdomain_partition_data;

    const auto & assembler_mass_matrices =
      [](const SubdomainHandler<dim, double> &                                 data,
         std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>> & matrices,
         const bool &,
         const std::pair<unsigned int, unsigned int> & subdomain_range) {
        std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
        for(unsigned int comp = 0; comp < dim; ++comp)
          fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

        std::vector<CellMass> cell_mass_operations;
        for(unsigned int comp = 0; comp < dim; ++comp)
        {
          auto &               fd_eval = *(fd_evals[comp]);
          VectorizedMatrixType cell_mass_unit(fe_order, fe_order);
          fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
          cell_mass_operations.emplace_back(cell_mass_unit);
        }

        for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
          matrices[id] =
            FDMatrixIntegrator::assemble_mass_tensors(fd_evals, cell_mass_operations, id);
      };

    const auto & assembler_elasticity_matrices =
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

    const auto & assembler_offdiag_matrices =
      [](const SubdomainHandler<dim, double> &         data,
         std::vector<VectorizedMatrixType> &           matrices,
         const EquationData &                          equation_data,
         const std::pair<unsigned int, unsigned int> & subdomain_range,
         const unsigned int                            component_u,
         const unsigned int                            component_v) {
        std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
        for(unsigned int comp = 0; comp < dim; ++comp)
          fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

        for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
        {
          // // DEBUG
          //   matrices[id] = FDMatrixIntegrator::assemble_strain_mixed(fd_evals, equation_data,
          //   1, 0, id);
          // std::cout << "BLOCK01: " << std::endl;
          // table_to_fullmatrix(matrices[id]).print_formatted(std::cout);
          // std::cout << std::endl;

          matrices[id] = FDMatrixIntegrator::assemble_strain_mixed(
            fd_evals, equation_data, component_u, component_v, id);
          const auto & matrix_graddiv = FDMatrixIntegrator::assemble_graddiv_mixed(
            fd_evals, equation_data, component_u, component_v, id);
          matrices[id] = Tensors::sum(matrices[id], matrix_graddiv);

          // // DEBUG
          // std::cout << "transpose of BLOCK10: " << std::endl;
          // table_to_fullmatrix(Tensors::transpose(matrices[id])).print_formatted(std::cout);
          // std::cout << std::endl;
          // std::cout << "BLOCK10: " << std::endl;
          // table_to_fullmatrix(matrices[id]).print_formatted(std::cout);
          // std::cout << std::endl;
        }
      };

    basic_matrices.clear();
    basic_matrices.resize(partition_data.n_subdomains());
    action_matrices.clear();
    action_matrices.resize(partition_data.n_subdomains());
    std::vector<std::array<VectorizedMatrixType, dim>> gradmixed10_matrices;
    gradmixed10_matrices.clear();
    gradmixed10_matrices.resize(partition_data.n_subdomains());
    elasticity_blocks10.clear();
    elasticity_blocks10.resize(partition_data.n_subdomains());
    for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    {
      // *** MASS
      patch_storage
        ->template loop<bool, std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
          assembler_mass_matrices, basic_matrices, false, color);

      // *** ELASTICITY (diagonal)
      patch_storage
        ->template loop<EquationData,
                        std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
          assembler_elasticity_matrices,
          action_matrices,
          linelasticity_problem->equation_data,
          color);
      patch_storage
        ->template loop<EquationData,
                        std::vector<std::array<std::array<VectorizedMatrixType, dim>, dim>>>(
          assembler_graddiv_matrices, action_matrices, linelasticity_problem->equation_data, color);

      // *** ELASTICITY (off-diagonal)
      const unsigned int component_v = 1;
      const unsigned int component_u = 0;
      const auto &       assembler_offdiag_matrices_bind =
        [&](const SubdomainHandler<dim, double> &         data,
            std::vector<VectorizedMatrixType> &           matrices,
            const EquationData &                          equation_data,
            const std::pair<unsigned int, unsigned int> & subdomain_range) {
          return assembler_offdiag_matrices(
            data, matrices, equation_data, subdomain_range, component_u, component_v);
        };
      patch_storage->template loop<EquationData, std::vector<VectorizedMatrixType>>(
        assembler_offdiag_matrices_bind,
        elasticity_blocks10,
        linelasticity_problem->equation_data,
        color);
    }

    // *** TEST
    compare_blocks(params.test_diagonal_blocks, params.do_offdiagonal_blocks, print_details);
  }

  void
  compare_blocks(const bool test_diag,
                 const bool test_offdiag,
                 const bool print_details = false) const
  {
    const auto patch_variant =
      linelasticity_problem->rt_parameters.multigrid.pre_smoother.schwarz.patch_variant;
    if(patch_variant == TPSS::PatchVariant::cell)
      compare_blocks_cp_impl(test_diag, test_offdiag, print_details);
    else if(patch_variant == TPSS::PatchVariant::vertex)
      compare_blocks_vp_impl(print_details);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  void
  compare_blocks_cp_impl(const bool test_diag,
                         const bool test_offdiag,
                         const bool print_details = false) const
  {
    const auto     n_cells      = linelasticity_problem->triangulation.n_active_cells();
    const unsigned n_components = dim;
    // const auto     n_dofs       = linelasticity_problem->system_u.size();
    AssertDimension(1, linelasticity_problem->fe->n_base_elements());
    const auto n_dofs_per_cell_scalar = linelasticity_problem->fe->n_dofs_per_cell();
    const auto n_dofs_per_cell        = n_dofs_per_cell_scalar * n_components;

    const auto &                       patch_info = patch_storage->get_patch_info();
    TPSS::PatchWorker<dim, value_type> patch_worker{patch_info};
    std::ostringstream                 oss;

    /**
     * Comparison of diagonal blocks (w.r.t. components) within each
     * cell patch block (i.e. the diagonal blocks of the system
     * matrix)
     */
    if(test_diag)
      for(unsigned patch_id = 0; patch_id < basic_matrices.size(); ++patch_id)
        for(unsigned lane = 0; lane < patch_worker.n_lanes_filled(patch_id); ++lane)
          for(unsigned comp = 0; comp < n_components; ++comp)
          {
            // *** extract block from FD integrator
            // const unsigned patch_id = cell / macro_size;
            // const unsigned lane     = cell % macro_size;
            const auto fdmatrix = Tensors::assemble_separableKD(basic_matrices[patch_id][comp],
                                                                action_matrices[patch_id][comp]);
            const auto mat      = table_to_fullmatrix(fdmatrix, lane);
            if(print_details)
            {
              print_row(
                std::cout, 12, "\n patch_id:", patch_id, "comp:", comp, "lane:", lane, "\n");
              mat.print_formatted(std::cout);
            }
            print_row(oss, 15, "\n FD patch_id:", patch_id, "comp:", comp, "lane:", lane, "\n");
            mat.print_formatted(oss);

            // *** extract block from sparse matrix
            const auto     cell_collection = patch_worker.get_cell_collection(patch_id);
            const auto &   cell_iterator   = cell_collection[0][lane];
            const unsigned cell            = cell_iterator->index();
            AssertIndexRange(cell, n_cells);
            std::array<unsigned, 2> start;
            std::array<unsigned, 2> size;
            start.fill(cell * n_dofs_per_cell + comp * n_dofs_per_cell_scalar);
            size.fill(n_dofs_per_cell_scalar);
            const auto & block = extract_block(*sparse_matrix, start, size);
            if(print_details)
            {
              print_row(std::cout, 12, "\n cell_id:", cell, "comp:", comp, "\n");
              block.print_formatted(std::cout);
            }
            print_row(oss, 15, "\n cell_id:", cell, "comp:", comp, "\n");
            block.print_formatted(oss);

            // *** TEST comparison of blocks
            FullMatrix<value_type> diff{mat};
            diff.add(-1., block);
            oss << "||Block_fd - Block_sparse||_frob = " << diff.frobenius_norm()
                << " exceeds bounds!";
            AssertThrow(std::abs(diff.frobenius_norm()) < (1.e-10 * block.frobenius_norm()),
                        ExcMessage(oss.str()));
          }

    /**
     * Comparison of off-diagonal blocks (w.r.t. components) within
     * each cell patch block (i.e. the diagonal blocks of the system
     * matrix)
     */
    oss.str("");
    if(test_offdiag)
      for(unsigned patch_id = 0; patch_id < basic_matrices.size(); ++patch_id)
        for(unsigned lane = 0; lane < patch_worker.n_lanes_filled(patch_id); ++lane)
          for(unsigned comp_v = 0; comp_v < n_components; ++comp_v)
            for(unsigned comp_u = 0; comp_u < n_components; ++comp_u)
              if(comp_v != comp_u)
              {
                VectorizedMatrixType fdblock;
                if(comp_v == 1 && comp_u == 0)
                  fdblock = elasticity_blocks10[patch_id];
                else if(comp_v == 0 && comp_u == 1)
                  fdblock = Tensors::transpose(elasticity_blocks10[patch_id]);
                else
                  AssertThrow(false, ExcNotImplemented());
                const auto mat = table_to_fullmatrix(fdblock, lane);
                if(print_details)
                {
                  print_row(std::cout,
                            12,
                            "\n patch_id:",
                            patch_id,
                            "comp_v:",
                            comp_v,
                            "comp_u:",
                            comp_u,
                            "lane:",
                            lane,
                            "\n");
                  mat.print_formatted(std::cout);
                }
                print_row(oss,
                          12,
                          "\n patch_id:",
                          patch_id,
                          "comp_v:",
                          comp_v,
                          "comp_u:",
                          comp_u,
                          "lane:",
                          lane,
                          "\n");
                mat.print_formatted(oss);

                // *** extract block from sparse matrix
                const auto     cell_collection = patch_worker.get_cell_collection(patch_id);
                const auto &   cell_iterator   = cell_collection[0][lane];
                const unsigned cell            = cell_iterator->index();
                AssertIndexRange(cell, n_cells);
                std::array<unsigned, 2> start;
                std::array<unsigned, 2> size;
                start[0] = (cell * n_dofs_per_cell + comp_v * n_dofs_per_cell_scalar);
                start[1] = (cell * n_dofs_per_cell + comp_u * n_dofs_per_cell_scalar);
                size.fill(n_dofs_per_cell_scalar);
                const auto & block = extract_block(*sparse_matrix, start, size);
                if(print_details)
                {
                  print_row(
                    std::cout, 12, "\n cell_id:", cell, "comp_v:", comp_v, "comp_u:", comp_u, "\n");
                  block.print_formatted(std::cout);
                }
                print_row(oss, 15, "\n cell_id:", cell, "comp_v:", comp_v, "comp_u:", comp_u, "\n");
                block.print_formatted(oss);

                // *** TEST comparison of blocks
                FullMatrix<value_type> diff{mat};
                diff.add(-1., block);
                oss << "||Block_fd - Block_sparse||_frob = " << diff.frobenius_norm()
                    << " exceeds bounds!";
                AssertThrow(std::abs(diff.frobenius_norm()) < (1.e-10 * block.frobenius_norm()),
                            ExcMessage(oss.str()));
              }
  }

  void
  compare_blocks_vp_impl(const bool print_details) const
  {
    const unsigned n_components = dim;
    // const auto     n_dofs       = linelasticity_problem->system_u.size();
    AssertDimension(1, linelasticity_problem->fe->n_base_elements());
    const auto n_dofs_per_cell_scalar = linelasticity_problem->fe->n_dofs_per_cell();
    const auto n_dofs_per_cell        = n_dofs_per_cell_scalar * n_components;

    const auto &                       patch_info = patch_storage->get_patch_info();
    TPSS::PatchWorker<dim, value_type> patch_worker{patch_info};
    std::ostringstream                 oss;
    for(unsigned patch_id = 0; patch_id < basic_matrices.size(); ++patch_id)
      for(unsigned lane = 0; lane < patch_worker.n_lanes_filled(patch_id); ++lane)
        for(unsigned comp = 0; comp < n_components; ++comp)
        {
          // *** extract block from FD integrator
          // const unsigned patch_id = cell / macro_size;
          // const unsigned lane     = cell % macro_size;
          const auto fdmatrix = Tensors::assemble_separableKD(basic_matrices[patch_id][comp],
                                                              action_matrices[patch_id][comp]);
          const auto mat      = table_to_fullmatrix(fdmatrix, lane);
          if(print_details)
          {
            print_row(std::cout, 12, "\n patch_id:", patch_id, "comp:", comp, "lane:", lane, "\n");
            mat.print_formatted(std::cout);
          }
          print_row(oss, 15, "\n FD patch_id:", patch_id, "comp:", comp, "lane:", lane, "\n");
          mat.print_formatted(oss);

          // TODO

          // // *** extract block from sparse matrix
          // const auto     cell_collection = patch_worker.get_cell_collection(patch_id);
          // const auto &   cell_iterator   = cell_collection[0][lane];
          // const unsigned cell            = cell_iterator->index();
          // AssertIndexRange(cell, n_cells);
          // std::array<unsigned, 2> start;
          // std::array<unsigned, 2> size;
          // start.fill(cell * n_dofs_per_cell + comp * n_dofs_per_cell_scalar);
          // size.fill(n_dofs_per_cell_scalar);
          // const auto & block = extract_block(*sparse_matrix, start, size);
          // if(print_details)
          // {
          //   print_row(std::cout, 12, "\n cell_id:", cell, "comp:", comp, "\n");
          //   block.print_formatted(std::cout);
          // }
          // print_row(oss, 15, "\n cell_id:", cell, "comp:", comp, "\n");
          // block.print_formatted(oss);

          // // *** TEST comparison of blocks
          // FullMatrix<value_type> diff{mat};
          // diff.add(-1., block);
          // oss << "||Block_fd - Block_sparse||_frob = " << diff.frobenius_norm()
          //     << " exceeds bounds!";
          // AssertThrow(std::abs(diff.frobenius_norm()) < (1.e-10 * block.frobenius_norm()),
          //             ExcMessage(oss.str()));
        }
  }
};

int
main(int argc, char * argv[])
{
  // *** init TBB and MPI
  const unsigned int max_threads = 1; // serial
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  {
    TestParameter test_params;
    test_params.print_details = true;
    test_params.n_refinements = 0;
    // test_params.geometry_variant = Laplace::Parameter::GeometryVariant::SubdividedCubeoid;
    auto & equation_data = test_params.equation_data;
    // equation_data.lambda = 5.678;
    equation_data.mu = 10.; // 1.234;
    // test_params.patch_variant = TPSS::PatchVariant::vertex;
    // test_params.test_diagonal_blocks = false;
    Test<2, 1> test{test_params};
  }

  // for(unsigned n_refinements = 0; n_refinements < 3; ++n_refinements)
  // {
  //   TestParameter test_params;
  //   test_params.n_refinements = n_refinements;
  //   auto & equation_data  = test_params.equation_data;
  //   equation_data.lambda  = 123.4;
  //   equation_data.mu      = 5.678;
  //   Test<2, 1>{test_params};
  //   Test<2, 2>{test_params};
  //   Test<2, 5>{test_params};
  // }

  return 0;
}
