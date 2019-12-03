
/*
 * test functionality of laplace_integrator.h ...
 * ... fast diagonal integrator (cell patch)!
 *
 *  Created on: Aug 10, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

template<int dim, int fe_degree>
struct Test
{
  static constexpr unsigned macro_size = VectorizedArray<double>::n_array_elements;

  Test(ConditionalOStream &       pcout_in,
       const Laplace::Parameter & parameters_in,
       const bool                 print_test_details = false)
    : pcout(pcout_in),
      test_details(print_test_details ? std::cout.rdbuf() : null_stream.rdbuf()),
      parameters(parameters_in),
      laplace_problem(pcout, parameters)
  {
    laplace_problem.setup(false, false, true);
    global_level = laplace_problem.global_level;
    laplace_problem.assemble_matrix();
  }

  /*
   * Initializes a block smoother on the global level (each block
   * corresponds to one cell), by extracting blocks from the system
   * matrix. The boolean @p store_block_matrices determines if in
   * addition to the inverse block matrices the "forward" block
   * matrices are stored.
   */
  void
  assemble_smoother_cell_patch()
  {
    const auto & dof_handler   = laplace_problem.dof_handler;
    const auto & sparse_matrix = laplace_problem.sparse_matrix;

    Assert(sparse_matrix.m() == dof_handler.n_dofs(),
           ExcMessage("System matrix is not valid/initialized!"));

    // *** initialize block list defining the cell patches
    test_details << "Assembling BlockJacobi-smoother and -submatrices ...  \n\n";
    auto cpatch_additional_data = std::make_shared<typename RELAXATION_BLOCK::AdditionalData>();
    SchwarzSmootherData schwarz_smoother_data;
    schwarz_smoother_data.patch_variant    = TPSS::PatchVariant::cell;
    schwarz_smoother_data.smoother_variant = TPSS::SmootherVariant::additive;
    cpatch_additional_data->relaxation =
      TPSS::lookup_damping_factor(schwarz_smoother_data.patch_variant,
                                  schwarz_smoother_data.smoother_variant,
                                  dim);
    cpatch_additional_data->inversion = PreconditionBlockBase<double>::svd;
    DoFTools::make_cell_patches(cpatch_additional_data->block_list, dof_handler, global_level);
    if(cpatch_additional_data->block_list.n_rows() > 0)
      cpatch_additional_data->block_list.compress();
    block_smoother_data = cpatch_additional_data;

    // *** store the number of blocks
    n_blocks = block_smoother_data->block_list.n_rows();

    // *** initialize block smoother to access e.g. block matrices
    Assert(block_smoother_data != nullptr, ExcMessage("Smoother data not initialized!"));
    // NOTE we require a modified version of the RelaxationBlock class
    // --> see my branch on the matrixfree gitlab repo
    auto cpatch_smoother = std::make_shared<RELAXATION_BLOCK>(/*store_block_matrices*/ true);
    cpatch_smoother->initialize(sparse_matrix, *block_smoother_data);
    block_smoother = cpatch_smoother;

    auto block_jacobi = std::make_shared<RelaxationBlockJacobi<SparseMatrix<double>>>();
    block_jacobi->initialize(sparse_matrix, *block_smoother_data);
    block_jacobi_smoother = block_jacobi;

    test_details << " ... done!\n\n";
  }

  void
  assemble_fast_diagonal_submatrices()
  {
    using EVALUATOR = FDEvaluation<dim, fe_degree, fe_degree + 1, double>;
    // NOTE additional integration data can be transmitted by the integrator, here it is just a
    // dummy
    using INTEGRATION_DATA = typename Laplace::FD::MatrixIntegrator<dim, fe_degree, double>;

    // LAMBDA computes the 1D mass matrix in each dimension
    const auto & assembler_mass_matrices =
      [](const SubdomainHandler<dim, double> &                             data,
         std::vector<std::array<Table<2, VectorizedArray<double>>, dim>> & matrices,
         const INTEGRATION_DATA &,
         const std::pair<unsigned int, unsigned int> & subdomain_range) {
        EVALUATOR          fd_eval{data};
        constexpr unsigned fe_order = fe_degree + 1;

        Table<2, VectorizedArray<double>> cell_mass_unit{fe_order, fe_order};
        fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
        const auto cell_operation_mass = [&](const auto &                        fd_eval,
                                             Table<2, VectorizedArray<double>> & cell_matrix,
                                             const int                           direction,
                                             const int                           cell_no) {
          AssertIndexRange(direction, static_cast<int>(dim));
          EVALUATOR::CellAssembler::scale_matrix(fd_eval.get_h(direction, cell_no),
                                                 make_array_view(cell_mass_unit),
                                                 make_array_view(cell_matrix));
        };

        for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
        {
          fd_eval.reinit(id);
          fd_eval.evaluate(true);

          std::array<Table<2, VectorizedArray<double>>, dim> && mass_matrices =
            fd_eval.patch_action(cell_operation_mass);

          matrices[id] = std::move(mass_matrices);
        }
      };

    // LAMBDA computes the 1D laplace matrix in each dimension
    const auto & assembler_laplace_matrices =
      [](const SubdomainHandler<dim, double> &                             data,
         std::vector<std::array<Table<2, VectorizedArray<double>>, dim>> & matrices,
         const INTEGRATION_DATA &,
         const std::pair<unsigned int, unsigned int> & subdomain_range) {
        EVALUATOR fd_eval{data};

        using INTEGRATOR  = typename Laplace::FD::MatrixIntegrator<dim, fe_degree, double>;
        using CELL_ACTION = typename INTEGRATOR::template CellLaplace<EVALUATOR>;
        using FACE_ACTION = typename INTEGRATOR::template FaceLaplace<EVALUATOR>;
        for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
        {
          fd_eval.reinit(id);
          fd_eval.evaluate(true);

          std::array<Table<2, VectorizedArray<double>>, dim> && laplace_matrices =
            fd_eval.patch_action(CELL_ACTION{}, FACE_ACTION{}, /*dummy*/ FACE_ACTION{});

          matrices[id] = std::move(laplace_matrices);
        }
      };

    // *** assemble the basic matrices, i.e. mass matrices
    test_details << "Assembling basic and action matrices (fast diagonal submatrices) ... \n\n";
    const auto mf_storage_global = laplace_problem.template build_mf_storage<double>(global_level);
    const auto patch_storage =
      laplace_problem.template build_patch_storage<double>(global_level, mf_storage_global);
    const auto & partition_data = patch_storage->get_patch_info().subdomain_partition_data;
    basic_matrices.clear();
    basic_matrices.resize(partition_data.n_subdomains());
    for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    {
      patch_storage->template loop<INTEGRATION_DATA, std::vector<MATRIX_TENSOR>>(
        assembler_mass_matrices, basic_matrices, INTEGRATION_DATA{}, color);
    }

    // *** assemble the action matrices, i.e. the Laplace DG formulation
    action_matrices.clear();
    action_matrices.resize(partition_data.n_subdomains());
    for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    {
      patch_storage->template loop<INTEGRATION_DATA, std::vector<MATRIX_TENSOR>>(
        assembler_laplace_matrices, action_matrices, INTEGRATION_DATA{}, color);
    }

    AssertDimension(basic_matrices.size(), action_matrices.size());
    test_details << " ... done!\n\n";
  }

  void
  operator()()
  {
    // LAMBDA prints the block matrices assembled by laplace_problem
    const auto print_block_matrix = [this](const unsigned block_id) {
      print_row(test_details, 12, "\n block_id:", block_id, "\n");
      const auto & block_matrix = block_smoother->get_block_matrix(block_id);
      block_matrix.print_formatted(test_details);
    };

    // LAMBDA computes the separable Kronecker Decomposition and prints subspace matrices
    const auto print_fast_diagonal_matrix = [this](const unsigned patch_id) {
      const auto fdmatrix =
        Tensors::assemble_separableKD(basic_matrices[patch_id], action_matrices[patch_id]);
      for(unsigned lane = 0; lane < macro_size; ++lane)
      {
        print_row(test_details, 12, "\n patch_id:", patch_id, "lane:", lane, "\n");
        const auto mat = table_to_fullmatrix(fdmatrix, lane);
        mat.print_formatted(test_details);
      }
    };

    assemble_smoother_cell_patch();
    test_details << "Printing the block matrices (cell patch) ... \n\n";
    for(unsigned id = 0; id < n_blocks; ++id)
      print_block_matrix(id);
    test_details << std::endl;

    assemble_fast_diagonal_submatrices();
    test_details << "Printing the fast diagonal subspace matrices (cell patch) ... \n\n";
    for(unsigned id = 0; id < basic_matrices.size(); ++id)
      print_fast_diagonal_matrix(id);
    test_details << std::endl;

    pcout << "Comparing BlockJacobi and Schwarz smoother ... \n\n";
    auto     dst{laplace_problem.solution};
    unsigned i = 0;
    for(auto it = dst.begin(); it != dst.end(); ++it, ++i)
      *it = (double)rand() / RAND_MAX; //(double)i;

    // *** fill src and dst vectors
    const auto           src{dst};
    auto                 dst2{dst};
    const Vector<double> src_copy(dst.begin(), dst.end());
    Vector<double>       dst_copy(dst.begin(), dst.end());

    // *** block Jacobi smoother
    block_jacobi_smoother->step(dst_copy, src_copy);
    print_row(test_details, 20, "dst:", "BlockJacobiRelaxation\n");
    dst_copy.print(test_details);
    test_details << std::endl;

    // *** Schwarz smoother
    const auto & mg_schwarz_smoother = laplace_problem.mg_smoother;
    mg_schwarz_smoother.smooth(global_level, dst2, src);
    print_row(test_details, 20, "dst:", "SchwarzSmoother\n");
    dst2.print(test_details);
    test_details << std::endl;

    // *** Schwarz smoother (MGSmootherPrecondition + SchwarzPreconditioner)
    const auto & mg_schwarz_smoother_precondition = laplace_problem.mg_smoother_precondition;
    mg_schwarz_smoother_precondition.smooth(global_level, dst, src);
    print_row(test_details, 20, "dst:", "Schwarz smoother (MGSmootherPrecondition)\n");
    dst.print(test_details);
    test_details << std::endl;

    // *** check resulting vectors
    Vector<double> temp(dst.begin(), dst.end());
    temp -= dst_copy;
    Assert(std::abs(temp.l2_norm()) < (1.e-12 * dst_copy.l2_norm()),
           ExcMessage(
             "BlockJacobiRelaxation and SchwarzSmoother(MGSmootherPrecondition) don't match!"));
    Vector<double> temp2(dst2.begin(), dst2.end());
    temp2 -= dst_copy;
    Assert(std::abs(temp2.l2_norm()) < (1.e-12 * dst_copy.l2_norm()),
           ExcMessage("BlockJacobiRelaxation and Schwarz smoother don't match!"));

    pcout << " ... success!\n\n";
  }

  ConditionalOStream &                            pcout;
  NullOStream                                     null_stream;
  std::ostream                                    test_details;
  const Laplace::Parameter                        parameters;
  Laplace::MatrixOperator<dim, fe_degree, double> laplace_problem;
  unsigned                                        global_level = 0;

  using RELAXATION_BLOCK = RelaxationBlock<SparseMatrix<double>, double, Vector<double>>;
  std::shared_ptr<const typename RELAXATION_BLOCK::AdditionalData>   block_smoother_data;
  std::shared_ptr<const RELAXATION_BLOCK>                            block_smoother;
  std::shared_ptr<const RelaxationBlockJacobi<SparseMatrix<double>>> block_jacobi_smoother;
  unsigned                                                           n_blocks = 0;

  using MATRIX_TENSOR = typename std::array<Table<2, VectorizedArray<double>>, dim>;
  std::vector<MATRIX_TENSOR> basic_matrices;
  std::vector<MATRIX_TENSOR> action_matrices;
};

int
main(int argc, char * argv[])
{
  using namespace dealii;

  // *** init MPI (1 proc only!)
  constexpr int                    max_threads = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // *** test parameters
  Laplace::Parameter parameters;
  parameters.n_refines = 1;

  // *** run tests
  parameters.n_refines = 1;
  Test<2, 1>{pcout, parameters, /*print_test_details?*/ true}();

  parameters.n_refines = 2;
  Test<2, 4>{pcout, parameters}();

  parameters.n_refines = 1;
  Test<3, 1>{pcout, parameters, /*print_test_details?*/ true}();

  parameters.n_refines = 2;
  Test<3, 2>{pcout, parameters}();

  return 0;
}
