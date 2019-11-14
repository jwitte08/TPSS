
/*
 * Linear Elasticity (Test MatrixFree Integrators)
 *
 *  Created on: Jun 27, 2019
 *      Author: witte
 */

#include "Plaplace_fdss.h"
#include "ct_parameter.h"
#include "linelasticity_problem.h"

using namespace dealii;
using namespace LinElasticity;



struct TestParameter
{
  unsigned int n_refines     = 0;
  bool         print_details = false;
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const TestParameter test_parameters)
{
  using LinElasticityOperator = typename LinElasticity::MatrixOperator<dim, fe_degree, value_type>;

  const bool         is_mpi_process0 = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  const bool         print_details   = is_mpi_process0 && test_parameters.print_details;
  ConditionalOStream pcout(std::cout, print_details);
  Laplace::Parameter parameters;
  parameters.allow_one_level_only = true;
  parameters.n_refines            = test_parameters.n_refines;
  parameters.geometry_variant     = Laplace::Parameter::GeometryVariant::Cube;

  LinElasticityOperator linelasticity_problem(pcout, parameters);
  // linelasticity_problem.create_triangulation_finest();
  linelasticity_problem.distribute_dofs();
  linelasticity_problem.prepare_system(false, /*compute_rhs?*/ false);
  linelasticity_problem.assemble_matrix();

  const auto & sparse_matrix = linelasticity_problem.sparse_matrix;
  const auto & mf_operator   = linelasticity_problem.system_matrix;
  if(pcout.is_active())
  {
    pcout << "SPARSE MATRIX\n\n";
    sparse_matrix.print_formatted(std::cout);
    pcout << std::endl;
  }

  const auto     n_cells          = linelasticity_problem.triangulation.n_active_cells();
  const unsigned n_blocks         = dim;
  const unsigned n_components     = dim;
  const auto     n_dofs           = linelasticity_problem.system_u.size();
  const auto     n_dofs_per_block = linelasticity_problem.system_u.block(0).size();
  AssertDimension(1, linelasticity_problem.fe->n_base_elements());
  const auto n_dofs_per_cell_scalar = linelasticity_problem.fe->n_dofs_per_cell();
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
    auto temp{linelasticity_problem.system_u};
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
    mf_matrix.print_formatted(std::cout);
    pcout << std::endl;
  }

  { // TEST: comparison of vmults
    auto     temp{linelasticity_problem.system_u};
    unsigned i = 0;
    for(auto it = temp.begin(); it != temp.end(); ++it, ++i)
      *it = (double)rand() / RAND_MAX;
    const auto           src{temp};
    const Vector<double> src_copy(src.begin(), src.end());
    // src_copy.print(std::cout);

    // auto dst_unsorted{src};
    // mf_operator.vmult(dst_unsorted, src);
    // // dst_unsorted.print(std::cout);
    auto dst{src_copy};
    mf_matrix.vmult(dst, src_copy);
    // dst.print(std::cout);

    auto dst_sparse{src_copy};
    sparse_matrix.vmult(dst_sparse, src_copy);
    // dst_sparse.print(std::cout);

    Vector<double> diff(dst.begin(), dst.end());
    diff -= dst_sparse;
    std::ostringstream oss;
    oss << "||v_mf - v_sparse||_l2 = " << diff.l2_norm() << " exceeds bounds!";
    Assert(std::abs(diff.l2_norm()) < (1.e-12 * dst_sparse.l2_norm()), ExcMessage(oss.str()));
    std::cout << " ... success!\n\n";
  } // end TEST
}

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
    test_params.n_refines     = 1;
    // test<3, 1>(test_params);
    test<2, 1>(test_params);
  }
  { // TODO works on domains with one cell so far
    TestParameter test_params;
    // for(unsigned n = 1; n < 3; ++n)
    // {
    //   test_params.n_refines = n;
    test<2, 5>(test_params);
    test<3, 2>(test_params);
    // }
  }

  return 0;
}
