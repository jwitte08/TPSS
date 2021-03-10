/*
 * matrix_utilities.h
 *
 *  Created on: Mar 10, 2021
 *      Author: witte
 */

#ifndef MATRIX_UTILITIES_H_
#define MATRIX_UTILITIES_H_



#include <deal.II/base/aligned_vector.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>


#include "block_matrix.h"
#include "matrix_helper.h"



namespace Util
{
using namespace dealii;

/**
 * A square submatrix restricted to locally relevant dof indices of a parallel
 * square matrix @p matrix is returned. Locally relevant indices are identified
 * by the MPI partitioner @p partitioner. Indices are ordered: locally owned
 * first, then all ghost indices. For each global dof index j the matrix is
 * multiplied with canonical basis vector e_j, ghost values are communicated and
 * the respective local column of the submatrix is filled. This process is cost
 * intensive, in particular the submatrix returned is not compressed, thus this
 * function is only recommended for small matrices @p matrix.
 */
FullMatrix<double>
extract_locally_relevant_matrix(
  const TrilinosWrappers::SparseMatrix &                   matrix,
  const std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_row,
  const std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_column)
{
  AssertIndexRange(matrix.m(), 1000); // small matrices!
  AssertIndexRange(matrix.n(), 1000); // small matrices!

  const auto locally_relevant_row_size =
    partitioner_row->local_size() + partitioner_row->n_ghost_indices();
  const auto locally_relevant_column_size =
    partitioner_column->local_size() + partitioner_column->n_ghost_indices();

  FullMatrix<double> submatrix(locally_relevant_row_size, locally_relevant_column_size);

  LinearAlgebra::distributed::Vector<double> e_j(partitioner_column);
  LinearAlgebra::distributed::Vector<double> dst(partitioner_row);

  AssertDimension(matrix.m(), dst.size());
  AssertDimension(matrix.n(), e_j.size());

  /// DEBUG
  // const bool mpi_rank =
  // Utilities::MPI::this_mpi_process(partitioner_row->get_mpi_communicator()); std::ostringstream
  // oss; oss << get_filename(partitioner_row->get_mpi_communicator()) << "." << mpi_rank;
  // std::ofstream                       ofs;
  // ofs.open(oss.str(), std::ios_base::out);

  for(types::global_dof_index j = 0; j < e_j.size(); ++j)
  {
    e_j = 0.;
    dst = 0.;

    if(partitioner_column->in_local_range(j))
      e_j.local_element(partitioner_column->global_to_local(j)) = 1.;

    // e_j.update_ghost_values();
    dst.zero_out_ghosts();

    /// DEBUG
    // {
    //   std::ostringstream oss;
    //   oss << "e_" << j << " : " << std::endl;
    //   e_j.print(oss);
    //   ofs << oss.str();
    // }

    matrix.vmult(dst, e_j);

    /// DEBUG
    // const bool j_is_ghost = partitioner_column->is_ghost_entry(j) ? 1 : 0;
    // const bool j_is_ghost_on_any_proc =
    //   Utilities::MPI::max<int>(j_is_ghost, partitioner_column->get_mpi_communicator());
    // {
    //   std::ostringstream oss;
    //   oss << "dst = A e_" << j << " : " << std::endl;
    //   dst.print(oss);
    //   ofs << oss.str();
    // }

    dst.compress(VectorOperation::add);
    dst.update_ghost_values();

    /// DEBUG
    // {
    //   std::ostringstream oss;
    //   oss << "dst = A e_" << j << " : " << std::endl;
    //   dst.print(oss);
    //   ofs << oss.str();
    // }

    if(partitioner_column->in_local_range(j) || partitioner_column->is_ghost_entry(j))
    {
      const auto jj = partitioner_column->global_to_local(j);
      for(auto ii = 0U; ii < locally_relevant_row_size; ++ii)
        submatrix(ii, jj) = dst.local_element(ii);
    }
  }

  return submatrix;
}



/**
 * Same as above but more convenient for square matrices.
 */
FullMatrix<double>
extract_locally_relevant_matrix(
  const TrilinosWrappers::SparseMatrix &                   matrix,
  const std::shared_ptr<const Utilities::MPI::Partitioner> partitioner)
{
  AssertDimension(matrix.m(), matrix.n());
  return extract_locally_relevant_matrix(matrix, partitioner, partitioner);
}



/**
 * Gathers a distributed block matrix on a single mpi process, all remaining mpi
 * processes return empty matrices or their locally relevant part. The chosen
 * mpi process is either 0 if force_zero_rank_master is true, otherwise it is
 * the mpi process which owns the most rows/columns.
 *
 * The block matrix returned has no sparsity, thus this function should be used
 * with caution and currently exists only for debugging purposes.
 */
template<bool force_zero_rank_master = true>
std::shared_ptr<Tensors::BlockMatrixBasic<MatrixAsTable<double>>>
gather_distributed_blockmatrix(const TrilinosWrappers::BlockSparseMatrix & distributed_blockmatrix,
                               const unsigned int level = numbers::invalid_unsigned_int)
{
  using local_matrix_type = Tensors::BlockMatrixBasic<MatrixAsTable<double>>;

  AssertDimension(distributed_blockmatrix.m(), distributed_blockmatrix.n());
  AssertDimension(distributed_blockmatrix.n_block_rows(), distributed_blockmatrix.n_block_cols());
  const auto n_blocks = distributed_blockmatrix.n_block_rows();

  const auto this_mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  (void)level;
  /// DEBUG
  std::ostringstream oss;
  oss << "debug_l" << level << "_p" << this_mpi_rank << ".txt";
  std::ofstream ofs;
  ofs.open(oss.str(), std::ios_base::out);

  std::vector<IndexSet> global_ranges;
  std::vector<IndexSet> local_ranges;
  unsigned int          n_locally_owned_indices = 0;
  for(auto b = 0U; b < n_blocks; ++b)
  {
    const auto & this_matrix = distributed_blockmatrix.block(b, b);
    AssertDimension(this_matrix.m(), this_matrix.n());
    auto & this_global_range = global_ranges.emplace_back(this_matrix.m());
    this_global_range.add_range(0, this_matrix.m());
    this_global_range.compress();
    auto & this_local_range             = local_ranges.emplace_back(this_matrix.m());
    const auto [local_begin, local_end] = this_matrix.local_range();
    this_local_range.add_range(local_begin, local_end);
    this_local_range.compress();
    n_locally_owned_indices += this_matrix.local_size();
  }

  AssertDimension(Utilities::MPI::sum(n_locally_owned_indices, MPI_COMM_WORLD),
                  distributed_blockmatrix.m());
  AssertDimension(global_ranges.size(), n_blocks);
  AssertDimension(local_ranges.size(), n_blocks);

  auto master_mpi_rank = 0U;
  if(!force_zero_rank_master)
  {
    const auto n_loi_max = Utilities::MPI::max(n_locally_owned_indices, MPI_COMM_WORLD);
    master_mpi_rank = Utilities::MPI::max(n_locally_owned_indices == n_loi_max ? this_mpi_rank : 0U,
                                          MPI_COMM_WORLD);
  }

  const bool is_master_proc = this_mpi_rank == master_mpi_rank;

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
  for(auto b = 0U; b < n_blocks; ++b)
    partitioners.emplace_back(std::make_shared<const Utilities::MPI::Partitioner>(
      local_ranges[b], is_master_proc ? global_ranges[b] : local_ranges[b], MPI_COMM_WORLD));

  const auto locally_relevant_blockmatrix = std::make_shared<local_matrix_type>();
  locally_relevant_blockmatrix->resize(n_blocks, n_blocks);

  for(auto brow = 0U; brow < n_blocks; ++brow)
  {
    const auto                partitioner_row = partitioners[brow];
    std::vector<unsigned int> this_g2l_row;
    if(is_master_proc)
      for(types::global_dof_index dof = 0; dof < partitioner_row->size(); ++dof)
        this_g2l_row.emplace_back(partitioner_row->global_to_local(dof));

    /// DEBUG
    ofs << "g2l_row: " << vector_to_string(this_g2l_row) << std::endl;

    for(auto bcol = 0U; bcol < n_blocks; ++bcol)
    {
      const auto partitioner_col = partitioners[bcol];

      std::vector<unsigned int> this_g2l_col;
      if(is_master_proc)
        for(types::global_dof_index dof = 0; dof < partitioner_col->size(); ++dof)
          this_g2l_col.emplace_back(partitioner_col->global_to_local(dof));

      /// DEBUG
      ofs << "g2l_col: " << vector_to_string(this_g2l_col) << std::endl;

      /// this is a collective operation thus all processes need to call this method
      const auto & locally_relevant_matrix =
        Util::extract_locally_relevant_matrix(distributed_blockmatrix.block(brow, bcol),
                                              partitioner_row,
                                              partitioner_col);

      /// DEBUG
      ofs << "block: " << brow << "-" << bcol << std::endl;
      ofs << "size: " << locally_relevant_matrix.m() << "x" << locally_relevant_matrix.n()
          << std::endl;
      locally_relevant_matrix.print_formatted(ofs);
      ofs << std::endl;

      if(is_master_proc)
      {
        FullMatrix<double> tmp(locally_relevant_matrix.m(), locally_relevant_matrix.n());
        tmp.extract_submatrix_from(locally_relevant_matrix, this_g2l_row, this_g2l_col);
        locally_relevant_blockmatrix->get_block(brow, bcol) = tmp;
      }
    }
  }

  return locally_relevant_blockmatrix;
}

} // namespace Util

#endif // MATRIX_UTILITIES_H_
