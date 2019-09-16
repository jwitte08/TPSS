/*
 * tensors.h
 *
 *  Created on: Dec 06, 2018
 *      Author: witte
 */

#ifndef TENSORS_H_
#define TENSORS_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/tensor_product_matrix.h>

#include "TPSS.h"

#include <sstream>

using namespace dealii;

namespace Tensors
{
template<int order>
std::string
multiindex_to_string(const std::array<unsigned int, order> multiindex)
{
  std::ostringstream osstream;
  osstream << "(";
  static_assert(order > 0, "0-order multiindex is not permitted.");
  for(unsigned int k = 0; k < multiindex.size() - 1; ++k)
    osstream << multiindex[k] << ", ";
  osstream << multiindex.back() << ")";
  return osstream.str();
}

/*
 * transforms an (isotropic) multi-index into the canonical
 * uni-index with respect to lexicographical order.
 *
 * order : the order of the multi-index
 * size  : isotropic size of each index set (mode)
 */
template<int order>
unsigned int
multi_to_uniindex(const std::array<unsigned int, order> multiindex, const unsigned int size)
{
  for(unsigned int k = 0; k < multiindex.size(); ++k)
    AssertIndexRange(multiindex[k], size);
  unsigned int uniindex{0};
  for(unsigned int k = 0; k < order; ++k)
    uniindex += std::pow(size, k) * multiindex[k];

  AssertIndexRange(uniindex, Utilities::pow(size, order));
  return uniindex;
}

/*
 * transforms an uni-index into the canonical (isotropic)
 * multi-index with respect to lexicographical order.
 *
 * order : the order of the multi-index
 * size  : isotropic size of each index set (mode)
 */
template<int order>
std::array<unsigned int, order>
uni_to_multiindex(unsigned int index, const unsigned int size)
{
  AssertIndexRange(index, Utilities::pow(size, order));
  std::array<unsigned int, order> multiindex;
  for(int k = order - 1; k >= 0; --k)
  {
    const unsigned int sizek = std::pow(size, k);
    multiindex[k]            = index / sizek;
    index                    = index % sizek;
  }
  Assert(index == 0, ExcMessage("Uni-index has remainder after multi-index extraction."));

  for(unsigned int k = 0; k < multiindex.size(); ++k)
    AssertIndexRange(multiindex[k], size);
  return multiindex;
}

/*
 * returns the fibre of (uni)-indices of an @p order -order
 * (isotropic) multiindex running along the @p mode mode with
 * respect to lexicographical order.
 *
 * order : the order of the multi-index
 * mode  : traversing mode of the fibre
 * size  : isotropic size of each index set (mode)
 */
template<int order>
std::vector<unsigned int>
index_fibre(const std::array<unsigned int, order - 1> index,
            const int                                 mode,
            const unsigned int                        size)
{
  AssertIndexRange(mode, order);
  for(unsigned int k = 0; k < index.size(); ++k)
    AssertIndexRange(index[k], size);
  std::vector<unsigned int>       fibre;
  std::array<unsigned int, order> multiindex;
  // std::cout << Tensors::multiindex_to_string<order-1>(index) << std::endl;
  auto elem = index.cbegin();
  std::generate(multiindex.begin(), multiindex.end(), [&, mode, k = int{0}]() mutable {
    return (k++ != mode) ? *(elem++) : 0U;
  });
  for(unsigned int i = 0; i < size; ++i)
  {
    multiindex[mode] = i;
    // std::cout << Tensors::multiindex_to_string<order>(multiindex) << std::endl;
    fibre.push_back(multi_to_uniindex<order>(multiindex, size));
  }
  return fibre;
}

template<int dim, typename Number, int size = -1>
class BlockMatrixDiagonal
{
public:
  using matrix_type = TensorProductMatrixSymmetricSum<dim, Number, size>;
  using value_type  = typename matrix_type::value_type;

  /**
   * Deletes old and resizes to new block structure.
   */
  void
  resize(const std::size_t n_blocks)
  {
    blocks.clear();
    blocks.resize(n_blocks);
  }

  matrix_type &
  get_block(const std::size_t block_index)
  {
    AssertIndexRange(block_index, blocks.size());
    return blocks[block_index];
  }

  const matrix_type &
  get_block(const std::size_t block_index) const
  {
    AssertIndexRange(block_index, blocks.size());
    return blocks[block_index];
  }

  unsigned int
  m() const
  {
    const unsigned int n_rows =
      std::accumulate(blocks.begin(), blocks.end(), 0, [](const auto sum, const auto & matrix) {
        return sum + matrix.m();
      });
    return n_rows;
  }

  unsigned int
  n() const
  {
    const unsigned int n_columns =
      std::accumulate(blocks.begin(), blocks.end(), 0, [](const auto sum, const auto & matrix) {
        return sum + matrix.m();
      });
    return n_columns;
  }

  void
  vmult(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             matrix,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { matrix.vmult(dst, src); },
                     dst,
                     src);
  }

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             matrix,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { matrix.apply_inverse(dst, src); },
                     dst,
                     src);
  }

private:
  /**
   * Applies the function @p action to each matrix of the block
   * diagonal and the associated vector slices of @p dst and @p src.
   * The signature of the action function should be equivalent to
   * action(matrix_type& m, ArrayView<...> dst, ArrayView<const ...> src)
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const
  {
    AssertDimension(src.size(), n());
    AssertDimension(dst.size(), m());
    AssertThrow(src.size() == dst.size(), ExcNotImplemented()); // quadratic !
    AssertThrow(size == -1 || src.size() % size == 0,
                ExcMessage("Input vector sizes are not a multiple of the static size."));
    AssertThrow(!blocks.empty(), ExcMessage("Blocks are not initialized."));

    // *** apply sliced vectors to the associated block matrix
    std::size_t row_end = 0;
    std::size_t col_end = 0;
    for(const auto & matrix : blocks)
    {
      const auto row_start = row_end;
      const auto col_start = col_end;
      row_end += matrix.m();
      col_end += matrix.n();
      const ArrayView<Number>       dst_block(dst.begin() + row_start, matrix.m());
      const ArrayView<const Number> src_block(src.begin() + col_start, matrix.n());
      action(matrix, dst_block, src_block);
    }
    AssertThrow(row_end == dst.size() && col_end == src.size(),
                ExcMessage("Inconsistent slicing."));
  }

  /**
   * The vector containing the matrix blocks.
   */
  AlignedVector<matrix_type> blocks;
};

/**
 * Compute the Kronecker product of two matrices. Each input
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeIn2, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
kronecker_product(MatrixTypeIn1 && left_matrix, MatrixTypeIn2 && right_matrix)
{
  auto && matrix1 = std::forward<MatrixTypeIn1>(left_matrix);
  auto && matrix0 = std::forward<MatrixTypeIn2>(right_matrix);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();
  const unsigned int n_rows1 = matrix1.n_rows();
  const unsigned int n_cols1 = matrix1.n_cols();
  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows1 * n_rows0, n_cols1 * n_cols0);

  for(unsigned int i1 = 0; i1 < n_rows1; ++i1)
    for(unsigned int j1 = 0; j1 < n_cols1; ++j1)
      for(unsigned int i0 = 0; i0 < n_rows0; ++i0)
        for(unsigned int j0 = 0; j0 < n_cols0; ++j0)
          matrix_out(i1 * n_rows0 + i0, j1 * n_cols0 + j0) = matrix1(i1, j1) * matrix0(i0, j0);

  return matrix_out;
}

/**
 * Computes the sum of two equally sized matrices. Each input
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeIn2, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
sum(MatrixTypeIn1 && left_matrix, MatrixTypeIn2 && right_matrix)
{
  using namespace dealii;
  auto && matrix1 = std::forward<MatrixTypeIn1>(left_matrix);
  auto && matrix0 = std::forward<MatrixTypeIn2>(right_matrix);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();
#ifdef DEBUG
  const unsigned int n_rows1 = matrix1.n_rows();
  const unsigned int n_cols1 = matrix1.n_cols();
  AssertDimension(n_rows0, n_rows1);
  AssertDimension(n_cols0, n_cols1);
#endif

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows0, n_cols0);

  for(unsigned int i = 0; i < n_rows0; ++i)
    for(unsigned int j = 0; j < n_cols0; ++j)
      matrix_out(i, j) = matrix1(i, j) + matrix0(i, j);

  return matrix_out;
}

/**
 * Returns the transpose of the input matrix @p matrix_in
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
// scale(MatrixTypeIn1 && matrix_in,const typename ExtractScalarType<typename
// MatrixTypeIn1::value_type>::type factor)
scale(const double factor, MatrixTypeIn1 && matrix_in) // TODO
{
  auto && matrix0 = std::forward<MatrixTypeIn1>(matrix_in);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows0, n_cols0);

  for(unsigned int i = 0; i < n_rows0; ++i)
    for(unsigned int j = 0; j < n_cols0; ++j)
      matrix_out(i, j) = factor * matrix0(i, j);

  return matrix_out;
}

/**
 * Returns the transpose of the input matrix @p matrix_in
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
transpose(MatrixTypeIn1 && matrix_in)
{
  auto && matrix0 = std::forward<MatrixTypeIn1>(matrix_in);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows0, n_cols0);

  for(unsigned int i = 0; i < n_rows0; ++i)
    for(unsigned int j = 0; j < n_cols0; ++j)
      matrix_out(i, j) = matrix0(j, i);

  return matrix_out;
}

/**
 * Assembles the separable Kronecker product form of a collection of
 * "mass matrices" @p mass_matrix and "derivative matrices" @p
 * derivative_matrix. For a detailed description see the
 * documentation on deal.II's class TensorProductMatrixSymmetricSum.
 */
template<std::size_t dim, typename Number>
Table<2, Number>
assemble_separableKD(const std::array<Table<2, Number>, dim> & mass_matrix,
                     const std::array<Table<2, Number>, dim> & derivative_matrix)
{
  using namespace dealii;
  Table<2, Number> tpmatrix;

  if(dim == 1)
    tpmatrix = derivative_matrix[0];

  else if(dim == 2)
  {
    auto && MxA{kronecker_product(mass_matrix[1], derivative_matrix[0])};
    auto && AxM{kronecker_product(derivative_matrix[1], mass_matrix[0])};
    tpmatrix = std::move(sum(MxA, AxM));
  }

  else if(dim == 3)
  {
    auto && MxA         = kronecker_product(mass_matrix[1], derivative_matrix[0]);
    auto && AxM         = kronecker_product(derivative_matrix[1], mass_matrix[0]);
    auto && MxAxM_MxMxA = kronecker_product(mass_matrix[2], sum(AxM, MxA));
    auto && AxMxM =
      kronecker_product(derivative_matrix[2], kronecker_product(mass_matrix[1], mass_matrix[0]));
    tpmatrix = std::move(sum(AxMxM, MxAxM_MxMxA));
  }

  else
    Assert(false, ExcNotImplemented());

  return tpmatrix;
}

/**
 * Extracts and converts the matrix associated to the lane @p lane
 * of the vectorized matrix @p table into the FullMatrix format.
 */
template<typename Number>
FullMatrix<Number>
vectorized_table_to_fullmatrix(const Table<2, VectorizedArray<Number>> & table,
                               const unsigned int                        lane = 0)
{
  AssertIndexRange(lane, VectorizedArray<Number>::n_array_elements);
  FullMatrix<Number> matrix{table.n_rows(), table.n_cols()};
  for(unsigned int i = 0; i < table.n_rows(); ++i)
    for(unsigned int j = 0; j < table.n_cols(); ++j)
      matrix(i, j) = (table(i, j))[lane];
  return matrix;
}

} // namespace Tensors

#endif // TENSORS_H_
