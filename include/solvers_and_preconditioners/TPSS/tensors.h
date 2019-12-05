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
#include "generic_functionalities.h"

#include <sstream>

using namespace dealii;

namespace Tensors
{
template<int order, typename IntType = unsigned int>
std::string
multiindex_to_string(const std::array<IntType, order> multiindex)
{
  std::ostringstream osstream;
  osstream << "(";
  static_assert(order > 0, "0-order multiindex is not permitted.");
  for(IntType k = 0; k < multiindex.size() - 1; ++k)
    osstream << multiindex[k] << ", ";
  osstream << multiindex.back() << ")";
  return osstream.str();
}

/*
 * transforms an (anisotropic) multi-index into the canonical
 * uni-index with respect to lexicographical order.
 *
 * order : the order of the multi-index
 * sizes  : (anisotropic) size of each independent variable (mode)
 */
template<int order, typename IntType = unsigned int>
IntType
multi_to_uniindex(const std::array<IntType, order> & multiindex,
                  const std::array<IntType, order> & sizes)
{
  for(IntType k = 0; k < multiindex.size(); ++k)
    AssertIndexRange(multiindex[k], sizes[k]);
  IntType uniindex{0};
  for(int k = order - 1; k >= 0; --k)
  {
    // has no effect on purpose for k == order-1 (uniindex is zero)
    uniindex *= sizes[k];
    uniindex += multiindex[k];
  }
  const auto n_elem = std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<IntType>());
  (void)n_elem;
  AssertIndexRange(uniindex, n_elem);

  return uniindex;
}

/*
 * transforms an (isotropic) multi-index into the canonical
 * uni-index with respect to lexicographical order.
 *
 * order : the order of the multi-index
 * size  : isotropic size of each index set (mode)
 */
template<int order, typename IntType = unsigned int>
IntType
multi_to_uniindex(const std::array<IntType, order> & multiindex, const IntType size)
{
  std::array<IntType, order> sizes;
  sizes.fill(size);
  return multi_to_uniindex<order>(multiindex, sizes);
}

/*
 * transforms an uni-index into the canonical (anisotropic)
 * multi-index with respect to lexicographical order.
 *
 * order : the order of the multi-index
 * sizes : sizes of each independent variable (mode)
 */
template<int order, typename IntType = unsigned int>
std::array<IntType, order>
uni_to_multiindex(IntType index, const std::array<IntType, order> & sizes)
{
  const auto n_elem = std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<IntType>());
  (void)n_elem;
  AssertIndexRange(index, n_elem);
  std::array<IntType, order> multiindex;
  for(int k = 0; k < order; ++k)
  {
    multiindex[k] = index % sizes[k];
    index         = index / sizes[k];
  }
  Assert(index == 0, ExcMessage("Uni-index has remainder after multi-index extraction."));
  for(IntType k = 0; k < multiindex.size(); ++k)
    AssertIndexRange(multiindex[k], sizes[k]);

  return multiindex;
}

/*
 * transforms an uni-index into the canonical (isotropic)
 * multi-index with respect to lexicographical order.
 *
 * order : the order of the multi-index
 * size  : isotropic size of each index set (mode)
 */
template<int order, typename IntType = unsigned int>
std::array<IntType, order>
uni_to_multiindex(IntType index, const IntType size)
{
  std::array<IntType, order> sizes;
  sizes.fill(size);
  return uni_to_multiindex<order>(index, sizes);
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
template<int order, typename IntType = unsigned int>
std::vector<IntType>
index_fibre(const std::array<IntType, order - 1> index, const int mode, const IntType size)
{
  AssertIndexRange(mode, order);
  for(IntType k = 0; k < index.size(); ++k)
    AssertIndexRange(index[k], size);
  std::vector<IntType>       fibre;
  std::array<IntType, order> multiindex;
  // std::cout << Tensors::multiindex_to_string<order-1>(index) << std::endl;
  auto elem = index.cbegin();
  std::generate(multiindex.begin(), multiindex.end(), [&, mode, k = int{0}]() mutable {
    return (k++ != mode) ? *(elem++) : 0U;
  });
  for(IntType i = 0; i < size; ++i)
  {
    multiindex[mode] = i;
    // std::cout << Tensors::multiindex_to_string<order>(multiindex) << std::endl;
    fibre.push_back(multi_to_uniindex<order>(multiindex, size));
  }
  return fibre;
}

/**
 * Converts a matrix into a two dimensional table. MatrixType has to fulfill
 * following interface:
 *
 * method m() returning number of rows
 * method n() returning number of cols
 * typedef value_type
 * method vmult(ArrayView,ArrayView)
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
Table<2, Number>
matrix_to_table(const MatrixType & matrix)
{
  Table<2, Number>      table(matrix.m(), matrix.n());
  AlignedVector<Number> e_j(matrix.n());
  AlignedVector<Number> col_j(matrix.m());
  for(unsigned int j = 0; j < matrix.n(); ++j)
  {
    e_j.fill(static_cast<Number>(0.));
    col_j.fill(static_cast<Number>(0.));
    e_j[j]                = static_cast<Number>(1.);
    const auto e_j_view   = make_array_view<const Number>(e_j.begin(), e_j.end());
    const auto col_j_view = make_array_view<Number>(col_j.begin(), col_j.end());
    matrix.vmult(col_j_view, e_j_view);
    for(unsigned int i = 0; i < matrix.m(); ++i)
      table(i, j) = col_j[i];
  }
  return table;
}

/**
 * Converts a matrix into a two dimensional table. MatrixType has to fulfill
 * following interface:
 *
 * method m() returning number of rows
 * method n() returning number of cols
 * typedef value_type
 * method vmult(ArrayView,ArrayView)
 */
template<typename Number = double>
Table<2, Number>
lapack_matrix_to_table(const LAPACKFullMatrix<Number> & matrix)
{
  Table<2, Number> table(matrix.m(), matrix.n());
  Vector<Number>   e_j(matrix.n());
  Vector<Number>   col_j(matrix.m());
  for(unsigned int j = 0; j < matrix.n(); ++j)
  {
    std::fill(e_j.begin(), e_j.end(), 0.);
    std::fill(col_j.begin(), col_j.end(), 0.);
    e_j[j] = 1.;
    matrix.vmult(col_j, e_j);
    for(unsigned int i = 0; i < matrix.m(); ++i)
      table(i, j) = col_j[i];
  }
  return table;
}

/**
 * Converts the inverse of a matrix into a two dimensional table. MatrixType has
 * to fulfill following interface:
 *
 * method m() returning number of rows
 * method n() returning number of cols
 * typedef value_type
 * method apply_inverse(ArrayView,ArrayView)
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
Table<2, Number>
inverse_matrix_to_table(const MatrixType & matrix)
{
  Table<2, Number>      table(matrix.m(), matrix.n());
  AlignedVector<Number> e_j(matrix.n());
  AlignedVector<Number> col_j(matrix.m());
  for(unsigned int j = 0; j < matrix.n(); ++j)
  {
    e_j.fill(static_cast<Number>(0.));
    col_j.fill(static_cast<Number>(0.));
    e_j[j]                = static_cast<Number>(1.);
    const auto e_j_view   = make_array_view<const Number>(e_j.begin(), e_j.end());
    const auto col_j_view = make_array_view<Number>(col_j.begin(), col_j.end());
    matrix.apply_inverse(col_j_view, e_j_view);
    for(unsigned int i = 0; i < matrix.m(); ++i)
      table(i, j) = col_j[i];
  }
  return table;
}

template<typename MatrixType1, typename MatrixType2 = MatrixType1>
void
insert_block(MatrixType1 &       dst,
             const MatrixType2 & src,
             const unsigned int  row_dst = 0,
             const unsigned int  col_dst = 0)
{
  AssertIndexRange(row_dst + src.m(), dst.m() + 1);
  AssertIndexRange(col_dst + src.n(), dst.n() + 1);
  for(unsigned i = 0; i < src.m(); ++i)
    for(unsigned j = 0; j < src.n(); ++j)
      dst(row_dst + i, col_dst + j) = src(i, j);
}

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

} // namespace Tensors

#endif // TENSORS_H_
