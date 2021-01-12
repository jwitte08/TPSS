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
#include "alignedlinalg.h"
#include "generic_functionalities.h"
#include "tensor_indices.h"
#include "vectorization.h"

#include <sstream>

using namespace dealii;



namespace Tensors
{
/**
 * Generates a tensor of zero matrices with order @p order. The number of rows and
 * columns are defined by @p rows and @p columns for each tensor direction.
 */
template<int order, typename Number, typename IntType = std::size_t>
std::array<Table<2, Number>, order>
make_zero_tensor(const std::array<IntType, order> rows, const std::array<IntType, order> columns)
{
  std::array<Table<2, Number>, order> tensor;
  for(auto d = 0U; d < order; ++d)
    tensor[d].reinit(rows[d], columns[d]);
  return tensor;
}



/**
 * Generates a vector of rank-1 tensors of zero matrices with order @p
 * order. Matrices are sized according to @p rows and @p columns (see
 * make_zero_tensor() for more details).
 */
template<int order, typename Number, typename IntType = std::size_t>
std::vector<std::array<Table<2, Number>, order>>
make_zero_rank1_tensors(const std::size_t                rank,
                        const std::array<IntType, order> rows,
                        const std::array<IntType, order> columns)
{
  std::vector<std::array<Table<2, Number>, order>> rank1_tensors;
  std::fill_n(std::back_inserter(rank1_tensors),
              rank,
              make_zero_tensor<order, Number, IntType>(rows, columns));
  return rank1_tensors;
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
template<typename MatrixType, typename Number = typename MatrixType::value_type>
Table<2, Number>
transpose_matrix_to_table(const MatrixType & matrix)
{
  Table<2, Number>      table(matrix.n(), matrix.m());
  AlignedVector<Number> e_j(matrix.m());
  AlignedVector<Number> col_j(matrix.n());
  for(unsigned int j = 0; j < matrix.m(); ++j)
  {
    e_j.fill(static_cast<Number>(0.));
    col_j.fill(static_cast<Number>(0.));
    e_j[j]                = static_cast<Number>(1.);
    const auto e_j_view   = make_array_view<const Number>(e_j.begin(), e_j.end());
    const auto col_j_view = make_array_view<Number>(col_j.begin(), col_j.end());
    matrix.Tvmult(col_j_view, e_j_view);
    for(unsigned int i = 0; i < matrix.n(); ++i)
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

template<int order, typename Number>
std::vector<std::array<Table<2, Number>, order>>
product(const std::vector<std::array<Table<2, Number>, order>> & tensors1,
        const std::vector<std::array<Table<2, Number>, order>> & tensors2)
{
  std::vector<std::array<Table<2, Number>, order>> prod_of_tensors(tensors1.size() *
                                                                   tensors2.size());
  for(auto i2 = 0U; i2 < tensors2.size(); ++i2)
    for(auto i1 = 0U; i1 < tensors1.size(); ++i1)
      std::transform(tensors1[i1].cbegin(),
                     tensors1[i1].cend(),
                     tensors2[i2].cbegin(),
                     prod_of_tensors[i1 + tensors1.size() * i2].begin(),
                     [](const auto & A, const auto & B) {
                       Assert(A.n_rows() > 0, ExcMessage("Empty."));
                       Assert(B.n_cols() > 0, ExcMessage("Empty."));
                       auto C = matrix_multiplication(A, B);
                       AssertDimension(C.n_rows(), A.n_rows());
                       AssertDimension(C.n_cols(), B.n_cols());
                       return C;
                     });
  return prod_of_tensors;
}

template<int order, typename Number>
std::vector<std::array<Table<2, Number>, order>>
Tproduct(const std::vector<std::array<Table<2, Number>, order>> & tensors1,
         const std::vector<std::array<Table<2, Number>, order>> & tensors2)
{
  std::vector<std::array<Table<2, Number>, order>> prod_of_tensors(tensors1.size() *
                                                                   tensors2.size());
  for(auto i2 = 0U; i2 < tensors2.size(); ++i2)
    for(auto i1 = 0U; i1 < tensors1.size(); ++i1)
      std::transform(tensors1[i1].cbegin(),
                     tensors1[i1].cend(),
                     tensors2[i2].cbegin(),
                     prod_of_tensors[i1 + tensors1.size() * i2].begin(),
                     [](const auto & A, const auto & B) {
                       Assert(A.n_rows() > 0, ExcMessage("Empty."));
                       Assert(B.n_cols() > 0, ExcMessage("Empty."));
                       auto C = matrix_transpose_multiplication(A, B);
                       AssertDimension(C.n_rows(), A.n_rows());
                       AssertDimension(C.n_cols(), B.n_cols());
                       return C;
                     });
  return prod_of_tensors;
}

template<int order, typename Number>
std::vector<std::array<Table<2, Number>, order>>
productT(const std::vector<std::array<Table<2, Number>, order>> & tensors1,
         const std::vector<std::array<Table<2, Number>, order>> & tensors2)
{
  std::vector<std::array<Table<2, Number>, order>> prod_of_tensors(tensors1.size() *
                                                                   tensors2.size());
  for(auto i2 = 0U; i2 < tensors2.size(); ++i2)
    for(auto i1 = 0U; i1 < tensors1.size(); ++i1)
      std::transform(tensors1[i1].cbegin(),
                     tensors1[i1].cend(),
                     tensors2[i2].cbegin(),
                     prod_of_tensors[i1 + tensors1.size() * i2].begin(),
                     [](const auto & A, const auto & B) {
                       Assert(A.n_rows() > 0, ExcMessage("Empty."));
                       Assert(B.n_cols() > 0, ExcMessage("Empty."));
                       auto C = matrix_multiplication_transpose(A, B);
                       AssertDimension(C.n_rows(), A.n_rows());
                       AssertDimension(C.n_cols(), B.n_cols());
                       return C;
                     });
  return prod_of_tensors;
}

template<int order, typename Number>
std::vector<std::array<Table<2, Number>, order>>
scale(const Number & factor, const std::vector<std::array<Table<2, Number>, order>> & tensors)
{
  std::vector<std::array<Table<2, Number>, order>> scaled_tensors(tensors.size());
  for(auto i = 0U; i < tensors.size(); ++i)
  {
    auto &       scaled_tensor = scaled_tensors[i];
    const auto & tensor        = tensors[i];
    scaled_tensor.front()      = factor * tensor.front();
    for(auto d = 1U; d < order; ++d)
      scaled_tensor[d] = tensor[d];
  }
  return scaled_tensors;
}

template<typename Number>
bool
is_nearly_zero(const Table<2, Number> & matrix)
{
  const auto           m     = matrix.size(0);
  const auto           n     = matrix.size(1);
  const Number * const begin = &(matrix(0, 0));
  const Number * const end   = std::next(&(matrix(m - 1, n - 1)));
  return std::all_of(begin, end, has_nearly_zero_abs<Number>);
}

template<int N, typename Number>
bool
is_nearly_zero(const std::array<Table<2, Number>, N> & array)
{
  return std::all_of(array.cbegin(), array.cend(), is_nearly_zero<Number>);
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
 * Matrix multiplication between left-hand matrix @p lhs and right-hand matrix
 * @p rhs.
 */
template<typename Number>
Table<2, Number>
mmult(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return matrix_multiplication(lhs, rhs);
}



/**
 * Matrix multiplication between the transpose of left-hand matrix @p lhs and
 * right-hand matrix @p rhs.
 */
template<typename Number>
Table<2, Number>
Tmmult(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return matrix_transpose_multiplication(lhs, rhs);
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
  auto && matrix_src = std::forward<MatrixTypeIn1>(matrix_in);

  const unsigned int n_rows_src = matrix_src.n_rows();
  const unsigned int n_cols_src = matrix_src.n_cols();

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_cols_src, n_rows_src);

  for(unsigned int i = 0; i < n_rows_src; ++i)
    for(unsigned int j = 0; j < n_cols_src; ++j)
      matrix_out(j, i) = matrix_src(i, j);

  return matrix_out;
}

template<int order, typename Number>
// std::array<Table<2, VectorizedArray<Number>>, order>
void
transpose_tensor(std::array<Table<2, VectorizedArray<Number>>, order> & tensor)
{
  for(auto & matrix : tensor)
    matrix = transpose(matrix);
}

// template<int order, typename Number>
// std::vector<std::array<Table<2, Number>, order>> elementary_tensors;
// assemble_elementary_tensors(const Table<2, Number> * mass, const Table<2, Number> * driv)
// {
//   std::vector<std::array<Table<2, Number>, order>> elementary_tensors(order);

// }

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
