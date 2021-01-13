#ifndef ALIGNEDLINALG_H_
#define ALIGNEDLINALG_H_

#include <numeric>

#include <deal.II/base/table.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include "generic_functionalities.h"
#include "vectorization.h"

using namespace dealii;



namespace LinAlg
{
template<typename Number>
AlignedVector<Number>
sum(const AlignedVector<Number> & lhs, const AlignedVector<Number> & rhs)
{
  AssertDimension(lhs.size(), rhs.size());
  AlignedVector<Number> sum;
  std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(sum), std::plus<Number>{});
  return sum;
}



/**
 * Scales a vector @p vec by a factor @p scalar.
 */
template<typename Number>
AlignedVector<Number>
scaling(const AlignedVector<Number> & vec, const Number & scalar)
{
  AlignedVector<Number> scaled_vec(vec.size());
  std::transform(vec.begin(), vec.end(), scaled_vec.begin(), [&](const auto & elem) {
    return scalar * elem;
  });
  return scaled_vec;
}



/**
 * Computes the Euclidean inner product between a left-hand vector @p lhs and
 * right-hand vector @p rhs.
 */
template<typename Number>
Number
inner_product(const AlignedVector<Number> & lhs, const AlignedVector<Number> & rhs)
{
  AssertDimension(lhs.size(), rhs.size());
  return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), static_cast<Number>(0.));
}



/**
 * Computes the Euclidean norm of a vector @p vec.
 */
template<typename Number>
Number
euclidean_norm(const AlignedVector<Number> & vec)
{
  return std::sqrt(inner_product(vec, vec));
}



/**
 * Scales the vector @p vec by the inverse of @p scalar. For vectorized
 * arithmetic type @p Number each lane where the corresponding value of @p
 * scalar is nearly zero is scaled by zero instead of the actual inverse scalar
 * avoiding a division by zero.
 */
template<typename Number>
AlignedVector<Number>
inverse_scaling_if(const AlignedVector<Number> & vec, const Number & scalar)
{
  const auto & inverse_scalar = inverse_scalar_if(scalar);
  return scaling(vec, inverse_scalar);
}



template<typename Number, bool transpose>
void
vect_of_block_impl(Table<2, Number> &                 dst,
                   const std::size_t                  fixed_col_or_row,
                   const Table<2, Number> &           src,
                   const std::array<std::size_t, 2> & row_range,
                   const std::array<std::size_t, 2> & column_range)
{
  const auto & [row_begin, row_end] = row_range;
  Assert(row_begin < row_end, ExcMessage("Empty range of rows."));
  AssertIndexRange(row_end, src.size(0) + 1);

  const auto & [column_begin, column_end] = column_range;
  Assert(column_begin < column_end, ExcMessage("Empty range of columns."));
  AssertIndexRange(column_end, src.size(1) + 1);

  const auto m = row_end - row_begin;
  const auto n = column_end - column_begin;

  AssertIndexRange(fixed_col_or_row, (transpose ? dst.size(0) : dst.size(1)) + 1U);
  AssertDimension(m * n, (transpose ? dst.size(1) : dst.size(0)));

  for(std::size_t i = 0; i < m; ++i)
    for(std::size_t j = 0; j < n; ++j)
    {
      if(transpose)
        dst(fixed_col_or_row, i + j * m) = src(i + row_begin, j + column_begin);
      else
        dst(i + j * m, fixed_col_or_row) = src(i + row_begin, j + column_begin);
    }
}



template<typename Number, bool transpose>
Table<2, Number>
vect_impl(const Table<2, Number> & M)
{
  const std::size_t m = M.size(0);
  const std::size_t n = M.size(1);
  Table<2, Number>  vect(transpose ? 1U : m * n, transpose ? m * n : 1U);

  vect_of_block_impl<Number, transpose>(vect, 0U, M, {0U, M.size(0)}, {0U, M.size(1)});
  return vect;
}



/**
 * Returns the (mathematical) vectorization of a matrix @p M (in short vect(M))
 * which stacks all columns of @p M (from left- to right-hand) into one long
 * column-vector (from top to bottom) that is returned in the end.
 */
template<typename Number>
Table<2, Number>
vect(const Table<2, Number> & M)
{
  return vect_impl<Number, false>(M);
}



/**
 * The transpose of the (mathematical) vectorization operator of a matrix
 * M. For more details see vect() above.
 */
template<typename Number>
Table<2, Number>
Tvect(const Table<2, Number> & M)
{
  return vect_impl<Number, true>(M);
}



template<typename Number, typename OtherNumber, bool do_scaling, bool transpose_src>
Table<2, Number>
folding_impl(const Table<2, Number> & src,
             const unsigned int       n_rows,
             const unsigned int       n_columns,
             const OtherNumber &      alpha,
             const unsigned int       fixed_column_or_row)
{
  AssertIndexRange(fixed_column_or_row, transpose_src ? src.size(0) : src.size(1));
  AssertDimension(transpose_src ? src.size(1) : src.size(0), n_rows * n_columns);
  Table<2, Number> matrix(n_rows, n_columns);
  for(auto i = 0U; i < n_rows; ++i)
    for(auto j = 0U; j < n_columns; ++j)
      if(do_scaling)
      {
        matrix(i, j) = alpha * (transpose_src ? src(fixed_column_or_row, i + j * n_rows) :
                                                src(i + j * n_rows, fixed_column_or_row));
      }
      else
      {
        (void)alpha;
        matrix(i, j) = (transpose_src ? src(fixed_column_or_row, i + j * n_rows) :
                                        src(i + j * n_rows, fixed_column_or_row));
      }
  return matrix;
}



template<typename Number, typename OtherNumber>
Table<2, Number>
sfolding_impl(const Table<2, Number> & src,
              const unsigned int       n_rows,
              const unsigned int       n_columns,
              const OtherNumber &      alpha,
              const unsigned int       fixed_column)
{
  return folding_impl<Number, OtherNumber, true, false>(
    src, n_rows, n_columns, alpha, fixed_column);
}



template<typename Number, typename OtherNumber>
Table<2, Number>
sfoldingT_impl(const Table<2, Number> & src,
               const unsigned int       n_rows,
               const unsigned int       n_columns,
               const OtherNumber &      alpha,
               const unsigned int       fixed_row)
{
  return folding_impl<Number, OtherNumber, true, true>(src, n_rows, n_columns, alpha, fixed_row);
}



/**
 * Folds the column vector @p src into a matrix with size according to @p n_rows
 * and @p n_columns and returns the matrix. For safety the table @p src must not
 * have more than one column and, obviously, the product of @p n_rows and @p
 * n_columns must equal the length of @p src. Note that this is the inverse
 * operation of vectorization vect().
 */
template<typename Number, typename OtherNumber = Number>
Table<2, Number>
folding(const Table<2, Number> & src, const unsigned int n_rows, const unsigned int n_columns)
{
  AssertDimension(src.size(1), 1U);
  return folding_impl<Number, OtherNumber, false, false>(src, n_rows, n_columns, 1., 0U);
}



/**
 * Folds the row vector @p src into a matrix with size according to @p n_rows
 * and @p n_columns and returns the matrix. For safety the table @p src must not
 * have more than one row and, obviously, the product of @p n_rows and @p
 * n_columns must equal the length of @p src. Note that this is the inverse
 * operation of vectorization vectT().
 */
template<typename Number, typename OtherNumber = Number>
Table<2, Number>
foldingT(const Table<2, Number> & src, const unsigned int n_rows, const unsigned int n_columns)
{
  AssertDimension(src.size(1), 1U);
  return folding_impl<Number, OtherNumber, false, true>(src, n_rows, n_columns, 1., 0U);
}



/**
 * Adds a vector @p src to a vector @p dst, that is in pseudo-code dst += src.
 */
template<typename Number>
void
add(AlignedVector<Number> & dst, const AlignedVector<Number> & src)
{
  dst = std::move(sum(dst, src));
}



/**
 * Adds a scaled vector @p src to a vector @p dst, that reads in pseudo-code
 * dst += alpha * src with @p alpha being the scaling factor.
 */
template<typename Number>
void
sadd(AlignedVector<Number> & dst, const Number & alpha, const AlignedVector<Number> & src)
{
  dst = std::move(sum(dst, scaling(src, alpha)));
}



/**
 * Orthogonalizes the last vector of a family of vectors @p vecs by means of a
 * Gram-Schmidt process assuming all previous vectors are mutually orthogonal
 * (but not necessarily normalized). For vectorized arithmetic type @p Number
 * division by zero on any lane is avoided: the algorithm is aware of lanes
 * consisting only of zeros and replaces inverse scalars by zero (all remaining
 * lanes are processed as usual).
 */
template<typename Number>
void
orthogonalize_full(std::vector<AlignedVector<Number>> & vecs)
{
  const auto n   = vecs.size();
  auto &     r_n = vecs.back();
  for(auto j = 0U; j < n - 1; ++j)
  {
    const auto & r_j = vecs[j];
    sadd(r_n, -inverse_scalar_if(inner_product(r_j, r_j)) * inner_product(r_n, r_j), r_j);
  }
}



template<typename Number>
std::pair<std::bitset<get_macro_size<Number>()>, AlignedVector<Number>>
is_positive_definite_impl(const Table<2, Number> & matrix)
{
  AssertDimension(matrix.size(0), matrix.size(1));

  std::bitset<get_macro_size<Number>()> is_posdef;
  is_posdef.flip();

  AlignedVector<Number> eigenvalues_real(matrix.size(0));

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto & eigenvalues = compute_eigenvalues(table_to_fullmatrix(matrix, lane));
    Assert(std::all_of(eigenvalues.cbegin(),
                       eigenvalues.cend(),
                       [](const auto & val) { return has_nearly_zero_abs(val.imag()); }),
           ExcMessage("Not all eigenvalues are real!"));

    const bool is_positive = eigenvalues.back().real() > 0.;
    is_posdef[lane]        = is_positive;

    AssertDimension(eigenvalues.size(), eigenvalues_real.size());
    for(auto i = 0U; i < eigenvalues.size(); ++i)
      scalar_value(eigenvalues_real[i], lane) = eigenvalues[i].real();
  }

  return {is_posdef, eigenvalues_real};
}



/**
 * Checks for each vectorization lane if matrix @p matrix is positive definite
 * by means of its eigendecomposition. We assume @p matrix is Hermitian.
 */
template<typename Number>
std::bitset<get_macro_size<Number>()>
is_positive_definite(const Table<2, Number> & matrix)
{
  const auto & [is_posdef, dummy] = is_positive_definite_impl(matrix);
  (void)dummy;
  return is_posdef;
}



/**
 * Computes the Frobenius norm of matrix @p matrix.
 */
template<typename Number>
Number
frobenius_norm(const Table<2, Number> & matrix)
{
  Number norm(0.);
  for(auto i = 0U; i < matrix.size(0); ++i)
    for(auto j = 0U; j < matrix.size(1); ++j)
      norm += matrix(i, j) * matrix(i, j);
  return std::sqrt(norm);
}



template<typename Number, typename Number2, bool scale_lhs, bool scale_sum, bool scale_rhs>
Table<2, Number>
sum_impl(const Table<2, Number> & lhs,
         const Table<2, Number> & rhs,
         const Number2 &          lhs_factor = 1.,
         const Number2 &          sum_factor = 1.,
         const Number2 &          rhs_factor = 1.)
{
  AssertDimension(lhs.size(0), rhs.size(0));
  AssertDimension(lhs.size(1), rhs.size(1));
  Table<2, Number> sum(lhs.size(0), lhs.size(1));

  for(auto i = 0U; i < lhs.size(0); ++i)
    for(auto j = 0U; j < lhs.size(1); ++j)
    {
      sum(i, j) = (scale_lhs ? lhs_factor * lhs(i, j) : lhs(i, j)) +
                  (scale_rhs ? rhs_factor * rhs(i, j) : rhs(i, j));
      if(scale_sum)
        sum(i, j) = sum_factor * sum(i, j);
    }

  return sum;
}



/**
 * Returns the sum between matrices @p lhs and @p rhs.
 */
template<typename Number>
Table<2, Number>
sum(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return sum_impl<Number, Number, false, false, false>(lhs, rhs);
}



/**
 * Sums both matrices @p lhs and @p rhs first and returns the sum scaled by
 * factor @p alpha.
 */
template<typename Number, typename Number2 = Number>
Table<2, Number>
scaled_sum(const Number2 & alpha, const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return sum_impl<Number, Number2, false, true, false>(lhs, rhs, 1., alpha, 1.);
}


/**
 * Scales the matrix @p lhs by @p alpha first and returns the sum with matrix @p
 * rhs.
 */
template<typename Number, typename Number2 = Number>
Table<2, Number>
ssum(const Number2 & alpha, const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return sum_impl<Number, Number2, true, false, false>(lhs, rhs, alpha, 1., 1.);
}



/**
 * Scales the matrix @p rhs by @p beta first and returns the sum with matrix @p
 * lhs.
 */
template<typename Number, typename Number2 = Number>
Table<2, Number>
sums(const Table<2, Number> & lhs, const Number2 & beta, const Table<2, Number> & rhs)
{
  return sum_impl<Number, Number2, false, false, true>(lhs, rhs, 1., 1., beta);
}



/**
 * First scales both matrices @p lhs and @p rhs by @p alpha and @p beta,
 * respectively, and returns the sum between scaled matrices.
 */
template<typename Number, typename Number2 = Number>
Table<2, Number>
ssums(const Number2 &          alpha,
      const Table<2, Number> & lhs,
      const Number2 &          beta,
      const Table<2, Number> & rhs)
{
  return sum_impl<Number, Number2, true, false, true>(lhs, rhs, alpha, 1., beta);
}



template<typename Number, typename Number2>
Table<2, Number>
scaling_impl(const Number2 & factor, const Table<2, Number> & matrix)
{
  Table<2, Number> scaled_matrix(matrix.size(0), matrix.size(1));
  for(auto i = 0U; i < scaled_matrix.size(0); ++i)
    for(auto j = 0U; j < scaled_matrix.size(1); ++j)
      scaled_matrix(i, j) = factor * matrix(i, j);
  return scaled_matrix;
}



/**
 * Scales the matrix @p matrix by a scalar @p factor.
 */
template<typename Number, typename Number2 = Number>
Table<2, Number>
scaling(const Table<2, Number> & matrix, const Number2 & factor)
{
  return scaling_impl(factor, matrix);
}



/**
 * Scales the matrix @p matrix by a scalar @p factor.
 */
template<typename Number, typename Number2 = Number>
Table<2, Number>
scaling(const Number2 & factor, const Table<2, Number> & matrix)
{
  return scaling_impl(factor, matrix);
}



template<typename Number,
         typename Number2 = Number,
         bool scale,
         bool transpose_lhs,
         bool transpose_rhs>
Table<2, Number>
product_impl(const Table<2, Number> & lhs,
             const Table<2, Number> & rhs,
             const Number2 &          factor = 1.)
{
  const auto lm = transpose_lhs ? lhs.size(1) : lhs.size(0);
  const auto ln = transpose_lhs ? lhs.size(0) : lhs.size(1);
  const auto rm = transpose_rhs ? rhs.size(1) : rhs.size(0);
  const auto rn = transpose_rhs ? rhs.size(0) : rhs.size(1);
  AssertDimension(ln, rm);
  (void)rm;
  const auto & n = ln;

  Table<2, Number> prod(lm, rn);

  for(auto i = 0U; i < lm; ++i)
    for(auto j = 0U; j < rn; ++j)
    {
      Number inner_prod(0.);
      for(auto k = 0U; k < n; ++k)
        inner_prod +=
          (transpose_lhs ? lhs(k, i) : lhs(i, k)) * (transpose_rhs ? rhs(j, k) : rhs(k, j));
      prod(i, j) = scale ? factor * inner_prod : inner_prod;
    }

  return prod;
}



/**
 * Returns the matrix-matrix product between left-hand matrix @p lhs and
 * right-hand matrix @p rhs.
 */
template<typename Number>
Table<2, Number>
product(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return product_impl<Number, Number, false, false, false>(lhs, rhs);
}



/**
 * Returns the matrix-matrix product between the transpose of left-hand matrix @p lhs and
 * right-hand matrix @p rhs.
 */
template<typename Number>
Table<2, Number>
Tproduct(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return product_impl<Number, Number, false, true, false>(lhs, rhs);
}



/**
 * Returns the matrix-matrix product between left-hand matrix @p lhs and the
 * transpose of right-hand matrix @p rhs.
 */
template<typename Number>
Table<2, Number>
productT(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return product_impl<Number, Number, false, false, true>(lhs, rhs);
}



/**
 * Returns the matrix-matrix product between the transpose of left-hand matrix
 * @p lhs and the transpose of right-hand matrix @p rhs.
 */
template<typename Number>
Table<2, Number>
TproductT(const Table<2, Number> & lhs, const Table<2, Number> & rhs)
{
  return product_impl<Number, Number, false, true, true>(lhs, rhs);
}



/**
 * Computes the singular value decomposition of the matrix @p matrix in the form
 * U Sigma VT first and returns the triplet of matrices [UT, Sigma^{-1}, V]
 * which defines a "pseudo" inverse of @p matrix. If singular values are below
 * the given threshold @p threshold their reciprocal is set to zero avoiding by
 * (nearly) zero. Each vectorization lane of arithmetic type @p Number is
 * treated on its own.
 */
template<typename Number>
std::array<Table<2, Number>, 3>
compute_inverse_svd(const Table<2, Number> & matrix)
{
  /// TODO for non-square matrices...
  AssertDimension(matrix.size(0), matrix.size(1));

  std::array<Table<2, Number>, 3> inverse_svd;
  auto &                          V = inverse_svd[0];
  V.reinit(matrix.size(1), matrix.size(1));
  auto & invSigma = inverse_svd[1];
  invSigma.reinit(matrix.size(1), matrix.size(0));
  auto & UT = inverse_svd[2];
  UT.reinit(matrix.size(0), matrix.size(0));

  using scalar_value_type = typename ExtractScalarType<Number>::type;
  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    LAPACKFullMatrix<scalar_value_type> micro_matrix(matrix.size(0), matrix.size(1));
    micro_matrix = table_to_fullmatrix(matrix, lane);

    const auto & [tildeV, micro_invSigma, tildeUT] = ::compute_inverse_svd(micro_matrix);

    for(auto i = 0U; i < UT.size(0); ++i)
      for(auto j = 0U; j < UT.size(1); ++j)
        scalar_value(UT(i, j), lane) = tildeUT(i, j);

    for(auto i = 0U; i < V.size(0); ++i)
      for(auto j = 0U; j < V.size(1); ++j)
        scalar_value(V(i, j), lane) = tildeV(i, j);

    for(auto i = 0U; i < std::min<unsigned int>(invSigma.size(0), invSigma.size(1)); ++i)
      scalar_value(invSigma(i, i), lane) = micro_invSigma(i, i);
  }

  return inverse_svd;
}



template<typename Number, bool transpose_mat, bool is_matvec>
AlignedVector<Number>
product_impl(const Table<2, Number> & mat, const AlignedVector<Number> & vec)
{
  /// is_matvec means mat*vec (otherwise transpose(vec)*mat is computed)
  const auto m = transpose_mat ? mat.size(1) : mat.size(0);
  const auto n = transpose_mat ? mat.size(0) : mat.size(1);
  AssertDimension(is_matvec ? n : m, vec.size());

  AlignedVector<Number> prod(is_matvec ? m : n);

  for(auto i = 0U; i < m; ++i)
    for(auto j = 0U; j < n; ++j)
      if(is_matvec) // mat*vec
        prod[i] = (transpose_mat ? mat(j, i) : mat(i, j)) * vec[j];
      else // vecT*mat
        prod[j] = vec[i] * (transpose_mat ? mat(j, i) : mat(i, j));

  return prod;
}



/**
 * Computes the matrix-vector multiplication w = A * v returning the column
 * vector w.
 */
template<typename Number>
AlignedVector<Number>
product(const Table<2, Number> & A, const AlignedVector<Number> & v)
{
  return product_impl<Number, false, true>(A, v);
}



/**
 * Computes the matrix-vector multiplication w = A^T * v, where A^T denotes the
 * transpose of matrix A, returning the column vector w.
 */
template<typename Number>
AlignedVector<Number>
Tproduct(const Table<2, Number> & A, const AlignedVector<Number> & v)
{
  return product_impl<Number, true, true>(A, v);
}



/**
 * Computes the matrix-vector multiplication w = v^T * A, where v^T denotes the
 * transpose of column vector v, returning the row vector w.
 */
template<typename Number>
AlignedVector<Number>
product(const AlignedVector<Number> & v, const Table<2, Number> & A)
{
  return product_impl<Number, false, false>(A, v);
}



/**
 * Computes the matrix-vector multiplication w = v^T * A^T, where v^T denotes
 * the transpose of column vector v and A^T the transpose of matrix A, returning
 * the row vector w.
 */
template<typename Number>
AlignedVector<Number>
productT(const AlignedVector<Number> & v, const Table<2, Number> & A)
{
  return product_impl<Number, true, false>(A, v);
}



template<typename Number>
Table<2, Number>
transpose(const Table<2, Number> & matrix)
{
  Table<2, Number> transpose_matrix(matrix.size(1), matrix.size(0));
  for(unsigned int i = 0; i < transpose_matrix.size(0); ++i)
    for(unsigned int j = 0; j < transpose_matrix.size(1); ++j)
      transpose_matrix(i, j) = matrix(j, i);
  return transpose_matrix;
}

} // namespace LinAlg



template<typename Number>
Table<2, Number>
operator*(const Table<2, Number> & matrix, const Number & factor)
{
  return LinAlg::scaling(factor, matrix);
}



template<typename Number>
Table<2, Number>
operator*(const Number & factor, const Table<2, Number> & matrix)
{
  return LinAlg::scaling(factor, matrix);
}



template<typename Number>
Table<2, VectorizedArray<Number>>
operator*(const Table<2, VectorizedArray<Number>> & matrix, const Number & factor)
{
  return LinAlg::scaling(factor, matrix);
}



template<typename Number>
Table<2, VectorizedArray<Number>>
operator*(const Number & factor, const Table<2, VectorizedArray<Number>> & matrix)
{
  return LinAlg::scaling(factor, matrix);
}



// compute the Khatri-Rao product of two matrices
template<typename Number>
Table<2, Number>
khatri_rao(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
  AssertDimension(in1.size()[1], in2.size()[1]);
  Table<2, Number> ret(in1.size()[0] * in2.size()[0], in1.size()[1]);
  for(std::size_t i = 0; i < in1.size()[0]; i++)
    for(std::size_t j = 0; j < in2.size()[0]; j++)
      for(std::size_t k = 0; k < in1.size()[1]; k++)
        ret(i * in2.size()[0] + j, k) = in1(i, k) * in2(j, k);
  return ret;
}



// compute the Hadamard product of two matrices
template<typename Number>
Table<2, Number>
hadamard(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
  AssertDimension(in1.size()[1], in2.size()[1]);
  AssertDimension(in1.size()[0], in2.size()[0]);
  Table<2, Number> ret(in1.size()[0], in1.size()[1]);
  for(std::size_t i = 0; i < in1.size()[0]; i++)
    for(std::size_t j = 0; j < in1.size()[1]; j++)
      ret(i, j) = in1(i, j) * in2(i, j);
  return ret;
}



// Flatten Table to AlignedVector
template<typename Number>
AlignedVector<Number>
vectorize_matrix(const Table<2, Number> & tab)
{
  std::size_t           m = tab.size()[0];
  std::size_t           n = tab.size()[1];
  AlignedVector<Number> ret(m * n);
  for(std::size_t k = 0; k < m; k++)
    for(std::size_t l = 0; l < n; l++)
      ret[k * n + l] = tab(k, l);
  return ret;
}



// use the svd to compute a pseudo inverse, only for quadratic matrices!
template<typename Number>
Table<2, Number>
pseudo_inverse(const Table<2, Number> & matrix)
{
  const auto & [V, invSigma, UT] = LinAlg::compute_inverse_svd(matrix);
  auto inverse_matrix            = LinAlg::product(LinAlg::product(V, invSigma), UT);
  AssertDimension(inverse_matrix.size(0), matrix.size(1));
  AssertDimension(inverse_matrix.size(1), matrix.size(0));
  return inverse_matrix;
}



// Calculate the unfolding matrix in some direction of a third order tensor with polyadic rank one,
// that is a tensor given as the polyadic product of three vectors
template<typename Number>
Table<2, Number>
unfold_rank1(std::array<AlignedVector<Number>, 3> polyadic_factors, std::size_t direction)
{
  AlignedVector<Number> first;
  AlignedVector<Number> second;
  AlignedVector<Number> third;
  if(direction == 0)
  {
    first  = polyadic_factors[0];
    second = polyadic_factors[2];
    third  = polyadic_factors[1];
  }
  if(direction == 1)
  {
    first  = polyadic_factors[1];
    second = polyadic_factors[2];
    third  = polyadic_factors[0];
  }
  if(direction == 2)
  {
    first  = polyadic_factors[2];
    second = polyadic_factors[1];
    third  = polyadic_factors[0];
  }
  Table<2, Number> ret(first.size(), second.size() * third.size());
  for(std::size_t i = 0; i < first.size(); i++)
    for(std::size_t j = 0; j < second.size(); j++)
      for(std::size_t k = 0; k < third.size(); k++)
        ret(i, j * third.size() + k) = first[i] * second[j] * third[k];
  return ret;
}



// Calculate the unfolding matrix in some direction of a third order tensor with higher polyadic
// rank, that is a tensor given as the sum of polyadic products of three vectors, here each polyadic
// product is given as one entry of the vector polyadic_factors
template<typename Number>
Table<2, Number>
unfold_rankk(std::vector<std::array<AlignedVector<Number>, 3>> polyadic_factors,
             std::size_t                                       direction)
{
  Table<2, Number> ret = unfold_rank1(polyadic_factors[0], direction);
  for(std::size_t i = 1; i < polyadic_factors.size(); i++)
    ret = LinAlg::sum(ret, unfold_rank1(polyadic_factors[i], direction));
  return ret;
}



#endif
