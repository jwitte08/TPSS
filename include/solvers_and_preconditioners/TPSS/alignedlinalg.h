#ifndef ALIGNEDLINALG_H_
#define ALIGNEDLINALG_H_

#include <numeric>

#include <deal.II/base/table.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/lapack_full_matrix.h>

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
 * scalar is nearly zero is scaled by zero instead of the inverse scalar
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
Table<2, Number>
vect_impl(const Table<2, Number> & M)
{
  const unsigned int m = M.size(0);
  const unsigned int n = M.size(1);
  Table<2, Number>   vect(transpose ? 1U : m * n, transpose ? m * n : 1U);

  for(unsigned int i = 0; i < m; ++i)
    for(unsigned int j = 0; j < n; ++j)
      if(transpose)
        vect(0U, i + j * m) = M(i, j);
      else
        vect(i + j * m, 0U) = M(i, j);
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

} // namespace LinAlg



// Add two AlignedVectors
template<typename Number>
AlignedVector<Number>
vector_addition(const AlignedVector<Number> & in1, const AlignedVector<Number> & in2)
{
  AssertDimension(in1.size(), in2.size());
  AlignedVector<Number> ret = AlignedVector<Number>(in1);
  for(std::size_t i = 0; i < in1.size(); i++)
    ret[i] = in1[i] + in2[i];
  return ret;
}



// Multiply AlignedVector with scalar
template<typename Number>
AlignedVector<Number>
vector_scaling(const AlignedVector<Number> & in, const Number & scalar)
{
  AlignedVector<Number> ret = AlignedVector<Number>(in);
  for(std::size_t i = 0; i < in.size(); i++)
    ret[i] = in[i] * scalar;
  return ret;
}



// Calculate inner product of two AlignedVectors
template<typename Number>
Number
inner_product(const AlignedVector<Number> & in1, const AlignedVector<Number> & in2)
{
  AssertDimension(in1.size(), in2.size());
  Number ret = Number(0);
  for(std::size_t i = 0; i < in1.size(); i++)
    ret += in1[i] * in2[i];
  return ret;
}


// Add two Tables
template<typename Number>
Table<2, Number>
matrix_addition(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
  AssertDimension(in1.size()[0], in2.size()[0]);
  AssertDimension(in1.size()[1], in2.size()[1]);
  Table<2, Number> ret = Table<2, Number>(in1);
  for(std::size_t i = 0; i < in1.size()[0]; i++)
    for(std::size_t j = 0; j < in1.size()[1]; j++)
      ret(i, j) = in1(i, j) + in2(i, j);
  return ret;
}

// Multiply Table with scalar
template<typename Number>
Table<2, Number>
matrix_scaling(const Table<2, Number> & in, const Number & scalar)
{
  Table<2, Number> ret = Table<2, Number>(in);
  for(std::size_t i = 0; i < in.size()[0]; i++)
    for(std::size_t j = 0; j < in.size()[1]; j++)
      ret(i, j) = in(i, j) * scalar;
  return ret;
}



template<typename Number>
Table<2, Number>
operator*(const Table<2, Number> & matrix, const Number & factor)
{
  return matrix_scaling(matrix, factor);
}



template<typename Number>
Table<2, Number>
operator*(const Number & factor, const Table<2, Number> & matrix)
{
  return matrix * factor;
}



template<typename Number>
Table<2, VectorizedArray<Number>>
operator*(const Table<2, VectorizedArray<Number>> & matrix, const Number & factor)
{
  return matrix * make_vectorized_array<Number>(factor);
}



template<typename Number>
Table<2, VectorizedArray<Number>>
operator*(const Number & factor, const Table<2, VectorizedArray<Number>> & matrix)
{
  return matrix * make_vectorized_array<Number>(factor);
}



// invert a number, if number is zero return zero
template<typename Number>
Number
invert_safe(const Number x)
{
  if(std::abs(x) <= std::numeric_limits<Number>::epsilon())
    return Number(0);
  else
    return Number(1) / x;
}



// invert a VectirizedArray, if a component is zero return zero for that component
template<typename Number>
VectorizedArray<Number>
invert_safe(const VectorizedArray<Number> x)
{
  VectorizedArray<Number> ret;
  constexpr std::size_t   macro_size = VectorizedArray<Number>::size();
  for(std::size_t lane = 0; lane < macro_size; lane++)
  {
    if(std::abs(x[lane]) <= std::numeric_limits<Number>::epsilon())
      ret[lane] = Number(0);
    else
      ret[lane] = Number(1) / x[lane];
  }
  return ret;
}



// Divide AlignedVector by scalar, if vector is zero allow scalar to be zero
template<typename Number>
AlignedVector<Number>
vector_inverse_scaling(const AlignedVector<Number> & in, const Number & scalar)
{
  AlignedVector<Number> ret(in.size());
  for(std::size_t i = 0; i < in.size(); i++)
    ret[i] = in[i] * invert_safe(scalar);
  return ret;
}



// Multiply Matrix by vector
template<typename Number>
AlignedVector<Number>
matrix_vector_multiplication(const Table<2, Number> & in_mat, const AlignedVector<Number> & in_vec)
{
  AssertDimension(in_mat.size()[1], in_vec.size());
  AlignedVector<Number> ret(in_mat.size()[0]);
  for(std::size_t i = 0; i < in_mat.size()[0]; i++)
    for(std::size_t j = 0; j < in_vec.size(); j++)
      ret[i] += in_mat(i, j) * in_vec[j];
  return ret;
}



// Multiply transpose of Matrix by vector
template<typename Number>
AlignedVector<Number>
matrix_transpose_vector_multiplication(const Table<2, Number> &      in_mat,
                                       const AlignedVector<Number> & in_vec)
{
  AssertDimension(in_mat.size()[0], in_vec.size());
  AlignedVector<Number> ret(in_mat.size()[1]);
  for(std::size_t i = 0; i < in_mat.size()[1]; i++)
    for(std::size_t j = 0; j < in_vec.size(); j++)
      ret[i] += in_mat(j, i) * in_vec[j];
  return ret;
}



// Multiply Matrix @p in1 by Matrix @p in2
template<typename Number>
Table<2, Number>
matrix_multiplication(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
  AssertDimension(in1.size()[1], in2.size()[0]);
  Table<2, Number> ret(in1.size()[0], in2.size()[1]);
  for(std::size_t i = 0; i < in1.size()[0]; i++)
    for(std::size_t j = 0; j < in2.size()[1]; j++)
      for(std::size_t k = 0; k < in2.size()[0]; k++)
        ret(i, j) += in1(i, k) * in2(k, j);
  return ret;
}



// Multiply transpose of Matrix @p in1 by Matrix @p in2
template<typename Number>
Table<2, Number>
matrix_transpose_multiplication(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
  AssertDimension(in1.size()[0], in2.size()[0]);
  Table<2, Number> ret(in1.size()[1], in2.size()[1]);
  for(std::size_t i = 0; i < in1.size()[1]; i++)
    for(std::size_t j = 0; j < in2.size()[1]; j++)
      for(std::size_t k = 0; k < in2.size()[0]; k++)
        ret(i, j) += in1(k, i) * in2(k, j);
  return ret;
}



// Multiply Matrix by transpose of Matrix
template<typename Number>
Table<2, Number>
matrix_multiplication_transpose(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
  AssertDimension(in1.size()[0], in2.size()[0]);
  Table<2, Number> ret(in1.size()[1], in2.size()[1]);
  for(std::size_t i = 0; i < in1.size()[1]; i++)
    for(std::size_t j = 0; j < in2.size()[1]; j++)
      for(std::size_t k = 0; k < in2.size()[0]; k++)
        ret(i, j) += in1(i, k) * in2(j, k);
  return ret;
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



// TODO remove
// For vectorizedarray check if > holds for all elements
template<typename Number>
bool
operator>(VectorizedArray<Number> a, VectorizedArray<Number> b)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::size();
  for(std::size_t lane = 0; lane < macro_size; lane++)
    if(a[lane] <= b[lane])
      return false;
  return true;
}



// For vectorizedarray check if > holds for all elements
template<typename Number>
bool
operator>(VectorizedArray<Number> a, Number b)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::size();
  for(std::size_t lane = 0; lane < macro_size; lane++)
    if(a[lane] <= b)
      return false;
  return true;
}



namespace std
{
template<typename Number>
class numeric_limits<VectorizedArray<Number>>
{
public:
  static Number
  epsilon()
  {
    return numeric_limits<Number>::epsilon();
  };
};
} // namespace std



// Two Matrices are considered equal if all of their components are equal up to machine epsilon
template<typename Number>
bool
operator==(Table<2, Number> tab1, Table<2, Number> tab2)
{
  AssertDimension(tab1.size()[0], tab2.size()[0]);
  AssertDimension(tab1.size()[1], tab2.size()[1]);
  std::size_t m                 = tab1.size()[0];
  std::size_t n                 = tab1.size()[1];
  Number      max_table_element = Number(0);
  for(std::size_t i = 0; i < m; i++)
    for(std::size_t j = 0; j < n; j++)
    {
      if(std::abs(tab1(i, j)) > max_table_element)
        max_table_element = std::abs(tab1(i, j));
      if(std::abs(tab2(i, j)) > max_table_element)
        max_table_element = std::abs(tab2(i, j));
    }
  for(std::size_t i = 0; i < m; i++)
    for(std::size_t j = 0; j < n; j++)
      if(std::abs(tab1(i, j) - tab2(i, j)) >
         std::numeric_limits<Number>::epsilon() * max_table_element * Number(10.0))
      {
        std::cout << std::abs(tab1(i, j) - tab2(i, j)) << " "
                  << std::numeric_limits<Number>::epsilon() * max_table_element << "\n";
        return false;
      }
  return true;
}



// print a table up to digits sginificant digits
template<typename Number>
void
printTable(Table<2, Number> tab, double digits = 2)
{
  std::size_t m = tab.size()[0];
  std::size_t n = tab.size()[1];
  std::cout << "------------------------------------\n";
  for(std::size_t i = 0; i < m; i++)
  {
    for(std::size_t j = 0; j < n; j++)
      std::cout << ((int)(tab(i, j) * std::pow(10.0, digits) + 0.5)) / std::pow(10.0, digits)
                << "\t";
    std::cout << "\n";
  }
  std::cout << "------------------------------------\n";
}



template<typename Number>
void
printTable(Table<2, VectorizedArray<Number>> tab)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::size();
  std::size_t           m          = tab.size()[0];
  std::size_t           n          = tab.size()[1];
  std::cout << "------------------------------------\n";
  for(std::size_t lane = 0; lane < macro_size; lane++)
  {
    std::cout << "-----------------\n";
    for(std::size_t i = 0; i < m; i++)
    {
      for(std::size_t j = 0; j < n; j++)
        std::cout << ((int)(tab(i, j)[lane] * 100 + 0.5)) / 100.0 << "\t";
      std::cout << "\n";
    }
    std::cout << "-----------------\n";
  }
  std::cout << "------------------------------------\n";
}



// print aligned vector
template<typename Number>
void
printAlignedVector(AlignedVector<Number> vec)
{
  std::size_t m = vec.size();
  std::cout << "######################################\n";
  for(std::size_t i = 0; i < m; i++)
    std::cout << ((int)(vec[i] * 100 + 0.5)) / 100.0 << "\t";
  std::cout << "\n######################################\n";
}



template<typename Number>
void
printAlignedVector(AlignedVector<VectorizedArray<Number>> vec)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::size();
  std::size_t           m          = vec.size();
  std::cout << "######################################";
  for(std::size_t lane = 0; lane < macro_size; lane++)
  {
    std::cout << "\n-----------------\n";
    for(std::size_t i = 0; i < m; i++)
      std::cout << ((int)(vec[i][lane] * 100 + 0.5)) / 100.0 << "\t";
  }
  std::cout << "\n######################################\n";
}



// compute svd of a table by tranforming it into a lapack matrix and computing the svd there
template<typename Number>
void
svd(const Number *    matrix_begin,
    const std::size_t m,
    const std::size_t n,
    Number *          U_begin,
    Number *          singular_values_begin,
    Number *          VT_begin)
{
  LAPACKFullMatrix<Number> mat(m, n);
  for(unsigned int mm = 0; mm < m; ++mm)
    for(unsigned int nn = 0; nn < n; ++nn)
      mat(mm, nn) = *(matrix_begin++);
  mat.compute_svd();
  LAPACKFullMatrix<Number> U_  = mat.get_svd_u();
  LAPACKFullMatrix<Number> VT_ = mat.get_svd_vt();
  for(std::size_t i = 0; i < U_.size()[0]; i++)
    for(std::size_t j = 0; j < U_.size()[1]; j++)
      *(U_begin++) = U_(i, j);
  for(std::size_t i = 0; i < VT_.size()[0]; i++)
  {
    *(singular_values_begin++) = mat.singular_value(i);
    for(std::size_t j = 0; j < VT_.size()[1]; j++)
    {
      *(VT_begin++) = VT_(i, j);
    }
  }
}



template<typename Number>
void
svd(const Table<2, Number>  matrix,
    Table<2, Number> &      U,
    AlignedVector<Number> & singular_values,
    Table<2, Number> &      VT)
{
  svd<Number>(&(matrix(0, 0)),
              matrix.size()[0],
              matrix.size()[1],
              &(U(0, 0)),
              &(singular_values[0]),
              &(VT(0, 0)));
}



// compute the svd lane wise and put everything together again
template<typename Number>
void
svd(const Table<2, VectorizedArray<Number>> & matrix,
    Table<2, VectorizedArray<Number>> &       U,
    AlignedVector<VectorizedArray<Number>> &  singular_values,
    Table<2, VectorizedArray<Number>> &       VT)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::size();
  for(std::size_t lane = 0; lane < macro_size; lane++)
  {
    Table<2, Number>      lane_matrix(matrix.size()[0], matrix.size()[1]);
    Table<2, Number>      lane_U(U.size()[0], U.size()[1]);
    AlignedVector<Number> lane_singular_values(singular_values.size());
    Table<2, Number>      lane_VT(VT.size()[0], VT.size()[1]);
    for(std::size_t i = 0; i < matrix.size()[0]; i++)
      for(std::size_t j = 0; j < matrix.size()[1]; j++)
        lane_matrix(i, j) = matrix(i, j)[lane];

    svd<Number>(lane_matrix, lane_U, lane_singular_values, lane_VT);
    for(std::size_t i = 0; i < U.size()[0]; i++)
      for(std::size_t j = 0; j < U.size()[1]; j++)
        U(i, j)[lane] = lane_U(i, j);
    for(std::size_t i = 0; i < VT.size()[0]; i++)
    {
      singular_values[i][lane] = lane_singular_values[i];
      for(std::size_t j = 0; j < VT.size()[1]; j++)
        VT(i, j)[lane] = lane_VT(i, j);
    }
  }
}



// use the svd to compute a pseudo inverse, only for quadratic matrices!
template<typename Number>
Table<2, Number>
pseudo_inverse(const Table<2, Number> matrix)
{
  AssertDimension(matrix.size()[0], matrix.size()[1]);
  std::size_t           n = matrix.size()[0];
  Table<2, Number>      ret(n, n);
  Table<2, Number>      U(n, n);
  Table<2, Number>      VT(n, n);
  Table<2, Number>      sing_inv_mat(n, n);
  AlignedVector<Number> sing_vect(n);
  svd(matrix, U, sing_vect, VT);
  for(std::size_t i = 0; i < n; i++)
    sing_inv_mat(i, i) = invert_safe(sing_vect[i]);
  ret = matrix_multiplication(matrix_multiplication(U, sing_inv_mat), VT);
  return ret;
}



#endif
