/*
 * kroneckersvd.h
 *
 *  Created on: Nov 14, 2019
 *      Author: schmidt
 */

#ifndef KRONECKERSVD_H_
#define KRONECKERSVD_H_

#include <deal.II/base/table.h>
#include <deal.II/lac/lapack_full_matrix.h>

using namespace dealii;

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

/*
  Divide AlignedVector by scalar, if vector is zero allow scalar to be zero.
  Alowing this makes it possible to simultaneously execute a Lanczos algorithm
  on matrices with different Kronecker rank. We check for zero, so for
  vectorizedarrays we need to do it comonent wise
*/

template<typename Number>
AlignedVector<VectorizedArray<Number>>
vector_inverse_scaling(const AlignedVector<VectorizedArray<Number>> & in,
                       const VectorizedArray<Number> &                scalar)
{
  AlignedVector<VectorizedArray<Number>> ret(in.size());

  constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
  for(std::size_t lane = 0; lane < macro_size; lane++)
  {
    for(std::size_t i = 0; i < in.size(); i++)
    {
      if(std::abs(scalar[lane]) >=
         std::numeric_limits<Number>::epsilon() * std::abs(scalar[lane] + in[i][lane]))
        ret[i][lane] = in[i][lane] / scalar[lane];
      else
      {
        ret[i][lane] = 0;
      }
    }
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
  {
    if(std::abs(scalar) >= std::numeric_limits<Number>::epsilon() * std::abs(scalar + in[i]))
      ret[i] = in[i] / scalar;
    else
    {
      ret[i] = 0;
    }
  }
  return ret;
}

// Multiply Matrix by Matrix
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

// Multiply transpose of Matrix by Matrix
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

// Calculate the product of a Rank 1 matrix, given as dyadic product of two
// vectors, with a vector: (in⊗inT)vec where ⊗ is the dyadic product
template<typename Number>
AlignedVector<Number>
rank1_vector_multiplication(const AlignedVector<Number> & in,
                            const AlignedVector<Number> & inT,
                            const AlignedVector<Number> & vec)
{
  AssertDimension(inT.size(), vec.size());
  Number inTMvec = inner_product(inT, vec);
  return vector_scaling(in, inTMvec);
}

// calculates the product of a rank k matrix, given as k dyadic products of two
// vectors, with a vector
template<typename Number>
AlignedVector<Number>
rankk_vector_multiplication(const std::vector<AlignedVector<Number>> & in,
                            const std::vector<AlignedVector<Number>> & inT,
                            const AlignedVector<Number> &              vec)
{
  AlignedVector<Number> ret(in[0].size());
  for(std::size_t i = 0; i < in.size(); i++)
    ret = vector_addition(ret, rank1_vector_multiplication(in[i], inT[i], vec));
  return ret;
}

// transform the last vector of a family of vectors such that the family becomes
// orthogonal
template<typename Number>
void
orthogonalize(std::vector<AlignedVector<Number>> & vectors)
{
  for(std::size_t j = 0; j < vectors.size() - 1; j++) // orthogonalize r
    vectors.back() = vector_addition(vectors.back(),
                                     vector_scaling(vectors[j],
                                                    -inner_product(vectors.back(), vectors[j]) /
                                                      inner_product(vectors[j], vectors[j])));
}

// For vectorizedarray check if > holds for all elements
template<typename Number>
bool
operator>(VectorizedArray<Number> a, VectorizedArray<Number> b)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
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
  constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
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

// compute svd of bidiagonal matrix
template<typename Number>
void
bidiagonal_svd(const AlignedVector<Number> & diagonal,
               const AlignedVector<Number> & upper_diagonal,
               Table<2, Number> &            U,
               AlignedVector<Number> &       singular_values,
               Table<2, Number> &            VT)
{
  std::size_t base_len = diagonal.size();
  AssertDimension(base_len, upper_diagonal.size() + 1);
  AssertDimension(base_len, U.size()[0]);
  AssertDimension(base_len, U.size()[1]);
  AssertDimension(base_len, singular_values.size());
  AssertDimension(base_len, VT.size()[0]);
  AssertDimension(base_len, VT.size()[1]);
  LAPACKFullMatrix<Number> bidiag_mat(base_len, base_len);
  for(std::size_t i = 0; i < base_len; i++)
  {
    bidiag_mat(i, i) = diagonal[i];
    if(i < base_len - 1)
      bidiag_mat(i, i + 1) = upper_diagonal[i];
  }
  bidiag_mat.compute_svd();
  LAPACKFullMatrix<Number> U_  = bidiag_mat.get_svd_u();
  LAPACKFullMatrix<Number> VT_ = bidiag_mat.get_svd_vt();
  for(std::size_t i = 0; i < base_len; i++)
    for(std::size_t j = 0; j < base_len; j++)
    {
      U(i, j)            = U_(i, j);
      VT(i, j)           = VT_(i, j);
      singular_values[i] = bidiag_mat.singular_value(i);
    }
}

// compute lane-wise svd of bidiagonal matrix
template<typename Number>
void
bidiagonal_svd(const AlignedVector<VectorizedArray<Number>> & diagonal,
               const AlignedVector<VectorizedArray<Number>> & upper_diagonal,
               Table<2, VectorizedArray<Number>> &            U,
               AlignedVector<VectorizedArray<Number>> &       singular_values,
               Table<2, VectorizedArray<Number>> &            VT)
{
  constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
  std::size_t           base_len   = diagonal.size();
  AlignedVector<Number> diag(base_len);
  AlignedVector<Number> updiag(base_len - 1);
  AlignedVector<Number> sing_values(base_len);
  Table<2, Number>      U_(base_len, base_len);
  Table<2, Number>      VT_(base_len, base_len);
  for(std::size_t lane = 0; lane < macro_size; lane++)
  {
    for(std::size_t i = 0; i < base_len; i++)
    {
      diag[i] = diagonal[i][lane];
      if(i < base_len - 1)
        updiag[i] = upper_diagonal[i][lane];
    }
    bidiagonal_svd(diag, updiag, U_, sing_values, VT_);
    for(std::size_t i = 0; i < base_len; i++)
    {
      for(std::size_t j = 0; j < base_len; j++)
      {
        U(i, j)[lane]  = U_(i, j);
        VT(i, j)[lane] = VT_(i, j);
      }
      singular_values[i][lane] = sing_values[i];
    }
  }
}

/*
  Compute the low Kronecker rank approximation, i.e. the ksvd, of a matrix of
  the form M = Σ A_i⊗B_i where A_i is called the big matrix and B_i is called
  the small matrix. We first reshuffle M and then compute the first few singular
  values/vectors by using the Lanczos algorithm. The matricization of these
  singular vectors then is the low Kronecker rank approximation. The matrix M is
  passed in "in", and the low rank approximation is passed in "out"
*/
template<int dim, typename Number, int out_rank>
void
compute_ksvd(const std::vector<std::array<Table<2, Number>, dim>>            &in,
             std::array<std::array<Table<2, Number>, dim>, out_rank> & out,
             const std::size_t lanczos_iterations = out_rank * out_rank + 10)
{
  std::size_t in_rank = in.size();
  std::size_t big_m   = in[0][0].size()[0];
  std::size_t big_n   = in[0][0].size()[1];
  std::size_t small_m = in[0][1].size()[0];
  std::size_t small_n = in[0][1].size()[1];
  if(dim == 2)
  {
    std::vector<AlignedVector<Number>> big_matrices_vectorized;
    std::vector<AlignedVector<Number>> small_matrices_vectorized;
    for(std::size_t i = 0; i < in_rank; i++)
    {
      small_matrices_vectorized.push_back(vectorize_matrix(in[i][1]));
      big_matrices_vectorized.push_back(vectorize_matrix(in[i][0]));
    }
    AlignedVector<Number> beta;
    beta.push_back(Number(1)); // we artificially introduce a first value for
                               // beta to define beta.back()
    AlignedVector<Number>              alpha;
    std::vector<AlignedVector<Number>> r;
    std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(small_m * small_n)};
    p.back()[0]                          = Number(1);
    std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(big_m * big_n)};
    std::vector<AlignedVector<Number>> v;
    for(std::size_t i = 0;
        i < lanczos_iterations && std::abs(beta.back()) > std::numeric_limits<Number>::epsilon();
        i++)
    {
      beta.push_back(std::sqrt(inner_product(p.back(), p.back())));
      v.push_back(vector_inverse_scaling(p.back(), beta.back()));
      r.push_back(vector_scaling(u.back(), -beta.back()));
      r.back() = vector_addition(r.back(),
                                 rankk_vector_multiplication(big_matrices_vectorized,
                                                             small_matrices_vectorized,
                                                             v.back()));
      orthogonalize(r);
      alpha.push_back(std::sqrt(inner_product(r.back(), r.back())));
      u.push_back(vector_inverse_scaling(r.back(), alpha.back()));
      p.push_back(vector_scaling(v.back(), -alpha.back()));
      p.back() = vector_addition(p.back(),
                                 rankk_vector_multiplication(small_matrices_vectorized,
                                                             big_matrices_vectorized,
                                                             u.back()));
      orthogonalize(p);
    }
    std::size_t           base_len = alpha.size() - 1;
    Table<2, Number>      U(base_len, big_m * big_n); // discard first value of u since it is zero
    Table<2, Number>      V(base_len,
                       small_m * small_n); // discard last value of v since it is zero
    AlignedVector<Number> real_beta(base_len -
                                    1); // discard first two values of beta, first is artificially
                                        // introduced, second only depends on inital guess
    AlignedVector<Number> real_alpha(base_len); // discard last value of alpha since it is zero
    for(std::size_t i = 0; i < base_len; i++)
    {
      real_alpha[i] = alpha[i];
      if(i < base_len - 1)
        real_beta[i] = beta[i + 2];
      for(std::size_t j = 0; j < big_m * big_n; j++)
        U(i, j) = u[i + 1][j];
      for(std::size_t j = 0; j < small_m * small_n; j++)
        V(i, j) = v[i][j];
    }

    AlignedVector<Number> singular_values(base_len);
    Table<2, Number>      tildeU(base_len, base_len);
    Table<2, Number>      tildeVT(base_len, base_len);

    bidiagonal_svd(real_alpha, real_beta, tildeU, singular_values, tildeVT);

    Table<2, Number> left_singular_vectors  = matrix_transpose_multiplication(tildeU, U);
    Table<2, Number> right_singular_vectors = matrix_multiplication(tildeVT, V);

    for(std::size_t i = 0; i < out_rank; i++)
    {
      for(std::size_t k = 0; k < big_m; k++)
        for(std::size_t l = 0; l < big_n; l++)
          out[i][0](k, l) = left_singular_vectors(i, k * big_n + l) * std::sqrt(singular_values[i]);
      for(std::size_t k = 0; k < small_m; k++)
        for(std::size_t l = 0; l < small_n; l++)
          out[i][1](k, l) =
            right_singular_vectors(i, k * small_n + l) * std::sqrt(singular_values[i]);
    }
  }
}

#endif
