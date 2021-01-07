/*
 * kroneckersvd_new.h
 *
 *  Created on: Nov 14, 2019
 *      Author: schmidt
 */

#ifndef KRONECKERSVD_NEW_H_
#define KRONECKERSVD_NEW_H_

#include <deal.II/base/table.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include "alignedlinalg.h"
#include "kroneckersvd.h"

using namespace dealii;



/*
  Compute the low Kronecker rank approximation, i.e. the ksvd, of full Matrix M.
  We first reshuffle M and then compute the first few singular
  values/vectors by using the Lanczos algorithm. The matricization of these
  singular vectors then is the low Kronecker rank approximation. The matrix M is
  passed in "in", and the low rank approximation is passed in "out"
  to be consistent with dealII kroneckerproducts are passed as {B_i,A_i}
*/
template<typename Number>
void
compute_ksvd_new(const Table<2, Number> &                       in,
                 std::vector<std::array<Table<2, Number>, 2>> & out,
                 std::size_t                                    lanczos_iterations = -1)
{
  std::size_t out_rank = out.size();
  if(lanczos_iterations == (std::size_t)-1)
  {
    lanczos_iterations = out_rank * out_rank + 10;
  }

  std::size_t           big_m        = out[0][1].size()[0];
  std::size_t           big_n        = out[0][1].size()[1];
  std::size_t           small_m      = out[0][0].size()[0];
  std::size_t           small_n      = out[0][0].size()[1];
  Table<2, Number>      reshuffledIn = reshuffle_full(in, big_m, big_n, small_m, small_n);
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
    r.back() = vector_addition(r.back(), matrix_vector_multiplication(reshuffledIn, v.back()));
    orthogonalize(r);
    alpha.push_back(std::sqrt(inner_product(r.back(), r.back())));
    u.push_back(vector_inverse_scaling(r.back(), alpha.back()));
    p.push_back(vector_scaling(v.back(), -alpha.back()));
    p.back() =
      vector_addition(p.back(), matrix_transpose_vector_multiplication(reshuffledIn, u.back()));
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

  /// TODO mismatch between out_rank and base_len
  for(std::size_t i = 0; i < out_rank; i++)
  {
    for(std::size_t k = 0; k < big_m; k++)
      for(std::size_t l = 0; l < big_n; l++)
        out[i][1](k, l) = left_singular_vectors(i, k * big_n + l) * std::sqrt(singular_values[i]);
    for(std::size_t k = 0; k < small_m; k++)
      for(std::size_t l = 0; l < small_n; l++)
        out[i][0](k, l) =
          right_singular_vectors(i, k * small_n + l) * std::sqrt(singular_values[i]);
  }
}



/*
  Compute the low Kronecker rank approximation, i.e. the ksvd, of a matrix of
  the form M = Σ A_i⊗B_i where A_i is called the big matrix and B_i is called
  the small matrix. We first reshuffle M and then compute the first few singular
  values/vectors by using the Lanczos algorithm. The matricization of these
  singular vectors then is the low Kronecker rank approximation. The matrix M is
  passed in "in", and the low rank approximation is passed in "out"
  to be consistent with dealII kroneckerproducts are passed as {B_i,A_i}
*/
template<typename Number>
void
compute_ksvd_new(const std::vector<std::array<Table<2, Number>, 2>> & in,
                 std::vector<std::array<Table<2, Number>, 2>> &       out,
                 std::size_t                                          lanczos_iterations = -1)
{
  std::size_t in_rank  = in.size();
  std::size_t out_rank = out.size();
  if(lanczos_iterations == (std::size_t)-1)
  {
    lanczos_iterations = out_rank * out_rank + 10;
  }

  std::size_t                        big_m   = in[0][1].size()[0];
  std::size_t                        big_n   = in[0][1].size()[1];
  std::size_t                        small_m = in[0][0].size()[0];
  std::size_t                        small_n = in[0][0].size()[1];
  std::vector<AlignedVector<Number>> big_matrices_vectorized;
  std::vector<AlignedVector<Number>> small_matrices_vectorized;
  for(std::size_t i = 0; i < in_rank; i++)
  {
    small_matrices_vectorized.push_back(vectorize_matrix(in[i][0]));
    big_matrices_vectorized.push_back(vectorize_matrix(in[i][1]));
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

  /// TODO mismatch between out_rank and base_len
  AssertThrow(out_rank <= base_len,
              ExcMessage("TODO base_len determines the maximal Kronecker rank? ask Simon!"));
  for(std::size_t i = 0; i < out_rank; i++)
  {
    for(std::size_t k = 0; k < big_m; k++)
      for(std::size_t l = 0; l < big_n; l++)
        out[i][1](k, l) = left_singular_vectors(i, k * big_n + l) * std::sqrt(singular_values[i]);
    for(std::size_t k = 0; k < small_m; k++)
      for(std::size_t l = 0; l < small_n; l++)
        out[i][0](k, l) =
          right_singular_vectors(i, k * small_n + l) * std::sqrt(singular_values[i]);
  }
}


/*
  Compute the low Kronecker rank approximation, i.e. the ksvd, of a diagonal matrix.
  We compute the first few singular
  values/vectors by using the Lanczos algorithm. The matricization of these
  singular vectors then is the low Kronecker rank approximation. The matrix M is
  passed in "in", and the low rank approximation is passed in "out"
  to be consistent with dealII kroneckerproducts are passed as {B_i,A_i}
*/

template<typename Number>
void
compute_ksvd_new(const AlignedVector<Number> &                  in,
                 std::vector<std::array<Table<2, Number>, 2>> & out,
                 std::size_t                                    lanczos_iterations = -1)
{
  std::size_t out_rank = out.size();
  if(lanczos_iterations == (std::size_t)-1)
  {
    lanczos_iterations = out_rank * out_rank + 10;
  }

  std::size_t           big_m   = out[0][1].size()[0];
  std::size_t           small_m = out[0][0].size()[0];
  AlignedVector<Number> beta;
  beta.push_back(Number(1)); // we artificially introduce a first value for
  // beta to define beta.back()
  AlignedVector<Number>              alpha;
  std::vector<AlignedVector<Number>> r;
  std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(small_m * small_m)};
  p.back()[0]                          = Number(1);
  std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(big_m * big_m)};
  std::vector<AlignedVector<Number>> v;
  for(std::size_t i = 0;
      i < lanczos_iterations && std::abs(beta.back()) > std::numeric_limits<Number>::epsilon();
      i++)
  {
    beta.push_back(std::sqrt(inner_product(p.back(), p.back())));
    v.push_back(vector_inverse_scaling(p.back(), beta.back()));
    r.push_back(vector_scaling(u.back(), -beta.back()));
    r.back() = vector_addition(r.back(),
                               reshuffled_diag_vector_multiplication(in, v.back(), big_m, small_m));
    orthogonalize(r);
    alpha.push_back(std::sqrt(inner_product(r.back(), r.back())));
    u.push_back(vector_inverse_scaling(r.back(), alpha.back()));
    p.push_back(vector_scaling(v.back(), -alpha.back()));
    p.back() = vector_addition(
      p.back(), transpose_reshuffled_diag_vector_multiplication(in, u.back(), big_m, small_m));
    orthogonalize(p);
  }
  std::size_t           base_len = alpha.size() - 1;
  Table<2, Number>      U(base_len, big_m * big_m); // discard first value of u since it is zero
  Table<2, Number>      V(base_len,
                     small_m * small_m); // discard last value of v since it is zero
  AlignedVector<Number> real_beta(base_len -
                                  1); // discard first two values of beta, first is artificially
  // introduced, second only depends on inital guess
  AlignedVector<Number> real_alpha(base_len); // discard last value of alpha since it is zero
  for(std::size_t i = 0; i < base_len; i++)
  {
    real_alpha[i] = alpha[i];
    if(i < base_len - 1)
      real_beta[i] = beta[i + 2];
    for(std::size_t j = 0; j < big_m * big_m; j++)
      U(i, j) = u[i + 1][j];
    for(std::size_t j = 0; j < small_m * small_m; j++)
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
      for(std::size_t l = 0; l < big_m; l++)
        out[i][1](k, l) = left_singular_vectors(i, k * big_m + l) * std::sqrt(singular_values[i]);
    for(std::size_t k = 0; k < small_m; k++)
      for(std::size_t l = 0; l < small_m; l++)
        out[i][0](k, l) =
          right_singular_vectors(i, k * small_m + l) * std::sqrt(singular_values[i]);
  }
}

#endif
