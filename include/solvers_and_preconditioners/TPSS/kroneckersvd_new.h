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
#include "tensor_product_matrix.h"

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
    // v.push_back(vector_inverse_scaling(p.back(), beta.back()));
    v.push_back(LinAlg::inverse_scaling_if(p.back(), beta.back()));
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
compute_ksvd_new(const std::vector<std::array<Table<2, Number>, 2>> & src,
                 std::vector<std::array<Table<2, Number>, 2>> &       dst,
                 const std::size_t                                    lanczos_iterations_in = -1)
{
  std::cout << "tensor path" << std::endl;

  const std::size_t m_B = src.front()[0].size(0);
  const std::size_t n_B = src.front()[0].size(1);
  const std::size_t m_A = src.front()[1].size(0);
  const std::size_t n_A = src.front()[1].size(1);

  /// the shuffled matrix reads SUM_i vect(A_i) (x) vect(B_i)^T
  std::vector<std::array<Table<2, Number>, 2>> rank1_tensors_shuffled;
  std::transform(src.cbegin(),
                 src.cend(),
                 std::back_inserter(rank1_tensors_shuffled),
                 [&](const auto & tensor) -> std::array<Table<2, Number>, 2> {
                   const auto & [B, A] = tensor;
                   return {LinAlg::Tvect(B), LinAlg::vect(A)};
                 });
  Tensors::TensorProductMatrix<2, Number> shuffled_matrix(rank1_tensors_shuffled);

  const std::size_t max_tensor_rank_src = src.size();
  std::size_t       tensor_rank_dst     = dst.size();
  const auto        max_tensor_rank_dst =
    std::min<std::size_t>(max_tensor_rank_src,
                          std::min<std::size_t>(shuffled_matrix.m(), shuffled_matrix.n()));
  AssertIndexRange(tensor_rank_dst, max_tensor_rank_dst + 1);

  const std::size_t lanczos_iterations = lanczos_iterations_in == static_cast<std::size_t>(-1) ?
                                           tensor_rank_dst * tensor_rank_dst + 10 :
                                           lanczos_iterations_in;
  /// TODO Number of iterations should not exceed maximal tensor rank !!!
  // const std::size_t lanczos_iterations =
  //   lanczos_iterations_in == static_cast<std::size_t>(-1) ?
  //     std::min<std::size_t>(tensor_rank_dst * tensor_rank_dst + 10, max_tensor_rank_dst) :
  //     lanczos_iterations_in;
  // AssertIndexRange(lanczos_iterations, max_tensor_rank_dst + 1);

  AlignedVector<Number> beta;
  // we artificially introduce a first value for
  // beta to define beta.back()
  beta.push_back(Number(1));

  AlignedVector<Number> alpha;

  std::vector<AlignedVector<Number>> r;

  std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(m_B * n_B)};
  p.back()[0]                          = Number(1);

  std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(m_A * n_A)};

  std::vector<AlignedVector<Number>> v;

  for(std::size_t i = 0;
      i < lanczos_iterations && std::abs(beta.back()) > std::numeric_limits<Number>::epsilon();
      i++)
  {
    /// beta_i = || p_i ||_2
    beta.push_back(LinAlg::euclidean_norm(p.back()));

    /// v_i = p_i / beta_i
    v.push_back(LinAlg::inverse_scaling_if(p.back(), beta.back()));

    /// r_i = R(A) v_i - beta * u_i
    r.push_back(LinAlg::scaling(u.back(), -beta.back()));
    shuffled_matrix.vmult_add(r.back(), v.back());

    orthogonalize(r);

    /// alpha_i = || r_i ||_2
    alpha.push_back(LinAlg::euclidean_norm(r.back()));

    /// u_{i+1} = r_i / alpha_i
    u.push_back(LinAlg::inverse_scaling_if(r.back(), alpha.back()));

    /// p_{i+1} = R(A)^T u_{i+1} - alpha_i * v_i
    p.push_back(LinAlg::scaling(v.back(), -alpha.back()));
    shuffled_matrix.Tvmult_add(p.back(), u.back());

    orthogonalize(p);
  }

  std::size_t base_len = alpha.size() - 1;

  Table<2, Number>      U(base_len, m_A * n_A); // discard first value of u since it is zero
  Table<2, Number>      V(base_len,
                     m_B * n_B); // discard last value of v since it is zero
  AlignedVector<Number> real_beta(base_len -
                                  1); // discard first two values of beta, first is artificially
  // introduced, second only depends on inital guess
  AlignedVector<Number> real_alpha(base_len); // discard last value of alpha since it is zero
  for(std::size_t i = 0; i < base_len; i++)
  {
    real_alpha[i] = alpha[i];
    if(i < base_len - 1)
      real_beta[i] = beta[i + 2];
    for(std::size_t j = 0; j < m_A * n_A; j++)
      U(i, j) = u[i + 1][j];
    for(std::size_t j = 0; j < m_B * n_B; j++)
      V(i, j) = v[i][j];
  }

  AlignedVector<Number> singular_values(base_len);
  Table<2, Number>      tildeU(base_len, base_len);
  Table<2, Number>      tildeVT(base_len, base_len);

  bidiagonal_svd(real_alpha, real_beta, tildeU, singular_values, tildeVT);

  Table<2, Number> left_singular_vectors  = matrix_transpose_multiplication(tildeU, U);
  Table<2, Number> right_singular_vectors = matrix_multiplication(tildeVT, V);

  /// TODO mismatch between tensor_rank_dst and base_len
  AssertIndexRange(tensor_rank_dst, base_len + 2);
  AssertThrow(tensor_rank_dst <= base_len,
              ExcMessage("TODO base_len determines the maximal Kronecker rank? ask Simon!"));
  for(std::size_t i = 0; i < tensor_rank_dst; i++)
  {
    for(std::size_t k = 0; k < m_A; k++)
      for(std::size_t l = 0; l < n_A; l++)
        dst[i][1](k, l) = left_singular_vectors(i, k * n_A + l) * std::sqrt(singular_values[i]);
    for(std::size_t k = 0; k < m_B; k++)
      for(std::size_t l = 0; l < n_B; l++)
        dst[i][0](k, l) = right_singular_vectors(i, k * n_B + l) * std::sqrt(singular_values[i]);
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
  std::size_t tensor_rank_out = out.size();
  if(lanczos_iterations == (std::size_t)-1)
  {
    lanczos_iterations = tensor_rank_out * tensor_rank_out + 10;
  }

  std::size_t           big_m   = out[0][1].size(0);
  std::size_t           small_m = out[0][0].size(0);
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

  for(std::size_t i = 0; i < tensor_rank_out; i++)
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
