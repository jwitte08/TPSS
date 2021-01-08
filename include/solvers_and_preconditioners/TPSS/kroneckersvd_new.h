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



namespace intern
{
template<typename Number>
void
compute_bidiagonal_svd(const AlignedVector<Number> & diagonal,
                       const AlignedVector<Number> & super_diagonal,
                       Table<2, Number> &            U,
                       AlignedVector<Number> &       singular_values,
                       Table<2, Number> &            VT)
{
  AssertDimension(diagonal.size(), super_diagonal.size() + 1);
  AssertIndexRange(singular_values.size(), diagonal.size() + 1);
  AssertDimension(diagonal.size(), U.size(0));
  AssertDimension(singular_values.size(), U.size(1));
  AssertDimension(diagonal.size(), VT.size(1));
  AssertDimension(singular_values.size(), VT.size(0));

  using scalar_value_type = typename ExtractScalarType<Number>::type;
  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    LAPACKFullMatrix<scalar_value_type> bidiagonal_matrix(diagonal.size());
    for(auto i = 0U; i < diagonal.size(); ++i)
      bidiagonal_matrix(i, i) = scalar_value(diagonal[i], lane);
    for(auto i = 0U; i < super_diagonal.size(); ++i)
      bidiagonal_matrix(i, i + 1) = scalar_value(super_diagonal[i], lane);

    /// TODO compute only the singular_values.size() first singular values and
    /// not the complete SVD
    const auto & [tildeU, Sigma, tildeVT] = compute_svd(bidiagonal_matrix);

    for(auto i = 0U; i < U.size(0); ++i)
      for(auto j = 0U; j < U.size(1); ++j)
        scalar_value(U(i, j), lane) = tildeU(i, j);

    for(auto i = 0U; i < VT.size(0); ++i)
      for(auto j = 0U; j < VT.size(1); ++j)
        scalar_value(VT(i, j), lane) = tildeVT(i, j);

    for(auto i = 0U; i < singular_values.size(); ++i)
      scalar_value(singular_values[i], lane) = Sigma(i, i);
  }
}

} // namespace intern

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


  // {
  //   const auto & [B, A] = src.front();
  //   std::cout << "A" << std::endl;
  //   table_to_fullmatrix(A).print_formatted(std::cout);
  //   std::cout << "vectA" << std::endl;
  //   table_to_fullmatrix(LinAlg::vect(A)).print_formatted(std::cout);
  //   std::cout << "folding vectA" << std::endl;
  //   table_to_fullmatrix(LinAlg::folding(LinAlg::vect(A),A.size(0),A.size(1))).print_formatted(std::cout);
  // }


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
  beta.push_back(Number(1.));

  AlignedVector<Number> alpha;

  std::vector<AlignedVector<Number>> r;

  std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(m_B * n_B)};
  p.back()[0]                          = Number(1.);

  std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(m_A * n_A)};

  std::vector<AlignedVector<Number>> v;

  Assert(has_nearly_zero_abs(LinAlg::euclidean_norm(p.back()) - (Number)1.),
         ExcMessage("Euclidean norm of p_0 must be one!"));
  for(std::size_t i = 0; i < lanczos_iterations && !has_nearly_zero_abs(std::abs(beta.back())); ++i)
  {
    /// beta_i = || p_i ||_2
    beta.push_back(LinAlg::euclidean_norm(p.back()));
    std::cout << "beta_" << i << ": " << varray_to_string(beta.back()) << std::endl;
    std::cout << std::boolalpha << "is nearly zero: " << has_nearly_zero_abs(std::abs(beta.back()))
              << std::endl;
    std::cout << std::boolalpha << "first lane is nearly zero: "
              << has_nearly_zero_abs(std::abs(scalar_value(beta.back(), 0))) << std::endl;

    /// v_i = p_i / beta_i
    v.push_back(LinAlg::inverse_scaling_if(p.back(), beta.back()));
    std::cout << "||v_" << i << "||: " << varray_to_string(LinAlg::euclidean_norm(v.back()))
              << std::endl;

    /// r_i = R(A) v_i - beta * u_i
    r.push_back(LinAlg::scaling(u.back(), -beta.back()));
    shuffled_matrix.vmult_add(r.back(), v.back());
    std::cout << "||r_" << i << "||: " << varray_to_string(LinAlg::euclidean_norm(r.back()))
              << std::endl;

    // orthogonalize(r);
    LinAlg::orthogonalize_full(r);

    std::cout << "||r_" << i << "||: " << varray_to_string(LinAlg::euclidean_norm(r.back()))
              << std::endl;

    /// alpha_i = || r_i ||_2
    alpha.push_back(LinAlg::euclidean_norm(r.back()));
    std::cout << "alpha_" << i << ": " << varray_to_string(alpha.back()) << std::endl;
    std::cout << std::boolalpha << "is nearly zero: " << has_nearly_zero_abs(std::abs(alpha.back()))
              << std::endl;
    std::cout << std::boolalpha << "first lane is nearly zero: "
              << has_nearly_zero_abs(std::abs(scalar_value(alpha.back(), 0))) << std::endl;

    /// u_{i+1} = r_i / alpha_i
    u.push_back(LinAlg::inverse_scaling_if(r.back(), alpha.back()));

    /// p_{i+1} = R(A)^T u_{i+1} - alpha_i * v_i
    p.push_back(LinAlg::scaling(v.back(), -alpha.back()));
    shuffled_matrix.Tvmult_add(p.back(), u.back());

    // orthogonalize(p);
    LinAlg::orthogonalize_full(p);
  }

  std::size_t base_len = alpha.size() - 1;

  // discard first value of u since it is zero
  Table<2, Number> U(base_len, m_A * n_A);

  // discard last value of v since it is zero
  Table<2, Number> V(base_len, m_B * n_B);

  // discard first two values of beta, first is artificially
  // introduced, second only depends on inital guess
  AlignedVector<Number> real_beta(base_len - 1);

  // discard last value of alpha since it is zero
  AlignedVector<Number> real_alpha(base_len);

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

  // bidiagonal_svd(real_alpha, real_beta, tildeU, singular_values, tildeVT);
  intern::compute_bidiagonal_svd(real_alpha, real_beta, tildeU, singular_values, tildeVT);

  Table<2, Number> left_singular_vectors  = matrix_transpose_multiplication(tildeU, U);
  Table<2, Number> right_singular_vectors = matrix_multiplication(tildeVT, V);

  /// TODO mismatch between tensor_rank_dst and base_len
  AssertIndexRange(tensor_rank_dst, base_len + 1);
  AssertThrow(tensor_rank_dst <= base_len,
              ExcMessage("TODO base_len determines the maximal Kronecker rank? ask Simon!"));

  // for(std::size_t i = 0; i < tensor_rank_dst; i++)
  // {
  //   for(std::size_t k = 0; k < m_A; k++)
  //     for(std::size_t l = 0; l < n_A; l++)
  //       dst[i][1](k, l) = left_singular_vectors(i, k * n_A + l) * std::sqrt(singular_values[i]);
  //   for(std::size_t k = 0; k < m_B; k++)
  //     for(std::size_t l = 0; l < n_B; l++)
  //       dst[i][0](k, l) = right_singular_vectors(i, k * n_B + l) * std::sqrt(singular_values[i]);
  // }

  /// NOTE U is actually UT
  const auto & left = Tensors::Tmmult(U, tildeU);
  // table_to_fullmatrix(std::sqrt(singular_values[0]) *
  // left_singular_vectors).print_formatted(std::cout); std::cout << std::endl;
  // table_to_fullmatrix(std::sqrt(singular_values[0]) * left).print_formatted(std::cout);
  // std::cout << std::endl;

  /// NOTE V is actually VT
  const auto & rightT = Tensors::mmult(tildeVT, V);
  // table_to_fullmatrix(std::sqrt(singular_values[0]) *
  // right_singular_vectors).print_formatted(std::cout); std::cout << "rsv" << std::endl;
  // table_to_fullmatrix(std::sqrt(singular_values[0]) * rightT).print_formatted(std::cout);
  // std::cout << "rightT" << std::endl;
  const auto & right = Tensors::transpose(rightT);

  for(std::size_t i = 0; i < std::min(tensor_rank_dst, singular_values.size()); ++i)
  {
    // table_to_fullmatrix(dst[i][1]).print_formatted(std::cout);
    // std::cout << i << std::endl;
    dst[i][1] = LinAlg::sfolding_impl(
      left, dst[i][1].size(0), dst[i][1].size(1), std::sqrt(singular_values[i]), i);
    // table_to_fullmatrix(dst[i][1]).print_formatted(std::cout);
    // std::cout << i << std::endl;

    // table_to_fullmatrix(dst[i][0]).print_formatted(std::cout);
    // std::cout << "right" << i << std::endl;
    dst[i][0] = LinAlg::sfolding_impl(
      right, dst[i][0].size(0), dst[i][0].size(1), std::sqrt(singular_values[i]), i);
    // table_to_fullmatrix(dst[i][0]).print_formatted(std::cout);
    // std::cout << "right" << i << std::endl;
  }


  // {
  //   const auto lane = 0U;
  //   const auto myU = table_to_fullmatrix(Tensors::Tmmult(U, tildeU),lane);
  //   const auto myV = table_to_fullmatrix(Tensors::transpose(Tensors::mmult(tildeVT, V)),lane);
  //   AssertDimension(myU.n(), myV.n());
  //   FullMatrix<typename ExtractScalarType<Number>::type> Sigma(singular_values.size());
  //   for (auto i = 0U; i < singular_values.size(); ++i)
  //     Sigma(i,i) = scalar_value(singular_values[i],lane);
  //   std::cout << "U Sigma VT" << std::endl;
  //   merge_decomposition(myU, Sigma, myV).print_formatted(std::cout);
  //   std::cout << "shuffled matrix" << std::endl;
  //   table_to_fullmatrix(shuffled_matrix.as_table(),lane).print_formatted(std::cout);
  // }
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
