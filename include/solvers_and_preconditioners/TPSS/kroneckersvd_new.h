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
/**
 * Computes the (partial) singular value decomposition of a bidiagonal matrix
 * defined by @p diagonal and @p super_diagonal. Singular values are passed in
 * @p singular_values. Left singular vectors are juxtaposed column-by-column in
 * @p U and right singular vectors are stacked row-by-row in @p VT (explaining
 * the "T" in "VT" as transpose of matrix V).
 *
 * When @p U, @p singular_values and @p VT are passed empty they will be resized
 * to the length of @p diagonal and the COMPLETE SVD will be computed. By
 * passing those fields with compatible sizes only a PARTIAL SVD is computed
 * with as many singular values as the size of @p singular_values.
 */
template<typename Number>
void
compute_bidiagonal_svd(const ArrayView<const Number> & diagonal,
                       const ArrayView<const Number> & super_diagonal,
                       Table<2, Number> &              U,
                       AlignedVector<Number> &         singular_values,
                       Table<2, Number> &              VT)
{
  AssertDimension(diagonal.size(), super_diagonal.size() + 1);

  /// resize data if empty -> computes complete SVD
  const bool svd_data_fields_are_empty = U.empty() && singular_values.empty() && VT.empty();
  if(svd_data_fields_are_empty)
  {
    U.reinit(diagonal.size(), diagonal.size());
    singular_values.resize(diagonal.size());
    VT.reinit(diagonal.size(), diagonal.size());
  }

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

  // for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  //   std::cout << lane << " | " << vector_to_string(alignedvector_to_vector(singular_values, lane))
  //             << std::endl;
  Assert(std::is_sorted(singular_values.begin(),
                        singular_values.end(),
                        [](const auto & lhs, const auto & rhs) {
                          return less_than(lhs, rhs).none();
                        }),
         ExcMessage("Singular values are not sorted from large to small."));
}



template<typename Number>
void
compute_bidiagonal_svd(const AlignedVector<Number> & diagonal,
                       const AlignedVector<Number> & super_diagonal,
                       Table<2, Number> &            U,
                       AlignedVector<Number> &       singular_values,
                       Table<2, Number> &            VT)
{
  const ArrayView<const Number> diagonal_view(diagonal.data(), diagonal.size());
  const ArrayView<const Number> super_diagonal_view(super_diagonal.data(), super_diagonal.size());
  compute_bidiagonal_svd<Number>(diagonal_view, super_diagonal_view, U, singular_values, VT);
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



/**
 * Computes a low Kronecker rank approximation of the tensor product matrix @p
 * src defined by
 *
 *    src = SUM_i A_i (x) B_i,
 *
 * returning a tensor of matrices @p dst. The low rank decomposition is obtained
 * by the so-called Kronecker SVD (KSVD) and of the form
 *
 *    dst = SUM_i^r U_i (x) V_i.
 *
 * The tensor rank r is fixed by the size of @p dst and the sizes of U_i and V_i
 * are fixed by the elements of @p dst (it's the user's responsibility to pass a
 * tensor of matrices @p dst of appropriate size.
 *
 * Details on a Kronecker SVD are provided by Van Loan and Pitsianis: First
 * reshuffle the matrix @ src, then compute r many singular values of the
 * reshuffled matrix by means of a Lanczos biorthogonalization. Finally, fold
 * both left and right singular vectors scaled (each scaled by the square of
 * corresponding singular value) into the matrices U_i and V_i, respectively. To
 * be consistent with deal.II indexing (in particular TensorProductMatrix)
 * tensors of matrices are passed as {B_i, A_i} and {V_i, U_i}, respectively.
 */
template<typename Number>
void
compute_ksvd_new(const std::vector<std::array<Table<2, Number>, 2>> & src,
                 std::vector<std::array<Table<2, Number>, 2>> &       dst,
                 const std::size_t                                    lanczos_iterations_in = -1)
{
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
  const std::size_t tensor_rank_dst     = dst.size();
  const auto        max_matrix_rank_shuffled =
    std::min<std::size_t>(shuffled_matrix.m(), shuffled_matrix.n());
  const auto max_tensor_rank_dst =
    std::min<std::size_t>(max_tensor_rank_src, max_matrix_rank_shuffled);
  AssertIndexRange(tensor_rank_dst, max_tensor_rank_dst + 1);

  const std::size_t lanczos_iterations =
    lanczos_iterations_in == static_cast<std::size_t>(-1) ?
      std::min<std::size_t>(/*Simon: tensor_rank_dst * tensor_rank_dst + 10*/ tensor_rank_dst + 10,
                            max_matrix_rank_shuffled) :
      lanczos_iterations_in;
  AssertIndexRange(lanczos_iterations, max_matrix_rank_shuffled + 1);

  std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(shuffled_matrix.n())};
  p.back()[0]                          = Number(1.);
  AlignedVector<Number>              beta(1U, static_cast<Number>(1.));
  std::vector<AlignedVector<Number>> v;

  std::vector<AlignedVector<Number>> r;
  AlignedVector<Number>              alpha;
  std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(shuffled_matrix.m())};

  Assert(has_nearly_zero_abs(LinAlg::euclidean_norm(p.back()) - (Number)1.),
         ExcMessage("Euclidean norm of p_0 must be one!"));
  for(std::size_t i = 0; i < lanczos_iterations /*&& !has_nearly_zero_abs(std::abs(beta.back()))*/;
      ++i)
  {
    /// v_i = p_i / beta_i
    v.push_back(LinAlg::inverse_scaling_if(p.back(), beta.back()));
    // std::cout << "||v_" << i << "||: " << varray_to_string(LinAlg::euclidean_norm(v.back()))
    //           << std::endl;

    /// r_i = R(A) v_i - beta * u_i
    r.push_back(LinAlg::scaling(u.back(), -beta.back()));
    shuffled_matrix.vmult_add(r.back(), v.back());
    // std::cout << "||r_" << i << "||: " << varray_to_string(LinAlg::euclidean_norm(r.back()))
    //           << std::endl;

    // orthogonalize(r);
    LinAlg::orthogonalize_full(r);
    // std::cout << "||r_" << i << "||: " << varray_to_string(LinAlg::euclidean_norm(r.back()))
    //           << std::endl;

    /// alpha_i = || r_i ||_2
    alpha.push_back(LinAlg::euclidean_norm(r.back()));
    // std::cout << "alpha_" << i << ": " << varray_to_string(alpha.back()) << std::endl;

    /// u_{i+1} = r_i / alpha_i
    u.push_back(LinAlg::inverse_scaling_if(r.back(), alpha.back()));
    // std::cout << "||u_" << i + 1 << "||: " << varray_to_string(LinAlg::euclidean_norm(u.back()))
    //           << std::endl;

    /// p_{i+1} = R(A)^T u_{i+1} - alpha_i * v_i
    p.push_back(LinAlg::scaling(v.back(), -alpha.back()));
    shuffled_matrix.Tvmult_add(p.back(), u.back());

    // orthogonalize(p);
    LinAlg::orthogonalize_full(p);

    /// beta_{i+1} = || p_{i+1} ||_2
    beta.push_back(LinAlg::euclidean_norm(p.back()));
    // std::cout << "beta_" << i + 1 << ": " << varray_to_string(beta.back()) << std::endl;

    if(has_nearly_zero_abs(std::abs(beta.back())))
      break;
  }

  AssertDimension(u.size(), alpha.size() + 1);
  AssertDimension(v.size(), alpha.size());
  AssertDimension(alpha.size() + 1, beta.size());

  /// for safety it's the user's responsibility to pass a reasonable tensor rank
  AssertIndexRange(tensor_rank_dst, alpha.size() + 1);

  /// discard u_0 which is zero (initial guess)
  Table<2, Number> U(shuffled_matrix.m(), alpha.size());
  for(std::size_t i = 0; i < shuffled_matrix.m(); ++i)
    for(std::size_t j = 0; j < alpha.size(); ++j)
      U(i, j) = u[j + 1][i];

  Table<2, Number> VT(alpha.size(), shuffled_matrix.n());
  for(std::size_t i = 0; i < alpha.size(); ++i)
    for(std::size_t j = 0; j < shuffled_matrix.n(); ++j)
      VT(i, j) = v[i][j];

  const ArrayView<const Number> alpha_view(alpha.data(), alpha.size());

  /// discard the first value of beta (initial guess) and the last value (nearly zero)
  const ArrayView<const Number> beta_view(beta.data() + 1, alpha.size() - 1);

  AlignedVector<Number> singular_values(tensor_rank_dst);
  Table<2, Number>      tildeU(alpha.size(), tensor_rank_dst);
  Table<2, Number>      tildeVT(tensor_rank_dst, alpha.size());

  intern::compute_bidiagonal_svd(alpha_view, beta_view, tildeU, singular_values, tildeVT);

  const auto & left_singular_vectors   = Tensors::mmult(U, tildeU);
  const auto & right_singular_vectorsT = Tensors::mmult(tildeVT, VT);

  /// fill tensors of matrices of the form SUM_i A_i (x) B_i
  for(std::size_t i = 0; i < std::min(tensor_rank_dst, singular_values.size()); ++i)
  {
    auto & [B_i, A_i] = dst[i];
    /// A_i are stored at first and B_i at zeroth position
    A_i = LinAlg::sfolding_impl(
      left_singular_vectors, A_i.size(0), A_i.size(1), std::sqrt(singular_values[i]), i);
    B_i = LinAlg::sfoldingT_impl(
      right_singular_vectorsT, B_i.size(0), B_i.size(1), std::sqrt(singular_values[i]), i);
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
