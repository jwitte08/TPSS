/*
 * kroneckersvd.h
 *
 *  Created on: Nov 14, 2019
 *      Author: schmidt, witte
 */

#ifndef KRONECKERSVD_H_
#define KRONECKERSVD_H_

#include <deal.II/base/table.h>
#include <deal.II/lac/lapack_full_matrix.h>


#include "alignedlinalg.h"
#include "kroneckersvd.h"
#include "matrix_helper.h"
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



template<typename Number, typename ShuffledMatrixType>
AlignedVector<Number>
compute_ksvd_impl(std::vector<std::array<Table<2, Number>, 2>> & dst,
                  const ShuffledMatrixType &                     shuffled_matrix,
                  const std::size_t                              lanczos_iterations_in = -1,
                  const std::size_t                              max_tensor_rank_guess = -1)
{
  /// TODO in special cases we already have a guess of the maximal tensor rank!
  /// can we exploit this information?!
  AssertDimension(max_tensor_rank_guess, static_cast<std::size_t>(-1));

  const std::size_t demanded_ksvd_rank = dst.size();
  const auto        max_matrix_rank_shuffled =
    std::min<std::size_t>(shuffled_matrix.m(), shuffled_matrix.n());
  const auto max_demanded_ksvd_rank =
    std::min<std::size_t>(max_tensor_rank_guess, max_matrix_rank_shuffled);

  (void)max_demanded_ksvd_rank;
  AssertIndexRange(demanded_ksvd_rank, max_demanded_ksvd_rank + 1);

  /// TODO tests have shown that the actual tensor rank of the inserted matrix
  /// plus one leads to exact KSVD approximations... does it hold in general?!
  const std::size_t lanczos_iterations =
    lanczos_iterations_in == static_cast<std::size_t>(-1) ?
      std::min<std::size_t>(
        /*Simon: demanded_ksvd_rank * demanded_ksvd_rank + 10*/ demanded_ksvd_rank + 10,
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
  AssertIndexRange(demanded_ksvd_rank, alpha.size() + 1);

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

  AlignedVector<Number> singular_values(demanded_ksvd_rank);
  Table<2, Number>      tildeU(alpha.size(), demanded_ksvd_rank);
  Table<2, Number>      tildeVT(demanded_ksvd_rank, alpha.size());

  intern::compute_bidiagonal_svd(alpha_view, beta_view, tildeU, singular_values, tildeVT);

  const auto & left_singular_vectors   = LinAlg::product(U, tildeU);
  const auto & right_singular_vectorsT = LinAlg::product(tildeVT, VT);

  /// fill tensors of matrices of the form SUM_i A_i (x) B_i
  const auto actual_ksvd_rank = std::min<std::size_t>(demanded_ksvd_rank, singular_values.size());
  dst.resize(actual_ksvd_rank);
  dst.shrink_to_fit();
  for(std::size_t i = 0; i < actual_ksvd_rank; ++i)
  {
    auto & [B_i, A_i] = dst[i];
    /// A_i are stored at first and B_i at zeroth position
    A_i = LinAlg::sfolding_impl(
      left_singular_vectors, A_i.size(0), A_i.size(1), std::sqrt(singular_values[i]), i);
    B_i = LinAlg::sfoldingT_impl(
      right_singular_vectorsT, B_i.size(0), B_i.size(1), std::sqrt(singular_values[i]), i);
  }

  return singular_values;
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
AlignedVector<Number>
compute_ksvd(const std::vector<std::array<Table<2, Number>, 2>> & src,
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
                   /// due to conventions the rightmost Kronecker factor is
                   /// inserted first...
                   return {LinAlg::Tvect(B), LinAlg::vect(A)};
                 });
  Tensors::TensorProductMatrix<2, Number> shuffled_matrix(rank1_tensors_shuffled);

  /// TODO...
  // const std::size_t max_tensor_rank_src = src.size();

  return compute_ksvd_impl<Number, Tensors::TensorProductMatrix<2, Number>>(dst,
                                                                     shuffled_matrix,
                                                                     lanczos_iterations_in/*
                                                                     ,max_tensor_rank_src*/);
}



/**
 * Same as above except that a plain matrix @p src is passed instead of a tensor
 * product matrix. The required matrix "reshuffling" is detailed in Van Loan and
 * Pitsianis and implemented here in MatrixAsTable::shuffle().
 */
template<typename Number>
AlignedVector<Number>
compute_ksvd(const Table<2, Number> &                       src,
             std::vector<std::array<Table<2, Number>, 2>> & dst,
             const std::size_t                              lanczos_iterations_in = -1)
{
  Assert(!dst.empty(), ExcMessage("dst is empty."));
  const auto & [B_0, A_0] = dst.front();
  AssertDimension(A_0.size(0) * B_0.size(0), src.size(0));
  AssertDimension(A_0.size(1) * B_0.size(1), src.size(1));

  MatrixAsTable<Number> shuffled_matrix;
  shuffled_matrix.as_table() = src;
  shuffled_matrix.shuffle({A_0.size(0), A_0.size(1)}, {B_0.size(0), B_0.size(1)});

  return compute_ksvd_impl<Number, MatrixAsTable<Number>>(dst,
                                                          shuffled_matrix,
                                                          lanczos_iterations_in);
}



/**
 * Same as above except that a diagonal matrix @p src is passed instead of a
 * tensor product matrix. This reduces the computational effort of
 * matrix-vector-multiplatications with the shuffled matrix (or its
 * transpose). Internally DiagonalMatrixWrap is used allowing for vectorized
 * arithmetic type @p Number.
 *
 * Note that only squared diagonal matrices are supported so far.
 */
template<typename Number>
AlignedVector<Number>
compute_ksvd(const AlignedVector<Number> &                  diagonal,
             std::vector<std::array<Table<2, Number>, 2>> & dst,
             const std::size_t                              lanczos_iterations_in = -1)
{
  Assert(!dst.empty(), ExcMessage("dst is empty."));
  const auto & [B_0, A_0] = dst.front();
  AssertDimension(A_0.size(0) * B_0.size(0), diagonal.size());
  AssertDimension(A_0.size(1) * B_0.size(1), diagonal.size());

  DiagonalMatrixWrap<Number> shuffled_matrix;
  shuffled_matrix.reinit(diagonal);
  shuffled_matrix.shuffle({A_0.size(0), A_0.size(1)}, {B_0.size(0), B_0.size(1)});

  return compute_ksvd_impl<Number, DiagonalMatrixWrap<Number>>(dst,
                                                               shuffled_matrix,
                                                               lanczos_iterations_in);
}



/**
 * Compute the low Kronecker rank approximation, we use the ALS decomposition to
 * get a Candecomp/Parafac decomposition, of a matrix of the form M = Σ
 * A_i⊗B_i⊗C_i. We first reshuffle M and then compute the first few singular
 * values/vectors by using the Lanczos algorithm. The matricization of these
 * singular vectors then is the low Kronecker rank approximation. The matrix M
 * is passed in "in", and the low rank approximation is passed in "out" to be
 * consistent with dealII kroneckerproducts are passed as {C_i,B_i,A_i}
 */
template<typename Number>
void
compute_kcp(const std::vector<std::array<Table<2, Number>, 3>> & in,
            std::vector<std::array<Table<2, Number>, 3>> &       out,
            std::size_t                                          als_iterations = -1)
{
  AssertDimension(in[0][0].size()[0], out[0][0].size()[0]);
  AssertDimension(in[0][1].size()[0], out[0][1].size()[0]);
  AssertDimension(in[0][2].size()[0], out[0][2].size()[0]);
  AssertDimension(in[0][0].size()[1], out[0][0].size()[1]);
  AssertDimension(in[0][1].size()[1], out[0][1].size()[1]);
  AssertDimension(in[0][2].size()[1], out[0][2].size()[1]);
  std::size_t in_rank = in.size();

  const std::size_t out_rank = out.size();
  if(als_iterations == (std::size_t)-1)
  {
    als_iterations = out_rank * out_rank + 10;
  }
  for(std::size_t i = 0; i < out_rank; i++)
  {
    AssertDimension(out[0][0].size()[0], out[i][0].size()[0]);
    AssertDimension(out[0][1].size()[0], out[i][1].size()[0]);
    AssertDimension(out[0][2].size()[0], out[i][2].size()[0]);
    AssertDimension(out[0][0].size()[1], out[i][0].size()[1]);
    AssertDimension(out[0][1].size()[1], out[i][1].size()[1]);
    AssertDimension(out[0][2].size()[1], out[i][2].size()[1]);
  }
  std::vector<std::array<AlignedVector<Number>, 3>> matrices_vectorized;
  for(std::size_t i = 0; i < in_rank; i++)
  {
    AssertDimension(in[0][0].size()[0], in[i][0].size()[0]);
    AssertDimension(in[0][1].size()[0], in[i][1].size()[0]);
    AssertDimension(in[0][2].size()[0], in[i][2].size()[0]);
    AssertDimension(in[0][0].size()[1], in[i][0].size()[1]);
    AssertDimension(in[0][1].size()[1], in[i][1].size()[1]);
    AssertDimension(in[0][2].size()[1], in[i][2].size()[1]);
    std::array<AlignedVector<Number>, 3> matrix_vect = {vectorize_matrix(in[i][2]),
                                                        vectorize_matrix(in[i][1]),
                                                        vectorize_matrix(in[i][0])};
    matrices_vectorized.push_back(matrix_vect);
  }
  std::vector<Table<2, Number>> parafac_components;
  Table<2, Number>              A0(matrices_vectorized[0][0].size(), out_rank);
  Table<2, Number>              A1(matrices_vectorized[0][1].size(), out_rank);
  Table<2, Number>              A2(matrices_vectorized[0][2].size(), out_rank);
  for(std::size_t i = 0; i < out_rank; i++)
  {
    for(std::size_t j = 0; j < matrices_vectorized[0][0].size(); j++)
      A0(j, i) = matrices_vectorized[i][0][j];
    for(std::size_t j = 0; j < matrices_vectorized[0][1].size(); j++)
      A1(j, i) = matrices_vectorized[i][1][j];
    for(std::size_t j = 0; j < matrices_vectorized[0][2].size(); j++)
      A2(j, i) = matrices_vectorized[i][2][j];
  }
  Table<2, Number> V;
  for(std::size_t i = 0; i < als_iterations; i++)
  {
    V  = hadamard(LinAlg::Tproduct(A1, A1), LinAlg::Tproduct(A2, A2));
    A0 = LinAlg::product(unfold_rankk(matrices_vectorized, 0),
                         LinAlg::product(khatri_rao(A2, A1), pseudo_inverse(V)));
    V  = hadamard(LinAlg::Tproduct(A0, A0), LinAlg::Tproduct(A2, A2));
    A1 = LinAlg::product(unfold_rankk(matrices_vectorized, 1),
                         LinAlg::product(khatri_rao(A2, A0), pseudo_inverse(V)));
    V  = hadamard(LinAlg::Tproduct(A0, A0), LinAlg::Tproduct(A1, A1));
    A2 = LinAlg::product(unfold_rankk(matrices_vectorized, 2),
                         LinAlg::product(khatri_rao(A1, A0), pseudo_inverse(V)));
  }


  for(std::size_t i = 0; i < out_rank; i++)
  {
    for(std::size_t j = 0; j < out[i][0].size()[0]; j++)
      for(std::size_t k = 0; k < out[i][0].size()[1]; k++)
        out[i][0](j, k) = A2(j * out[i][0].size()[1] + k, i);

    for(std::size_t j = 0; j < out[i][1].size()[0]; j++)
      for(std::size_t k = 0; k < out[i][1].size()[1]; k++)
        out[i][1](j, k) = A1(j * out[i][1].size()[1] + k, i);

    for(std::size_t j = 0; j < out[i][2].size()[0]; j++)
      for(std::size_t k = 0; k < out[i][2].size()[1]; k++)
        out[i][2](j, k) = A0(j * out[i][2].size()[1] + k, i);
  }
}



#endif
