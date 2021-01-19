/*
 * tensor_product_matrix.h
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#ifndef TENSOR_PRODUCT_MATRIX_H_
#define TENSOR_PRODUCT_MATRIX_H_

#include <deal.II/lac/lapack_full_matrix.h>

#include "matrix_helper.h"
#include "move_to_deal_ii.h"
#include "tensors.h"

using namespace dealii;

namespace Tensors
{
namespace internal
{
/**
 * This struct is a helper class to compute the sum factorizations with
 * (an)isotropic tensor structure required by TensorProductMatrix. Using the
 * generic evaluator TensorProductEvaluator as kernel it adheres to the
 * deal.II convention that the matrix of dimension 0 is the rightmost
 * Kronecker factor. Furthermore, it facilitates static or dynamic loop bounds.
 *
 * The static path allows only for isotropic tensors of quadratic matrices.
 *
 * The dynamic path supports anisotropic tensors of matrices.
 */
template<int order, typename Number, int n_rows_1d>
struct Evaluator
  : public dealii::internal::My::EvaluatorTensorProduct<order, Number, n_rows_1d, n_rows_1d>
{
  using Base = dealii::internal::My::EvaluatorTensorProduct<order, Number, n_rows_1d, n_rows_1d>;

  Evaluator(const std::array<unsigned int, order> & n_rows_foreach_dimension,
            const std::array<unsigned int, order> & n_columns_foreach_dimension)
    : Base(n_rows_foreach_dimension[0], n_columns_foreach_dimension[0]),
      thelper_rows(n_rows_foreach_dimension),
      thelper_columns(n_columns_foreach_dimension)
  {
  }

  /**
   * Returns the correct collapsed mode sizes following the conventions of
   * TensorProductEvaluator. Furthermore, the number of rows and columns of the
   * 1D matrix of dimension @p direction are returned.
   */
  std::array<int, 4>
  anisotropic_indices(const int direction) const
  {
    const int n_pre     = thelper_columns.collapsed_size_pre(direction);
    const int n_post    = thelper_rows.collapsed_size_post(direction);
    const int n_rows    = thelper_rows.size(direction);
    const int n_columns = thelper_columns.size(direction);
    return {n_pre, n_post, n_rows, n_columns};
  };

  /**
   * Returns the maximum number of elements intermediate tensors have when a sum
   * factorization is applied following the conventions of
   * TensorProductEvaluator.
   */
  template<bool transpose>
  unsigned int
  max_intermediate_size() const
  {
    unsigned int max_size = 0;
    for(auto d = 0; d < order; ++d)
    {
      const auto [n_pre, n_post, n_rows, n_columns] = anisotropic_indices(d);
      const unsigned int n_elements_of_direction    = transpose ? n_columns : n_rows;
      const unsigned int size_tmp                   = n_pre * n_elements_of_direction * n_post;
      max_size                                      = std::max(max_size, size_tmp);
    }
    return max_size;
  }

  template<int direction, bool contract_over_rows, bool add>
  void
  apply(const Number * matrix,
        const Number * tensor_of_vectors_src,
        Number *       tensor_of_vectors_dst) const
  {
    const auto [n_pre, n_post, n_rows, n_columns] = anisotropic_indices(direction);
    Base::template apply<direction, contract_over_rows, add>(
      matrix, tensor_of_vectors_src, tensor_of_vectors_dst, n_pre, n_post, n_rows, n_columns);
  }

  TensorHelper<order> thelper_rows;
  TensorHelper<order> thelper_columns;
};

/// TODO static path using template parameter n_rows_1d
template<int order, typename Number, int n_rows_1d = -1>
struct ComputeGeneralizedEigendecomposition
{
  void
  operator()(std::array<AlignedVector<Number>, order> &               eigenvalues,
             std::array<Table<2, Number>, order> &                    eigenvectors,
             const std::vector<std::array<Table<2, Number>, order>> & rank1_tensors,
             const std::bitset<order>                                 B_mask)
  {
    AssertDimension(rank1_tensors.size(), 2U);
    for(auto dimension = 0; dimension < order; ++dimension)
    {
      const bool   B_index = B_mask[dimension];
      const auto & A       = rank1_tensors[static_cast<std::size_t>(!B_index)][dimension];
      const auto & B       = rank1_tensors[static_cast<std::size_t>(B_index)][dimension];
      AssertDimension(A.n_rows(), B.n_rows());
      AssertDimension(A.n_cols(), B.n_cols());
      auto & lambdas = eigenvalues[dimension];
      AssertDimension(A.n_rows(), A.n_cols());
      lambdas.resize(A.n_rows());
      auto & Q = eigenvectors[dimension];
      Q.reinit(A.n_rows(), A.n_cols());

      /// computes a generalized eigenvalue problem of the form:
      ///    A q = \lambda B q
      /// eigenvectors q are stored column-wise in Q
      dealii::internal::TensorProductMatrix::spectral_assembly<Number>(
        &(B(0, 0)), &(A(0, 0)), A.n_rows(), A.n_cols(), lambdas.begin(), &(Q(0, 0)));
    }
  }
};



/// TODO static path using template parameter n_rows_1d
template<int order, typename Number, int n_rows_1d>
struct ComputeGeneralizedEigendecomposition<order, VectorizedArray<Number>, n_rows_1d>
{
  void
  operator()(
    std::array<AlignedVector<VectorizedArray<Number>>, order> &               eigenvalues,
    std::array<Table<2, VectorizedArray<Number>>, order> &                    eigenvectors,
    const std::vector<std::array<Table<2, VectorizedArray<Number>>, order>> & rank1_tensors,
    const std::bitset<order>                                                  B_mask)
  {
    AssertDimension(rank1_tensors.size(), 2U);

    /// assume one-dimensional square matrices
    constexpr unsigned int macro_size         = VectorizedArray<Number>::size();
    unsigned int           n_rows_max_dynamic = 0;
    for(unsigned int d = 0; d < order; ++d)
      n_rows_max_dynamic =
        std::max<unsigned int>(n_rows_max_dynamic, rank1_tensors.front()[d].n_rows());
    const unsigned int n_flat_max =
      n_rows_1d > 0 ? n_rows_1d * macro_size : n_rows_max_dynamic * macro_size;
    const unsigned int nm_flat_max = n_rows_1d > 0 ?
                                       n_rows_1d * n_rows_1d * macro_size :
                                       n_rows_max_dynamic * n_rows_max_dynamic * macro_size;

    std::vector<Number> B_flat;
    std::vector<Number> A_flat;
    std::vector<Number> lambdas_flat;
    std::vector<Number> Q_flat;
    B_flat.resize(nm_flat_max);
    A_flat.resize(nm_flat_max);
    lambdas_flat.resize(n_flat_max);
    Q_flat.resize(nm_flat_max);
    std::array<unsigned int, macro_size> offsets_nm;
    std::array<unsigned int, macro_size> offsets_n;

    for(auto dimension = 0; dimension < order; ++dimension)
    {
      const bool   B_index = B_mask[dimension];
      const auto & A       = rank1_tensors[static_cast<std::size_t>(!B_index)][dimension];
      const auto & B       = rank1_tensors[static_cast<std::size_t>(B_index)][dimension];
      AssertDimension(A.n_rows(), B.n_rows());
      AssertDimension(A.n_cols(), B.n_cols());
      auto & lambdas = eigenvalues[dimension];
      AssertDimension(A.n_rows(), A.n_cols());
      lambdas.resize(A.n_rows());
      auto & Q = eigenvectors[dimension];
      Q.reinit(A.n_rows(), A.n_cols());

      const unsigned int m  = A.n_rows();
      const unsigned int n  = A.n_cols();
      const unsigned int nm = n * m;
      for(auto lane = 0U; lane < macro_size; ++lane)
        offsets_nm[lane] = nm * lane;

      vectorized_transpose_and_store(false, nm, &(B(0, 0)), offsets_nm.data(), B_flat.data());
      vectorized_transpose_and_store(false, nm, &(A(0, 0)), offsets_nm.data(), A_flat.data());

      const Number * B_cbegin      = B_flat.data();
      const Number * A_cbegin      = A_flat.data();
      Number *       Q_begin       = Q_flat.data();
      Number *       lambdas_begin = lambdas_flat.data();

      /// computes a generalized eigenvalue problem of the form:
      ///    A q = \lambda B q
      /// eigenvectors q are stored column-wise in Q
      for(auto lane = 0U; lane < macro_size; ++lane)
        dealii::internal::TensorProductMatrix::spectral_assembly<Number>(B_cbegin + nm * lane,
                                                                         A_cbegin + nm * lane,
                                                                         m,
                                                                         n,
                                                                         lambdas_begin + n * lane,
                                                                         Q_begin + nm * lane);

      for(auto lane = 0U; lane < macro_size; ++lane)
        offsets_n[lane] = n * lane;

      vectorized_load_and_transpose(n, lambdas_flat.data(), offsets_n.data(), lambdas.begin());
      vectorized_load_and_transpose(nm, Q_flat.data(), offsets_nm.data(), &(Q(0, 0)));
    }
  }
};

} // namespace internal



/// TODO refactor documentation for this and derived classes...
/**
 * This struct implements a tensor product matrix in the sense of a sum over
 * rank-1 tensors of matrices (so-called elementary tensors) where the tensor
 * structure is leveraged reducing memory footprint and computational
 * complexity, for example used in a matrix-vector multiplication. If the tensor
 * structure qualifies for a more specific group of tensor product matrices,
 * one might use this specific State upon initialization (resulting in even
 * more efficient memory compression and/or application routines).
 *
 * The tensor product of matrices '(x)' is frequently called Kronecker product
 * and a rank-1 tensor of order d
 *
 *    A^(d) (x) A^(d-1) (x) ... (x) A^(1)
 *
 * some times Kronecker tensor. Any order-d tensor product matrix reads
 *
 *    SUM_r=1^R  A_r^(d) (x) A_r^(d-1) (x) ... (x) A_r^(1)
 *
 * where its tensor rank (or Kronecker rank) is less or equal to R (depending
 * on the linear independence of rank-1 tensors).
 *
 * We note that due to (deal.II) conventions the first matrix A_r^(1) (that is
 * elementary_tensors[r][0] in the notation of reinit()) is the rightmost factor
 * in the Kronecker product although stored at zeroth array position.
 *
 * For the generic case (State::basic) matrix-vector multiplications are
 * efficiently evaluated by means of sum factorization. Unfortunately,
 * applying the inverse does not profit from the tensor structure as the
 * forward problem does, except for separable tensor product matrices
 * (State::separable) where the fast diagonalization method is applicable
 * (read more on this in TensorProductMatrixBaseSymmetricSum's documentation).
 *
 * Some functionalities are supported only for a specific State but more details
 * are provided in the documentation there.
 */
template<int order, typename Number, int n_rows_1d = -1>
class TensorProductMatrixBase
{
public:
  using value_type        = Number;
  using scalar_value_type = typename ExtractScalarType<Number>::type;
  using evaluator_type    = internal::Evaluator<order, Number, n_rows_1d>;
  using matrix_type_1d    = Table<2, Number>;
  using tensor_type       = std::array<matrix_type_1d, order>;


  TensorProductMatrixBase();

  TensorProductMatrixBase &
  operator=(const TensorProductMatrixBase & other);

  void
  reinit(const std::vector<tensor_type> & elementary_tensors);

  void
  clear();

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  bool
  empty() const;

  unsigned int
  m() const;

  unsigned int
  n() const;

  unsigned int
  m(unsigned int dimension) const;

  unsigned int
  n(unsigned int dimension) const;

  /**
   * Returns the number of rows for all tensor directions as multi-index.
   */
  const std::array<unsigned int, order> &
  tensor_m() const;

  /**
   * Returns the number of columns for all tensor directions as multi-index.
   */
  const std::array<unsigned int, order> &
  tensor_n() const;

  const std::vector<tensor_type> &
  get_elementary_tensors() const;

  const matrix_type_1d &
  get_matrix_1d(const unsigned int tensor_index, const unsigned int dimension) const;

protected:
  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  bool
  check_n_rows_1d_static(const std::vector<tensor_type> & tensors) const;

  bool
  check_n_rows_and_columns_1d(const std::vector<tensor_type> & tensors) const;

  bool
  check_n_rows_and_columns_1d_impl(const std::vector<tensor_type> & tensors,
                                   const unsigned int               direction,
                                   const unsigned int               n_rows,
                                   const unsigned int               n_cols) const;

  /**
   * The vector of elementary tensors, that is rank-1 tensors of
   * "one-dimensional" matrices. The sum of these tensors determines the matrix
   * this class represents. Given all rank-1 tensors are linear independent, the
   * vector length denotes the tensor rank (or Kronecker rank).
   *
   * We note that after initialization elementary tensors are only stored here
   * for the basic state. For the separable state the base class
   * TensorProductMatrixBaseSymmetricSum stores the elementary tensors in a
   * compressed fashion.
   */
  std::vector<tensor_type> elementary_tensors;

  /**
   * Helper struct which represents the tensor product index set of rows.
   */
  std::shared_ptr<const TensorHelper<order>> tensor_helper_row;

  /**
   * Helper struct which represents the tensor product index set of columns.
   */
  std::shared_ptr<const TensorHelper<order>> tensor_helper_column;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};



/**
 * TODO
 */
template<int order, typename Number, int n_rows_1d = -1>
class TensorProductMatrix : public TensorProductMatrixBase<order, Number, n_rows_1d>
{
public:
  enum class State
  {
    invalid,
    basic,
    separable,
    ranktwo,
    rankone
  };


  struct AdditionalData
  {
    /**
     * The state switches which functionalities are accessible:
     *
     * basic     : TensorProductMatrix
     * separable : ...
     * ranktwo   : ...
     * rankone   : TODO...
     */
    State state = State::invalid;

    /**
     * In case of State::ranktwo defines which one-dimensional matrix is
     * symmetric, positive definite and on the right-hand side of the generalized
     * eigenvalue problem.
     */
    std::bitset<order> spd_mask;

    /**
     * TODO...
     */
    bool force_positive_definite_inverse = false;
  };


  using Base              = TensorProductMatrixBase<order, Number, n_rows_1d>;
  using value_type        = typename Base::value_type;
  using scalar_value_type = typename Base::scalar_value_type;
  using evaluator_type    = typename Base::evaluator_type;
  using matrix_type_1d    = typename Base::matrix_type_1d;
  using tensor_type       = typename Base::tensor_type;


  TensorProductMatrix();

  TensorProductMatrix(const tensor_type & rank1_tensor);

  TensorProductMatrix(const std::vector<tensor_type> & elementary_tensors,
                      const AdditionalData &           additional_data = {State::basic,
                                                                std::bitset<order>{},
                                                                false});

  /**
   * Same as constructor above when passing AdditionalData with variables @p
   * state and @p spd_mask set to @p state_in and @p spd_mask_in, respectively.
   */
  TensorProductMatrix(const std::vector<tensor_type> & elementary_tensors,
                      const State                      state_in,
                      const std::bitset<order>         spd_mask_in = std::bitset<order>{});

  TensorProductMatrix &
  operator=(const TensorProductMatrix & other);

  /**
   * Depending on the matrix state @p additional_data_in.state the rank-1
   * tensors of matrices @p elementary_tensors_in initialize this tensor product
   * matrix.
   *
   * basic : All rank-1 tensors are treated in the way they are passed, see the
   * class' description.
   *
   * ranktwo : Accepts two and only two rank-1 tensors passed by @p
   * elementary_tensors. During initialization a generalized eigendecomposition
   * is computed which enables an efficient implementation of apply_inverse()
   * (see the class' description.) The bitset @p additional_data_in.spd_mask
   * defines for each direction which matrix is symmetric, positive definite
   * (requirement for generalized eigenvalue problem). The bit position
   * coincides with the tensor direction.
   *
   * separable : Accepts two and only two rank-1 tensors passed by @p
   * elementary_tensors. During initialization a generalized eigendecomposition
   * is computed which enables an efficient implementation of apply_inverse()
   * (see the class' description). The first rank-1 tensor needs to contain mass
   * matrices (symmetric, positive definite) and the second derivative matrices,
   * in the context of Rice, Lynch and Thomas.
   *
   * rankone : Accepts one and only one rank-1 tensor passed in @p
   * elementary_tensors. During initialization a specific eigendecomposition is
   * computed which enables an efficient implementation of apply_inverse() (see
   * the class' description for more details).
   */
  void
  reinit(const std::vector<tensor_type> & elementary_tensors_in,
         const AdditionalData & additional_data_in = {State::basic, std::bitset<order>{}, false});


  /**
   * Same as reinit() above when passing AdditionalData with variables @p state
   * and @p spd_mask set to @p state_in and @p spd_mask_in, respectively.
   */
  void
  reinit(const std::vector<tensor_type> & elementary_tensors_in,
         const State                      state_in,
         const std::bitset<order>         spd_mask_in = std::bitset<order>{});

  /**
   * Clears all data, sizes are reset to zero and the matrix state becomes
   * State::invalid. Only after a call to reinit() the matrix is usable again.
   */
  void
  clear();

  /**
   * Depending on the current state we apply the (fast) inverse of the
   * underlying tensor product matrix.
   *
   * basic : We do not compute the inverse during (re-)initialization such that
   * non-invertible matrices are possible. The first time @p apply_inverse() is
   * called after (re-)initialization the inverse is computed (thus requiring @p
   * basic_inverse to be mutable).
   *
   * ranktwo : We use the (generalized) eigendecomposition computed during
   * (re-)initialization to apply the inverse via fast diagonalization.
   *
   * separable : We use the (generalized) eigendecomposition computed during
   * (re-)initialization to apply the inverse via fast diagonalization.
   *
   * rankone : We use the eigendecomposition computed during (re-)initialization
   * to apply the inverse via fast diagonalization.
   *
   * Note that for each vectorization lane of arithmetic type @p Number the
   * inverse of (nearly) zero eigenvalues is replaced by zero avoiding division
   * by zero.
   */
  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * Sum-factorization is used to efficiently apply this matrix to a vector @p
   * src_view resulting in the vector @p dst_view: to be precise we accumulate
   * all products of rank-1 tensors with the source vector.
   */
  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * Same as above with different argument types.
   */
  void
  vmult(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  /**
   * Same as vmult() but adding into the destination vector @p dst_view.
   */
  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * Same as above with different argument types.
   */
  void
  vmult_add(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  /**
   * Same as vmult() but multiplying with the transpose matrix.
   */
  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * Same as above with different argument types.
   */
  void
  Tvmult(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  /**
   * Same as Tvmult() but adding into the destination vector @p dst_view.
   */
  void
  Tvmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * Same as above with different argument types.
   */
  void
  Tvmult_add(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  /// TODO does this conversion make sense?!...
  operator std::vector<std::vector<std::array<Table<2, Number>, order>>> const &() const
  {
    return this->elementary_tensors;
  }

  /**
   * Returns the number of rank-1 tensors used to initialize this tensor product
   * matrix. Depending on the linear independence of those tensors the actual
   * tensor rank is less than or equal to the maximal rank returned.
   */
  unsigned int
  n_max_rank() const;

  /**
   * Returns the eigenvalues of the underlying matrix, to be precise for the @p
   * m() times @p n() matrix with Kronecker products resolved (note that this
   * requires the number of rows @p m() and number of columns @p n() to be
   * equal).
   *
   * This functionality is only supported in separable/ranktwo/rankone state.
   */
  AlignedVector<Number>
  get_eigenvalues() const;

  /**
   * Returns the tensor of vectors containing the (generalized) eigenvalues for
   * each direction computed during reinit().
   *
   * This functionality is only supported in separable/ranktwo/rankone state.
   */
  const std::array<AlignedVector<Number>, order> &
  get_eigenvalue_tensor() const;

  /**
   * Returns the (generalized) eigenvectors of the underlying matrix, to be
   * precise for the @p m() times @p n() matrix with Kronecker products resolved
   * (note that this requires the number of rows @p m() and number of columns @p
   * n() to be equal). The (generalized) eigenvectors are representable as
   * rank-1 tensor of univariate eigenvectors if the matrix is appropriate (see
   * the class' documentation on the requirements for a tensor product matrix
   * with valid ranktwo/separable/rankone state.
   *
   * This functionality is only supported in separable/ranktwo/rankone state.
   */
  Table<2, Number>
  get_eigenvectors() const;

  /**
   * Returns the tensor of (generalized) eigenvectors. The univariate
   * eigenvectors are stored column-wise.
   *
   * This functionality is only supported in separable/ranktwo/rankone state.
   */
  const tensor_type &
  get_eigenvector_tensor() const;

  /**
   * Assembles and returns the tensor of mass matrices if in separable state.
   */
  tensor_type
  get_mass_tensor() const;

  /**
   * Returns the current matrix state.
   */
  State
  get_state() const;

  /**
   * Returns the m() times n() matrix after resolving all kronecker products.
   */
  Table<2, Number>
  as_table() const;

  /**
   * Returns the inverse of this m() times n() matrix after resolving all
   * kronecker products.
   */
  Table<2, Number>
  as_inverse_table() const;

  /**
   * Returns the transpose of this m() times n() matrix after resolving all
   * kronecker products.
   */
  Table<2, Number>
  as_transpose_table() const;

private:
  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  AlignedVector<Number>
  compute_eigenvalues_impl_ranktwo() const;

  AlignedVector<Number>
  compute_eigenvalues_impl_separable() const;

  AlignedVector<Number>
  compute_eigenvalues_impl_rankone() const;

  void
  apply_inverse_impl(const ArrayView<Number> &       dst_view,
                     const ArrayView<const Number> & src_view) const;

  void
  apply_inverse_impl_basic(const ArrayView<Number> &       dst_view,
                           const ArrayView<const Number> & src_view) const;

  template<bool zero_out_negative_eigenvalues>
  void
  apply_inverse_impl_eigen(const ArrayView<Number> &       dst_view,
                           const ArrayView<const Number> & src_view) const;

  /**
   * Additional data storing (essential) matrix features.
   */
  AdditionalData additional_data;

  /**
   * The (naively computed) inverse of the underlying matrix if in basic state.
   */
  mutable std::shared_ptr<const InverseTable<Number>> basic_inverse;

  /**
   * The eigenvalues stored in tensor product form if in
   * ranktwo/separable/rankone state. The eigenvalues of the underlying m()
   * times n() matrix can be computed as sum over Kronecker products of diagonal
   * matrices.
   */
  std::array<AlignedVector<Number>, order> eigenvalues;

  /**
   * The (generalized) eigenvectors stored in tensor product form if in
   * rankone/ranktwo/separable state. The (generalized) eigenvectors of the
   * underlying m() times n() matrix are a rank-1 tensor of univariate
   * eigenvectors.
   */
  TensorProductMatrixBase<order, Number, n_rows_1d> eigenvectors;
};



////////// Definitions



template<int order, typename Number, int n_rows_1d>
inline const std::vector<std::array<Table<2, Number>, order>> &
TensorProductMatrixBase<order, Number, n_rows_1d>::get_elementary_tensors() const
{
  return elementary_tensors;
}



template<int order, typename Number, int n_rows_1d>
inline const typename TensorProductMatrixBase<order, Number, n_rows_1d>::matrix_type_1d &
TensorProductMatrixBase<order, Number, n_rows_1d>::get_matrix_1d(const unsigned int tensor_index,
                                                                 const unsigned int dimension) const
{
  AssertIndexRange(tensor_index, elementary_tensors.size());
  AssertIndexRange(dimension, order);
  return elementary_tensors[tensor_index][dimension];
}



template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrixBase<order, Number, n_rows_1d>::m() const
{
  Assert(tensor_helper_row, ExcMessage("tensor_helper_row is not initialized."));
  return tensor_helper_row->n_flat();
}



template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrixBase<order, Number, n_rows_1d>::n() const
{
  Assert(tensor_helper_column, ExcMessage("tensor_helper_column is not initialized."));
  return tensor_helper_column->n_flat();
}



template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrixBase<order, Number, n_rows_1d>::m(unsigned int dimension) const
{
  AssertIndexRange(dimension, order);
  Assert(tensor_helper_row, ExcMessage("tensor_helper_row is not initialized."));
  return tensor_helper_row->size(dimension);
}



template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrixBase<order, Number, n_rows_1d>::n(unsigned int dimension) const
{
  AssertIndexRange(dimension, order);
  Assert(tensor_helper_column, ExcMessage("tensor_helper_column is not initialized."));
  return tensor_helper_column->size(dimension);
}



template<int order, typename Number, int n_rows_1d>
inline const std::array<unsigned int, order> &
TensorProductMatrixBase<order, Number, n_rows_1d>::tensor_m() const
{
  Assert(tensor_helper_row, ExcMessage("tensor_helper_row is not initialized."));
  return tensor_helper_row->size();
}



template<int order, typename Number, int n_rows_1d>
inline const std::array<unsigned int, order> &
TensorProductMatrixBase<order, Number, n_rows_1d>::tensor_n() const
{
  Assert(tensor_helper_column, ExcMessage("tensor_helper_column is not initialized."));
  return tensor_helper_column->size();
}



template<int order, typename Number, int n_rows_1d>
inline bool
TensorProductMatrixBase<order, Number, n_rows_1d>::empty() const
{
  return m() == 0U || n() == 0U;
}



template<int order, typename Number, int n_rows_1d>
inline bool
TensorProductMatrixBase<order, Number, n_rows_1d>::check_n_rows_and_columns_1d_impl(
  const std::vector<std::array<Table<2, Number>, order>> & tensors,
  const unsigned int                                       direction,
  const unsigned int                                       n_rows,
  const unsigned int                                       n_cols) const
{
  return std::all_of(tensors.cbegin(),
                     tensors.cend(),
                     [direction, n_rows, n_cols](const auto & tensor) {
                       const auto & matrix = tensor[direction];
                       return matrix.size(0) == n_rows && matrix.size(1) == n_cols;
                     });
}



template<int order, typename Number, int n_rows_1d>
inline bool
TensorProductMatrixBase<order, Number, n_rows_1d>::check_n_rows_and_columns_1d(
  const std::vector<std::array<Table<2, Number>, order>> & tensors) const
{
  Assert(!tensors.empty(), ExcMessage("The vector of tensors is empty."));
  Assert(tensor_helper_row, ExcMessage("tensor_helper isn't initialized."));
  Assert(tensor_helper_column, ExcMessage("tensor_helper isn't initialized."));

  for(const auto tensor_of_matrices : tensors)
  {
    std::array<unsigned int, order> n_rows_foreach_dimension;
    std::array<unsigned int, order> n_columns_foreach_dimension;
    for(auto d = 0; d < order; ++d)
    {
      const auto & matrix            = tensor_of_matrices[d];
      n_rows_foreach_dimension[d]    = matrix.size(0);
      n_columns_foreach_dimension[d] = matrix.size(1);
    }

    const bool has_equal_rows    = *tensor_helper_row == n_rows_foreach_dimension;
    const bool has_equal_columns = *tensor_helper_column == n_columns_foreach_dimension;

    if(!has_equal_rows || !has_equal_columns)
      return false;
  }

  return true;
}



template<int order, typename Number, int n_rows_1d>
inline bool
TensorProductMatrixBase<order, Number, n_rows_1d>::check_n_rows_1d_static(
  const std::vector<std::array<Table<2, Number>, order>> & tensors) const
{
  if(n_rows_1d == -1)
    return true;
  if(n_rows_1d > 0)
  {
    std::vector<unsigned int> list_of_dimensions(order);
    std::iota(list_of_dimensions.begin(), list_of_dimensions.end(), 0);
    return std::all_of(
      list_of_dimensions.cbegin(), list_of_dimensions.cend(), [&](const auto & direction) {
        return check_n_rows_and_columns_1d_impl(tensors, direction, n_rows_1d, n_rows_1d);
      });
  }
  return false;
}



template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrix<order, Number, n_rows_1d>::n_max_rank() const
{
  if(additional_data.state == State::basic)
  {
    return this->elementary_tensors.size();
  }
  else if(additional_data.state == State::ranktwo)
  {
    AssertDimension(this->elementary_tensors.size(), 2U);
    return 2;
  }
  else if(additional_data.state == State::separable)
  {
    AssertDimension(this->elementary_tensors.size(), order); // TODO
    return order;
  }
  else if(additional_data.state == State::rankone)
  {
    AssertDimension(this->elementary_tensors.size(), 1U);
    return 1U;
  }
  else if(additional_data.state == State::invalid)
  {
    AssertThrow(false, ExcMessage("State is invalid."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return numbers::invalid_unsigned_int;
}


template<int order, typename Number, int n_rows_1d>
inline typename TensorProductMatrix<order, Number, n_rows_1d>::State
TensorProductMatrix<order, Number, n_rows_1d>::get_state() const
{
  return this->additional_data.state;
}

} // namespace Tensors

#include "tensor_product_matrix.templates.h"

#endif // TENSOR_PRODUCT_MATRIX_H_
