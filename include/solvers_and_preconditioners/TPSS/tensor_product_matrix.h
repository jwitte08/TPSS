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
   * Returns the maximum number of elements intermediate tensors have while a
   * sum factorization is applied following the conventions of
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



/**
 * This struct implements a tensor product matrix in the sense of a sum over
 * rank-1 tensors of matrices (so-called elementary tensors) where the tensor
 * structure is leveraged reducing memory footprint and computational
 * complexity, for example for matrix-vector multiplication. If the tensor
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
 * on the linear independence of rank-1 tensors). We note that due to
 * (deal.II) conventions the first matrix A_r^(1) (that is
 * elementary_tensors[r][0] in the notation of reinit()) is the rightmost
 * factor in the Kronecker product although stored at zeroth array position.
 *
 * For the generic case (State::basic) matrix-vector multiplications are
 * efficiently evaluated by means of sum factorization. Unfortunately,
 * applying the inverse does not profit from the tensor structure as the
 * forward problem does, except for separable tensor product matrices
 * (State::separable) where the fast diagonalization method is applicable
 * (read more on this in TensorProductMatrixSymmetricSum's documentation).
 *
 * Some functionalities of this struct are only valid for a specific State but
 * more details are provided in the documentation there.
 */
template<int order, typename Number, int n_rows_1d = -1>
class TensorProductMatrix : public TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>
{
public:
  enum class State
  {
    invalid,
    basic,
    separable,
    ranktwo
  };


  using separable_matrix_type = TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>;
  using value_type            = Number;
  using scalar_value_type     = typename ExtractScalarType<Number>::type;
  using evaluator_type        = internal::Evaluator<order, Number, n_rows_1d>;
  using matrix_type_1d        = Table<2, Number>;
  using tensor_type           = std::array<matrix_type_1d, order>;


  TensorProductMatrix() = default;

  TensorProductMatrix(const tensor_type & rank1_tensor);

  TensorProductMatrix(const std::vector<tensor_type> & elementary_tensors,
                      const State                      state_in    = State::basic,
                      const std::bitset<order>         spd_mask_in = std::bitset<order>{});

  TensorProductMatrix &
  operator=(const TensorProductMatrix & other);

  /**
   * For State::basic the @p elementary_tensors_in has to be filled as in the
   * class's documentation. For State::separable @p elementary_tensors_in[0]
   * should pass "mass" matrices and @p elementary_tensors_in[1] "derivative"
   * matrices (for more details see the documentation of
   * dealii::TensorProductMatrixSymmetricSum).
   */
  void
  reinit(const std::vector<tensor_type> & elementary_tensors_in,
         const State                      state_in    = State::basic,
         const std::bitset<order>         spd_mask_in = std::bitset<order>{});

  void
  clear();

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  unsigned int
  m() const;

  unsigned int
  n() const;

  unsigned int
  m(unsigned int dimension) const;

  unsigned int
  n(unsigned int dimension) const;

  /// TODO !!! This function has been changed without testing...
  AlignedVector<Number>
  get_eigenvalues() const;

  const tensor_type &
  get_eigenvectors() const;

  std::vector<tensor_type>
  get_elementary_tensors() const;

  const tensor_type &
  get_mass() const;

  const matrix_type_1d &
  get_matrix_1d(const unsigned int tensor_index, const unsigned int dimension) const;

  State
  get_state() const;

  Table<2, Number>
  as_table() const;

  Table<2, Number>
  as_inverse_table() const;

  Table<2, Number>
  as_transpose_table() const;

protected:
  /**
   * The vector of elementary tensors, that is rank-1 tensors of
   * "one-dimensional" matrices. The sum of these tensors determines the matrix
   * this class represents. Given all rank-1 tensors are linear independent, the
   * vector length denotes the tensor rank (or Kronecker rank).
   *
   * We note that after initialization elementary tensors are only stored here
   * for the basic state. For the separable state the base class
   * TensorProductMatrixSymmetricSum stores the elementary tensors in a
   * compressed fashion.
   */
  std::vector<tensor_type> elementary_tensors;

  /**
   * The state switches which functionalities are accessible:
   *
   * basic     : TensorProductMatrix
   * separable : TensorProductMatrixSymmetricSum
   * ranktwo   : TODO...
   */
  State state = State::invalid;

  /**
   * In case of State::ranktwo defines which one-dimensional matrix is
   * symmetric, positive definite and on the right-hand side of the generalized
   * eigenvalue problem.
   */
  std::bitset<order> spd_mask;

private:
  void
  reinit_ranktwo_impl(const std::vector<tensor_type> & elementary_tensors_in,
                      const std::bitset<order>         spd_mask_in);

  void
  apply_inverse_impl(const ArrayView<Number> &       dst_view,
                     const ArrayView<const Number> & src_view) const;

  void
  apply_inverse_impl_basic_static(const ArrayView<Number> &       dst_view,
                                  const ArrayView<const Number> & src_view) const;

  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  template<bool add, bool transpose>
  void
  vmult_separable_impl(const ArrayView<Number> &       dst_view,
                       const ArrayView<const Number> & src_view) const;

  template<bool add, bool transpose>
  void
  vmult_basic_impl(const ArrayView<Number> &       dst_view,
                   const ArrayView<const Number> & src_view) const;

  AlignedVector<Number>
  get_eigenvalues_ranktwo_impl() const;

  bool
  check_n_rows_1d_static(const std::vector<tensor_type> & tensors) const;

  bool
  check_n_rows_and_columns_1d(const std::vector<tensor_type> & tensors,
                              const unsigned int               direction) const;

  bool
  check_n_rows_and_columns_1d_impl(const std::vector<tensor_type> & tensors,
                                   const unsigned int               direction,
                                   const unsigned int               n_rows,
                                   const unsigned int               n_cols) const;

  /**
   * Helper struct which represents the tensor product index set of rows.
   */
  std::shared_ptr<const TensorHelper<order>> tensor_helper_row;

  /**
   * Helper struct which represents the tensor product index set of columns.
   */
  std::shared_ptr<const TensorHelper<order>> tensor_helper_column;

  /**
   * The naive inverse of the underlying matrix for the basic state.
   */
  mutable std::shared_ptr<const InverseTable<Number>> basic_inverse;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};



template<int order, typename Number, int n_rows_1d>
inline const std::array<Table<2, Number>, order> &
TensorProductMatrix<order, Number, n_rows_1d>::get_mass() const
{
  AssertThrow(state == State::separable, ExcMessage("Not implemented."));
  return this->mass_matrix;
}


template<int order, typename Number, int n_rows_1d>
inline const std::array<Table<2, Number>, order> &
TensorProductMatrix<order, Number, n_rows_1d>::get_eigenvectors() const
{
  AssertThrow(state == State::separable || state == State::ranktwo, ExcMessage("Not implemented."));
  return this->eigenvectors;
}


template<int order, typename Number, int n_rows_1d>
inline std::vector<std::array<Table<2, Number>, order>>
TensorProductMatrix<order, Number, n_rows_1d>::get_elementary_tensors() const
{
  Assert(state != State::invalid, ExcMessage("Invalid State"));

  if(state == State::separable)
  {
    std::vector<std::array<Table<2, Number>, order>> tensors(order);
    const auto &                                     mass       = this->mass_matrix;
    const auto &                                     derivative = this->derivative_matrix;
    for(auto i = 0U; i < order; ++i)
    {
      auto & tensor = tensors[i];
      for(auto j = 0U; j < order; ++j)
        tensor[j] = i == j ? derivative[j] : mass[j];
    }
    return tensors;
  }

  return elementary_tensors;
}


template<int order, typename Number, int n_rows_1d>
inline const typename TensorProductMatrix<order, Number, n_rows_1d>::matrix_type_1d &
TensorProductMatrix<order, Number, n_rows_1d>::get_matrix_1d(const unsigned int tensor_index,
                                                             const unsigned int dimension) const
{
  AssertIndexRange(tensor_index, elementary_tensors.size());
  AssertIndexRange(dimension, order);
  return elementary_tensors[tensor_index][dimension];
}


template<int order, typename Number, int n_rows_1d>
inline typename TensorProductMatrix<order, Number, n_rows_1d>::State
TensorProductMatrix<order, Number, n_rows_1d>::get_state() const
{
  return state;
}


template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrix<order, Number, n_rows_1d>::m() const
{
  if(state == State::separable)
    return separable_matrix_type::m();
  Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
  Assert(tensor_helper_row, ExcMessage("tensor_helper is not initialized."));
  return tensor_helper_row->n_flat();
}


template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrix<order, Number, n_rows_1d>::n() const
{
  if(state == State::separable)
    return separable_matrix_type::n();
  Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
  Assert(tensor_helper_column, ExcMessage("tensor_helper is not initialized."));
  return tensor_helper_column->n_flat();
}


template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrix<order, Number, n_rows_1d>::m(unsigned int dimension) const
{
  AssertIndexRange(dimension, order);
  if(state == State::separable)
    return separable_matrix_type::eigenvalues[dimension].size();
  Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
  /// TODO use tensor_helper_row
  return elementary_tensors.front()[dimension].n_rows();
}


template<int order, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrix<order, Number, n_rows_1d>::n(unsigned int dimension) const
{
  AssertIndexRange(dimension, order);
  if(state == State::separable)
    return separable_matrix_type::eigenvalues[dimension].size();
  Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
  /// TODO use tensor_helper_column
  return elementary_tensors.front()[dimension].n_cols();
}


template<int order, typename Number, int n_rows_1d>
inline bool
TensorProductMatrix<order, Number, n_rows_1d>::check_n_rows_and_columns_1d_impl(
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
TensorProductMatrix<order, Number, n_rows_1d>::check_n_rows_and_columns_1d(
  const std::vector<std::array<Table<2, Number>, order>> & tensors,
  const unsigned int                                       direction) const
{
  Assert(!tensors.empty(), ExcMessage("The vector of tensors is empty."));
  const unsigned int n_rows = tensors.front()[direction].size(0);
  const unsigned int n_cols = tensors.front()[direction].size(1);
  return check_n_rows_and_columns_1d_impl(tensors, direction, n_rows, n_cols);
}


template<int order, typename Number, int n_rows_1d>
inline bool
TensorProductMatrix<order, Number, n_rows_1d>::check_n_rows_1d_static(
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

} // namespace Tensors

#include "tensor_product_matrix.templates.h"

#endif // TENSOR_PRODUCT_MATRIX_H_
