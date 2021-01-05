/*
 * tensor_product_matrix_new.h
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#ifndef TENSOR_PRODUCT_MATRIX_NEW_H_
#define TENSOR_PRODUCT_MATRIX_NEW_H_

#include <deal.II/lac/lapack_full_matrix.h>

#include "matrix_helper.h"
#include "move_to_deal_ii.h"
#include "tensor_product_matrix.h"
#include "tensors.h"

using namespace dealii;

namespace Tensors
{
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
 * (read more on this in TensorProductMatrixBaseSymmetricSum's documentation).
 *
 * Some functionalities of this struct are only valid for a specific State but
 * more details are provided in the documentation there.
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


  unsigned int
  m() const;

  unsigned int
  n() const;

  unsigned int
  m(unsigned int dimension) const;

  unsigned int
  n(unsigned int dimension) const;

  std::vector<tensor_type>
  get_elementary_tensors() const;

  const matrix_type_1d &
  get_matrix_1d(const unsigned int tensor_index, const unsigned int dimension) const;

protected:
  TensorProductMatrixBase();

  TensorProductMatrixBase &
  operator=(const TensorProductMatrixBase & other);

  void
  reinit(const std::vector<tensor_type> & elementary_tensors);

  void
  clear();

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

private:
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
class TensorProductMatrix_new : public TensorProductMatrixBase<order, Number, n_rows_1d>
{
public:
  enum class State
  {
    invalid,
    basic,
    separable,
    ranktwo
  };


  using Base              = TensorProductMatrixBase<order, Number, n_rows_1d>;
  using value_type        = typename Base::value_type;
  using scalar_value_type = typename Base::scalar_value_type;
  using evaluator_type    = typename Base::evaluator_type;
  using matrix_type_1d    = typename Base::matrix_type_1d;
  using tensor_type       = typename Base::tensor_type;


  TensorProductMatrix_new();

  TensorProductMatrix_new(const tensor_type & rank1_tensor);

  TensorProductMatrix_new(const std::vector<tensor_type> & elementary_tensors,
                          const State                      state_in    = State::basic,
                          const std::bitset<order>         spd_mask_in = std::bitset<order>{});

  TensorProductMatrix_new &
  operator=(const TensorProductMatrix_new & other);

  /**
   * TODO
   */
  void
  reinit(const std::vector<tensor_type> & elementary_tensors_in,
         const State                      state_in    = State::basic,
         const std::bitset<order>         spd_mask_in = std::bitset<order>{});

  void
  clear();

  /**
   * TODO
   */
  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  // /**
  //  * TODO
  //  */
  // AlignedVector<Number>
  // get_eigenvalues() const;

  // /**
  //  * TODO
  //  */
  // const tensor_type &
  // get_eigenvectors() const;

  // std::vector<tensor_type>
  // get_elementary_tensors() const;

  State
  get_state() const;

  Table<2, Number>
  as_table() const;

  Table<2, Number>
  as_inverse_table() const;

  Table<2, Number>
  as_transpose_table() const;

private:
  /**
   * The state switches which functionalities are accessible:
   *
   * basic     : TensorProductMatrix_new
   * separable : TensorProductMatrix_newSymmetricSum
   * ranktwo   : TODO...
   */
  State state = State::invalid;

  /**
   * In case of State::ranktwo defines which one-dimensional matrix is
   * symmetric, positive definite and on the right-hand side of the generalized
   * eigenvalue problem.
   */
  std::bitset<order> spd_mask;

  void
  apply_inverse_impl(const ArrayView<Number> &       dst_view,
                     const ArrayView<const Number> & src_view) const;

  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * The naive inverse of the underlying matrix for the basic state.
   */
  mutable std::shared_ptr<const InverseTable<Number>> basic_inverse;

  std::shared_ptr<const TensorProductMatrixBase<order, Number, n_rows_1d>> eigenvectors;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};



////////// Definitions



template<int order, typename Number, int n_rows_1d>
inline std::vector<std::array<Table<2, Number>, order>>
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

} // namespace Tensors

#include "tensor_product_matrix_new.templates.h"

#endif // TENSOR_PRODUCT_MATRIX_NEW_H_
