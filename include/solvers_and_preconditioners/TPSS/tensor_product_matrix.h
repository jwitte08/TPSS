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
    separable
  };


  using separable_matrix_type = TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>;
  using value_type            = Number;
  using scalar_value_type     = typename ExtractScalarType<Number>::type;
  using matrix_type_1d        = Table<2, Number>;
  using tensor_type           = std::array<matrix_type_1d, order>;


  TensorProductMatrix() = default;

  TensorProductMatrix(const std::vector<tensor_type> & elementary_tensors,
                      const State                      state_in = State::basic);

  TensorProductMatrix &
  operator=(const TensorProductMatrix & other);

  void
  reinit(const std::vector<tensor_type> & elementary_tensors_in,
         const State                      state_in = State::basic);
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
   */
  State state = State::invalid;

private:
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
inline TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix(
  const std::vector<typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type> &
              elementary_tensors,
  const State state_in)
{
  reinit(elementary_tensors, state_in);
}

template<int order, typename Number, int n_rows_1d>
inline TensorProductMatrix<order, Number, n_rows_1d> &
TensorProductMatrix<order, Number, n_rows_1d>::operator=(const TensorProductMatrix & other)
{
  reinit(other.elementary_tensors, other.state);
  return *this;
}

template<int order, typename Number, int n_rows_1d>
inline AlignedVector<Number>
TensorProductMatrix<order, Number, n_rows_1d>::get_eigenvalues() const
{
  AssertThrow(state == State::separable, ExcMessage("Not implemented."));

  const auto & eigenvalues_foreach_dimension = separable_matrix_type::eigenvalues;
  for(auto direction = 0; direction < order; ++direction)
  {
    AssertDimension(tensor_helper_row->n_flat(), tensor_helper_column->n_flat());
    AssertDimension(tensor_helper_row->n_flat(), eigenvalues_foreach_dimension[direction].size());
  }

  // /// OLD OLD
  // std::array<unsigned int, order> sizes;
  // std::transform(eigenvalues_foreach_dimension.cbegin(),
  //                eigenvalues_foreach_dimension.cend(),
  //                sizes.begin(),
  //                [](const auto & evs) { return evs.size(); });
  // AlignedVector<Number> eigenvalues(m());
  // for(unsigned int i = 0; i < eigenvalues.size(); ++i)
  // {
  //   const auto & ii     = uni_to_multiindex<order>(i, sizes);
  //   Number       lambda = eigenvalues_foreach_dimension[0][ii[0]];
  //   for(auto d = 1; d < order; ++d)
  //     lambda += eigenvalues_foreach_dimension[d][ii[d]];
  //   eigenvalues[i] = lambda;
  // }
  /// TODO needs to be verified
  AlignedVector<Number> eigenvalues(m());
  for(unsigned int i = 0; i < eigenvalues.size(); ++i)
  {
    const auto & ii     = tensor_helper_row->multi_index(i);
    Number       lambda = eigenvalues_foreach_dimension[0][ii[0]];
    for(auto d = 1; d < order; ++d)
      lambda += eigenvalues_foreach_dimension[d][ii[d]];
    eigenvalues[i] = lambda;
  }


  // TODO
  const bool has_zero_eigenvalues =
    std::any_of(eigenvalues.begin(), eigenvalues.end(), Tensors::is_nearly_zero_value<Number>);
  AssertThrow(!has_zero_eigenvalues, ExcMessage("Has zero eigenvalues."));

  return eigenvalues;
}


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
  AssertThrow(state == State::separable, ExcMessage("Not implemented."));
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
  return elementary_tensors.front()[dimension].n_cols();
}


template<int order, typename Number, int n_rows_1d>
inline Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}


template<int order, typename Number, int n_rows_1d>
inline Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}


template<int order, typename Number, int n_rows_1d>
inline Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::as_transpose_table() const
{
  return Tensors::transpose_matrix_to_table(*this);
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
