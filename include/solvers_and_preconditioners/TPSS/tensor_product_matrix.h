/*
 * tensor_product_matrix.h
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#ifndef TENSOR_PRODUCT_MATRIX_H_
#define TENSOR_PRODUCT_MATRIX_H_

#include "tensors.h"

using namespace dealii;

namespace Tensors
{
template<int order, typename Number, int n_rows_1d = -1>
class TensorProductMatrix : public TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>
{
public:
  enum class State
  {
    invalid,
    basic,
    skd
  };

  using SKDMatrix = TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>;
  using SKDMatrix::value_type;

  TensorProductMatrix() = default;

  TensorProductMatrix(const std::vector<std::array<Table<2, Number>, order>> & elementary_tensors,
                      const State state_in = State::basic)
  {
    static_assert(order == 2, "TODO");
    reinit(elementary_tensors, state_in);
  }

  void
  reinit(const std::vector<std::array<Table<2, Number>, order>> & elementary_tensors,
         const State                                              state_in = State::basic)
  {
    // AssertThrow(check_n_rows_1d(left_or_mass_in) && check_n_rows_1d(right_or_derivative_in),
    //             ExcMessage(
    //               "Not all univariate matrices inserted are of size (n_rows_1d x n_rows_1d)."));

    if(state_in == State::basic)
    {
      left_owned.resize(elementary_tensors.size());
      right_owned.resize(elementary_tensors.size());
      std::transform(elementary_tensors.cbegin(),
                     elementary_tensors.cend(),
                     left_owned.begin(),
                     [](const auto & tensor) { return tensor[0]; });
      std::transform(elementary_tensors.cbegin(),
                     elementary_tensors.cend(),
                     right_owned.begin(),
                     [](const auto & tensor) { return tensor[1]; });
      left_or_mass        = make_array_view(left_owned);
      right_or_derivative = make_array_view(right_owned);
    }

    else if(state_in == State::skd)
    {
      AssertThrow(elementary_tensors.size() == order,
                  ExcMessage("The number of mass/derivative matrices and orderension differ."));
      // std::array<Table<2, Number>, order> mass_matrices;
      // std::array<Table<2, Number>, order> derivative_matrices;
      // std::copy(left_or_mass_in.begin(), left_or_mass_in.end(), mass_matrices.begin());
      // std::copy(right_or_derivative_in.begin(),
      //           right_or_derivative_in.end(),
      //           derivative_matrices.begin());
      // SKDMatrix::reinit(mass_matrices, derivative_matrices);
      SKDMatrix::reinit(elementary_tensors[0], elementary_tensors[1]);
      left_or_mass = make_array_view(SKDMatrix::mass_matrix.begin(), SKDMatrix::mass_matrix.end());
      right_or_derivative =
        make_array_view(SKDMatrix::derivative_matrix.begin(), SKDMatrix::derivative_matrix.end());
    }
    else
      AssertThrow(false, ExcMessage("Invalid state at initialization."));

    state = state_in;
  }

  unsigned int
  m() const
  {
    if(state == State::skd)
      return SKDMatrix::m();
    Assert(left_or_mass.size() > 0, ExcMessage("Not initialized."));
    const unsigned int m_left  = left(0).size(0);
    const unsigned int m_right = right(0).size(0);
    return m_left * m_right;
  }

  unsigned int
  n() const
  {
    if(state == State::skd)
      return SKDMatrix::n();
    Assert(left_or_mass.size() > 0, ExcMessage("Not initialized."));
    const unsigned int n_left  = left(0).size(1);
    const unsigned int n_right = right(0).size(1);
    return n_left * n_right;
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ false>(dst_view, src_view);
  }

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ true>(dst_view, src_view);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

protected:
  const Table<2, Number> &
  left(unsigned int r) const
  {
    AssertIndexRange(r, left_or_mass.size());
    return left_or_mass[r];
  }

  const Table<2, Number> &
  right(unsigned int r) const
  {
    AssertIndexRange(r, right_or_derivative.size());
    return right_or_derivative[r];
  }

  /**
   * An array view pointing to the left or mass matrices, respectively,
   * depending on the active state.
   */
  ArrayView<Table<2, Number>> left_or_mass;

  /**
   * A vector containing left matrices, that is left factors of the sum over
   * Kronecker products.
   */
  std::vector<Table<2, Number>> left_owned;

  /**
   * An array view pointing to the right or derivative matrices, respectively,
   * depending on the active state.
   */
  ArrayView<Table<2, Number>> right_or_derivative;

  /**
   * A vector containing right matrices, that is right factors of the sum over
   * Kronecker products.
   */
  std::vector<Table<2, Number>> right_owned;

  /**
   * The state switches which functionalities are faciliated:
   *
   * basic:  TensorProductMatrix
   * skd:    TensorProductMatrixSymmetricSum
   */
  State state = State::invalid;

private:
  template<bool add>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    if(state == State::basic)
      vmult_impl_basic_static<add>(dst_view, src_view);
    else if(state == State::skd)
    {
      AlignedVector<Number> initial_dst;
      if(add)
      {
        initial_dst.resize_fast(dst_view.size());
        std::copy(dst_view.cbegin(), dst_view.cend(), initial_dst.begin());
      }
      // const auto initial_dst_view = make_array_view(initial_dst.begin(),initial_dst.end());
      SKDMatrix::vmult(dst_view, src_view);
      if(add)
        std::transform(dst_view.cbegin(),
                       dst_view.cend(),
                       initial_dst.begin(),
                       dst_view.begin(),
                       [](const auto & elem1, const auto & elem2) { return elem1 + elem2; });
    }
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }


  template<bool add>
  void
  vmult_impl_basic_static(const ArrayView<Number> &       dst_view,
                          const ArrayView<const Number> & src_view) const
  {
    AssertDimension(dst_view.size(), m());
    AssertDimension(src_view.size(), n());
    std::lock_guard<std::mutex> lock(this->mutex);
    const unsigned int          mm =
      n_rows_1d > 0 ? Utilities::fixed_power<order>(n_rows_1d) : right(0).size(0) * left(0).size(1);
    tmp_array.resize_fast(mm);
    constexpr int kernel_size = n_rows_1d > 0 ? n_rows_1d : 0;
    internal::
      EvaluatorTensorProduct<internal::evaluate_general, order, kernel_size, kernel_size, Number>
                   eval(AlignedVector<Number>{},
             AlignedVector<Number>{},
             AlignedVector<Number>{},
             std::max(left(0).size(0),
                      right(0).size(0)), // TODO size of left and right matrices differs
             std::max(left(0).size(1),
                      right(0).size(1))); // TODO size of left and right matrices differs
    Number *       tmp     = tmp_array.begin();
    const Number * src     = src_view.begin();
    Number *       dst     = dst_view.data();
    const Number * left_0  = &(left(0)(0, 0));
    const Number * right_0 = &(right(0)(0, 0));
    eval.template apply</*direction*/ 0, /*contract_over_rows*/ false, /*add*/ false>(right_0,
                                                                                      src,
                                                                                      tmp);
    eval.template apply<1, false, add>(left_0, tmp, dst);
    for(std::size_t r = 1; r < left_or_mass.size(); ++r)
    {
      const Number * left_r  = &(left(r)(0, 0));
      const Number * right_r = &(right(r)(0, 0));
      eval.template apply<0, false, false>(right_r, src, tmp);
      eval.template apply<1, false, true>(left_r, tmp, dst);
    }
  }

  // bool
  // check_n_rows_1d(const std::vector<Table<2, Number>> & tables)
  // {
  //   if(n_rows_1d == -1)
  //     return true;
  //   if(n_rows_1d > 0)
  //     return std::all_of(tables.cbegin(), tables.cend(), [](const auto & tab) {
  //       return tab.size(0) == static_cast<unsigned>(n_rows_1d) &&
  //              tab.size(1) == static_cast<unsigned>(n_rows_1d);
  //     });
  //   return false;
  // }

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;
};

} // namespace Tensors

#endif // TENSOR_PRODUCT_MATRIX_H_
