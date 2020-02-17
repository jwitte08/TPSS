/*
 * matrix_helper.h
 *
 *  Created on: Feb 17, 2020
 *      Author: witte
 */

#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_

#include <deal.II/lac/lapack_full_matrix.h>

#include "tensors.h"

using namespace dealii;



/**
 * Matrix wrapper that treats constraints arising from strong homogeneous
 * Dirichlet boundary conditions on a local index set.
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
struct ConstrainedMatrix
{
  using value_type                         = Number;
  using scalar_value_type                  = typename ExtractScalarType<Number>::type;
  static constexpr unsigned int macro_size = get_macro_size<Number>();

  ConstrainedMatrix() = default;

  ~ConstrainedMatrix() = default;

  ConstrainedMatrix &
  operator=(const ConstrainedMatrix & other) = delete;

  unsigned int
  m() const
  {
    return matrix.m();
  }

  unsigned int
  n() const
  {
    return matrix.n();
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    tmp_array.resize_fast(src_view.size());
    const ArrayView<Number> constrained_src_view(tmp_array.begin(), src_view.size());
    std::copy(src_view.begin(), src_view.end(), constrained_src_view.begin());
    zero_out_constrained_dofs_col(constrained_src_view);

    matrix.vmult(dst_view, constrained_src_view);

    zero_out_constrained_dofs_row(dst_view);
  }

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    matrix.vmult_add(dst_view, src_view);
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    AssertDimension(dst_view.size(), src_view.size());
    Assert(constrained_dof_indices_row == constrained_dof_indices_col,
           ExcMessage("Row and column constraints don't coincide."));

    // tmp_array.resize_fast(src_view.size());
    // const ArrayView<Number> constrained_src_view(tmp_array.begin(), src_view.size());
    // std::copy(src_view.begin(), src_view.end(), constrained_src_view.begin());
    // zero_out_constrained_dofs_row(constrained_src_view);

    // matrix.apply_inverse(dst_view, constrained_src_view);
    matrix.apply_inverse(dst_view, src_view);

    // zero_out_constrained_dofs_col(dst_view);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
  }

  MatrixType                                     matrix;
  std::array<std::set<unsigned int>, macro_size> constrained_dof_indices_row;
  std::array<std::set<unsigned int>, macro_size> constrained_dof_indices_col;
  scalar_value_type                              diagonal_factor = 1.;

  void
  zero_out_constrained_dofs_row(const ArrayView<Number> & dst_view) const
  {
    zero_out_constrained_dofs_impl<true>(dst_view);
  }

  void
  zero_out_constrained_dofs_col(const ArrayView<Number> & dst_view) const
  {
    zero_out_constrained_dofs_impl<false>(dst_view);
  }

  template<bool is_row>
  void
  zero_out_constrained_dofs_impl(const ArrayView<Number> & dst_view) const
  {
    const auto & constrained_dof_indices =
      is_row ? constrained_dof_indices_row : constrained_dof_indices_col;
    for(auto lane = 0U; lane < macro_size; ++lane)
      for(const auto dof_index : constrained_dof_indices[lane])
        dst_view[dof_index][lane] = 0.;
  }

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};



#endif // MATRIX_HELPER_H_
