/*
 * matrix_helper.h
 *
 *  Created on: Feb 17, 2020
 *      Author: witte
 */

#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_

#include <deal.II/lac/lapack_full_matrix.h>

#include "generic_functionalities.h"
#include "tensors.h"

using namespace dealii;



template<typename Number>
struct InverseTable
{
  using value_type                         = Number;
  using scalar_value_type                  = typename ExtractScalarType<Number>::type;
  static constexpr unsigned int macro_size = get_macro_size<Number>();

  InverseTable() = default;

  InverseTable(const Table<2, Number> & matrix_in)
  {
    reinit(matrix_in);
  }

  void
  reinit(const Table<2, Number> & matrix_in)
  {
    Assert(matrix_in.size(0) == matrix_in.size(1), ExcMessage("Matrix is not square."));
    clear();

    auto inverses = std::make_shared<std::array<LAPACKFullMatrix<scalar_value_type>, macro_size>>();
    const unsigned int n_rows = matrix_in.size(0);
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      auto & inverse = (*inverses)[lane];
      inverse.reinit(n_rows);
      inverse = table_to_fullmatrix(matrix_in, lane);
      inverse.invert();
    }
    /// ALTERNATIVE: FullMatrix
    // auto inverses = std::make_shared<std::array<FullMatrix<scalar_value_type>, macro_size>>();
    // const unsigned int n_rows = matrix_in.size(0);
    // for(auto lane = 0U; lane < macro_size; ++lane)
    // {
    //   auto & inverse = (*inverses)[lane];
    //   inverse.reinit(n_rows, n_rows);
    //   inverse.invert(table_to_fullmatrix(matrix_in, lane));
    // }
    this->inverses = inverses;
  }

  void
  clear()
  {
    inverses.reset();
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl(dst_view, src_view);
  }

  void
  vmult_impl(const ArrayView<scalar_value_type> &       dst_view,
             const ArrayView<const scalar_value_type> & src_view,
             const unsigned int                         lane = 0) const
  {
    Vector<scalar_value_type> dst(dst_view.size()), src(src_view.cbegin(), src_view.cend());
    const auto &              inverse = (*inverses)[lane];
    inverse.vmult(dst, src);
    std::copy(dst.begin(), dst.end(), dst_view.begin());
  }

  void
  vmult_impl(const ArrayView<VectorizedArray<scalar_value_type>> &       dst_view,
             const ArrayView<const VectorizedArray<scalar_value_type>> & src_view) const
  {
    Vector<scalar_value_type> dst_lane(dst_view.size()), src_lane(src_view.size());
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      std::transform(src_view.cbegin(),
                     src_view.cend(),
                     src_lane.begin(),
                     [lane](const auto & value) { return value[lane]; });
      const auto src_view_lane = make_array_view(src_lane);
      const auto dst_view_lane = make_array_view(dst_lane);
      vmult_impl(dst_view_lane, src_view_lane, lane);
      for(auto i = 0U; i < dst_lane.size(); ++i)
        dst_view[i][lane] = dst_lane[i];
    }
  }

  std::shared_ptr<const std::array<LAPACKFullMatrix<scalar_value_type>, macro_size>> inverses;
  /// ALTERNATIVE: FullMatrix
  // std::shared_ptr<const std::array<FullMatrix<scalar_value_type>, macro_size>> inverses;
};



template<typename PatchOperator>
struct PatchMatrix
{
  using scalar_value_type                  = typename PatchOperator::value_type;
  using value_type                         = VectorizedArray<scalar_value_type>;
  static constexpr unsigned int macro_size = VectorizedArray<scalar_value_type>::n_array_elements;

  void
  reinit(const PatchOperator * patch_operator_in, const unsigned int patch_id)
  {
    patch_operator   = patch_operator_in;
    current_patch_id = patch_id;
  }

  unsigned int
  m() const
  {
    return patch_operator->m(current_patch_id);
  }

  unsigned int
  n() const
  {
    return patch_operator->n(current_patch_id);
  }

  void
  vmult(const ArrayView<VectorizedArray<scalar_value_type>> &       dst_view,
        const ArrayView<const VectorizedArray<scalar_value_type>> & src_view) const
  {
    patch_operator->vmult(dst_view, src_view, current_patch_id);
  }

  void
  apply_inverse(const ArrayView<VectorizedArray<scalar_value_type>> &       dst_view,
                const ArrayView<const VectorizedArray<scalar_value_type>> & src_view) const
  {
    AssertThrow(false, ExcMessage("todo"));
    // TODO
    // patch_operator.apply_inverse(dst_view, src_view, current_patch_id);
  }

  Table<2, VectorizedArray<scalar_value_type>>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  FullMatrix<scalar_value_type>
  as_fullmatrix(const unsigned int lane)
  {
    return table_to_fullmatrix(as_table(), lane);
  }

  const PatchOperator * patch_operator   = nullptr;
  unsigned int          current_patch_id = numbers::invalid_unsigned_int;
};



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

    tmp_array.resize_fast(src_view.size());
    const ArrayView<Number> constrained_src_view(tmp_array.begin(), src_view.size());
    std::copy(src_view.begin(), src_view.end(), constrained_src_view.begin());
    zero_out_constrained_dofs_row(constrained_src_view);

    matrix.apply_inverse(dst_view, constrained_src_view);

    zero_out_constrained_dofs_col(dst_view);
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
