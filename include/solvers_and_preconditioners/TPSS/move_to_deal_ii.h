/*
 * move_to_deal_ii.h
 *
 *  Created on: Jun 10, 2020
 *      Author: witte
 */

#ifndef MOVE_TO_DEAL_II_H_
#define MOVE_TO_DEAL_II_H_

#include <deal.II/matrix_free/tensor_product_kernels.h>

namespace dealii
{
namespace internal
{
namespace My
{
/**
 * This function computes the d-mode product @p out for the tensor of vectors
 * @p in with matrix @p shape_data. The direction of contraction, that is the
 * 'd' in 'd-mode product', is @p direction.
 *
 * The destination and source tensor of vectors @p out and @p in are passed as
 * flat arrays, respectively, thus reading from @p in or writing into @out
 * requires the mode sizes of previous and subsequent dimensions. Fortunately,
 * the d-mode product is reduced to a 1-mode product of order-3 tensors, where
 * the new 0th mode is obtained by collapsing all previous modes to dimension
 * @p direction and the new 2nd mode by collapsing all subsequent modes. The
 * size of collapsed modes is @p n_pre or @p n_post, respectively.
 *
 * For the d-mode product with the transpose of @p shape_data, which is
 * triggered by @p contract_over_rows equal to true, we simply contract over
 * columns instead of explicitly transposing the values in @p shape_data (note
 * that the size of mode @p direction of the source tensor has to be @p n_rows
 * instead of @p n_columns and vice versa for the destination tensor). By
 * template parameter @p add we switch between adding values into @p out or
 * assigning values (that is overwritting previous values).
 */
template<int dim,
         typename Number,
         int  n_rows,
         int  n_columns,
         int  direction,
         int  n_pre,
         int  n_post,
         bool contract_over_rows,
         bool add,
         typename Number2 = Number>
void
contract_general_impl(const Number2 * DEAL_II_RESTRICT shape_data,
                      const Number *                   in,
                      Number *                         out,
                      const int                        n_rows_dynamic    = -1,
                      const int                        n_columns_dynamic = -1,
                      const int                        n_pre_dynamic     = -1,
                      const int                        n_post_dynamic    = -1)
{
  Assert(shape_data != nullptr,
         ExcMessage("The given array shape_data must not be the null pointer!"));
  AssertIndexRange(direction, dim);

  constexpr bool is_dynamic_mode = n_rows == -1 && n_columns == -1 && n_pre == -1 && n_post == -1;

  if(is_dynamic_mode) /// TODO static if-else
  {
    Assert(n_rows_dynamic > 0, ExcMessage("Any reasonable n_rows_dynamic is at least one."));
    Assert(n_columns_dynamic > 0, ExcMessage("Any reasonable n_columns_dynamic is at least one."));
    Assert(n_pre_dynamic > 0, ExcMessage("Any reasonable n_pre_dynamic is at least one."));
    Assert(n_post_dynamic > 0, ExcMessage("Any reasonable n_post_dynamic is at least one."));
  }
  else
  {
    /// TODO static_asserts if static if-else is used
    Assert(n_rows > 0, ExcMessage("Any reasonable n_rows is at least one."));
    Assert(n_columns > 0, ExcMessage("Any reasonable n_columns is at least one."));
    Assert(n_pre > 0, ExcMessage("Any reasonable n_pre is at least one."));
    Assert(n_post > 0, ExcMessage("Any reasonable n_post is at least one."));
  }

  constexpr int mm_static = contract_over_rows ? n_rows : n_columns;
  constexpr int nn_static = contract_over_rows ? n_columns : n_rows;

  const int mm =
    is_dynamic_mode ? (contract_over_rows ? n_rows_dynamic : n_columns_dynamic) : mm_static;
  const int nn =
    is_dynamic_mode ? (contract_over_rows ? n_columns_dynamic : n_rows_dynamic) : nn_static;
  const int n_columns_actual = is_dynamic_mode ? n_columns_dynamic : n_columns;
  const int stride           = is_dynamic_mode ? n_pre_dynamic : n_pre;
  const int n_blocks1        = is_dynamic_mode ? n_pre_dynamic : n_pre;
  const int n_blocks2        = is_dynamic_mode ? n_post_dynamic : n_post;

  Assert(n_columns_actual > 0, ExcMessage("Any reasonable n_columns_actual is at least one."));
  Assert(mm <= 128,
         ExcMessage("For dynamic mm intermediate array x is initialized with static size 128."));

  for(int i2 = 0; i2 < n_blocks2; ++i2)
  {
    for(int i1 = 0; i1 < n_blocks1; ++i1)
    {
      Number x[is_dynamic_mode ? 128 : mm_static];
      for(int i = 0; i < mm; ++i)
        x[i] = in[stride * i];
      for(int col = 0; col < nn; ++col)
      {
        Number2 val0;
        if(contract_over_rows == true)
          val0 = shape_data[col];
        else
          val0 = shape_data[col * n_columns_actual];
        Number res0 = val0 * x[0];
        for(int i = 1; i < mm; ++i)
        {
          if(contract_over_rows == true)
            val0 = shape_data[i * n_columns_actual + col];
          else
            val0 = shape_data[col * n_columns_actual + i];
          res0 += val0 * x[i];
        }
        if(add == false)
          out[stride * col] = res0;
        else
          out[stride * col] += res0;
      }

      // if (one_line == false)
      {
        ++in;
        ++out;
      }
    }
    // if (one_line == false)
    {
      in += stride * (mm - 1);
      out += stride * (nn - 1);
    }
  }
}

/**
 * Tensor product evaluator with static loop bounds (assuming isotropic tensors by default).
 */
template<int dim, typename Number, int n_rows, int n_columns, typename Number2 = Number>
struct EvaluatorTensorProduct
{
  EvaluatorTensorProduct(const int /*dummy*/ = -1, const int /*dummy*/ = -1){};

  template<int  direction,
           bool contract_over_rows,
           bool add,
           int  n_pre            = -1,
           int  n_post           = -1,
           int  n_rows_actual    = n_rows,
           int  n_columns_actual = n_columns>
  void
  apply(const Number2 * DEAL_II_RESTRICT matrix,
        const Number *                   tensor_of_vectors_src,
        Number *                         tensor_of_vectors_dst,
        const int /*n_rows_dynamic*/    = -1,
        const int /*n_columns_dynamic*/ = -1,
        const int /*n_pre_dynamic*/     = -1,
        const int /*n_post_dynamic*/    = -1)
  {
    constexpr bool is_isotropic_path = n_pre == -1 && n_post == -1;
    constexpr int  reverse_direction = dim - direction - 1;

    constexpr int n_pre_actual =
      is_isotropic_path ? Utilities::pow(n_columns_actual, direction) : n_pre;
    constexpr int n_post_actual =
      is_isotropic_path ? Utilities::pow(n_rows_actual, reverse_direction) : n_post;

    contract_general_impl<dim,
                          Number,
                          n_rows_actual,
                          n_columns_actual,
                          direction,
                          n_pre_actual,
                          n_post_actual,
                          contract_over_rows,
                          add,
                          Number2>(matrix, tensor_of_vectors_src, tensor_of_vectors_dst);
  }
};

/**
 * Tensor product evaluator with dynamic loop bounds.
 */
template<int dim, typename Number, typename Number2>
struct EvaluatorTensorProduct<dim, Number, -1, -1, Number2>
{
  static_assert(1 <= dim && dim <= 3, "This struct is implemented for up to 3 dimensions.");

  EvaluatorTensorProduct(const int n_rows_in, const int n_columns_in)
    : n_rows(n_rows_in), n_columns(n_columns_in)
  {
    Assert(n_rows > 0, ExcMessage("Any reasonable n_rows is at least one."));
    Assert(n_columns > 0, ExcMessage("Any reasonable n_columns is at least one."));
  }

  template<int  direction,
           bool contract_over_rows,
           bool add,
           int /*n_pre*/            = -1,
           int /*n_post*/           = -1,
           int /*n_rows_actual*/    = -1,
           int /*n_columns_actual*/ = -1>
  void
  apply(const Number2 * DEAL_II_RESTRICT matrix,
        const Number *                   tensor_of_vectors_src,
        Number *                         tensor_of_vectors_dst,
        const int                        n_pre        = -1,
        const int                        n_post       = -1,
        const int                        n_rows_in    = -1,
        const int                        n_columns_in = -1)
  {
    const int n_rows_actual    = n_rows_in == -1 ? n_rows : n_rows_in;
    const int n_columns_actual = n_columns_in == -1 ? n_columns : n_columns_in;

    const bool    is_isotropic_path = n_pre == -1 && n_post == -1;
    constexpr int reverse_direction = dim - direction - 1;

    const int n_pre_actual =
      is_isotropic_path ? Utilities::pow(n_columns_actual, direction) : n_pre;
    const int n_post_actual =
      is_isotropic_path ? Utilities::pow(n_rows_actual, reverse_direction) : n_post;

    // const int n_pre_actual = collapse_sizes_pre(direction, size0_src, size1_src, size2_src);
    // const int n_post_actual = collapse_sizes_post(direction, size0_dst, size1_dst, size2_dst);

    contract_general_impl<dim, Number, -1, -1, direction, -1, -1, contract_over_rows, add, Number2>(
      matrix,
      tensor_of_vectors_src,
      tensor_of_vectors_dst,
      n_rows_actual,
      n_columns_actual,
      n_pre_actual,
      n_post_actual);
  }

  const int n_rows;
  const int n_columns;
};

} // namespace My

} // namespace internal

} // namespace dealii

#endif // MOVE_TO_DEAL_II_H_
