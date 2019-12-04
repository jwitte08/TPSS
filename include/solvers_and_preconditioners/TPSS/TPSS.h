#ifndef TPSS_H
#define TPSS_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/evaluation_kernels.h>

#include <array>
#include <memory>
#include <type_traits>

template<typename Number>
struct ExtractScalarType
{
  using type = Number;
};

template<typename Number>
struct ExtractScalarType<typename dealii::VectorizedArray<Number>>
{
  using type = Number;
};

template<typename Number>
constexpr unsigned int
get_macro_size()
{
  using UnvectorizedNumber = typename ExtractScalarType<Number>::type;
  return (std::is_same<Number, UnvectorizedNumber>::value == true) ?
           1U :
           VectorizedArray<UnvectorizedNumber>::n_array_elements;
}

namespace TPSS
{
// *** ENUM: Patch Variant
enum class PatchVariant
{
  invalid,
  cell,
  vertex
};

template<int dim>
struct UniversalInfo
{
  static constexpr unsigned int
  n_cells_per_direction(const PatchVariant variant)
  {
    constexpr unsigned int n_cells_per_direction_[] = {0, 1, 2};
    return n_cells_per_direction_[(int)variant];
  }
  static constexpr unsigned int
  n_cells(const PatchVariant variant)
  {
    constexpr unsigned int n_cells_[] = {0, 1, 1 << dim};
    return n_cells_[(int)variant];
  }
};

// *** ENUM: Smoother Variant
enum class SmootherVariant
{
  invalid,
  additive,
  multiplicative
};

double
lookup_damping_factor(const TPSS::PatchVariant    patch_variant,
                      const TPSS::SmootherVariant smoother_variant,
                      const int                   dim)
{
  double r{99.};
  switch(smoother_variant)
  {
    case TPSS::SmootherVariant::multiplicative:
    {
      r = 1.;
      break;
    }
    case TPSS::SmootherVariant::additive:
    {
      switch(patch_variant)
      {
        case TPSS::PatchVariant::cell:
        {
          r = dim == 2 ? 0.7 : r;
          r = dim == 3 ? 0.7 : r;
          break;
        }
        case TPSS::PatchVariant::vertex:
        {
          r = dim == 2 ? 0.25 : r;
          r = dim == 3 ? 0.15 : r;
          break;
        }
        case TPSS::PatchVariant::invalid:
        {
          break;
        }
      }
    }
    case TPSS::SmootherVariant::invalid:
    {
      break;
    }
  }

  AssertThrow(r < 10., dealii::ExcMessage("This case is not listed!"));
  return r;
}

std::string
getstr_schwarz_variant(const TPSS::PatchVariant    patch_variant,
                       const TPSS::SmootherVariant smoother_variant,
                       const bool                  overlapping_vp = true)
{
  if(patch_variant == TPSS::PatchVariant::cell)
  {
    if(smoother_variant == TPSS::SmootherVariant::additive)
      return "ACP";
    else if(smoother_variant == TPSS::SmootherVariant::multiplicative)
      return "MCP";
    else
      AssertThrow(false, dealii::ExcMessage("Invalid smoother variant."));
  }
  else if(patch_variant == TPSS::PatchVariant::vertex)
  {
    if(smoother_variant == TPSS::SmootherVariant::additive)
      return overlapping_vp ? "AVP" : "AVPno";
    else if(smoother_variant == TPSS::SmootherVariant::multiplicative)
      return overlapping_vp ? "MVP" : "MVPno";
    else
      AssertThrow(false, dealii::ExcMessage("Invalid smoother variant."));
  }
  else
    AssertThrow(false, dealii::ExcMessage("Invalid patch variant."));
  return "invalid";
}

/**
 * Internal assembler for 1D dof-related matrices exploiting the tensor product form
 * of the shape functions
 */
template<int fe_order, int n_q_points, typename Number>
class MatrixEvaluator
{
public:
  MatrixEvaluator() = default;

  /**
   * Constructor, filling the shape function data fields.
   */
  MatrixEvaluator(const dealii::ArrayView<const Number> & shape_values_in,
                  const dealii::ArrayView<const Number> & shape_gradients_in,
                  const dealii::ArrayView<const Number> & dx_in)
    : shape_values(shape_values_in.data()),
      shape_gradients(shape_gradients_in.data()),
      dx(dx_in.data())
  {
    AssertDimension(dx_in.size(), n_q_points);
    AssertDimension(shape_values_in.size(), n_q_points * fe_order);
    AssertDimension(shape_gradients_in.size(), n_q_points * fe_order);
  }

  void
  reset(const dealii::ArrayView<const Number> & shape_values_in,
        const dealii::ArrayView<const Number> & shape_gradients_in,
        const dealii::ArrayView<const Number> & dx_in)
  {
    AssertDimension(dx_in.size(), n_q_points);
    AssertDimension(shape_values_in.size(), n_q_points * fe_order);
    AssertDimension(shape_gradients_in.size(), n_q_points * fe_order);

    shape_values    = shape_values_in.data();
    shape_gradients = shape_gradients_in.data();
    dx              = dx_in.data();
  }

  template<bool add>
  static void
  assemble(const dealii::ArrayView<const Number> & ansatz_data,
           const dealii::ArrayView<const Number> & test_data,
           const dealii::ArrayView<const Number> & dx,
           const dealii::ArrayView<Number> &       matrix)
  {
    AssertDimension(dx.size(), n_q_points);
    AssertDimension(ansatz_data.size(), n_q_points * fe_order);
    AssertDimension(test_data.size(), n_q_points * fe_order);
    AssertDimension(matrix.size(), fe_order * fe_order);
    apply_impl<add>(ansatz_data.data(), test_data.data(), dx.data(), matrix.data());
  }

  template<bool add>
  void
  assemble_mass(const dealii::ArrayView<Number> & matrix) const
  {
    AssertDimension(matrix.size(), fe_order * fe_order);
    Assert(shape_values != nullptr, dealii::ExcNotInitialized());
    Assert(shape_gradients != nullptr, dealii::ExcNotInitialized());
    Assert(dx != nullptr, dealii::ExcNotInitialized());
    apply_impl<add>(shape_values, shape_values, dx, matrix.data());
  }

  static void
  scale_matrix(const Number &                          factor,
               const dealii::ArrayView<const Number> & matrix_in,
               const dealii::ArrayView<Number> &       matrix_out)
  {
    AssertDimension(matrix_in.size(), fe_order * fe_order);
    AssertDimension(matrix_out.size(), fe_order * fe_order);
    scale_array2dim<fe_order, fe_order, false>(&factor, matrix_in.data(), matrix_out.data());
  }

private:
  template<int n_rows, int n_cols, bool variable>
  static void
  scale_array2dim(const Number * factor, const Number * array_in, Number * array_out)
  {
    constexpr int stride = n_cols;

    for(int r = 0; r < n_rows; ++r)
      for(int c = 0; c < n_cols; ++c)
        array_out[r * stride + c] = array_in[r * stride + c] * factor[variable ? c : 0];
  }

  template<bool add>
  static void
  apply_impl(const Number * ansatz_data,
             const Number * test_data,
             const Number * dx,
             Number *       matrix)
  {
    constexpr int stride = n_q_points;

    dealii::AlignedVector<Number> temp;
    temp.resize_fast(n_q_points * fe_order);
    for(int i = 0; i < fe_order; ++i)
      for(int q = 0; q < n_q_points; ++q)
        temp[i * stride + q] = test_data[i * stride + q] * dx[q];
    Number * test_data_dx = temp.begin();

    for(int j = 0; j < fe_order; ++j)
      for(int i = 0; i < fe_order; ++i)
      {
        Number m_ji;
        m_ji = 0.;
        for(int q = 0; q < n_q_points; ++q)
          m_ji += ansatz_data[i * stride + q] * test_data_dx[j * stride + q];
        if(add)
          *matrix += m_ji;
        else
          *matrix = m_ji;
        ++matrix;
      }
  }

  const Number * shape_values    = nullptr;
  const Number * shape_gradients = nullptr;
  const Number * dx              = nullptr;
};

/*** CLASS: Matrix Assembler for 1D problems ***/
/**
 * Generic MatrixAssembler framework
 */
template<dealii::internal::EvaluatorVariant variant,
         int                                dim,
         int                                fe_order,
         int                                n_q_points,
         typename number>
struct MatrixAssembler
{
};

/**
 * Internal assembler for 1D dof-related matrices exploiting the tensor product form
 * of the shape functions
 */
template<int dim, int fe_order, int n_q_points_1d, typename number>
struct MatrixAssembler<dealii::internal::EvaluatorVariant::evaluate_general,
                       dim,
                       fe_order,
                       n_q_points_1d,
                       number>
{
  static constexpr unsigned int dofs_per_cell = dealii::Utilities::pow(fe_order, dim);
  static constexpr unsigned int n_q_points    = dealii::Utilities::pow(n_q_points_1d, dim);

  /**
   * Empty constructor. Does nothing. Be careful when using 'unit_mass' and
   * related methods because they need to be filled with the other pointer
   */
  MatrixAssembler()
    : shape_values(nullptr),
      shape_gradients(nullptr),
      shape_hessians(nullptr),
      quad_weights(nullptr)
  {
  }

  /**
   * Constructor, taking the data from ShapeInfo
   */
  MatrixAssembler(const dealii::AlignedVector<number> & shape_values_,
                  const dealii::AlignedVector<number> & shape_gradients_,
                  const dealii::AlignedVector<number> & shape_hessians_,
                  const dealii::AlignedVector<number> & quad_weights_,
                  const unsigned int                    dummy1 = 0,
                  const unsigned int                    dummy2 = 0)
    : shape_values(shape_values_.begin()),
      shape_gradients(shape_gradients_.begin()),
      shape_hessians(shape_hessians_.begin()),
      quad_weights(quad_weights_.begin())
  {
    using namespace dealii;

    Assert(quad_weights_.size() > 0, dealii::ExcNotInitialized());
    // TODO how do we handle the face assembly case, where we use n_q_points_1d == 1 ?
    AssertDimension(static_cast<int>(quad_weights_.size()), n_q_points_1d);

    (void)dummy1;
    (void)dummy2;
  }

  template<bool add>
  void
  unit_mass(number * matrix) const
  {
    apply<add>(shape_values, shape_values, quad_weights, matrix);
  }

  template<bool add>
  static void
  apply(const number * ansatz_data, const number * test_data, const number * dx, number * matrix);

  const number * shape_values;
  const number * shape_gradients;
  const number * shape_hessians;
  const number * quad_weights;
};

} // end namespace TPSS

#include "TPSS.templates.h"

#endif // end inclusion guard
