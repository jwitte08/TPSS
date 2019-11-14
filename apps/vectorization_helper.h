/*
 * vectorization_helper.h
 *
 * Helper functionalities dealing with VectorizedArray.
 *
 *  Created on: Feb 03, 2018
 *      Author: witte
 */

#ifndef VECTORIZATIONHELPER_H_
#define VECTORIZATIONHELPER_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/vectorization.h>

#include <sstream>

using namespace dealii;

namespace VHelper
{
/**
 * Calls value() of the function @p func for each vectorization lane
 * of @p macro_point.
 */
template<int dim, typename Number>
VectorizedArray<Number>
value(const Function<dim, Number> &               func,
      const Point<dim, VectorizedArray<Number>> & macro_point,
      const unsigned int                          component = 0)
{
  VectorizedArray<Number> value;
  for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
  {
    Point<dim> single_point;
    for(unsigned int d = 0; d < dim; ++d)
      single_point[d] = macro_point[d][v];
    value[v] = func.value(single_point, component);
  }
  return value;
}

/**
 * Calls value() of the function @p func for each vectorization lane
 * of @p macro_point.
 */
template<int dim, typename Number>
Tensor<1, dim, VectorizedArray<Number>>
value(const TensorFunction<1, dim, Number> &      func,
      const Point<dim, VectorizedArray<Number>> & macro_point)
{
  Tensor<1, dim, VectorizedArray<Number>> value;
  for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
  {
    Point<dim> single_point;
    for(unsigned int d = 0; d < dim; ++d)
      single_point[d] = macro_point[d][v];
    const auto single_value = func.value(single_point);
    for(unsigned int d = 0; d < dim; ++d)
      value[d][v] = single_value[d];
  }
  return value;
}

template<int dim, typename Number>
std::string
print_point(const Point<dim, VectorizedArray<Number>> & macro_point)
{
  constexpr auto     macro_size = VectorizedArray<Number>::n_array_elements;
  std::ostringstream osstream;
  osstream << std::scientific << std::setprecision(2) << "{ ";
  for(unsigned int lane = 0; lane < macro_size; ++lane)
  {
    Point<dim> micro_point;
    for(unsigned int d = 0; d < dim; ++d)
      micro_point[d] = macro_point[d][lane];
    osstream << micro_point << (lane < macro_size - 1 ? " | " : "");
  }
  osstream << " }";
  return osstream.str();
}

template<typename Number>
std::string
print_value(const VectorizedArray<Number> & macro_value)
{
  constexpr auto     macro_size = VectorizedArray<Number>::n_array_elements;
  std::ostringstream osstream;
  osstream << std::scientific << std::setprecision(2) << "{ ";
  for(unsigned int lane = 0; lane < macro_size; ++lane)
  {
    const Number & micro_value = macro_value[lane];
    osstream << micro_value << (lane < macro_size - 1 ? " | " : "");
  }
  osstream << " }";
  return osstream.str();
}
} // namespace VHelper

#endif // VECTORIZATIONHELPER_H_
