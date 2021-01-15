/*
 * vectorization.h
 *
 *  Created on: Aug 07, 2020
 *      Author: witte
 */

#ifndef TPSS_VECTORIZATION_H_
#define TPSS_VECTORIZATION_H_

#include <deal.II/base/vectorization.h>

#include <bitset>
#include <type_traits>

using namespace dealii;



template<typename Number>
struct ZeroThresholdImpl
{
  static_assert(std::is_floating_point_v<Number>, "Number is no floating point type.");

  static const Number value;
};

template<typename Number>
const Number ZeroThresholdImpl<Number>::value = std::numeric_limits<Number>::epsilon() * 100.;



template<typename Number>
struct ZeroThresholdImpl<VectorizedArray<Number>>
{
  static const VectorizedArray<Number> value;
};

template<typename Number>
const VectorizedArray<Number> ZeroThresholdImpl<VectorizedArray<Number>>::value =
  ZeroThresholdImpl<Number>::value;



/**
 * Defines a threshold below which all positive numbers are treated as zero.
 */
template<typename Number>
static const Number zero_threshold = ZeroThresholdImpl<Number>::value;



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
           dealii::VectorizedArray<UnvectorizedNumber>::size();
}



template<typename Number>
Number &
scalar_value(Number & value, const unsigned int /*dummy*/ = 0)
{
  using UnvectorizedNumber = typename ExtractScalarType<Number>::type;
  static_assert(std::is_same<Number, UnvectorizedNumber>::value == true,
                "Implemented for unvectorized number type.");
  return value;
}



template<typename Number>
Number &
scalar_value(VectorizedArray<Number> & value, const unsigned int lane = 0)
{
  AssertIndexRange(lane, VectorizedArray<Number>::size());
  return value[lane];
}



template<typename Number>
const Number &
scalar_value(const Number & value, const unsigned int /*dummy*/ = 0)
{
  using UnvectorizedNumber = typename ExtractScalarType<Number>::type;
  static_assert(std::is_same<Number, UnvectorizedNumber>::value == true,
                "Implemented for unvectorized number type.");
  return value;
}



template<typename Number>
const Number &
scalar_value(const VectorizedArray<Number> & value, const unsigned int lane = 0)
{
  AssertIndexRange(lane, VectorizedArray<Number>::size());
  return value[lane];
}



/**
 * Returns true if the absolute value of @p value is below a certain
 * threshold. For vectorized arithmetic type @p Number this function returns
 * only true if all lanes are nearly zero.
 */
template<typename Number, typename scalar_value_type = typename ExtractScalarType<Number>::type>
bool
has_nearly_zero_abs_impl(const Number &            value,
                         const scalar_value_type & threshold = zero_threshold<scalar_value_type>)
{
  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const bool is_nearly_zero = std::abs(scalar_value(value, lane)) < threshold;
    if(!is_nearly_zero)
      return false;
  }
  return true;
}



/**
 * Returns true if the absolute value of @p value is below a certain
 * threshold. For vectorized arithmetic type @p Number this function returns
 * only true if all lanes are nearly zero.
 */
template<typename Number>
bool
has_nearly_zero_abs(const Number & value)
{
  return has_nearly_zero_abs_impl(value);
}



template<typename Number, typename scalar_value_type = typename ExtractScalarType<Number>::type>
Number
inverse_scalar_if_impl(const Number &          scalar,
                       const scalar_value_type threshold = zero_threshold<scalar_value_type>)
{
  Number inverse_scalar(0.);
  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
    scalar_value(inverse_scalar, lane) =
      has_nearly_zero_abs_impl(scalar_value(scalar, lane), threshold) ?
        static_cast<scalar_value_type>(0.) :
        1. / scalar_value(scalar, lane);
  return inverse_scalar;
}



/**
 * Returns the inverse of @p scalar. For vectorized arithmetic type @p Number
 * each lane of @p scalar being nearly zero is set to zero in the returned
 * inverse scalar (avoiding a division by zero). This procedure helps to write
 * vectorized iterative algorithms reaching a state where at least one lane
 * divides by zero before the remaining lanes have reached their stopping
 * criterion. What "nearly zero" means is defined by the free function
 * has_nearly_zero_abs().
 */
template<typename Number>
Number
inverse_scalar_if(const Number & scalar)
{
  return inverse_scalar_if_impl(scalar);
}



template<typename Number>
std::bitset<VectorizedArray<Number>::size()>
less_than(const VectorizedArray<Number> & lhs, const VectorizedArray<Number> & rhs)
{
  std::bitset<VectorizedArray<Number>::size()> flag;
  for(auto lane = 0U; lane < VectorizedArray<Number>::size(); ++lane)
    flag[lane] = lhs[lane] < rhs[lane];
  return flag;
}



template<typename Number>
std::bitset<1>
less_than(const Number & lhs, const Number & rhs)
{
  std::bitset<1> flag;
  flag[0] = lhs < rhs;
  return flag;
}



template<typename NumberType>
bool
less_than_all_lanes(const NumberType & lhs, const NumberType & rhs)
{
  const auto & flag = less_than(lhs, rhs);
  return flag.all();
}



template<typename Number>
Number
zero_out_negative_value(const Number & scalar)
{
  return scalar < 0. ? zero_threshold<Number> : scalar;
}



template<typename Number>
VectorizedArray<Number>
zero_out_negative_value(const VectorizedArray<Number> & scalar)
{
  VectorizedArray<Number> positive_scalar;
  for(auto lane = 0U; lane < VectorizedArray<Number>::size(); ++lane)
    positive_scalar[lane] = zero_out_negative_value<Number>(scalar[lane]);
  return positive_scalar;
}



template<typename Number>
std::string
varray_to_string(const VectorizedArray<Number> & array)
{
  std::ostringstream osstream;
  osstream << "[";
  constexpr auto n_elems = VectorizedArray<Number>::size();
  for(unsigned int k = 0; k < n_elems - 1; ++k)
    osstream << array[k] << "|";
  osstream << array[n_elems - 1] << "]";
  return osstream.str();
}



template<typename Number>
std::string
varray_to_string(const Number & value)
{
  std::ostringstream osstream;
  osstream << "[" << value << "]";
  return osstream.str();
}



/**
 * Extracts and converts the matrix associated to the lane @p lane
 * of the vectorized matrix @p table into the FullMatrix format.
 */
template<typename Number>
FullMatrix<Number>
vectorized_table_to_fullmatrix(const Table<2, VectorizedArray<Number>> & table,
                               const unsigned int                        lane = 0)
{
  AssertIndexRange(lane, VectorizedArray<Number>::size());
  FullMatrix<Number> matrix{table.n_rows(), table.n_cols()};
  for(unsigned int i = 0; i < table.n_rows(); ++i)
    for(unsigned int j = 0; j < table.n_cols(); ++j)
      matrix(i, j) = (table(i, j))[lane];
  return matrix;
}



template<typename Number>
FullMatrix<Number>
table_to_fullmatrix(const Table<2, VectorizedArray<Number>> & table, const unsigned int lane = 0)
{
  return vectorized_table_to_fullmatrix(table, lane);
}



template<typename Number>
FullMatrix<Number>
table_to_fullmatrix(const Table<2, Number> & table, const unsigned int dummy = 0)
{
  (void)dummy;
  FullMatrix<Number> matrix{table.n_rows(), table.n_cols()};
  for(unsigned int i = 0; i < table.n_rows(); ++i)
    for(unsigned int j = 0; j < table.n_cols(); ++j)
      matrix(i, j) = table(i, j);
  return matrix;
}



template<typename Number>
std::vector<typename ExtractScalarType<Number>::type>
alignedvector_to_vector(const AlignedVector<Number> & avec, const unsigned int lane)
{
  std::vector<typename ExtractScalarType<Number>::type> vec;
  std::transform(avec.begin(), avec.end(), std::back_inserter(vec), [lane](const auto & val) {
    return scalar_value(val, lane);
  });
  return vec;
}

#endif
