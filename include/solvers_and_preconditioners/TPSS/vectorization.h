/*
 * vectorization.h
 *
 *  Created on: Aug 07, 2020
 *      Author: witte
 */

#ifndef TPSS_VECTORIZATION_H_
#define TPSS_VECTORIZATION_H_

// #include <deal.II/base/conditional_ostream.h>
// #include <deal.II/base/mpi.h>
// #include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <bitset>
#include <type_traits>


using namespace dealii;



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

#endif
