/**
 * test_utilites.h
 *
 * collection of helper functions
 *
 *  Created on: Nov 24, 2019
 *      Author: witte
 */

#ifndef TEST_UTILITIES_H_
#define TEST_UTILITIES_H_

#include <deal.II/base/utilities.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <gtest/gtest.h>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Util
{
template<typename Number>
constexpr Number numeric_eps = std::pow(10, std::log10(std::numeric_limits<Number>::epsilon()) / 2);

struct PrintFormat
{
  static constexpr unsigned int precision   = 3;
  static constexpr bool         scientific  = true;
  static constexpr unsigned int width       = 0;
  static constexpr char const * zero_string = " ";
  static constexpr double       denominator = 1.;
  static constexpr double       threshold   = numeric_eps<double>;
  static constexpr unsigned int max_size    = 200;
};



/// Compare pair of matrices of FullMatrix type
template<typename Number>
void
compare_matrix(const FullMatrix<Number> & matrix,
               const FullMatrix<Number> & other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  ASSERT_EQ(matrix.m(), other.m()) << "mismatching number of rows";
  ASSERT_EQ(matrix.n(), other.n()) << "mismatching number of columns";

  std::ostringstream oss;
  const bool         print_details =
    pcout.is_active() && matrix.m() < PrintFormat::max_size && matrix.n() < PrintFormat::max_size;

  if(print_details)
  {
    oss << "Matrix:\n";
    matrix.print_formatted(oss,
                           PrintFormat::precision,
                           PrintFormat::scientific,
                           PrintFormat::width,
                           PrintFormat::zero_string,
                           PrintFormat::denominator,
                           PrintFormat::threshold);
    oss << "Reference matrix:\n";
    other.print_formatted(oss,
                          PrintFormat::precision,
                          PrintFormat::scientific,
                          PrintFormat::width,
                          PrintFormat::zero_string,
                          PrintFormat::denominator,
                          PrintFormat::threshold);
  }
  else
    oss << "...printing is suppressed!\n";

  pcout << oss.str();

  auto diff(matrix);
  diff.add(-1., other);
  const auto threshold =
    std::max(numeric_eps<Number> * other.frobenius_norm(), numeric_eps<Number>);
  EXPECT_PRED_FORMAT2(testing::DoubleLE, diff.frobenius_norm(), threshold) << oss.str();
}



/// Compare inverse matrix by multiplying with reference matrix, both of
/// FullMatrix type
template<typename Number>
void
compare_inverse_matrix(const FullMatrix<Number> & inverse_matrix,
                       const FullMatrix<Number> & other,
                       const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  LAPACKFullMatrix<Number> lapack_other;
  lapack_other.reinit(other.m());
  lapack_other = other;
  lapack_other.compute_inverse_svd();
  const auto & lapack_other_as_table = Tensors::lapack_matrix_to_table(lapack_other);
  const auto   inverse_other         = table_to_fullmatrix(lapack_other_as_table);

  std::ostringstream oss;
  if(pcout.is_active())
  {
    oss << "Inverse matrix:\n";
    inverse_matrix.print_formatted(oss,
                                   PrintFormat::precision,
                                   PrintFormat::scientific,
                                   PrintFormat::width,
                                   PrintFormat::zero_string,
                                   PrintFormat::denominator,
                                   PrintFormat::threshold);
    oss << "Reference inverse matrix (LAPACK):\n";
    inverse_other.print_formatted(oss,
                                  PrintFormat::precision,
                                  PrintFormat::scientific,
                                  PrintFormat::width,
                                  PrintFormat::zero_string,
                                  PrintFormat::denominator,
                                  PrintFormat::threshold);
  }
  else
    oss << "...printing is suppressed!\n";

  FullMatrix<Number> id(inverse_matrix.m(), inverse_matrix.n());
  inverse_matrix.mmult(id, other);
  if(pcout.is_active())
  {
    oss << "A^{-1} A:\n";
    id.print_formatted(oss,
                       PrintFormat::precision,
                       PrintFormat::scientific,
                       PrintFormat::width,
                       PrintFormat::zero_string,
                       PrintFormat::denominator,
                       PrintFormat::threshold);
  }

  for(auto i = 0U; i < id.m(); ++i)
  {
    EXPECT_NEAR(id(i, i), 1., numeric_eps<Number>);
    for(auto j = 0U; j < id.m(); ++j)
      if(i != j)
      {
        EXPECT_NEAR(id(i, j),
                    // std::numeric_limits<Number>::epsilon(),
                    0.,
                    numeric_eps<Number>);
      }
  }
  pcout << oss.str();
}

/// Compare two deal.II Vectors element-wise
template<typename Number>
void
compare_vector(const Vector<Number> &     vector,
               const Vector<Number> &     other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  AssertDimension(vector.size(), other.size());
  std::ostringstream oss;
  if(pcout.is_active())
  {
    oss << "Vector:\n";
    vector.print(oss, PrintFormat::precision, PrintFormat::scientific);
    oss << "Reference vector:\n";
    other.print(oss, PrintFormat::precision, PrintFormat::scientific);
  }
  else
    oss << "...printing is suppressed!\n";

  for(auto i = 0U; i < vector.size(); ++i)
  {
    const auto value       = vector[i];
    const auto other_value = other[i];
    const auto diff        = std::abs(value - other_value);
    const auto threshold   = numeric_eps<Number> * std::abs(other_value);
    EXPECT_PRED_FORMAT2(testing::DoubleLE, diff, threshold)
      << "diff " << diff << " exceeds threshold " << threshold << " at position " << i;
  }
  pcout << oss.str();
}


/// Compare two vectors of MPI parallel vector type.
template<typename Number = double>
void
compare_vector(const LinearAlgebra::distributed::Vector<Number> & vector,
               const LinearAlgebra::distributed::Vector<Number> & other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  AssertDimension(vector.size(), other.size());
  std::ostringstream oss;
  if(vector.size() <= PrintFormat::max_size)
  {
    oss << "Vector:\n";
    vector.print(oss, PrintFormat::precision, PrintFormat::scientific);
    oss << "Reference vector:\n";
    other.print(oss, PrintFormat::precision, PrintFormat::scientific);
  }
  else
    oss << "...printing is suppressed!\n";

  for(auto i = 0U; i < vector.local_size(); ++i)
  {
    const auto value       = vector.local_element(i);
    const auto other_value = other.local_element(i);
    const auto diff        = std::abs(value - other_value);
    const auto threshold   = numeric_eps<Number> * std::abs(other_value);
    EXPECT_PRED_FORMAT2(testing::DoubleLE, diff, threshold)
      << "diff " << diff << " exceeds threshold " << threshold << " at position " << i;
  }
  pcout << oss.str();
}


/// Compare two block vectors of MPI parallel vector type.
template<typename Number = double>
void
compare_vector(const LinearAlgebra::distributed::BlockVector<Number> & vector,
               const LinearAlgebra::distributed::BlockVector<Number> & other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  AssertDimension(vector.n_blocks(), other.n_blocks());
  for(auto b = 0U; b < vector.n_blocks(); ++b)
  {
    pcout << "Block: " << b << std::endl;
    compare_vector(vector.block(b), other.block(b), pcout);
  }
}


/// Compare two vectors with type VectorType element by element.
/// VectorType has to provide size() and operator[]() for element access
template<typename VectorType, typename Number = typename VectorType::value_type>
void
compare_vector(const VectorType &         vector,
               const VectorType &         other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  AssertDimension(vector.size(), other.size());
  std::ostringstream oss;
  if(pcout.is_active())
  {
    oss << "Vector:\n";
    vector.print(oss, PrintFormat::precision, PrintFormat::scientific);
    oss << "Reference vector:\n";
    other.print(oss, PrintFormat::precision, PrintFormat::scientific);
  }
  else
    oss << "...printing is suppressed!\n";

  for(auto i = 0U; i < vector.size(); ++i)
  {
    const auto value       = vector[i];
    const auto other_value = other[i];
    const auto diff        = std::abs(value - other_value);
    const auto threshold   = numeric_eps<Number> * std::abs(other_value);
    EXPECT_PRED_FORMAT2(testing::DoubleLE, diff, threshold)
      << "diff " << diff << " exceeds threshold " << threshold << " at position " << i;
  }
  pcout << oss.str();
}


template<typename Number>
void
print_matrix(const FullMatrix<Number> & matrix,
             const std::string &        description,
             const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  const bool do_print =
    pcout.is_active() && std::max(matrix.m(), matrix.n()) < PrintFormat::max_size;
  pcout << description << std::endl;
  if(do_print)
    matrix.print_formatted(pcout.get_stream());
  else
    pcout << "...printing is suppressed!";
}



template<typename Number>
Table<2, Number>
make_random_matrix(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> rndm_matrix;
  fill_matrix_with_random_values(rndm_matrix, n_rows, n_cols);
  return rndm_matrix;
}



template<typename Number>
Table<2, Number>
make_random_matrix_symm(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> rndm_matrix;
  fill_matrix_with_random_values(rndm_matrix, n_rows, n_cols);

  /// symmetrize
  Table<2, Number> matrix;
  matrix.reinit(n_rows, n_cols);
  for(auto i = 0U; i < n_rows; ++i)
    for(auto j = 0U; j < n_cols; ++j)
      matrix(i, j) = (rndm_matrix(i, j) + rndm_matrix(j, i)) / 2.;
  return matrix;
}



template<typename Number>
Table<2, Number>
make_random_matrix_spd(const unsigned int n_rows, const unsigned int n_cols)
{
  /// symmetric
  Table<2, Number> matrix = make_random_matrix_symm<Number>(n_rows, n_cols);
  /// positive definite?
  for(auto i = 0U; i < std::min(n_rows, n_cols); ++i)
    matrix(i, i) += static_cast<Number>(std::max(n_rows, n_cols));
  return matrix;
}



template<typename Number>
Table<2, Number>
make_one_matrix(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> matrix;
  matrix.reinit(n_rows, n_cols);
  matrix.fill(static_cast<Number>(1.));
  return matrix;
}



template<typename Number>
Table<2, Number>
make_identity_matrix(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> matrix;
  matrix.reinit(n_rows, n_cols);
  for(auto i = 0U; i < std::min(n_rows, n_cols); ++i)
    matrix(i, i) = static_cast<Number>(1.);
  return matrix;
}



/// Convert any array-type into a tuple
template<typename Array, std::size_t... I>
auto
a2t_impl(const Array & a, std::index_sequence<I...>)
{
  return std::make_tuple(a[I]...);
}

/// Convert std::array into a tuple
template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
auto
make_tuple(const std::array<T, N> & a)
{
  return a2t_impl(a, Indices{});
}

template<typename Func, typename... Args>
struct PackExpansionDelayed
{
  std::tuple<Args...> params;

  template<typename Indices = std::make_index_sequence<sizeof...(Args)>>
  auto
  eval()
  {
    return eval_impl(Indices{});
  }

  template<std::size_t... index>
  auto eval_impl(std::index_sequence<index...>)
  {
    return Func{}(std::get<index>(params)...);
  }
};

template<typename T, typename... Ts>
constexpr auto
make_array(Ts... args)
{
  static_assert(std::conjunction_v<std::is_convertible<T, Ts>...>,
                "At least one type in Ts is not convertible to T.");
  std::array<T, sizeof...(Ts)> array = {args...};
  return array;
}

template<auto... args>
struct NonTypeParams
{
  static constexpr std::size_t                   size   = sizeof...(args);
  static constexpr std::tuple<decltype(args)...> params = {args...};

  template<int index>
  static constexpr auto
  value()
  {
    return std::get<index>(params);
  }

  template<typename T>
  static constexpr auto
  value()
  {
    return std::get<T>(params);
  }
};

template<class... Types>
struct TypeList
{
  template<std::size_t N>
  using type = typename std::tuple_element<N, std::tuple<Types...>>::type;
};
} // end namespace Util

#endif /* TEST_UTILITIES_H_ */
