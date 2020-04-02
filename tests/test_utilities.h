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
#include <deal.II/lac/lapack_full_matrix.h>

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Util
{
struct PrintFormat
{
  static constexpr unsigned int precision   = 3;
  static constexpr bool         scientific  = true;
  static constexpr unsigned int width       = 0;
  static constexpr char const * zero_string = " ";
  static constexpr double       denominator = 1.;
  static constexpr double       threshold   = std::numeric_limits<double>::epsilon() * 100.;
};

/// Compare pair of matrices of FullMatrix type
template<typename Number>
void
compare_matrix(const FullMatrix<Number> & matrix,
               const FullMatrix<Number> & other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  std::ostringstream oss;
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
  auto diff(matrix);
  diff.add(-1., other);
  const double n_entries = other.m() * other.n();
  EXPECT_PRED_FORMAT2(testing::FloatLE,
                      diff.frobenius_norm(),
                      std::numeric_limits<Number>::epsilon() * 100. *
                        std::max(n_entries, other.frobenius_norm()))
    << oss.str();
  pcout << oss.str();
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

  FullMatrix<Number> id(inverse_matrix.m(), inverse_matrix.n());
  inverse_matrix.mmult(id, other);
  oss << "A^{-1} A:\n";
  id.print_formatted(oss,
                     PrintFormat::precision,
                     PrintFormat::scientific,
                     PrintFormat::width,
                     PrintFormat::zero_string,
                     PrintFormat::denominator,
                     PrintFormat::threshold);
  const double n_entries = id.m() * id.n();
  for(auto i = 0U; i < id.m(); ++i)
  {
    EXPECT_NEAR(id(i, i), 1., std::numeric_limits<Number>::epsilon() * 100. * n_entries);
    for(auto j = 0U; j < id.m(); ++j)
      if(i != j)
      {
        EXPECT_NEAR(id(i, j),
                    std::numeric_limits<Number>::epsilon(),
                    std::numeric_limits<Number>::epsilon() * 100. * n_entries);
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
  oss << "Vector:\n";
  vector.print(oss, PrintFormat::precision, PrintFormat::scientific);
  oss << "Reference vector:\n";
  other.print(oss, PrintFormat::precision, PrintFormat::scientific);
  for(auto i = 0U; i < vector.size(); ++i)
  {
    const auto value       = vector[i];
    const auto other_value = other[i];
    const auto diff        = std::abs(value - other_value);
    EXPECT_PRED_FORMAT2(testing::FloatLE,
                        diff,
                        std::numeric_limits<Number>::epsilon() * std::max(1000., other_value))
      << oss.str();
  }
  pcout << oss.str();
}

/// Compare two vectors of VectorType element-wise.
/// VectorType has to provide size() and operator[]() for element access
template<typename VectorType, typename Number = typename VectorType::value_type>
void
compare_vector(const VectorType &         vector,
               const VectorType &         other,
               const ConditionalOStream & pcout = ConditionalOStream(std::cout, true))
{
  AssertDimension(vector.size(), other.size());
  std::ostringstream oss;
  oss << "Vector:\n";
  vector.print(oss, PrintFormat::precision, PrintFormat::scientific);
  oss << "Reference vector:\n";
  other.print(oss, PrintFormat::precision, PrintFormat::scientific);
  for(auto i = 0U; i < vector.size(); ++i)
  {
    const auto value       = vector[i];
    const auto other_value = other[i];
    const auto diff        = std::abs(value - other_value);
    EXPECT_PRED_FORMAT2(testing::FloatLE,
                        diff,
                        std::numeric_limits<Number>::epsilon() * std::max(1000., other_value))
      << oss.str();
  }
  pcout << oss.str();
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
  auto
  eval_impl(std::index_sequence<index...>)
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
