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

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Util
{
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
                "One type in Ts is not convertible to T.");
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
