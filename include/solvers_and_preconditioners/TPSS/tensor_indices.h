/*
 * tensor_indices.h
 *
 *  Created on: Dec 06, 2018
 *      Author: witte
 */

#ifndef TENSOR_INDICES_H_
#define TENSOR_INDICES_H_

// #include "generic_functionalities.h"
// #include "vectorization.h"

#include <array>
#include <sstream>

using namespace dealii;



namespace Tensors
{
template<int order, typename IntType = unsigned int>
std::string
multiindex_to_string(const std::array<IntType, order> multiindex)
{
  std::ostringstream osstream;
  osstream << "(";
  static_assert(order > 0, "0-order multiindex is not permitted.");
  for(IntType k = 0; k < multiindex.size() - 1; ++k)
    osstream << multiindex[k] << ", ";
  osstream << multiindex.back() << ")";
  return osstream.str();
}


/*
 * transforms an (anisotropic) multi-index into the canonical uni-index with
 * respect to lexicographical order. That is the first index of the multi-index
 * runs faster than second and so on.
 *
 * order : the order of the multi-index
 * sizes  : (anisotropic) size of each independent variable (mode)
 */
template<int order, typename IntType = unsigned int>
IntType
multi_to_uniindex(const std::array<IntType, order> & multiindex,
                  const std::array<IntType, order> & sizes)
{
  for(IntType k = 0; k < multiindex.size(); ++k)
    AssertIndexRange(multiindex[k], sizes[k]);
  IntType uniindex{0};
  for(int k = order - 1; k >= 0; --k)
  {
    // has no effect on purpose for k == order-1 (uniindex is zero)
    uniindex *= sizes[k];
    uniindex += multiindex[k];
  }
  const auto n_elem = std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<IntType>());
  (void)n_elem;
  AssertIndexRange(uniindex, n_elem);

  return uniindex;
}


/*
 * transforms an (isotropic) multi-index into the canonical uni-index with
 * respect to lexicographical order. That is the first index of the multi-index
 * runs faster than second and so on.
 *
 * order : the order of the multi-index
 * size  : isotropic size of each index set (mode)
 */
template<int order, typename IntType = unsigned int>
IntType
multi_to_uniindex(const std::array<IntType, order> & multiindex, const IntType size)
{
  std::array<IntType, order> sizes;
  sizes.fill(size);
  return multi_to_uniindex<order>(multiindex, sizes);
}


/*
 * transforms an uni-index into the canonical (anisotropic) multi-index with
 * respect to lexicographical order. That is the first index of the multi-index
 * runs faster than second and so on.
 *
 * order : the order of the multi-index
 * sizes : sizes of each independent variable (mode)
 */
template<int order, typename IntType = unsigned int>
std::array<IntType, order>
uni_to_multiindex(IntType index, const std::array<IntType, order> & sizes)
{
  const auto n_elem = std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<IntType>());
  (void)n_elem;
  AssertIndexRange(index, n_elem);
  std::array<IntType, order> multiindex;
  for(int k = 0; k < order; ++k)
  {
    multiindex[k] = index % sizes[k];
    index         = index / sizes[k];
  }
  Assert(index == 0, ExcMessage("Uni-index has remainder after multi-index extraction."));
  for(IntType k = 0; k < multiindex.size(); ++k)
    AssertIndexRange(multiindex[k], sizes[k]);

  return multiindex;
}


/*
 * transforms an uni-index into the canonical (isotropic) multi-index with
 * respect to lexicographical order. That is the first index of the multi-index
 * runs faster than second and so on.
 *
 * order : the order of the multi-index
 * size  : isotropic size of each index set (mode)
 */
template<int order, typename IntType = unsigned int>
std::array<IntType, order>
uni_to_multiindex(IntType index, const IntType size)
{
  std::array<IntType, order> sizes;
  sizes.fill(size);
  return uni_to_multiindex<order>(index, sizes);
}


/*
 * returns the fibre of (uni)-indices of an @p order -order (isotropic)
 * multiindex running along the @p mode mode with respect to lexicographical
 * order. That is the first index of the multi-index runs faster than second and
 * so on.
 *
 * order : the order of the multi-index
 * mode  : traversing mode of the fibre
 * size  : isotropic size of each index set (mode)
 */
template<int order, typename IntType = unsigned int>
std::vector<IntType>
index_fibre(const std::array<IntType, order - 1> index, const int mode, const IntType size)
{
  AssertIndexRange(mode, order);
  for(IntType k = 0; k < index.size(); ++k)
    AssertIndexRange(index[k], size);
  std::vector<IntType>       fibre;
  std::array<IntType, order> multiindex;
  // std::cout << Tensors::multiindex_to_string<order-1>(index) << std::endl;
  auto elem = index.cbegin();
  std::generate(multiindex.begin(), multiindex.end(), [&, mode, k = int{0}]() mutable {
    return (k++ != mode) ? *(elem++) : 0U;
  });
  for(IntType i = 0; i < size; ++i)
  {
    multiindex[mode] = i;
    // std::cout << Tensors::multiindex_to_string<order>(multiindex) << std::endl;
    fibre.push_back(multi_to_uniindex<order>(multiindex, size));
  }
  return fibre;
}



// TODO could be constexpr?
template<int order, typename IntType = unsigned int>
struct TensorHelper
{
  TensorHelper(const std::array<IntType, order> & sizes) : n(sizes)
  {
  }

  TensorHelper(const IntType size)
    : n([size]() {
        std::array<IntType, order> sizes;
        sizes.fill(size);
        return sizes;
      }())
  {
  }


  /**
   * If we think of the @p order dimensional index set as hypercube, then, we
   * have 2*order hyperfaces of order @p order-1 following a lexicographical
   * ordering, see for example dealii::GeometryInfo. In analogy to the
   * one-dimensional case we refer to hyperfaces as edge numbers @p edge_no. If
   * index @p index is in the interior of the imaginary hypercube, the set of
   * edge numbers is empty. If the index is located at a vertex dim edge numbers
   * are returned.
   */
  std::vector<unsigned int>
  get_edge_numbers(const IntType index) const
  {
    AssertIndexRange(index, n_flat());
    std::vector<unsigned int> edge_numbers;
    const auto &              multi_index = this->multi_index(index);
    for(auto mode = 0U; mode < order; ++mode)
    {
      const auto edge_no_1d = get_edge_no_1d(multi_index, mode);
      if(edge_no_1d != -1U)
        edge_numbers.emplace_back(2 * mode + edge_no_1d);
    }
    return edge_numbers;
  }

  /**
   * Returns the one-dimensional edge number of the @p mode'th mode of
   * multi-index @p multi_index. That is, 0 for the first, 1 for the last and -1
   * for all interior indices with respect to mode @p mode is reutrned.
   */
  unsigned int
  get_edge_no_1d(const std::array<IntType, order> & multi_index, const unsigned int mode) const
  {
    AssertIndexRange(mode, order);
    if(is_first_index_1d(multi_index, mode))
      return 0;
    else if(is_last_index_1d(multi_index, mode))
      return 1;
    return -1;
  }

  bool
  is_edge_index(const IntType index) const
  {
    AssertIndexRange(index, n_flat());
    const auto & multiindex = this->multi_index(index);
    for(auto mode = 0U; mode < order; ++mode)
      if(is_edge_index_1d(multiindex, mode))
        return true;
    return false;
  }

  bool
  is_first_index_1d(const std::array<IntType, order> & multi_index, const unsigned int mode) const
  {
    AssertIndexRange(mode, order);
    return multi_index[mode] == static_cast<IntType>(0);
  }

  bool
  is_last_index_1d(const std::array<IntType, order> & multi_index, const unsigned int mode) const
  {
    AssertIndexRange(mode, order);
    return multi_index[mode] == (size(mode) - 1);
  }

  bool
  is_edge_index_1d(const std::array<IntType, order> & multi_index, const unsigned int mode) const
  {
    AssertIndexRange(mode, order);
    return is_first_index_1d(multi_index, mode) || is_last_index_1d(multi_index, mode);
  }

  std::array<IntType, order>
  multi_index(const IntType index) const
  {
    return Tensors::uni_to_multiindex<order, IntType>(index, n);
  }

  IntType
  uni_index(const std::array<IntType, order> & multi_index) const
  {
    return Tensors::multi_to_uniindex<order, IntType>(multi_index, n);
  }

  std::vector<IntType>
  sliced_indices(const IntType index, const unsigned int mode) const
  {
    AssertThrow(order > 0, ExcMessage("Not implemented."));

    std::vector<IntType> indices;
    AssertIndexRange(mode, order);
    AssertIndexRange(index, size(mode));
    if(order == 1)
    {
      indices.emplace_back(index);
      return indices;
    }

    const auto restrict = [&](const std::array<IntType, order> & multiindex) {
      std::array<IntType, order - 1> slicedindex;
      for(auto m = 0U; m < mode; ++m)
        slicedindex[m] = multiindex[m];
      for(auto m = mode + 1; m < order; ++m)
        slicedindex[m - 1] = multiindex[m];
      return slicedindex;
    };
    const auto prolongate = [&](const std::array<IntType, order - 1> & slicedindex) {
      std::array<IntType, order> multiindex;
      for(auto m = 0U; m < mode; ++m)
        multiindex[m] = slicedindex[m];
      multiindex[mode] = index;
      for(auto m = mode + 1; m < order; ++m)
        multiindex[m] = slicedindex[m - 1];
      return multiindex;
    };

    TensorHelper<order - 1, IntType> slice(restrict(this->n));
    for(auto i = 0U; i < slice.n_flat(); ++i)
    {
      const auto sliced_index = slice.multi_index(i);
      const auto multi_index  = prolongate(sliced_index);
      indices.emplace_back(this->uni_index(multi_index));
    }
    return indices;
  }

  bool
  is_isotropic() const
  {
    for(auto direction = 0; direction < order; ++direction)
      if(size(0) != size(direction))
        return false;
    return true;
  }

  bool
  operator==(const TensorHelper<order, IntType> & other) const
  {
    return std::equal(n.cbegin(),
                      n.cend(),
                      other.n.cbegin(),
                      other.n.cend(),
                      [](const auto i, const auto j) { return i == j; });
  }

  IntType
  n_flat() const
  {
    return std::accumulate(n.cbegin(),
                           n.cend(),
                           static_cast<IntType>(1),
                           std::multiplies<IntType>());
  }

  IntType
  size(const unsigned int mode) const
  {
    AssertIndexRange(mode, order);
    return n[mode];
  }

  const std::array<IntType, order> &
  size() const
  {
    return n;
  }

  IntType
  collapsed_size_pre(const unsigned int direction) const
  {
    return std::accumulate(n.begin(), n.begin() + direction, 1, std::multiplies<IntType>{});
  }

  IntType
  collapsed_size_post(const unsigned int direction) const
  {
    return std::accumulate(n.begin() + direction + 1, n.end(), 1, std::multiplies<IntType>{});
  }

  const std::array<IntType, order> n;
};

} // namespace Tensors

#endif // TENSOR_INDICES_H_
