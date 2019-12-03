/*
 * tensors.h
 *
 *  Created on: Dec 06, 2018
 *      Author: witte
 */

#ifndef TENSORS_H_
#define TENSORS_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/tensor_product_matrix.h>

#include "TPSS.h"

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
 * transforms an (anisotropic) multi-index into the canonical
 * uni-index with respect to lexicographical order.
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
 * transforms an (isotropic) multi-index into the canonical
 * uni-index with respect to lexicographical order.
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
 * transforms an uni-index into the canonical (anisotropic)
 * multi-index with respect to lexicographical order.
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
 * transforms an uni-index into the canonical (isotropic)
 * multi-index with respect to lexicographical order.
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
 * returns the fibre of (uni)-indices of an @p order -order
 * (isotropic) multiindex running along the @p mode mode with
 * respect to lexicographical order.
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

/**
 * Converts a matrix into a two dimensional table. MatrixType has to fulfill
 * following interface:
 *
 * method m() returning number of rows
 * method n() returning number of cols
 * typedef value_type
 * method vmult(ArrayView,ArrayView)
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
Table<2, Number>
matrix_to_table(const MatrixType & matrix)
{
  Table<2, Number>      table(matrix.m(), matrix.n());
  AlignedVector<Number> e_j(matrix.n());
  AlignedVector<Number> col_j(matrix.m());
  for(unsigned int j = 0; j < matrix.n(); ++j)
  {
    e_j.fill(static_cast<Number>(0.));
    col_j.fill(static_cast<Number>(0.));
    e_j[j]                = static_cast<Number>(1.);
    const auto e_j_view   = make_array_view<const Number>(e_j.begin(), e_j.end());
    const auto col_j_view = make_array_view<Number>(col_j.begin(), col_j.end());
    matrix.vmult(col_j_view, e_j_view);
    for(unsigned int i = 0; i < matrix.m(); ++i)
      table(i, j) = col_j[i];
  }
  return table;
}

template<typename MatrixType1, typename MatrixType2 = MatrixType1>
void
insert_block(MatrixType1 &       dst,
             const MatrixType2 & src,
             const unsigned int  row_dst = 0,
             const unsigned int  col_dst = 0)
{
  AssertIndexRange(row_dst + src.m(), dst.m() + 1);
  AssertIndexRange(col_dst + src.n(), dst.n() + 1);
  for(unsigned i = 0; i < src.m(); ++i)
    for(unsigned j = 0; j < src.n(); ++j)
      dst(row_dst + i, col_dst + j) = src(i, j);
}

template<int dim, typename Number, int n_rows_1d = -1>
class TensorProductMatrix : public TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>
{
public:
  enum class State
  {
    invalid,
    basic,
    skd
  };

  using SKDMatrix = TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>;
  using SKDMatrix::value_type;

  TensorProductMatrix() = default;

  TensorProductMatrix(const std::vector<Table<2, Number>> & left_or_mass_in,
                      const std::vector<Table<2, Number>> & right_or_derivative_in,
                      const State                           state_in = State::basic)
  {
    reinit(left_or_mass_in, right_or_derivative_in, state_in);
  }

  void
  reinit(const std::vector<Table<2, Number>> & left_or_mass_in,
         const std::vector<Table<2, Number>> & right_or_derivative_in,
         const State                           state_in = State::basic)
  {
    Assert(
      left_or_mass_in.size() == right_or_derivative_in.size(),
      ExcMessage(
        "The number of left/mass matrices is not equal to the number of right/derivative matrices."));
    AssertThrow(check_n_rows_1d(left_or_mass_in) && check_n_rows_1d(right_or_derivative_in),
                ExcMessage(
                  "Not all univariate matrices inserted are of size (n_rows_1d x n_rows_1d)."));

    if(state_in == State::basic)
    {
      left_owned          = left_or_mass_in;
      left_or_mass        = make_array_view(left_owned);
      right_owned         = right_or_derivative_in;
      right_or_derivative = make_array_view(right_owned);
    }

    else if(state_in == State::skd)
    {
      Assert(left_or_mass_in.size() == dim,
             ExcMessage("The number of mass/derivative matrices and dimension differ."));
      std::array<Table<2, Number>, dim> mass_matrices;
      std::array<Table<2, Number>, dim> derivative_matrices;
      std::copy(left_or_mass_in.begin(), left_or_mass_in.end(), mass_matrices.begin());
      std::copy(right_or_derivative_in.begin(),
                right_or_derivative_in.end(),
                derivative_matrices.begin());
      SKDMatrix::reinit(mass_matrices, derivative_matrices);
      left_or_mass = make_array_view(SKDMatrix::mass_matrix.begin(), SKDMatrix::mass_matrix.end());
      right_or_derivative =
        make_array_view(SKDMatrix::derivative_matrix.begin(), SKDMatrix::derivative_matrix.end());
    }
    else
      AssertThrow(false, ExcMessage("Invalid state at initialization."));

    state = state_in;
  }

  unsigned int
  m() const
  {
    if(state == State::skd)
      return SKDMatrix::m();
    Assert(left_or_mass.size() > 0, ExcMessage("Not initialized."));
    const unsigned int m_left  = left(0).size(0);
    const unsigned int m_right = right(0).size(0);
    return m_left * m_right;
  }

  unsigned int
  n() const
  {
    if(state == State::skd)
      return SKDMatrix::n();
    Assert(left_or_mass.size() > 0, ExcMessage("Not initialized."));
    const unsigned int n_left  = left(0).size(1);
    const unsigned int n_right = right(0).size(1);
    return n_left * n_right;
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ false>(dst_view, src_view);
  }

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ true>(dst_view, src_view);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

protected:
  const Table<2, Number> &
  left(unsigned int r) const
  {
    AssertIndexRange(r, left_or_mass.size());
    return left_or_mass[r];
  }

  const Table<2, Number> &
  right(unsigned int r) const
  {
    AssertIndexRange(r, right_or_derivative.size());
    return right_or_derivative[r];
  }

  /**
   * An array view pointing to the left or mass matrices, respectively,
   * depending on the active state.
   */
  ArrayView<Table<2, Number>> left_or_mass;

  /**
   * A vector containing left matrices, that is left factors of the sum over
   * Kronecker products.
   */
  std::vector<Table<2, Number>> left_owned;

  /**
   * An array view pointing to the right or derivative matrices, respectively,
   * depending on the active state.
   */
  ArrayView<Table<2, Number>> right_or_derivative;

  /**
   * A vector containing right matrices, that is right factors of the sum over
   * Kronecker products.
   */
  std::vector<Table<2, Number>> right_owned;

  /**
   * The state switches which functionalities are faciliated:
   *
   * basic:  TensorProductMatrix
   * skd:    TensorProductMatrixSymmetricSum
   */
  State state = State::invalid;

private:
  template<bool add>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    if(state == State::basic)
      vmult_impl_basic_static<add>(dst_view, src_view);
    else if(state == State::skd)
    {
      AlignedVector<Number> initial_dst;
      if(add)
      {
        initial_dst.resize_fast(dst_view.size());
        std::copy(dst_view.cbegin(), dst_view.cend(), initial_dst.begin());
      }
      // const auto initial_dst_view = make_array_view(initial_dst.begin(),initial_dst.end());
      SKDMatrix::vmult(dst_view, src_view);
      if(add)
        std::transform(dst_view.cbegin(),
                       dst_view.cend(),
                       initial_dst.begin(),
                       dst_view.begin(),
                       [](const auto & elem1, const auto & elem2) { return elem1 + elem2; });
    }
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }


  template<bool add>
  void
  vmult_impl_basic_static(const ArrayView<Number> &       dst_view,
                          const ArrayView<const Number> & src_view) const
  {
    AssertDimension(dst_view.size(), m());
    AssertDimension(src_view.size(), n());
    std::lock_guard<std::mutex> lock(this->mutex);
    const unsigned int          mm =
      n_rows_1d > 0 ? Utilities::fixed_power<dim>(n_rows_1d) : right(0).size(0) * left(0).size(1);
    tmp_array.resize_fast(mm);
    constexpr int kernel_size = n_rows_1d > 0 ? n_rows_1d : 0;
    internal::
      EvaluatorTensorProduct<internal::evaluate_general, dim, kernel_size, kernel_size, Number>
                   eval(AlignedVector<Number>{},
             AlignedVector<Number>{},
             AlignedVector<Number>{},
             std::max(left(0).size(0),
                      right(0).size(0)), // TODO size of left and right matrices differs
             std::max(left(0).size(1),
                      right(0).size(1))); // TODO size of left and right matrices differs
    Number *       tmp     = tmp_array.begin();
    const Number * src     = src_view.begin();
    Number *       dst     = dst_view.data();
    const Number * left_0  = &(left(0)(0, 0));
    const Number * right_0 = &(right(0)(0, 0));
    eval.template apply</*direction*/ 0, /*contract_over_rows*/ false, /*add*/ false>(right_0,
                                                                                      src,
                                                                                      tmp);
    eval.template apply<1, false, add>(left_0, tmp, dst);
    for(std::size_t r = 1; r < left_or_mass.size(); ++r)
    {
      const Number * left_r  = &(left(r)(0, 0));
      const Number * right_r = &(right(r)(0, 0));
      eval.template apply<0, false, false>(right_r, src, tmp);
      eval.template apply<1, false, true>(left_r, tmp, dst);
    }
  }

  bool
  check_n_rows_1d(const std::vector<Table<2, Number>> & tables)
  {
    if(n_rows_1d == -1)
      return true;
    if(n_rows_1d > 0)
      return std::all_of(tables.cbegin(), tables.cend(), [](const auto & tab) {
        return tab.size(0) == static_cast<unsigned>(n_rows_1d) &&
               tab.size(1) == static_cast<unsigned>(n_rows_1d);
      });
    return false;
  }

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;
};

template<int dim, typename Number, int n_rows_1d = -1>
class BlockMatrixDiagonal
{
public:
  using matrix_type = TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>;
  using value_type  = typename matrix_type::value_type;

  /**
   * Deletes old and resizes to new block structure.
   */
  void
  resize(const std::size_t n_blocks)
  {
    blocks.clear();
    blocks.resize(n_blocks);
  }

  matrix_type &
  get_block(const std::size_t block_index)
  {
    AssertIndexRange(block_index, blocks.size());
    return blocks[block_index];
  }

  const matrix_type &
  get_block(const std::size_t block_index) const
  {
    AssertIndexRange(block_index, blocks.size());
    return blocks[block_index];
  }

  unsigned int
  m() const
  {
    const unsigned int n_rows =
      std::accumulate(blocks.begin(), blocks.end(), 0, [](const auto sum, const auto & matrix) {
        return sum + matrix.m();
      });
    return n_rows;
  }

  unsigned int
  n() const
  {
    const unsigned int n_columns =
      std::accumulate(blocks.begin(), blocks.end(), 0, [](const auto sum, const auto & matrix) {
        return sum + matrix.m();
      });
    return n_columns;
  }

  void
  vmult(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             matrix,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { matrix.vmult(dst, src); },
                     dst,
                     src);
  }

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             matrix,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { matrix.apply_inverse(dst, src); },
                     dst,
                     src);
  }

private:
  /**
   * Applies the function @p action to each matrix of the block
   * diagonal and the associated vector slices of @p dst and @p src.
   * The signature of the action function should be equivalent to
   * action(matrix_type& m, ArrayView<...> dst, ArrayView<const ...> src)
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const
  {
    AssertDimension(src.size(), n());
    AssertDimension(dst.size(), m());
    AssertThrow(src.size() == dst.size(), ExcNotImplemented()); // quadratic !
    AssertThrow(n_rows_1d == -1 || src.size() % n_rows_1d == 0,
                ExcMessage("Input vector sizes are not a multiple of the static size."));
    AssertThrow(!blocks.empty(), ExcMessage("Blocks are not initialized."));

    // *** apply sliced vectors to the associated block matrix
    std::size_t row_end = 0;
    std::size_t col_end = 0;
    for(const auto & matrix : blocks)
    {
      const auto row_start = row_end;
      const auto col_start = col_end;
      row_end += matrix.m();
      col_end += matrix.n();
      const ArrayView<Number>       dst_block(dst.begin() + row_start, matrix.m());
      const ArrayView<const Number> src_block(src.begin() + col_start, matrix.n());
      action(matrix, dst_block, src_block);
    }
    AssertThrow(row_end == dst.size() && col_end == src.size(),
                ExcMessage("Inconsistent slicing."));
  }

  /**
   * The vector containing the matrix blocks.
   */
  AlignedVector<matrix_type> blocks;
};

template<int dim, typename Number, int n_rows_1d = -1>
class BlockMatrix
{
public:
  using matrix_type = TensorProductMatrix<dim, Number, n_rows_1d>;
  using value_type  = typename matrix_type::value_type;

  /**
   * Deletes old and resizes to square block structure.
   */
  void
  resize(const std::size_t n_rows)
  {
    blocks.clear();
    blocks.resize(n_rows * n_rows);
    n_[0] = n_[1] = n_rows;
  }

  /**
   * Deletes old and resizes to new block structure.
   */
  void
  resize(const std::size_t n_rows, const std::size_t n_cols)
  {
    blocks.clear();
    blocks.resize(n_rows * n_cols);
    n_[0] = n_rows;
    n_[1] = n_cols;
  }

  matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index)
  {
    AssertIndexRange(row_index, n_block_rows());
    AssertIndexRange(col_index, n_block_cols());
    return blocks[block_index(row_index, col_index)];
  }

  const matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index) const
  {
    AssertIndexRange(row_index, n_block_rows());
    AssertIndexRange(col_index, n_block_cols());
    return blocks[block_index(row_index, col_index)];
  }

  std::size_t
  m(const std::size_t row_index) const
  {
    AssertIndexRange(row_index, n_block_rows());
    return get_block(row_index, 0).m();
  }

  std::size_t
  n(const std::size_t col_index) const
  {
    AssertIndexRange(col_index, n_block_cols());
    return get_block(0, col_index).n();
  }

  std::size_t
  m() const
  {
    Assert(check_row_sizes(), ExcMessage("Inconsistent number of rows of block matrices."));
    std::size_t n_rows = 0;
    for(std::size_t row_index = 0; row_index < n_block_rows(); ++row_index)
      n_rows += m(row_index);
    return n_rows;
  }

  std::size_t
  n() const
  {
    Assert(check_col_sizes(), ExcMessage("Inconsistent number of rows of block matrices."));
    std::size_t n_cols = 0;
    for(std::size_t col_index = 0; col_index < n_block_cols(); ++col_index)
      n_cols += n(col_index);
    return n_cols;
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  void
  vmult(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             block,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { block.vmult_add(dst, src); },
                     dst,
                     src);
  }

  // // TODO
  // void
  // apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  // {
  //   blockwise_action([](const matrix_type &             matrix,
  //                       const ArrayView<Number> &       dst,
  //                       const ArrayView<const Number> & src) { matrix.apply_inverse(dst, src); },
  //                    dst,
  //                    src);
  // }

  std::array<std::size_t, 2>
  size() const
  {
    return n_;
  }

  std::size_t
  n_block_rows() const
  {
    return n_[0];
  }

  std::size_t
  n_block_cols() const
  {
    return n_[1];
  }

private:
  std::size_t
  block_index(const std::size_t row_index, const std::size_t col_index) const
  {
    const std::array<std::size_t, 2> row_and_col = {row_index, col_index};
    return Tensors::multi_to_uniindex<2>(row_and_col, n_);
  }

  /**
   * Applies the function @p action to each matrix of the block
   * diagonal and the associated vector slices of @p dst and @p src.
   * The signature of the action function should be equivalent to
   * action(matrix_type& m, ArrayView<...> dst, ArrayView<const ...> src)
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const
  {
    AssertDimension(src.size(), n());
    AssertDimension(dst.size(), m());
    Assert(src.size() == dst.size(),
           ExcMessage("TODO BlockMatrix is not quadratic.")); // quadratic !
    Assert(n_rows_1d == -1 || src.size() % n_rows_1d == 0,
           ExcMessage("Input vector sizes are not a multiple of the static size."));
    Assert(!blocks.empty(), ExcMessage("Blocks are not initialized."));
    Assert(is_valid(), ExcMessage("Matrix sizes of blocks mismatch."));
    std::fill(dst.begin(), dst.end(), static_cast<Number>(0.));

    // *** apply sliced vectors to the associated block matrix
    std::size_t row_start = 0;
    for(std::size_t row = 0; row < n_block_rows(); ++row)
    {
      std::size_t col_start = 0;
      for(std::size_t col = 0; col < n_block_cols(); ++col)
      {
        const ArrayView<Number>       dst_block(dst.begin() + row_start, m(row));
        const ArrayView<const Number> src_block(src.begin() + col_start, n(col));
        action(get_block(row, col), dst_block, src_block);
        col_start += n(col);
      }
      Assert(col_start == src.size(), ExcMessage("Inconsistent slicing."));
      row_start += m(row);
    }
    Assert(row_start == dst.size(), ExcMessage("Inconsistent slicing."));
  }

  bool
  check_row_sizes() const
  {
    for(std::size_t row = 0; row < n_block_rows(); ++row)
      for(std::size_t col = 0; col < n_block_cols(); ++col)
        if(get_block(row, col).m() != get_block(row, 0).m())
          return false;
    return true;
  }

  bool
  check_col_sizes() const
  {
    for(std::size_t col = 0; col < n_block_cols(); ++col)
      for(std::size_t row = 0; row < n_block_rows(); ++row)
        if(get_block(row, col).n() != get_block(0, col).n())
          return false;
    return true;
  }

  bool
  is_valid() const
  {
    return check_row_sizes() && check_col_sizes();
  }

  /**
   * The number of blocks per row and column
   */
  std::array<std::size_t, 2> n_ = {0, 0};

  /**
   * The vector containing the matrix blocks.
   */
  AlignedVector<matrix_type> blocks;
};

/**
 * Compute the Kronecker product of two matrices. Each input
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeIn2, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
kronecker_product(MatrixTypeIn1 && left_matrix, MatrixTypeIn2 && right_matrix)
{
  auto && matrix1 = std::forward<MatrixTypeIn1>(left_matrix);
  auto && matrix0 = std::forward<MatrixTypeIn2>(right_matrix);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();
  const unsigned int n_rows1 = matrix1.n_rows();
  const unsigned int n_cols1 = matrix1.n_cols();
  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows1 * n_rows0, n_cols1 * n_cols0);

  for(unsigned int i1 = 0; i1 < n_rows1; ++i1)
    for(unsigned int j1 = 0; j1 < n_cols1; ++j1)
      for(unsigned int i0 = 0; i0 < n_rows0; ++i0)
        for(unsigned int j0 = 0; j0 < n_cols0; ++j0)
          matrix_out(i1 * n_rows0 + i0, j1 * n_cols0 + j0) = matrix1(i1, j1) * matrix0(i0, j0);

  return matrix_out;
}

/**
 * Computes the sum of two equally sized matrices. Each input
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeIn2, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
sum(MatrixTypeIn1 && left_matrix, MatrixTypeIn2 && right_matrix)
{
  using namespace dealii;
  auto && matrix1 = std::forward<MatrixTypeIn1>(left_matrix);
  auto && matrix0 = std::forward<MatrixTypeIn2>(right_matrix);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();
#ifdef DEBUG
  const unsigned int n_rows1 = matrix1.n_rows();
  const unsigned int n_cols1 = matrix1.n_cols();
  AssertDimension(n_rows0, n_rows1);
  AssertDimension(n_cols0, n_cols1);
#endif

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows0, n_cols0);

  for(unsigned int i = 0; i < n_rows0; ++i)
    for(unsigned int j = 0; j < n_cols0; ++j)
      matrix_out(i, j) = matrix1(i, j) + matrix0(i, j);

  return matrix_out;
}

/**
 * Returns the transpose of the input matrix @p matrix_in
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
// scale(MatrixTypeIn1 && matrix_in,const typename ExtractScalarType<typename
// MatrixTypeIn1::value_type>::type factor)
scale(const double factor, MatrixTypeIn1 && matrix_in) // TODO
{
  auto && matrix0 = std::forward<MatrixTypeIn1>(matrix_in);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows0, n_cols0);

  for(unsigned int i = 0; i < n_rows0; ++i)
    for(unsigned int j = 0; j < n_cols0; ++j)
      matrix_out(i, j) = factor * matrix0(i, j);

  return matrix_out;
}

/**
 * Returns the transpose of the input matrix @p matrix_in
 * MatrixType must contain at least the operator(n,m) to acces the
 * elements at row n and column m.
 */
template<typename MatrixTypeIn1, typename MatrixTypeOut = MatrixTypeIn1>
typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type
transpose(MatrixTypeIn1 && matrix_in)
{
  auto && matrix0 = std::forward<MatrixTypeIn1>(matrix_in);

  const unsigned int n_rows0 = matrix0.n_rows();
  const unsigned int n_cols0 = matrix0.n_cols();

  typename std::remove_const<typename std::remove_reference<MatrixTypeOut>::type>::type matrix_out;
  matrix_out.reinit(n_rows0, n_cols0);

  for(unsigned int i = 0; i < n_rows0; ++i)
    for(unsigned int j = 0; j < n_cols0; ++j)
      matrix_out(i, j) = matrix0(j, i);

  return matrix_out;
}

/**
 * Assembles the separable Kronecker product form of a collection of
 * "mass matrices" @p mass_matrix and "derivative matrices" @p
 * derivative_matrix. For a detailed description see the
 * documentation on deal.II's class TensorProductMatrixSymmetricSum.
 */
template<std::size_t dim, typename Number>
Table<2, Number>
assemble_separableKD(const std::array<Table<2, Number>, dim> & mass_matrix,
                     const std::array<Table<2, Number>, dim> & derivative_matrix)
{
  using namespace dealii;
  Table<2, Number> tpmatrix;

  if(dim == 1)
    tpmatrix = derivative_matrix[0];

  else if(dim == 2)
  {
    auto && MxA{kronecker_product(mass_matrix[1], derivative_matrix[0])};
    auto && AxM{kronecker_product(derivative_matrix[1], mass_matrix[0])};
    tpmatrix = std::move(sum(MxA, AxM));
  }

  else if(dim == 3)
  {
    auto && MxA         = kronecker_product(mass_matrix[1], derivative_matrix[0]);
    auto && AxM         = kronecker_product(derivative_matrix[1], mass_matrix[0]);
    auto && MxAxM_MxMxA = kronecker_product(mass_matrix[2], sum(AxM, MxA));
    auto && AxMxM =
      kronecker_product(derivative_matrix[2], kronecker_product(mass_matrix[1], mass_matrix[0]));
    tpmatrix = std::move(sum(AxMxM, MxAxM_MxMxA));
  }

  else
    Assert(false, ExcNotImplemented());

  return tpmatrix;
}

} // namespace Tensors

#endif // TENSORS_H_
