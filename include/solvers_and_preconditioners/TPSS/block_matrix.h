/*
 * block_matrix.h
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#ifndef BLOCK_MATRIX_H_
#define BLOCK_MATRIX_H_

#include "tensor_product_matrix.h"

using namespace dealii;

namespace Tensors
{
/**
 * Schur complement S = D - C * A^{-1} * B
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
class SchurComplement
{
public:
  using matrix_type = MatrixType;
  using value_type  = typename matrix_type::value_type;

  SchurComplement(const matrix_type & A_in,
                  const matrix_type & B_in,
                  const matrix_type & C_in,
                  const matrix_type & D_in)
    : A(A_in), B(B_in), C(C_in), D(D_in)
  {
    Sinv.reinit(as_table());
  }

  unsigned int
  m() const
  {
    AssertDimension(D.m(), C.m());
    return D.m();
  }

  unsigned int
  n() const
  {
    AssertDimension(D.n(), B.n());
    return D.n();
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    AssertDimension(D.n(), src_view.size());
    AssertDimension(D.m(), dst_view.size());
    AssertDimension(B.n(), src_view.size());
    AssertDimension(A.n(), A.m());
    AssertDimension(A.m(), B.m());
    AssertDimension(C.n(), A.m());
    AssertDimension(C.m(), dst_view.size());

    std::lock_guard<std::mutex> lock(this->mutex);
    tmp_array.clear();
    tmp_array.resize(B.m()); // TODO resize to max
    const auto dst_view_of_B = ArrayView(tmp_array.begin(), B.m());
    B.vmult(dst_view_of_B, src_view);
    Assert(A.m() <= dst_view.size(), ExcMessage("TODO dst_view not usable as temporary array."));
    const auto dst_view_of_Ainv = ArrayView(dst_view.begin(), A.m());
    A.apply_inverse(dst_view_of_Ainv, dst_view_of_B);
    tmp_array.clear();
    tmp_array.resize(C.n()); // TODO
    const auto dst_view_of_C = ArrayView(tmp_array.begin(), C.m());
    C.vmult(dst_view_of_C, dst_view_of_Ainv);

    D.vmult(dst_view, src_view);
    std::transform(dst_view.cbegin(),
                   dst_view.cend(),
                   dst_view_of_C.cbegin(),
                   dst_view.begin(),
                   std::minus<Number>{});
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    Sinv.vmult(dst_view, src_view);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
  }

private:
  const matrix_type &       A;
  const matrix_type &       B;
  const matrix_type &       C;
  const matrix_type &       D;
  VectorizedInverse<Number> Sinv;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};

/**
 * Computes the block Gaussian elimination of the 2x2 block matrix
 *
 *    | A B |
 *    | C D |
 *
 * using the Schur complement S for D, namely S = D - C * A^{-1} * B. The inverse
 * of the 2x2 block matrix is given by
 *
 *    | I   -A^{-1} * B | | A^{-1}   0      | | I             0 |
 *    | 0   I           | | 0        S^{-1} | | -C * A^{-1}   I |
 *
 * where the inverse of S is the dominating complexity.
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
class BlockGaussianInverse
{
public:
  using matrix_type = MatrixType;
  using value_type  = typename matrix_type::value_type;

  BlockGaussianInverse(const matrix_type & A_in,
                       const matrix_type & B_in,
                       const matrix_type & C_in,
                       const matrix_type & D_in)
    : A(A_in), B(B_in), C(C_in), D(D_in), S(A_in, B_in, C_in, D_in)
  {
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    VectorizedInverse<Number> inverse;
    AssertDimension(D.m(), D.n()); // -> S^{-1} possible
    AssertDimension(A.m(), A.n()); // -> A^{-1} possible
    AssertDimension(A.m(), B.m());
    AssertDimension(C.m(), D.m());
    AssertDimension(A.n(), C.n());
    AssertDimension(B.n(), D.n());
    const unsigned int m0 = A.m();
    const unsigned int m1 = D.m();
    AssertDimension(src_view.size(), m0 + m1);
    AssertDimension(dst_view.size(), m0 + m1);

    /// MEMORY INEFFICENT CODE

    /// src1[0,m0) <- [I 0] src_view = src_view[0,m0);
    const auto src1_view_m0 = ArrayView(src_view.begin(), m0);
    /// src1[m0,m0+m1) <- [-CA^{-1}  I] src_view = - CA^{-1} src_view[0,m0) + src_view[m0,m0+m1)
    AlignedVector<Number> tmp1_array(m0);
    const auto            dst_view_Ainv = ArrayView(tmp1_array.begin(), m0);
    const auto            src_view_Ainv = ArrayView(src_view.begin(), m0);
    A.apply_inverse(dst_view_Ainv, src_view_Ainv);
    AlignedVector<Number> tmp2_array(m1);
    const auto            dst_view_CAinv = ArrayView(tmp2_array.begin(), m1);
    C.vmult(dst_view_CAinv, dst_view_Ainv);
    AlignedVector<Number> tmp3_array(m1);
    const auto            src1_view_m1 = ArrayView(tmp3_array.begin(), m1);
    const auto            src_view_m1  = ArrayView(src_view.begin() + m0, m1);
    std::transform(src_view_m1.cbegin(),
                   src_view_m1.end(),
                   dst_view_CAinv.cbegin(),
                   src1_view_m1.begin(),
                   std::minus<Number>());

    /// src2[0,m0) <- [A^{-1}  0] src1 = A^{-1} src1_view_m0
    AlignedVector<Number> tmp4_array(m0);
    const auto            src2_view_m0 = ArrayView(tmp4_array.begin(), m0);
    A.apply_inverse(src2_view_m0, src1_view_m0);
    /// src2[m0,m0+m1) <- [0  S^{-1}] src1 = S^{-1} src_1_view_m1
    AlignedVector<Number> tmp5_array(m1);
    const auto            src2_view_m1 = ArrayView(tmp5_array.begin(), m1);
    S.apply_inverse(src2_view_m1, src1_view_m1);

    /// dst[0,m0) <- [I  -A^{-1}B] src2 = src2_view_m0 - A^{-1}B src2_view_m1
    AlignedVector<Number> tmp6_array(m0);
    const auto            dst_view_B = ArrayView(tmp6_array.begin(), m0);
    B.vmult(dst_view_B, src2_view_m1);
    AlignedVector<Number> tmp7_array(m0);
    const auto            dst_view_AinvB = ArrayView(tmp7_array.begin(), m0);
    A.apply_inverse(dst_view_AinvB, dst_view_B);
    std::transform(src2_view_m0.cbegin(),
                   src2_view_m0.cend(),
                   dst_view_AinvB.cbegin(),
                   dst_view.begin(),
                   std::minus<Number>());
    /// dst[m0,m0+m1) <- [0  I] src2 = src2_view_m1
    std::copy(src2_view_m1.cbegin(), src2_view_m1.cend(), dst_view.begin() + m0);
  }

private:
  const matrix_type &         A;
  const matrix_type &         B;
  const matrix_type &         C;
  const matrix_type &         D;
  SchurComplement<MatrixType> S;
};


template<int dim, typename Number, int n_rows_1d = -1>
class BlockMatrix
{
public:
  using matrix_type = TensorProductMatrix<dim, Number, n_rows_1d>;
  using value_type  = typename matrix_type::value_type;

  BlockMatrix() = default;

  BlockMatrix &
  operator=(const BlockMatrix & other)
  {
    resize(other.n_block_rows, other.n_block_cols);
    blocks = other.blocks;
    return *this;
  }

  /**
   * Deletes current block structure.
   */
  void
  clear()
  {
    std::fill(n_.begin(), n_.end(), 0U);
    blocks.clear();
    inverse_2x2.reset();
  }

  /**
   * Deletes old and resizes to square block structure.
   */
  void
  resize(const std::size_t n_rows)
  {
    resize(n_rows, n_rows);
  }

  /**
   * Deletes old and resizes to new block structure.
   */
  void
  resize(const std::size_t n_rows, const std::size_t n_cols)
  {
    clear();
    blocks.resize(n_rows * n_cols);
    n_[0] = n_rows;
    n_[1] = n_cols;
  }

  matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index)
  {
    AssertIndexRange(row_index, n_block_rows());
    AssertIndexRange(col_index, n_block_cols());
    /// possible change of block requires recomputation of inverse
    inverse_2x2.reset();
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

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
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

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    const bool is_2x2_block_matrix = n_block_rows() == 2 && n_block_cols() == 2;
    AssertThrow(is_2x2_block_matrix, ExcMessage("TODO"));
    if(!inverse_2x2)
      inverse_2x2 = std::make_shared<BlockGaussianInverse<matrix_type>>(get_block(0, 0),
                                                                        get_block(0, 1),
                                                                        get_block(1, 0),
                                                                        get_block(1, 1));
    inverse_2x2->vmult(dst, src);
    /// ALTERNATIVE: standard inverse based on LAPACK
    // basic_inverse = std::make_shared<const VectorizedInverse<Number>>(as_table());
    // basic_inverse->vmult(dst, src);
  }

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

  /**
   * The inverse of a 2 x 2 block matrix based on block Gaussian elimination.
   */
  mutable std::shared_ptr<const BlockGaussianInverse<matrix_type>> inverse_2x2;
  /// ALTERNATIVE: standard inverse based on LAPACK
  // mutable std::shared_ptr<const VectorizedInverse<Number>>         basic_inverse;
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

} // namespace Tensors

#endif // BLOCK_MATRIX_H_
