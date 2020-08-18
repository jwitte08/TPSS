
namespace Tensors
{
////////// BlockMatrixBase



template<typename MatrixType, typename Number>
BlockMatrixBase<MatrixType, Number> &
BlockMatrixBase<MatrixType, Number>::operator=(const BlockMatrixBase<MatrixType, Number> & other)
{
  resize(other.n_block_rows, other.n_block_cols);
  blocks = other.blocks;
  return *this;
}



template<typename MatrixType, typename Number>
void
BlockMatrixBase<MatrixType, Number>::clear()
{
  std::fill(n_blocks.begin(), n_blocks.end(), 0U);
  blocks.clear();
}



template<typename MatrixType, typename Number>
void
BlockMatrixBase<MatrixType, Number>::resize(const std::size_t n_rows)
{
  resize(n_rows, n_rows);
}



template<typename MatrixType, typename Number>
void
BlockMatrixBase<MatrixType, Number>::resize(const std::size_t n_rows, const std::size_t n_cols)
{
  clear();
  blocks.resize(n_rows * n_cols);
  n_blocks[0] = n_rows;
  n_blocks[1] = n_cols;
}



template<typename MatrixType, typename Number>
MatrixType &
BlockMatrixBase<MatrixType, Number>::get_block(const std::size_t row_index,
                                               const std::size_t col_index)
{
  AssertIndexRange(row_index, n_block_rows());
  AssertIndexRange(col_index, n_block_cols());
  return blocks[block_index(row_index, col_index)];
}



template<typename MatrixType, typename Number>
const MatrixType &
BlockMatrixBase<MatrixType, Number>::get_block(const std::size_t row_index,
                                               const std::size_t col_index) const
{
  AssertIndexRange(row_index, n_block_rows());
  AssertIndexRange(col_index, n_block_cols());
  return blocks[block_index(row_index, col_index)];
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::m(const std::size_t row_index) const
{
  AssertIndexRange(row_index, n_block_rows());
  return get_block(row_index, 0).m();
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::n(const std::size_t col_index) const
{
  AssertIndexRange(col_index, n_block_cols());
  return get_block(0, col_index).n();
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::m() const
{
  Assert(has_consistent_block_rows(), ExcMessage("Inconsistent block structure."));
  std::size_t n_rows = 0;
  for(std::size_t row_index = 0; row_index < n_block_rows(); ++row_index)
    n_rows += m(row_index);
  return n_rows;
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::n() const
{
  Assert(has_consistent_block_columns(), ExcMessage("Inconsistent block structure."));
  std::size_t n_cols = 0;
  for(std::size_t col_index = 0; col_index < n_block_cols(); ++col_index)
    n_cols += n(col_index);
  return n_cols;
}



template<typename MatrixType, typename Number>
Table<2, Number>
BlockMatrixBase<MatrixType, Number>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



// template<typename MatrixType, typename Number>
// Table<2, Number>
// BlockMatrixBase<MatrixType, Number>::as_transpose_table() const
// {
//   return Tensors::transpose_matrix_to_table(*this);
// }



template<typename MatrixType, typename Number>
void
BlockMatrixBase<MatrixType, Number>::vmult(const ArrayView<Number> &       dst,
                                           const ArrayView<const Number> & src) const
{
  blockwise_action([](const MatrixType &              block,
                      const ArrayView<Number> &       dst,
                      const ArrayView<const Number> & src) { block.vmult_add(dst, src); },
                   dst,
                   src);
}



template<typename MatrixType, typename Number>
std::array<std::size_t, 2>
BlockMatrixBase<MatrixType, Number>::size() const
{
  return n_blocks;
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::n_block_rows() const
{
  return n_blocks[0];
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::n_block_cols() const
{
  return n_blocks[1];
}



template<typename MatrixType, typename Number>
std::size_t
BlockMatrixBase<MatrixType, Number>::block_index(const std::size_t row_index,
                                                 const std::size_t col_index) const
{
  const std::array<std::size_t, 2> row_and_col = {row_index, col_index};
  return Tensors::multi_to_uniindex<2>(row_and_col, n_blocks);
}



/**
 * Applies the function @p action to each matrix of the block
 * diagonal and the associated vector slices of @p dst and @p src.
 * The signature of the action function should be equivalent to
 * action(MatrixType& m, ArrayView<...> dst, ArrayView<const ...> src)
 *
 * TODO This method has not been tested for non-quadratic blocks.
 */
template<typename MatrixType, typename Number>
template<typename ActionType>
void
BlockMatrixBase<MatrixType, Number>::blockwise_action(const ActionType &              action,
                                                      const ArrayView<Number> &       dst,
                                                      const ArrayView<const Number> & src) const
{
  AssertDimension(src.size(), n());
  AssertDimension(dst.size(), m());
  // Assert(src.size() == dst.size(),
  //        ExcMessage("TODO BlockMatrixBase is not quadratic.")); // quadratic !
  // Assert(n_rows_1d == -1 || src.size() % n_rows_1d == 0,
  //        ExcMessage("Input vector sizes are not a multiple of the static size."));
  Assert(!blocks.empty(), ExcMessage("Blocks are not initialized."));
  Assert(is_valid(), ExcMessage("Matrix sizes of blocks mismatch."));
  std::fill(dst.begin(), dst.end(), static_cast<Number>(0.));

  // *** apply sliced vectors to the associated block matrix
  std::size_t row_start = 0;
  for(std::size_t row = 0; row < n_block_rows(); ++row)
  {
    const ArrayView<Number> dst_block(dst.begin() + row_start, m(row));
    std::size_t             col_start = 0;
    for(std::size_t col = 0; col < n_block_cols(); ++col)
    {
      const ArrayView<const Number> src_block(src.begin() + col_start, n(col));
      action(get_block(row, col), dst_block, src_block);
      col_start += n(col);
    }
    Assert(col_start == src.size(), ExcMessage("Inconsistent slicing."));
    row_start += m(row);
  }
  Assert(row_start == dst.size(), ExcMessage("Inconsistent slicing."));
}



template<typename MatrixType, typename Number>
bool
BlockMatrixBase<MatrixType, Number>::has_consistent_block_rows() const
{
  for(std::size_t col = 0; col < n_block_cols(); ++col)
    for(std::size_t row = 0; row < n_block_rows(); ++row)
      if(get_block(row, col).m() != get_block(row, 0).m())
        return false;
  return true;
}



template<typename MatrixType, typename Number>
bool
BlockMatrixBase<MatrixType, Number>::has_consistent_block_columns() const
{
  for(std::size_t row = 0; row < n_block_rows(); ++row)
    for(std::size_t col = 0; col < n_block_cols(); ++col)
      if(get_block(row, col).n() != get_block(0, col).n())
        return false;
  return true;
}



template<typename MatrixType, typename Number>
bool
BlockMatrixBase<MatrixType, Number>::is_valid() const
{
  return has_consistent_block_rows() && has_consistent_block_columns();
}



////////// BlockMatrixBasic



template<typename MatrixType, typename Number>
BlockMatrixBasic<MatrixType, Number> &
BlockMatrixBasic<MatrixType, Number>::operator=(const BlockMatrixBasic<MatrixType, Number> & other)
{
  Base::operator=(other);
  basic_inverse = other.basic_inverse;
}



template<typename MatrixType, typename Number>
void
BlockMatrixBasic<MatrixType, Number>::clear()
{
  basic_inverse.reset();
  Base::clear();
}



template<typename MatrixType, typename Number>
MatrixType &
BlockMatrixBasic<MatrixType, Number>::get_block(const std::size_t row_index,
                                                const std::size_t col_index)
{
  /// modification of any blocks results in an invalid inverse
  basic_inverse.reset();

  return Base::get_block(row_index, col_index);
}



template<typename MatrixType, typename Number>
void
BlockMatrixBasic<MatrixType, Number>::invert(
  const typename InverseTable<Number>::AdditionalData & additional_data)
{
  basic_inverse = std::make_shared<InverseTable<Number>>(this->as_table(), additional_data);
}



template<typename MatrixType, typename Number>
Table<2, Number>
BlockMatrixBasic<MatrixType, Number>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



template<typename MatrixType, typename Number>
void
BlockMatrixBasic<MatrixType, Number>::apply_inverse(const ArrayView<Number> &       dst,
                                                    const ArrayView<const Number> & src) const
{
  AssertThrow(basic_inverse,
              ExcMessage("Have you called invert() after finishing the submission of entries?"));
  basic_inverse->vmult(dst, src);
}



////////// BlockMatrixBasic2x2



template<typename MatrixType, typename Number>
BlockMatrixBasic2x2<MatrixType, Number>::BlockMatrixBasic2x2(const MatrixType & A_in,
                                                             const MatrixType & B_in,
                                                             const MatrixType & C_in,
                                                             const MatrixType & D_in)
{
  Base::resize(2U, 2U);
  Base::get_block(0, 0) = A_in;
  Base::get_block(0, 1) = B_in;
  Base::get_block(1, 0) = C_in;
  Base::get_block(1, 1) = D_in;
}



template<typename MatrixType, typename Number>
BlockMatrixBasic2x2<MatrixType, Number> &
BlockMatrixBasic2x2<MatrixType, Number>::operator=(
  const BlockMatrixBasic2x2<MatrixType, Number> & other)
{
  Base::operator=(other);
  basic_inverse = other.basic_inverse;
}



template<typename MatrixType, typename Number>
void
BlockMatrixBasic2x2<MatrixType, Number>::clear()
{
  basic_inverse.reset();
  Base::clear();
}



// template<typename MatrixType, typename Number>
// MatrixType &
// BlockMatrixBasic2x2<MatrixType, Number>::get_block(const std::size_t row_index,
//                                                 const std::size_t col_index)
// {
//   /// modification of any blocks results in an invalid inverse
//   basic_inverse.reset();

//   return Base::get_block(row_index, col_index);
// }



template<typename MatrixType, typename Number>
void
BlockMatrixBasic2x2<MatrixType, Number>::invert()
{
  const auto & A = Base::get_block(0, 0);
  const auto & B = Base::get_block(0, 1);
  const auto & C = Base::get_block(1, 0);
  const auto & D = Base::get_block(1, 1);
  basic_inverse =
    std::make_shared<BlockGaussianInverse<MatrixType, SchurComplement<MatrixType>, Number>>(A,
                                                                                            B,
                                                                                            C,
                                                                                            D);
}



template<typename MatrixType, typename Number>
Table<2, Number>
BlockMatrixBasic2x2<MatrixType, Number>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



template<typename MatrixType, typename Number>
void
BlockMatrixBasic2x2<MatrixType, Number>::apply_inverse(const ArrayView<Number> &       dst,
                                                       const ArrayView<const Number> & src) const
{
  AssertThrow(basic_inverse,
              ExcMessage("Have you called invert() after finishing the submission of entries?"));
  basic_inverse->vmult(dst, src);
}



} // namespace Tensors
