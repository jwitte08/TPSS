
namespace Tensors
{
////////// SchurComplementBase



template<typename MatrixType, typename Number>
SchurComplementBase<MatrixType, Number>::SchurComplementBase(const matrix_type & A_in,
                                                             const matrix_type & B_in,
                                                             const matrix_type & C_in,
                                                             const matrix_type & D_in)
  : A(&A_in), B(&B_in), C(&C_in), D(&D_in)
{
  AssertDimension(A->m(), A->n());
  AssertDimension(A->m(), B->m());
  AssertDimension(A->n(), C->n());
  AssertDimension(D->n(), B->n());
  AssertDimension(D->m(), C->m());
}

template<typename MatrixType, typename Number>
unsigned int
SchurComplementBase<MatrixType, Number>::m() const
{
  AssertDimension(D->m(), C->m());
  return D->m();
}

template<typename MatrixType, typename Number>
unsigned int
SchurComplementBase<MatrixType, Number>::n() const
{
  AssertDimension(D->n(), B->n());
  return D->n();
}

template<typename MatrixType, typename Number>
void
SchurComplementBase<MatrixType, Number>::vmult(const ArrayView<Number> &       dst_view,
                                               const ArrayView<const Number> & src_view) const
{
  AssertDimension(D->n(), src_view.size());
  AssertDimension(D->m(), dst_view.size());
  AssertDimension(B->n(), src_view.size());
  AssertDimension(A->n(), A->m());
  AssertDimension(A->m(), B->m());
  AssertDimension(C->n(), A->m());
  AssertDimension(C->m(), dst_view.size());
  const auto n_rows_max = std::max(B->m(), C->m());

  std::lock_guard<std::mutex> lock(this->mutex);

  /// compute vmult for C A^{-1} B
  tmp_array.resize(n_rows_max + A->n());
  const auto dst_view_of_B = ArrayView(tmp_array.begin(), B->m());
  B->vmult(dst_view_of_B, src_view);
  const auto dst_view_of_AinvB = ArrayView(tmp_array.begin() + n_rows_max, A->n());
  A->apply_inverse(dst_view_of_AinvB, dst_view_of_B);
  const auto dst_view_of_CAinvB = ArrayView(tmp_array.begin(), C->m());
  C->vmult(dst_view_of_CAinvB, dst_view_of_AinvB);

  /// compute vmult for D - C A^{-1} B
  D->vmult(dst_view, src_view);
  std::transform(dst_view.cbegin(),
                 dst_view.cend(),
                 dst_view_of_CAinvB.cbegin(),
                 dst_view.begin(),
                 std::minus<Number>{});
}

template<typename MatrixType, typename Number>
Table<2, Number>
SchurComplementBase<MatrixType, Number>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



////////// SchurComplementBasic



template<typename MatrixType, typename Number>
SchurComplementBasic<MatrixType, Number>::SchurComplementBasic(const MatrixType & A_in,
                                                               const MatrixType & B_in,
                                                               const MatrixType & C_in,
                                                               const MatrixType & D_in)
  : SchurComplementBase<MatrixType, Number>(A_in, B_in, C_in, D_in)
{
}



template<typename MatrixType, typename Number>
void
SchurComplementBasic<MatrixType, Number>::invert(const AdditionalData & additional_data)
{
  if(additional_data.kronecker_rank != numbers::invalid_unsigned_int)
  {
    const auto [m_0, n_0] = additional_data.size_of_kronecker_factors[0];
    const auto [m_1, n_1] = additional_data.size_of_kronecker_factors[1];
    AssertDimension(m_0 * m_1, Base::m());
    AssertDimension(n_0 * n_1, Base::n());

    using matrix_type_1d = Table<2, Number>;
    using tensor_type    = std::array<matrix_type_1d, 2>;

    std::vector<tensor_type> low_rank_representation(additional_data.kronecker_rank);
    for(auto & tensor : low_rank_representation)
    {
      tensor[0].reinit(m_0, n_0);
      tensor[1].reinit(m_1, n_1);
    }

    compute_ksvd(Base::as_table(), low_rank_representation, additional_data.lanczos_iterations);

    TensorProductMatrix<2, Number> low_rank_matrix(low_rank_representation);
    Sinv = std::make_shared<InverseTable<Number>>(low_rank_matrix.as_table(),
                                                  additional_data.inverse_table);

    // // !!!
    // AssertThrow(false, ExcMessage("Todo..."));
  }
  else
  {
    Sinv = std::make_shared<InverseTable<Number>>(Base::as_table(), additional_data.inverse_table);
  }
}



template<typename MatrixType, typename Number>
void
SchurComplementBasic<MatrixType, Number>::apply_inverse(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  AssertThrow(Sinv, ExcMessage("Have you called invert()?"));
  Sinv->vmult(dst_view, src_view);
}



template<typename MatrixType, typename Number>
Table<2, Number>
SchurComplementBasic<MatrixType, Number>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



////////// BlockGaussianInverseBase



template<typename MatrixType, typename SchurType, typename Number>
BlockGaussianInverseBase<MatrixType, SchurType, Number>::BlockGaussianInverseBase(
  const matrix_type & A_in,
  const matrix_type & B_in,
  const matrix_type & C_in,
  const schur_type &  S_in)
  : A(A_in), B(B_in), C(C_in), S(S_in)
{
  AssertDimension(A.m(), A.n());
  AssertDimension(S.m(), S.n());
  AssertDimension(A.m(), B.m());
  AssertDimension(A.n(), C.n());
  AssertDimension(S.n(), B.n());
  AssertDimension(S.m(), C.m());
}



template<typename MatrixType, typename SchurType, typename Number>
unsigned int
BlockGaussianInverseBase<MatrixType, SchurType, Number>::m() const
{
  return A.n() + S.n();
}



template<typename MatrixType, typename SchurType, typename Number>
unsigned int
BlockGaussianInverseBase<MatrixType, SchurType, Number>::n() const
{
  return A.m() + S.m();
}



template<typename MatrixType, typename SchurType, typename Number>
void
BlockGaussianInverseBase<MatrixType, SchurType, Number>::vmult(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  const unsigned int m0 = A.m();
  const unsigned int m1 = S.m();
  AssertDimension(src_view.size(), m0 + m1);
  AssertDimension(dst_view.size(), m0 + m1);

  /// MEMORY INEFFICENT CODE (TODO)

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



template<typename MatrixType, typename SchurType, typename Number>
Table<2, Number>
BlockGaussianInverseBase<MatrixType, SchurType, Number>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



////////// BlockGaussianInverseBasic



template<typename MatrixType, typename Number>
BlockGaussianInverseBasic<MatrixType, Number>::BlockGaussianInverseBasic(
  const MatrixType &   A_in,
  const MatrixType &   B_in,
  const MatrixType &   C_in,
  const MatrixType &   D_in,
  const AdditionalData additional_data)
  : Base(A_in, B_in, C_in, [&]() {
      SchurComplementBasic<MatrixType, Number> S(A_in, B_in, C_in, D_in);
      S.invert(additional_data);
      return S;
    }())
{
}



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



template<typename MatrixType, typename Number>
void
BlockMatrixBasic2x2<MatrixType, Number>::invert(const AdditionalData & additional_data)
{
  const auto & A = Base::get_block(0, 0);
  const auto & B = Base::get_block(0, 1);
  const auto & C = Base::get_block(1, 0);
  const auto & D = Base::get_block(1, 1);
  basic_inverse =
    std::make_shared<BlockGaussianInverseBasic<MatrixType, Number>>(A, B, C, D, additional_data);
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
