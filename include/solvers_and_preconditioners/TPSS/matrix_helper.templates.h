////////// InverseTable



template<typename Number>
InverseTable<Number>::InverseTable(const Table<2, Number> & matrix_in,
                                   const AdditionalData &   additional_data_in)
{
  reinit(matrix_in, additional_data_in);
}



template<typename Number>
void
InverseTable<Number>::reinit(const Table<2, Number> & matrix_in,
                             const AdditionalData &   additional_data_in)
{
  Assert(matrix_in.size(0) == matrix_in.size(1), ExcMessage("Matrix is not square."));
  clear();

  auto inverses = std::make_shared<std::array<LAPACKFullMatrix<scalar_value_type>, macro_size>>();
  const unsigned int n_rows = matrix_in.size(0);
  for(auto lane = 0U; lane < macro_size; ++lane)
  {
    auto & inverse = (*inverses)[lane];
    inverse.reinit(n_rows);
    inverse = table_to_fullmatrix(matrix_in, lane);
    if(additional_data_in.kernel_size != numbers::invalid_unsigned_int)
      inverse.compute_inverse_svd_with_kernel(additional_data_in.kernel_size);
    else
      inverse.compute_inverse_svd(additional_data_in.threshold);
  }
  /// ALTERNATIVE: FullMatrix
  // auto inverses = std::make_shared<std::array<FullMatrix<scalar_value_type>, macro_size>>();
  // const unsigned int n_rows = matrix_in.size(0);
  // for(auto lane = 0U; lane < macro_size; ++lane)
  // {
  //   auto & inverse = (*inverses)[lane];
  //   inverse.reinit(n_rows, n_rows);
  //   inverse.invert(table_to_fullmatrix(matrix_in, lane));
  // }
  this->inverses = inverses;
}



template<typename Number>
void
InverseTable<Number>::clear()
{
  inverses.reset();
}



template<typename Number>
void
InverseTable<Number>::vmult(const ArrayView<Number> &       dst_view,
                            const ArrayView<const Number> & src_view) const
{
  vmult_impl(dst_view, src_view);
}



template<typename Number>
void
InverseTable<Number>::vmult_impl(const ArrayView<scalar_value_type> &       dst_view,
                                 const ArrayView<const scalar_value_type> & src_view,
                                 const unsigned int                         lane) const
{
  Vector<scalar_value_type> dst(dst_view.size()), src(src_view.cbegin(), src_view.cend());
  const auto &              inverse = (*inverses)[lane];
  inverse.vmult(dst, src);
  std::copy(dst.begin(), dst.end(), dst_view.begin());
}



template<typename Number>
void
InverseTable<Number>::vmult_impl(
  const ArrayView<VectorizedArray<scalar_value_type>> &       dst_view,
  const ArrayView<const VectorizedArray<scalar_value_type>> & src_view) const
{
  Vector<scalar_value_type> dst_lane(dst_view.size()), src_lane(src_view.size());
  for(auto lane = 0U; lane < macro_size; ++lane)
  {
    std::transform(src_view.cbegin(),
                   src_view.cend(),
                   src_lane.begin(),
                   [lane](const auto & value) { return value[lane]; });
    const auto src_view_lane = make_array_view(src_lane);
    const auto dst_view_lane = make_array_view(dst_lane);
    vmult_impl(dst_view_lane, src_view_lane, lane);
    for(auto i = 0U; i < dst_lane.size(); ++i)
      dst_view[i][lane] = dst_lane[i];
  }
}



////////// MatrixAsTable



template<typename Number>
MatrixAsTable<Number>::MatrixAsTable(const MatrixAsTable & other)
  : matrix(other.matrix), inverse_matrix(other.inverse_matrix)
{
}



template<typename Number>
MatrixAsTable<Number>::MatrixAsTable(MatrixAsTable && other)
  : matrix(std::move(other.matrix)), inverse_matrix(other.inverse_matrix)
{
}



template<typename Number>
MatrixAsTable<Number> &
MatrixAsTable<Number>::operator=(const MatrixAsTable & other)
{
  if(this == &other)
    return *this;
  matrix         = other.matrix;
  inverse_matrix = other.inverse_matrix;
  return *this;
}



template<typename Number>
MatrixAsTable<Number> &
MatrixAsTable<Number>::operator=(MatrixAsTable && other)
{
  if(this == &other)
    return *this;
  matrix         = std::move(other.matrix);
  inverse_matrix = other.inverse_matrix;
  return *this;
}



template<typename Number>
MatrixAsTable<Number> &
MatrixAsTable<Number>::operator=(const Table<2, Number> & other)
{
  if(&(this->matrix) == &other)
    return *this;
  matrix = other;
  inverse_matrix.reset();
  return *this;
}



template<typename Number>
MatrixAsTable<Number> &
MatrixAsTable<Number>::operator=(Table<2, Number> && other)
{
  if(&(this->matrix) == &other)
    return *this;
  matrix = std::move(other);
  inverse_matrix.reset();
  return *this;
}



template<typename Number>
MatrixAsTable<Number>::operator const Table<2, Number> &() const
{
  return matrix;
}



template<typename Number>
MatrixAsTable<Number>::operator Table<2, Number> &()
{
  return matrix;
}



template<typename Number>
unsigned int
MatrixAsTable<Number>::m() const
{
  return this->matrix.size(0);
}



template<typename Number>
unsigned int
MatrixAsTable<Number>::n() const
{
  return this->matrix.size(1);
}



template<typename Number>
template<bool add, bool transpose>
void
MatrixAsTable<Number>::vmult_impl(const ArrayView<Number> &       dst_view,
                                  const ArrayView<const Number> & src_view) const
{
  const unsigned int m = transpose ? this->n() : this->m();
  const unsigned int n = transpose ? this->m() : this->n();
  AssertDimension(dst_view.size(), m);
  AssertDimension(src_view.size(), n);

  for(auto i = 0U; i < m; ++i)
  {
    Number value(0.);
    for(auto j = 0U; j < n; ++j)
      value += matrix(transpose ? j : i, transpose ? i : j) * src_view[j];
    if(add)
      dst_view[i] += value;
    else
      dst_view[i] = value;
  }
}



template<typename Number>
template<bool add, bool transpose>
void
MatrixAsTable<Number>::vmult_impl(const ArrayView<Number> &       dst_view,
                                  const Number &                  factor,
                                  const ArrayView<const Number> & src_view) const
{
  const unsigned int m = transpose ? this->n() : this->m();
  const unsigned int n = transpose ? this->m() : this->n();
  AssertDimension(dst_view.size(), m);
  AssertDimension(src_view.size(), n);

  for(auto i = 0U; i < m; ++i)
  {
    Number value(0.);
    for(auto j = 0U; j < n; ++j)
      value += matrix(transpose ? j : i, transpose ? i : j) * src_view[j];
    if(add)
      dst_view[i] += factor * value;
    else
      dst_view[i] = factor * value;
  }
}



template<typename Number>
void
MatrixAsTable<Number>::vmult(const ArrayView<Number> &       dst_view,
                             const ArrayView<const Number> & src_view) const
{
  vmult_impl<false, /*transpose*/ false>(dst_view, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::vmult_add(const ArrayView<Number> &       dst_view,
                                 const ArrayView<const Number> & src_view) const
{
  vmult_impl<true, /*transpose*/ false>(dst_view, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::vmult_add(AlignedVector<Number> &       dst,
                                 const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  vmult_add(dst_view, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::vmult_sadd(const ArrayView<Number> &       dst_view,
                                  const Number &                  scalar,
                                  const ArrayView<const Number> & src_view) const
{
  vmult_impl<true, /*transpose*/ false>(dst_view, scalar, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::Tvmult(const ArrayView<Number> &       dst_view,
                              const ArrayView<const Number> & src_view) const
{
  vmult_impl<false, /*transpose*/ true>(dst_view, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::Tvmult_add(const ArrayView<Number> &       dst_view,
                                  const ArrayView<const Number> & src_view) const
{
  vmult_impl<true, /*transpose*/ true>(dst_view, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::Tvmult_add(AlignedVector<Number> &       dst,
                                  const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  Tvmult_add(dst_view, src_view);
}



template<typename Number>
void
MatrixAsTable<Number>::shuffle(const std::array<std::size_t, 2> & size_of_blockmatrix,
                               const std::array<std::size_t, 2> & size_of_blocks)
{
  const auto & [m_of_blocks, n_of_blocks] = size_of_blocks;
  const auto & [m, n]                     = size_of_blockmatrix;

  AssertDimension(m * m_of_blocks, this->m());
  AssertDimension(n * n_of_blocks, this->n());

  Table<2, Number> shuffled_matrix(m * n, m_of_blocks * n_of_blocks);
  for(auto i = 0U; i < m; ++i)
    for(auto j = 0U; j < n; ++j)
    {
      const auto fixed_row = i + j * m;

      LinAlg::vect_of_block_impl<Number, /*transpose*/ true>(
        shuffled_matrix,
        fixed_row,
        this->matrix,
        /*range of rows*/ {i * m_of_blocks, (i + 1) * m_of_blocks},
        /*range of columns*/ {j * n_of_blocks, (j + 1) * n_of_blocks});
    }

  this->matrix = std::move(shuffled_matrix);
}



template<typename Number>
void
MatrixAsTable<Number>::invert(const typename InverseTable<Number>::AdditionalData & additional_data)
{
  inverse_matrix = std::make_shared<InverseTable<Number>>(matrix, additional_data);
}



template<typename Number>
void
MatrixAsTable<Number>::apply_inverse(const ArrayView<Number> &       dst_view,
                                     const ArrayView<const Number> & src_view) const
{
  AssertThrow(inverse_matrix, ExcMessage("Have you called invert() after modifying entries?"));
  inverse_matrix->vmult(dst_view, src_view);
}



template<typename Number>
Table<2, Number> &
MatrixAsTable<Number>::as_table()
{
  inverse_matrix.reset();
  return matrix;
}



template<typename Number>
const Table<2, Number> &
MatrixAsTable<Number>::as_table() const
{
  return matrix;
}



template<typename Number>
Table<2, Number>
MatrixAsTable<Number>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



template<typename Number>
template<bool fill_transpose>
void
MatrixAsTable<Number>::fill_submatrix(const Table<2, Number> & other,
                                      const unsigned int       row_start,
                                      const unsigned int       column_start)
{
  AssertIndexRange(row_start + (fill_transpose ? other.size(1) : other.size(0)) - 1, m());
  AssertIndexRange(column_start + (fill_transpose ? other.size(0) : other.size(1)) - 1, n());
  if constexpr(fill_transpose == false)
  {
    for(auto i = 0U; i < other.size(0); ++i)
      for(auto j = 0U; j < other.size(1); ++j)
        matrix(row_start + i, column_start + j) = other(i, j);
  }
  else
  {
    for(auto j = 0U; j < other.size(1); ++j)
      for(auto i = 0U; i < other.size(0); ++i)
        matrix(row_start + j, column_start + i) = other(i, j);
  }
  inverse_matrix.reset();
}



template<typename Number>
template<typename OtherNumber, bool fill_transpose>
void
MatrixAsTable<Number>::fill_submatrix(const Table<2, OtherNumber> & other,
                                      const unsigned int            row_start,
                                      const unsigned int            column_start,
                                      const unsigned int            lane)
{
  fill_or_add_submatrix_impl<OtherNumber, fill_transpose, false>(other,
                                                                 row_start,
                                                                 column_start,
                                                                 lane);
}



template<typename Number>
template<typename OtherNumber, bool fill_transpose>
void
MatrixAsTable<Number>::add_submatrix(const Table<2, OtherNumber> & other,
                                     const unsigned int            row_start,
                                     const unsigned int            column_start,
                                     const unsigned int            lane)
{
  fill_or_add_submatrix_impl<OtherNumber, fill_transpose, true>(other,
                                                                row_start,
                                                                column_start,
                                                                lane);
}



template<typename Number>
template<typename OtherNumber, bool fill_transpose, bool add>
void
MatrixAsTable<Number>::fill_or_add_submatrix_impl(const Table<2, OtherNumber> & other,
                                                  const unsigned int            row_start,
                                                  const unsigned int            column_start,
                                                  const unsigned int            lane)
{
  AssertIndexRange(row_start + (fill_transpose ? other.size(1) : other.size(0)) - 1, m());
  AssertIndexRange(column_start + (fill_transpose ? other.size(0) : other.size(1)) - 1, n());
  if constexpr(fill_transpose == false)
  {
    for(auto i = 0U; i < other.size(0); ++i)
      for(auto j = 0U; j < other.size(1); ++j)
        if constexpr(add)
          scalar_value(matrix(row_start + i, column_start + j), lane) +=
            scalar_value(other(i, j), lane);
        else
          scalar_value(matrix(row_start + i, column_start + j), lane) =
            scalar_value(other(i, j), lane);
  }
  else
  {
    for(auto j = 0U; j < other.size(1); ++j)
      for(auto i = 0U; i < other.size(0); ++i)
        if constexpr(add)
          scalar_value(matrix(row_start + j, column_start + i), lane) +=
            scalar_value(other(i, j), lane);
        else
          scalar_value(matrix(row_start + j, column_start + i), lane) =
            scalar_value(other(i, j), lane);
  }
  inverse_matrix.reset();
}



////////// DiagonalMatrixWrap



template<typename Number>
DiagonalMatrixWrap<Number>::DiagonalMatrixWrap()
{
  reinit(AlignedVector<Number>());
}



template<typename Number>
DiagonalMatrixWrap<Number>::DiagonalMatrixWrap(const AlignedVector<Number> &    diagonal_in,
                                               const std::array<std::size_t, 2> size_in)
{
  reinit(diagonal_in, size_in);
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::reinit(const AlignedVector<Number> &    diagonal_in,
                                   const std::array<std::size_t, 2> size_in)
{
  const auto [m_in, n_in] = size_in;
  const bool treat_as_square =
    m_in == static_cast<std::size_t>(-1) && n_in == static_cast<std::size_t>(-1);
  size =
    treat_as_square ? std::array<std::size_t, 2>{diagonal_in.size(), diagonal_in.size()} : size_in;

  const auto [m, n] = size;
  (void)m, (void)n;
  Assert(diagonal_in.size() == m || diagonal_in.size() == n, ExcMessage("Mismatching size."));
  AssertIndexRange(diagonal_in.size(), m + 1);
  AssertIndexRange(diagonal_in.size(), n + 1);
  diagonal = diagonal_in;

  size_of_blockmatrix = {0U, 0U};
  size_of_blocks      = {0U, 0U};
  is_shuffled         = false;
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::clear()
{
  size = {0U, 0U};
  diagonal.clear();
  size_of_blockmatrix = {0U, 0U};
  size_of_blocks      = {0U, 0U};
  is_shuffled         = false;
}



template<typename Number>
unsigned int
DiagonalMatrixWrap<Number>::m() const
{
  return this->size[0];
}



template<typename Number>
unsigned int
DiagonalMatrixWrap<Number>::n() const
{
  return this->size[1];
}



template<typename Number>
template<bool add, bool transpose>
void
DiagonalMatrixWrap<Number>::vmult_impl(const ArrayView<Number> &       dst_view,
                                       const ArrayView<const Number> & src_view) const
{
  AssertDimension(dst_view.size(), transpose ? this->n() : this->m());
  AssertDimension(src_view.size(), transpose ? this->m() : this->n());

  /// clear dst if not adding into it
  if(!add)
    std::fill(dst_view.begin(), dst_view.end(), static_cast<Number>(0.));

  if(is_shuffled)
  {
    const auto & [m, n]   = size_of_blocks;
    const auto & [mm, nn] = size_of_blockmatrix;

    /// assuming squared blocks TODO rectangular blocks...
    AssertDimension(m, n);
    /// assuming squared blockmatrix TODO rectangular...
    AssertDimension(mm, nn);

    for(auto ii = 0U; ii < mm; ++ii)
      for(auto i = 0U; i < m; ++i)
        dst_view[transpose ? i + i * m : ii + ii * mm] +=
          diagonal[i + ii * m] * src_view[transpose ? ii + ii * mm : i + i * m];
  }

  else
  {
    /// if the matrix isn't squared the diagonal has either less elements than
    /// rows or than columns, thus, the loop is bound by diagonal.size()
    for(auto i = 0U; i < diagonal.size(); ++i)
    {
      if(add)
        dst_view[i] += diagonal[i] * src_view[i];
      else
        dst_view[i] = diagonal[i] * src_view[i];
    }
  }
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::vmult(const ArrayView<Number> &       dst_view,
                                  const ArrayView<const Number> & src_view) const
{
  vmult_impl<false, /*transpose*/ false>(dst_view, src_view);
}


template<typename Number>
void
DiagonalMatrixWrap<Number>::vmult_add(const ArrayView<Number> &       dst_view,
                                      const ArrayView<const Number> & src_view) const
{
  vmult_impl<true, /*transpose*/ false>(dst_view, src_view);
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::vmult_add(AlignedVector<Number> &       dst,
                                      const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  vmult_add(dst_view, src_view);
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::Tvmult(const ArrayView<Number> &       dst_view,
                                   const ArrayView<const Number> & src_view) const
{
  vmult_impl<false, /*transpose*/ true>(dst_view, src_view);
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::Tvmult_add(const ArrayView<Number> &       dst_view,
                                       const ArrayView<const Number> & src_view) const
{
  vmult_impl<true, /*transpose*/ true>(dst_view, src_view);
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::Tvmult_add(AlignedVector<Number> &       dst,
                                       const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  Tvmult_add(dst_view, src_view);
}



template<typename Number>
void
DiagonalMatrixWrap<Number>::shuffle(const std::array<std::size_t, 2> & size_of_blockmatrix_in,
                                    const std::array<std::size_t, 2> & size_of_blocks_in)
{
  const auto & [m_of_blocks, n_of_blocks] = size_of_blocks_in;
  const auto & [m, n]                     = size_of_blockmatrix_in;

  AssertDimension(m * m_of_blocks, this->m());
  AssertDimension(n * n_of_blocks, this->n());

  size                = {m * n, m_of_blocks * n_of_blocks};
  size_of_blockmatrix = size_of_blockmatrix_in;
  size_of_blocks      = size_of_blocks_in;
  is_shuffled         = true;
}



template<typename Number>
const AlignedVector<Number> &
DiagonalMatrixWrap<Number>::get_diagonal() const
{
  return diagonal;
}



template<typename Number>
Table<2, Number>
DiagonalMatrixWrap<Number>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



////////// PatchMatrix



template<typename PatchOperator>
void
PatchMatrix<PatchOperator>::reinit(const PatchOperator * patch_operator_in,
                                   const unsigned int    patch_id)
{
  patch_operator   = patch_operator_in;
  current_patch_id = patch_id;
}



template<typename PatchOperator>
unsigned int
PatchMatrix<PatchOperator>::m() const
{
  return patch_operator->m(current_patch_id);
}



template<typename PatchOperator>
unsigned int
PatchMatrix<PatchOperator>::n() const
{
  return patch_operator->n(current_patch_id);
}



template<typename PatchOperator>
void
PatchMatrix<PatchOperator>::vmult(
  const ArrayView<typename PatchMatrix<PatchOperator>::value_type> &       dst_view,
  const ArrayView<const typename PatchMatrix<PatchOperator>::value_type> & src_view) const
{
  patch_operator->vmult(dst_view, src_view, current_patch_id);
}



template<typename PatchOperator>
void
PatchMatrix<PatchOperator>::apply_inverse(
  const ArrayView<typename PatchMatrix<PatchOperator>::value_type> &       dst_view,
  const ArrayView<const typename PatchMatrix<PatchOperator>::value_type> & src_view) const
{
  AssertThrow(false, ExcMessage("todo"));
  // TODO
  // patch_operator.apply_inverse(dst_view, src_view, current_patch_id);
}



template<typename PatchOperator>
Table<2, typename PatchMatrix<PatchOperator>::value_type>
PatchMatrix<PatchOperator>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



template<typename PatchOperator>
FullMatrix<typename PatchMatrix<PatchOperator>::scalar_value_type>
PatchMatrix<PatchOperator>::as_fullmatrix(const unsigned int lane)
{
  return table_to_fullmatrix(as_table(), lane);
}



////////// ConstrainedMatrix



template<typename MatrixType, typename Number>
unsigned int
ConstrainedMatrix<MatrixType, Number>::m() const
{
  return matrix.m();
}



template<typename MatrixType, typename Number>
unsigned int
ConstrainedMatrix<MatrixType, Number>::n() const
{
  return matrix.n();
}



template<typename MatrixType, typename Number>
void
ConstrainedMatrix<MatrixType, Number>::vmult(const ArrayView<Number> &       dst_view,
                                             const ArrayView<const Number> & src_view) const
{
  tmp_array.resize_fast(src_view.size());
  const ArrayView<Number> constrained_src_view(tmp_array.begin(), src_view.size());
  std::copy(src_view.begin(), src_view.end(), constrained_src_view.begin());
  zero_out_constrained_dofs_col(constrained_src_view);

  matrix.vmult(dst_view, constrained_src_view);

  zero_out_constrained_dofs_row(dst_view);
}



template<typename MatrixType, typename Number>
void
ConstrainedMatrix<MatrixType, Number>::vmult_add(const ArrayView<Number> &       dst_view,
                                                 const ArrayView<const Number> & src_view) const
{
  matrix.vmult_add(dst_view, src_view);
}



template<typename MatrixType, typename Number>
void
ConstrainedMatrix<MatrixType, Number>::apply_inverse(const ArrayView<Number> &       dst_view,
                                                     const ArrayView<const Number> & src_view) const
{
  AssertDimension(dst_view.size(), src_view.size());
  Assert(constrained_dof_indices_row == constrained_dof_indices_col,
         ExcMessage("Row and column constraints don't coincide."));

  tmp_array.resize_fast(src_view.size());
  const ArrayView<Number> constrained_src_view(tmp_array.begin(), src_view.size());
  std::copy(src_view.begin(), src_view.end(), constrained_src_view.begin());
  zero_out_constrained_dofs_row(constrained_src_view);

  matrix.apply_inverse(dst_view, constrained_src_view);

  zero_out_constrained_dofs_col(dst_view);
}



template<typename MatrixType, typename Number>
Table<2, Number>
ConstrainedMatrix<MatrixType, Number>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



template<typename MatrixType, typename Number>
Table<2, Number>
ConstrainedMatrix<MatrixType, Number>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



template<typename MatrixType, typename Number>
void
ConstrainedMatrix<MatrixType, Number>::zero_out_constrained_dofs_row(
  const ArrayView<Number> & dst_view) const
{
  zero_out_constrained_dofs_impl<true>(dst_view);
}



template<typename MatrixType, typename Number>
void
ConstrainedMatrix<MatrixType, Number>::zero_out_constrained_dofs_col(
  const ArrayView<Number> & dst_view) const
{
  zero_out_constrained_dofs_impl<false>(dst_view);
}



template<typename MatrixType, typename Number>
template<bool is_row>
void
ConstrainedMatrix<MatrixType, Number>::zero_out_constrained_dofs_impl(
  const ArrayView<Number> & dst_view) const
{
  const auto & constrained_dof_indices =
    is_row ? constrained_dof_indices_row : constrained_dof_indices_col;
  for(auto lane = 0U; lane < macro_size; ++lane)
    for(const auto dof_index : constrained_dof_indices[lane])
      dst_view[dof_index][lane] = 0.;
}
