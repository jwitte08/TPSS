
namespace Tensors
{
template<int order, typename Number, int n_rows_1d>
TensorProductMatrixBase<order, Number, n_rows_1d>::TensorProductMatrixBase()
  : tensor_helper_row(std::make_shared<const TensorHelper<order>>(0)),
    tensor_helper_column(std::make_shared<const TensorHelper<order>>(0))
{
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrixBase<order, Number, n_rows_1d> &
TensorProductMatrixBase<order, Number, n_rows_1d>::operator=(const TensorProductMatrixBase & other)
{
  reinit(other.elementary_tensors);
  return *this;
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrixBase<order, Number, n_rows_1d>::reinit(
  const std::vector<typename TensorProductMatrixBase<order, Number, n_rows_1d>::tensor_type> &
    elementary_tensors_in)
{
  clear();

  /// TODO is it better throw to an exception ?!
  if(elementary_tensors_in.empty())
  {
    tensor_helper_row    = std::make_shared<const TensorHelper<order>>(0);
    tensor_helper_column = std::make_shared<const TensorHelper<order>>(0);
    return;
  }

  std::array<unsigned int, order> n_rows_foreach_dimension;
  std::array<unsigned int, order> n_columns_foreach_dimension;
  for(auto d = 0; d < order; ++d)
  {
    const auto & matrix            = elementary_tensors_in.front()[d];
    n_rows_foreach_dimension[d]    = matrix.size(0);
    n_columns_foreach_dimension[d] = matrix.size(1);
  }
  tensor_helper_row    = std::make_shared<const TensorHelper<order>>(n_rows_foreach_dimension);
  tensor_helper_column = std::make_shared<const TensorHelper<order>>(n_columns_foreach_dimension);

  Assert(check_n_rows_1d_static(elementary_tensors_in),
         ExcMessage("Not all univariate matrices are of size (n_rows_1d x n_rows_1d)."));
  Assert(check_n_rows_and_columns_1d(elementary_tensors_in),
         ExcMessage("Mismatching sizes of univariate matrices."));

  this->elementary_tensors.clear();
  std::copy(elementary_tensors_in.cbegin(),
            elementary_tensors_in.cend(),
            std::back_inserter(this->elementary_tensors));
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrixBase<order, Number, n_rows_1d>::clear()
{
  tmp_array.clear();
  tensor_helper_row    = std::make_shared<const TensorHelper<order>>(0);
  tensor_helper_column = std::make_shared<const TensorHelper<order>>(0);
  elementary_tensors.clear();
}



template<int order, typename Number, int n_rows_1d>
template<bool add, bool transpose>
void
TensorProductMatrixBase<order, Number, n_rows_1d>::vmult_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  AssertDimension(dst_view.size(), transpose ? n() : m());
  AssertDimension(src_view.size(), transpose ? m() : n());
  Assert(tensor_helper_row, ExcMessage("Did you initialize tensor_helper_row?"));
  Assert(tensor_helper_column, ExcMessage("Did you initialize tensor_helper_column?"));

  constexpr bool is_dynamic_path = n_rows_1d == -1;
  if(!is_dynamic_path)
  {
    Assert(tensor_helper_row->is_isotropic(),
           ExcMessage("Currently, we support only isotropic tensors for the static path."));
    Assert(tensor_helper_column->is_isotropic(),
           ExcMessage("Currently, we support only isotropic tensors for the static path."));
  }

  /// The evaluator is aware of (possible) anisotropic tensor structure (for the dynamic path).
  evaluator_type eval(tensor_helper_row->size(), tensor_helper_column->size());

  /// TODO n_rows_1d =/= n_columns_1d (for the static path)
  constexpr int mm_static = Utilities::pow(n_rows_1d, order);

  /// We reserve the maximal amount of memory with respect to intermediate
  /// tensors occuring during sum factorization. The evaluator has to take care
  /// of not writing noise into (maybe to large) arrays.
  const unsigned int mm_dynamic = eval.template max_intermediate_size<transpose>();

  const unsigned int mm = is_dynamic_path ? mm_dynamic : mm_static;

  std::lock_guard<std::mutex> lock(this->mutex);
  const Number *              src = src_view.begin();
  Number *                    dst = dst_view.data();

  /// Apply elementary tensor of first rank.
  if(order == 1)
  {
    /// matrix_<tensor index>_<direction>
    const Number * matrix_0 = &(get_matrix_1d(0, 0)(0, 0));
    eval.template apply<0, transpose, add>(matrix_0, src, dst);

    for(std::size_t r = 1; r < elementary_tensors.size(); ++r)
    {
      const Number * matrix_r = &(get_matrix_1d(r, 0)(0, 0));
      eval.template apply<0, transpose, true>(matrix_r, src, dst);
    }
  }

  else if(order == 2)
  {
    tmp_array.clear();
    tmp_array.resize(mm);
    Number * tmp = tmp_array.begin();

    const Number * left_0  = &(get_matrix_1d(0, 1)(0, 0));
    const Number * right_0 = &(get_matrix_1d(0, 0)(0, 0));
    if(transpose)
    {
      eval.template apply<0, transpose, false>(right_0, src, tmp);
      eval.template apply<1, transpose, add>(left_0, tmp, dst);
    }
    else
    {
      eval.template apply<1, transpose, false>(left_0, src, tmp);
      eval.template apply<0, transpose, add>(right_0, tmp, dst);
    }

    /// Apply elementary tensors of remaining ranks.
    for(std::size_t r = 1; r < elementary_tensors.size(); ++r)
    {
      const Number * left_r  = &(get_matrix_1d(r, 1)(0, 0));
      const Number * right_r = &(get_matrix_1d(r, 0)(0, 0));
      if(transpose)
      {
        eval.template apply<0, transpose, false>(right_r, src, tmp);
        eval.template apply<1, transpose, true>(left_r, tmp, dst);
      }
      else
      {
        eval.template apply<1, transpose, false>(left_r, src, tmp);
        eval.template apply<0, transpose, true>(right_r, tmp, dst);
      }
    }
  }

  else if(order == 3)
  {
    tmp_array.clear();
    tmp_array.resize(2 * mm);
    Number * tmp  = tmp_array.begin();
    Number * tmp2 = tmp_array.begin() + mm;

    const Number * left_0  = &(get_matrix_1d(0, 2)(0, 0));
    const Number * mid_0   = &(get_matrix_1d(0, 1)(0, 0));
    const Number * right_0 = &(get_matrix_1d(0, 0)(0, 0));
    if(transpose)
    {
      eval.template apply<0, transpose, false>(right_0, src, tmp);
      eval.template apply<1, transpose, false>(mid_0, tmp, tmp2);
      eval.template apply<2, transpose, add>(left_0, tmp2, dst);
    }
    else
    {
      eval.template apply<2, transpose, false>(left_0, src, tmp);
      eval.template apply<1, transpose, false>(mid_0, tmp, tmp2);
      eval.template apply<0, transpose, add>(right_0, tmp2, dst);
    }

    /// Apply elementary tensors of remaining ranks.
    for(std::size_t r = 1; r < elementary_tensors.size(); ++r)
    {
      const Number * left_r  = &(get_matrix_1d(r, 2)(0, 0));
      const Number * mid_r   = &(get_matrix_1d(r, 1)(0, 0));
      const Number * right_r = &(get_matrix_1d(r, 0)(0, 0));
      if(transpose)
      {
        eval.template apply<0, transpose, false>(right_r, src, tmp);
        eval.template apply<1, transpose, false>(mid_r, tmp, tmp2);
        eval.template apply<2, transpose, true>(left_r, tmp2, dst);
      }
      else
      {
        eval.template apply<2, transpose, false>(left_r, src, tmp);
        eval.template apply<1, transpose, false>(mid_r, tmp, tmp2);
        eval.template apply<0, transpose, true>(right_r, tmp2, dst);
      }
    }
  }

  else
    AssertThrow(false, ExcMessage("Not implemented."));
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix_new<order, Number, n_rows_1d>::TensorProductMatrix_new()
  : TensorProductMatrixBase<order, Number, n_rows_1d>(),
    state(State::invalid),
    spd_mask(std::bitset<order>{})
{
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix_new<order, Number, n_rows_1d>::TensorProductMatrix_new(
  const std::vector<typename TensorProductMatrix_new<order, Number, n_rows_1d>::tensor_type> &
                           elementary_tensors_in,
  const State              state_in,
  const std::bitset<order> spd_mask_in)
{
  reinit(elementary_tensors_in, state_in, spd_mask_in);
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix_new<order, Number, n_rows_1d>::TensorProductMatrix_new(
  const typename TensorProductMatrix_new<order, Number, n_rows_1d>::tensor_type & rank1_tensor)
  : TensorProductMatrix_new<order, Number, n_rows_1d>::TensorProductMatrix_new(
      std::vector<typename TensorProductMatrix_new<order, Number, n_rows_1d>::tensor_type>{
        rank1_tensor})
{
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix_new<order, Number, n_rows_1d> &
TensorProductMatrix_new<order, Number, n_rows_1d>::operator=(const TensorProductMatrix_new & other)
{
  reinit(other.elementary_tensors, other.state, other.spd_mask);
  return *this;
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix_new<order, Number, n_rows_1d>::reinit(
  const std::vector<typename TensorProductMatrix_new<order, Number, n_rows_1d>::tensor_type> &
                           elementary_tensors_in,
  const State              state_in,
  const std::bitset<order> spd_mask_in)
{
  clear();

  state    = state_in;
  spd_mask = spd_mask_in;

  if(state == State::basic)
    Base::reinit(elementary_tensors_in);
  // else if(state == State::invalid)
  //   AssertThrow(false, ExcMessage("State is invalid."));
  else
    AssertThrow(false, ExcMessage("Not implemented."));
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix_new<order, Number, n_rows_1d>::clear()
{
  tmp_array.clear();
  state    = State::invalid;
  spd_mask = std::bitset<order>{};
  Base::clear();
}



template<int order, typename Number, int n_rows_1d>
template<bool add, bool transpose>
void
TensorProductMatrix_new<order, Number, n_rows_1d>::vmult_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  if(state == State::basic)
    Base::template vmult_impl<add, transpose>(dst_view, src_view);
  // else if(state == State::invalid)
  //   AssertThrow(false, ExcMessage("State is invalid."));
  else
    AssertThrow(false, ExcMessage("Not implemented."));
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix_new<order, Number, n_rows_1d>::vmult(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ false, /*transpose*/ false>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix_new<order, Number, n_rows_1d>::vmult_add(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ true, /*transpose*/ false>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix_new<order, Number, n_rows_1d>::Tvmult(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ false, /*transpose*/ true>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix_new<order, Number, n_rows_1d>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix_new<order, Number, n_rows_1d>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix_new<order, Number, n_rows_1d>::as_transpose_table() const
{
  return Tensors::transpose_matrix_to_table(*this);
}

} // namespace Tensors
