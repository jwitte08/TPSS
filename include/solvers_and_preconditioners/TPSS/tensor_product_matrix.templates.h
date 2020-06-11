
namespace Tensors
{
template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::clear()
{
  tmp_array.clear();
  tensor_helper_row.reset();
  tensor_helper_column.reset();
  basic_inverse.reset();
  state = State::invalid;
  elementary_tensors.clear();
  /// TODO clear underlying TensorProductMatrixSymmetricSum
}

template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::reinit(
  const std::vector<std::array<Table<2, Number>, order>> & elementary_tensors_in,
  const State                                              state_in)
{
  Assert(check_n_rows_1d_static(elementary_tensors_in),
         ExcMessage("Not all univariate matrices are of size (n_rows_1d x n_rows_1d)."));

  /// clear data
  clear();
  /// is it better throw to an exception ?!
  if(elementary_tensors_in.empty())
    return;

  /// initialize data
  state = state_in;

  if(state == State::basic)
  {
    for(unsigned i = 0; i < order; ++i)
      Assert(check_n_rows_and_columns_1d(elementary_tensors_in, i),
             ExcMessage("Mismatching sizes of univariate matrices."));
    std::copy(elementary_tensors_in.cbegin(),
              elementary_tensors_in.cend(),
              std::back_inserter(elementary_tensors));

    const auto &                    first_elementary_tensor = elementary_tensors.front();
    std::array<unsigned int, order> n_rows_foreach_dimension;
    std::array<unsigned int, order> n_columns_foreach_dimension;
    for(auto d = 0; d < order; ++d)
    {
      const auto & matrix            = first_elementary_tensor[d];
      n_rows_foreach_dimension[d]    = matrix.size(0);
      n_columns_foreach_dimension[d] = matrix.size(1);
    }
    tensor_helper_row    = std::make_shared<const TensorHelper<order>>(n_rows_foreach_dimension);
    tensor_helper_column = std::make_shared<const TensorHelper<order>>(n_columns_foreach_dimension);
  }

  else if(state == State::separable)
  {
    AssertThrow(
      elementary_tensors_in.size() == 2,
      ExcMessage(
        "For this special case two rank-1 tensors are required, first the tensor of mass matrices and second the tensor of derivative matrices."));
    /// TODO avoid duplication
    // elementary_tensor = elementary_tensor_in;
    separable_matrix_type::reinit(elementary_tensors_in[0], elementary_tensors_in[1]);
  }

  else
    AssertThrow(false, ExcMessage("Did youe set a valid matrix state?"));
}


template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::apply_inverse(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  apply_inverse_impl(dst_view, src_view);
}

template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::apply_inverse_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  if(state == State::basic)
    apply_inverse_impl_basic_static(dst_view, src_view);
  else if(state == State::separable)
    separable_matrix_type::apply_inverse(dst_view, src_view);
  else
    AssertThrow(false, ExcMessage("Not implemented."));
}


template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::apply_inverse_impl_basic_static(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  AssertDimension(dst_view.size(), m());
  AssertDimension(src_view.size(), n());
  if(!basic_inverse)
    basic_inverse = std::make_shared<const InverseTable<Number>>(as_table());
  basic_inverse->vmult(dst_view, src_view);
}


template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::vmult(const ArrayView<Number> &       dst_view,
                                                     const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ false, /*transpose*/ false>(dst_view, src_view);
}


template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::vmult_add(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ true, /*transpose*/ false>(dst_view, src_view);
}


template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::Tvmult(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ false, /*transpose*/ true>(dst_view, src_view);
}


template<int order, typename Number, int n_rows_1d>
template<bool add, bool transpose>
void
TensorProductMatrix<order, Number, n_rows_1d>::vmult_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  if(state == State::basic)
    vmult_basic_impl<add, transpose>(dst_view, src_view);
  else if(state == State::separable)
    vmult_separable_impl<add, transpose>(dst_view, src_view);
  else
    AssertThrow(false, ExcMessage("Not implemented."));
}


template<int order, typename Number, int n_rows_1d>
template<bool add, bool transpose>
void
TensorProductMatrix<order, Number, n_rows_1d>::vmult_separable_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  AssertThrow(!transpose, ExcMessage("Not implemented."));

  AlignedVector<Number> initial_dst;
  if(add)
  {
    initial_dst.resize_fast(dst_view.size());
    std::copy(dst_view.cbegin(), dst_view.cend(), initial_dst.begin());
  }

  separable_matrix_type::vmult(dst_view, src_view);

  if(add)
  {
    std::transform(dst_view.cbegin(),
                   dst_view.cend(),
                   initial_dst.begin(),
                   dst_view.begin(),
                   [](const auto & elem1, const auto & elem2) { return elem1 + elem2; });
  }
}

template<int order, typename Number, int n_rows_1d>
template<bool add, bool transpose>
void
TensorProductMatrix<order, Number, n_rows_1d>::vmult_basic_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  static_assert(order == 2, "TODO");
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
    const Number * matrix_0_0 = &(get_matrix_1d(0, 0)(0, 0));
    eval.template apply<0, transpose, add>(matrix_0_0, src, dst);
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
    AssertThrow(false, ExcMessage("TODO"));
  }

  else
    AssertThrow(false, ExcMessage("Currently, this case is not supported."));
}

} // namespace Tensors
