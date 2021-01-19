
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
void
TensorProductMatrixBase<order, Number, n_rows_1d>::vmult(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ false, /*transpose*/ false>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrixBase<order, Number, n_rows_1d>::vmult_add(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ true, /*transpose*/ false>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrixBase<order, Number, n_rows_1d>::Tvmult(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ false, /*transpose*/ true>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrixBase<order, Number, n_rows_1d>::Tvmult_add(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ true, /*transpose*/ true>(dst_view, src_view);
}



////////// Definitions: TensorProductMatrix



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix()
  : TensorProductMatrixBase<order, Number, n_rows_1d>()
{
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix(
  const std::vector<typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type> &
                         elementary_tensors_in,
  const AdditionalData & additional_data_in)
{
  reinit(elementary_tensors_in, additional_data_in);
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix(
  const std::vector<typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type> &
                           elementary_tensors_in,
  const State              state_in,
  const std::bitset<order> spd_mask_in)
  : TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix(elementary_tensors_in,
                                                                       {state_in, spd_mask_in})
{
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix(
  const typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type & rank1_tensor)
  : TensorProductMatrix<order, Number, n_rows_1d>::TensorProductMatrix(
      std::vector<typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type>{
        rank1_tensor})
{
}



template<int order, typename Number, int n_rows_1d>
TensorProductMatrix<order, Number, n_rows_1d> &
TensorProductMatrix<order, Number, n_rows_1d>::operator=(const TensorProductMatrix & other)
{
  reinit(other.elementary_tensors, other.additional_data);
  return *this;
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::reinit(
  const std::vector<typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type> &
                         elementary_tensors_in,
  const AdditionalData & additional_data_in)
{
  clear();

  additional_data = additional_data_in;

  if(additional_data.state == State::basic)
  {
    Base::reinit(elementary_tensors_in);
  }

  else if(additional_data.state == State::ranktwo)
  {
    AssertDimension(elementary_tensors_in.size(), 2U);

    Base::reinit(elementary_tensors_in);

    tensor_type eigenvector_tensor;
    internal::ComputeGeneralizedEigendecomposition<order, Number, n_rows_1d>{}(
      this->eigenvalues, eigenvector_tensor, elementary_tensors_in, additional_data_in.spd_mask);
    this->eigenvectors.reinit(std::vector<tensor_type>{eigenvector_tensor});
  }

  else if(additional_data.state == State::separable)
  {
    AssertThrow(
      elementary_tensors_in.size() == 2U,
      ExcMessage(
        "elementary_tensors_in must contain exactly two elements: a tensor of mass matrices first and a tensor of derivative matrices second!"));
    const auto & MM = elementary_tensors_in.front();
    const auto & DD = elementary_tensors_in.back();

    std::vector<tensor_type> expanded_tensors(order);
    for(auto r = 0U; r < order; ++r)
      for(auto d = 0U; d < order; ++d)
      {
        if(r == d)
          expanded_tensors[r][d] = DD[d];
        else
          expanded_tensors[r][d] = MM[d];
      }

    Base::reinit(expanded_tensors);
    AssertDimension(this->elementary_tensors.size(), order);

    Assert(additional_data_in.spd_mask == std::bitset<order>{},
           ExcMessage("mass muss be first and s.p.d."));
    tensor_type eigenvector_tensor;
    internal::ComputeGeneralizedEigendecomposition<order, Number, n_rows_1d>{}(
      this->eigenvalues, eigenvector_tensor, elementary_tensors_in, std::bitset<order>{});
    this->eigenvectors.reinit(std::vector<tensor_type>{eigenvector_tensor});
  }

  else if(additional_data.state == State::rankone)
  {
    AssertDimension(elementary_tensors_in.size(), 1U);

    Base::reinit(elementary_tensors_in);

    /// due to laziness compute gen. eigendecomp. w.r.t. identity matrices
    std::vector<tensor_type> ranktwo_tensors;
    ranktwo_tensors.emplace_back(
      make_id_tensor<order, Number, unsigned int>(this->tensor_m(), this->tensor_n()));
    ranktwo_tensors.emplace_back(elementary_tensors_in.front());
    Assert(this->check_n_rows_and_columns_1d(ranktwo_tensors),
           ExcMessage("Mismatching matrix sizes."));
    tensor_type eigenvector_tensor;
    internal::ComputeGeneralizedEigendecomposition<order, Number, n_rows_1d>{}(
      this->eigenvalues, eigenvector_tensor, ranktwo_tensors, std::bitset<order>{});
    this->eigenvectors.reinit(std::vector<tensor_type>{eigenvector_tensor});
  }

  else if(additional_data.state == State::invalid)
  {
    AssertThrow(false, ExcMessage("State is invalid."));
  }

  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::reinit(
  const std::vector<typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type> &
                           elementary_tensors_in,
  const State              state_in,
  const std::bitset<order> spd_mask_in)
{
  AdditionalData additional_data_in;
  additional_data_in.state    = state_in;
  additional_data_in.spd_mask = spd_mask_in;
  reinit(elementary_tensors_in, additional_data_in);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::clear()
{
  additional_data = {State::invalid, std::bitset<order>{}, false};
  basic_inverse.reset();
  Base::clear();
}



template<int order, typename Number, int n_rows_1d>
template<bool add, bool transpose>
void
TensorProductMatrix<order, Number, n_rows_1d>::vmult_impl(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  /// TODO
  Base::template vmult_impl<add, transpose>(dst_view, src_view);
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
TensorProductMatrix<order, Number, n_rows_1d>::vmult(AlignedVector<Number> &       dst,
                                                     const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  vmult(dst_view, src_view);
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
TensorProductMatrix<order, Number, n_rows_1d>::vmult_add(AlignedVector<Number> &       dst,
                                                         const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  vmult_add(dst_view, src_view);
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
void
TensorProductMatrix<order, Number, n_rows_1d>::Tvmult(AlignedVector<Number> &       dst,
                                                      const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  Tvmult(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::Tvmult_add(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  vmult_impl</*add*/ true, /*transpose*/ true>(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::Tvmult_add(AlignedVector<Number> &       dst,
                                                          const AlignedVector<Number> & src) const
{
  const ArrayView<Number>       dst_view(dst.begin(), dst.size());
  const ArrayView<const Number> src_view(src.begin(), src.size());
  Tvmult_add(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
AlignedVector<Number>
TensorProductMatrix<order, Number, n_rows_1d>::get_eigenvalues() const
{
  if(additional_data.state == State::ranktwo)
  {
    return compute_eigenvalues_impl_ranktwo();
  }
  else if(additional_data.state == State::separable)
  {
    return compute_eigenvalues_impl_separable();
  }
  else if(additional_data.state == State::rankone)
  {
    return compute_eigenvalues_impl_rankone();
  }
  else if(additional_data.state == State::invalid)
  {
    AssertThrow(false, ExcMessage("State is invalid."));
  }
  else
  {
    Assert(false, ExcMessage("Current state isn't supported."));
  }

  return AlignedVector<Number>{};
}



template<int order, typename Number, int n_rows_1d>
const std::array<AlignedVector<Number>, order> &
TensorProductMatrix<order, Number, n_rows_1d>::get_eigenvalue_tensor() const
{
  Assert(additional_data.state == State::ranktwo || additional_data.state == State::separable ||
           additional_data.state == State::rankone,
         ExcMessage("Functionality isn't supported in current state."));
  return eigenvalues;
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::get_eigenvectors() const
{
  Assert(additional_data.state == State::ranktwo || additional_data.state == State::separable ||
           additional_data.state == State::rankone,
         ExcMessage("Current state isn't supported."));
  Assert(!eigenvectors.empty(), ExcMessage("eigenvectors is empty."));
  return Tensors::matrix_to_table(eigenvectors);
}



template<int order, typename Number, int n_rows_1d>
const typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type &
TensorProductMatrix<order, Number, n_rows_1d>::get_eigenvector_tensor() const
{
  Assert(additional_data.state == State::ranktwo || additional_data.state == State::separable ||
           additional_data.state == State::rankone,
         ExcMessage("Functionality isn't supported in current state."));
  AssertDimension(this->eigenvectors.get_elementary_tensors().size(), 1U);
  return eigenvectors.get_elementary_tensors().front();
}



template<int order, typename Number, int n_rows_1d>
typename TensorProductMatrix<order, Number, n_rows_1d>::tensor_type
TensorProductMatrix<order, Number, n_rows_1d>::get_mass_tensor() const
{
  Assert(additional_data.state == State::separable,
         ExcMessage("Current matrix state isn't supported."));
  AssertDimension(this->elementary_tensors.size(), order);
  tensor_type mass_tensor;
  for(auto d = 0U; d < order; ++d)
    mass_tensor[d] = this->elementary_tensors[d][(d + 1) % order];
  return mass_tensor;
}



template<int order, typename Number, int n_rows_1d>
AlignedVector<Number>
TensorProductMatrix<order, Number, n_rows_1d>::compute_eigenvalues_impl_ranktwo() const
{
  AssertDimension(this->m(), this->n()); // square matrix?

  const auto & eigenvalues_foreach_dimension = this->eigenvalues;

  Number                lambda_r(0.);
  AlignedVector<Number> eigenvalues(this->m());
  for(unsigned int i = 0; i < eigenvalues.size(); ++i)
  {
    const auto & ii = this->tensor_helper_row->multi_index(i);

    AssertDimension(n_max_rank(), 2U);
    for(auto r = 0U; r < 2U; ++r)
    {
      const bool is_one = r == static_cast<unsigned int>(additional_data.spd_mask[0]);
      lambda_r = is_one ? static_cast<Number>(1.) : eigenvalues_foreach_dimension[0][ii[0]];
      for(auto d = 1; d < order; ++d)
      {
        const bool is_one = r == static_cast<unsigned int>(additional_data.spd_mask[d]);
        lambda_r *= is_one ? static_cast<Number>(1.) : eigenvalues_foreach_dimension[d][ii[d]];
      }
      eigenvalues[i] += lambda_r;
    }
  }

  return eigenvalues;
}



template<int order, typename Number, int n_rows_1d>
AlignedVector<Number>
TensorProductMatrix<order, Number, n_rows_1d>::compute_eigenvalues_impl_rankone() const
{
  AssertDimension(this->m(), this->n()); // square matrix?
  AssertDimension(n_max_rank(), 1U);

  const auto & eigenvalues_foreach_dimension = this->eigenvalues;

  Number                lambda_r(0.);
  AlignedVector<Number> eigenvalues(this->m());
  for(unsigned int i = 0; i < eigenvalues.size(); ++i)
  {
    const auto & ii = this->tensor_helper_row->multi_index(i);

    lambda_r = eigenvalues_foreach_dimension[0][ii[0]];
    for(auto d = 1; d < order; ++d)
      lambda_r *= eigenvalues_foreach_dimension[d][ii[d]];

    eigenvalues[i] += lambda_r;
  }

  return eigenvalues;
}



template<int order, typename Number, int n_rows_1d>
AlignedVector<Number>
TensorProductMatrix<order, Number, n_rows_1d>::compute_eigenvalues_impl_separable() const
{
  AssertDimension(this->m(), this->n()); // square matrix?

  const auto & eigenvalues_foreach_dimension = this->eigenvalues;

  AssertDimension(n_max_rank(), order);
  AlignedVector<Number> eigenvalues(this->m());
  for(unsigned int i = 0; i < eigenvalues.size(); ++i)
  {
    const auto & ii = this->tensor_helper_row->multi_index(i);
    for(auto r = 0U; r < order; ++r)
      eigenvalues[i] += eigenvalues_foreach_dimension[r][ii[r]];
  }

  return eigenvalues;
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
  AssertDimension(this->m(), this->n());
  AssertDimension(dst_view.size(), this->m());
  AssertDimension(src_view.size(), this->m());

  if(additional_data.state == State::basic)
  {
    Assert(!additional_data.force_positive_definite_inverse, ExcMessage("Not supported."));
    apply_inverse_impl_basic(dst_view, src_view);
  }

  else if(additional_data.state == State::ranktwo || additional_data.state == State::separable ||
          additional_data.state == State::rankone)
  {
    if(additional_data.force_positive_definite_inverse)
      apply_inverse_impl_eigen<true>(dst_view, src_view);
    else
      apply_inverse_impl_eigen<false>(dst_view, src_view);
  }

  else if(additional_data.state == State::invalid)
  {
    AssertThrow(false, ExcMessage("The state is invalid."));
  }

  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}



template<int order, typename Number, int n_rows_1d>
template<bool zero_out_negative_eigenvalues>
void
TensorProductMatrix<order, Number, n_rows_1d>::apply_inverse_impl_eigen(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  AssertDimension(this->m(), this->n());
  AssertDimension(eigenvectors.m(), eigenvectors.n());
  AssertDimension(eigenvectors.m(), this->m());

  this->tmp_array.clear();
  this->tmp_array.resize(this->m());
  ArrayView<Number> tmp_view = make_array_view(this->tmp_array.begin(), this->tmp_array.end());

  eigenvectors.Tvmult(tmp_view, src_view);

  const auto & eigenvalues = get_eigenvalues();
  // if(additional_data.force_positive_definite_inverse)
  //   std::transform(eigenvalues.begin(),
  //                  eigenvalues.end(),
  //                  eigenvalues.begin(),
  //                  [](const auto & lambda) { return zero_out_negative_value(lambda); });
  std::transform(tmp_view.cbegin(),
                 tmp_view.cend(),
                 eigenvalues.begin(),
                 tmp_view.begin(),
                 /// avoid division by zero for each lane
                 [](const auto & value, const auto & lambda) {
                   const auto & inverse_lambda =
                     (zero_out_negative_eigenvalues ?
                        zero_out_negative_value(inverse_scalar_if<Number>(lambda)) :
                        inverse_scalar_if<Number>(lambda));
                   return value * inverse_lambda;
                 });

  eigenvectors.vmult(dst_view, tmp_view);
}



template<int order, typename Number, int n_rows_1d>
void
TensorProductMatrix<order, Number, n_rows_1d>::apply_inverse_impl_basic(
  const ArrayView<Number> &       dst_view,
  const ArrayView<const Number> & src_view) const
{
  if(!basic_inverse)
    basic_inverse = std::make_shared<const InverseTable<Number>>(as_table());
  basic_inverse->vmult(dst_view, src_view);
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::as_table() const
{
  return Tensors::matrix_to_table(*this);
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::as_inverse_table() const
{
  return Tensors::inverse_matrix_to_table(*this);
}



template<int order, typename Number, int n_rows_1d>
Table<2, Number>
TensorProductMatrix<order, Number, n_rows_1d>::as_transpose_table() const
{
  return Tensors::transpose_matrix_to_table(*this);
}

} // namespace Tensors
