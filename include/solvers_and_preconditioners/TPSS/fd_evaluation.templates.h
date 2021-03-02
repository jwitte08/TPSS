template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::evaluate(const bool do_gradients,
                                                               const bool do_hessians)
{
  /// univariate Jacobian, that is h_d, times quadrature weight
  const VectorizedArray<Number> * weight = this->q_weights_unit;
  for(unsigned int d = 0; d < dim; ++d)
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      const auto h = get_h_impl(d, cell_no);
      for(unsigned int q = 0; q < n_q_points_1d_static; ++q)
        get_JxW_impl(q, d, cell_no) = h * weight[q]; // JxW
    }

  if(do_gradients)
    evaluate_gradients();

  if(do_hessians)
    evaluate_hessians();
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::evaluate_gradients()
{
  /// scale univariate reference gradients with h_d^{-1}
  for(unsigned int d = 0; d < dim; ++d)
  {
    const auto & shape_data         = get_shape_data(d, component);
    const auto   n_q_points_1d      = this->n_q_points_1d(d);
    const auto   n_dofs_per_cell_1d = this->n_dofs_per_cell_1d(d);
    const auto * unit_grads_begin   = shape_data.shape_gradients.begin();
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      const auto h_inv = 1. / get_h_impl(d, cell_no);
      for(unsigned int dof = 0; dof < n_dofs_per_cell_1d; ++dof)
      {
        const auto * unit_grad_begin = unit_grads_begin + dof * n_q_points_1d;
        auto *       grad            = &(shape_gradient_impl(dof, 0, d, cell_no));
        std::transform(unit_grad_begin,
                       unit_grad_begin + n_q_points_1d,
                       grad,
                       [h_inv](const auto & unit_grad) { return unit_grad * h_inv; });
      }

      for(const int face_no : {0, 1})
      {
        const auto * unit_grads_on_face = shape_data.shape_data_on_face[face_no].begin() + fe_order;
        auto *       grad_on_face       = &(shape_gradient_face_impl(0, face_no, d, cell_no));
        std::transform(unit_grads_on_face,
                       unit_grads_on_face + fe_order,
                       grad_on_face,
                       [h_inv](const auto & unit_grad) { return unit_grad * h_inv; });
      }
    }
  }

  gradients_filled = true;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::evaluate_hessians()
{
  /// scale univariate reference hessians with h_d^{-2}
  for(unsigned int d = 0; d < dim; ++d)
  {
    const auto & shape_data          = get_shape_data(d, component);
    const auto   n_q_points_1d       = this->n_q_points_1d(d);
    const auto   n_dofs_per_cell_1d  = this->n_dofs_per_cell_1d(d);
    const auto * unit_hessians_begin = shape_data.shape_hessians.begin();
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      const auto h_inv = 1. / get_h_impl(d, cell_no);
      for(unsigned int dof = 0; dof < n_dofs_per_cell_1d; ++dof)
      {
        const auto * unit_hessian_begin = unit_hessians_begin + dof * n_q_points_1d;
        auto *       hessian            = &(shape_hessian_impl(dof, 0, d, cell_no));
        std::transform(unit_hessian_begin,
                       unit_hessian_begin + n_q_points_1d,
                       hessian,
                       [h_inv](const auto & unit_hessian) { return unit_hessian * h_inv * h_inv; });
      }

      for(const int face_no : {0, 1})
      {
        const auto * unit_hessians_on_face =
          shape_data.shape_data_on_face[face_no].begin() + 2 * fe_order;
        auto * hessian_on_face = &(shape_hessian_face_impl(0, face_no, d, cell_no));
        std::transform(unit_hessians_on_face,
                       unit_hessians_on_face + fe_order,
                       hessian_on_face,
                       [h_inv](const auto & unit_hessian) { return unit_hessian * h_inv * h_inv; });
      }
    }
  }

  hessians_filled = true;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<typename CellOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action(
  CellOperation && cell_operation) const
{
  return patch_action(*this, std::forward<CellOperation>(cell_operation));
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<int fe_degree_ansatz, int n_q_points_ansatz, typename CellOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action(
  const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
  CellOperation &&                                                       cell_operation) const
{
  auto matrices = patch_action_impl(eval_ansatz, std::forward<CellOperation>(cell_operation));

  /// Setting already the one-dimensional boundary conditions avoids singular
  /// matrices.
  post_process_constraints(matrices, eval_ansatz);

  return matrices;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<typename CellOperation, typename FaceOperation, typename InterfaceOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action(
  const CellOperation &      cell_operation,
  const FaceOperation &      face_operation,
  const InterfaceOperation & interface_operation) const
{
  return patch_action(*this, cell_operation, face_operation, interface_operation);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<int fe_degree_ansatz,
         int n_q_points_ansatz,
         typename CellOperation,
         typename FaceOperation,
         typename InterfaceOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action(
  const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
  const CellOperation &                                                  cell_operation,
  const FaceOperation &                                                  face_operation,
  const InterfaceOperation &                                             interface_operation) const
{
  auto matrices =
    patch_action_impl(eval_ansatz, cell_operation, face_operation, interface_operation);

  /// Setting already the one-dimensional boundary conditions avoids singular
  /// matrices.
  post_process_constraints(matrices, eval_ansatz);

  return matrices;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<int fe_degree_ansatz, int n_q_points_ansatz, typename CellOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action_impl(
  const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
  CellOperation &&                                                       cell_operation) const
{
  using MatrixType                     = Table<2, VectorizedArray<Number>>;
  const auto * patch_dof_tensor_ansatz = &(eval_ansatz.get_dof_tensor());

  std::array<MatrixType, dim> matrices;
  for(int direction = 0; direction < dim; ++direction)
  {
    const auto n_dofs_plain_1d_test      = this->n_dofs_plain_1d(direction);
    const auto n_dofs_plain_1d_ansatz    = eval_ansatz.n_dofs_plain_1d(direction);
    const auto n_dofs_per_cell_1d_test   = this->n_dofs_per_cell_1d(direction);
    const auto n_dofs_per_cell_1d_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

    /// assemble one-dimensional matrices on each cell (i.e. interval)
    MatrixType & matrix = matrices[direction];
    matrix.reinit(n_dofs_plain_1d_test, n_dofs_plain_1d_ansatz);
    for(unsigned int cell_no = 0; cell_no < n_cells_1d(direction); ++cell_no)
    {
      MatrixType cell_matrix(n_dofs_per_cell_1d_test, n_dofs_per_cell_1d_ansatz);
      std::forward<decltype(cell_operation)>(
        cell_operation)(eval_ansatz, *this, cell_matrix, direction, cell_no);
      submit_cell_matrix(matrix, cell_matrix, cell_no, cell_no, direction, patch_dof_tensor_ansatz);
    }
    AssertDimension(n_dofs_plain_1d_test, matrices.at(direction).n_rows());
    AssertDimension(n_dofs_plain_1d_ansatz, matrices.at(direction).n_cols());
  }
  return matrices;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<int fe_degree_ansatz,
         int n_q_points_ansatz,
         typename CellOperation,
         typename FaceOperation,
         typename InterfaceOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action_impl(
  const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
  CellOperation &&                                                       cell_operation,
  FaceOperation &&                                                       face_operation,
  InterfaceOperation &&                                                  interface_operation) const
{
  using MatrixType                     = Table<2, VectorizedArray<Number>>;
  const auto * patch_dof_tensor_ansatz = &(eval_ansatz.get_dof_tensor());

  /// Assemble 1D matrices on each 1D cell @p cell_no for each spatial
  /// dimension.
  auto matrices =
    patch_action_impl(eval_ansatz, std::forward<decltype(cell_operation)>(cell_operation));

  for(int direction = 0; direction < dim; ++direction)
  {
    const auto n_dofs_per_cell_1d_test   = this->n_dofs_per_cell_1d(direction);
    const auto n_dofs_per_cell_1d_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

    auto & matrix = matrices[direction];
    for(unsigned int cell_no = 0; cell_no < n_cells_1d(direction); ++cell_no)
    {
      /// Assemble 1D matrices on each 1D face @p face_no with respect to
      /// the shape functions on 1D cell @p cell_no.
      for(const int face_no : {0, 1})
      {
        MatrixType cell_matrix(n_dofs_per_cell_1d_test, n_dofs_per_cell_1d_ansatz);
        std::forward<decltype(face_operation)>(
          face_operation)(eval_ansatz, *this, cell_matrix, direction, cell_no, face_no);
        /// TODO note, it is safer to submit after each operation but not efficient ...
        submit_cell_matrix(
          matrix, cell_matrix, cell_no, cell_no, direction, patch_dof_tensor_ansatz);
      }

      /// Assemble 1D matrices on 1D interfaces between left cell and right
      /// cell with respect to spatial direction. We compute interactions
      /// between test functions from left cell and ansatz functions from
      /// right cell (@p cell_matrix_leftright) and vice versa (@p
      /// cell_matrix_rightleft).
      const bool has_adjacent_cell_on_rhs = cell_no + 1 < n_cells_1d(direction);
      if(has_adjacent_cell_on_rhs)
      {
        const auto cell_no_left  = cell_no;
        const auto cell_no_right = cell_no + 1;
        MatrixType cell_matrix_leftright(n_dofs_per_cell_1d_test, n_dofs_per_cell_1d_ansatz);
        MatrixType cell_matrix_rightleft(n_dofs_per_cell_1d_test, n_dofs_per_cell_1d_ansatz);
        std::forward<decltype(interface_operation)>(interface_operation)(eval_ansatz,
                                                                         *this,
                                                                         cell_matrix_leftright,
                                                                         cell_matrix_rightleft,
                                                                         cell_no_left,
                                                                         direction);
        submit_cell_matrix(matrix,
                           cell_matrix_leftright,
                           cell_no_left,
                           cell_no_right,
                           direction,
                           patch_dof_tensor_ansatz);
        submit_cell_matrix(matrix,
                           cell_matrix_rightleft,
                           cell_no_right,
                           cell_no_left,
                           direction,
                           patch_dof_tensor_ansatz);
      }
    }
    AssertDimension(this->n_dofs_plain_1d(direction), matrices.at(direction).n_rows());
    AssertDimension(eval_ansatz.n_dofs_plain_1d(direction), matrices.at(direction).n_cols());
  }

  return matrices;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<int fe_degree_ansatz, int n_q_points_ansatz>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::post_process_constraints(
  std::array<Table<2, VectorizedArray<Number>>, dim> &                   matrices,
  const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz) const
{
  if(TPSS::DoFLayout::DGQ == get_dof_layout())
    return;

  else if(TPSS::DoFLayout::Q == get_dof_layout())
  {
    for(auto direction = 0U; direction < dim; ++direction)
    {
      auto & matrix = matrices[direction];

      if(patch_variant == TPSS::PatchVariant::vertex)
      {
        restrict_matrix_qvp(matrix, patch_worker, eval_ansatz.patch_worker);
        AssertDimension(this->patch_worker.n_dofs_1d(direction), matrices.at(direction).n_rows());
        AssertDimension(eval_ansatz.patch_worker.n_dofs_1d(direction),
                        matrices.at(direction).n_cols());
      }
      else
        AssertThrow(false, ExcMessage("Patch variant is not supported."));
    }
    return;
  }

  AssertThrow(false, ExcMessage("Post processing of constraints is not implemented."));
  return;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::restrict_matrix_qvp(
  Table<2, VectorizedArray<Number>> &       subdomain_matrix,
  const TPSS::PatchDoFWorker<dim, Number> & patch_worker_row,
  const TPSS::PatchDoFWorker<dim, Number> & patch_worker_col) const
{
  const auto   subdomain_matrix_plain = subdomain_matrix;
  const auto & additional_data_row    = patch_worker_row.get_dof_info().get_additional_data();
  const auto & additional_data_col    = patch_worker_col.get_dof_info().get_additional_data();
  const auto   n_rows_restricted      = additional_data_row.force_no_boundary_condition ?
                                   subdomain_matrix_plain.n_rows() :
                                   subdomain_matrix_plain.n_rows() - 2;
  const auto n_cols_restricted = additional_data_col.force_no_boundary_condition ?
                                   subdomain_matrix_plain.n_cols() :
                                   subdomain_matrix_plain.n_cols() - 2;

  subdomain_matrix.reinit(n_rows_restricted, n_cols_restricted);
  for(auto i = 0U; i < subdomain_matrix.n_rows(); ++i)
    for(auto j = 0U; j < subdomain_matrix.n_cols(); ++j)
    {
      const auto ii          = additional_data_row.force_no_boundary_condition ? i : i + 1;
      const auto jj          = additional_data_col.force_no_boundary_condition ? j : j + 1;
      subdomain_matrix(i, j) = subdomain_matrix_plain(ii, jj);
    }
}


/// TODO this functionality should be provided by PatchLocalTensorHelper ???
template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::submit_cell_matrix(
  Table<2, VectorizedArray<Number>> &       subdomain_matrix,
  const Table<2, VectorizedArray<Number>> & cell_matrix,
  const unsigned int                        cell_no_row,
  const unsigned int                        cell_no_col,
  const unsigned int                        dimension,
  const TPSS::PatchLocalTensorHelper<dim> * patch_dof_tensor_ansatz) const
{
  const auto & patch_dof_tensor_row = get_dof_tensor();
  const auto & patch_dof_tensor_col =
    patch_dof_tensor_ansatz ? *patch_dof_tensor_ansatz : patch_dof_tensor_row;
  const auto n_dofs_per_cell_1d_row = patch_dof_tensor_row.get_cell_dof_tensor().size(dimension);
  const auto n_dofs_per_cell_1d_col = patch_dof_tensor_col.get_cell_dof_tensor().size(dimension);
  /// assuming isotropy ...
  AssertDimension(fe_order, n_dofs_per_cell_1d_row);
  AssertIndexRange(dimension, dim);
  AssertDimension(cell_matrix.n_rows(), n_dofs_per_cell_1d_row);
  AssertDimension(cell_matrix.n_cols(), n_dofs_per_cell_1d_col);
  AssertDimension(patch_dof_tensor_row.n_dofs_1d(dimension), subdomain_matrix.n_rows());
  AssertDimension(patch_dof_tensor_col.n_dofs_1d(dimension), subdomain_matrix.n_cols());
  for(unsigned int i = 0; i < n_dofs_per_cell_1d_row; ++i)
    for(unsigned int j = 0; j < n_dofs_per_cell_1d_col; ++j)
    {
      const auto ii = patch_dof_tensor_row.plain.dof_index_1d(cell_no_row, i, dimension);
      const auto jj = patch_dof_tensor_col.plain.dof_index_1d(cell_no_col, j, dimension);
      subdomain_matrix(ii, jj) += cell_matrix(i, j);
    }
}
