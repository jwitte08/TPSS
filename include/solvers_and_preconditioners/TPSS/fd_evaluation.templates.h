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
  return patch_action_impl(eval_ansatz, std::forward<CellOperation>(cell_operation));
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
  return patch_action_impl(eval_ansatz, cell_operation, face_operation, interface_operation);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<int fe_degree_ansatz, int n_q_points_ansatz, typename CellOperation>
std::array<Table<2, VectorizedArray<Number>>, dim>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::patch_action_impl(
  const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
  CellOperation &&                                                       cell_operation) const
{
  using MatrixType                                    = Table<2, VectorizedArray<Number>>;
  const auto *                patch_dof_tensor_ansatz = &(eval_ansatz.get_dof_tensor());
  std::array<MatrixType, dim> matrices;
  for(int direction = 0; direction < dim; ++direction)
  {
    const auto n_dofs_1d_test            = this->n_dofs_1d(direction);
    const auto n_dofs_1d_ansatz          = patch_dof_tensor_ansatz->n_dofs_1d(direction);
    const auto n_dofs_per_cell_1d_test   = this->n_dofs_per_cell_1d(direction);
    const auto n_dofs_per_cell_1d_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

    /// assemble one-dimensional matrices on each cell (i.e. interval)
    MatrixType & matrix = matrices[direction];
    matrix.reinit(n_dofs_1d_test, n_dofs_1d_ansatz);
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      MatrixType cell_matrix(n_dofs_per_cell_1d_test, n_dofs_per_cell_1d_ansatz);
      std::forward<decltype(cell_operation)>(
        cell_operation)(eval_ansatz, *this, cell_matrix, direction, cell_no);
      submit_cell_matrix(matrix, cell_matrix, cell_no, cell_no, direction, patch_dof_tensor_ansatz);
    }
    AssertDimension(n_dofs_1d_test, matrices.at(direction).n_rows());
    AssertDimension(n_dofs_1d_test, matrices.at(direction).n_cols());
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
    AssertDimension(n_dofs_1d_test, matrices.at(direction).n_rows());
    AssertDimension(n_dofs_1d_test, matrices.at(direction).n_cols());
  }

  return matrices;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::submit_cell_matrix(
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
  const auto n_dofs_per_cell_1d_row = patch_dof_tensor_row.n_dofs_per_cell_1d(dimension);
  const auto n_dofs_per_cell_1d_col = patch_dof_tensor_col.n_dofs_per_cell_1d(dimension);
  /// assuming isotropy ...
  AssertDimension(fe_order, n_dofs_per_cell_1d_row);
  AssertDimension(fe_order, n_dofs_per_cell_1d_col);
  AssertIndexRange(dimension, dim);
  AssertDimension(cell_matrix.n_rows(), n_dofs_per_cell_1d_row);
  AssertDimension(cell_matrix.n_cols(), n_dofs_per_cell_1d_col);
  AssertDimension(patch_dof_tensor_row.n_dofs_1d(dimension), subdomain_matrix.n_rows());
  AssertDimension(patch_dof_tensor_col.n_dofs_1d(dimension), subdomain_matrix.n_cols());
  for(unsigned int i = 0; i < n_dofs_per_cell_1d_row; ++i)
    for(unsigned int j = 0; j < n_dofs_per_cell_1d_col; ++j)
    {
      const auto ii = patch_dof_tensor_row.dof_index_1d(cell_no_row, i, dimension);
      const auto jj = patch_dof_tensor_col.dof_index_1d(cell_no_col, j, dimension);
      subdomain_matrix(ii, jj) += cell_matrix(i, j);
    }
}
