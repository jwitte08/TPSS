template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::submit_cell_matrix(
  Table<2, VectorizedArray<Number>> &       subdomain_matrix,
  const Table<2, VectorizedArray<Number>> & cell_matrix,
  const unsigned int                        cell_no_row,
  const unsigned int                        cell_no_col)
{
  AssertDimension(subdomain_matrix.n_rows() % fe_order, 0);
  AssertDimension(subdomain_matrix.n_rows(), subdomain_matrix.n_cols()); // is quadratic
  AssertDimension(cell_matrix.n_rows(), fe_order);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols()); // is quadratic
  AssertIndexRange((cell_no_row + 1) * fe_order, subdomain_matrix.n_rows() + 1);
  AssertIndexRange((cell_no_col + 1) * fe_order, subdomain_matrix.n_cols() + 1);

  const unsigned int row_start = cell_no_row * fe_order;
  const unsigned int col_start = cell_no_col * fe_order;
  for(unsigned int row = 0; row < fe_order; ++row)
    for(unsigned int col = 0; col < fe_order; ++col)
      subdomain_matrix(row_start + row, col_start + col) += cell_matrix(row, col);
}

template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::evaluate(const bool do_gradients)
{
  /// univariate Jacobian, that is h_d, times quadrature weight
  const VectorizedArray<Number> * weight = this->q_weights_unit;
  for(unsigned int d = 0; d < dim; ++d)
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      const auto h = get_h(d, cell_no);
      for(unsigned int q = 0; q < n_q_points_1d_static; ++q)
        get_JxW_impl(q, d, cell_no) = h * weight[q]; // JxW
    }

  if(do_gradients)
  {
    /// scale univariate reference gradients with h_d^{-1}
    const auto & patch_dof_tensor = patch_worker.get_dof_tensor();
    for(unsigned int d = 0; d < dim; ++d)
    {
      const auto & shape_info         = get_shape_info(d);
      const auto   n_q_points_1d      = this->n_q_points_1d(d);
      const auto   n_dofs_per_cell_1d = this->n_dofs_per_cell_1d(d);
      const auto * unit_grads_begin   = shape_info.shape_gradients.begin();
      for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
      {
        const auto h_inv = 1. / get_h(d, cell_no);
        for(unsigned int dof = 0; dof < n_dofs_per_cell_1d; ++dof)
        {
          const auto * unit_grad_begin = unit_grads_begin + dof * n_q_points_1d;
          auto *       grad            = &(shape_gradient_impl(dof, 0, d, cell_no));
          std::transform(unit_grad_begin,
                         unit_grad_begin + n_q_points_1d,
                         grad,
                         [h_inv](const auto & unit_grad) { return unit_grad * h_inv; });
        }
      }
    }

    /*** scale the 1d reference gradients in x=0 and x=1 with h_d^-1 in each direction d ***/
    for(const int face_no : {0, 1})
      for(unsigned int d = 0; d < dim; ++d)
      {
        const VectorizedArray<Number> * unit_grad =
          patch_worker.get_shape_info(d).shape_data_on_face[face_no].begin() + fe_order;
        // const VectorizedArray<Number> * unit_grad =
        //   shape_info.shape_data_on_face[face_no].begin() + fe_order;

        for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
        {
          const auto h_inv =
            1. / get_h(d, cell_no); //_inverses[d * n_cells_per_direction + cell_no];
          VectorizedArray<Number> * grad = this->gradients_face[d] + fe_order * face_no +
                                           fe_order * n_cells_per_direction * cell_no;

          for(unsigned int dof = 0; dof < fe_order; ++grad, ++dof)
            *grad = unit_grad[dof] * h_inv;

          Assert((cell_no == n_cells_per_direction - 1 && face_no == 1) ?
                   (d < dim - 1 ? grad == this->gradients_face[d + 1] : grad == JxWs) :
                   true,
                 ExcInternalError());
        }
      }
    gradients_filled = true;
  }
}
