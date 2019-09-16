template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
void FDPatchEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::submit_cell_matrix(
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

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
void
FDPatchEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::evaluate(
  const bool do_gradients)
{
  /*** scale the reference weights in each direction to obtain JxW in 1D ***/
  if(!weights_filled)
    for(unsigned int d = 0; d < dim; ++d)
      for(unsigned int cell_no_1d = 0; cell_no_1d < n_cells_per_direction; ++cell_no_1d)
      {
        const auto                      h = h_lengths[d * n_cells_per_direction + cell_no_1d];
        const VectorizedArray<Number> * unit_weight = this->quad_weights_unit;
        auto                            JxW =
          this->JxWs + n_cells_per_direction * n_q_points_1d * d + n_q_points_1d * cell_no_1d;
        for(unsigned int quad_no = 0; quad_no < n_q_points_1d; ++JxW, ++unit_weight, ++quad_no)
          *JxW = (*unit_weight) * h;
        // std::cout << std::distance (JxW, scratch_fedata_end) << std::endl;
        Assert((cell_no_1d == n_cells_per_direction - 1 && d == dim - 1) ?
                 JxW == scratch_fedata_end :
                 true,
               ExcInternalError());
        weights_filled = true;
      }

  if(do_gradients)
  {
    /*** scale the 1d reference gradients with h_d^-1 in each direction d ***/
    for(unsigned int d = 0; d < dim; ++d)
      for(unsigned int cell_no_1d = 0; cell_no_1d < n_cells_per_direction; ++cell_no_1d)
      {
        const auto                      h_inv = h_inverses[d * n_cells_per_direction + cell_no_1d];
        const VectorizedArray<Number> * unit_grad = shape_info.shape_gradients.begin();
        VectorizedArray<Number> * grad = this->gradients[d] + n_q_points_1d * fe_order * cell_no_1d;

        for(unsigned int dof = 0; dof < fe_order; ++dof)
          for(unsigned int quad_no = 0; quad_no < n_q_points_1d; ++unit_grad, ++grad, ++quad_no)
          {
            *grad = (*unit_grad) * h_inv;
          }

        Assert(cell_no_1d == n_cells_per_direction - 1 ?
                 (d < dim - 1 ? grad == this->gradients[d + 1] : grad == this->gradients_face[0]) :
                 true,
               ExcInternalError());
      }
    /*** scale the 1d reference gradients in x=0 and x=1 with h_d^-1 in each direction d ***/
    for(const int face_no_1d : {0, 1})
    {
      const VectorizedArray<Number> * unit_grad =
        shape_info.shape_data_on_face[face_no_1d].begin() + fe_order;
      for(unsigned int d = 0; d < dim; ++d)
        for(unsigned int cell_no_1d = 0; cell_no_1d < n_cells_per_direction; ++cell_no_1d)
        {
          const auto                h_inv = h_inverses[d * n_cells_per_direction + cell_no_1d];
          VectorizedArray<Number> * grad  = this->gradients_face[d] + fe_order * face_no_1d +
                                           fe_order * n_cells_per_direction * cell_no_1d;

          for(unsigned int dof = 0; dof < fe_order; ++grad, ++dof)
            *grad = unit_grad[dof] * h_inv;

          Assert((cell_no_1d == n_cells_per_direction - 1 && face_no_1d == 1) ?
                   (d < dim - 1 ? grad == this->gradients_face[d + 1] : grad == JxWs) :
                   true,
                 ExcInternalError());
        }
    }
    gradients_filled = true;
  }
}
