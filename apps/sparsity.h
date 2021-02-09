/**
 * sparsity.h
 *
 * collection of helper functions initializing sparsity patterns
 *
 *  Created on: Feb 07, 2021
 *      Author: witte
 */

#ifndef SPARSITY_H_
#define SPARSITY_H_

#include <deal.II/dofs/dof_tools.h>



using namespace dealii;

namespace Tools
{
template<int dim, int spacedim, typename SparsityPatternType>
void
make_sparsity_pattern(const DoFHandler<dim, spacedim> & dofh_row,
                      const DoFHandler<dim, spacedim> & dofh_column,
                      SparsityPatternType &             sparsity)
// const types::subdomain_id subdomain_id = numbers::invalid_subdomain_id)
{
  const types::global_dof_index n_dofs_row    = dofh_row.n_dofs();
  const types::global_dof_index n_dofs_column = dofh_column.n_dofs();
  (void)n_dofs_row, (void)n_dofs_column;

  Assert(sparsity.n_rows() == n_dofs_row, ExcDimensionMismatch(sparsity.n_rows(), n_dofs_row));
  Assert(sparsity.n_cols() == n_dofs_column,
         ExcDimensionMismatch(sparsity.n_cols(), n_dofs_column));

  Assert(&dofh_row.get_triangulation() == &dofh_column.get_triangulation(),
         ExcMessage("This function requires both DoFHandlers to share the same triangulation"));

  std::vector<types::global_dof_index> row_dof_indices_on_this_cell;
  row_dof_indices_on_this_cell.reserve(dofh_row.get_fe_collection().max_dofs_per_cell());
  std::vector<types::global_dof_index> column_dof_indices_on_this_cell;
  column_dof_indices_on_this_cell.reserve(dofh_column.get_fe_collection().max_dofs_per_cell());

  AffineConstraints<double> constraints_dummy;

  for(const auto & cell : dofh_row.active_cell_iterators())
    if(cell->is_locally_owned())
    {
      using cell_iterator_type = typename std::decay<decltype(cell)>::type;
      const cell_iterator_type cell_column(&dofh_column.get_triangulation(),
                                           cell->level(),
                                           cell->index(),
                                           &dofh_column);

      const unsigned int n_dofs_per_cell_row = cell->get_fe().n_dofs_per_cell();
      row_dof_indices_on_this_cell.resize(n_dofs_per_cell_row);
      cell->get_dof_indices(row_dof_indices_on_this_cell);

      const unsigned int n_dofs_per_cell_column = cell_column->get_fe().n_dofs_per_cell();
      column_dof_indices_on_this_cell.resize(n_dofs_per_cell_column);
      cell_column->get_dof_indices(column_dof_indices_on_this_cell);

      /// use "empty" constraints to add entries to the sparsity pattern
      constraints_dummy.add_entries_local_to_global(row_dof_indices_on_this_cell,
                                                    column_dof_indices_on_this_cell,
                                                    sparsity,
                                                    /*keep_constrained_dofs*/ false);
    }
}

} // namespace Tools

#endif /* SPARSITY_H_ */
