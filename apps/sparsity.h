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
template<bool is_multigrid, int dim, int spacedim, typename SparsityPatternType>
void
make_sparsity_pattern_impl(const DoFHandler<dim, spacedim> & dofh_row,
                           const DoFHandler<dim, spacedim> & dofh_column,
                           SparsityPatternType &             sparsity,
                           const unsigned int                level = numbers::invalid_unsigned_int)
{
  const types::global_dof_index n_dofs_row =
    is_multigrid ? dofh_row.n_dofs(level) : dofh_row.n_dofs();
  const types::global_dof_index n_dofs_column =
    is_multigrid ? dofh_column.n_dofs(level) : dofh_column.n_dofs();
  (void)n_dofs_row, (void)n_dofs_column;

  Assert(sparsity.n_rows() == n_dofs_row, ExcDimensionMismatch(sparsity.n_rows(), n_dofs_row));
  Assert(sparsity.n_cols() == n_dofs_column,
         ExcDimensionMismatch(sparsity.n_cols(), n_dofs_column));

  Assert(&dofh_row.get_triangulation() == &dofh_column.get_triangulation(),
         ExcMessage("This function requires both DoFHandlers to share the same triangulation"));

  if(level != numbers::invalid_unsigned_int)
    AssertIndexRange(level, dofh_row.get_triangulation().n_global_levels());

  std::vector<types::global_dof_index> row_dof_indices_on_this_cell;
  row_dof_indices_on_this_cell.reserve(dofh_row.get_fe_collection().max_dofs_per_cell());
  std::vector<types::global_dof_index> column_dof_indices_on_this_cell;
  column_dof_indices_on_this_cell.reserve(dofh_column.get_fe_collection().max_dofs_per_cell());

  AffineConstraints<double> constraints_dummy;

  /// is_locally_owned() is only implemented for active cell iterators, thus, we
  /// need a common interface for mg and active cell iteratos
  const auto & is_locally_owned = [&](const auto & cell) -> bool {
    if constexpr(is_multigrid)
    {
      const bool is_serial_tria =
        dofh_row.get_triangulation().locally_owned_subdomain() == numbers::invalid_subdomain_id;
      const bool is_locally_owned =
        cell->level_subdomain_id() == dofh_row.get_triangulation().locally_owned_subdomain();
      return is_serial_tria || is_locally_owned;
    }
    else
      return cell->is_locally_owned();
  };

  const auto & add_entries_foreach_cell = [&](const auto & range_of_cells_row) {
    for(const auto & cell : range_of_cells_row)
      if(is_locally_owned(cell))
      {
        using cell_iterator_type = typename std::decay<decltype(cell)>::type;
        const cell_iterator_type cell_column(&dofh_column.get_triangulation(),
                                             cell->level(),
                                             cell->index(),
                                             &dofh_column);

        const unsigned int n_dofs_per_cell_row = cell->get_fe().n_dofs_per_cell();
        row_dof_indices_on_this_cell.resize(n_dofs_per_cell_row);
        cell->get_active_or_mg_dof_indices(row_dof_indices_on_this_cell);

        const unsigned int n_dofs_per_cell_column = cell_column->get_fe().n_dofs_per_cell();
        column_dof_indices_on_this_cell.resize(n_dofs_per_cell_column);
        cell_column->get_active_or_mg_dof_indices(column_dof_indices_on_this_cell);

        /// use "empty" constraints to add entries to the sparsity pattern
        constraints_dummy.add_entries_local_to_global(row_dof_indices_on_this_cell,
                                                      column_dof_indices_on_this_cell,
                                                      sparsity,
                                                      /*keep_constrained_dofs*/ false);
      }
  };

  if(is_multigrid)
    add_entries_foreach_cell(dofh_row.mg_cell_iterators_on_level(level));
  else
    add_entries_foreach_cell(dofh_row.active_cell_iterators());
}



template<int dim, int spacedim, typename SparsityPatternType>
void
make_sparsity_pattern(const DoFHandler<dim, spacedim> & dofh_row,
                      const DoFHandler<dim, spacedim> & dofh_column,
                      SparsityPatternType &             sparsity)
{
  make_sparsity_pattern_impl<false>(dofh_row, dofh_column, sparsity);
}



template<int dim, int spacedim, typename SparsityPatternType>
void
make_sparsity_pattern(const DoFHandler<dim, spacedim> & dofh_row,
                      const DoFHandler<dim, spacedim> & dofh_column,
                      SparsityPatternType &             sparsity,
                      const unsigned int                level)
{
  make_sparsity_pattern_impl<true>(dofh_row, dofh_column, sparsity, level);
}

} // namespace Tools

#endif /* SPARSITY_H_ */
