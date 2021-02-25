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
                                                      /*keep_constrained_dofs*/ true);
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


template<int dim, int spacedim>
Table<2, DoFTools::Coupling>
dof_couplings_from_component_couplings(const FiniteElement<dim, spacedim> & fe_row,
                                       const FiniteElement<dim, spacedim> & fe_col,
                                       const Table<2, DoFTools::Coupling> & component_couplings)
{
  Assert(component_couplings.n_rows() == fe_row.n_components(),
         ExcDimensionMismatch(component_couplings.n_rows(), fe_row.n_components()));
  Assert(component_couplings.n_cols() == fe_col.n_components(),
         ExcDimensionMismatch(component_couplings.n_cols(), fe_col.n_components()));

  const unsigned int n_dofs_row = fe_row.n_dofs_per_cell();
  const unsigned int n_dofs_col = fe_col.n_dofs_per_cell();

  Table<2, DoFTools::Coupling> dof_couplings(n_dofs_row, n_dofs_col);

  for(unsigned int i = 0; i < n_dofs_row; ++i)
  {
    const unsigned int comp_row =
      (fe_row.is_primitive(i) ? fe_row.system_to_component_index(i).first :
                                fe_row.get_nonzero_components(i).first_selected_component());
    Assert(comp_row < fe_row.n_components(), ExcInternalError());

    for(unsigned int j = 0; j < n_dofs_col; ++j)
    {
      const unsigned int comp_col =
        (fe_col.is_primitive(j) ? fe_col.system_to_component_index(j).first :
                                  fe_col.get_nonzero_components(j).first_selected_component());
      Assert(comp_col < fe_col.n_components(), ExcInternalError());

      dof_couplings(i, j) = component_couplings(comp_row, comp_col);
    }
  }

  return dof_couplings;
}



template<int dim, int spacedim, typename SparsityPatternType>
void
make_flux_sparsity_pattern(const DoFHandler<dim, spacedim> &    dofh_row,
                           const DoFHandler<dim, spacedim> &    dofh_column,
                           SparsityPatternType &                sparsity,
                           const Table<2, DoFTools::Coupling> & int_mask,
                           const Table<2, DoFTools::Coupling> & flux_mask,
                           const types::subdomain_id            subdomain_id)
{
  /// TODO asserts as in make_sparsity_pattern()...

  using Number = double;
  const AffineConstraints<Number> constraints;
  const bool                      keep_constrained_dofs = true;

  const FiniteElement<dim, spacedim> & fe_row    = dofh_row.get_fe();
  const FiniteElement<dim, spacedim> & fe_column = dofh_column.get_fe();

  std::vector<types::global_dof_index> row_dofs_on_this_cell(fe_row.n_dofs_per_cell());
  std::vector<types::global_dof_index> row_dofs_on_other_cell(fe_row.n_dofs_per_cell());
  std::vector<types::global_dof_index> column_dofs_on_this_cell(fe_column.n_dofs_per_cell());
  std::vector<types::global_dof_index> column_dofs_on_other_cell(fe_column.n_dofs_per_cell());

  const Table<2, DoFTools::Coupling>
    int_dof_mask  = dof_couplings_from_component_couplings(fe_row, fe_column, int_mask),
    flux_dof_mask = dof_couplings_from_component_couplings(fe_row, fe_column, flux_mask);

  Assert(!int_dof_mask.empty(), ExcMessage("Initialization failed."));
  Assert(!flux_dof_mask.empty(), ExcMessage("Initialization failed."));

  Table<2, bool> support_on_face_row(fe_row.n_dofs_per_cell(), GeometryInfo<dim>::faces_per_cell);
  for(unsigned int i = 0; i < fe_row.n_dofs_per_cell(); ++i)
    for(const unsigned int f : GeometryInfo<dim>::face_indices())
      support_on_face_row(i, f) = fe_row.has_support_on_face(i, f);

  Table<2, bool> support_on_face_column(fe_column.n_dofs_per_cell(),
                                        GeometryInfo<dim>::faces_per_cell);
  for(unsigned int i = 0; i < fe_column.n_dofs_per_cell(); ++i)
    for(const unsigned int f : GeometryInfo<dim>::face_indices())
      support_on_face_column(i, f) = fe_column.has_support_on_face(i, f);

  // Convert the int_dof_mask to bool_int_dof_mask so we can pass it
  // to constraints.add_entries_local_to_global()
  Table<2, bool> boolean_int_dof_mask(int_dof_mask.n_rows(), int_dof_mask.n_cols());
  boolean_int_dof_mask.fill(false);
  for(unsigned int i = 0; i < int_dof_mask.n_rows(); ++i)
    for(unsigned int j = 0; j < int_dof_mask.n_cols(); ++j)
      if(int_dof_mask(i, j) != DoFTools::Coupling::none)
        boolean_int_dof_mask(i, j) = true;
  Assert(!boolean_int_dof_mask.empty(), ExcInternalError());
  const auto begin_boolean_int_dof_mask = &(boolean_int_dof_mask(0, 0));
  const auto end_boolean_int_dof_mask =
    begin_boolean_int_dof_mask + boolean_int_dof_mask.n_elements();
  const bool all_int_dofs_couple = std::all_of(begin_boolean_int_dof_mask,
                                               end_boolean_int_dof_mask,
                                               [&](const bool is_coupled) { return is_coupled; });

  // If all dofs couple with respect to cell integrals there is no need in
  // passing a mask to constraints.add_entries_local_to_global(). Note that
  // currently passing a non-empty dof_mask to add_entries_local_to_global() is
  // not implemented... TODO
  if(all_int_dofs_couple)
    boolean_int_dof_mask.reinit(0U, 0U);

  /// TODO treat multigrid cell iterators...
  typename DoFHandler<dim, spacedim>::active_cell_iterator cell = dofh_row.begin_active(),
                                                           endc = dofh_row.end();
  /// TODO get rid off subdomain_id...
  for(; cell != endc; ++cell)
    if(((subdomain_id == numbers::invalid_subdomain_id) ||
        (subdomain_id == cell->subdomain_id())) &&
       cell->is_locally_owned())
    {
      using cell_iterator_type = typename std::decay<decltype(cell)>::type;
      const cell_iterator_type cell_column(&dofh_column.get_triangulation(),
                                           cell->level(),
                                           cell->index(),
                                           &dofh_column);

      cell->get_dof_indices(row_dofs_on_this_cell);
      cell_column->get_dof_indices(column_dofs_on_this_cell);

      /// TODO what if we actually use constraints? QA: do AffineConstraints
      /// treat non-square matrices, in particular, is
      /// add_entries_local_to_global() the appropriate for function for our
      /// purpose here?

      /// make sparsity pattern for this cell
      constraints.add_entries_local_to_global(row_dofs_on_this_cell,
                                              column_dofs_on_this_cell,
                                              sparsity,
                                              keep_constrained_dofs,
                                              boolean_int_dof_mask);

      /// Loop over all interior neighbors
      for(const unsigned int face_n : cell->face_indices())
      {
        const auto & cell_face        = cell->face(face_n);
        const auto & cell_column_face = cell_column->face(face_n);

        Assert(!(cell->has_periodic_neighbor(face_n)), ExcMessage("Periodicity is not handled..."));
        /// remove variable periodic_neighbor
        const bool periodic_neighbor = false;

        if(cell->at_boundary(face_n) && (!periodic_neighbor))
        {
          for(unsigned int i = 0; i < fe_row.n_dofs_per_cell(); ++i)
          {
            const bool i_non_zero_i = support_on_face_row(i, face_n);
            for(unsigned int j = 0; j < fe_column.n_dofs_per_cell(); ++j)
            {
              const bool j_non_zero_i = support_on_face_column(j, face_n);

              if(flux_dof_mask(i, j) == DoFTools::Coupling::always ||
                 (flux_dof_mask(i, j) == DoFTools::Coupling::nonzero && i_non_zero_i &&
                  j_non_zero_i))
                sparsity.add(row_dofs_on_this_cell[i], column_dofs_on_this_cell[j]);
            }
          }
        }

        else
        {
          // if(!face_has_flux_coupling(cell, face_n))
          //   continue;

          // WHY do we switch here from active to level cell iterator???
          typename DoFHandler<dim, spacedim>::level_cell_iterator neighbor =
            cell->neighbor_or_periodic_neighbor(face_n);
          typename DoFHandler<dim, spacedim>::level_cell_iterator neighbor_column =
            cell_column->neighbor_or_periodic_neighbor(face_n);

          /// double-check that neighbor and neighbor_column have the same cell index and level
          AssertDimension(neighbor->index(), neighbor_column->index());
          AssertDimension(neighbor->level(), neighbor_column->level());

          /// If the cells are on the same level (and both are
          /// active, locally-owned cells) then only add to the
          /// sparsity pattern if the current cell is 'greater'
          /// in the total ordering.
          if(neighbor->level() == cell->level() && neighbor->index() > cell->index() &&
             neighbor->is_active() && neighbor->is_locally_owned())
            continue;
          /// If we are more refined then the neighbor, then we
          /// will automatically find the active neighbor cell
          /// when we call 'neighbor (face_n)' above. The
          /// opposite is not true; if the neighbor is more
          /// refined then the call 'neighbor (face_n)' will
          /// *not* return an active cell. Hence, only add things
          /// to the sparsity pattern if (when the levels are
          /// different) the neighbor is coarser than the current
          /// cell.
          ///
          /// Like above, do not use this optimization if the
          /// neighbor is not locally owned.
          if(neighbor->level() != cell->level() &&
             ((!periodic_neighbor && !cell->neighbor_is_coarser(face_n)) ||
              (periodic_neighbor && !cell->periodic_neighbor_is_coarser(face_n))) &&
             neighbor->is_locally_owned())
            continue; // (the neighbor is finer)

          const unsigned int neighbor_face_n = periodic_neighbor ?
                                                 cell->periodic_neighbor_face_no(face_n) :
                                                 cell->neighbor_face_no(face_n);
          AssertDimension(neighbor_face_n,
                          periodic_neighbor ? cell_column->periodic_neighbor_face_no(face_n) :
                                              cell_column->neighbor_face_no(face_n));

          // In 1D, go straight to the cell behind this
          // particular cell's most terminal cell. This makes us
          // skip the if (neighbor->has_children()) section
          // below. We need to do this since we otherwise
          // iterate over the children of the face, which are
          // always 0 in 1D.
          if(dim == 1)
            while(neighbor->has_children())
              neighbor = neighbor->child(face_n == 0 ? 1 : 0);

          Assert(!(neighbor->has_children()),
                 ExcMessage("Anisotropic refinement is not supported..."));
          // if(neighbor->has_children())
          // {
          //   for(unsigned int sub_nr = 0; sub_nr != cell_face->n_children(); ++sub_nr)
          //   {
          //     const typename DoFHandler<dim, spacedim>::level_cell_iterator sub_neighbor =
          //       periodic_neighbor ? cell->periodic_neighbor_child_on_subface(face_n, sub_nr) :
          //                           cell->neighbor_child_on_subface(face_n, sub_nr);

          //     sub_neighbor->get_dof_indices(dofs_on_other_cell);
          //     for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          //     {
          //       const bool i_non_zero_i = support_on_face(i, face_n);
          //       const bool i_non_zero_e = support_on_face(i, neighbor_face_n);
          //       for(unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          //       {
          //         const bool j_non_zero_i = support_on_face(j, face_n);
          //         const bool j_non_zero_e = support_on_face(j, neighbor_face_n);

          //         if(flux_dof_mask(i, j) == always)
          //         {
          //           sparsity.add(dofs_on_this_cell[i], dofs_on_other_cell[j]);
          //           sparsity.add(dofs_on_other_cell[i], dofs_on_this_cell[j]);
          //           sparsity.add(dofs_on_this_cell[i], dofs_on_this_cell[j]);
          //           sparsity.add(dofs_on_other_cell[i], dofs_on_other_cell[j]);
          //         }
          //         else if(flux_dof_mask(i, j) == nonzero)
          //         {
          //           if(i_non_zero_i && j_non_zero_e)
          //             sparsity.add(dofs_on_this_cell[i], dofs_on_other_cell[j]);
          //           if(i_non_zero_e && j_non_zero_i)
          //             sparsity.add(dofs_on_other_cell[i], dofs_on_this_cell[j]);
          //           if(i_non_zero_i && j_non_zero_i)
          //             sparsity.add(dofs_on_this_cell[i], dofs_on_this_cell[j]);
          //           if(i_non_zero_e && j_non_zero_e)
          //             sparsity.add(dofs_on_other_cell[i], dofs_on_other_cell[j]);
          //         }

          //         if(flux_dof_mask(j, i) == always)
          //         {
          //           sparsity.add(dofs_on_this_cell[j], dofs_on_other_cell[i]);
          //           sparsity.add(dofs_on_other_cell[j], dofs_on_this_cell[i]);
          //           sparsity.add(dofs_on_this_cell[j], dofs_on_this_cell[i]);
          //           sparsity.add(dofs_on_other_cell[j], dofs_on_other_cell[i]);
          //         }
          //         else if(flux_dof_mask(j, i) == nonzero)
          //         {
          //           if(j_non_zero_i && i_non_zero_e)
          //             sparsity.add(dofs_on_this_cell[j], dofs_on_other_cell[i]);
          //           if(j_non_zero_e && i_non_zero_i)
          //             sparsity.add(dofs_on_other_cell[j], dofs_on_this_cell[i]);
          //           if(j_non_zero_i && i_non_zero_i)
          //             sparsity.add(dofs_on_this_cell[j], dofs_on_this_cell[i]);
          //           if(j_non_zero_e && i_non_zero_e)
          //             sparsity.add(dofs_on_other_cell[j], dofs_on_other_cell[i]);
          //         }
          //       }
          //     }
          //   }
          // }
          // else

          {
            neighbor->get_dof_indices(row_dofs_on_other_cell);
            neighbor_column->get_dof_indices(column_dofs_on_other_cell);

            for(unsigned int i = 0; i < fe_row.n_dofs_per_cell(); ++i)
            {
              const bool i_non_zero_i = support_on_face_row(i, face_n);
              const bool i_non_zero_e = support_on_face_row(i, neighbor_face_n);
              for(unsigned int j = 0; j < fe_column.n_dofs_per_cell(); ++j)
              {
                const bool j_non_zero_i = support_on_face_column(j, face_n);
                const bool j_non_zero_e = support_on_face_column(j, neighbor_face_n);
                if(flux_dof_mask(i, j) == DoFTools::Coupling::always)
                {
                  sparsity.add(row_dofs_on_this_cell[i], column_dofs_on_other_cell[j]);
                  sparsity.add(row_dofs_on_other_cell[i], column_dofs_on_this_cell[j]);
                  sparsity.add(row_dofs_on_this_cell[i], column_dofs_on_this_cell[j]);
                  sparsity.add(row_dofs_on_other_cell[i], column_dofs_on_other_cell[j]);
                }
                if(flux_dof_mask(i, j) == DoFTools::Coupling::nonzero)
                {
                  if(i_non_zero_i && j_non_zero_e)
                    sparsity.add(row_dofs_on_this_cell[i], column_dofs_on_other_cell[j]);
                  if(i_non_zero_e && j_non_zero_i)
                    sparsity.add(row_dofs_on_other_cell[i], column_dofs_on_this_cell[j]);
                  if(i_non_zero_i && j_non_zero_i)
                    sparsity.add(row_dofs_on_this_cell[i], column_dofs_on_this_cell[j]);
                  if(i_non_zero_e && j_non_zero_e)
                    sparsity.add(row_dofs_on_other_cell[i], column_dofs_on_other_cell[j]);
                }

                /// WHY do we add sparsity entries with row and column roles exchanged???

                // if(flux_dof_mask(j, i) == always)
                // {
                //   sparsity.add(dofs_on_this_cell[j], dofs_on_other_cell[i]);
                //   sparsity.add(dofs_on_other_cell[j], dofs_on_this_cell[i]);
                //   sparsity.add(dofs_on_this_cell[j], dofs_on_this_cell[i]);
                //   sparsity.add(dofs_on_other_cell[j], dofs_on_other_cell[i]);
                // }
                // if(flux_dof_mask(j, i) == nonzero)
                // {
                //   if(j_non_zero_i && i_non_zero_e)
                //     sparsity.add(dofs_on_this_cell[j], dofs_on_other_cell[i]);
                //   if(j_non_zero_e && i_non_zero_i)
                //     sparsity.add(dofs_on_other_cell[j], dofs_on_this_cell[i]);
                //   if(j_non_zero_i && i_non_zero_i)
                //     sparsity.add(dofs_on_this_cell[j], dofs_on_this_cell[i]);
                //   if(j_non_zero_e && i_non_zero_e)
                //     sparsity.add(dofs_on_other_cell[j], dofs_on_other_cell[i]);
                // }
              }
            }
          }
        }
      }
    }
}



} // namespace Tools

#endif /* SPARSITY_H_ */
