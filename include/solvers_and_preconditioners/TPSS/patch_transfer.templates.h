namespace TPSS
{
template<int dim, typename Number, int fe_degree>
template<typename VectorType>
AlignedVector<VectorizedArray<Number>>
PatchTransfer<dim, Number, fe_degree>::gather(const VectorType & src) const
{
  AssertIndexRange(patch_id, subdomain_handler.get_partition_data().n_subdomains());
  AssertDimension(src.size(), subdomain_handler.get_dof_handler().n_dofs(level));

  AlignedVector<VectorizedArray<Number>> dst(n_dofs_per_patch());

  if(!compressed)
  {
    std::vector<ArrayView<const CellIterator>> cell_collection =
      patch_worker.get_cell_collection_views(patch_id);
    const unsigned int n_lanes_filled = cell_collection.size();
    Assert(n_lanes_filled > 0, ExcMessage("No vectorization lane filled."));
    for(unsigned int lane = 0; lane < n_lanes_filled; ++lane)
    {
      const auto &       cell_view = cell_collection[lane];
      const unsigned int n_cells   = cell_view.size();
      Assert(n_cells > 0, ExcMessage("No cell contained in collection."));
      for(unsigned int cell_no = 0; cell_no < n_cells; ++cell_no)
      {
        const ArrayView<const unsigned> &    patch_dofs = patch_dofs_on_cell(cell_no);
        const CellIterator &                 cell       = cell_view[cell_no];
        std::vector<types::global_dof_index> global_dofs_on_cell;
        global_dofs_on_cell.resize(n_dofs_per_cell);
        cell->get_active_or_mg_dof_indices(global_dofs_on_cell);
        std::vector<Number> global_values;
        global_values.resize(n_dofs_per_cell);
        empty_constraints.get_dof_values(src,
                                         global_dofs_on_cell.cbegin(),
                                         global_values.begin(),
                                         global_values.end());
        auto dof = patch_dofs.cbegin();
        for(auto value = global_values.cbegin(); value != global_values.cend(); ++value, ++dof)
          dst[*dof][lane] = *value;
      }
    }
    //: fill the unused lanes with meaningful data to avoid divison by zero in LAC solvers
    for(unsigned int lane = n_lanes_filled; lane < VectorizedArray<Number>::n_array_elements;
        ++lane)
      for(auto & elem : dst)
        elem[lane] = elem[0];
  }
  else // compressed
  {
    AssertDimension(dst.size(), patch_to_global_indices.size());
    auto global_dof = patch_to_global_indices.cbegin();
    for(auto dst_value = dst.begin(); dst_value != dst.end(); ++global_dof, ++dst_value)
      for(unsigned int lane = 0; lane < macro_size; ++lane)
        (*dst_value)[lane] = src((*global_dof)[lane]);
  }

  AssertDimension(dst.size(), n_dofs_per_patch());
  return dst;
}

template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::gather_add(const ArrayView<VectorizedArray<Number>> dst,
                                                  const VectorType & src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto & src_local = gather(src);
  std::transform(dst.begin(),
                 dst.end(),
                 src_local.begin(),
                 dst.begin(),
                 [](const auto & dst, const auto & src) { return dst + src; });
}

template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::gather_add(AlignedVector<VectorizedArray<Number>> & dst,
                                                  const VectorType & src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto dst_view = make_array_view<VectorizedArray<Number>>(dst.begin(), dst.end());
  gather_add(dst_view, src);
}

template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::scatter_add(
  VectorType &                                   dst,
  const ArrayView<const VectorizedArray<Number>> src) const
{
  AssertIndexRange(patch_id, subdomain_handler.get_partition_data().n_subdomains());
  AssertDimension(dst.size(), subdomain_handler.get_dof_handler().n_dofs(level));
  AssertDimension(src.size(), n_dofs_per_patch());

  if(!compressed)
  {
    std::vector<ArrayView<const CellIterator>> cell_collection =
      patch_worker.get_cell_collection_views(patch_id);
    const unsigned int n_lanes_filled = cell_collection.size();
    Assert(n_lanes_filled > 0, ExcMessage("No vectorization lane filled."));
    std::vector<Number> src_per_cell;
    src_per_cell.resize(n_dofs_per_cell);
    for(unsigned int lane = 0; lane < n_lanes_filled; ++lane)
    {
      const auto &       cell_view = cell_collection[lane];
      const unsigned int n_cells   = cell_view.size();
      Assert(n_cells > 0, ExcMessage("No cell contained in collection."));
      for(unsigned int cell_no = 0; cell_no < n_cells; ++cell_no)
      {
        const ArrayView<const unsigned> &    patch_dofs = patch_dofs_on_cell(cell_no);
        const CellIterator &                 cell       = cell_view[cell_no];
        std::vector<types::global_dof_index> global_dofs_on_cell;
        global_dofs_on_cell.resize(n_dofs_per_cell);
        cell->get_active_or_mg_dof_indices(global_dofs_on_cell);
        const unsigned * dof = patch_dofs.begin();
        for(auto out = src_per_cell.begin(); out != src_per_cell.end(); ++out, ++dof)
          *out = src[*dof][lane];
        empty_constraints.distribute_local_to_global(src_per_cell, global_dofs_on_cell, dst);
      }
    }
  }
  else // compressed
  {
    AssertDimension(src.size(), patch_to_global_indices.size());
    auto global_dof = patch_to_global_indices.cbegin();
    for(auto src_value = src.cbegin(); src_value != src.cend(); ++global_dof, ++src_value)
      for(unsigned int lane = 0; lane < patch_worker.n_lanes_filled(patch_id); ++lane)
        dst((*global_dof)[lane]) += (*src_value)[lane];
  }
}

template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::scatter_add(
  VectorType &                                   dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  const auto src_view = make_array_view<const VectorizedArray<Number>>(src.begin(), src.end());
  scatter_add(dst, src_view);
}

} // end namespace TPSS
