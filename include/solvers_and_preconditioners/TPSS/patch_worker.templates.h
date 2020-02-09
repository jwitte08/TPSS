namespace TPSS
{
template<int dim, typename number>
void
PatchWorker<dim, number>::partition_patches(PatchInfo<dim> & info)
{
  /*
   * Initialize the partitioning into subdomain batches (vectorization).
   */
  compute_partition_data(info.subdomain_partition_data,
                         info.get_internal_data(),
                         &(info.patch_starts));

  /*
   * Initialize the boundary identifiers for each face of the subdomain.
   *
   * TODO treat each macro_cells at the patch boundary instead of one representative?
   */
  const auto get_mask =
    [](auto && macro_cell, const unsigned int direction, const unsigned int face_no_1d) {
      std::bitset<macro_size> bitset_mask;
      const auto              face_no = 2 * direction + face_no_1d;
      for(unsigned int vv = 0; vv < macro_size; ++vv)
        bitset_mask[vv] = macro_cell[vv]->face(face_no)->at_boundary();
      return bitset_mask;
    };

  info.at_boundary_mask.clear();
  info.at_boundary_mask.reserve(get_partition_data().n_subdomains() *
                                GeometryInfo<dim>::faces_per_cell);
  for(unsigned int color = 0; color < get_partition_data().n_colors(); ++color)
  {
    const auto patch_range = get_partition_data().get_patch_range(color);
    for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
    {
      //: face_no < direction
      std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> masks;
      const auto cell_collection{std::move(get_cell_collection(patch_id))};
      for(unsigned int d = 0; d < dim; ++d)
      {
        masks[d * 2]     = get_mask(cell_collection.front(), d, /*face_no*/ 0);
        masks[d * 2 + 1] = get_mask(cell_collection.back(), d, /*face_no*/ 1);
      }
      for(const auto & mask : masks)
        info.at_boundary_mask.emplace_back(static_cast<unsigned int>(mask.to_ulong()));
    }
  }
  AssertDimension(info.at_boundary_mask.size(),
                  get_partition_data().n_subdomains() * GeometryInfo<dim>::faces_per_cell);
}


template<int dim, typename number>
void
PatchWorker<dim, number>::connect_to_matrixfree(MatrixFreeConnect<dim, number> & mf_connect)
{
  Assert(patch_info != nullptr, ExcNotInitialized());

  const auto &       mf_storage              = *(mf_connect.mf_storage);
  const unsigned int n_cell_batches          = mf_storage.n_cell_batches();
  const auto &       internal_data           = *(patch_info->get_internal_data());
  const unsigned     level                   = internal_data.level;
  const auto &       dof_handler             = mf_storage.get_dof_handler();
  const auto &       tria                    = dof_handler.get_triangulation();
  const bool         proc_has_cells_on_level = (level < tria.n_levels());
  //: N of locally stored cell iterators (including ghost and artificial)
  const unsigned int n_cells_stored = proc_has_cells_on_level ? tria.n_cells(level) : 0;

  /**
   * Each process enumerates its owned, ghosted & artificial cells. The cell index is accessible via
   * TriaIterator::index(level) on each level. The maximal cell index is bounded by the number of
   * cells, Triangulation::n_cells(level).
   *
   * In the MatrixFree framework cells are stored as batches due to vectorization. In the following,
   * we map cells identified by its cell index to their counterpart in the MatrixFree object
   * identified by the pair of batch index (bindex) and vectorization lane (bcomp).
   */
  std::vector<std::pair<unsigned int, unsigned int>> cindex_to_bindex_bcomp_pair;
  cindex_to_bindex_bcomp_pair.resize(n_cells_stored); // we don't care about the accurate size
  for(unsigned int bid = 0; bid < n_cell_batches; ++bid)
    for(unsigned int comp = 0; comp < macro_size; ++comp)
      if(comp < mf_storage.n_components_filled(bid))
      {
        const auto         cell = mf_storage.get_cell_iterator(bid, comp, /*dof_handler_index*/ 0);
        const unsigned int cindex = cell->index();
        AssertIndexRange(cindex, n_cells_stored);
        cindex_to_bindex_bcomp_pair[cindex] = std::make_pair(bid, comp);
      }

  /**
   * For each cell iterator we store the associated macro cell (batch) and the vectorization lane
   * representing the same cell in the MatrixFree framework.
   */
  const auto n_cells_patch_info = patch_info->n_cells_plain();
  auto &     bindex_and_bcomp   = mf_connect.batch_and_lane;
  bindex_and_bcomp.clear();
  bindex_and_bcomp.reserve(n_cells_patch_info);
  for(auto i = 0U; i < n_cells_patch_info; ++i)
  {
    const auto [cindex, dummy] = patch_info->get_cell_level_and_index(i);
    (void)dummy;
    bindex_and_bcomp.emplace_back(cindex_to_bindex_bcomp_pair[cindex]);
  }
  AssertDimension(bindex_and_bcomp.size(), patch_info->n_cells_plain());
}
} // end namespace TPSS
