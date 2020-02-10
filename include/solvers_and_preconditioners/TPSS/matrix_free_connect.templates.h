namespace TPSS
{
template<int dim, typename Number>
void
MatrixFreeConnect<dim, Number>::initialize(const MatrixFree<dim, Number> * mf_storage_in,
                                           const TPSS::PatchInfo<dim> *    patch_info_in)
{
  Assert(patch_info_in, ExcMessage("patch_info_in not initialized."));
  Assert(mf_storage_in, ExcMessage("mf_storage_in not initialized."));

  clear();

  this->mf_storage = mf_storage_in;
  this->patch_info = patch_info_in;

  const unsigned int n_cell_batches          = mf_storage->n_cell_batches();
  const auto &       internal_data           = *(patch_info->get_internal_data());
  const unsigned     level                   = internal_data.level;
  const auto &       tria                    = patch_info->get_triangulation();
  const bool         proc_has_cells_on_level = (level < tria.n_levels());
  //: N of locally stored cell iterators (including ghost and artificial)
  const unsigned int n_cells_stored = proc_has_cells_on_level ? tria.n_cells(level) : 0;

  /**
   * Each process enumerates its owned, ghosted & artificial cells. The cell
   * index is accessible via TriaIterator::index(level) on each level. The
   * maximal cell index is bounded by the Number of cells,
   * Triangulation::n_cells(level).
   *
   * In the MatrixFree framework cells are stored as batches due to
   * vectorization. In the following, we map cells identified by its cell index
   * to their counterpart in the MatrixFree object identified by the pair of
   * batch index (bindex) and vectorization lane (bcomp).
   */
  std::vector<std::pair<unsigned int, unsigned int>> cindex_to_bindex_bcomp_pair;
  cindex_to_bindex_bcomp_pair.resize(n_cells_stored); // we don't care about the accurate size
  for(unsigned int bid = 0; bid < n_cell_batches; ++bid)
    for(unsigned int comp = 0; comp < macro_size; ++comp)
      if(comp < mf_storage->n_components_filled(bid))
      {
        const auto         cell   = mf_storage->get_cell_iterator(bid, comp, /*dofh_index*/ 0);
        const unsigned int cindex = cell->index();
        AssertIndexRange(cindex, n_cells_stored);
        cindex_to_bindex_bcomp_pair[cindex] = std::make_pair(bid, comp);
      }

  /**
   * For each cell iterator we store the associated macro cell (batch) and the
   * vectorization lane representing the same cell in the MatrixFree framework.
   */
  const auto n_cells_patch_info = patch_info->n_cells_plain();
  auto &     bindex_and_bcomp   = batch_and_lane_index_pairs;
  bindex_and_bcomp.clear();
  bindex_and_bcomp.reserve(n_cells_patch_info);
  for(auto i = 0U; i < n_cells_patch_info; ++i)
  {
    const auto [cell_index, dummy] = patch_info->get_cell_level_and_index(i);
    (void)dummy;
    bindex_and_bcomp.emplace_back(cindex_to_bindex_bcomp_pair[cell_index]);
  }
  AssertDimension(bindex_and_bcomp.size(), patch_info->n_cells_plain());
}



} // end namespace TPSS
