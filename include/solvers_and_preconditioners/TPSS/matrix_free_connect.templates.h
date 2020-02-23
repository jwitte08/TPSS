namespace TPSS
{
template<int dim, typename Number>
void
MatrixFreeConnect<dim, Number>::initialize(
  const MatrixFree<dim, Number> *                     mf_storage_in,
  const ArrayView<const TPSS::DoFInfo<dim, Number>> & dof_infos_in)
{
  Assert(dof_infos_in.size() != 0, ExcMessage("No dof_info is passed."));
  const auto patch_info_0 = dof_infos_in[0].patch_info;
  for(const auto dof_info : dof_infos_in)
    Assert(patch_info_0 == dof_info.patch_info, ExcMessage("PatchInfos do not coincide."));
  Assert(mf_storage_in, ExcMessage("mf_storage_in not initialized."));

  clear();

  this->mf_storage = mf_storage_in;
  this->dof_infos.reinit(dof_infos_in.data(), dof_infos_in.size());
  this->patch_info = patch_info_0;

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

  dof_infos_local.resize(dof_infos.size());
  for(auto dofh_index = 0; dofh_index < dof_infos.size(); ++dofh_index)
  {
    const auto & dof_info       = dof_infos[dofh_index];
    auto &       dof_info_local = dof_infos_local[dofh_index];

    TPSS::PatchDoFWorker<dim, Number> patch_dof_worker(dof_info);
    const auto             fe_order = dof_info.dof_handler->get_fe().tensor_degree() + 1;
    PatchLocalHelperQ<dim> local_helper(fe_order);
    const auto             n_subdomains = patch_dof_worker.get_partition_data().n_subdomains();
    for(auto patch_id = 0U; patch_id < n_subdomains; ++patch_id)
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_id); ++lane)
      {
        local_helper.reinit(patch_dof_worker.fill_face_infos(patch_id, lane));
        std::map<types::global_dof_index, unsigned int> global_to_patch_dof_index;
        for(auto cell_no = 0U; local_helper.n_cells(); ++cell_no)
        {
          const auto   n_dofs_per_cell_unique = local_helper.n_dofs_per_cell_unique(cell_no);
          const auto & global_dof_indices =
            patch_dof_worker.get_dof_indices_on_cell(patch_id, cell_no, lane);
          for(auto cell_dof_no = 0U; cell_dof_no < n_dofs_per_cell_unique; ++cell_dof_no)
          {
            const unsigned int patch_dof_index = local_helper.dof_index(cell_no, cell_dof_no);
            const auto         cell_dof_index  = local_helper.cell_dof_index(patch_dof_index);
            const types::global_dof_index global_dof_index      = global_dof_indices[patch_dof_index];
            const auto [it, is_inserted] =
              global_to_patch_dof_index.emplace(global_dof_index, patch_dof_index);
	    (void)it;
            Assert(
              is_inserted,
              ExcMessage("Tried to insert a global dof index again. Check if the patch_dof_index is really unique. Presently the code is only tested for regular patches of cells, that is a patch with tensor product structure. On irregular patches the PatchLocalHelper might not consider all possible cases of neighboring cells."));
          }
        }
      }
  }
}



} // end namespace TPSS
