namespace TPSS
{
template<int dim, typename number>
typename PatchWorker<dim, number>::cell_iterator_type
PatchWorker<dim, number>::get_cell_iterator(const unsigned int patch_id,
                                            const unsigned int cell_no,
                                            const unsigned int lane) const
{
  Assert(patch_info, ExcMessage("patch_info is empty."));

  const auto cell_position            = get_cell_position(patch_id, cell_no, lane);
  const auto [cell_level, cell_index] = patch_info->get_cell_level_and_index(cell_position);
  const auto & tria                   = patch_info->get_triangulation();
  return cell_iterator_type(&tria, cell_level, cell_index, nullptr);
}



template<int dim, typename number>
template<typename CellIteratorType, typename MakeCellIterator>
std::vector<CellIteratorType>
PatchWorker<dim, number>::make_cell_collection_impl(const MakeCellIterator & make_cell_iterator,
                                                    const unsigned int       patch_index,
                                                    const unsigned int       lane) const
{
  AssertIndexRange(lane, n_lanes_filled(patch_index));
  AssertIndexRange(patch_index, get_partition_data().n_subdomains());
  std::vector<CellIteratorType> collection;
  const auto                    patch_size = n_cells_per_subdomain();
  for(auto cell_no = 0U; cell_no < patch_size; ++cell_no)
    collection.emplace_back(make_cell_iterator(patch_index, cell_no, lane));
  return collection;
}



template<int dim, typename number>
template<typename CellIteratorType, typename MakeCellIterator>
std::vector<std::array<CellIteratorType, PatchWorker<dim, number>::macro_size>>
PatchWorker<dim, number>::make_cell_collection_impl(const MakeCellIterator & make_cell_iterator,
                                                    const unsigned int       patch_index) const
{
  std::vector<std::array<CellIteratorType, macro_size>> collection(n_cells_per_subdomain());
  for(auto lane = 0U; lane < macro_size; ++lane)
  {
    const auto   actual_lane = lane < this->n_lanes_filled(patch_index) ? lane : 0U;
    const auto & collection_per_lane =
      make_cell_collection_impl<CellIteratorType, MakeCellIterator>(make_cell_iterator,
                                                                    patch_index,
                                                                    actual_lane);
    AssertDimension(collection.size(), collection_per_lane.size());
    auto cell_per_lane = collection_per_lane.begin();
    for(auto & macro_cell : collection)
      macro_cell[lane] = *(cell_per_lane++);
    Assert(cell_per_lane == collection_per_lane.end(), ExcMessage("..."));
  }
  return collection;
}



template<int dim, typename number>
inline std::vector<typename PatchWorker<dim, number>::cell_iterator_type>
PatchWorker<dim, number>::get_cell_collection(const unsigned int patch_index,
                                              const unsigned int lane) const
{
  return make_cell_collection_impl<cell_iterator_type>(
    [&](const auto p, const auto c, const auto l) { return get_cell_iterator(p, c, l); },
    patch_index,
    lane);
}



template<int dim, typename number>
std::vector<std::array<typename PatchWorker<dim, number>::cell_iterator_type,
                       PatchWorker<dim, number>::macro_size>>
PatchWorker<dim, number>::get_cell_collection(const unsigned int patch_index) const
{
  return make_cell_collection_impl<cell_iterator_type>(
    [&](const auto p, const auto c, const auto l) { return get_cell_iterator(p, c, l); },
    patch_index);
}



template<int dim, typename number>
void
PatchWorker<dim, number>::partition_patches(PatchInfo<dim> & info)
{
  // Initialize the partitioning of macro patches.
  compute_partition_data(info.subdomain_partition_data,
                         info.get_internal_data(),
                         &(info.patch_starts));
}

} // end namespace TPSS
