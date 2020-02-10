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



} // end namespace TPSS
