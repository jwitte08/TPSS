namespace TPSS
{
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
