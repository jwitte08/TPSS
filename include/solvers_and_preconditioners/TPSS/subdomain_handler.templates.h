
template<int dim, typename number>
void
SubdomainHandler<dim, number>::reinit(
  const std::shared_ptr<const dealii::MatrixFree<dim, number>> mf_storage_in,
  const AdditionalData                                         additional_data_in)
{
  clear();

  // *** submit the input (take ownership of matrix-free storage)
  owned_mf_storage = mf_storage_in;
  mf_storage       = owned_mf_storage.get();
  additional_data  = additional_data_in;

  internal_reinit();
}

template<int dim, typename number>
void
SubdomainHandler<dim, number>::reinit(const dealii::MatrixFree<dim, number> * mf_storage_in,
                                      const AdditionalData                    additional_data_in)
{
  clear();

  // *** submit the input
  mf_storage      = mf_storage_in;
  additional_data = additional_data_in;

  internal_reinit();
}

template<int dim, typename number>
void
SubdomainHandler<dim, number>::internal_reinit()
{
  using namespace dealii;

  // *** check if additional_data is reasonable
  Assert(additional_data.level != static_cast<unsigned int>(-1), ExcNotImplemented());
  Assert(additional_data.patch_variant != TPSS::PatchVariant::invalid, ExcInvalidState());
  Assert(additional_data.smoother_variant != TPSS::SmootherVariant::invalid, ExcInvalidState());

  // *** set internal data
  mf_connect.mf_storage   = mf_storage;
  dof_handler             = &(mf_storage->get_dof_handler());
  const auto & shape_info = mf_storage->get_shape_info();
  for(int d = 0; d < dim; ++d)
    quadrature_storage.emplace_back(shape_info.quadrature);

  // *** requirements: tensor product elements, ...
  Assert(mf_storage->is_supported(dof_handler->get_fe()), ExcNotImplemented());
  AssertDimension(Utilities::fixed_power<dim>(dof_handler->get_fe().tensor_degree() + 1),
                  dof_handler->get_fe().n_dofs_per_cell());

  const auto level = additional_data.level;
  AssertIndexRange(level, dof_handler->get_triangulation().n_global_levels());

  // *** gather patches (in vectorization batches) & colorize them
  typename TPSS::PatchInfo<dim>::AdditionalData patch_info_data;
  patch_info_data.patch_variant         = additional_data.patch_variant;
  patch_info_data.smoother_variant      = additional_data.smoother_variant;
  patch_info_data.level                 = additional_data.level;
  patch_info_data.coloring_func         = additional_data.coloring_func;
  patch_info_data.manual_gathering_func = additional_data.manual_gathering_func;
  patch_info_data.print_details         = additional_data.print_details;
  patch_info.initialize(dof_handler, patch_info_data);
  for(const auto & info : patch_info.time_data)
    time_data.emplace_back(info.time, info.description, info.unit);

  // *** initialize the MPI-partitioner
  const IndexSet owned_indices = std::move(dof_handler->locally_owned_mg_dofs(level));
  IndexSet       ghost_indices;
  DoFTools::extract_locally_relevant_level_dofs(*dof_handler, level, ghost_indices);
  vector_partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(owned_indices, ghost_indices, MPI_COMM_WORLD);

  // *** map the patch batches to MatrixFree's cell batches (used for the patch-local transfers)
  TPSS::PatchWorker<dim, number> patch_worker{patch_info};
  patch_worker.connect_to_matrixfree(mf_connect);

  // *** compute the surrogate patches which pertain the tensor structure
  typename TPSS::MappingInfo<dim, number>::AdditionalData mapping_info_data;
  mapping_info_data.n_q_points      = additional_data.n_q_points_surrogate;
  mapping_info_data.normalize_patch = additional_data.normalize_surrogate_patch;
  mapping_info_data.use_arc_length  = additional_data.use_arc_length;
  mapping_info.initialize_storage(patch_info, mf_connect, mapping_info_data);

  // *** check if the initialization was successful TODO
  AssertThrow(quadrature_storage.size() == dim, ExcNotImplemented()); // each direction is filled
  for(const auto & quad : quadrature_storage)
    AssertThrow(quad == quadrature_storage[0], ExcMessage("Quadrature storage is not isotropic!"));
}

template<int dim, typename number>
unsigned int
SubdomainHandler<dim, number>::guess_grain_size(const unsigned int n_subdomain_batches) const
{
  unsigned int grain_size = additional_data.grain_size;
  if(grain_size == 0)
  {
    // we would like to have enough work to do, so as first guess, try
    // to get 16 times as many chunks as we have threads on the system.
    if(additional_data.n_threads > MultithreadInfo::n_threads())
      std::cout << "WARNING: AdditionalData::n_threads = " << additional_data.n_threads
                << "is higher than MultithreadInfo::n_threads() = " << MultithreadInfo::n_threads()
                << std::endl;
    const unsigned int n_threads =
      additional_data.n_threads == 0 ? MultithreadInfo::n_threads() : additional_data.n_threads;
    grain_size = n_subdomain_batches / (n_threads * 16);
    /* if (additional_data.patch_variant == TPSS::PatchVariant::cell) */
    /*   grain_size *= 1<<dim; */

    // if there are too few degrees of freedom per cell, need to
    // increase the grain size
    //    const unsigned int minimum_parallel_grain_size = 400; //Martin:200 // J:?
    const unsigned int minimum_parallel_grain_size = 200; // Martin:200 // J:?
    const unsigned int dofs_per_cell               = dof_handler->get_fe().dofs_per_cell;
    if(dofs_per_cell * grain_size < minimum_parallel_grain_size)
      grain_size = (minimum_parallel_grain_size / dofs_per_cell + 1);
    if(dofs_per_cell * grain_size > 10000) // J:?
      grain_size /= 4;

    grain_size = 1 << (unsigned int)(std::log2(grain_size + 1));
  }
  if(grain_size > n_subdomain_batches)
    grain_size = std::max(1U, n_subdomain_batches);
  return grain_size;
}

template<int dim, typename number>
template<typename Input, typename Output>
void
SubdomainHandler<dim, number>::loop(
  const std::function<void(const SubdomainHandler<dim, number> &,
                           Output &,
                           const Input &,
                           const std::pair<unsigned int, unsigned int> &)> & patch_operation,
  Output &                                                                   output,
  const Input &                                                              input,
  const unsigned int                                                         color) const
{
  const auto & subdomain_partition_data = patch_info.subdomain_partition_data;
  const bool   contains_subdomains      = (subdomain_partition_data.n_subdomains(color) > 0);
  if(!contains_subdomains)
    return; //: nothing to do

  // TODO use explicit coloring algorithm to avoid conflicts by means of overlaps
  const bool is_AVP = (additional_data.patch_variant == TPSS::PatchVariant::vertex) &&
                      (additional_data.smoother_variant == TPSS::SmootherVariant::additive);

  const auto   range                 = subdomain_partition_data.get_patch_range(color);
  const auto & operation_on_subrange = [&](const unsigned int begin_patch,
                                           const unsigned int end_patch) {
    const std::pair<unsigned int, unsigned int> subrange{begin_patch, end_patch};
    patch_operation(*this, output, input, subrange);
  };
  const auto n_subdomain_batches = subdomain_partition_data.n_subdomains(color);
  const auto grain_size          = guess_grain_size(n_subdomain_batches);
  if(is_AVP) // serial, TODO implement a parallel scheme
    operation_on_subrange(range.first, range.second);
  else // parallel, e.g. multiplicative variant is non-conflicting due to the coloring
    parallel::apply_to_subranges(range.first, range.second, operation_on_subrange, grain_size);
}

template<int dim, typename number>
template<typename Input, typename Output>
void
SubdomainHandler<dim, number>::parloop(
  const std::function<void(const SubdomainHandler<dim, number> &,
                           Output &,
                           const Input &,
                           const std::pair<unsigned int, unsigned int> &)> & patch_operation,
  Output &                                                                   output,
  const Input &                                                              input) const
{
  using namespace dealii;

  const auto & subdomain_partition_data = patch_info.subdomain_partition_data;
  const bool   contains_subdomains      = (subdomain_partition_data.n_subdomains() > 0);
  if(!contains_subdomains)
    return; //: nothing to do

  const auto n_subdomain_batches = subdomain_partition_data.n_subdomains();
  const auto grain_size          = guess_grain_size(n_subdomain_batches);
  const auto total_range         = subdomain_partition_data.get_patch_range();
  AssertThrow(total_range.second == n_subdomain_batches, ExcMessage("Invalid patch range."));
  const auto operation_on_subrange = [&](const unsigned int begin_patch,
                                         const unsigned int end_patch) {
    const std::pair<unsigned int, unsigned int> subrange{begin_patch, end_patch};
    patch_operation(*this, output, input, subrange);
  };
  parallel::apply_to_subranges(total_range.first,
                               total_range.second,
                               operation_on_subrange,
                               grain_size);
}
