
template<int dim, typename number>
void
SubdomainHandler<dim, number>::reinit(
  const std::shared_ptr<const dealii::MatrixFree<dim, number>> mf_storage_in,
  const AdditionalData                                         additional_data_in)
{
  clear();

  // *** submit the input (take ownership of matrix-free storage)
  mf_storage_owned = mf_storage_in;
  mf_storage       = mf_storage_owned.get();
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
  // *** check if additional_data is reasonable
  Assert(additional_data.level != static_cast<unsigned int>(-1), ExcNotImplemented());
  Assert(additional_data.patch_variant != TPSS::PatchVariant::invalid, ExcInvalidState());
  Assert(additional_data.smoother_variant != TPSS::SmootherVariant::invalid, ExcInvalidState());


  // *** compress dof handlers
  {
    const auto check_dof_handler = [&](const auto dof_handler) {
      AssertThrow(mf_storage->is_supported(dof_handler->get_fe()),
                  ExcMessage("Finite element not supported."));
      AssertDimension(Utilities::fixed_power<dim>(dof_handler->get_fe().tensor_degree() + 1),
                      dof_handler->get_fe().n_dofs_per_cell());
      AssertIndexRange(additional_data.level, dof_handler->get_triangulation().n_global_levels());
      AssertDimension(dof_handler->get_fe().n_base_elements(), 1);
    };
    std::vector<const dealii::DoFHandler<dim> *> unique_dof_handlers;
    const auto                                   first_dofh_index = 0U;
    const auto & first_dofh = mf_storage->get_dof_handler(first_dofh_index);
    check_dof_handler(&first_dofh);
    unique_dof_handlers.push_back(&first_dofh);
    dofh_indices.push_back(first_dofh_index);
    for(auto dofh_index = 1U; dofh_index < mf_storage->n_components(); ++dofh_index)
    {
      const auto & dofh = mf_storage->get_dof_handler(dofh_index);
      check_dof_handler(&dofh);
      const auto         it                = std::find_if(unique_dof_handlers.begin(),
                                   unique_dof_handlers.end(),
                                   [&](const auto unique_dofh) {
                                     return dofh.get_fe() == unique_dofh->get_fe();
                                   });
      const unsigned int unique_dofh_index = std::distance(unique_dof_handlers.begin(), it);
      if(unique_dofh_index == unique_dof_handlers.size()) // dofh is unique
        unique_dof_handlers.push_back(&dofh);
      dofh_indices.push_back(unique_dofh_index);
    }
    AssertDimension(mf_storage->n_components(), dofh_indices.size());
    this->dof_handlers = unique_dof_handlers;
  }


  // *** gather patches as vectorized batches and colorize them
  typename TPSS::PatchInfo<dim>::AdditionalData patch_info_data;
  patch_info_data.patch_variant         = additional_data.patch_variant;
  patch_info_data.smoother_variant      = additional_data.smoother_variant;
  patch_info_data.level                 = additional_data.level;
  patch_info_data.coloring_func         = additional_data.coloring_func;
  patch_info_data.manual_gathering_func = additional_data.manual_gathering_func;
  patch_info_data.caching_strategy      = additional_data.caching_strategy;
  patch_info_data.print_details         = additional_data.print_details;
  patch_info.initialize(dof_handlers.front(), patch_info_data);
  for(const auto & info : patch_info.time_data)
    time_data.emplace_back(info.time, info.description, info.unit);
  // *** constructor partitions patches with respect to vectorization
  TPSS::PatchWorker<dim, number> patch_worker{patch_info};


  // *** map the patch batches to MatrixFree's cell batches
  mf_connect.initialize(mf_storage, &patch_info);


  { // *** (partially) store dof indices and patch-local dof information
    dof_infos.resize(dof_handlers.size());
    typename TPSS::DoFInfo<dim, number>::AdditionalData dof_info_data;
    dof_info_data.level = additional_data.level;
    std::set<unsigned int> initialized_indices;
    for(auto dofh_index = 0U; dofh_index < n_components(); ++dofh_index)
    {
      const auto unique_dofh_index = get_unique_dofh_index(dofh_index);
      if(!additional_data.dirichlet_ids.empty())
      {
        AssertIndexRange(unique_dofh_index, additional_data.dirichlet_ids.size());
        dof_info_data.dirichlet_ids = additional_data.dirichlet_ids.at(unique_dofh_index);
      }
      auto & dof_info                     = dof_infos[unique_dofh_index];
      const auto [dummy, not_initialized] = initialized_indices.insert(unique_dofh_index);
      (void)dummy;
      if(not_initialized)
        dof_info.initialize(dof_handlers.at(unique_dofh_index),
                            &patch_info,
                            &(get_shape_info(dofh_index)),
                            dof_info_data);
    }
  }
  { // *** initialize MPI vector partitioners
    if(TPSS::PatchVariant::cell == additional_data.patch_variant)
    {
      std::set<unsigned int> initialized_indices;
      for(auto dofh_index = 0U; dofh_index < n_components(); ++dofh_index)
      {
        const auto unique_dofh_index        = get_unique_dofh_index(dofh_index);
        auto &     dof_info                 = dof_infos[unique_dofh_index];
        const auto [dummy, not_initialized] = initialized_indices.insert(unique_dofh_index);
        (void)dummy;
        if(not_initialized)
          dof_info.vector_partitioner = mf_storage->get_vector_partitioner(dofh_index);
      }
    }
    else // entire assembly of vector partitioners
    {
      for(auto & dof_info : dof_infos)
      {
        TPSS::PatchDoFWorker<dim, number> patch_dof_worker(dof_info);
        dof_info.vector_partitioner = patch_dof_worker.initialize_vector_partitioner();
      }
    }
  }


  // *** compute the surrogate patches which pertain the tensor structure
  typename TPSS::MappingInfo<dim, number>::AdditionalData mapping_info_data;
  mapping_info_data.n_q_points      = additional_data.n_q_points_surrogate;
  mapping_info_data.normalize_patch = additional_data.normalize_surrogate_patch;
  mapping_info_data.use_arc_length  = additional_data.use_arc_length;
  mapping_info.initialize_storage(patch_info, mf_connect, mapping_info_data);


  // TODO
  // // *** if possible compress the data
  // if(additional_data.compressed)
  //   patch_info.get_internal_data()->compress();
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
    const unsigned int dofs_per_cell               = get_dof_handler(0).get_fe().dofs_per_cell;
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
