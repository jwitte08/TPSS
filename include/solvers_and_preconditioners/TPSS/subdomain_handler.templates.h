
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

  // *** initialize internal data
  mf_connect.mf_storage = mf_storage;
  for(auto i = 0U; i < mf_storage->n_components(); ++i)
  {
    const auto   dof_handler    = &(mf_storage->get_dof_handler(i));
    const auto & finite_element = dof_handler->get_fe();
    Assert(mf_storage->is_supported(finite_element), ExcMessage("Finite element not supported."));
    AssertDimension(Utilities::fixed_power<dim>(finite_element.tensor_degree() + 1),
                    finite_element.n_dofs_per_cell());
    AssertIndexRange(additional_data.level, dof_handler->get_triangulation().n_global_levels());
    AssertDimension(finite_element.n_base_elements(), 1);
    dof_handlers.push_back(dof_handler);
  }
  {
    std::reverse(dof_handlers.begin(), dof_handlers.end());
    std::vector<const dealii::DoFHandler<dim> *> unique_dof_handlers;
    unique_dof_handlers.push_back(dof_handlers.back());
    dof_handlers.pop_back();
    dofh_indices.push_back(0);
    for(const auto dofh : dof_handlers)
    {
      const auto         it         = std::find_if(unique_dof_handlers.begin(),
                                   unique_dof_handlers.end(),
                                   [&](const auto unique_dofh) {
                                     return dofh->get_fe() == unique_dofh->get_fe();
                                   });
      const unsigned int dofh_index = std::distance(unique_dof_handlers.begin(), it);
      if(dofh_index < unique_dof_handlers.size()) // dofh is not unique
      {
        dofh_indices.push_back(dofh_index);
      }
      else // dofh is unique
      {
        dofh_indices.push_back(dofh_index);
        unique_dof_handlers.push_back(dofh);
      }
    }
    AssertDimension(dof_handlers.size() + 1, dofh_indices.size());
    dof_handlers = unique_dof_handlers;
  }

  // std::cout << "n_dof_handlers: " << dofh_indices.size() << std::endl;
  // std::cout << "n_dof_handlers (unique): " << dof_handlers.size() << std::endl;

  // TODO
  const auto & shape_info = mf_storage->get_shape_info();
  for(int d = 0; d < dim; ++d)
    quadrature_storage.emplace_back(shape_info.quadrature);

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

  // *** (partially) store dofs
  dof_infos.resize(dof_handlers.size());
  for(auto i = 0U; i < dof_infos.size(); ++i)
    dof_infos.at(i).initialize(dof_handlers.at(i), &patch_info);

  // *** map the patch batches to MatrixFree's cell batches
  mf_connect.initialize(mf_storage, &patch_info);

  // *** initialize the MPI-partitioner
  vector_partitioners.resize(n_components());
  vector_partitioners[0] = initialize_vector_partitioner(patch_worker);
  // TODO different dof handlers !?
  for(unsigned int dofh_index = 1; dofh_index < n_components(); ++dofh_index)
    vector_partitioners[dofh_index] = vector_partitioners[0];
  AssertThrow(vector_partitioners.size() == n_components(),
              ExcMessage("Mismatching number of partitioners."));

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

  // TODO
  // // *** if possible compress the data
  // if(additional_data.compressed)
  //   patch_info.get_internal_data()->compress();
}

template<int dim, typename number>
std::shared_ptr<const Utilities::MPI::Partitioner>
SubdomainHandler<dim, number>::initialize_vector_partitioner(
  const TPSS::PatchWorker<dim, number> & patch_worker) const
{
  const auto &                                       additional_data = get_additional_data();
  const unsigned                                     level           = additional_data.level;
  const auto                                         patch_variant = additional_data.patch_variant;
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
  if(TPSS::PatchVariant::cell == patch_variant)
    partitioner = get_matrix_free().get_vector_partitioner();
  else
  {
    const IndexSet owned_indices = std::move(dof_handlers.front()->locally_owned_mg_dofs(level));

    // Note: For certain meshes (in particular in 3D and with many
    // processors), it is really necessary to cache intermediate data. After
    // trying several objects such as std::set, a vector that is always kept
    // sorted, and a vector that is initially unsorted and sorted once at the
    // end, the latter has been identified to provide the best performance.
    // Martin Kronbichler
    const auto   my_subdomain_id   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto & is_ghost_on_level = [my_subdomain_id](const auto & cell) {
      const bool is_owned      = cell->level_subdomain_id() == my_subdomain_id;
      const bool is_artificial = cell->level_subdomain_id() == numbers::artificial_subdomain_id;
      return !is_owned && !is_artificial;
    };
    std::vector<types::global_dof_index> dof_indices;
    std::vector<types::global_dof_index> dofs_on_ghosts;
    const unsigned int                   n_colors = patch_worker.get_partition_data().n_colors();
    for(unsigned int color = 0; color < n_colors; ++color)
    {
      const auto range = patch_worker.get_ghost_range(color);
      for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
      {
        const auto & collection_views = patch_worker.get_cell_collection_views(patch_id);
        for(const auto & collection : collection_views)
          for(const auto & cell : collection)
            if(is_ghost_on_level(cell))
            {
              dof_indices.resize(cell->get_fe().dofs_per_cell);
              cell->get_mg_dof_indices(dof_indices);
              for(const auto dof_index : dof_indices)
                if(!owned_indices.is_element(dof_index))
                  dofs_on_ghosts.push_back(dof_index);
            }
      }
    }

    // sort, compress out duplicates, fill into index set
    std::sort(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    IndexSet   ghost_indices(owned_indices.size());
    const auto end_without_duplicates = std::unique(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    ghost_indices.add_indices(dofs_on_ghosts.begin(), end_without_duplicates);
    ghost_indices.compress();
    IndexSet larger_ghost_indices(owned_indices.size());
    DoFTools::extract_locally_relevant_level_dofs(get_dof_handler(0), level, larger_ghost_indices);
    larger_ghost_indices.subtract_set(owned_indices);

    const auto tmp_partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(owned_indices, MPI_COMM_WORLD);
    tmp_partitioner->set_ghost_indices(ghost_indices, larger_ghost_indices);
    // tmp_partitioner->set_ghost_indices(larger_ghost_indices);
    partitioner = tmp_partitioner;

    // // DEBUG
    // std::ostringstream oss;
    // oss << "info on mpi proc " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " on level
    // " << level << std::endl; oss << "ghost index set:" << std::endl; for (const auto i :
    // ghost_indices)
    //   oss << i << " ";
    // oss << std::endl;
    // oss << "larger ghost index set:" << std::endl;
    // for (const auto i : larger_ghost_indices)
    //   oss << i << " ";
    // oss << std::endl;
    // const auto import_targets = partitioner->import_targets();
    // oss << "import targets: (proc, n_targets)" << std::endl;
    // for (const auto p : import_targets)
    //   oss << "(" << p.first << ", " << p.second << ") ";
    // oss << std::endl;
    // const auto ghost_targets = partitioner->ghost_targets();
    // oss << "ghost targets: (proc, n_targets)" << std::endl;
    // for (const auto p : ghost_targets)
    //   oss << "(" << p.first << ", " << p.second << ") ";
    // oss << std::endl;
    // std::cout << oss.str() << std::endl;
  }
  return partitioner;
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
