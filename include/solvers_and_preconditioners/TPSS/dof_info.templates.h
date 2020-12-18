#include "patch_dof_worker.h"

namespace TPSS
{
template<int dim, typename Number>
void
DoFInfo<dim, Number>::initialize(
  const DoFHandler<dim> *                                                   dof_handler_in,
  const PatchInfo<dim> *                                                    patch_info_in,
  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> * shape_info_in,
  const AdditionalData &                                                    additional_data_in)
{
  Assert(patch_info_in->get_internal_data()->level != numbers::invalid_unsigned_int,
         ExcMessage("Handles level cells only."));
  Assert(DoFLayout::invalid != TPSS::get_dof_layout(dof_handler_in->get_fe()),
         ExcMessage("Finite element is not supported."));
  Assert(additional_data_in.level != numbers::invalid_unsigned_int,
         ExcMessage("Implemented for level cells only"));
  Assert(!shape_info_in->data.empty(), ExcMessage("shape_info_in is empty."));

  clear();
  dof_handler     = dof_handler_in;
  patch_info      = patch_info_in;
  shape_info      = shape_info_in;
  additional_data = additional_data_in;

  //: fill the lexicographic-to-hierarchic-numbering map
  l2h = shape_info->lexicographic_numbering;

  //: fill the cell-wise number of dofs per component (assume isotropy)
  n_dofs_on_cell_per_comp.resize(shape_info_in->n_components);
  std::fill(n_dofs_on_cell_per_comp.begin(),
            n_dofs_on_cell_per_comp.end(),
            shape_info_in->dofs_per_component_on_cell);

  initialize_impl();
}


template<int dim, typename Number>
void
DoFInfo<dim, Number>::initialize_impl()
{
  /// Cache global dof indices once for each cell owned by this processor
  /// (including ghost cells)
  {
    const auto my_subdomain_id = dof_handler->get_triangulation().locally_owned_subdomain();

    /// LAMBDA checks if cell is a ghost cell on the current level
    const auto & is_ghost_on_level_impl = [](const auto & cell, const auto my_subdomain_id) {
      const bool is_owned      = cell.level_subdomain_id() == my_subdomain_id;
      const bool is_artificial = cell.level_subdomain_id() == numbers::artificial_subdomain_id;
      return !is_owned && !is_artificial;
    };

    /// LAMBDA convenience function
    const auto & is_ghost_on_level = [&, my_subdomain_id](const auto & cell) {
      return is_ghost_on_level_impl(cell, my_subdomain_id);
    };

    /// LAMBDA checks if cell has a neighbor being a ghost cell on the current level
    const auto & has_ghost_neighbor_on_level = [&](const auto & cell) {
      for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        if(cell.neighbor_level(face_no) == static_cast<int>(additional_data.level))
          if(is_ghost_on_level(*(cell.neighbor(face_no))))
            return true;
      return false;
    };

    std::vector<types::global_dof_index> dof_indices_on_ghosts;
    const IndexSet                       owned_dof_indices =
      std::move(dof_handler->locally_owned_mg_dofs(additional_data.level));
    IndexSet ghost_dof_indices(owned_dof_indices.size());

    PatchWorker<dim, Number> patch_worker(*patch_info);
    const auto               n_cells_plain = patch_info->n_cells_plain();
    const auto               n_subdomains  = patch_worker.get_partition_data().n_subdomains();
    const auto               n_cells_per_subdomain = patch_worker.n_cells_per_subdomain();

    /// First we require a mapping between the process-local @p cell_index and all
    /// associated cells within patches stored by the underlying @p patch_info. Each
    /// cell contained in the @p patch_info has a unique @p cell_position.
    std::map<int, std::vector<unsigned int>> cell_index_to_cell_position;
    for(auto patch = 0U; patch < n_subdomains; ++patch)
    {
      const auto n_lanes_filled = patch_worker.n_lanes_filled(patch);
      for(auto cell_no = 0U; cell_no < n_cells_per_subdomain; ++cell_no)
        for(auto lane = 0U; lane < n_lanes_filled; ++lane)
        {
          const auto cell_position = patch_worker.get_cell_position(patch, cell_no, lane);
          const auto [cell_level, cell_index] = patch_info->get_cell_level_and_index(cell_position);
          (void)cell_level;
          AssertDimension(cell_level, additional_data.level);
          cell_index_to_cell_position[cell_index].push_back(cell_position);
        }
    }

    /// Cache global dof indices in @p global_dof_indices_cellwise (later
    /// compressed to @p dof_indices_cellwise) cell by
    /// cell. For any cell identified by its @p cell_position we store the
    /// access points to the range of global dof indices in @p
    /// global_dof_indices_cellwise via @p
    /// start_and_number_of_dof_indices_cellwise.
    std::vector<types::global_dof_index> global_dof_indices_cellwise;
    start_and_number_of_dof_indices_cellwise.resize(n_cells_plain);
    for(const auto & [cell_index, cell_positions] : cell_index_to_cell_position)
    {
      const auto & cell = get_level_dof_accessor_impl(cell_index, additional_data.level);
      const auto   level_dof_indices = fill_level_dof_indices_impl(cell);

      //: set dof start and quantity
      const auto dof_start = global_dof_indices_cellwise.size();
      const auto n_dofs    = level_dof_indices.size(); // compress?
      for(auto cell_position : cell_positions)
        start_and_number_of_dof_indices_cellwise[cell_position] = std::make_pair(dof_start, n_dofs);

      //: submit global dof indices
      std::copy(level_dof_indices.cbegin(),
                level_dof_indices.cend(),
                std::back_inserter(global_dof_indices_cellwise));

      //: update index set of ghost dof indices
      if(is_ghost_on_level(cell) || has_ghost_neighbor_on_level(cell))
        for(const auto dof_index : level_dof_indices)
          if(!owned_dof_indices.is_element(dof_index))
            dof_indices_on_ghosts.push_back(dof_index);
    }

    /// First, sort and compress duplicates of ghosted global dof indices. Then,
    /// fill ghost index set.
    std::sort(dof_indices_on_ghosts.begin(), dof_indices_on_ghosts.end());
    const auto end_without_duplicates =
      std::unique(dof_indices_on_ghosts.begin(), dof_indices_on_ghosts.end());
    ghost_dof_indices.add_indices(dof_indices_on_ghosts.begin(), end_without_duplicates);
    ghost_dof_indices.compress();

    /// Initialize vector partitioner based on locally owned and ghosted dof indices.
    const auto partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(owned_dof_indices, MPI_COMM_WORLD);
    partitioner->set_ghost_indices(ghost_dof_indices);
    this->vector_partitioner   = partitioner;
    const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1); // !!!
    if(!is_mpi_parallel)
      AssertDimension(dof_handler->n_dofs(additional_data.level), vector_partitioner->size());

    /// Convert global dof indices into process local dof indices
    dof_indices_cellwise.clear();
    std::transform(global_dof_indices_cellwise.cbegin(),
                   global_dof_indices_cellwise.cend(),
                   std::back_inserter(dof_indices_cellwise),
                   [&](const auto global_dof_index) {
                     return vector_partitioner->global_to_local(global_dof_index);
                   });
  }

  /// Completely cache all global dof indices for each macro patch.
  if(additional_data.caching_strategy == TPSS::CachingStrategy::Cached)
  {
    /// At this point, we are able to use a reduced but sufficient set of
    /// PatchDoFWorker's functionality to cache the global dof indices
    /// patch-wise (to be precise, we use the process local dof index to save
    /// memory).
    PatchDoFWorker<dim, Number> patch_dof_worker(*this);
    const auto &                partition_data = patch_dof_worker.get_partition_data();
    const auto                  n_subdomains   = partition_data.n_subdomains();

    start_of_dof_indices_patchwise.clear();
    dof_indices_patchwise.clear();
    for(auto patch_id = 0U; patch_id < n_subdomains; ++patch_id)
    {
      start_of_dof_indices_patchwise.emplace_back(dof_indices_patchwise.size());
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_id); ++lane)
      {
        const auto & dof_indices_on_patch =
          patch_dof_worker.fill_dof_indices_on_patch(patch_id, lane);
        std::copy(dof_indices_on_patch.cbegin(),
                  dof_indices_on_patch.cend(),
                  std::back_inserter(dof_indices_patchwise));
      }
    }
    start_of_dof_indices_patchwise.emplace_back(dof_indices_patchwise.size());
  }

  if(additional_data.compute_ras_boolean_weights)
    compute_restricted_dofs_impl();
  /// TODO !!!
  // compress();
}



template<int dim, typename Number>
void
DoFInfo<dim, Number>::compute_restricted_dofs_impl()
{
  const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
  AssertThrow(!is_mpi_parallel, ExcMessage("MPI is not supported."));

  const auto n_components = shape_info->n_components;
  AssertThrow(patch_info, ExcMessage("patch_info isn't set."));
  AssertThrow(shape_info, ExcMessage("shape_info isn't set."));
  AssertThrow(dof_handler, ExcMessage("dof_handler isn't set."));
  if(!patch_info->empty())
    AssertThrow(!dof_indices_patchwise.empty(),
                ExcMessage("dof indices have to be cached patchwise."));

  AssertThrow(n_components == 1U, ExcMessage("TODO more than scalar elements"));
  const auto patch_variant    = patch_info->get_additional_data().patch_variant;
  const auto smoother_variant = patch_info->get_additional_data().smoother_variant;
  AssertThrow(patch_variant == TPSS::PatchVariant::vertex, ExcMessage("TODO"));
  AssertThrow(smoother_variant == TPSS::SmootherVariant::additive,
              ExcMessage("Additive scheme is only supported."));

  PatchDoFWorker<dim, Number> patch_dof_worker(*this);
  const auto &                partition_data = patch_dof_worker.get_partition_data();
  const auto                  n_subdomains   = partition_data.n_subdomains();

  std::array<unsigned int, dim> inner_sizes;
  for(auto d = 0U; d < dim; ++d)
    inner_sizes[d] = patch_dof_worker.n_dofs_1d(d);
  // const auto & patch_dof_tensor = patch_dof_worker.get_dof_tensor();
  // auto         inner_sizes      = patch_dof_tensor.size();
  // for(auto d = 0U; d < dim; ++d)
  //   inner_sizes[d] -= 2U;
  Tensors::TensorHelper<dim> dof_tensor_inner(inner_sizes);

  const auto comp_distance_impl_q = [&](const unsigned int index) {
    const auto & mindex = dof_tensor_inner.multi_index(index);
    AssertDimension(dof_tensor_inner.n_flat() % 2, 1); // odd
    const auto   index_at_vertex = dof_tensor_inner.n_flat() / 2;
    const auto & mindex_mid      = dof_tensor_inner.multi_index(index_at_vertex);
    unsigned int dist            = 0U;
    for(auto d = 0U; d < dim; ++d)
      dist = std::max(dist,
                      (mindex[d] < mindex_mid[d]) ? (mindex_mid[d] - mindex[d]) :
                                                    (mindex[d] - mindex_mid[d]));
    return dist;
  };

  const auto comp_distance_impl_dgq = [&](const unsigned int index) {
    std::array<double, dim> mindex_mid;
    for(auto d = 0U; d < dim; ++d)
      mindex_mid[d] = patch_dof_worker.n_dofs_1d(d) / 2. - 0.5;
    const auto & mindex = dof_tensor_inner.multi_index(index);
    double       dist   = 0.;
    for(auto d = 0U; d < dim; ++d)
    {
      const double diff = mindex_mid[d] - static_cast<double>(mindex[d]);
      dist              = std::max<double>(dist, std::fabs(diff));
    }
    return static_cast<unsigned int>(dist);
  };

  const auto comp_distance = [&](const unsigned int index) {
    if(get_dof_layout() == DoFLayout::Q)
      return comp_distance_impl_q(index);
    else if(get_dof_layout() == DoFLayout::DGQ)
      return comp_distance_impl_dgq(index);
    else
      AssertThrow(false, ExcMessage("Dof layout isn't supported"));
  };

  std::map<unsigned int, std::vector<std::array<unsigned int, 5>>> dof_indices_to_weights;
  unsigned int                                                     dof_index_max = 0U;
  const unsigned int                                               comp          = 0; // TODO
  for(auto patch_id = 0U; patch_id < n_subdomains; ++patch_id)
  {
    for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_id); ++lane)
    {
      const auto dof_indices_on_patch =
        patch_dof_worker.get_dof_indices_on_patch(patch_id, lane, comp);

      const auto comp_weight = [&](const unsigned int index) -> std::array<unsigned int, 5> {
        AssertDimension(dof_tensor_inner.n_flat(), dof_indices_on_patch.size());
        return {comp_distance(index), index, patch_id, lane, comp};
      };

      for(auto i = 0U; i < dof_indices_on_patch.size(); ++i)
      {
        const auto dof_index = dof_indices_on_patch[i];
        dof_indices_to_weights[dof_index].emplace_back(comp_weight(i));
        dof_index_max = std::max(dof_index, dof_index_max);
      }
    }
  }

  const auto get_dof_position_patchwise = [&](const unsigned int patch_id,
                                              const unsigned int lane,
                                              const unsigned int component,
                                              const unsigned int dof_no) {
    const auto [dof_start, n_dofs_at_comp] =
      patch_dof_worker.get_dof_start_and_quantity_on_patch(patch_id, lane, component);
    return dof_start + dof_no;
  };

  restricted_dof_flags_patchwise.clear();
  restricted_dof_flags_patchwise.resize(dof_indices_patchwise.size(), true);

  for(auto & [dof_index, weights] : dof_indices_to_weights)
  {
    Assert(!weights.empty(), ExcMessage("No weights stored."));
    std::sort(weights.begin(), weights.end());

    const auto & [ignore, dof_no, patch_id, lane, comp] = weights.front();
    (void)ignore;
    const auto dof_position = get_dof_position_patchwise(patch_id, lane, comp, dof_no);
    AssertDimension(dof_indices_patchwise.at(dof_position), dof_index);
    restricted_dof_flags_patchwise[dof_position] = false;
  }

  // TODO in case of MPI we have to communicate at this point or earlier ...
}



} // end namespace TPSS
