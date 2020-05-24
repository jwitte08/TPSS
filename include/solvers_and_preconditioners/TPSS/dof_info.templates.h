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


// template<int dim, typename Number>
// void
// DoFInfo<dim, Number>::initialize(
//   const DoFHandler<dim> * dof_handler_in,
//   const PatchInfo<dim> *  patch_info_in,
//   const std::array<internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> *, dim> &
//                          shape_infos_in,
//   const AdditionalData & additional_data_in)
// {
//   Assert(patch_info_in->get_internal_data()->level != numbers::invalid_unsigned_int,
//          ExcMessage("Handles level cells only."));
//   Assert(DoFLayout::invalid != TPSS::get_dof_layout(dof_handler_in->get_fe()),
//          ExcMessage("Finite element is not supported."));
//   Assert(additional_data_in.level != numbers::invalid_unsigned_int,
//          ExcMessage("Implemented for level cells only"));

//   clear();
//   dof_handler = dof_handler_in;
//   patch_info  = patch_info_in;
//   std::copy(shape_infos_in.cbegin(), shape_infos_in.cend(), shape_infos.begin());
//   additional_data = additional_data_in;

//   initialize_impl();
// }


template<int dim, typename Number>
void
DoFInfo<dim, Number>::initialize_impl()
{
  /// cache global dof indices once for each cell owned by this processor
  /// (including ghost cells)
  {
    /// LAMBDA checks if cell is ghost on current level
    const auto   my_subdomain_id   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto & is_ghost_on_level = [my_subdomain_id](const auto & cell) {
      const bool is_owned      = cell.level_subdomain_id() == my_subdomain_id;
      const bool is_artificial = cell.level_subdomain_id() == numbers::artificial_subdomain_id;
      return !is_owned && !is_artificial;
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
    /// associated cells within patches stored by the underlying patch_info. Each
    /// cell contained in the patch_info has a unique @p cell_position.
    std::vector<std::vector<unsigned int>> cell_index_to_cell_position;
    cell_index_to_cell_position.resize(n_cells_plain);
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
          if(static_cast<unsigned>(cell_index) >= cell_index_to_cell_position.size())
            cell_index_to_cell_position.resize(cell_index + 1);
          cell_index_to_cell_position[cell_index].emplace_back(cell_position);
        }
    }

    /// Cache the global dof indices in @p global_dof_indices_cellwise. Given
    /// the @p cell_position we access the associated dof indices via @p
    /// start_and_number_of_dof_indices_cellwise.
    global_dof_indices_cellwise.clear();
    start_and_number_of_dof_indices_cellwise.resize(n_cells_plain);
    for(auto cell_index = 0U; cell_index < cell_index_to_cell_position.size(); ++cell_index)
      if(!cell_index_to_cell_position[cell_index].empty())
      {
        const auto & cell = get_level_dof_accessor_impl(cell_index, additional_data.level);
        const auto   level_dof_indices = fill_level_dof_indices_impl(cell);

        //: set dof start and quantity
        const auto   dof_start      = global_dof_indices_cellwise.size();
        const auto   n_dofs         = level_dof_indices.size(); // compress?
        const auto & cell_positions = cell_index_to_cell_position[cell_index];
        for(auto cell_position : cell_positions)
          start_and_number_of_dof_indices_cellwise[cell_position] =
            std::make_pair(dof_start, n_dofs);

        //: submit global dof indices
        std::copy(level_dof_indices.cbegin(),
                  level_dof_indices.cend(),
                  std::back_inserter(global_dof_indices_cellwise));

        //: update index set of ghost dof indices
        if(is_ghost_on_level(cell))
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
    this->vector_partitioner = partitioner;

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

  // compress(); // TODO !!!
}



} // end namespace TPSS
