namespace TPSS
{
// !!! TODO what about BlockVectors?
template<int dim, typename Number>
void
PatchDoFWorker<dim, Number>::initialize_dof_vector(
  LinearAlgebra::distributed::Vector<Number> & vec) const
{
  Assert(dof_info, ExcMessage("dof_info is not set."));
  Assert(dof_info->vector_partitioner, ExcMessage("vector_partitioner is not set."));
  vec.reinit(dof_info->vector_partitioner);
}



template<int dim, typename Number>
void
PatchDoFWorker<dim, Number>::initialize_dof_vector(Vector<Number> & vec) const
{
  vec.reinit(n_global_dofs());
}



template<int dim, typename Number>
typename PatchDoFWorker<dim, Number>::cell_iterator_type
PatchDoFWorker<dim, Number>::get_cell_iterator(const unsigned int patch_index,
                                               const unsigned int cell_no,
                                               const unsigned int lane) const
{
  Assert(this->patch_info, ExcMessage("patch_info is empty."));
  Assert(dof_info, ExcMessage("dof_info is empty."));
  Assert(dof_info->dof_handler, ExcMessage("dof_handler is empty."));
  const auto cell_position            = this->get_cell_position(patch_index, cell_no, lane);
  const auto [cell_level, cell_index] = this->patch_info->get_cell_level_and_index(cell_position);
  const auto & tria                   = this->patch_info->get_triangulation();
  return cell_iterator_type(&tria, cell_level, cell_index, dof_info->dof_handler);
}



template<int dim, typename Number>
std::vector<typename PatchDoFWorker<dim, Number>::cell_iterator_type>
PatchDoFWorker<dim, Number>::get_cell_collection(const unsigned int patch_index,
                                                 const unsigned int lane) const
{
  return Base::template make_cell_collection_impl<cell_iterator_type>(
    [&](const auto p, const auto c, const auto l) { return get_cell_iterator(p, c, l); },
    patch_index,
    lane);
}



template<int dim, typename Number>
std::vector<std::array<typename PatchDoFWorker<dim, Number>::cell_iterator_type,
                       PatchDoFWorker<dim, Number>::macro_size>>
PatchDoFWorker<dim, Number>::get_cell_collection(const unsigned int patch_index) const
{
  return Base::template make_cell_collection_impl<cell_iterator_type>(
    [&](const auto p, const auto c, const auto l) { return get_cell_iterator(p, c, l); },
    patch_index);
}


// template<int dim, typename number>
// std::shared_ptr<const Utilities::MPI::Partitioner>
// PatchDoFWorker<dim, number>::initialize_vector_partitioner() const
// {
//   const auto     level            = dof_info->get_additional_data().level;
//   const auto     caching_strategy = dof_info->get_additional_data().caching_strategy;
//   const IndexSet owned_indices    =
//   std::move(dof_info->dof_handler->locally_owned_mg_dofs(level));

//   // Note: For certain meshes (in particular in 3D and with many
//   // processors), it is really necessary to cache intermediate data. After
//   // trying several objects such as std::set, a vector that is always kept
//   // sorted, and a vector that is initially unsorted and sorted once at the
//   // end, the latter has been identified to provide the best performance.
//   // Martin Kronbichler
//   const unsigned int                   n_colors = this->get_partition_data().n_colors();
//   std::vector<types::global_dof_index> dofs_on_ghosts;
//   const auto                           insert_ghost_indices = [&](auto && dof_indices) {
//     for(const auto dof_index : dof_indices)
//       if(!owned_indices.is_element(dof_index))
//         dofs_on_ghosts.push_back(dof_index);
//   };
//   for(unsigned int color = 0; color < n_colors; ++color)
//   {
//     const auto range = this->get_ghost_range(color);
//     for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
//       for(auto lane = 0U; lane < this->n_lanes_filled(patch_id); ++lane)
//       {
//         if(caching_strategy != TPSS::CachingStrategy::Cached)
//           insert_ghost_indices(fill_dof_indices_on_patch(patch_id, lane));
//         else
//           insert_ghost_indices(get_dof_indices_on_patch(patch_id, lane));
//       }
//   }

//   /// sort, compress out duplicates, fill into index set
//   std::sort(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
//   IndexSet   ghost_indices(owned_indices.size());
//   const auto end_without_duplicates = std::unique(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
//   ghost_indices.add_indices(dofs_on_ghosts.begin(), end_without_duplicates);
//   ghost_indices.compress();

//   const auto partitioner =
//     std::make_shared<Utilities::MPI::Partitioner>(owned_indices, MPI_COMM_WORLD);

//   // /// ALTERNATIVE LARGER
//   // IndexSet larger_ghost_indices(owned_indices.size());
//   // DoFTools::extract_locally_relevant_level_dofs(*(dof_info->dof_handler),
//   //                                               level,
//   //                                               larger_ghost_indices);
//   // larger_ghost_indices.subtract_set(owned_indices);
//   // partitioner->set_ghost_indices(ghost_indices, larger_ghost_indices);

//   /// ALTERNATIVE SMALL
//   partitioner->set_ghost_indices(ghost_indices);

//   return partitioner;
// }

} // end namespace TPSS
