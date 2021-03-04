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

} // end namespace TPSS
