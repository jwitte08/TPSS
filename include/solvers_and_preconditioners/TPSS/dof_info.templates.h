namespace TPSS
{
template<int dim>
void
DoFInfo<dim>::initialize(const DoFHandler<dim> * dof_handler_in,
                         const PatchInfo<dim> *  patch_info_in)
{
  Assert(patch_info_in->get_internal_data()->level != numbers::invalid_unsigned_int,
         ExcMessage("Handles level cells only."));
  Assert(DoFLayout::invalid != TPSS::get_dof_layout(dof_handler_in->get_fe()),
         ExcMessage("Finite element is not supported."));

  clear();
  dof_handler = dof_handler_in;
  patch_info  = patch_info_in;

  /// fill the lexicographic-to-hierarchic-numbering map
  if(DoFLayout::Q == get_dof_layout())
    l2h = FETools::lexicographic_to_hierarchic_numbering(dof_handler->get_fe());

  /// store global dof indices
  const auto n_cells = patch_info->n_cells_plain();
  dof_indices.reserve(n_cells);
  for(auto i = 0U; i < n_cells; ++i)
    dof_indices.emplace_back(get_level_dof_indices(i));
  dof_indices.shrink_to_fit();
}



} // end namespace TPSS
