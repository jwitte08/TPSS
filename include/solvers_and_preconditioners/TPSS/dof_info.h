#ifndef DOF_INFO_H
#define DOF_INFO_H

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_tools.h>

#include "TPSS.h"
#include "generic_functionalities.h"
#include "patch_info.h"
#include "patch_worker.h"

#include <array>
#include <memory>

namespace TPSS
{
template<int dim>
struct DoFInfo
{
  void
  initialize(const DoFHandler<dim> * dof_handler_in, const PatchInfo<dim> * patch_info_in);

  void
  clear()
  {
    patch_info = nullptr;
    dof_indices.clear();
    dof_handler = nullptr;
    l2h.clear();
  }

  DoFLayout
  get_dof_layout() const
  {
    Assert(dof_handler, ExcMessage("DoF handler not initialized."));
    return TPSS::get_dof_layout(dof_handler->get_fe());
  }

  const DoFHandler<dim> * dof_handler = nullptr;

  /*
   * Array storing for each cell in @p cell_iterators the first dof index.
   */
  std::vector<std::vector<types::global_dof_index>> dof_indices;

  const PatchInfo<dim> * patch_info = nullptr;

  DoFAccessor<dim, DoFHandler<dim>, true>
  get_level_dof_accessor(const unsigned int cell_position) const
  {
    Assert(patch_info, ExcMessage("Patch info not initialized."));
    const auto [cell_level, cell_index] = patch_info->get_cell_level_and_index(cell_position);
    const auto & tria                   = dof_handler->get_triangulation();
    return DoFAccessor<dim, DoFHandler<dim>, true>(&tria, cell_level, cell_index, dof_handler);
  }

  std::vector<types::global_dof_index>
  get_level_dof_indices(const unsigned int cell_position) const
  {
    const auto                           n_dofs_per_cell = dof_handler->get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> level_dof_indices(n_dofs_per_cell);
    const auto &                         cell = get_level_dof_accessor(cell_position);
    cell.get_mg_dof_indices(cell.level(), level_dof_indices);

    /// reorder level dof indices lexicographically
    if(DoFLayout::Q == get_dof_layout())
    {
      AssertDimension(level_dof_indices.size(), l2h.size());
      std::vector<types::global_dof_index> level_dof_indices_lxco;
      std::transform(l2h.cbegin(),
                     l2h.cend(),
                     std::back_inserter(level_dof_indices_lxco),
                     [&](const auto & h) { return level_dof_indices[h]; });
      return level_dof_indices_lxco;
    }

    return level_dof_indices;
  }

  std::vector<types::global_dof_index> l2h;
};



template<int dim, typename number>
class PatchDoFWorker : public PatchWorker<dim, number>
{
public:
  using patch_worker                       = PatchWorker<dim, number>;
  static constexpr unsigned int macro_size = patch_worker::macro_size;

  PatchDoFWorker() = delete;

  PatchDoFWorker(const DoFInfo<dim> & dof_info_in);

  PatchDoFWorker(const PatchDoFWorker &) = delete;

  PatchDoFWorker &
  operator=(const PatchDoFWorker &) = delete;

  ArrayView<const types::global_dof_index>
  get_dof_indices_on_cell(const unsigned int patch_id,
                          const unsigned int cell_no,
                          const unsigned int lane) const;

  std::array<ArrayView<const types::global_dof_index>, macro_size>
  get_dof_indices_on_cell(const unsigned int patch_id, const unsigned int cell_no) const;

private:
  const DoFInfo<dim> * const dof_info;
};



// --------------------------------   PatchDoFWorker   --------------------------------



template<int dim, typename number>
inline PatchDoFWorker<dim, number>::PatchDoFWorker(const DoFInfo<dim> & dof_info_in)
  : PatchWorker<dim, number>(*(dof_info_in.patch_info)), dof_info(&dof_info_in)
{
}

template<int dim, typename number>
inline ArrayView<const types::global_dof_index>
PatchDoFWorker<dim, number>::get_dof_indices_on_cell(const unsigned int patch_id,
                                                     const unsigned int cell_no,
                                                     const unsigned int lane) const
{
  const unsigned int n_lanes_filled = this->n_lanes_filled(patch_id);
  const unsigned int position       = [&]() {
    AssertIndexRange(lane, this->macro_size);
    if(lane < n_lanes_filled)
      return this->get_cell_position(patch_id, cell_no, lane);
    else
      return this->get_cell_position(patch_id, cell_no, 0);
  }();
  const auto &                             dof_indices = dof_info->dof_indices;
  ArrayView<const types::global_dof_index> view;
  const auto &                             dof_indices_on_cell = dof_indices[position];
  view.reinit(dof_indices_on_cell.data(), dof_indices_on_cell.size());
  return view;
}


template<int dim, typename number>
inline std::array<ArrayView<const types::global_dof_index>, PatchDoFWorker<dim, number>::macro_size>
PatchDoFWorker<dim, number>::get_dof_indices_on_cell(const unsigned int patch_id,
                                                     const unsigned int cell_no) const
{
  std::array<ArrayView<const types::global_dof_index>, macro_size> views;
  for(auto lane = 0U; lane < macro_size; ++lane)
  {
    const auto & view = get_dof_indices_on_cell(patch_id, cell_no, lane);
    views[lane].reinit(view.data(), view.size());
  }
  return views;
}



} // end namespace TPSS

#include "dof_info.templates.h"

#endif // end inclusion guard
