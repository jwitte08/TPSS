#ifndef DOF_INFO_H
#define DOF_INFO_H

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_tools.h>

#include "TPSS.h"
#include "generic_functionalities.h"
#include "patch_info.h"
#include "patch_worker.h"
#include "tensors.h"

#include <array>
#include <memory>



namespace TPSS
{
/**
 * Helps with the (one-dimensional) patch local dof indexing depending on the
 * underlying finite element. For example, for Q-like finite elements we have
 * to treat the dofs at cell boundaries belonging to more than one cell.
 */
template<int n_dimensions>
class PatchLocalIndexHelper
{
public:
  PatchLocalIndexHelper(const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
                        const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
                        const DoFLayout                             dof_layout_in);

  /**
   * Returns the one-dimensional patch local dof index with cell local dof index
   * @p cell_dof_index within local cell @p cell_no in spatial dimension @p
   * dimension. Local cells and cell local dof indices are subject to
   * lexicographical ordering.
   *
   * For DGQ-like finite elements this mapping is bijective. For Q-like finite
   * elements this mapping is surjective but not injective.
   */
  unsigned int
  dof_index_1d(const unsigned int cell_no,
               const unsigned int cell_dof_index,
               const int          dimension) const;

  const Tensors::TensorHelper<n_dimensions> * cell_dof_tensor;
  const Tensors::TensorHelper<n_dimensions> * cell_tensor;
  const DoFLayout                             dof_layout;

private:
  /**
   * Implementation of @p dof_index_1d for Q-like finite elements.
   */
  unsigned int
  dof_index_1d_q_impl(const unsigned int cell_no,
                      const unsigned int cell_dof_index,
                      const int          dimension) const;

  /**
   * Implementation of @p dof_index_1d for DGQ-like finite elements.
   */
  unsigned int
  dof_index_1d_dgq_impl(const unsigned int cell_no,
                        const unsigned int cell_dof_index,
                        const int          dimension) const;
};



/**
 * Helps with the d-dimensional patch local dof indexing depending on the finite element.
 */
template<int n_dimensions>
class PatchLocalTensorHelper : public PatchLocalIndexHelper<n_dimensions>,
                               public Tensors::TensorHelper<n_dimensions>
{
public:
  using IndexHelperBase  = PatchLocalIndexHelper<n_dimensions>;
  using TensorHelperBase = Tensors::TensorHelper<n_dimensions>;

  PatchLocalTensorHelper(const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
                         const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
                         const DoFLayout                             dof_layout_in);

  /**
   * Returns the patch local dof index as multi-index subject to lexicographical
   * ordering. For more details see PatchLocalIndexHelper::dof_index_1d.
   */
  std::array<unsigned int, n_dimensions>
  dof_multi_index(const unsigned int cell_no, const unsigned int cell_dof_index) const;

  /**
   * Returns the patch local dof index as univariate index subject to
   * lexicographical ordering. For more details see
   * PatchLocalIndexHelper::dof_index_1d.
   */
  unsigned int
  dof_index(const unsigned int cell_no, const unsigned int cell_dof_index) const;
};



template<int dim>
struct DoFInfo
{
  struct AdditionalData;

  void
  initialize(const DoFHandler<dim> * dof_handler,
             const PatchInfo<dim> *  patch_info,
             const AdditionalData &  additional_data = AdditionalData{});

  void
  clear()
  {
    patch_info = nullptr;
    dof_indices.clear();
    dof_handler     = nullptr;
    additional_data = AdditionalData{};
    l2h.clear();
  }

  const AdditionalData &
  get_additional_data() const
  {
    return additional_data;
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

  std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner;

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

  AdditionalData additional_data;

  std::vector<types::global_dof_index> l2h;
};



template<int dim>
struct DoFInfo<dim>::AdditionalData
{
  unsigned int level = numbers::invalid_unsigned_int;
};



template<int dim, typename number>
class PatchDoFWorker : public PatchWorker<dim, number>
{
public:
  using patch_worker_type                  = PatchWorker<dim, number>;
  static constexpr unsigned int macro_size = patch_worker_type::macro_size;

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

  const DoFInfo<dim> &
  get_dof_info() const;

  std::shared_ptr<const Utilities::MPI::Partitioner>
  initialize_vector_partitioner() const;

private:
  const DoFInfo<dim> * const dof_info;

  PatchLocalTensorHelper<dim> patch_dof_tensor;
};



// -----------------------------   PatchLocalIndexHelper   ----------------------------

template<int n_dimensions>
inline PatchLocalIndexHelper<n_dimensions>::PatchLocalIndexHelper(
  const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
  const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
  const DoFLayout                             dof_layout_in)
  : cell_dof_tensor(&cell_dof_tensor_in), cell_tensor(&cell_tensor_in), dof_layout(dof_layout_in)
{
}


template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index_1d(const unsigned int cell_no,
                                                  const unsigned int cell_dof_index,
                                                  const int          dimension) const
{
  if(dof_layout == DoFLayout::DGQ)
    return dof_index_1d_dgq_impl(cell_no, cell_dof_index, dimension);
  else if(dof_layout == DoFLayout::Q)
    return dof_index_1d_q_impl(cell_no, cell_dof_index, dimension);
  AssertThrow(false, ExcMessage("Finite element not supported."));
  return numbers::invalid_unsigned_int;
}

template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index_1d_q_impl(const unsigned int cell_no,
                                                         const unsigned int cell_dof_index,
                                                         const int          dimension) const
{
  AssertIndexRange(cell_no, cell_tensor->n[dimension]);
  AssertIndexRange(cell_dof_index, cell_dof_tensor->n[dimension]);
  AssertIndexRange(dimension, n_dimensions);
  const auto & n_dofs_per_cell_1d = cell_dof_tensor->n[dimension];
  return cell_no * n_dofs_per_cell_1d + cell_dof_index - cell_no;
}


template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index_1d_dgq_impl(const unsigned int cell_no,
                                                           const unsigned int cell_dof_index,
                                                           const int          dimension) const
{
  AssertIndexRange(cell_no, cell_tensor->n[dimension]);
  AssertIndexRange(cell_dof_index, cell_dof_tensor->n[dimension]);
  AssertIndexRange(dimension, n_dimensions);
  const auto & n_dofs_per_cell_1d = cell_dof_tensor->n[dimension];
  return cell_no * n_dofs_per_cell_1d + cell_dof_index;
}



// -----------------------------   PatchLocalTensorHelper   ----------------------------



template<int n_dimensions>
inline PatchLocalTensorHelper<n_dimensions>::PatchLocalTensorHelper(
  const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
  const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
  const DoFLayout                             dof_layout_in)
  : IndexHelperBase(cell_tensor_in, cell_dof_tensor_in, dof_layout_in), TensorHelperBase([&]() {
      std::array<unsigned int, n_dimensions> sizes;
      for(auto d = 0U; d < n_dimensions; ++d)
        sizes[d] =
          IndexHelperBase::dof_index_1d(cell_tensor_in.n[d] - 1, cell_dof_tensor_in.n[d] - 1, d) +
          1;
      return sizes;
    }())
{
}


template<int n_dimensions>
inline std::array<unsigned int, n_dimensions>

PatchLocalTensorHelper<n_dimensions>::dof_multi_index(const unsigned int cell_no,
                                                      const unsigned int cell_dof_index) const
{
  const auto & cell_no_multi        = IndexHelperBase::cell_tensor->multi_index(cell_no);
  const auto & cell_dof_index_multi = IndexHelperBase::cell_dof_tensor->multi_index(cell_dof_index);
  std::array<unsigned int, n_dimensions> patch_dof_index_multi;
  for(auto d = 0U; d < n_dimensions; ++d)
    patch_dof_index_multi[d] =
      IndexHelperBase::dof_index_1d(cell_no_multi[d], cell_dof_index_multi[d], d);
  return patch_dof_index_multi;
}


template<int n_dimensions>
inline unsigned int
PatchLocalTensorHelper<n_dimensions>::dof_index(const unsigned int cell_no,
                                                const unsigned int cell_dof_index) const
{
  const auto & patch_dof_index_multi = dof_multi_index(cell_no, cell_dof_index);
  return TensorHelperBase::uni_index(patch_dof_index_multi);
}



// --------------------------------   PatchDoFWorker   --------------------------------



template<int dim, typename number>
inline PatchDoFWorker<dim, number>::PatchDoFWorker(const DoFInfo<dim> & dof_info_in)
  : PatchWorker<dim, number>(*(dof_info_in.patch_info)),
    dof_info(&dof_info_in),
    patch_dof_tensor(this->patch_size,
                     dof_info_in.dof_handler->get_fe().tensor_degree(),
                     dof_info_in.get_dof_layout())
{
  Assert(dof_info->get_additional_data().level != numbers::invalid_unsigned_int,
         ExcMessage("Implemented for level cells only."));
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


template<int dim, typename number>
inline const DoFInfo<dim> &
PatchDoFWorker<dim, number>::get_dof_info() const
{
  return *dof_info;
}



} // end namespace TPSS

#include "dof_info.templates.h"

#endif // end inclusion guard
