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

  std::pair<unsigned int, unsigned int>
  dof_range_1d(const unsigned int cell_no, const unsigned int dimension) const;

  unsigned int
  n_cells_1d(const unsigned int dimension) const;

  unsigned int
  n_dofs_per_cell_1d(const unsigned int dimension) const;

  const Tensors::TensorHelper<n_dimensions> cell_dof_tensor;
  const Tensors::TensorHelper<n_dimensions> cell_tensor;
  const DoFLayout                           dof_layout;

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

  unsigned int
  n_dofs_1d(const unsigned int dimension) const;
};



template<int dim, typename Number>
struct DoFInfo
{
  struct AdditionalData;

  /// assuming isotropy of tensor product finite elements and quadrature
  void
  initialize(const DoFHandler<dim> *                                                   dof_handler,
             const PatchInfo<dim> *                                                    patch_info,
             const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> * shape_info,
             const AdditionalData & additional_data = AdditionalData{});

  void
  initialize(const DoFHandler<dim> * dof_handler,
             const PatchInfo<dim> *  patch_info,
             const std::array<internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> *,
                              dim> & shape_infos,
             const AdditionalData &  additional_data = AdditionalData{});

  void
  initialize_impl();

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

  std::array<const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> *, dim>
    shape_infos;

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



template<int dim, typename Number>
struct DoFInfo<dim, Number>::AdditionalData
{
  unsigned int level = numbers::invalid_unsigned_int;
};



template<int dim, typename Number>
class PatchDoFWorker : public PatchWorker<dim, Number>
{
public:
  using patch_worker_type                  = PatchWorker<dim, Number>;
  static constexpr unsigned int macro_size = patch_worker_type::macro_size;

  PatchDoFWorker() = delete;

  PatchDoFWorker(const DoFInfo<dim, Number> & dof_info_in);

  PatchDoFWorker(const PatchDoFWorker &) = delete;

  PatchDoFWorker &
  operator=(const PatchDoFWorker &) = delete;

  ArrayView<const types::global_dof_index>
  get_dof_indices_on_cell(const unsigned int patch_id,
                          const unsigned int cell_no,
                          const unsigned int lane) const;

  std::array<ArrayView<const types::global_dof_index>, macro_size>
  get_dof_indices_on_cell(const unsigned int patch_id, const unsigned int cell_no) const;

  const DoFInfo<dim, Number> &
  get_dof_info() const;

  const PatchLocalTensorHelper<dim> &
  get_dof_tensor() const;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
  get_shape_info(const unsigned int dimension) const;

  std::shared_ptr<const Utilities::MPI::Partitioner>
  initialize_vector_partitioner() const;

private:
  const DoFInfo<dim, Number> * const dof_info;

  PatchLocalTensorHelper<dim> patch_dof_tensor;
};



// -----------------------------   PatchLocalIndexHelper   ----------------------------

template<int n_dimensions>
inline PatchLocalIndexHelper<n_dimensions>::PatchLocalIndexHelper(
  const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
  const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
  const DoFLayout                             dof_layout_in)
  : cell_dof_tensor(cell_dof_tensor_in), cell_tensor(cell_tensor_in), dof_layout(dof_layout_in)
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
  AssertIndexRange(cell_no, cell_tensor.n[dimension]);
  AssertIndexRange(cell_dof_index, cell_dof_tensor.n[dimension]);
  AssertIndexRange(dimension, n_dimensions);
  const auto & n_dofs_per_cell_1d = cell_dof_tensor.n[dimension];
  return cell_no * n_dofs_per_cell_1d + cell_dof_index - cell_no;
}


template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index_1d_dgq_impl(const unsigned int cell_no,
                                                           const unsigned int cell_dof_index,
                                                           const int          dimension) const
{
  AssertIndexRange(cell_no, cell_tensor.n[dimension]);
  AssertIndexRange(cell_dof_index, cell_dof_tensor.n[dimension]);
  AssertIndexRange(dimension, n_dimensions);
  const auto & n_dofs_per_cell_1d = cell_dof_tensor.n[dimension];
  return cell_no * n_dofs_per_cell_1d + cell_dof_index;
}


template<int n_dimensions>
inline std::pair<unsigned int, unsigned int>
PatchLocalIndexHelper<n_dimensions>::dof_range_1d(const unsigned int cell_no,
                                                  const unsigned int dimension) const
{
  const auto         last_cell_dof_index = this->n_dofs_per_cell_1d(dimension) - 1;
  const unsigned int first               = dof_index_1d(cell_no, 0, dimension);
  const unsigned int last                = dof_index_1d(cell_no, last_cell_dof_index, dimension);
  return {first, last + 1};
}


template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::n_cells_1d(const unsigned int dimension) const
{
  return this->cell_tensor.size(dimension);
}


template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::n_dofs_per_cell_1d(const unsigned int dimension) const
{
  return this->cell_dof_tensor.size(dimension);
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
      {
        const auto last_cell_no        = this->n_cells_1d(d) - 1;
        const auto last_cell_dof_index = this->n_dofs_per_cell_1d(d) - 1;
        sizes[d] = this->dof_index_1d(last_cell_no, last_cell_dof_index, d) + 1;
      }
      return sizes;
    }())
{
}


template<int n_dimensions>
inline std::array<unsigned int, n_dimensions>

PatchLocalTensorHelper<n_dimensions>::dof_multi_index(const unsigned int cell_no,
                                                      const unsigned int cell_dof_index) const
{
  const auto & cell_no_multi        = IndexHelperBase::cell_tensor.multi_index(cell_no);
  const auto & cell_dof_index_multi = IndexHelperBase::cell_dof_tensor.multi_index(cell_dof_index);
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


template<int n_dimensions>
inline unsigned int
PatchLocalTensorHelper<n_dimensions>::n_dofs_1d(const unsigned int dimension) const
{
  return TensorHelperBase::size(dimension);
}


// --------------------------------   PatchDoFWorker   --------------------------------



template<int dim, typename Number>
inline PatchDoFWorker<dim, Number>::PatchDoFWorker(const DoFInfo<dim, Number> & dof_info_in)
  : PatchWorker<dim, Number>(*(dof_info_in.patch_info)),
    dof_info(&dof_info_in),
    patch_dof_tensor(TPSS::UniversalInfo<dim>::n_cells_per_direction(
                       this->patch_info->get_additional_data().patch_variant),
                     /// currently assuming isotropy ...
                     get_shape_info(0).fe_degree + 1,
                     dof_info_in.get_dof_layout())
{
  for(auto d = 0U; d < dim; ++d)
    AssertDimension(get_dof_tensor().n_dofs_per_cell_1d(d), get_shape_info(d).fe_degree + 1);
  Assert(dof_info->get_additional_data().level != numbers::invalid_unsigned_int,
         ExcMessage("Implemented for level cells only."));
}


template<int dim, typename Number>
inline ArrayView<const types::global_dof_index>
PatchDoFWorker<dim, Number>::get_dof_indices_on_cell(const unsigned int patch_id,
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


template<int dim, typename Number>
inline std::array<ArrayView<const types::global_dof_index>, PatchDoFWorker<dim, Number>::macro_size>
PatchDoFWorker<dim, Number>::get_dof_indices_on_cell(const unsigned int patch_id,
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


template<int dim, typename Number>
inline const DoFInfo<dim, Number> &
PatchDoFWorker<dim, Number>::get_dof_info() const
{
  return *dof_info;
}


template<int dim, typename Number>
inline const PatchLocalTensorHelper<dim> &
PatchDoFWorker<dim, Number>::get_dof_tensor() const
{
  return patch_dof_tensor;
}


template<int dim, typename Number>
inline const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
PatchDoFWorker<dim, Number>::get_shape_info(const unsigned int dimension) const
{
  AssertIndexRange(dimension, dim);
  const auto * shape_info = dof_info->shape_infos[dimension];
  Assert(shape_info, ExcMessage("Shape info not set."));
  return *shape_info;
}



} // end namespace TPSS

#include "dof_info.templates.h"

#endif // end inclusion guard
