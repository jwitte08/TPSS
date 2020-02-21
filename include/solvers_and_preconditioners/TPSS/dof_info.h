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
   * Returns wether the patch local dof index @p patch_dof_index is part of the
   * outermost layer of dofs.
   */
  bool
  is_boundary_face_dof(const unsigned int patch_dof_index) const;

  /**
   * Returns wether the patch local dof index @p patch_dof_index is part of the
   * outermost layer of dofs.
   */
  bool
  is_boundary_face_dof_1d(const unsigned int patch_dof_index, const unsigned int dimension) const;

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
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

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
    start_and_number_of_dof_indices.clear();
    global_dof_indices_cellwise.clear();
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

  std::vector<std::pair<unsigned int, unsigned int>> start_and_number_of_dof_indices;

  /*
   * Array uniquely storing for each cell stored in patch_info global dof indices.
   */
  std::vector<types::global_dof_index> global_dof_indices_cellwise;

  std::vector<unsigned int> start_of_global_dof_indices;

  std::vector<types::global_dof_index> global_dof_indices_patchwise;

  const PatchInfo<dim> * patch_info = nullptr;

  std::array<const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> *, dim>
    shape_infos;

  std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner;

  DoFAccessor<dim, DoFHandler<dim>, true>
  get_level_dof_accessor(const unsigned int cell_position) const
  {
    Assert(patch_info, ExcMessage("Patch info not initialized."));
    const auto [cell_level, cell_index] = patch_info->get_cell_level_and_index(cell_position);
    return get_level_dof_accessor_impl(cell_index, cell_level);
  }

  DoFAccessor<dim, DoFHandler<dim>, true>
  get_level_dof_accessor_impl(const unsigned int cell_index, const unsigned int level) const
  {
    const auto & tria = dof_handler->get_triangulation();
    return DoFAccessor<dim, DoFHandler<dim>, true>(&tria, level, cell_index, dof_handler);
  }

  std::vector<types::global_dof_index>
  get_level_dof_indices(const unsigned int cell_position) const
  {
    const auto & cell = get_level_dof_accessor(cell_position);
    return get_level_dof_indices_impl(cell);
  }

  std::vector<types::global_dof_index>
  get_level_dof_indices_impl(const DoFAccessor<dim, DoFHandler<dim>, true> & cell) const
  {
    const auto                           n_dofs_per_cell = dof_handler->get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> level_dof_indices(n_dofs_per_cell);
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
  unsigned int                 level = numbers::invalid_unsigned_int;
  std::set<types::boundary_id> dirichlet_ids;
  TPSS::CachingStrategy        caching_strategy = TPSS::CachingStrategy::Cached;
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

  std::set<unsigned int>
  get_constrained_local_dof_indices_1d(const unsigned int patch_id,
                                       const unsigned int dimension,
                                       const unsigned int lane) const;

  std::set<unsigned int>
  get_constrained_local_dof_indices(const unsigned int patch_id, const unsigned int lane) const;

  ArrayView<const types::global_dof_index>
  get_dof_indices_on_cell(const unsigned int patch_id,
                          const unsigned int cell_no,
                          const unsigned int lane) const;

  std::array<ArrayView<const types::global_dof_index>, macro_size>
  get_dof_indices_on_cell(const unsigned int patch_id, const unsigned int cell_no) const;

  std::pair<unsigned int, unsigned int>
  get_start_and_number_of_dof_indices(const unsigned int patch_id,
                                             const unsigned int lane) const
  {
    AssertIndexRange(lane, this->n_lanes_filled(patch_id));
    Assert(dof_info, ExcMessage("DoF info is not set."));
    Assert(!(dof_info->global_dof_indices_patchwise.empty()), ExcMessage("No indices are cached."));
    AssertDimension(dof_info->start_of_global_dof_indices.size(),
                    this->get_partition_data().n_subdomains() + 1);
    AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
    const unsigned int start          = dof_info->start_of_global_dof_indices[patch_id];
    const unsigned int end            = dof_info->start_of_global_dof_indices[patch_id + 1];
    const auto         n_lanes_filled = this->n_lanes_filled(patch_id);
    /// Assuming the same number of dofs on each lane !
    AssertDimension((end - start) % n_lanes_filled, 0);
    const unsigned int n_dofs_per_lane = (end - start) / n_lanes_filled;
    const unsigned int start_at_lane   = start + lane * n_dofs_per_lane;
    AssertIndexRange(end - 1, dof_info->global_dof_indices_patchwise.size());
    return std::make_pair(start_at_lane, n_dofs_per_lane);
  }

  // std::pair<unsigned int, unsigned int>
  // get_start_and_number_of_dof_indices(const unsigned int patch_id) const
  // {
  //   Assert(dof_info, ExcMessage("DoF info is not set."));
  //   AssertDimension(dof_info->start_of_global_dof_indices.size(),
  //                   this->get_partition_data().n_subdomains() + 1);
  //   AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
  //   const unsigned int start  = dof_info->start_of_global_dof_indices[patch_id];
  //   const unsigned int end    = dof_info->start_of_global_dof_indices[patch_id + 1];
  //   const unsigned int n_dofs = end - start;
  //   AssertIndexRange(end - 1, dof_info->global_dof_indices_patchwise.size());
  //   return std::make_pair(start, n_dofs);
  // }

  ArrayView<const types::global_dof_index>
  get_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const
  {
    AssertIndexRange(lane, macro_size);
    /// Return indices of first lane if the current lane @lane is not filled.
    if(lane >= this->n_lanes_filled(patch_id))
      return get_dof_indices_on_patch(patch_id, 0);

    const auto [dof_start, n_dofs] = get_start_and_number_of_dof_indices(patch_id, lane);
    const auto begin               = dof_info->global_dof_indices_patchwise.data() + dof_start;
    return ArrayView<const types::global_dof_index>(begin, n_dofs);
  }

  ArrayView<const std::array<types::global_dof_index, macro_size>>
  get_dof_indices_on_patch(const unsigned int patch_id) const
  {
    const auto & dof_indices = get_dof_indices_on_patch(patch_id, 0);
    dofs_tmp.clear();
    dofs_tmp.resize(dof_indices.size());
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      const auto & dof_indices = get_dof_indices_on_patch(patch_id, lane);
      for(auto i = 0U; i < dof_indices.size(); ++i)
        dofs_tmp[i][lane] = dof_indices[i];
    }
    return ArrayView<const std::array<types::global_dof_index, macro_size>>(dofs_tmp.data(),
                                                                            dofs_tmp.size());
  }

  std::vector<types::global_dof_index>
  fill_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const
  {
    AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
    AssertIndexRange(lane, this->n_lanes_filled(patch_id));
    const auto n_cells         = patch_dof_tensor.cell_tensor.n_flat();
    const auto n_dofs_per_cell = patch_dof_tensor.cell_dof_tensor.n_flat();
    std::vector<types::global_dof_index> global_dof_indices;

    if(dof_layout == DoFLayout::DGQ)
    {
      std::vector<types::global_dof_index> global_dof_indices_plain(patch_dof_tensor.n_flat());
      for(auto cell_no = 0U; cell_no < n_cells; ++cell_no)
      {
        const auto global_dof_indices_on_cell = get_dof_indices_on_cell(patch_id, cell_no, lane);
        for(auto cell_dof_index = 0U; cell_dof_index < n_dofs_per_cell; ++cell_dof_index)
        {
          const unsigned int patch_dof_index = patch_dof_tensor.dof_index(cell_no, cell_dof_index);
          global_dof_indices_plain[patch_dof_index] = global_dof_indices_on_cell[cell_dof_index];
        }
      }
      std::swap(global_dof_indices, global_dof_indices_plain);
    }

    else if(dof_layout == DoFLayout::Q)
    {
      if(this->patch_variant == TPSS::PatchVariant::vertex)
      {
        /// Fill global dof indices regarding a patch local lexicographical
        /// ordering. Dof indices at the patch boundary are marked as invalid.
        std::vector<types::global_dof_index> global_dof_indices_plain(patch_dof_tensor.n_flat());
        for(auto cell_no = 0U; cell_no < n_cells; ++cell_no)
        {
          const auto global_dof_indices_on_cell = get_dof_indices_on_cell(patch_id, cell_no, lane);
          for(auto cell_dof_index = 0U; cell_dof_index < n_dofs_per_cell; ++cell_dof_index)
          {
            const unsigned int patch_dof_index =
              patch_dof_tensor.dof_index(cell_no, cell_dof_index);
            const bool is_boundary_dof = patch_dof_tensor.is_boundary_face_dof(patch_dof_index);
            global_dof_indices_plain[patch_dof_index] =
              is_boundary_dof ? numbers::invalid_dof_index :
                                global_dof_indices_on_cell[cell_dof_index];
          }
        }

        /// Copy global dof indices neglecting all indices at the patch boundary.
        std::copy_if(global_dof_indices_plain.cbegin(),
                     global_dof_indices_plain.cend(),
                     std::back_inserter(global_dof_indices),
                     [](const auto dof_index) { return dof_index != numbers::invalid_dof_index; });
      }
      else
        AssertThrow(false, ExcMessage("Patch variant is not supported."));
    }
    else
      AssertThrow(false, ExcMessage("Finite element is not supported."));
    return global_dof_indices;
  }

  std::vector<std::array<types::global_dof_index, macro_size>>
  fill_dof_indices_on_patch(const unsigned int patch_id) const
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  const DoFInfo<dim, Number> &
  get_dof_info() const;

  const PatchLocalTensorHelper<dim> &
  get_dof_tensor() const;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
  get_shape_info(const unsigned int dimension) const;

  std::pair<unsigned int, unsigned int>
  get_start_and_number_of_dof_indices(const unsigned int patch_id,
                                      const unsigned int cell_no,
                                      const unsigned int lane) const;

  std::shared_ptr<const Utilities::MPI::Partitioner>
  initialize_vector_partitioner() const;

  unsigned int
  n_dofs() const;

  unsigned int
  n_dofs_1d(const unsigned int dimension) const;

  unsigned int
  n_dofs_plain_1d(const unsigned int dimension) const;

private:
  const DoFInfo<dim, Number> * const dof_info;

  const TPSS::DoFLayout dof_layout;

  PatchLocalTensorHelper<dim> patch_dof_tensor;

  mutable std::vector<std::array<types::global_dof_index, macro_size>> dofs_tmp;
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
inline bool
PatchLocalTensorHelper<n_dimensions>::is_boundary_face_dof(const unsigned int patch_dof_index) const
{
  const auto & patch_dof_index_multi = TensorHelperBase::multi_index(patch_dof_index);
  for(auto d = 0U; d < n_dimensions; ++d)
    if(is_boundary_face_dof_1d(patch_dof_index_multi[d], d))
      return true;
  return false;
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
inline bool
PatchLocalTensorHelper<n_dimensions>::is_boundary_face_dof_1d(const unsigned int patch_dof_index,
                                                              const unsigned int dimension) const
{
  AssertIndexRange(patch_dof_index, n_dofs_1d(dimension));
  const auto last_patch_dof_index = n_dofs_1d(dimension) - 1;
  return patch_dof_index == 0 || patch_dof_index == last_patch_dof_index;
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
    dof_layout(dof_info_in.get_dof_layout()),
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
inline std::set<unsigned>
PatchDoFWorker<dim, Number>::get_constrained_local_dof_indices_1d(const unsigned int patch_id,
                                                                  const unsigned int dimension,
                                                                  const unsigned int lane) const
{
  Assert(this->patch_info, ExcMessage("Patch info is not initialized."));
  std::set<unsigned int> constrained_dof_indices;
  const auto             n_dofs_1d       = patch_dof_tensor.n_dofs_1d(dimension);
  const auto             first_dof_index = 0U;
  const auto             last_dof_index  = n_dofs_1d - 1;

  if(TPSS::PatchVariant::vertex == this->patch_variant)
  {
    constrained_dof_indices.insert({first_dof_index, last_dof_index});
    return constrained_dof_indices;
  }

  else if(TPSS::PatchVariant::cell == this->patch_variant)
  {
    const auto & boundary_ids    = this->get_boundary_ids(patch_id);
    const auto & additional_data = dof_info->get_additional_data();
    const auto   local_face_no   = [](const auto dimension, const auto face_no_1d) {
      /// two faces per dimension, namely endpoints of intervals
      return dimension * 2 + face_no_1d;
    };

    {
      const types::boundary_id boundary_id = boundary_ids[local_face_no(dimension, 0)][lane];
      const bool               at_dirichlet_boundary =
        additional_data.dirichlet_ids.find(boundary_id) != additional_data.dirichlet_ids.cend();
      if(at_dirichlet_boundary)
        constrained_dof_indices.insert(first_dof_index);
    }
    {
      const types::boundary_id boundary_id = boundary_ids[local_face_no(dimension, 1)][lane];
      const bool               at_dirichlet_boundary =
        additional_data.dirichlet_ids.find(boundary_id) != additional_data.dirichlet_ids.cend();
      if(at_dirichlet_boundary)
        constrained_dof_indices.insert(last_dof_index);
    }
    return constrained_dof_indices;
  }

  AssertThrow(false, ExcMessage("Access variant is not implemented."));
  return constrained_dof_indices;
}


template<int dim, typename Number>
inline std::set<unsigned>
PatchDoFWorker<dim, Number>::get_constrained_local_dof_indices(const unsigned int patch_id,
                                                               const unsigned int lane) const
{
  AssertIndexRange(lane, macro_size);
  if(lane >= this->n_lanes_filled(patch_id))
    return get_constrained_local_dof_indices(patch_id, 0);

  std::set<unsigned int> constrained_dof_indices;
  for(auto dimension = 0U; dimension < dim; ++dimension)
  {
    const auto & constrained_dof_indices_1d =
      get_constrained_local_dof_indices_1d(patch_id, dimension, lane);
    for(const auto index : constrained_dof_indices_1d)
    {
      const auto & indices = patch_dof_tensor.sliced_indices(index, dimension);
      constrained_dof_indices.insert(indices.cbegin(), indices.cend());
    }
  }
  return constrained_dof_indices;
}

template<int dim, typename Number>
inline ArrayView<const types::global_dof_index>
PatchDoFWorker<dim, Number>::get_dof_indices_on_cell(const unsigned int patch_id,
                                                     const unsigned int cell_no,
                                                     const unsigned int lane) const
{
  AssertIndexRange(lane, macro_size);
  const unsigned int n_lanes_filled = this->n_lanes_filled(patch_id);
  if(lane >= n_lanes_filled)
    return get_dof_indices_on_cell(patch_id, cell_no, 0);

  const auto [dof_start, n_dofs] = get_start_and_number_of_dof_indices(patch_id, cell_no, lane);
  const auto begin               = dof_info->global_dof_indices_cellwise.data() + dof_start;
  return ArrayView<const types::global_dof_index>(begin, n_dofs);
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


template<int dim, typename Number>
inline std::pair<unsigned int, unsigned int>
PatchDoFWorker<dim, Number>::get_start_and_number_of_dof_indices(const unsigned int patch_id,
                                                                 const unsigned int cell_no,
                                                                 const unsigned int lane) const
{
  AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
  AssertIndexRange(cell_no, this->n_cells_per_subdomain());
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  Assert(dof_info, ExcMessage("DoF info is not set."));
  const auto cell_position = this->get_cell_position(patch_id, cell_no, lane);
  AssertIndexRange(cell_position, dof_info->start_and_number_of_dof_indices.size());
  return dof_info->start_and_number_of_dof_indices[cell_position];
}


template<int dim, typename Number>
inline unsigned int
PatchDoFWorker<dim, Number>::n_dofs() const
{
  unsigned int n_dofs = 1;
  for(auto d = 0U; d < dim; ++d)
    n_dofs *= n_dofs_1d(d);
  return n_dofs;
}


template<int dim, typename Number>
inline unsigned int
PatchDoFWorker<dim, Number>::n_dofs_1d(const unsigned dimension) const
{
  AssertIndexRange(dimension, dim);
  if(dof_layout == TPSS::DoFLayout::Q)
    if(this->patch_variant == TPSS::PatchVariant::vertex)
      return n_dofs_plain_1d(dimension) - 2;
  return n_dofs_plain_1d(dimension);
}


template<int dim, typename Number>
inline unsigned int
PatchDoFWorker<dim, Number>::n_dofs_plain_1d(const unsigned dimension) const
{
  AssertIndexRange(dimension, dim);
  return get_dof_tensor().n_dofs_1d(dimension);
}



} // end namespace TPSS

#include "dof_info.templates.h"

#endif // end inclusion guard
