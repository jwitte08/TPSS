#ifndef PATCH_DOF_WORKER_H
#define PATCH_DOF_WORKER_H

#include "dof_info.h"

#include <array>
#include <memory>



namespace TPSS
{
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

  /**
   * Returns cached global dof indices on local cell @p cell_no within patch
   * @patch_id subject to lexicographical ordering.
   */
  std::array<ArrayView<const types::global_dof_index>, macro_size>
  get_dof_indices_on_cell(const unsigned int patch_id, const unsigned int cell_no) const;

  /**
   * Same as above, but returning only global dof indices for vectorization lane
   * @lane.
   */
  ArrayView<const types::global_dof_index>
  get_dof_indices_on_cell(const unsigned int patch_id,
                          const unsigned int cell_no,
                          const unsigned int lane) const;

  /**
   * Returns cached global dof indices on patch @patch_id at vectorization lane
   * @p lane. The returned array is subject to lexicographical ordering.
   */
  ArrayView<const types::global_dof_index>
  get_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Returns global dof indices on the fly on patch @patch_id at vectorization
   * lane @p lane. The returned array is subject to lexicographical ordering.
   */
  std::vector<types::global_dof_index>
  fill_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  const DoFInfo<dim, Number> &
  get_dof_info() const;

  const PatchLocalTensorHelper<dim> &
  get_dof_tensor() const;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
  get_shape_info(const unsigned int dimension) const;

  /**
   * Returns the start position of global dof indices on patch @patch_id at
   * vectorization lane @p lane stored in the flat field @p
   * global_dof_indices_patchwise of the underlying dof info. In addition,
   * returns the number of dof indices on the given patch.
   */
  std::pair<unsigned int, unsigned int>
  get_start_and_number_of_dof_indices(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Returns the start position of global dof indices on local cell @p cell_no
   * within patch @patch_id at vectorization lane @p lane which are stored in
   * the flat field @p global_dof_indices_cellwise of the underlying dof
   * info. In addition, returns the number of dof indices on the given cell.
   */
  std::pair<unsigned int, unsigned int>
  get_start_and_number_of_dof_indices(const unsigned int patch_id,
                                      const unsigned int cell_no,
                                      const unsigned int lane) const;

  /**
   * TODO
   */
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
};



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
inline std::vector<types::global_dof_index>
PatchDoFWorker<dim, Number>::fill_dof_indices_on_patch(const unsigned int patch_id,
                                                       const unsigned int lane) const
{
  AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  const auto                           n_cells         = patch_dof_tensor.cell_tensor.n_flat();
  const auto                           n_dofs_per_cell = patch_dof_tensor.cell_dof_tensor.n_flat();
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
          const unsigned int patch_dof_index = patch_dof_tensor.dof_index(cell_no, cell_dof_index);
          const bool is_boundary_dof = patch_dof_tensor.is_boundary_face_dof(patch_dof_index);
          global_dof_indices_plain[patch_dof_index] = is_boundary_dof ?
                                                        numbers::invalid_dof_index :
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
ArrayView<const types::global_dof_index>
PatchDoFWorker<dim, Number>::get_dof_indices_on_patch(const unsigned int patch_id,
                                                      const unsigned int lane) const
{
  AssertIndexRange(lane, macro_size);
  Assert(dof_info, ExcMessage("Dof info is not set."));
  Assert(!(dof_info->global_dof_indices_patchwise.empty()),
         ExcMessage("Dof indices aren't cached."));
  /// Return indices of first lane if the current lane @lane is not filled.
  if(lane >= this->n_lanes_filled(patch_id))
    return get_dof_indices_on_patch(patch_id, 0);

  const auto [dof_start, n_dofs] = get_start_and_number_of_dof_indices(patch_id, lane);
  const auto begin               = dof_info->global_dof_indices_patchwise.data() + dof_start;
  return ArrayView<const types::global_dof_index>(begin, n_dofs);
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
                                                                 const unsigned int lane) const
{
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  Assert(dof_info, ExcMessage("DoF info is not set."));
  Assert(!(dof_info->global_dof_indices_patchwise.empty()), ExcMessage("No indices are cached."));
  AssertDimension(dof_info->start_of_dof_indices_patchwise.size(),
                  this->get_partition_data().n_subdomains() + 1);
  AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
  const unsigned int start          = dof_info->start_of_dof_indices_patchwise[patch_id];
  const unsigned int end            = dof_info->start_of_dof_indices_patchwise[patch_id + 1];
  const auto         n_lanes_filled = this->n_lanes_filled(patch_id);
  /// Assuming the same number of dofs on each lane !
  AssertDimension((end - start) % n_lanes_filled, 0);
  const unsigned int n_dofs_per_lane = (end - start) / n_lanes_filled;
  const unsigned int start_at_lane   = start + lane * n_dofs_per_lane;
  AssertIndexRange(end - 1, dof_info->global_dof_indices_patchwise.size());
  return std::make_pair(start_at_lane, n_dofs_per_lane);
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
  AssertIndexRange(cell_position, dof_info->start_and_number_of_dof_indices_cellwise.size());
  return dof_info->start_and_number_of_dof_indices_cellwise[cell_position];
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

#include "patch_dof_worker.templates.h"

#endif // end inclusion guard
