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
  using Base = PatchWorker<dim, Number>;

  using Base::macro_size;

  using patch_worker_type  = Base;
  using cell_iterator_type = typename PatchInfo<dim>::dof_cell_iterator_type;

  PatchDoFWorker() = delete;

  PatchDoFWorker(const DoFInfo<dim, Number> & dof_info_in);

  PatchDoFWorker(const PatchDoFWorker &) = delete;

  PatchDoFWorker &
  operator=(const PatchDoFWorker &) = delete;

  cell_iterator_type
  get_cell_iterator(const unsigned int patch_index,
                    const unsigned int cell_no,
                    const unsigned int lane) const;

  std::vector<cell_iterator_type>
  get_cell_collection(const unsigned int patch_index, const unsigned int lane) const;

  std::vector<std::array<cell_iterator_type, macro_size>>
  get_cell_collection(const unsigned int patch_index) const;

  std::set<unsigned int>
  get_constrained_local_dof_indices_1d(const unsigned int patch_id,
                                       const unsigned int dimension,
                                       const unsigned int lane) const;

  std::set<unsigned int>
  get_constrained_local_dof_indices(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Returns cached global dof indices of local cell @p cell_no subject to a
   * cell local lexicographical ordering. We emphasize that all global dof
   * indices are returned, even those that might not be part of patch @p
   * patch_id (for example vertex patches based on conforming finite
   * elements exclude dofs at the patch boundary).
   */
  std::array<ArrayView<const unsigned int>, macro_size>
  get_dof_indices_on_cell(const unsigned int patch_id, const unsigned int cell_no) const;

  /**
   * Same as above, but returning only global dof indices for vectorization lane
   * @lane and shape function component @p component (assuming the underlying
   * finite element is primitive).
   */
  ArrayView<const unsigned int>
  get_dof_indices_on_cell(const unsigned int patch_id,
                          const unsigned int cell_no,
                          const unsigned int lane,
                          const unsigned int component) const;

  /**
   * Returns cached global dof indices on patch @patch_id at vectorization lane
   * @p lane. The returned array is subject to patch local lexicographical ordering.
   */
  ArrayView<const unsigned int>
  get_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Returns cached global dof indices on patch @patch_id at vectorization lane
   * @p lane of vector component @component. The returned array is subject to
   * patch local lexicographical ordering.
   */
  ArrayView<const unsigned int>
  get_dof_indices_on_patch(const unsigned int patch_id,
                           const unsigned int lane,
                           const unsigned int component) const;

  /**
   * Returns global dof indices on patch @patch_id at vectorization lane @p lane
   * in its unique numbering among all mpi processes.
   */
  std::vector<types::global_dof_index>
  get_global_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Returns global dof indices on patch @patch_id at vectorization lane @p lane
   * of vector component @component in its unique numbering among all mpi
   * processes.
   */
  std::vector<types::global_dof_index>
  get_global_dof_indices_on_patch(const unsigned int patch_id,
                                  const unsigned int lane,
                                  const unsigned int component) const;

  /**
   * Returns global dof indices on the fly on patch @patch_id at vectorization
   * lane @p lane. The returned array is subject to lexicographical ordering.
   */
  std::vector<unsigned int>
  fill_dof_indices_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  const DoFInfo<dim, Number> &
  get_dof_info() const;

  const PatchLocalTensorHelper<dim> &
  get_dof_tensor() const;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
  get_shape_info() const;

  /**
   * Returns the start position of global dof indices on patch @patch_id at
   * vectorization lane @p lane stored in the flat field @p
   * dof_indices_patchwise of the underlying dof info. In addition,
   * returns the number of dof indices on the given patch.
   */
  std::pair<unsigned int, unsigned int>
  get_dof_start_and_quantity_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Same as above but we return only the start position and quantity of dofs
   * for the specified component @p component.
   */
  std::pair<unsigned int, unsigned int>
  get_dof_start_and_quantity_on_patch(const unsigned int patch_id,
                                      const unsigned int lane,
                                      const unsigned int component) const;

  /**
   * Returns the start position of global dof indices on local cell @p cell_no
   * within patch @patch_id at vectorization lane @p lane which are stored in
   * the flat field @p dof_indices_cellwise of the underlying dof
   * info. In addition, returns the number of dof indices on the given cell.
   */
  std::pair<unsigned int, unsigned int>
  get_dof_start_and_quantity_on_cell(const unsigned int patch_id,
                                     const unsigned int cell_no,
                                     const unsigned int lane) const;

  /**
   * For each dof in the range returned by @p get_dof_indices_on_patch() we
   * return a boolean that indicates whether the dof is restricted or not. A
   * restricted dof is ignored during prolongation of patch-local values into
   * global values. For restricted additive Schwarz methods (RAS) we might use a
   * partition of unity at the algebraic level: let $\tilde{R}_j^T$ denote the
   * restricted prolongation operator associated to the standard prolongation
   * $R_j^T$, then it holds $$\sum_j \R_j^T = id$$, where $id$ is the identity.
   */
  std::vector<bool>
  get_restricted_dof_flags(const unsigned int patch_id, const unsigned int lane) const;

  // /**
  //  * TODO ... this method is not tested
  //  * the DoFInfo takes care of setting up a MPI::Partitioner during initialization
  //  */
  // std::shared_ptr<const Utilities::MPI::Partitioner>
  // initialize_vector_partitioner() const;

  /**
   * The MPI::Partitioner in the underlying DoFInfo is used to initialize global
   * dof vectors with respect to locally owned and ghosted dof indices.
   */
  void
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> & vec) const;

  /**
   * Same as above except that in the serial case only the number of global dofs
   * on the current level suffices to initialize @p vec.
   */
  void
  initialize_dof_vector(Vector<Number> & vec) const;

  /**
   * Number of degrees of freedom on patch (assuming same number on all patches)
   * TODO...
   */
  unsigned int
  n_dofs() const;

  /**
   * Return the total number of global dofs on the current mesh level. In case
   * of a parallel distributed mesh this corresponds to the number of global
   * dofs accumulated over all mpi processes.
   */
  types::global_dof_index
  n_global_dofs() const;

  /**
   * The actual number of degrees of freedom on a patch in dimension @p
   * dimension. This excludes degrees of freedom at the boundary of a patch
   * which are usually neglected in favor of homogeneous boundary conditions.
   */
  unsigned int
  n_dofs_1d(const unsigned int dimension) const;

  /**
   * The number of degrees of freedom on a patch in dimension @p dimension if
   * degrees of freedom at the boundary of a patch are not excluded.
   */
  unsigned int
  n_dofs_plain_1d(const unsigned int dimension) const;

  unsigned int
  n_dofs_on_cell(const unsigned int component) const
  {
    AssertIndexRange(component, n_components);
    return dof_info->n_dofs_on_cell_per_comp[component];
  }

  unsigned int
  n_preceding_dofs_on_cell(const unsigned int component) const
  {
    AssertIndexRange(component, n_components);
    return n_preceding_dofs_on_cell_[component];
  }

private:
  std::vector<types::global_dof_index>
  proc_local_to_global_dof_indices_on_patch_impl(
    const ArrayView<const unsigned int> & local_view) const;

  const DoFInfo<dim, Number> * const dof_info;

  const TPSS::DoFLayout dof_layout;

  PatchLocalTensorHelper<dim> patch_dof_tensor;

  const unsigned int n_components;

  std::vector<unsigned int> n_preceding_dofs_on_cell_;
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
                     get_shape_info().get_shape_data().fe_degree + 1,
                     dof_info_in.get_dof_layout()),
    n_components(dof_info_in.shape_info->n_components)
{
  for(auto component = 0U; component < n_components; ++component)
  {
    unsigned int n_dofs_preceding = 0;
    for(auto c = 0U; c < component; ++c)
      n_dofs_preceding += n_dofs_on_cell(c);
    n_preceding_dofs_on_cell_.push_back(n_dofs_preceding);
  }
  AssertDimension(n_preceding_dofs_on_cell_.size(), n_components);

  for(auto d = 0U; d < dim; ++d)
    AssertDimension(get_dof_tensor().n_dofs_per_cell_1d(d),
                    get_shape_info().get_shape_data(d).fe_degree + 1);
  Assert(dof_info->get_additional_data().level != numbers::invalid_unsigned_int,
         ExcMessage("Implemented for level cells only."));
}


template<int dim, typename Number>
inline std::vector<unsigned int>
PatchDoFWorker<dim, Number>::fill_dof_indices_on_patch(const unsigned int patch_id,
                                                       const unsigned int lane) const
{
  AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  const auto & additional_data = dof_info->get_additional_data();
  const auto n_cells = this->n_cells_per_subdomain(); // !!! patch_dof_tensor.cell_tensor.n_flat();
  const auto n_dofs_per_cell_per_comp = dof_layout == DoFLayout::DGP ?
                                          get_shape_info().dofs_per_component_on_cell :
                                          patch_dof_tensor.cell_dof_tensor.n_flat();
  std::vector<unsigned int> global_dof_indices;

  const bool is_L2_conforming =
    dof_layout == DoFLayout::DGQ ||
    (dof_layout == DoFLayout::Q && additional_data.force_no_boundary_condition) ||
    dof_layout == DoFLayout::DGP;
  const bool is_H1_conforming =
    (dof_layout == DoFLayout::Q && !additional_data.force_no_boundary_condition);

  if(is_H1_conforming)
  {
    if(this->patch_variant == TPSS::PatchVariant::vertex)
    {
      /// Fill global dof indices with a patch local lexicographical
      /// ordering. Dof indices at the patch boundary are marked as invalid.
      const unsigned            n_patch_dofs_per_component = patch_dof_tensor.n_flat();
      std::vector<unsigned int> global_dof_indices_plain(n_components * n_patch_dofs_per_component);
      for(auto comp = 0U; comp < n_components; ++comp)
      {
        AssertDimension(n_dofs_per_cell_per_comp, n_dofs_on_cell(comp));
        const unsigned int patch_dof_stride = comp * n_patch_dofs_per_component;
        for(auto cell_no = 0U; cell_no < n_cells; ++cell_no)
        {
          const auto global_dof_indices_on_cell =
            get_dof_indices_on_cell(patch_id, cell_no, lane, comp);
          for(auto cell_dof_index = 0U; cell_dof_index < n_dofs_per_cell_per_comp; ++cell_dof_index)
          {
            const unsigned int patch_dof_index_per_comp =
              patch_dof_tensor.dof_index(cell_no, cell_dof_index);
            const bool is_boundary_dof =
              patch_dof_tensor.is_boundary_face_dof(patch_dof_index_per_comp);
            global_dof_indices_plain[patch_dof_stride + patch_dof_index_per_comp] =
              is_boundary_dof ? numbers::invalid_unsigned_int :
                                global_dof_indices_on_cell[cell_dof_index];
          }
        }
      }

      /// Copy global dof indices neglecting all indices at the patch boundary
      /// (which have been marked in a previous step).
      std::copy_if(global_dof_indices_plain.cbegin(),
                   global_dof_indices_plain.cend(),
                   std::back_inserter(global_dof_indices),
                   [](const auto dof_index) { return dof_index != numbers::invalid_unsigned_int; });
    }
    else
      AssertThrow(false, ExcMessage("Patch variant is not supported."));
  }

  else if(is_L2_conforming)
  {
    const unsigned int n_patch_dofs_per_component =
      dof_layout == DoFLayout::DGP ? n_cells * get_shape_info().dofs_per_component_on_cell :
                                     patch_dof_tensor.n_flat();
    AssertIndexRange(n_patch_dofs_per_component, patch_dof_tensor.n_flat() + 1);
    std::vector<unsigned int> global_dof_indices_plain(n_components * n_patch_dofs_per_component);
    for(auto comp = 0U; comp < n_components; ++comp)
    {
      AssertDimension(n_dofs_per_cell_per_comp, n_dofs_on_cell(comp));
      const unsigned int patch_dof_stride = comp * n_patch_dofs_per_component;
      for(auto cell_no = 0U; cell_no < n_cells; ++cell_no)
      {
        const auto global_dof_indices_on_cell =
          get_dof_indices_on_cell(patch_id, cell_no, lane, comp);
        AssertDimension(n_dofs_per_cell_per_comp, global_dof_indices_on_cell.size()); // ???
        for(auto cell_dof_index = 0U; cell_dof_index < n_dofs_per_cell_per_comp; ++cell_dof_index)
        {
          /// We dont obtain a patch-local lexicographical ordering for DGP
          /// (TODO is it possible to have a truncated lexicographical
          /// ordering?).
          const unsigned int patch_dof_index =
            patch_dof_stride + (dof_layout == DoFLayout::DGP ?
                                  cell_no * n_dofs_per_cell_per_comp + cell_dof_index :
                                  patch_dof_tensor.dof_index(cell_no, cell_dof_index));
          global_dof_indices_plain[patch_dof_index] = global_dof_indices_on_cell[cell_dof_index];
        }
      }
    }
    std::swap(global_dof_indices, global_dof_indices_plain);
  }

  else
    AssertThrow(false, ExcMessage("Finite element is not supported."));

  // if(dof_layout != DoFLayout::DGP) // !!! TODO n_dofs() is based on a full tensor structure
  AssertDimension(n_dofs(), global_dof_indices.size());
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
inline ArrayView<const unsigned int>
PatchDoFWorker<dim, Number>::get_dof_indices_on_cell(const unsigned int patch_id,
                                                     const unsigned int cell_no,
                                                     const unsigned int lane,
                                                     const unsigned int component) const
{
  AssertIndexRange(lane, macro_size);
  AssertIndexRange(component, n_components);
  const unsigned int n_lanes_filled = this->n_lanes_filled(patch_id);
  if(lane >= n_lanes_filled) // !!! TODO throw exc instead?
    return get_dof_indices_on_cell(patch_id, cell_no, 0, component);

  const auto [dof_start, n_dofs] = get_dof_start_and_quantity_on_cell(patch_id, cell_no, lane);
  (void)n_dofs;
  AssertDimension(n_preceding_dofs_on_cell(n_components - 1) + n_dofs_on_cell(n_components - 1),
                  n_dofs);
  const unsigned int n_dofs_component = n_dofs_on_cell(component);
  const unsigned int n_dofs_preceding = n_preceding_dofs_on_cell(component);
  const auto         begin = dof_info->dof_indices_cellwise.data() + dof_start + n_dofs_preceding;

  return ArrayView<const unsigned int>(begin, n_dofs_component);
}


template<int dim, typename Number>
inline std::array<ArrayView<const unsigned int>, PatchDoFWorker<dim, Number>::macro_size>
PatchDoFWorker<dim, Number>::get_dof_indices_on_cell(const unsigned int patch_id,
                                                     const unsigned int cell_no) const
{
  std::array<ArrayView<const unsigned int>, macro_size> views;
  for(auto lane = 0U; lane < macro_size; ++lane)
  {
    const auto [dof_start, n_dofs] = get_dof_start_and_quantity_on_cell(patch_id, cell_no, lane);
    (void)dof_start;
    const auto & view = get_dof_indices_on_cell(patch_id, cell_no, lane, 0);
    views[lane].reinit(view.data(), n_dofs);
  }
  return views;
}


template<int dim, typename Number>
ArrayView<const unsigned int>
PatchDoFWorker<dim, Number>::get_dof_indices_on_patch(const unsigned int patch_id,
                                                      const unsigned int lane) const
{
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  Assert(dof_info, ExcMessage("dof_info is not set."));
  Assert(!(dof_info->dof_indices_patchwise.empty()), ExcMessage("Dof indices aren't cached."));

  const auto [dof_start, n_dofs] = get_dof_start_and_quantity_on_patch(patch_id, lane);
  const auto begin               = dof_info->dof_indices_patchwise.data() + dof_start;
  return ArrayView<const unsigned int>(begin, n_dofs);
}


template<int dim, typename Number>
ArrayView<const unsigned int>
PatchDoFWorker<dim, Number>::get_dof_indices_on_patch(const unsigned int patch_id,
                                                      const unsigned int lane,
                                                      const unsigned int component) const
{
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  AssertIndexRange(component, n_components);
  Assert(dof_info, ExcMessage("dof_info is not set."));
  Assert(!(dof_info->dof_indices_patchwise.empty()), ExcMessage("Dof indices aren't cached."));

  const auto [dof_start, n_dofs] = get_dof_start_and_quantity_on_patch(patch_id, lane, component);
  const auto begin               = dof_info->dof_indices_patchwise.data() + dof_start;
  return ArrayView<const unsigned int>(begin, n_dofs);
}


template<int dim, typename Number>
std::vector<types::global_dof_index>
PatchDoFWorker<dim, Number>::proc_local_to_global_dof_indices_on_patch_impl(
  const ArrayView<const unsigned int> & local_view) const
{
  Assert(dof_info, ExcMessage("dof_info is not set."));
  Assert(dof_info->vector_partitioner, ExcMessage("vector_partitioner is not set."));
  std::vector<types::global_dof_index> global_dof_indices;
  std::transform(local_view.begin(),
                 local_view.end(),
                 std::back_inserter(global_dof_indices),
                 [&](const auto & local_index) {
                   return dof_info->vector_partitioner->local_to_global(local_index);
                 });
  AssertDimension(global_dof_indices.size(), local_view.size());
  return global_dof_indices;
}


template<int dim, typename Number>
std::vector<types::global_dof_index>
PatchDoFWorker<dim, Number>::get_global_dof_indices_on_patch(const unsigned int patch_id,
                                                             const unsigned int lane) const
{
  return proc_local_to_global_dof_indices_on_patch_impl(get_dof_indices_on_patch(patch_id, lane));
}


template<int dim, typename Number>
std::vector<types::global_dof_index>
PatchDoFWorker<dim, Number>::get_global_dof_indices_on_patch(const unsigned int patch_id,
                                                             const unsigned int lane,
                                                             const unsigned int component) const
{
  return proc_local_to_global_dof_indices_on_patch_impl(
    get_dof_indices_on_patch(patch_id, lane, component));
}


template<int dim, typename Number>
std::vector<bool>
PatchDoFWorker<dim, Number>::get_restricted_dof_flags(const unsigned int patch_id,
                                                      const unsigned int lane) const
{
  Assert(dof_info, ExcMessage("dof_info is not set."));
  Assert(!(dof_info->dof_indices_patchwise.empty()), ExcMessage("Dof indices aren't cached."));

  const auto [dof_start, n_dofs] = get_dof_start_and_quantity_on_patch(patch_id, lane);
  std::vector<bool> flags;
  const auto        begin = dof_info->restricted_dof_flags_patchwise.begin() + dof_start;
  std::copy_n(begin, n_dofs, std::back_inserter(flags));
  return flags;
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
PatchDoFWorker<dim, Number>::get_shape_info() const
{
  Assert(dof_info->shape_info, ExcMessage("shape_info is not set."));
  return *(dof_info->shape_info);
}


template<int dim, typename Number>
inline std::pair<unsigned int, unsigned int>
PatchDoFWorker<dim, Number>::get_dof_start_and_quantity_on_patch(const unsigned int patch_id,
                                                                 const unsigned int lane) const
{
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  Assert(dof_info, ExcMessage("DoFInfo is not set."));
  Assert(!(dof_info->dof_indices_patchwise.empty()), ExcMessage("No indices are cached."));
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
  AssertIndexRange(end - 1, dof_info->dof_indices_patchwise.size());
  return std::make_pair(start_at_lane, n_dofs_per_lane);
}


template<int dim, typename Number>
inline std::pair<unsigned int, unsigned int>
PatchDoFWorker<dim, Number>::get_dof_start_and_quantity_on_patch(const unsigned int patch_id,
                                                                 const unsigned int lane,
                                                                 const unsigned int component) const
{
  const auto [dof_start, n_dofs] = get_dof_start_and_quantity_on_patch(patch_id, lane);
  /// assume isotropy w.r.t. components
  AssertDimension(n_dofs % n_components, 0U);
  const auto n_dofs_per_component   = n_dofs / n_components;
  const auto dof_start_at_component = dof_start + component * n_dofs_per_component;
  AssertIndexRange(dof_start_at_component + n_dofs_per_component - 1,
                   dof_info->dof_indices_patchwise.size());
  return {dof_start_at_component, n_dofs_per_component};
}


template<int dim, typename Number>
inline std::pair<unsigned int, unsigned int>
PatchDoFWorker<dim, Number>::get_dof_start_and_quantity_on_cell(const unsigned int patch_id,
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
  /// DGP has truncated tensor structure (special case)
  if(dof_layout == DoFLayout::DGP)
  {
    const auto n_cells         = Base::n_cells_per_subdomain();
    const auto n_dofs_per_cell = get_shape_info().dofs_per_component_on_cell;
    return n_cells * n_dofs_per_cell * n_components;
  }

  /// TODO currently, assuming isotropy for each dimension
  unsigned int n_dofs = 1;
  for(auto d = 0U; d < dim; ++d)
    n_dofs *= n_dofs_1d(d);
  /// TODO currently, assuming isotropy for each component
  n_dofs *= n_components;
  return n_dofs;
}


template<int dim, typename Number>
inline types::global_dof_index
PatchDoFWorker<dim, Number>::n_global_dofs() const
{
  Assert(dof_info, ExcMessage("dof_info is not set."));
  const auto vector_partitioner = dof_info->vector_partitioner;
  Assert(vector_partitioner, ExcMessage("vector_partitioner is not set."));
  AssertDimension(vector_partitioner->size(),
                  dof_info->dof_handler->n_dofs(dof_info->get_additional_data().level));
  return vector_partitioner->size();
}


template<int dim, typename Number>
inline unsigned int
PatchDoFWorker<dim, Number>::n_dofs_1d(const unsigned dimension) const
{
  AssertIndexRange(dimension, dim);
  const auto & additional_data = dof_info->get_additional_data();
  if(dof_layout == TPSS::DoFLayout::Q && !additional_data.force_no_boundary_condition)
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
