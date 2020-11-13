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
template<int n_dimensions>
class PatchLocalHelperQ
{
public:
  PatchLocalHelperQ(const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in);

  void
  reinit(const std::vector<FaceInfoLocal<n_dimensions>> & face_infos_in);

  void
  clear();

  unsigned int
  n_cells() const;

  /**
   * Checks if the degree of freedom with local index @p cell_dof_index relative
   * to cell @p cell_no belongs to the patch boundary.
   */
  bool
  at_patch_boundary(const unsigned int cell_no, const unsigned int cell_dof_index) const;

  /**
   * Returns the set of cell dof indices at cell @p cell_no being part of the
   * patch. This includes also the indices which are uniquely assigned to other
   * cells of the patch.
   */
  ArrayView<const unsigned int>
  get_cell_dof_indices(const unsigned int cell_no) const;

  unsigned int
  get_dof_start_plain(const unsigned int cell_no) const;

  /**
   * Returns the number of dofs at cell @p cell_no. This includes all dofs being
   * not part of the patch.
   */
  unsigned int
  n_dofs_per_cell(const unsigned int cell_no) const;

  /**
   * Returns the number of patch local dofs located at cell @p cell_no.
   */
  unsigned int
  n_dofs_per_cell_on_patch(const unsigned int cell_no) const;

  const Tensors::TensorHelper<n_dimensions> cell_dof_tensor;

  std::vector<FaceInfoLocal<n_dimensions>> face_infos;
  std::vector<unsigned int>                dof_starts_plain;
  std::vector<unsigned int>                cell_dof_indices_plain;
};



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

  /**
   * Implementation of @p dof_index_1d for DGP-like finite elements.
   */
  // !!! TODO the current implementation is not correct. actually this function
  // !!! makes no sense on a truncated tensor, but we need this interface for
  // !!! compatibility ...
  unsigned int
  dof_index_1d_dgp_impl(const unsigned int cell_no,
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


/**
 * TODO description missing ...
 */
// !!! TODO do not pass level as additional data, directly use the level of patch_info_in
template<int dim, typename Number>
struct DoFInfo
{
  static constexpr unsigned int macro_size = VectorizedArray<Number>::size();

  struct AdditionalData;

  /**
   * Builds up the data structures for degrees of freedom according to the
   * distribution of (vectorized) subdomains in @p patch_info. The underlying
   * finite element of @p dof_handler and @p shape_info has to be the
   * same. Currently, we support scalar- and vector-valued (but primitive)
   * finite elements.
   */
  void
  initialize(const DoFHandler<dim> *                                                   dof_handler,
             const PatchInfo<dim> *                                                    patch_info,
             const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> * shape_info,
             const AdditionalData & additional_data = AdditionalData{});

  void
  initialize_impl();

  void
  initialize_restricted_dofs_impl();

  void
  clear();

  void
  compress();

  const AdditionalData &
  get_additional_data() const;

  DoFLayout
  get_dof_layout() const;

  DoFAccessor<dim, dim, dim, true>
  get_level_dof_accessor(const unsigned int cell_position) const;

  DoFAccessor<dim, dim, dim, true>
  get_level_dof_accessor_impl(const unsigned int cell_index, const unsigned int level) const;

  std::vector<types::global_dof_index>
  fill_level_dof_indices(const unsigned int cell_position) const;

  std::vector<types::global_dof_index>
  fill_level_dof_indices_impl(const DoFAccessor<dim, dim, dim, true> & cell) const;

  const DoFHandler<dim> * dof_handler = nullptr;

  /**
   * Stores the starting position of cached dof indices in @p
   * dof_indices_cellwise for each patch local cell stored by @p
   * patch_info as well as the number of dofs. Each element of this array is
   * uniquely associated to patch local cell identified by @p
   * PatchWorker::cell_position()
   */
  std::vector<std::pair<unsigned int, unsigned int>> start_and_number_of_dof_indices_cellwise;

  /**
   * This flat array uniquely caches global dof indices subject to
   * lexicographical for each cell stored in @p patch_info. Global dof indices
   * are stored as proc-local index granting local data access. Actual indices
   * of type types::global_dof_index are obtained by means of the underlying
   * vector partitioner.
   */
  std::vector<unsigned> dof_indices_cellwise;

  /**
   * Stores the starting position of cached dof indices in @p
   * dof_indices_patchwise for each patch stored by @p patch_info. Each element
   * of this array uniquely corresponds to a physical patch, which is identified
   * by its macro patch index and vectorization lane. The lane index runs faster
   * than the macro patch index.
   */
  std::vector<unsigned int> start_of_dof_indices_patchwise;

  /**
   * The array caches the proc-local global dof indices for each macro patch stored in @p
   * patch_info in flat format.
   */
  std::vector<unsigned int> dof_indices_patchwise;

  std::vector<bool> restricted_dof_flags_patchwise;

  const PatchInfo<dim> * patch_info = nullptr;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> * shape_info = nullptr;

  std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner;

  AdditionalData additional_data;

  std::vector<unsigned int> l2h;

  std::vector<unsigned int> n_dofs_on_cell_per_comp;
};



template<int dim, typename Number>
struct DoFInfo<dim, Number>::AdditionalData
{
  unsigned int                 level = numbers::invalid_unsigned_int;
  std::set<types::boundary_id> dirichlet_ids;
  TPSS::CachingStrategy        caching_strategy            = TPSS::CachingStrategy::Cached;
  bool                         force_no_boundary_condition = false;
  bool                         compute_ras_weights         = false;
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
  else if(dof_layout == DoFLayout::DGP)
    return dof_index_1d_dgp_impl(cell_no, cell_dof_index, dimension);
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
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index_1d_dgp_impl(const unsigned int cell_no,
                                                           const unsigned int cell_dof_index,
                                                           const int          dimension) const
{
  AssertIndexRange(cell_no, cell_tensor.n[dimension]);
  AssertIndexRange(cell_dof_index, cell_dof_tensor.n[dimension]);
  AssertIndexRange(dimension, n_dimensions);
  // !!! TODO currently, the actually truncated tensor structure is treated as
  // !!! "full" tensor (i.e. isotropic w.r.t. the polynomial degree)
  // FIX? one could return the actual 1D dof index if it is part of the
  // truncated and otherwhise return numbers::invalid_unsigned_int
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



// -----------------------------   PatchLocalHelperQ   ----------------------------



template<int n_dimensions>
PatchLocalHelperQ<n_dimensions>::PatchLocalHelperQ(
  const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in)
  : cell_dof_tensor(cell_dof_tensor_in)
{
}


template<int n_dimensions>
void
PatchLocalHelperQ<n_dimensions>::reinit(
  const std::vector<FaceInfoLocal<n_dimensions>> & face_infos_in)
{
  clear();
  std::copy(face_infos_in.cbegin(), face_infos_in.cend(), std::back_inserter(face_infos));

  /// Fill for each cell @p cell_no the local cell dof indices on this patch.
  for(auto cell_no = 0U; cell_no < n_cells(); ++cell_no)
  {
    const auto n_cell_dofs = n_dofs_per_cell(cell_no);
    dof_starts_plain.emplace_back(cell_dof_indices_plain.size());
    for(auto cell_dof_index = 0U; cell_dof_index < n_cell_dofs; ++cell_dof_index)
      if(!at_patch_boundary(cell_no, cell_dof_index))
        cell_dof_indices_plain.emplace_back(cell_dof_index);
  }
  dof_starts_plain.emplace_back(cell_dof_indices_plain.size());
}


template<int n_dimensions>
void
PatchLocalHelperQ<n_dimensions>::clear()
{
  face_infos.clear();
  dof_starts_plain.clear();
  cell_dof_indices_plain.clear();
}


template<int n_dimensions>
unsigned int
PatchLocalHelperQ<n_dimensions>::n_cells() const
{
  return face_infos.size();
}


template<int n_dimensions>
bool
PatchLocalHelperQ<n_dimensions>::at_patch_boundary(const unsigned int cell_no,
                                                   const unsigned int cell_dof_index) const
{
  const auto & face_info = face_infos[cell_no];
  if(!cell_dof_tensor.is_edge_index(cell_dof_index))
    return false;
  const auto & edge_numbers     = cell_dof_tensor.get_edge_numbers(cell_dof_index);
  const auto & boundary_numbers = face_info.get_face_numbers_at_patch_boundary();
  for(const auto edge_no : edge_numbers)
    if(std::any_of(boundary_numbers.cbegin(),
                   boundary_numbers.cend(),
                   [edge_no](const auto boundary_no) { return boundary_no == edge_no; }))
      return true;
  return false;
}


template<int n_dimensions>
unsigned int
PatchLocalHelperQ<n_dimensions>::n_dofs_per_cell(const unsigned int cell_no) const
{
  AssertIndexRange(cell_no, n_cells());
  (void)cell_no;
  return cell_dof_tensor.n_flat();
}


template<int n_dimensions>
unsigned int
PatchLocalHelperQ<n_dimensions>::n_dofs_per_cell_on_patch(const unsigned int cell_no) const
{
  const auto dof_start     = get_dof_start_plain(cell_no);
  const auto dof_start_end = get_dof_start_plain(cell_no + 1);
  return dof_start_end - dof_start;
}


template<int n_dimensions>
ArrayView<const unsigned int>
PatchLocalHelperQ<n_dimensions>::get_cell_dof_indices(const unsigned int cell_no) const
{
  const auto start = get_dof_start_plain(cell_no);
  const auto end   = get_dof_start_plain(cell_no + 1);
  AssertIndexRange(end - 1, cell_dof_indices_plain.size());
  return make_array_view<const unsigned int>(cell_dof_indices_plain.data() + start,
                                             cell_dof_indices_plain.data() + end);
}


template<int n_dimensions>
unsigned int
PatchLocalHelperQ<n_dimensions>::get_dof_start_plain(const unsigned int cell_no) const
{
  AssertIndexRange(cell_no, dof_starts_plain.size());
  AssertDimension(dof_starts_plain.size(), n_cells() + 1);
  return dof_starts_plain[cell_no];
}



// -----------------------------   DoFInfo   ----------------------------



template<int dim, typename Number>
inline void
DoFInfo<dim, Number>::clear()
{
  patch_info = nullptr;
  start_and_number_of_dof_indices_cellwise.clear();
  // global_dof_indices_cellwise.clear();
  dof_indices_cellwise.clear();
  start_of_dof_indices_patchwise.clear();
  dof_indices_patchwise.clear();
  dof_handler     = nullptr;
  additional_data = AdditionalData{};
  l2h.clear();
  n_dofs_on_cell_per_comp.clear();
}


template<int dim, typename Number>
inline void
DoFInfo<dim, Number>::compress()
{
  if(additional_data.caching_strategy == TPSS::CachingStrategy::Cached)
  {
    start_and_number_of_dof_indices_cellwise.clear();
    // global_dof_indices_cellwise.clear();
    dof_indices_cellwise.clear();
  }
}


template<int dim, typename Number>
inline const typename DoFInfo<dim, Number>::AdditionalData &
DoFInfo<dim, Number>::get_additional_data() const
{
  return additional_data;
}


template<int dim, typename Number>
inline DoFLayout
DoFInfo<dim, Number>::get_dof_layout() const
{
  Assert(dof_handler, ExcMessage("DoF handler not initialized."));
  return TPSS::get_dof_layout(dof_handler->get_fe());
}


template<int dim, typename Number>
inline DoFAccessor<dim, dim, dim, true>
DoFInfo<dim, Number>::get_level_dof_accessor(const unsigned int cell_position) const
{
  Assert(patch_info, ExcMessage("Patch info not initialized."));
  const auto [cell_level, cell_index] = patch_info->get_cell_level_and_index(cell_position);
  return get_level_dof_accessor_impl(cell_index, cell_level);
}


template<int dim, typename Number>
inline DoFAccessor<dim, dim, dim, true>
DoFInfo<dim, Number>::get_level_dof_accessor_impl(const unsigned int cell_index,
                                                  const unsigned int level) const
{
  const auto & tria = dof_handler->get_triangulation();
  return DoFAccessor<dim, dim, dim, true>(&tria, level, cell_index, dof_handler);
}


template<int dim, typename Number>
inline std::vector<types::global_dof_index>
DoFInfo<dim, Number>::fill_level_dof_indices(const unsigned int cell_position) const
{
  const auto & cell = get_level_dof_accessor(cell_position);
  return fill_level_dof_indices_impl(cell);
}


template<int dim, typename Number>
inline std::vector<types::global_dof_index>
DoFInfo<dim, Number>::fill_level_dof_indices_impl(
  const DoFAccessor<dim, dim, dim, true> & cell) const
{
  const auto                           n_dofs_per_cell = dof_handler->get_fe().n_dofs_per_cell();
  std::vector<types::global_dof_index> level_dof_indices(n_dofs_per_cell);
  cell.get_mg_dof_indices(cell.level(), level_dof_indices);

  /// Reorder level dof indices lexicographically within each component and
  /// concatenate dof indices per component.
  AssertDimension(level_dof_indices.size(), l2h.size());
  std::vector<types::global_dof_index> level_dof_indices_lxco;
  std::transform(l2h.cbegin(),
                 l2h.cend(),
                 std::back_inserter(level_dof_indices_lxco),
                 [&](const auto & h) { return level_dof_indices[h]; });
  return level_dof_indices_lxco;
}



} // end namespace TPSS

#include "dof_info.templates.h"

#endif // end inclusion guard
