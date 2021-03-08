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
   * Returns the number of patch-local dofs located at cell @p cell_no.
   */
  unsigned int
  n_dofs_per_cell_on_patch(const unsigned int cell_no) const;

  const Tensors::TensorHelper<n_dimensions> cell_dof_tensor;

  std::vector<FaceInfoLocal<n_dimensions>> face_infos;
  std::vector<unsigned int>                dof_starts_plain;
  std::vector<unsigned int>                cell_dof_indices_plain;
};



/**
 * Helps with the d-dimensional plain patch-local dof indexing exploiting the
 * tensor structure of the finite element and patch of cells. "Plain" means that
 * patch-local constraints are not taken into account.
 *
 * For example, conforming finite elements are subject to a certain degree of
 * inter-element continuity, thus their degrees of freedom might belong to more
 * than one cell, which affects the patch-local dof indexing.
 */
template<int n_dimensions>
class PatchLocalIndexHelper : public Tensors::TensorHelper<n_dimensions>
{
  using Base = Tensors::TensorHelper<n_dimensions>;

public:
  /// TODO description...
  PatchLocalIndexHelper(const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
                        const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
                        const DoFLayout                             dof_layout_in);

  /**
   * Returns the one-dimensional patch-local dof index associated with
   * cell-local one-dimensional dof index @p cell_dof_index_1d on the cell with
   * patch-local one-dimensional numbering @p cell_no_1d in spatial dimension @p
   * dimension. Patch-local cell and cell-local dof index numbering are as well
   * subject to lexicographical ordering such that @p cell_no_1d and @p
   * cell_dof_index_1d are well-defined.
   *
   * For DGQ-like finite elements this mapping is bijective. For Q-like finite
   * elements this mapping is surjective but not injective. For RT-like finite
   * elements the same holds and we have to take the vector component into
   * account. For DGP-like finite elements this mapping is incorrect but has to
   * be provided to satisfy TPSS interfaces.
   */
  unsigned int
  dof_index_1d(const unsigned int cell_no_1d,
               const unsigned int cell_dof_index_1d,
               const int          dimension) const;

  /**
   * Same as above but this time the patch-local dof index is returned as
   * multi-index: for each dimension the one-dimensional index is given by @p
   * dof_index_1d() (therefore one-dimensional patch-local cell and cell-local
   * dof index numberings are determined by the underlying cell and cell dof
   * tensor.
   */
  std::array<unsigned int, n_dimensions>
  dof_multi_index(const unsigned int cell_no, const unsigned int cell_dof_index) const;

  /**
   * Similar as above but here the patch-local dof index subject to
   * lexicographical ordering (the first dimension runs faster than the second
   * and so on) is returned: given the tensor structure of dof indices on the
   * plain patch this univariate index is one-to-one with respect to the
   * multi-index given by @p dof_multi_index().
   */
  unsigned int
  dof_index(const unsigned int cell_no, const unsigned int cell_dof_index) const;

  std::pair<unsigned int, unsigned int>
  cell_dof_range_1d(const unsigned int cell_no_1d, const unsigned int dimension) const;

  /// DEPRECATED: use cell_tensor.size(dimension) instead...
  // unsigned int
  // n_cells_1d(const unsigned int dimension) const;

  /// DEPRECATED: use cell_dof_tensor.size(dimension) instead...
  // unsigned int
  // n_dofs_per_cell_1d(const unsigned int dimension) const;

  const Tensors::TensorHelper<n_dimensions> cell_tensor;

  const Tensors::TensorHelper<n_dimensions> cell_dof_tensor;

  const DoFLayout dof_layout;

  const unsigned int dimension_with_high_degree;
};



/**
 * Helps with the d-dimensional patch-local dof indexing taking the tensor
 * structure of the finite element and patch-local constraints into account.
 *
 * See also PatchLocalIndexHelper, in particular for the understanding of
 * "plain" dof indices.
 */
template<int n_dimensions>
struct PatchLocalTensorHelper : public Tensors::TensorHelper<n_dimensions>
{
  using Base = Tensors::TensorHelper<n_dimensions>;

  PatchLocalTensorHelper(const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
                         const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
                         const DoFLayout                             dof_layout_in);

  /**
   * Determines whether the plain patch-local dof index @p plain_patch_dof_index is
   * part of the outermost layer of dofs (also referred to as edge or hyperface
   * in analogy to TensorHelper).
   */
  bool
  is_plain_edge_dof(const unsigned int plain_patch_dof_index) const;

  /// DEPRECATED use is_plain_edge_dof()
  // bool
  // is_boundary_face_dof_1d(const unsigned int patch_dof_index, const unsigned int dimension)
  // const;

  /**
   * Determines whether the one-dimensional plain dof index @p
   * plain_patch_dof_index_1d is constrained.
   */
  bool
  is_constrained_1d(const unsigned int plain_dof_index_1d, const unsigned int dimension) const;

  /**
   * Determines whether any one-dimensional index of the multi-index associated
   * with plain dof index @p plain_patch_dof_index is constrained.
   */
  bool
  is_constrained(const unsigned int plain_dof_index) const;

  bool
  has_constraints() const;


  /**
   * Returns the number of degrees of freedom seen from dimension @p dimension
   * after subtracting constrained dofs. This equals calling @p size(dimension)
   * from the underlying TensorHelper base.
   */
  unsigned int
  n_dofs_1d(const unsigned int dimension) const;

  /**
   * Returns the number of plain degrees of freedom seen from dimension @p
   * dimension, that is taking also constrained dofs into account. This equals
   * calling @p plain.size(dimension).
   */
  unsigned int
  n_plain_dofs_1d(const unsigned int dimension) const;

  /**
   * Returns the plain patch-local dof index subject to lexicographical ordering
   * (that is plain.dof_index_1d()), if it is unconstrained. Otherwise
   * numbers::invalid_unsigned_int is returned.
   */
  unsigned int
  plain_dof_index_if(const unsigned int cell_no, const unsigned int cell_dof_index) const;

  unsigned int
  plain_dof_index_if_1d(const unsigned int cell_no_1d, const unsigned int cell_dof_index_1d) const;

  /**
   * A "plain" square one-dimensional matrix @p subdomain_matrix representing
   * the univariate discretization on a subdomain with tensor structure is
   * passed. Then, all rows and columns associated with constrained dofs of
   * dimension @p dimension are deleted and the "constrained" matrix is returned
   * by reference @p subdomain_matrix. In other words, the matrix @p
   * subdomain_matrix passed follows the tensor structure of member @p plain and
   * the matrix returned by reference @p subdomain_matrix follows the tensor
   * structure of the base class TensorHelper.
   */
  template<typename Number>
  void
  apply_constraints(Table<2, Number> & subdomain_matrix, const unsigned int dimension) const;

  /**
   * Same as above, but for rectangular matrices, where this class is
   * responsible for rows and PatchLocalTensorHelper @p other_column for the
   * columns.
   */
  template<typename Number>
  void
  apply_constraints(Table<2, Number> &             subdomain_matrix,
                    const unsigned int             dimension,
                    const PatchLocalTensorHelper & other_column) const;

  /**
   * Submits the entries of one-dimensional matrix @p cell_matrix into the
   * "plain" one-dimensional matrix @p subdomain_matrix for dimension @p
   * dimension. Both matrices are assumed to be quadratic, this means that
   * univariate ansatz and test space coincide. For instance, different cell
   * numbers occur if we compute (DG) interface contributions for ansatz
   * functions on cell/interval with identifier @p cell_no_1d_row and test
   * functions on cell/interval with identifier @p cell_no_1d_col.
   */
  template<typename Number>
  void
  submit_cell_matrix_plain(Table<2, Number> &       subdomain_matrix,
                           const Table<2, Number> & cell_matrix,
                           const unsigned int       cell_no_1d_row,
                           const unsigned int       cell_no_1d_col,
                           const unsigned int       dimension) const;

  /**
   * Same as above, but matrices are not assumed to be square: actually this
   * PatchLocaLTensorHelper handles univariate test functions (columns) and the
   * PatchLocalTensorHelper @p ansatz univariate ansatz functions (rows).
   */
  template<typename Number>
  void
  submit_cell_matrix_plain(Table<2, Number> &             subdomain_matrix,
                           const Table<2, Number> &       cell_matrix,
                           const unsigned int             cell_no_1d_row,
                           const unsigned int             cell_no_1d_col,
                           const unsigned int             dimension,
                           const PatchLocalTensorHelper & ansatz) const;

  const Tensors::TensorHelper<n_dimensions> &
  get_cell_tensor() const;

  const Tensors::TensorHelper<n_dimensions> &
  get_cell_dof_tensor() const;

  template<typename Number>
  void
  submit_cell_matrix_plain_impl(Table<2, Number> &             subdomain_matrix,
                                const Table<2, Number> &       cell_matrix,
                                const unsigned int             cell_no_1d_row,
                                const unsigned int             cell_no_1d_col,
                                const unsigned int             dimension,
                                const PatchLocalTensorHelper & ansatz) const;

  template<typename Number>
  void
  apply_constraints_impl(Table<2, Number> &             subdomain_matrix,
                         const unsigned int             dimension,
                         const PatchLocalTensorHelper & other_column) const;

  const PatchLocalIndexHelper<n_dimensions> plain;

  const std::array<std::set<unsigned int>, n_dimensions> constrained_dof_mask;

  const std::array<std::vector<unsigned int>, n_dimensions> map_plain_dof_index_from;
};



/**
 * TODO description missing...
 */
template<int n_dimensions>
struct PatchLocalTensorIndices
{
  using Base = Tensors::TensorHelper<n_dimensions>;

  PatchLocalTensorIndices(
    const Tensors::TensorHelper<n_dimensions> &              cell_tensor_in,
    const std::vector<Tensors::TensorHelper<n_dimensions>> & cell_dof_tensors_in,
    const DoFLayout                                          dof_layout_in);

  /**
   * The set of dof indices @p plain_dof_indices subject to a lexicographical
   * order from the plain tensor dof structure, i.e. including all constrained
   * dofs, is restricted to the set of unconstrained dof indices (returned as
   * reference @p plain_dof_indices). For vector-valued finite elements dof
   * indices for each component are subject to lexicographical order. Indices
   * are juxtaposed in @p plain_dof_indices component by component.
   */
  template<typename DoFIndexType>
  void
  apply_constraints(std::vector<DoFIndexType> & plain_dof_indices) const;

  bool
  has_constraints() const;

  const PatchLocalTensorHelper<n_dimensions> &
  get_dof_tensor(unsigned int component) const;

  const unsigned int n_components;

  const std::vector<PatchLocalTensorHelper<n_dimensions>> dof_tensors;

  const unsigned int n_dofs;

  const unsigned int n_plain_dofs;

  const std::vector<unsigned int> n_preceding_plain_dofs;

  const std::vector<unsigned int> n_dofs_per_cell;

  const std::vector<unsigned int> n_preceding_dofs_per_cell;
};



/**
 * TODO description missing ...
 */
// !!! TODO do not pass level as additional data, directly use the level of patch_info_in
template<int dim, typename Number>
struct DoFInfo
{
  static constexpr unsigned int macro_size = VectorizedArray<Number>::size();

  using cell_iterator_type = typename PatchInfo<dim>::dof_cell_iterator_type;

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
  compute_restricted_dofs_impl();

  void
  clear();

  void
  compress();

  const AdditionalData &
  get_additional_data() const;

  DoFLayout
  get_dof_layout() const;

  /// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
  /// NOTE difference between 'accessor' and 'iterator'
  DoFAccessor<dim, dim, dim, true>
  get_level_dof_accessor(const unsigned int cell_position) const;

  /// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
  /// NOTE difference between 'accessor' and 'iterator'
  DoFAccessor<dim, dim, dim, true>
  get_level_dof_accessor_impl(const unsigned int cell_index, const unsigned int level) const;

  /// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
  /// NOTE difference between 'accessor' and 'iterator'
  std::vector<types::global_dof_index>
  fill_level_dof_indices(const unsigned int cell_position) const;

  /// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
  /// NOTE difference between 'accessor' and 'iterator'
  std::vector<types::global_dof_index>
  fill_level_dof_indices_impl(const DoFAccessor<dim, dim, dim, true> & cell) const;

  const DoFHandler<dim> * dof_handler = nullptr;

  DoFLayout dof_layout = DoFLayout::invalid;

  /**
   * Stores the starting position of cached dof indices in @p
   * dof_indices_cellwise for each patch-local cell stored by @p
   * patch_info as well as the number of dofs. Each element of this array is
   * uniquely associated to patch-local cell identified by @p
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

  std::shared_ptr<const PatchLocalTensorIndices<dim>> patch_dof_tensors;
};



template<int dim, typename Number>
struct DoFInfo<dim, Number>::AdditionalData
{
  unsigned int                 level = numbers::invalid_unsigned_int;
  std::set<types::boundary_id> dirichlet_ids;
  TPSS::CachingStrategy        caching_strategy            = TPSS::CachingStrategy::Cached;
  bool                         force_no_boundary_condition = false;
  bool                         compute_ras_boolean_weights = false;
};



// -----------------------------   PatchLocalIndexHelper   ----------------------------



template<int dim>
unsigned int
dof_index_1d_discontinuous(const Tensors::TensorHelper<dim> & cell_tensor,
                           const Tensors::TensorHelper<dim> & cell_dof_tensor,
                           const unsigned int &               cell_no_1d,
                           const unsigned int &               cell_dof_index_1d,
                           const int &                        dimension)
{
  (void)cell_tensor;
  AssertIndexRange(cell_no_1d, cell_tensor.n[dimension]);
  AssertIndexRange(cell_dof_index_1d, cell_dof_tensor.n[dimension]);
  AssertIndexRange(dimension, dim);
  const auto & n_dofs_per_cell_1d = cell_dof_tensor.n[dimension];
  return cell_no_1d * n_dofs_per_cell_1d + cell_dof_index_1d;
}



template<int dim>
unsigned int
dof_index_1d_continuous(const Tensors::TensorHelper<dim> & cell_tensor,
                        const Tensors::TensorHelper<dim> & cell_dof_tensor,
                        const unsigned int &               cell_no_1d,
                        const unsigned int &               cell_dof_index_1d,
                        const int &                        dimension)
{
  (void)cell_tensor;
  AssertIndexRange(cell_no_1d, cell_tensor.n[dimension]);
  AssertIndexRange(cell_dof_index_1d, cell_dof_tensor.n[dimension]);
  AssertIndexRange(dimension, dim);
  const auto & n_dofs_per_cell_1d = cell_dof_tensor.n[dimension];
  return cell_no_1d * n_dofs_per_cell_1d + cell_dof_index_1d - cell_no_1d;
}



template<int dim>
inline unsigned int
dof_index_1d_impl(const Tensors::TensorHelper<dim> & cell_tensor,
                  const Tensors::TensorHelper<dim> & cell_dof_tensor,
                  const DoFLayout                    dof_layout,
                  const unsigned int                 cell_no_1d,
                  const unsigned int                 cell_dof_index_1d,
                  const int                          dimension,
                  const int                          dimension_high = -1)
{
  if(dof_layout == DoFLayout::DGQ)
    return dof_index_1d_discontinuous(
      cell_tensor, cell_dof_tensor, cell_no_1d, cell_dof_index_1d, dimension);
  else if(dof_layout == DoFLayout::Q)
    return dof_index_1d_continuous(
      cell_tensor, cell_dof_tensor, cell_no_1d, cell_dof_index_1d, dimension);
  else if(dof_layout == DoFLayout::DGP)
    // !!! TODO currently, the truncated tensor structure is treated as
    // !!! "full" tensor (i.e. isotropic w.r.t. the polynomial degree)
    // FIX? one could return the actual 1D dof index if it is part of the
    // truncated and otherwhise return numbers::invalid_unsigned_int
    return dof_index_1d_discontinuous(
      cell_tensor, cell_dof_tensor, cell_no_1d, cell_dof_index_1d, dimension);
  else if(dof_layout == DoFLayout::RT)
  {
    if(dimension == dimension_high)
      return dof_index_1d_continuous(
        cell_tensor, cell_dof_tensor, cell_no_1d, cell_dof_index_1d, dimension);
    else
      return dof_index_1d_discontinuous(
        cell_tensor, cell_dof_tensor, cell_no_1d, cell_dof_index_1d, dimension);
  }

  AssertThrow(false, ExcMessage("Finite element not supported."));
  return numbers::invalid_unsigned_int;
}



template<int n_dimensions>
inline PatchLocalIndexHelper<n_dimensions>::PatchLocalIndexHelper(
  const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
  const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
  const DoFLayout                             dof_layout_in)
  : Base([&]() {
      Assert(dof_layout_in != DoFLayout::invalid, ExcMessage("Not supported."));
      /// double-check Raviart-Thomas tensor
      if(dof_layout_in == DoFLayout::RT)
        for(auto d = 0U; d < n_dimensions; ++d)
          if(d != cell_dof_tensor_in.max_dimension())
            AssertDimension(cell_dof_tensor_in.size(d) + 1, cell_dof_tensor_in.max_size());

      std::array<unsigned int, n_dimensions> sizes;
      for(auto d = 0U; d < n_dimensions; ++d)
      {
        const auto last_cell_no         = cell_tensor_in.size(d) - 1;
        const auto last_cell_dof_index  = cell_dof_tensor_in.size(d) - 1;
        const auto last_patch_dof_index = dof_index_1d_impl(cell_tensor_in,
                                                            cell_dof_tensor_in,
                                                            dof_layout_in,
                                                            last_cell_no,
                                                            last_cell_dof_index,
                                                            d,
                                                            cell_dof_tensor_in.max_dimension());
        sizes[d]                        = last_patch_dof_index + 1;
      }
      return sizes;
    }()),
    cell_tensor(cell_tensor_in),
    cell_dof_tensor(cell_dof_tensor_in),
    dof_layout(dof_layout_in),
    dimension_with_high_degree(cell_dof_tensor_in.max_dimension())
{
}



template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index_1d(const unsigned int cell_no_1d,
                                                  const unsigned int cell_dof_index_1d,
                                                  const int          dimension) const
{
  return dof_index_1d_impl(cell_tensor,
                           cell_dof_tensor,
                           dof_layout,
                           cell_no_1d,
                           cell_dof_index_1d,
                           dimension,
                           dimension_with_high_degree);
}



template<int n_dimensions>
inline std::array<unsigned int, n_dimensions>
PatchLocalIndexHelper<n_dimensions>::dof_multi_index(const unsigned int cell_no,
                                                     const unsigned int cell_dof_index) const
{
  const auto & cell_no_multi        = cell_tensor.multi_index(cell_no);
  const auto & cell_dof_index_multi = cell_dof_tensor.multi_index(cell_dof_index);
  std::array<unsigned int, n_dimensions> patch_dof_index_multi;
  for(auto d = 0U; d < n_dimensions; ++d)
    patch_dof_index_multi[d] = dof_index_1d(cell_no_multi[d], cell_dof_index_multi[d], d);
  return patch_dof_index_multi;
}



template<int n_dimensions>
inline unsigned int
PatchLocalIndexHelper<n_dimensions>::dof_index(const unsigned int cell_no,
                                               const unsigned int cell_dof_index) const
{
  const auto & patch_dof_index_multi = dof_multi_index(cell_no, cell_dof_index);
  return Base::uni_index(patch_dof_index_multi);
}



template<int n_dimensions>
inline std::pair<unsigned int, unsigned int>
PatchLocalIndexHelper<n_dimensions>::cell_dof_range_1d(const unsigned int cell_no_1d,
                                                       const unsigned int dimension) const
{
  AssertIndexRange(cell_no_1d, cell_tensor.n[dimension]);
  AssertIndexRange(dimension, n_dimensions);
  const auto         last_cell_dof_index_1d = cell_dof_tensor.size(dimension) - 1;
  const unsigned int first                  = dof_index_1d(cell_no_1d, 0, dimension);
  const unsigned int last = dof_index_1d(cell_no_1d, last_cell_dof_index_1d, dimension);
  return {first, last + 1};
}



// template<int n_dimensions>
// inline unsigned int
// PatchLocalIndexHelper<n_dimensions>::n_cells_1d(const unsigned int dimension) const
// {
//   AssertIndexRange(dimension, n_dimensions);
//   return this->cell_tensor.size(dimension);
// }



// template<int n_dimensions>
// inline unsigned int
// PatchLocalIndexHelper<n_dimensions>::n_dofs_per_cell_1d(const unsigned int dimension) const
// {
//   AssertIndexRange(dimension, n_dimensions);
//   return this->cell_dof_tensor.size(dimension);
// }



// -----------------------------   PatchLocalTensorHelper   ----------------------------



template<int dim>
std::array<std::set<unsigned int>, dim>
make_constrained_dof_mask_impl(const Tensors::TensorHelper<dim> &  plain_dof_tensor,
                               const Table<2, ConstraintVariant> & constrained_face_mask)
{
  AssertDimension(constrained_face_mask.size(0), dim);
  AssertDimension(constrained_face_mask.size(1), 2U);
  std::array<std::set<unsigned int>, dim> constrained_dof_mask;
  for(auto d = 0; d < dim; ++d)
  {
    if(constrained_face_mask(d, 0) == ConstraintVariant::Dirichlet)
      constrained_dof_mask[d].emplace(plain_dof_tensor.first_index(d));
    if(constrained_face_mask(d, 1) == ConstraintVariant::Dirichlet)
      constrained_dof_mask[d].emplace(plain_dof_tensor.last_index(d));
  }
  return constrained_dof_mask;
}



template<int n_dimensions>
inline PatchLocalTensorHelper<n_dimensions>::PatchLocalTensorHelper(
  const Tensors::TensorHelper<n_dimensions> & cell_tensor_in,
  const Tensors::TensorHelper<n_dimensions> & cell_dof_tensor_in,
  const DoFLayout                             dof_layout_in)
  : Base([&]() {
      const PatchLocalIndexHelper<n_dimensions> tmp_plain(cell_tensor_in,
                                                          cell_dof_tensor_in,
                                                          dof_layout_in);
      const auto tmp_constrained_dof_mask = make_constrained_dof_mask_impl<n_dimensions>(
        tmp_plain,
        make_constrained_hyperface_mask<n_dimensions>(dof_layout_in,
                                                      tmp_plain.dimension_with_high_degree));
      std::array<unsigned int, n_dimensions> sizes;
      for(auto d = 0U; d < n_dimensions; ++d)
        sizes[d] = tmp_plain.size(d) - tmp_constrained_dof_mask[d].size();
      return sizes;
    }()),
    plain(cell_tensor_in, cell_dof_tensor_in, dof_layout_in),
    constrained_dof_mask(make_constrained_dof_mask_impl<n_dimensions>(
      plain,
      make_constrained_hyperface_mask<n_dimensions>(dof_layout_in,
                                                    plain.dimension_with_high_degree))),
    map_plain_dof_index_from([&]() {
      std::array<std::vector<unsigned int>, n_dimensions> map;
      for(auto d = 0U; d < n_dimensions; ++d)
        for(auto i = 0U; i < plain.size(d); ++i)
          if(!(this->is_constrained_1d(i, d)))
            map[d].emplace_back(i);
      for(auto d = 0U; d < n_dimensions; ++d)
        AssertDimension(map[d].size(), this->size(d));
      return map;
    }())
{
  const bool is_single_cell = cell_tensor_in.n_flat() == 1U;
  if(is_single_cell)
    AssertThrow(dof_layout_in == DoFLayout::DGQ || dof_layout_in == DoFLayout::DGP,
                ExcMessage("A single cell patch is not supported for this dof layout..."));
}



template<int n_dimensions>
inline bool
PatchLocalTensorHelper<n_dimensions>::is_plain_edge_dof(const unsigned int plain_dof_index) const
{
  return plain.is_edge_index(plain_dof_index);
}



template<int n_dimensions>
inline bool
PatchLocalTensorHelper<n_dimensions>::is_constrained_1d(const unsigned int plain_dof_index_1d,
                                                        const unsigned int dimension) const
{
  AssertIndexRange(plain_dof_index_1d, plain.size(dimension));
  const auto & this_mask_1d = constrained_dof_mask[dimension];
  return this_mask_1d.cend() != this_mask_1d.find(plain_dof_index_1d);
}



template<int n_dimensions>
inline bool
PatchLocalTensorHelper<n_dimensions>::is_constrained(const unsigned int plain_dof_index) const
{
  const auto & plain_dof_index_multi = plain.multi_index(plain_dof_index);
  for(auto d = 0; d < n_dimensions; ++d)
    if(is_constrained_1d(plain_dof_index_multi[d], d))
      return true;
  return false;
}



template<int n_dimensions>
inline bool
PatchLocalTensorHelper<n_dimensions>::has_constraints() const
{
  const auto n_constraints =
    std::accumulate(constrained_dof_mask.cbegin(),
                    constrained_dof_mask.cend(),
                    0U,
                    [&](const unsigned int n, const auto & set) { return n + set.size(); });
  return n_constraints > 0;
}



template<int n_dimensions>
inline unsigned int
PatchLocalTensorHelper<n_dimensions>::plain_dof_index_if_1d(
  const unsigned int cell_no_1d,
  const unsigned int cell_dof_index_1d) const
{
  const auto & plain_dof_index_1d = plain.dof_index_1d(cell_no_1d, cell_dof_index_1d);
  const bool   is_constrained     = this->is_constrained_1d(plain_dof_index_1d);
  return is_constrained ? numbers::invalid_unsigned_int : plain_dof_index_1d;
}



template<int n_dimensions>
inline unsigned int
PatchLocalTensorHelper<n_dimensions>::plain_dof_index_if(const unsigned int cell_no,
                                                         const unsigned int cell_dof_index) const
{
  const auto & plain_dof_index = plain.dof_index(cell_no, cell_dof_index);
  const bool   is_constrained  = this->is_constrained(plain_dof_index);
  return is_constrained ? numbers::invalid_unsigned_int : plain_dof_index;
}



template<int n_dimensions>
inline unsigned int
PatchLocalTensorHelper<n_dimensions>::n_dofs_1d(const unsigned int dimension) const
{
  Assert(plain.dof_layout != DoFLayout::DGP,
         ExcMessage("Not supported as long as truncated tensor are not handled..."));
  return Base::size(dimension);
}



template<int n_dimensions>
inline unsigned int
PatchLocalTensorHelper<n_dimensions>::n_plain_dofs_1d(const unsigned int dimension) const
{
  Assert(plain.dof_layout != DoFLayout::DGP,
         ExcMessage("Not supported as long as truncated tensor are not handled..."));
  return plain.size(dimension);
}



template<int n_dimensions>
inline const Tensors::TensorHelper<n_dimensions> &
PatchLocalTensorHelper<n_dimensions>::get_cell_tensor() const
{
  return plain.cell_tensor;
}



template<int n_dimensions>
inline const Tensors::TensorHelper<n_dimensions> &
PatchLocalTensorHelper<n_dimensions>::get_cell_dof_tensor() const
{
  return plain.cell_dof_tensor;
}



// -----------------------------   PatchLocalTensorIndices   ----------------------------



template<int n_dimensions>
PatchLocalTensorIndices<n_dimensions>::PatchLocalTensorIndices(
  const Tensors::TensorHelper<n_dimensions> &              cell_tensor_in,
  const std::vector<Tensors::TensorHelper<n_dimensions>> & cell_dof_tensors_in,
  const DoFLayout                                          dof_layout_in)
  : n_components(cell_dof_tensors_in.size()),
    dof_tensors(std::move([&]() {
      Assert(!cell_dof_tensors_in.empty(), ExcMessage("Pass at least one cell dof tensor!"));
      std::vector<PatchLocalTensorHelper<n_dimensions>> dof_tensors_in;
      for(const auto & cell_dof_tensor_in : cell_dof_tensors_in)
        dof_tensors_in.emplace_back(cell_tensor_in, cell_dof_tensor_in, dof_layout_in);
      return dof_tensors_in;
    }())),
    n_dofs([&]() {
      return std::accumulate(dof_tensors.cbegin(),
                             dof_tensors.cend(),
                             0U,
                             [](const unsigned int sum, const auto & dof_tensor) {
                               return sum + dof_tensor.n_flat();
                             });
    }()),
    n_plain_dofs([&]() {
      return std::accumulate(dof_tensors.cbegin(),
                             dof_tensors.cend(),
                             0U,
                             [](const unsigned int sum, const auto & dof_tensor) {
                               return sum + dof_tensor.plain.n_flat();
                             });
    }()),
    n_preceding_plain_dofs([&]() {
      std::vector<unsigned int> n_preceding_plain_dofs_in;
      n_preceding_plain_dofs_in.emplace_back(0U);
      for(auto comp = 1U; comp < n_components; ++comp)
        n_preceding_plain_dofs_in.emplace_back(n_preceding_plain_dofs_in.back() +
                                               dof_tensors[comp - 1].plain.n_flat());
      return n_preceding_plain_dofs_in;
    }()),
    n_dofs_per_cell([&]() {
      std::vector<unsigned int> n_dofs_per_cell_in;
      for(const auto & dof_tensor : dof_tensors)
        n_dofs_per_cell_in.emplace_back(dof_tensor.get_cell_dof_tensor().n_flat());
      return n_dofs_per_cell_in;
    }()),
    n_preceding_dofs_per_cell([&]() {
      std::vector<unsigned int> n_preceding_dofs_per_cell_in;
      n_preceding_dofs_per_cell_in.emplace_back(0U);
      for(auto comp = 1U; comp < n_components; ++comp)
        n_preceding_dofs_per_cell_in.emplace_back(n_preceding_dofs_per_cell_in.back() +
                                                  n_dofs_per_cell[comp - 1]);
      return n_preceding_dofs_per_cell_in;
    }())
{
  if(dof_layout_in == DoFLayout::DGP || dof_layout_in == DoFLayout::DGQ)
    Assert(!has_constraints(), ExcMessage("Not implemented..."));

  AssertDimension(n_components, dof_tensors.size());
  AssertDimension(n_components, n_preceding_plain_dofs.size());
  AssertDimension(n_components, n_dofs_per_cell.size());
  AssertDimension(n_components, n_preceding_dofs_per_cell.size());

  AssertDimension(n_preceding_plain_dofs.back() + dof_tensors.back().plain.n_flat(), n_plain_dofs);
  AssertDimension(n_preceding_dofs_per_cell.back() + n_dofs_per_cell.back(),
                  std::accumulate(n_dofs_per_cell.cbegin(), n_dofs_per_cell.cend(), 0U));
}



template<int n_dimensions>
inline bool
PatchLocalTensorIndices<n_dimensions>::has_constraints() const
{
  return std::any_of(dof_tensors.cbegin(), dof_tensors.cend(), [&](const auto & dof_tensor) {
    return dof_tensor.has_constraints();
  });
}



template<int n_dimensions>
inline const PatchLocalTensorHelper<n_dimensions> &
PatchLocalTensorIndices<n_dimensions>::get_dof_tensor(unsigned int component) const
{
  AssertIndexRange(component, n_components);
  return dof_tensors[component];
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
  dof_layout      = DoFLayout::invalid;
  additional_data = AdditionalData{};
  l2h.clear();
  patch_dof_tensors.reset();
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
  Assert(dof_layout != DoFLayout::invalid, ExcMessage("Not a valid dof layout."));
  // return TPSS::get_dof_layout(dof_handler->get_fe());
  return dof_layout;
}


/// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
/// NOTE difference between 'accessor' and 'iterator'
template<int dim, typename Number>
inline DoFAccessor<dim, dim, dim, true>
DoFInfo<dim, Number>::get_level_dof_accessor(const unsigned int cell_position) const
{
  Assert(patch_info, ExcMessage("Patch info not initialized."));
  const auto [cell_level, cell_index] = patch_info->get_cell_level_and_index(cell_position);
  return get_level_dof_accessor_impl(cell_index, cell_level);
}


/// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
/// NOTE difference between 'accessor' and 'iterator'
template<int dim, typename Number>
inline DoFAccessor<dim, dim, dim, true>
DoFInfo<dim, Number>::get_level_dof_accessor_impl(const unsigned int cell_index,
                                                  const unsigned int level) const
{
  const auto & tria = dof_handler->get_triangulation();
  return DoFAccessor<dim, dim, dim, true>(&tria, level, cell_index, dof_handler);
}


/// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
/// NOTE difference between 'accessor' and 'iterator'
template<int dim, typename Number>
inline std::vector<types::global_dof_index>
DoFInfo<dim, Number>::fill_level_dof_indices(const unsigned int cell_position) const
{
  const auto & cell = get_level_dof_accessor(cell_position);
  return fill_level_dof_indices_impl(cell);
}


/// TODO replace DoFAccessor<dim, dim, dim, true> by cell_iterator_type
/// NOTE difference between 'accessor' and 'iterator'
template<int dim, typename Number>
inline std::vector<types::global_dof_index>
DoFInfo<dim, Number>::fill_level_dof_indices_impl(
  const DoFAccessor<dim, dim, dim, true> & cell) const
{
  const auto                           n_dofs_per_cell = dof_handler->get_fe().n_dofs_per_cell();
  std::vector<types::global_dof_index> level_dof_indices(n_dofs_per_cell);
  cell.get_mg_dof_indices(cell.level(), level_dof_indices);

  /// We assume here that l2h provides the renumbering from a cell-local
  /// lexicographical order to deal.II's hierarchical order for each vector
  /// component of the finite element.
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
