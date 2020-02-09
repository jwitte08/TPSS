#ifndef PATCHTRANSFER_H
#define PATCHTRANSFER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_tools.h>

#include "TPSS.h"
#include "subdomain_handler.h"
#include "tensors.h"

#include <array>
#include <memory>

using namespace dealii;

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

/**
 * Transfer class modeling the (mathematical) restriction operator from global
 * degrees of freedom onto subdomain degrees of freedom and its transpose
 * operator, namely the prolongation operator from subdomain to global space.
 * The restricted set of subdomain degrees of freedom is ordered
 * lexicographically.
 *
 * Mappings between cell local dof indices (subject to lexicographical ordering)
 * for each cell within a patch and the patch relevant dof indices are required.
 * Obviously, for "DGQ cell patches" it is the identity. For DGQ vertex patches
 * we have to take into account that each patch has 2 cells in each spatial
 * coordinate. For Q vertex patches we additionally have to treat the dofs on
 * (logical) cell boundaries belonging to more than one cell.
 *
 * Nevertheless, we restrict ourselves to patches with tensor product structure
 * of cells. The helper class PatchLocalTensorHelper provides all
 * cell-local-to-patch-local mappings. Whereas global dof indices are queried
 * from DoFInfo objects (identified by @p dofh_index) stored in the @p
 * subdomain_handler. To be precise, access to global dof indices is provided by
 * PatchDoFWorker.
 */
template<int dim, typename Number, int fe_degree>
class PatchTransfer
{
public:
  using CellIterator                            = typename PatchInfo<dim>::CellIterator;
  static constexpr unsigned int fe_order        = fe_degree + 1;
  static constexpr unsigned int n_dofs_per_cell = Utilities::pow(fe_order, dim);
  static constexpr unsigned int macro_size      = VectorizedArray<Number>::n_array_elements;

  // // TODO construct indices at compile time ?
  // struct GetIndexing;

  PatchTransfer(const SubdomainHandler<dim, Number> & subdomain_handler,
                const unsigned int                    dofh_index = 0);

  ~PatchTransfer() = default;

  PatchTransfer(const PatchTransfer & other) = delete;

  PatchTransfer &
  operator=(const PatchTransfer & other) = delete;

  void
  reinit(const unsigned int patch);

  /**
   * Reset the size of patch local vector @p vec.
   */
  void
  reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec) const;

  /**
   * Extract from the global dof values @p src the patch relevant dof values.
   */
  template<typename VectorType>
  AlignedVector<VectorizedArray<Number>>
  gather(const VectorType & src) const;

  /**
   * Same as above, but the patch relevant dof values from global vector @p src
   * are added to @p dst.
   */
  template<typename VectorType>
  void
  gather_add(AlignedVector<VectorizedArray<Number>> & dst, const VectorType & src) const;

  /**
   * Same as above, but @p dst is passed as ArrayView.
   */
  template<typename VectorType>
  void
  gather_add(const ArrayView<VectorizedArray<Number>> dst, const VectorType & src) const;

  /**
   * Add patch relevant dof values @p src to the global dof values @p dst.
   */
  template<typename VectorType>
  void
  scatter_add(VectorType & dst, const AlignedVector<VectorizedArray<Number>> & src) const;

  /**
   * Same as above, but @p dst is passed as ArrayView.
   */
  template<typename VectorType>
  void
  scatter_add(VectorType & dst, const ArrayView<const VectorizedArray<Number>> src) const;

  unsigned int
  n_dofs_per_patch() const;

  /**
   * Domain decomposition information.
   */
  const unsigned int level;
  const unsigned int n_subdomains;
  const unsigned int n_colors;

private:
  // /**
  //  * Read access to patch-local dof indices of cell @cell_no. The returned array
  //  * has a cell-based lexicographical ordering.
  //  */
  // ArrayView<const unsigned int>
  // patch_dof_indices_on_cell(const unsigned int cell_no) const;

  void
  fill_global_dof_indices(const unsigned int patch_id);

  Tensors::TensorHelper<dim> cell_dof_tensor;

  Tensors::TensorHelper<dim> cell_tensor;

  const unsigned int dofh_index;
  const DoFLayout    dof_layout;

  // /**
  //  * Empty constraints used to read and distribute global dof values
  //  * cell-by-cell.
  //  */
  // AffineConstraints<Number> empty_constraints;

  /**
   * A bijective map between patch local dof indices (position index) and their
   * associated global dof indices. The data field has patch-based
   * lexicographical ordering.
   */
  std::vector<std::array<types::global_dof_index, macro_size>> global_dof_indices;

  /**
   * Current patch id identifying a unique patch given by @p subdomain_handler.
   */
  unsigned int patch_id;

  PatchLocalTensorHelper<dim> patch_dof_tensor;

  /**
   * An interface accessing dof-related patch information stored in PatchInfo
   * and DoFInfo objects of the underlying subdomain_handler.
   */
  PatchDoFWorker<dim, Number> patch_dof_worker;

  /**
   * The underlying SubdomainHandler object.
   */
  const SubdomainHandler<dim, Number> & subdomain_handler;

  // mutable std::vector<unsigned int> cell_dof_indices_scratchpad;
};



/**
 * Transfer of patch relevant dof values for block vectors. Transfers for a
 * fixed block are provided by PatchTransfer object of appropriate component @p
 * dofh_index.
 *
 * Currently, each block has to be subject to a scalar-valued finite element (in
 * deal.II speak number of base elements equals one).
 *
 * TODO class has not been tested on blocks associated to different
 * scalar-valued finite elements.
 */
template<int dim, typename Number, int fe_degree>
class PatchTransferBlock
{
public:
  using BlockVectorType = typename LinearAlgebra::distributed::BlockVector<Number>;

  PatchTransferBlock(const SubdomainHandler<dim, Number> & subdomain_handler);

  ~PatchTransferBlock() = default;

  PatchTransferBlock(const PatchTransferBlock & other) = delete;

  PatchTransferBlock &
  operator=(const PatchTransferBlock & other) = delete;

  void
  reinit(const unsigned int patch_id);

  /**
   * Extract from the global dof values @p src the patch relevant dof values and return them.
   */
  AlignedVector<VectorizedArray<Number>>
  gather(const BlockVectorType & src) const;

  /**
   * Same as above, but the global dof values @p dst are added to @p src.
   */
  void
  gather_add(AlignedVector<VectorizedArray<Number>> & dst, const BlockVectorType & src) const;

  /**
   * Return the number of DoFs per patch accumulated over all components.
   */
  unsigned int
  n_dofs_per_patch() const;

  /**
   * Set the size of a patch local vector @p vec with flattened block-structure.
   */
  void
  reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec) const;

  /**
   * Add patch intern dof values @o src to the global dof values @p dst.
   */
  void
  scatter_add(BlockVectorType & dst, const AlignedVector<VectorizedArray<Number>> & src) const;

private:
  using transfer_type = PatchTransfer<dim, Number, fe_degree>;

  const unsigned int                          n_components;
  std::vector<std::shared_ptr<transfer_type>> transfers;
  unsigned int                                patch_id;
};



// ++++++++++++++++++++++++++++++   inline functors   ++++++++++++++++++++++++++++++

// template<int dim, typename Number, int fe_degree>
// struct PatchTransfer<dim, Number, fe_degree>::GetIndexing
// {
//   std::vector<unsigned int>
//   operator()(const TPSS::PatchVariant patch_variant, const bool dg = true) const
//   {
//     std::vector<unsigned int> indices;
//     if(dg)
//     {
//       if(patch_variant == TPSS::PatchVariant::cell)
//         indices = std::move(dg_cell_patch_index_map_impl());
//       else if(patch_variant == TPSS::PatchVariant::vertex)
//         indices = std::move(dg_vertex_patch_index_map_impl());
//       else
//         Assert(false, ExcNotImplemented());
//     }
//     // else
//     //   Assert(false, ExcNotImplemented());

//     return indices;
//   }

//   std::vector<unsigned int>
//   dg_cell_patch_index_map_impl() const
//   {
//     // TODO extract dof information from underlying SubdomainHandler
//     const std::size_t         n_dofs_per_subdomain = n_dofs_per_cell;
//     std::vector<unsigned int> indices;
//     indices.resize(n_dofs_per_subdomain);
//     std::iota(indices.begin(), indices.end(), 0);
//     return indices;
//   }

//   std::vector<unsigned int>
//   dg_vertex_patch_index_map_impl() const
//   {
//     // LAMBDA uniquely maps the (cell index, cell number) pair to the patch index
//     const auto cell_to_patch_index = [](const unsigned int cindex, const unsigned int cell) {
//       unsigned int patch_index{0};

//       constexpr int                 fe_order = fe_degree + 1;
//       std::array<unsigned int, dim> stride_length;
//       if(dim >= 1)
//         stride_length[0] = fe_order;
//       if(dim >= 2)
//         stride_length[1] = 2 * Utilities::pow(fe_order, 2);
//       if(dim >= 3)
//         stride_length[2] = 4 * Utilities::pow(fe_order, 3);

//       std::bitset<dim> multicell_index{cell};
//       unsigned int     stride{0};
//       for(unsigned int direction = 0; direction < dim; ++direction)
//         stride += multicell_index[direction] * stride_length[direction];

//       patch_index = stride;
//       if(dim == 3)
//       {
//         const unsigned int iz = cindex / static_cast<unsigned int>(Utilities::pow(fe_order, 2));
//         const unsigned int remainder_y =
//           cindex % static_cast<unsigned int>(Utilities::pow(fe_order, 2));
//         const unsigned int iy = static_cast<unsigned int>(remainder_y / fe_order);
//         const unsigned int ix = static_cast<unsigned int>(remainder_y % fe_order);
//         patch_index += iz * 4 * (fe_order * fe_order) + iy * 2 * fe_order + ix;
//         //      std::cout << "iz,ry,iy,pi " << iz << " " << remainder_y << " " << iy << " " <<
//         //      patch_index << std::endl ;
//       }
//       if(dim == 2)
//       {
//         const unsigned int iy = static_cast<unsigned int>(cindex / fe_order);
//         const unsigned int ix = static_cast<unsigned int>(cindex % fe_order);
//         patch_index += ix + iy * 2 * fe_order;
//       }
//       if(dim == 1)
//         patch_index += cindex;

//       return patch_index;
//     };

//     constexpr unsigned int vpatch_size = 1 << dim;
//     // TODO extract dof information from underlying SubdomainHandler
//     const std::size_t         n_dofs_per_subdomain = Utilities::pow(2, dim) * n_dofs_per_cell;
//     std::vector<unsigned int> indices;
//     indices.resize(n_dofs_per_subdomain);
//     auto patch_index = indices.begin();
//     for(unsigned int cell = 0; cell < vpatch_size; ++cell)
//       for(unsigned int dof = 0; dof < n_dofs_per_cell; ++dof, ++patch_index)
//         *patch_index = cell_to_patch_index(dof, cell);
//     return indices;
//   }
// };



// ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++



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



// -----------------------------   PatchTransfer   ----------------------------



template<int dim, typename Number, int fe_degree>
inline PatchTransfer<dim, Number, fe_degree>::PatchTransfer(
  const SubdomainHandler<dim, Number> & subdomain_handler_in,
  const unsigned int                    dofh_index_in)
  : level(subdomain_handler_in.get_additional_data().level),
    n_subdomains(subdomain_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(subdomain_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    // assume isotropic elements like Q or DGQ
    cell_dof_tensor(PatchTransfer<dim, Number, fe_degree>::fe_order),
    // assume isotropic patches
    cell_tensor(UniversalInfo<dim>::n_cells_per_direction(
      subdomain_handler_in.get_additional_data().patch_variant)),
    // cell_to_patch_indices(
    //   std::move(GetIndexing{}(subdomain_handler_in.get_additional_data().patch_variant,
    //                           dof_layout == DoFLayout::DGQ))),
    dofh_index(dofh_index_in),
    dof_layout(subdomain_handler_in.get_dof_layout(dofh_index_in)),
    patch_id(numbers::invalid_unsigned_int),
    patch_dof_tensor(cell_tensor, cell_dof_tensor, dof_layout),
    patch_dof_worker(subdomain_handler_in.get_dof_info(dofh_index_in)),
    subdomain_handler(subdomain_handler_in)
{
  AssertThrow(dof_layout != DoFLayout::invalid, ExcMessage("The finite element is not supported."));
  AssertThrow(n_dofs_per_cell == cell_dof_tensor.n_flat(),
              ExcMessage("Only isotropic elements supported."));
  // empty_constraints.close();
}


template<int dim, typename Number, int fe_degree>
inline void
PatchTransfer<dim, Number, fe_degree>::reinit(const unsigned int patch)
{
  AssertIndexRange(patch, n_subdomains);
  patch_id = patch;
  fill_global_dof_indices(patch_id);
  AssertDimension(global_dof_indices.size(), n_dofs_per_patch());

  // DEBUG
  // for(auto lane = 0U; lane < macro_size; ++lane)
  // {
  //   std::cout << "lane: " << lane << " ";
  //   for(const auto & dof : global_dof_indices)
  //     std::cout << dof[lane];
  //   std::cout << std::endl;
  // }
  // }
}


template<int dim, typename Number, int fe_degree>
inline void
PatchTransfer<dim, Number, fe_degree>::reinit_local_vector(
  AlignedVector<VectorizedArray<Number>> & vec) const
{
  Assert(patch_id != numbers::invalid_unsigned_int, ExcNotInitialized());
  vec.resize(n_dofs_per_patch());
}


template<int dim, typename Number, int fe_degree>
inline unsigned int
PatchTransfer<dim, Number, fe_degree>::n_dofs_per_patch() const
{
  return patch_dof_tensor.n_flat();
}


// template<int dim, typename Number, int fe_degree>
// inline ArrayView<const unsigned int>
// PatchTransfer<dim, Number, fe_degree>::patch_dof_indices_on_cell(const unsigned int cell_no)
// const
// {
//   const auto n_dofs_per_cell = cell_dof_tensor.n_flat();
//   cell_dof_indices_scratchpad.resize(n_dofs_per_cell);
//   for(auto cell_dof_index = 0U; cell_dof_index < n_dofs_per_cell; ++cell_dof_index)
//   {
//     const unsigned int patch_dof_index = patch_dof_tensor.dof_index(cell_no, cell_dof_index);
//     cell_dof_indices_scratchpad[cell_dof_index] = patch_dof_index;
//   }
//   return ArrayView<const unsigned>(cell_dof_indices_scratchpad);
// }


// TODO
template<int dim, typename Number, int fe_degree>
inline void
PatchTransfer<dim, Number, fe_degree>::fill_global_dof_indices(const unsigned int patch_id)
{
  AssertIndexRange(patch_id, subdomain_handler.get_partition_data().n_subdomains());
  std::vector<std::array<types::global_dof_index, macro_size>> global_dof_indices_plain(
    patch_dof_tensor.n_flat());
  const auto n_cells         = cell_tensor.n_flat();
  const auto n_dofs_per_cell = cell_dof_tensor.n_flat();
  for(auto cell_no = 0U; cell_no < n_cells; ++cell_no)
  {
    const auto global_dof_indices_on_cell =
      patch_dof_worker.get_dof_indices_on_cell(patch_id, cell_no);
    for(unsigned int lane = 0; lane < macro_size; ++lane)
      for(auto cell_dof_index = 0U; cell_dof_index < n_dofs_per_cell; ++cell_dof_index)
      {
        const unsigned int patch_dof_index = patch_dof_tensor.dof_index(cell_no, cell_dof_index);
        global_dof_indices_plain[patch_dof_index][lane] =
          global_dof_indices_on_cell[lane][cell_dof_index];
      }
  }

  if(dof_layout == DoFLayout::DGQ) // might change in future
    std::swap(global_dof_indices, global_dof_indices_plain);
  else if(dof_layout == DoFLayout::Q)
    std::swap(global_dof_indices, global_dof_indices_plain);
  else
    AssertThrow(false, ExcMessage("Finite element is not supported."));
}



// -----------------------------   PatchTransferBlock   ----------------------------



template<int dim, typename Number, int fe_degree>
inline PatchTransferBlock<dim, Number, fe_degree>::PatchTransferBlock(
  const SubdomainHandler<dim, Number> & subdomain_handler_in)
  : n_components(subdomain_handler_in.n_components()), patch_id(numbers::invalid_unsigned_int)
{
  const unsigned n_components = subdomain_handler_in.n_components();
  transfers.resize(n_components);
  for(unsigned int dofh_index = 0; dofh_index < n_components; ++dofh_index)
    transfers[dofh_index] = std::make_shared<transfer_type>(subdomain_handler_in, dofh_index);
}


template<int dim, typename Number, int fe_degree>
inline void
PatchTransferBlock<dim, Number, fe_degree>::reinit(const unsigned int patch_id)
{
  this->patch_id = patch_id;
  for(auto transfer : transfers)
    transfer->reinit(patch_id);
}


template<int dim, typename Number, int fe_degree>
inline unsigned int
PatchTransferBlock<dim, Number, fe_degree>::n_dofs_per_patch() const
{
  const unsigned int n_dofs_total = std::accumulate(transfers.begin(),
                                                    transfers.end(),
                                                    0,
                                                    [](const auto sum, const auto & transfer) {
                                                      return sum + transfer->n_dofs_per_patch();
                                                    });
  return n_dofs_total;
}


template<int dim, typename Number, int fe_degree>
inline void
PatchTransferBlock<dim, Number, fe_degree>::reinit_local_vector(
  AlignedVector<VectorizedArray<Number>> & vec) const
{
  Assert(patch_id != numbers::invalid_unsigned_int, ExcNotInitialized());
  vec.resize(n_dofs_per_patch());
}



} // end namespace TPSS

#include "patch_transfer.templates.h"

#endif // end inclusion guard
