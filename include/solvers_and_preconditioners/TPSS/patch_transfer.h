#ifndef PATCHTRANSFER_H
#define PATCHTRANSFER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include "TPSS.h"
#include "subdomain_handler.h"

#include <array>
#include <memory>

using namespace dealii;



namespace TPSS
{
namespace internal
{
/**
 * Read access to the proc-local entry at @p dof_index. The Vector class has
 * not to distinguish between proc-local and global entries and, thus, does
 * not require the interface local_element() compared to
 * LinearAlgebra::distributed::Vector.
 */
template<typename VectorType>
const typename VectorType::value_type
local_element(const VectorType & vec, const unsigned int dof_index)
{
  AssertThrow(false, ExcMessage("This VectorType is not supported."));
}

template<typename Number>
const Number
local_element(const Vector<Number> & vec, const unsigned int dof_index)
{
  return vec[dof_index];
}

template<typename Number>
const Number
local_element(const LinearAlgebra::distributed::Vector<Number> & vec, const unsigned int dof_index)
{
  return vec.local_element(dof_index);
}

/**
 * Write access to the proc-local entry at @p dof_index. The Vector class has
 * not to distinguish between proc-local and global entries and, thus, does
 * not require the interface local_element() compared to
 * LinearAlgebra::distributed::Vector.
 */
template<typename VectorType>
typename VectorType::value_type &
local_element(const VectorType & vec, const unsigned int dof_index)
{
  AssertThrow(false, ExcMessage("This VectorType is not supported."));
}

template<typename Number>
Number &
local_element(Vector<Number> & vec, const unsigned int dof_index)
{
  return vec[dof_index];
}

template<typename Number>
Number &
local_element(LinearAlgebra::distributed::Vector<Number> & vec, const unsigned int dof_index)
{
  return vec.local_element(dof_index);
}
} // namespace internal



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
template<int dim, typename Number>
class PatchTransfer
{
public:
  using CellIterator                       = typename PatchInfo<dim>::CellIterator;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::size();

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

  const PatchDoFWorker<dim, Number> &
  get_patch_dof_worker() const;

  ArrayView<const unsigned int>
  get_dof_indices(const unsigned int lane) const;

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

  const TPSS::PatchVariant patch_variant;

private:
  // /**
  //  * Read access to patch-local dof indices of cell @cell_no. The returned array
  //  * has a cell-based lexicographical ordering.
  //  */
  // ArrayView<const unsigned int>
  // patch_dof_indices_on_cell(const unsigned int cell_no) const;

  void
  fill_global_dof_indices(const unsigned int patch_id);

  /**
   * An interface accessing dof-related patch information stored in PatchInfo
   * and DoFInfo objects of the underlying subdomain_handler.
   */
  PatchDoFWorker<dim, Number> patch_dof_worker;

  /**
   * The underlying SubdomainHandler object.
   */
  const SubdomainHandler<dim, Number> & subdomain_handler;

  const PatchLocalTensorHelper<dim> & patch_dof_tensor;

  const Tensors::TensorHelper<dim> & cell_dof_tensor;

  const Tensors::TensorHelper<dim> & cell_tensor;

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
  std::array<std::vector<unsigned int>, macro_size> global_dof_indices;

  /**
   * Current patch id identifying a unique patch given by @p subdomain_handler.
   */
  unsigned int patch_id;

  TPSS::CachingStrategy caching_strategy;

  // mutable std::vector<unsigned int> cell_dof_indices_scratchpad;
};



/**
 * Transfer of patch relevant dof values for block vectors. Transfers for a
 * fixed block are provided by PatchTransfer object of appropriate component @p
 * dofh_index.
 *
 * TODO Currently, each block has to be subject to a scalar-valued finite element.
 *
 * TODO class has not been tested on blocks associated to different
 * scalar-valued finite elements.
 */
template<int dim, typename Number>
class PatchTransferBlock
{
public:
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
  template<typename BlockVectorType>
  AlignedVector<VectorizedArray<Number>>
  gather(const BlockVectorType & src) const;

  /**
   * Same as above, but the global dof values @p dst are added to @p src.
   */
  template<typename BlockVectorType>
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
  template<typename BlockVectorType>
  void
  scatter_add(BlockVectorType & dst, const AlignedVector<VectorizedArray<Number>> & src) const;

private:
  using transfer_type = PatchTransfer<dim, Number>;

  const unsigned int n_blocks;

  std::vector<std::shared_ptr<transfer_type>> transfers;

  unsigned int patch_id;

  const unsigned int n_dofs_total;
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
//     const std::size_t         n_dofs_per_subdomain = n_dofs_per_cell_static;
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
//     const std::size_t         n_dofs_per_subdomain = Utilities::pow(2, dim) *
//     n_dofs_per_cell_static; std::vector<unsigned int> indices;
//     indices.resize(n_dofs_per_subdomain);
//     auto patch_index = indices.begin();
//     for(unsigned int cell = 0; cell < vpatch_size; ++cell)
//       for(unsigned int dof = 0; dof < n_dofs_per_cell_static; ++dof, ++patch_index)
//         *patch_index = cell_to_patch_index(dof, cell);
//     return indices;
//   }
// };



// ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++



// -----------------------------   PatchTransfer   ----------------------------



template<int dim, typename Number>
inline PatchTransfer<dim, Number>::PatchTransfer(
  const SubdomainHandler<dim, Number> & subdomain_handler_in,
  const unsigned int                    dofh_index_in)
  : level(subdomain_handler_in.get_additional_data().level),
    n_subdomains(subdomain_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(subdomain_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    patch_variant(subdomain_handler_in.get_additional_data().patch_variant),
    patch_dof_worker(subdomain_handler_in.get_dof_info(dofh_index_in)),
    subdomain_handler(subdomain_handler_in),
    patch_dof_tensor(patch_dof_worker.get_dof_tensor()),
    cell_dof_tensor(patch_dof_worker.get_dof_tensor().cell_dof_tensor),
    cell_tensor(patch_dof_worker.get_dof_tensor().cell_tensor),
    dofh_index(dofh_index_in),
    dof_layout(subdomain_handler_in.get_dof_layout(dofh_index_in)),
    patch_id(numbers::invalid_unsigned_int),
    caching_strategy(patch_dof_worker.get_dof_info().get_additional_data().caching_strategy)
{
  AssertThrow(dof_layout != DoFLayout::invalid, ExcMessage("The finite element is not supported."));
}


template<int dim, typename Number>
inline void
PatchTransfer<dim, Number>::reinit(const unsigned int patch)
{
  AssertIndexRange(patch, n_subdomains);
  patch_id = patch;

  if(caching_strategy != TPSS::CachingStrategy::Cached)
  {
    fill_global_dof_indices(patch_id);
    AssertDimension(global_dof_indices.size(), n_dofs_per_patch());
  }
}


template<int dim, typename Number>
inline void
PatchTransfer<dim, Number>::reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec) const
{
  Assert(patch_id != numbers::invalid_unsigned_int, ExcNotInitialized());
  vec.resize(n_dofs_per_patch());
}


template<int dim, typename Number>
const PatchDoFWorker<dim, Number> &
PatchTransfer<dim, Number>::get_patch_dof_worker() const
{
  return patch_dof_worker;
}


template<int dim, typename Number>
ArrayView<const unsigned int>
PatchTransfer<dim, Number>::get_dof_indices(const unsigned int lane) const
{
  Assert(patch_id != numbers::invalid_unsigned_int, ExcNotInitialized());
  if(lane >= patch_dof_worker.n_lanes_filled(patch_id))
    return get_dof_indices(0);

  if(caching_strategy == TPSS::CachingStrategy::Cached)
    return patch_dof_worker.get_dof_indices_on_patch(patch_id, lane);

  AssertDimension(this->global_dof_indices[lane].size(), n_dofs_per_patch());
  return ArrayView<const unsigned int>(global_dof_indices[lane].data(),
                                       global_dof_indices[lane].size());
}


template<int dim, typename Number>
inline unsigned int
PatchTransfer<dim, Number>::n_dofs_per_patch() const
{
  return patch_dof_worker.n_dofs();
}


template<int dim, typename Number>
inline void
PatchTransfer<dim, Number>::fill_global_dof_indices(const unsigned int patch_id)
{
  AssertIndexRange(patch_id, n_subdomains);
  for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_id); ++lane)
  {
    auto && global_dof_indices_at_lane = patch_dof_worker.fill_dof_indices_on_patch(patch_id, lane);
    AssertDimension(global_dof_indices_at_lane.size(), n_dofs_per_patch());
    std::swap(this->global_dof_indices[lane], global_dof_indices_at_lane);
  }
}



// -----------------------------   PatchTransferBlock   ----------------------------



template<int dim, typename Number>
inline PatchTransferBlock<dim, Number>::PatchTransferBlock(
  const SubdomainHandler<dim, Number> & subdomain_handler_in)
  : n_blocks(subdomain_handler_in.n_dof_handlers()),
    transfers(std::move([this](const auto & subdomain_handler) {
      AssertDimension(n_blocks, subdomain_handler.n_dof_handlers());
      std::vector<std::shared_ptr<transfer_type>> transfers;
      for(auto dofh_index = 0U; dofh_index < n_blocks; ++dofh_index)
        transfers.emplace_back(std::make_shared<transfer_type>(subdomain_handler, dofh_index));
      return transfers;
    }(subdomain_handler_in))),
    patch_id(numbers::invalid_unsigned_int),
    n_dofs_total([this]() -> unsigned int {
      AssertDimension(transfers.size(), n_blocks);
      return std::accumulate(transfers.begin(),
                             transfers.end(),
                             0,
                             [](const auto sum, const auto & transfer) {
                               return sum + transfer->n_dofs_per_patch();
                             });
    }())
{
}


template<int dim, typename Number>
inline void
PatchTransferBlock<dim, Number>::reinit(const unsigned int patch_id)
{
  this->patch_id = patch_id;
  for(auto transfer : transfers)
    transfer->reinit(patch_id);
}


template<int dim, typename Number>
inline unsigned int
PatchTransferBlock<dim, Number>::n_dofs_per_patch() const
{
  // const unsigned int n_dofs_total = std::accumulate(transfers.begin(),
  //                                                   transfers.end(),
  //                                                   0,
  //                                                   [](const auto sum, const auto & transfer) {
  //                                                     return sum + transfer->n_dofs_per_patch();
  //                                                   });
  return n_dofs_total;
}


template<int dim, typename Number>
inline void
PatchTransferBlock<dim, Number>::reinit_local_vector(
  AlignedVector<VectorizedArray<Number>> & vec) const
{
  Assert(patch_id != numbers::invalid_unsigned_int, ExcNotInitialized());
  vec.resize(n_dofs_per_patch());
}



} // end namespace TPSS

#include "patch_transfer.templates.h"

#endif // end inclusion guard
