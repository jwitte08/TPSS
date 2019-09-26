#ifndef PATCHTRANSFER_H
#define PATCHTRANSFER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "TPSS.h"
#include "subdomain_handler.h"

#include <array>
#include <memory>

using namespace dealii;

namespace TPSS
{
// TODO replace FEEvaluation with read-write-operations -> reduces
// templates
// TODO merge Base and Derived PatchTransfer !!!
/**
 * Base class providing the interface to model the (mathematical)
 * restriction operator from global degrees of freedom onto subdomain
 * degrees of freedom and its transpose operator, namely the
 * prolongation operator from subdomain to global space.
 *
 * Therefore the information within the MatrixFreeConnect object,
 * stored in the underlying SubdomainHandler is combined with a
 * FEEvaluation object to implement the restriction and prolongation
 * cell-wise and lane-wise regarding the MatrixFree
 * read-write-operations @p distribute_local_to_global and @p
 * read_dof_values.
 *
 * Furthermore, a bijective map between local cell dofs
 * (lexicographical) for each cell within a patch and the local patch
 * dofs (lexicographical) is requested on construction. Obviously,for
 * a DG-cell patch it is the identity. For a DG-vertex patch we have
 * to take into account that each patch has 2 cells in each coordinate
 * direction. In other words, the map @p cell_to_patch_indices has to
 * be provided at construction by the deriving class.
 */
template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
class PatchTransferBase
{
public:
  using CellIterator                            = typename PatchInfo<dim>::CellIterator;
  static constexpr unsigned int fe_order        = fe_degree + 1;
  static constexpr unsigned int n_dofs_per_cell = Utilities::pow(fe_order, dim);

  unsigned int
  n_dofs_per_patch() const
  {
    return n_dofs;
  }

  /**
   * Reinitialize information of the (macro) patch with unique id @p patch.
   */
  void
  reinit(const unsigned int patch);

  /**
   * Set the size of a patch local vector @p vec.
   */
  void
  reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec) const;

  /**
   * Extract from the global dof values @p src the patch relevant dof values and return them.
   */
  template<typename VectorType>
  AlignedVector<VectorizedArray<Number>>
  gather(const VectorType & src) const;

  /**
   * Same as above, but the global dof values @p src are added to @p dst.
   */
  template<typename VectorType>
  void
  gather_add(AlignedVector<VectorizedArray<Number>> & dst, const VectorType & src) const;

  /**
   * Same as above, but the updated local dof values @p dst are passed
   * as ArrayView.
   */
  template<typename VectorType>
  void
  gather_add(const ArrayView<VectorizedArray<Number>> dst, const VectorType & src) const;

  /**
   * Add patch intern dof values @o src to the global dof values @p dst.
   */
  template<typename VectorType>
  void
  scatter_add(VectorType & dst, const ArrayView<const VectorizedArray<Number>> src) const;

  /**
   * Add patch intern dof values @o src to the global dof values @p dst.
   */
  template<typename VectorType>
  void
  scatter_add(VectorType & dst, const AlignedVector<VectorizedArray<Number>> & src) const;

  /**
   * Domain decomposition information.
   */
  const unsigned int level;
  const unsigned int n_subdomains;
  const unsigned int n_colors;

protected:
  /**
   * Pass the SubdomainHandler to obtain an interface to the
   * MatrixFree infrastructure through the underlying
   * MatrixFreeConnect object. The bijective map @p
   * cell_to_patch_indexing maps the pair (local_cell_dof, cell_no)
   * onto local_patch_dof.
   */
  PatchTransferBase(const SubdomainHandler<dim, Number> & sd_handler,
                    std::vector<unsigned int> &&          cell_to_patch_indexing,
                    const unsigned int                    dofh_index_in = 0);

  ~PatchTransferBase() = default;

  PatchTransferBase(const PatchTransferBase & other) = delete;

  PatchTransferBase &
  operator=(const PatchTransferBase & other) = delete;

  ArrayView<const unsigned>
  patch_dofs_on_cell(const unsigned int cell_no) const
  {
    const auto begin = cell_to_patch_indices.data() + cell_no * n_dofs_per_cell;
    AssertIndexRange((cell_no + 1) * n_dofs_per_cell, cell_to_patch_indices.size() + 1);
    return ArrayView<const unsigned>(begin, n_dofs_per_cell);
  }

  /**
   * Underlying SubdomainHandler that holds a MatrixFreeConnect object.
   */
  const SubdomainHandler<dim, Number> & sd_handler;

private:
  /**
   * A bijective map between local cell dofs (lexicographical) for
   * each cell within a patch and the local patch dofs (lexicographical).
   * Each element of the array is a unique patch index and the array is
   * lexicographically ordered as follows:
   *    local_cell_dof < cell_no
   */
  std::vector<unsigned int> cell_to_patch_indices;

  /**
   * Internally used to perform the restriction and prolongation operation
   * cell-wise and lane-wise based on the infrastructure provided by the
   * MatrixFree read-write-operations. Logically @p restrict or @p prolongate
   * member functions are const, that is why we require a mutable FEEvaluation.
   */
  mutable FEEvaluation<dim, fe_degree, n_q_points_1d, n_comp, Number> fe_eval;
  mutable std::mutex                                                  write_mutex;

  /**
   * Information required to map patch and cell dof indices.
   */
  const unsigned int n_dofs;

  /**
   * Variables uniquely determine a (macro) patch and the corresponding linkage
   * to the MatrixFree infrastructure.
   */
  const unsigned int                            dofh_index;
  unsigned int                                  patch_id;
  unsigned int                                  n_batches;
  const std::pair<unsigned int, unsigned int> * batch_count;
  const std::array<unsigned int, 3> *           batch_triple;

  PatchWorker<dim, Number>  patch_worker;
  // TODO pass meaningful constraints from the MatrixFree/SubdomainHandler
  AffineConstraints<Number> constraints;
  /**
   * Read-write-operations differ if not all vectorization lanes are meaningfully
   * filled.
   */
  bool is_incomplete_patch;
};

/**
 * Handling the transfer between patch-local and global degrees of
 * freedom depending on the underlying SubdomainHandler.
 */
template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
class PatchTransfer : public PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>
{
public:
  using Base = PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>;

  // TODO construct indices at compile time ?
  struct GetIndexing;

  PatchTransfer(const SubdomainHandler<dim, Number> & sd_handler,
                const unsigned int                    dofh_index = 0);

  ~PatchTransfer() = default;

  PatchTransfer(const PatchTransfer & other) = delete;

  PatchTransfer &
  operator=(const PatchTransfer & other) = delete;
};

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
class PatchTransferBlock
{
public:
  using BlockVectorType = typename LinearAlgebra::distributed::BlockVector<Number>;

  PatchTransferBlock(const SubdomainHandler<dim, Number> & sd_handler);

  ~PatchTransferBlock() = default;

  PatchTransferBlock(const PatchTransferBlock & other) = delete;

  PatchTransferBlock &
  operator=(const PatchTransferBlock & other) = delete;

  /**
   * Return the accumulated (over components) number of DoFs per patch.
   */
  unsigned int
  n_dofs_per_patch() const
  {
    const unsigned int n_dofs_total = std::accumulate(transfers.begin(),
                                                      transfers.end(),
                                                      0,
                                                      [](const auto sum, const auto & transfer) {
                                                        return sum + transfer->n_dofs_per_patch();
                                                      });
    return n_dofs_total;
  }

  // /**
  //  * Return the number of DoFs per patch corresponding to the
  //  * DoFHandler with index @p dofh_index.
  //  */
  // unsigned int
  // n_dofs_per_patch (std::size_t dofh_index) const
  // {
  //   return transfers[dofh_index]->n_dofs_per_patch();
  // }

  /**
   * Reinitialize information of the (macro) patch with unique id @p patch.
   */
  void
  reinit(const unsigned int patch_id)
  {
    this->patch_id = patch_id;
    for(auto transfer : transfers)
      transfer->reinit(patch_id);
  }

  /**
   * Set the size of a patch local vector @p vec with flattened block-structure.
   */
  void
  reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec) const
  {
    Assert(patch_id != static_cast<unsigned int>(-1), ExcNotInitialized());
    vec.resize(n_dofs_per_patch());
  }

  /**
   * Extract from the global dof values @p src the patch relevant dof values and return them.
   */
  AlignedVector<VectorizedArray<Number>>
  gather(const BlockVectorType & src) const
  {
    AlignedVector<VectorizedArray<Number>> dst;
    reinit_local_vector(dst);
    auto begin = dst.begin();
    for(std::size_t b = 0; b < n_components; ++b)
    {
      const auto                         transfer = transfers[b];
      const auto                         size     = transfer->n_dofs_per_patch();
      ArrayView<VectorizedArray<Number>> dst_block{begin, size};
      transfer->gather_add(dst_block, src.block(b));
      begin += size;
    }
    AssertThrow(begin == dst.end(), ExcMessage("Inconsistent slicing."));
    return dst;
  }

  /**
   * Same as above, but the global dof values @p dst are added to @p src.
   */
  void
  gather_add(AlignedVector<VectorizedArray<Number>> & dst, const BlockVectorType & src) const
  {
    AssertDimension(dst.size(), n_dofs_per_patch());
    const auto & src_local = gather(src);
    std::transform(dst.begin(),
                   dst.end(),
                   src_local.begin(),
                   dst.begin(),
                   [](const auto & dst, const auto & src) { return dst + src; });
  }

  /**
   * Add patch intern dof values @o src to the global dof values @p dst.
   */
  void
  scatter_add(BlockVectorType & dst, const AlignedVector<VectorizedArray<Number>> & src) const
  {
    auto begin = src.begin();
    for(std::size_t b = 0; b < n_components; ++b)
    {
      const auto                               transfer = transfers[b];
      const auto                               size     = transfer->n_dofs_per_patch();
      ArrayView<const VectorizedArray<Number>> src_block{begin, size};
      transfer->scatter_add(dst.block(b), src_block);
      begin += size;
    }
    AssertThrow(begin == src.end(), ExcMessage("Inconsistent slicing."));
  }

private:
  using transfer_type = PatchTransfer<dim, fe_degree, n_q_points_1d, 1, Number>;
  // template<typename UnaryFunctionType, typename InputType>
  // void
  // blockwise_unary(const UnaryFunctionType & unary_func, InputType && input) const
  // {
  //   for(unsigned int b = 0; b < n_components; ++b)
  //     unary_func(std::forward<InputType>(input));
  // }

  const unsigned int                          n_components;
  std::vector<std::shared_ptr<transfer_type>> transfers;
  unsigned int                                patch_id;
};


// ++++++++++++++++++++++++++++++   inline functors   ++++++++++++++++++++++++++++++

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
struct PatchTransfer<dim, fe_degree, n_q_points_1d, n_comp, Number>::GetIndexing
{
  std::vector<unsigned int>
  operator()(const TPSS::PatchVariant patch_variant, const bool dg = true) const
  {
    std::vector<unsigned int> indices;
    if(dg)
    {
      if(patch_variant == TPSS::PatchVariant::cell)
        indices = std::move(dg_cell_patch_index_map_impl());
      else if(patch_variant == TPSS::PatchVariant::vertex)
        indices = std::move(dg_vertex_patch_index_map_impl());
      else
        Assert(false, ExcNotImplemented());
    }
    else
      Assert(false, ExcNotImplemented());

    return indices;
  }

  std::vector<unsigned int>
  dg_cell_patch_index_map_impl() const
  {
    // TODO extract dof information from underlying SubdomainHandler
    const std::size_t         n_dofs_per_subdomain = Base::n_dofs_per_cell;
    std::vector<unsigned int> indices;
    indices.resize(n_dofs_per_subdomain);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  std::vector<unsigned int>
  dg_vertex_patch_index_map_impl() const
  {
    // LAMBDA uniquely maps the (cell index, cell number) pair to the patch index
    const auto cell_to_patch_index = [](const unsigned int cindex, const unsigned int cell) {
      unsigned int patch_index{0};

      constexpr int                 fe_order = fe_degree + 1;
      std::array<unsigned int, dim> stride_length;
      if(dim >= 1)
        stride_length[0] = fe_order;
      if(dim >= 2)
        stride_length[1] = 2 * Utilities::pow(fe_order, 2);
      if(dim >= 3)
        stride_length[2] = 4 * Utilities::pow(fe_order, 3);

      std::bitset<dim> multicell_index{cell};
      unsigned int     stride{0};
      for(unsigned int direction = 0; direction < dim; ++direction)
        stride += multicell_index[direction] * stride_length[direction];

      patch_index = stride;
      if(dim == 3)
      {
        const unsigned int iz = cindex / static_cast<unsigned int>(Utilities::pow(fe_order, 2));
        const unsigned int remainder_y =
          cindex % static_cast<unsigned int>(Utilities::pow(fe_order, 2));
        const unsigned int iy = static_cast<unsigned int>(remainder_y / fe_order);
        const unsigned int ix = static_cast<unsigned int>(remainder_y % fe_order);
        patch_index += iz * 4 * (fe_order * fe_order) + iy * 2 * fe_order + ix;
        //      std::cout << "iz,ry,iy,pi " << iz << " " << remainder_y << " " << iy << " " <<
        //      patch_index << std::endl ;
      }
      if(dim == 2)
      {
        const unsigned int iy = static_cast<unsigned int>(cindex / fe_order);
        const unsigned int ix = static_cast<unsigned int>(cindex % fe_order);
        patch_index += ix + iy * 2 * fe_order;
      }
      if(dim == 1)
        patch_index += cindex;

      return patch_index;
    };

    constexpr unsigned int vpatch_size = 1 << dim;
    // TODO extract dof information from underlying SubdomainHandler
    const std::size_t         n_dofs_per_subdomain = Utilities::pow(2, dim) * Base::n_dofs_per_cell;
    std::vector<unsigned int> indices;
    indices.resize(n_dofs_per_subdomain);
    auto patch_index = indices.begin();
    for(unsigned int cell = 0; cell < vpatch_size; ++cell)
      for(unsigned int dof = 0; dof < Base::n_dofs_per_cell; ++dof, ++patch_index)
        *patch_index = cell_to_patch_index(dof, cell);
    return indices;
  }
};



// ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++

// --------------------------------   PatchTransferBase   --------------------------------

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::PatchTransferBase(
  const SubdomainHandler<dim, Number> & sd_handler_in,
  std::vector<unsigned int> &&          cell_to_patch_indexing,
  const unsigned int                    dofh_index_in)
  : level(sd_handler_in.get_additional_data().level),
    n_subdomains(sd_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(sd_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    sd_handler(sd_handler_in),
    cell_to_patch_indices(std::move(cell_to_patch_indexing)),
    fe_eval(sd_handler.get_matrix_free()),
    n_dofs(cell_to_patch_indices.empty() ? static_cast<unsigned int>(-1) :
                                           cell_to_patch_indices.size()),
    dofh_index(dofh_index_in),
    patch_id(-1),
    n_batches(-1),
    batch_count(nullptr),
    batch_triple(nullptr),
    patch_worker(sd_handler_in.get_patch_info()),
    is_incomplete_patch(false)
{
  static_assert(n_comp == 1, "Handles only one scalar DoFHandler.");
  AssertThrow(n_dofs != static_cast<unsigned int>(-1),
              ExcMessage("The cell to patch index map is uninitialized!"));
  constraints.close();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline void
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::reinit(const unsigned int patch)
{
  AssertIndexRange(patch, n_subdomains);
  patch_id            = patch;
  is_incomplete_patch = sd_handler.get_patch_info().is_incomplete_patch[patch];

  const auto & mf_connect = sd_handler.get_matrixfree_connect();
  n_batches               = mf_connect.set_pointers_and_count(patch_id, batch_triple, batch_count);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline void
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::reinit_local_vector(
  AlignedVector<VectorizedArray<Number>> & vec) const
{
  Assert(patch_id != static_cast<unsigned int>(-1), ExcNotInitialized());
  vec.resize(n_dofs);
}

// -----------------------------   PatchTransfer   ----------------------------

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline PatchTransfer<dim, fe_degree, n_q_points_1d, n_comp, Number>::PatchTransfer(
  const SubdomainHandler<dim, Number> & sd_handler,
  const unsigned int                    dofh_index_in)
  : Base(sd_handler,
         GetIndexing{}(sd_handler.get_additional_data().patch_variant, true /*is DG?*/),
         dofh_index_in)
{
}



// -----------------------------   PatchTransferBlock   ----------------------------

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline PatchTransferBlock<dim, fe_degree, n_q_points_1d, Number>::PatchTransferBlock(
  const SubdomainHandler<dim, Number> & sd_handler_in)
  : n_components(sd_handler_in.n_components()), patch_id(static_cast<unsigned int>(-1))
{
  const unsigned n_components = sd_handler_in.n_components();
  transfers.resize(n_components);
  for(unsigned int dofh_index = 0; dofh_index < n_components; ++dofh_index)
    transfers[dofh_index] = std::make_shared<transfer_type>(sd_handler_in, dofh_index);
}



} // end namespace TPSS

#include "patch_transfer.templates.h"

#endif // end inclusion guard
