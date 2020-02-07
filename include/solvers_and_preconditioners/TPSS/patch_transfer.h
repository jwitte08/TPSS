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
enum class DoFLayout
{
  invalid,
  DGQ,
  Q
};

template<int dim>
DoFLayout
get_dof_layout(const FiniteElement<dim> & finite_element)
{
  auto dof_layout = DoFLayout::invalid;
  if(finite_element.get_name().find("FE_DGQ") != std::string::npos)
    dof_layout = DoFLayout::DGQ;
  else if(finite_element.get_name().find("FE_Q") != std::string::npos)
    dof_layout = DoFLayout::Q;
  return dof_layout;
}

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
template<int dim, typename Number, int fe_degree>
class PatchTransfer
{
public:
  using CellIterator                            = typename PatchInfo<dim>::CellIterator;
  static constexpr unsigned int fe_order        = fe_degree + 1;
  static constexpr unsigned int n_dofs_per_cell = Utilities::pow(fe_order, dim);
  static constexpr unsigned int macro_size      = VectorizedArray<Number>::n_array_elements;

  // TODO construct indices at compile time ?
  struct GetIndexing;

  PatchTransfer(const SubdomainHandler<dim, Number> & subdomain_handler,
                const unsigned int                    dofh_index = 0);

  ~PatchTransfer() = default;

  PatchTransfer(const PatchTransfer & other) = delete;

  PatchTransfer &
  operator=(const PatchTransfer & other) = delete;

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
  const DoFLayout    dof_layout;

private:
  /**
   * Read access to patch-local dof indices of cell @cell_no. The returned array
   * has a cell-based lexicographical ordering.
   */
  ArrayView<const unsigned int>
  patch_dofs_on_cell(const unsigned int cell_no) const;

  /**
   * TODO compressed mode ...
   */
  void
  reinit_patch_to_global_indices(const unsigned int patch_id);

  /**
   * A bijective map between local cell dofs (lexicographical) for
   * each cell within a patch and the local patch dofs (lexicographical).
   * Each element of the array is a unique patch index and the array is
   * lexicographically ordered as follows:
   *    local_cell_dof < cell_no
   */
  std::vector<unsigned int> cell_to_patch_indices;

  /**
   * Empty constraints used to read and distribute global dof values
   * cell-by-cell.
   */
  AffineConstraints<Number> empty_constraints;

  /**
   * A bijective map between patch-local dof indices (random access index) and
   * their associated global dof indices. The data field has patch-based
   * lexicographical ordering.
   */
  std::vector<std::array<types::global_dof_index, macro_size>> patch_to_global_indices;

  /**
   * A flag activating compressed mode. Then, the first global dof index of each
   * cell is stored. From this all global dof indices per cell are reconstructed
   * depending on the underlying finite element.
   */
  const bool compressed;

  // TODO dofh_index is not used
  const unsigned int dofh_index;

  /**
   * Current patch id identifying a unique patch given by @p subdomain_handler.
   */
  unsigned int patch_id;

  /**
   * An interface accessing patch information given by @p subdomain_handler.
   */
  PatchWorker<dim, Number> patch_worker;

  /**
   * The underlying SubdomainHandler object.
   */
  const SubdomainHandler<dim, Number> & subdomain_handler;
};



/**
 * TODO description
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

  /**
   * Return the number of DoFs per patch accumulated over all components.
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
    Assert(patch_id != numbers::invalid_unsigned_int, ExcNotInitialized());
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
  using transfer_type = PatchTransfer<dim, Number, fe_degree>;

  const unsigned int                          n_components;
  std::vector<std::shared_ptr<transfer_type>> transfers;
  unsigned int                                patch_id;
};



// ++++++++++++++++++++++++++++++   inline functors   ++++++++++++++++++++++++++++++

template<int dim, typename Number, int fe_degree>
struct PatchTransfer<dim, Number, fe_degree>::GetIndexing
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
    const std::size_t         n_dofs_per_subdomain = n_dofs_per_cell;
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
    const std::size_t         n_dofs_per_subdomain = Utilities::pow(2, dim) * n_dofs_per_cell;
    std::vector<unsigned int> indices;
    indices.resize(n_dofs_per_subdomain);
    auto patch_index = indices.begin();
    for(unsigned int cell = 0; cell < vpatch_size; ++cell)
      for(unsigned int dof = 0; dof < n_dofs_per_cell; ++dof, ++patch_index)
        *patch_index = cell_to_patch_index(dof, cell);
    return indices;
  }
};



// ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++

// -----------------------------   PatchTransfer   ----------------------------

// TODO use dofh_index
template<int dim, typename Number, int fe_degree>
inline PatchTransfer<dim, Number, fe_degree>::PatchTransfer(
  const SubdomainHandler<dim, Number> & subdomain_handler_in,
  const unsigned int                    dofh_index_in)
  : level(subdomain_handler_in.get_additional_data().level),
    n_subdomains(subdomain_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(subdomain_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    dof_layout(get_dof_layout(subdomain_handler_in.get_dof_handler().get_fe())),
    cell_to_patch_indices(
      std::move(GetIndexing{}(subdomain_handler_in.get_additional_data().patch_variant,
                              dof_layout == DoFLayout::DGQ))),
    compressed(subdomain_handler_in.get_additional_data().compressed),
    dofh_index(dofh_index_in),
    patch_id(numbers::invalid_unsigned_int),
    patch_worker(subdomain_handler_in.get_patch_info()),
    subdomain_handler(subdomain_handler_in)
{
  AssertThrow(!cell_to_patch_indices.empty(),
              ExcMessage("The cell to patch index map is uninitialized!"));
  AssertThrow(dof_layout != DoFLayout::invalid, ExcMessage("The finite element is not supported."));
  if(!compressed)
    AssertThrow(dof_layout == DoFLayout::DGQ,
                ExcMessage("Uncompressed mode only implemented for DGQ elements."));
  empty_constraints.close();
}



template<int dim, typename Number, int fe_degree>
inline void
PatchTransfer<dim, Number, fe_degree>::reinit(const unsigned int patch)
{
  AssertIndexRange(patch, n_subdomains);
  patch_id = patch;
  if(compressed)
  {
    reinit_patch_to_global_indices(patch_id);
    AssertDimension(patch_to_global_indices.size(), n_dofs_per_patch());
  }
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
  return cell_to_patch_indices.size();
}


template<int dim, typename Number, int fe_degree>
inline ArrayView<const unsigned int>
PatchTransfer<dim, Number, fe_degree>::patch_dofs_on_cell(const unsigned int cell_no) const
{
  const auto begin = cell_to_patch_indices.data() + cell_no * n_dofs_per_cell;
  AssertIndexRange((cell_no + 1) * n_dofs_per_cell, cell_to_patch_indices.size() + 1);
  return ArrayView<const unsigned>(begin, n_dofs_per_cell);
}

template<int dim, typename Number, int fe_degree>
inline void
PatchTransfer<dim, Number, fe_degree>::reinit_patch_to_global_indices(const unsigned int patch_id)
{
  AssertIndexRange(patch_id, subdomain_handler.get_partition_data().n_subdomains());
  patch_to_global_indices.resize(cell_to_patch_indices.size());
  std::vector<std::array<types::global_dof_index, macro_size>> first_dofs =
    patch_worker.get_dof_collection(patch_id);
  const unsigned n_cells = first_dofs.size();
  for(unsigned int cell_no = 0; cell_no < n_cells; ++cell_no)
  {
    const auto macro_dofs = first_dofs[cell_no];
    const auto patch_dofs = patch_dofs_on_cell(cell_no);
    for(unsigned int cell_dof = 0; cell_dof < n_dofs_per_cell; ++cell_dof)
    {
      const unsigned int patch_dof = patch_dofs[cell_dof];
      for(unsigned int lane = 0; lane < macro_size; ++lane)
        patch_to_global_indices[patch_dof][lane] = macro_dofs[lane] + cell_dof;
    }
  }
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



} // end namespace TPSS

#include "patch_transfer.templates.h"

#endif // end inclusion guard
