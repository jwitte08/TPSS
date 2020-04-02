#ifndef SUBDOMAINHANDLER_H
#define SUBDOMAINHANDLER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include "TPSS.h"
#include "dof_info.h"
#include "mapping_info.h"
#include "matrix_free_connect.h"
#include "patch_worker.h"
#include "solvers_and_preconditioners/TPSS/time_info.h"

#include <array>
#include <memory>

template<int dim, typename number = double>
class SubdomainHandler
{
public:
  struct AdditionalData;

  using CellIterator  = typename TPSS::PatchInfo<dim>::CellIterator;
  using PatchIterator = typename TPSS::PatchInfo<dim>::PatchIterator;

  SubdomainHandler() = default;

  SubdomainHandler(const SubdomainHandler<dim> &) = delete;

  ~SubdomainHandler() = default;

  SubdomainHandler<dim> &
  operator=(const SubdomainHandler<dim> &) = delete;

  void
  reinit(const dealii::MatrixFree<dim, number> * mf_storage, const AdditionalData additional_data);

  void
  reinit(std::shared_ptr<const dealii::MatrixFree<dim, number>> mf_storage,
         const AdditionalData                                   additional_data);

  void
  clear();

  unsigned int
  n_physical_subdomains() const;

  unsigned int
  n_components() const;

  const AdditionalData &
  get_additional_data() const;

  unsigned int
  get_unique_dofh_index(const unsigned int dofh_index = 0) const;

  const dealii::DoFHandler<dim> &
  get_dof_handler(const unsigned int dofh_index = 0) const;

  const TPSS::DoFInfo<dim, number> &
  get_dof_info(const unsigned int dofh_index = 0) const;

  ArrayView<const TPSS::DoFInfo<dim, number>>
  get_dof_infos() const;

  TPSS::DoFLayout
  get_dof_layout(const unsigned int dofh_index = 0) const;

  const typename TPSS::PatchInfo<dim>::PartitionData &
  get_partition_data() const;

  const TPSS::PatchInfo<dim> &
  get_patch_info() const;

  const TPSS::MappingInfo<dim, number> &
  get_mapping_info() const;

  const TPSS::MatrixFreeConnect<dim, number> &
  get_matrixfree_connect() const;

  const dealii::MatrixFree<dim, number> &
  get_matrix_free() const;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<number>> &
  get_shape_info(const unsigned int dofh_index = 0) const;

  std::vector<TimeInfo>
  get_time_data() const;

  std::shared_ptr<const Utilities::MPI::Partitioner>
  get_vector_partitioner(const unsigned int dofh_index = 0) const;

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
  get_vector_partitioners() const;

  unsigned int
  guess_grain_size(const unsigned int n_subdomain_batches) const;

  /**
   * The loop applies the user-defined function @p patch_operation
   * successively to the half-open patch ranges (partitions)
   * associated to color @color, whereby the patch operation results
   * in the output @p output given the input @p input. If possible the
   * operation is performed in parallel by means of threads.
   */
  template<typename Input, typename Output>
  void
  loop(const std::function<void(const SubdomainHandler<dim, number> &,
                                Output &,
                                const Input &,
                                const std::pair<unsigned int, unsigned int> &)> & patch_operation,
       Output &                                                                   output,
       const Input &                                                              input,
       const unsigned int                                                         color) const;

  /**
   * The loop applies the user-defined function @p patch_operation
   * successively to the total range of patches, regardless of any
   * color and partition, in parallel by means of
   * thread-parallelism. However, this loop is merely feasible if the
   * operation @p patch_operation does not come along with any mutual
   * conflict between patches, e.g. a race condition occurs by writing
   * data into the output @p output.
   */
  template<typename Input, typename Output>
  void
  parloop(
    const std::function<void(const SubdomainHandler<dim, number> &,
                             Output &,
                             const Input &,
                             const std::pair<unsigned int, unsigned int> &)> & patch_operation,
    Output &                                                                   output,
    const Input &                                                              input) const;

private:
  /**
   * setup patch infrastructure, compute surrogate mapping, link to
   * matrix-free macro cells ...
   */
  void
  internal_reinit();

  const dealii::MatrixFree<dim, number> *                mf_storage;
  std::shared_ptr<const dealii::MatrixFree<dim, number>> mf_storage_owned;
  std::vector<unsigned int>                              dofh_indices;
  std::vector<const dealii::DoFHandler<dim> *>           dof_handlers;

  TPSS::PatchInfo<dim>                    patch_info;
  std::vector<TPSS::DoFInfo<dim, number>> dof_infos;
  TPSS::MatrixFreeConnect<dim, number>    mf_connect;
  TPSS::MappingInfo<dim, number>          mapping_info;

  AdditionalData        additional_data;
  std::vector<TimeInfo> time_data;
};

template<int dim, typename number>
struct SubdomainHandler<dim, number>::AdditionalData
{
  unsigned int          level            = numbers::invalid_unsigned_int;
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::invalid;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;
  std::function<std::vector<std::vector<PatchIterator>>(
    const std::vector<std::vector<CellIterator>> &      patches,
    const typename TPSS::PatchInfo<dim>::AdditionalData additional_data)>
    coloring_func;
  std::function<void(const DoFHandler<dim> *                             dof_handler,
                     const typename TPSS::PatchInfo<dim>::AdditionalData additional_data,
                     std::vector<std::vector<CellIterator>> &            cell_collections)>
    manual_gathering_func;

  /*
   * Oth table direction : dofh_index
   * 1st table direction : dummy
   */
  Table<2, internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<number>>> shape_infos;
  std::vector<std::set<types::boundary_id>>                                   dirichlet_ids;

  unsigned int n_q_points_surrogate      = 5;
  bool         normalize_surrogate_patch = false;
  bool         use_arc_length            = true;

  unsigned int          n_threads        = 0;
  unsigned int          grain_size       = 0;
  TPSS::CachingStrategy caching_strategy = TPSS::CachingStrategy::Cached;

  bool print_details = false;
};

/*********************************** inline functions ***********************************/

template<int dim, typename number>
inline void
SubdomainHandler<dim, number>::clear()
{
  mf_storage = nullptr;
  mf_storage_owned.reset();
  dofh_indices.clear();
  dof_handlers.clear();

  patch_info.clear();
  dof_infos.clear();
  mf_connect = TPSS::MatrixFreeConnect<dim, number>{};
  mapping_info.clear();

  additional_data = AdditionalData{};
  time_data.clear();
}

template<int dim, typename number>
inline unsigned int
SubdomainHandler<dim, number>::n_physical_subdomains() const
{
  const auto &       c_patch_info = patch_info;
  const unsigned int n_subdomains =
    TPSS::PatchWorker<dim, number>{c_patch_info}.n_physical_subdomains();
  return n_subdomains;
}

template<int dim, typename number>
inline unsigned int
SubdomainHandler<dim, number>::n_components() const
{
  // TODO MatrixFree with DoFHandler with more than one base_element
  AssertThrow(mf_storage != nullptr, ExcNotInitialized());
  return mf_storage->n_components();
}

template<int dim, typename number>
inline const typename SubdomainHandler<dim, number>::AdditionalData &
SubdomainHandler<dim, number>::get_additional_data() const
{
  return additional_data;
}

template<int dim, typename number>
unsigned int
SubdomainHandler<dim, number>::get_unique_dofh_index(const unsigned int dofh_index) const
{
  AssertIndexRange(dofh_index, dofh_indices.size());
  return dofh_indices[dofh_index];
}

template<int dim, typename number>
inline const dealii::DoFHandler<dim> &
SubdomainHandler<dim, number>::get_dof_handler(const unsigned int dofh_index) const
{
  const auto unique_dofh_index = get_unique_dofh_index(dofh_index);
  return *(dof_handlers[unique_dofh_index]);
}

template<int dim, typename number>
inline const TPSS::DoFInfo<dim, number> &
SubdomainHandler<dim, number>::get_dof_info(const unsigned int dofh_index) const
{
  const auto unique_dofh_index = get_unique_dofh_index(dofh_index);
  return dof_infos[unique_dofh_index];
}

template<int dim, typename number>
inline ArrayView<const TPSS::DoFInfo<dim, number>>
SubdomainHandler<dim, number>::get_dof_infos() const
{
  return ArrayView<const TPSS::DoFInfo<dim, number>>(dof_infos);
}

template<int dim, typename number>
inline TPSS::DoFLayout
SubdomainHandler<dim, number>::get_dof_layout(const unsigned int dofh_index) const
{
  const auto & dof_info = get_dof_info(dofh_index);
  return dof_info.get_dof_layout();
}

template<int dim, typename number>
inline const TPSS::PatchInfo<dim> &
SubdomainHandler<dim, number>::get_patch_info() const
{
  return patch_info;
}

template<int dim, typename number>
inline const TPSS::MappingInfo<dim, number> &
SubdomainHandler<dim, number>::get_mapping_info() const
{
  return mapping_info;
}

template<int dim, typename number>
inline const TPSS::MatrixFreeConnect<dim, number> &
SubdomainHandler<dim, number>::get_matrixfree_connect() const
{
  return mf_connect;
}

template<int dim, typename number>
inline const dealii::MatrixFree<dim, number> &
SubdomainHandler<dim, number>::get_matrix_free() const
{
  return *mf_storage;
}

template<int dim, typename number>
inline const typename TPSS::PatchInfo<dim>::PartitionData &
SubdomainHandler<dim, number>::get_partition_data() const
{
  return patch_info.subdomain_partition_data;
}


template<int dim, typename number>
inline const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<number>> &
SubdomainHandler<dim, number>::get_shape_info(const unsigned int dofh_index) const
{
  AssertIndexRange(dofh_index, n_components());
  AssertDimension(additional_data.shape_infos.size(1), 1);
  if(!additional_data.shape_infos.empty())
    return additional_data.shape_infos(dofh_index, /*dummy*/ 0); // !!!
  /// assuming isotropy of tensor product elements and quadrature
  Assert(this->mf_storage, ExcMessage("MatrixFree object not initialized."));
  return get_matrix_free().get_shape_info(dofh_index, /*quad_index*/ 0);
}


template<int dim, typename number>
inline std::vector<TimeInfo>
SubdomainHandler<dim, number>::get_time_data() const
{
  return time_data;
}


template<int dim, typename number>
inline std::shared_ptr<const Utilities::MPI::Partitioner>
SubdomainHandler<dim, number>::get_vector_partitioner(const unsigned int dofh_index) const
{
  const auto & dof_info = get_dof_info(dofh_index);
  return dof_info.vector_partitioner;
}


template<int dim, typename number>
inline std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
SubdomainHandler<dim, number>::get_vector_partitioners() const
{
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
  for(auto dofh_index = 0U; dofh_index < n_components(); ++dofh_index)
    partitioners.push_back(get_vector_partitioner(dofh_index));
  return partitioners;
}

#include "subdomain_handler.templates.h"

#endif // inclusion guard
