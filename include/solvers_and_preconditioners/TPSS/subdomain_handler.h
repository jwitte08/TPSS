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
#include "mapping_info.h"
#include "patch_info.h"
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

  ~SubdomainHandler()
  {
    clear();
  }

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

  const dealii::DoFHandler<dim> &
  get_dof_handler() const;

  const dealii::Quadrature<1> &
  get_quadrature(int dimension = 0) const;

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

  std::vector<TimeInfo>
  get_time_data() const;

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
  std::shared_ptr<const dealii::MatrixFree<dim, number>> owned_mf_storage;
  const dealii::DoFHandler<dim> *                        dof_handler;

  TPSS::PatchInfo<dim>                 patch_info;
  TPSS::MatrixFreeConnect<dim, number> mf_connect;
  TPSS::MappingInfo<dim, number>       mapping_info;

  // TODO
  std::vector<dealii::Quadrature<1>> quadrature_storage;
  AdditionalData                     additional_data;
  std::vector<TimeInfo>              time_data;
};

template<int dim, typename number>
struct SubdomainHandler<dim, number>::AdditionalData
{
  unsigned int          level            = -1;
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::invalid;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;
  std::function<std::vector<std::vector<std::vector<CellIterator>>>(
    const DoFHandler<dim> *                             dof_handler,
    const typename TPSS::PatchInfo<dim>::AdditionalData additional_data)>
    manual_coloring_func_cp;
  std::function<std::vector<std::vector<PatchIterator>>(
    const DoFHandler<dim> *                             dof_handler,
    const typename TPSS::PatchInfo<dim>::AdditionalData additional_data,
    std::vector<std::vector<CellIterator>> &            cell_collections)>
    manual_coloring_func;
  std::function<void(const DoFHandler<dim> *                             dof_handler,
                     const typename TPSS::PatchInfo<dim>::AdditionalData additional_data,
                     std::vector<std::vector<CellIterator>> &            cell_collections)>
               manual_gathering_func;
  unsigned int n_q_points_surrogate      = 5;
  bool         normalize_surrogate_patch = false;
  bool         use_arc_length            = true;
  unsigned int n_threads                 = 0;
  unsigned int grain_size                = 0;
  bool         print_details             = false;
};

/*********************************** inline functions ***********************************/

template<int dim, typename number>
inline void
SubdomainHandler<dim, number>::clear()
{
  mf_storage = nullptr;
  owned_mf_storage.reset();
  dof_handler = nullptr;

  patch_info.clear();
  mf_connect = TPSS::MatrixFreeConnect<dim, number>{};
  mapping_info.clear();

  quadrature_storage.clear();
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
inline const dealii::DoFHandler<dim> &
SubdomainHandler<dim, number>::get_dof_handler() const
{
  Assert(dof_handler != nullptr, dealii::ExcNotInitialized());
  return *dof_handler;
}

template<int dim, typename number>
inline const dealii::Quadrature<1> &
SubdomainHandler<dim, number>::get_quadrature(int direction) const
{
  AssertIndexRange(direction, dim);
  Assert(quadrature_storage.size() == dim,
         dealii::ExcNotImplemented()); // each direction is filled

  return quadrature_storage[direction];
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
  return *(mf_connect.mf_storage);
}

template<int dim, typename number>
inline const typename TPSS::PatchInfo<dim>::PartitionData &
SubdomainHandler<dim, number>::get_partition_data() const
{
  return patch_info.subdomain_partition_data;
}

template<int dim, typename number>
inline std::vector<TimeInfo>
SubdomainHandler<dim, number>::get_time_data() const
{
  return time_data;
}

#include "subdomain_handler.templates.h"

#endif // inclusion guard
