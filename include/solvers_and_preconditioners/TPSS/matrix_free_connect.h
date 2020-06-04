#ifndef MATRIXFREECONNECT_H
#define MATRIXFREECONNECT_H

#include <deal.II/matrix_free/matrix_free.h>

#include "dof_info.h"
#include "patch_info.h"
#include "patch_worker.h"



namespace TPSS
{
using namespace dealii;

  template<int dim, typename Number>
struct MatrixFreeConnect
{
  /**
   * This struct stores the dof indices to transfer patch local vectors to cell
   * local vectors (gather or restrict) and vice versa (scatter or prolongate).
   */
  struct DoFInfoLocal;

  static constexpr unsigned int macro_size = VectorizedArray<Number>::size();

  void
  initialize(const MatrixFree<dim, Number> *                     mf_storage_in,
             const ArrayView<const TPSS::DoFInfo<dim, Number>> & dof_infos_in,
             const std::map<unsigned int, unsigned int> &        dofh_index_map_in);

  void
  clear();

  unsigned int
  n_cells_plain() const;

  const TPSS::DoFInfo<dim, Number> &
  get_dof_info(const unsigned int dofh_index) const;

  const DoFInfoLocal &
  get_dof_info_local(const unsigned int dofh_index) const;

  /**
   * The underlying MatrixFree object used to map matrix-free infrastructure
   * to the patch distribution stored in PatchInfo
   */
  const dealii::MatrixFree<dim, Number> * mf_storage = nullptr;

  const PatchInfo<dim> * patch_info = nullptr;

  ArrayView<const DoFInfo<dim, Number>> dof_infos;

  /**
   * The batch index and vectorization lane pairs identifying cells
   * stored in the underlying MatrixFree object and corresponding to
   * cells stored in @p patch_info's internal data.
   */
  std::vector<std::pair<unsigned int, unsigned int>> batch_and_lane_index_pairs;

  std::map<unsigned int, unsigned int> dofh_index_map;

  std::vector<DoFInfoLocal> dof_infos_local;
};



template<int dim, typename Number>
struct MatrixFreeConnect<dim, Number>::DoFInfoLocal
{
  std::vector<unsigned int>                          dof_starts;
  std::vector<std::pair<unsigned int, unsigned int>> cell_and_patch_dof_indices;
  bool                                               is_intentionally_uninitialized = false;

  void
  clear()
  {
    dof_starts.clear();
    cell_and_patch_dof_indices.clear();
  }
};



// --------------------------------   MatrixFreeConnect   --------------------------------



template<int dim, typename Number>
inline void
MatrixFreeConnect<dim, Number>::clear()
{
  dof_infos.reinit(NULL, 0);
  patch_info = nullptr;
  mf_storage = nullptr;
  batch_and_lane_index_pairs.clear();
  dof_infos_local.clear();
}


template<int dim, typename Number>
inline unsigned int
MatrixFreeConnect<dim, Number>::n_cells_plain() const
{
  return batch_and_lane_index_pairs.size();
}


template<int dim, typename Number>
inline const TPSS::DoFInfo<dim, Number> &
MatrixFreeConnect<dim, Number>::get_dof_info(const unsigned int dofh_index) const
{
  AssertIndexRange(dofh_index, dofh_index_map.size());
  const auto unique_dofh_index = dofh_index_map.at(dofh_index);
  AssertIndexRange(unique_dofh_index, dof_infos_local.size());
  return dof_infos[unique_dofh_index];
}


template<int dim, typename Number>
inline const typename MatrixFreeConnect<dim, Number>::DoFInfoLocal &
MatrixFreeConnect<dim, Number>::get_dof_info_local(const unsigned int dofh_index) const
{
  AssertIndexRange(dofh_index, dofh_index_map.size());
  const auto unique_dofh_index = dofh_index_map.at(dofh_index);
  AssertIndexRange(unique_dofh_index, dof_infos_local.size());
  return dof_infos_local[unique_dofh_index];
}



} // end namespace TPSS

#include "matrix_free_connect.templates.h"

#endif // end inclusion guard
