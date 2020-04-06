#ifndef FEEVALUATION_PATCH_H
#define FEEVALUATION_PATCH_H

#include <deal.II/matrix_free/fe_evaluation.h>

#include "patch_mf_worker.h"
#include "subdomain_handler.h"

using namespace dealii;



namespace TPSS
{
template<int dim, int fe_degree, int n_q_points_1d = fe_degree + 1, typename Number = double>
class FEEvaluationPatch : public PatchMFWorker<dim, Number>
{
public:
  using cell_evaluator_type                = FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>;
  using patch_worker_type                  = PatchMFWorker<dim, Number>;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::size();

  FEEvaluationPatch(const SubdomainHandler<dim, Number> & sd_handler_in,
                    const unsigned int                    dofh_index = 0);

  ~FEEvaluationPatch()
  {
    cell_eval.reset();
    current_patch_id   = numbers::invalid_unsigned_int;
    current_patch_lane = numbers::invalid_unsigned_int;
    current_cell_no    = numbers::invalid_unsigned_int;
  }

  void
  reinit(const unsigned int patch_id, const unsigned int patch_lane);

  void
  reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec) const;

  /**
   * Restrict patch dof values @p src to cell dof values at cell @p cell_no.
   */
  void
  gather(const unsigned int cell_no, const ArrayView<const VectorizedArray<Number>> & src);

  /**
   * We add the uniquely assigned cell dof values @p src of cell @p cell_no to
   * the corresponding patch dof values @p dst.
   */
  void
  scatter_add(const ArrayView<VectorizedArray<Number>> & dst);

  /**
   * TODO...  How do we treat cell evaluator: inherit? cheap enough to multiply
   * over all patches?
   */
  std::shared_ptr<cell_evaluator_type> cell_eval;

private:
  unsigned int current_patch_id   = numbers::invalid_unsigned_int;
  unsigned int current_patch_lane = numbers::invalid_unsigned_int;
  unsigned int current_cell_no    = numbers::invalid_unsigned_int;
};



// --------------------------------   FEEvaluationPatch   --------------------------------



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
FEEvaluationPatch<dim, fe_degree, n_q_points_1d, Number>::FEEvaluationPatch(
  const SubdomainHandler<dim, Number> & sd_handler_in,
  const unsigned int                    dofh_index)
  : PatchMFWorker<dim, Number>(sd_handler_in.get_matrixfree_connect(), dofh_index),
    cell_eval(std::make_shared<cell_evaluator_type>(sd_handler_in.get_matrix_free(), dofh_index))
{
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FEEvaluationPatch<dim, fe_degree, n_q_points_1d, Number>::reinit(const unsigned int patch_id,
                                                                 const unsigned int patch_lane)
{
  AssertIndexRange(patch_id, patch_worker_type::get_partition_data().n_subdomains());
  AssertIndexRange(patch_lane, patch_worker_type::n_lanes_filled(patch_id));
  current_patch_id   = patch_id;
  current_patch_lane = patch_lane;
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FEEvaluationPatch<dim, fe_degree, n_q_points_1d, Number>::reinit_local_vector(
  AlignedVector<VectorizedArray<Number>> & vec) const
{
  vec.resize(patch_worker_type::n_dofs_on_patch(current_patch_id, 0));
}


/**
 * Restrict patch dof values @p src to cell dof values at cell @p cell_no.
 */
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FEEvaluationPatch<dim, fe_degree, n_q_points_1d, Number>::gather(
  const unsigned int                               cell_no,
  const ArrayView<const VectorizedArray<Number>> & src)
{
  AssertIndexRange(current_patch_id, patch_worker_type::get_partition_data().n_subdomains());
  AssertIndexRange(current_patch_lane, patch_worker_type::n_lanes_filled(current_patch_id));
  AssertDimension(src.size(),
                  patch_worker_type::n_dofs_on_patch(current_patch_id, current_patch_lane));
  AssertIndexRange(cell_no, patch_worker_type::n_cells_per_subdomain());
  // TODO
  // AssertDimension(cell_eval->dofs_per_cell,
  //                 patch_worker_type::n_dofs_per_cell_on_patch(current_patch_id, cell_no,
  //                 current_patch_lane));
  current_cell_no = cell_no;

  /// Initialize FEEvaluation.
  const auto [batch_index, batch_lane] =
    this->get_batch_and_lane_index(current_patch_id, current_cell_no, current_patch_lane);
  (void)batch_lane;
  cell_eval->reinit(batch_index);

  /// Clear dof values.
  for(auto dof = 0U; dof < cell_eval->dofs_per_cell; ++dof)
    cell_eval->submit_dof_value(0., dof);

  /// Fill dof values (same value on each lane).
  const auto & cell_and_patch_dof_indices =
    patch_worker_type::get_local_cell_and_patch_dof_indices(current_patch_id,
                                                            cell_no,
                                                            current_patch_lane);
  for(const auto [cell_dof_index, patch_dof_index] : cell_and_patch_dof_indices)
    cell_eval->submit_dof_value(src[patch_dof_index][current_patch_lane], cell_dof_index);
}


/**
 * We add the uniquely assigned cell dof values @p src of cell @p cell_no to
 * the corresponding patch dof values @p dst.
 */
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FEEvaluationPatch<dim, fe_degree, n_q_points_1d, Number>::scatter_add(
  const ArrayView<VectorizedArray<Number>> & dst)
{
  AssertIndexRange(current_patch_id, patch_worker_type::get_partition_data().n_subdomains());
  AssertIndexRange(current_patch_lane, patch_worker_type::n_lanes_filled(current_patch_id));
  AssertDimension(dst.size(),
                  patch_worker_type::n_dofs_on_patch(current_patch_id, current_patch_lane));
  // TODO
  // AssertDimension(cell_eval->dofs_per_cell,
  //                 patch_worker_type::n_dofs_per_cell_on_patch(current_patch_id,
  //                                                current_cell_no,
  //                                                current_patch_lane));
  AssertIndexRange(current_cell_no, patch_worker_type::n_cells_per_subdomain());

  /// Unique distribution of dof values to patch local field @p dst
  const auto [batch_index, batch_lane] =
    this->get_batch_and_lane_index(current_patch_id, current_cell_no, current_patch_lane);
  (void)batch_index;
  const auto & cell_and_patch_dof_indices =
    patch_worker_type::get_local_cell_and_patch_dof_indices(current_patch_id,
                                                            current_cell_no,
                                                            current_patch_lane);
  for(const auto [cell_dof_index, patch_dof_index] : cell_and_patch_dof_indices)
    dst[patch_dof_index][current_patch_lane] +=
      (cell_eval->get_dof_value(cell_dof_index))[batch_lane];
}



} // end namespace TPSS

#endif // end inclusion guard
