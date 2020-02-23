#ifndef FEEVALUATION_PATCH_H
#define FEEVALUATION_PATCH_H

#include <deal.II/matrix_free/fe_evaluation.h>

#include "subdomain_handler.h"

using namespace dealii;



namespace TPSS
{
template<int dim, int fe_degree, int n_q_points_1d = fe_degree + 1, typename Number = double>
class FEEvaluationPatch
{
public:
  using evaluator_type                     = FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  FEEvaluationPatch(const SubdomainHandler<dim, Number> & sd_handler_in,
                    const unsigned int                    dofh_index = 0)
    : eval(std::make_shared<evaluator_type>(sd_handler_in.get_matrix_free(), dofh_index)),
      patch_mf_worker(sd_handler_in.get_matrixfree_connect()),
      index_helper(fe_degree + 1)
  {
  }

  void
  reinit(const unsigned int patch_id, const unsigned int patch_lane)
  {
    AssertIndexRange(patch_id, patch_mf_worker.get_partition_data().n_subdomains())
      AssertIndexRange(patch_lane, patch_mf_worker.n_lanes_filled(patch_id));
    current_patch_id   = patch_id;
    current_patch_lane = patch_lane;
    index_helper.reinit(std::move(patch_mf_worker.fill_face_infos(patch_id, patch_lane)));
  }

  void
  reinit_local_vector(AlignedVector<VectorizedArray<Number>> & vec,
                      const unsigned int                       cell_no) const
  {
    vec.resize(index_helper.n_dofs_per_cell_on_patch(cell_no));
  }

  std::pair<unsigned int, unsigned int>
  get_batch_and_lane_index(const unsigned int cell_no) const
  {
    const auto cell_position =
      patch_mf_worker.get_cell_position(current_patch_id, cell_no, current_patch_lane);
    return patch_mf_worker.get_batch_and_lane_index(cell_position);
  }

  /**
   * Restrict patch dof values @p src to cell dof values at cell @p cell_no.
   */
  // AlignedVector<VectorizedArray<Number>>
  void
  gather(const unsigned int cell_no, const ArrayView<const VectorizedArray<Number>> & src)
  {
    AssertIndexRange(current_patch_id, patch_mf_worker.get_partition_data().n_subdomains());
    AssertIndexRange(current_patch_lane, patch_mf_worker.n_lanes_filled(current_patch_id));
    AssertDimension(src.size(), index_helper.n_dofs());
    AssertDimension(eval->dofs_per_cell, index_helper.n_dofs_per_cell(cell_no));

    // /// Initialize FEEvaluation.
    // const auto [batch_index, batch_lane] = get_batch_and_lane_index(cell_no);
    // eval->reinit(batch_index);

    // /// Clear dof values.
    // for(auto dof = 0U; dof < eval->dofs_per_cell; ++dof)
    //   eval->submit_dof_value(0., dof);

    // /// Fill dof values.
    // const auto & cell_dof_indices = index_helper.get_cell_dof_indices(cell_no);
    // const auto   n_cell_dofs      = cell_dof_indices.size();
    // for(auto cell_dof_no = 0U; cell_dof_no < n_cell_dofs; ++cell_dof_no)
    // {
    //   const auto cell_dof_index  = cell_dof_indices[cell_dof_no];
    //   const auto patch_dof_index = index_helper.dof_index(cell_no, cell_dof_no);
    //   eval->submit_dof_value(src[patch_dof_index][current_patch_lane], cell_dof_index);
    // }
  }

  /**
   * Additive distribution of uniquely assigned cell dof values @p src at cell
   * @p cell_no to patch dof values @p dst.
   */
  template<typename VectorType>
  void
  scatter_add(const ArrayView<VectorizedArray<Number>> & dst)
  // const ArrayView<const VectorizedArray<Number>> & src,
  // const unsigned int cell_no)
  {
    AssertIndexRange(current_patch_id, patch_mf_worker.get_partition_data().n_subdomains());
    AssertIndexRange(current_patch_lane, patch_mf_worker.n_lanes_filled(current_patch_id));
    AssertDimension(dst.size(), index_helper.n_dofs());
    AssertDimension(eval->dofs_per_cell, index_helper.n_dofs_per_cell(current_cell_no));

    // /// Unique distribution of dof values to patch local field @p dst
    // const auto & cell_dof_indices = index_helper.get_cell_dof_indices_unique(current_cell_no);
    // const auto   n_cell_dofs      = cell_dof_indices.size();
    // for(auto cell_dof_no = 0U; cell_dof_no < n_cell_dofs; ++cell_dof_no)
    // {
    //   const auto cell_dof_index  = cell_dof_indices[cell_dof_no];
    //   const auto patch_dof_index = index_helper.dof_index(current_cell_no, cell_dof_no);
    //   dst[patch_dof_index][current_patch_lane] =
    //   (eval->get_dof_value(cell_dof_index))[batch_lane];
    // }
  }

  std::shared_ptr<evaluator_type> eval;

private:
  PatchMFWorker<dim, Number> patch_mf_worker;
  PatchLocalHelperQ<dim>     index_helper;
  unsigned int               current_patch_id   = numbers::invalid_unsigned_int;
  unsigned int               current_patch_lane = numbers::invalid_unsigned_int;
  unsigned int               current_cell_no    = numbers::invalid_unsigned_int;
};



} // end namespace TPSS

#endif // end inclusion guard
