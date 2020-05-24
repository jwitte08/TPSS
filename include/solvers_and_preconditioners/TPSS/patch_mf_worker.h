#ifndef PATCH_MF_WORKER_H
#define PATCH_MF_WORKER_H

#include "matrix_free_connect.h"
#include "patch_dof_worker.h"

using namespace dealii;



namespace TPSS
{
template<int dim, typename Number>
class PatchMFWorker : public PatchDoFWorker<dim, Number>
{
public:
  using Base                               = PatchDoFWorker<dim, Number>;
  static constexpr unsigned int macro_size = Base::macro_size;

  PatchMFWorker() = delete;

  PatchMFWorker(const MatrixFreeConnect<dim, Number> & mf_connect_in,
                const unsigned int                     dofh_index = 0);

  PatchMFWorker(const PatchMFWorker &) = delete;

  PatchMFWorker &
  operator=(const PatchMFWorker &) = delete;

  void
  connect_to_matrixfree(MatrixFreeConnect<dim, Number> & mf_connect);

  std::pair<unsigned int, unsigned int>
  get_batch_and_lane_index(const unsigned int patch_id,
                           const unsigned int cell_no,
                           const unsigned int lane) const;

  /**
   * Returns the collection of batch-index-and-lane pairs describing
   * the macro cell collection @p patch_id in MatrixFree speak.
   */
  std::vector<std::array<std::pair<unsigned int, unsigned int>, macro_size>>
  get_batch_collection(unsigned int patch_id) const;

  ArrayView<const std::pair<unsigned int, unsigned int>>
  get_local_cell_and_patch_dof_indices(const unsigned int patch_id,
                                       const unsigned int cell_no,
                                       const unsigned int lane) const;

  unsigned int
  get_local_dof_start(const unsigned int patch_id,
                      const unsigned int cell_no,
                      const unsigned int lane) const;

  unsigned int
  n_dofs_on_patch(const unsigned int patch_id, const unsigned int lane) const;

  unsigned int
  n_dofs_per_cell_on_patch(const unsigned int patch_id,
                           const unsigned int cell_no,
                           const unsigned int lane) const;

private:
  const MatrixFreeConnect<dim, Number> * const                        mf_connect;
  const typename MatrixFreeConnect<dim, Number>::DoFInfoLocal * const dof_info_local;
};



// --------------------------------   PatchMFWorker   --------------------------------



template<int dim, typename Number>
inline PatchMFWorker<dim, Number>::PatchMFWorker(
  const MatrixFreeConnect<dim, Number> & mf_connect_in,
  const unsigned int                     dofh_index)
  : PatchDoFWorker<dim, Number>(mf_connect_in.get_dof_info(dofh_index)),
    mf_connect(&mf_connect_in),
    dof_info_local(&(mf_connect_in.get_dof_info_local(dofh_index)))
{
  AssertThrow(mf_connect, ExcMessage("mf_connect is not set"));
  AssertThrow(mf_connect->patch_info, ExcMessage("mf_connect is not initialized."));
  /// Check wether we can stride through @p dof_starts and @p batch_and_lane_index_pairs by means of
  /// PatchWorker::get_cell_position()
  if(!dof_info_local->is_intentionally_uninitialized) // TODO !!!
    AssertDimension(this->patch_info->n_cells_plain() + 1, dof_info_local->dof_starts.size());
  AssertDimension(this->patch_info->n_cells_plain(), mf_connect->batch_and_lane_index_pairs.size());
}


template<int dim, typename Number>
inline std::pair<unsigned int, unsigned int>
PatchMFWorker<dim, Number>::get_batch_and_lane_index(const unsigned int patch_id,
                                                     const unsigned int cell_no,
                                                     const unsigned int lane) const
{
  Assert(mf_connect, ExcMessage("MatrixFreeConnect is not set."));
  AssertIndexRange(patch_id, this->get_partition_data().n_subdomains());
  AssertIndexRange(lane, this->n_lanes_filled(patch_id));
  AssertIndexRange(cell_no, this->n_cells_per_subdomain());
  const auto cell_position = this->get_cell_position(patch_id, cell_no, lane);
  return mf_connect->batch_and_lane_index_pairs[cell_position];
}


template<int dim, typename Number>
inline std::vector<
  std::array<std::pair<unsigned int, unsigned int>, PatchMFWorker<dim, Number>::macro_size>>
PatchMFWorker<dim, Number>::get_batch_collection(unsigned int patch_id) const
{
  AssertIndexRange(patch_id, this->patch_info->subdomain_partition_data.n_subdomains());
  const auto n_lanes_filled = this->n_lanes_filled(patch_id);

  std::vector<std::array<std::pair<unsigned int, unsigned int>, macro_size>> collection(
    this->n_cells_per_subdomain());
  for(auto cell_no = 0U; cell_no < collection.size(); ++cell_no)
  {
    for(auto lane = 0U; lane < n_lanes_filled; ++lane)
      collection[cell_no][lane] = get_batch_and_lane_index(patch_id, cell_no, lane);
    for(auto lane = n_lanes_filled; lane < macro_size; ++lane)
      collection[cell_no][lane] = get_batch_and_lane_index(patch_id, cell_no, 0);
  }
  return collection;
}


template<int dim, typename Number>
inline ArrayView<const std::pair<unsigned int, unsigned int>>
PatchMFWorker<dim, Number>::get_local_cell_and_patch_dof_indices(const unsigned int patch_id,
                                                                 const unsigned int cell_no,
                                                                 const unsigned int lane) const
{
  Assert(dof_info_local, ExcMessage("Local dof info is not set."));
  const auto dof_start = get_local_dof_start(patch_id, cell_no, lane);
  const auto n_dofs    = n_dofs_per_cell_on_patch(patch_id, cell_no, lane);
  AssertIndexRange(dof_start + n_dofs, dof_info_local->cell_and_patch_dof_indices.size() + 1);
  const auto start = dof_info_local->cell_and_patch_dof_indices.data() + dof_start;
  return ArrayView<const std::pair<unsigned int, unsigned int>>(start, n_dofs);
}


template<int dim, typename Number>
inline unsigned int
PatchMFWorker<dim, Number>::get_local_dof_start(const unsigned int patch_id,
                                                const unsigned int cell_no,
                                                const unsigned int lane) const
{
  const auto cell_position = this->get_cell_position(patch_id, cell_no, lane);
  const auto dof_start     = dof_info_local->dof_starts[cell_position];
  return dof_start;
}


template<int dim, typename Number>
inline unsigned int
PatchMFWorker<dim, Number>::n_dofs_on_patch(const unsigned int patch_id,
                                            const unsigned int lane) const
{
  // TODO !!!
  (void)patch_id;
  (void)lane;
  return Base::n_dofs(); // assuming isotropy
}


template<int dim, typename Number>
inline unsigned int
PatchMFWorker<dim, Number>::n_dofs_per_cell_on_patch(const unsigned int patch_id,
                                                     const unsigned int cell_no,
                                                     const unsigned int lane) const
{
  const auto cell_position = this->get_cell_position(patch_id, cell_no, lane);
  const auto dof_start     = dof_info_local->dof_starts[cell_position];
  AssertDimension(get_local_dof_start(patch_id, cell_no, lane), dof_start);
  const auto dof_start_next = dof_info_local->dof_starts[cell_position + 1];
  return dof_start_next - dof_start;
}



} // end namespace TPSS

#endif // end inclusion guard
