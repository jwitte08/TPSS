#ifndef MATRIXFREECONNECT_H
#define MATRIXFREECONNECT_H

#include "patch_info.h"
#include "patch_worker.h"



namespace TPSS
{
template<int dim, typename Number>
struct MatrixFreeConnect
{
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  void
  initialize(const MatrixFree<dim, Number> * mf_storage_in,
             const TPSS::PatchInfo<dim> *    patch_info_in);

  void
  clear()
  {
    patch_info = nullptr;
    mf_storage = nullptr;
    batch_and_lane_index_pairs.clear();
  }

  unsigned int
  n_cells_plain() const
  {
    return batch_and_lane_index_pairs.size();
  }

  std::pair<unsigned int, unsigned int>
  get_batch_and_lane_index(const unsigned int cell_position) const
  {
    Assert(patch_info, ExcMessage("patch_info is not set."));
    AssertDimension(patch_info->n_cells_plain(), n_cells_plain());
    AssertIndexRange(cell_position, n_cells_plain());
    return batch_and_lane_index_pairs[cell_position];
  }

  /**
   * The underlying MatrixFree object used to map matrix-free infrastructure
   * to the patch distribution stored in PatchInfo
   */
  const dealii::MatrixFree<dim, Number> * mf_storage = nullptr;

  const PatchInfo<dim> * patch_info = nullptr;

  /**
   * The batch index and vectorization lane pairs identifying cells
   * stored in the underlying MatrixFree object and corresponding to
   * cells stored in @p patch_info's internal data.
   */
  std::vector<std::pair<unsigned int, unsigned int>> batch_and_lane_index_pairs;
};



template<int dim, typename Number>
class PatchMFWorker : public PatchWorker<dim, Number>
{
public:
  using patch_worker                       = PatchWorker<dim, Number>;
  static constexpr unsigned int macro_size = patch_worker::macro_size;

  PatchMFWorker() = delete;

  PatchMFWorker(const MatrixFreeConnect<dim, Number> & mf_connect_in);

  PatchMFWorker(const PatchMFWorker &) = delete;

  PatchMFWorker &
  operator=(const PatchMFWorker &) = delete;

  void
  connect_to_matrixfree(MatrixFreeConnect<dim, Number> & mf_connect);

  /**
   * Returns the collection of batch-index-and-lane pairs describing
   * the macro cell collection @p patch_id in MatrixFree speak.
   */
  std::vector<std::array<std::pair<unsigned int, unsigned int>, macro_size>>
  get_batch_collection(unsigned int patch_id) const;

private:
  const MatrixFreeConnect<dim, Number> * const mf_connect;
};



template<int dim, typename Number>
inline PatchMFWorker<dim, Number>::PatchMFWorker(
  const MatrixFreeConnect<dim, Number> & mf_connect_in)
  : PatchWorker<dim, Number>(*(mf_connect_in.patch_info)), mf_connect(&mf_connect_in)
{
  AssertThrow(mf_connect, ExcMessage("mf_connect is not set"));
  AssertThrow(mf_connect->patch_info, ExcMessage("mf_connect is not iniatialized."));
}


template<int dim, typename Number>
inline std::vector<
  std::array<std::pair<unsigned int, unsigned int>, PatchMFWorker<dim, Number>::macro_size>>
PatchMFWorker<dim, Number>::get_batch_collection(unsigned int patch_id) const
{
  Assert(this->patch_info, ExcMessage("patch_info not initialized."));
  Assert(mf_connect, ExcMessage("mf_connect not initalized."));
  AssertIndexRange(patch_id, this->patch_info->subdomain_partition_data.n_subdomains());

  const auto n_lanes_filled = this->n_lanes_filled(patch_id);
  /// fill the empty vectorization lanes by copying the first lane
  const auto get_cell_position_filled = [&](const auto cell_no, const auto lane) {
    AssertIndexRange(lane, this->macro_size);
    if(lane < n_lanes_filled)
      return this->get_cell_position(patch_id, cell_no, lane);
    else
      return this->get_cell_position(patch_id, cell_no, 0);
  };

  std::vector<std::array<std::pair<unsigned int, unsigned int>, macro_size>> collection(
    this->patch_size);
  for(auto cell_no = 0U; cell_no < collection.size(); ++cell_no)
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      const auto cell_position  = get_cell_position_filled(cell_no, lane);
      collection[cell_no][lane] = mf_connect->get_batch_and_lane_index(cell_position);
    }
  return collection;
}



} // end namespace TPSS

#include "matrix_free_connect.templates.h"

#endif // end inclusion guard
