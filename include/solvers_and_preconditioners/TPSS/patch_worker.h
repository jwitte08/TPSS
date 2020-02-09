#ifndef PATCH_WORKER_H
#define PATCH_WORKER_H

#include "patch_info.h"

namespace TPSS
{
/**
 * A worker class re-interpreting the raw patch data in PatchInfo with
 * respect to a vectorized structure. The new alignment is uniquely
 * determined by the PartitionData @p subdomain_partition_data and @p
 * patch_starts stored in the PatchInfo. In addition, we extract
 * physical informations on subdomains and store them in the
 * PatchInfo, e.g. @p at_boundary_mask fields.
 *
 * Partitions per color are set as follows:
 * 1st partition = interior incomplete
 * 2nd partition = boundary incomplete
 * 3rd partition = interior complete
 * 4th partition = boundary complete
 *
 * An incomplete macro patch is a duplication of a physical patch
 * along all vectorization lanes. In other words incomplete patches
 * occur when the amount of physical patches is not divisible by the
 * vectorization length.  An interior macro patch contains several
 * interior patches.
 */
template<int dim, typename number>
class PatchWorker
{
public:
  static constexpr unsigned int macro_size = dealii::VectorizedArray<number>::n_array_elements;
  using CellIterator                       = typename PatchInfo<dim>::CellIterator;
  enum class RangeVariant
  {
    all,
    complete,
    incomplete
  };

  PatchWorker() = delete;

  PatchWorker(PatchInfo<dim> & patch_info);

  PatchWorker(const PatchInfo<dim> & patch_info);

  PatchWorker(const PatchInfo<dim> & patch_info, const MatrixFreeConnect<dim, number> & mf_connect);

  PatchWorker(const PatchWorker &) = delete;

  PatchWorker &
  operator=(const PatchWorker &) = delete;

  /**
   * Based on the data of the flat and unvectorized patch distribution
   * in PatchInfo::InternalData we are able to partition subdomains
   * into predicated groups.
   */
  void
  compute_partition_data(typename PatchInfo<dim>::PartitionData &      partition_data,
                         const typename PatchInfo<dim>::InternalData * internal_data,
                         std::vector<unsigned int> * patch_starts = nullptr) const;

  void
  connect_to_matrixfree(MatrixFreeConnect<dim, number> & mf_connect);


  std::array<unsigned int, GeometryInfo<dim>::faces_per_cell>
  get_at_boundary_masks_flat(const unsigned int patch) const;

  std::array<std::bitset<PatchWorker<dim, number>::macro_size>, GeometryInfo<dim>::faces_per_cell>
  get_at_boundary_masks(const unsigned int patch) const;

  /**
   * Returns the collection of batch-index-and-lane pairs describing
   * the macro cell collection @p patch_id in MatrixFree speak.
   */
  std::vector<std::array<std::pair<unsigned int, unsigned int>, macro_size>>
  get_batch_collection(unsigned int patch_id) const;

  /**
   * Returns the collection of macro cells describing the macro patch
   * @p patch_id subject to a lexicographical ordering.
   */
  std::vector<std::array<CellIterator, macro_size>>
  get_cell_collection(unsigned int patch_id) const;

  std::vector<ArrayView<const CellIterator>>
  get_cell_collection_views(unsigned int patch_id) const;

  std::size_t
  get_cell_position(const unsigned int patch_id,
                    const unsigned int cell_no,
                    const unsigned int lane) const
  {
    Assert(patch_info, ExcMessage("PatchInfo not initialized."));
    AssertIndexRange(patch_id, patch_info->subdomain_partition_data.n_subdomains());
    AssertIndexRange(cell_no, patch_size);
    AssertIndexRange(lane, n_lanes_filled(patch_id));
    return patch_info->patch_starts[patch_id] + lane * patch_size + cell_no;
  }

  const typename PatchInfo<dim>::PartitionData &
  get_partition_data() const;

  std::pair<unsigned int, unsigned int>
  get_owned_range(const unsigned int color,
                  const RangeVariant range_variant = RangeVariant::all) const
  {
    const auto & partition_data   = get_partition_data();
    const auto   complete_range   = partition_data.get_patch_range(0, color);
    const auto   incomplete_range = partition_data.get_patch_range(1, color);
    if(RangeVariant::all == range_variant)
    {
      AssertDimension(complete_range.second, incomplete_range.first);
      return {complete_range.first, incomplete_range.second};
    }
    else if(RangeVariant::complete == range_variant)
      return complete_range;
    else if(RangeVariant::incomplete == range_variant)
      return incomplete_range;
    else
      AssertThrow(false, ExcMessage("Invalid range variant."));
    return {0, 0};
  }

  std::pair<unsigned int, unsigned int>
  get_ghost_range(const unsigned int color,
                  const RangeVariant range_variant = RangeVariant::all) const
  {
    const auto & partition_data   = get_partition_data();
    const auto   complete_range   = partition_data.get_patch_range(2, color);
    const auto   incomplete_range = partition_data.get_patch_range(3, color);
    if(RangeVariant::all == range_variant)
    {
      AssertDimension(complete_range.second, incomplete_range.first);
      return {complete_range.first, incomplete_range.second};
    }
    else if(RangeVariant::complete == range_variant)
      return complete_range;
    else if(RangeVariant::incomplete == range_variant)
      return incomplete_range;
    else
      AssertThrow(false, ExcMessage("Invalid range variant."));
    return {0, 0};
  }

  unsigned int
  n_physical_subdomains() const;

  unsigned int
  n_lanes_filled(const unsigned int patch_id) const;

protected:
  /**
   * This method partitions the (unvectorized) patches, contained in
   * PatchInfo, into interior/boundary, incomplete/complete groups,
   * of vectorized patches, so-called macro patches.
   */
  void
  partition_patches(PatchInfo<dim> & patch_info);

  /**
   * Checks whether the partitions for the given color are valid.
   */
  static void
  has_valid_state(const typename PatchInfo<dim>::PartitionData & subdomain_partition_data,
                  const unsigned int                             color);

  unsigned int
  n_lanes_filled_impl(const unsigned int patch_id) const;

  const PatchInfo<dim> * const patch_info;

  const unsigned int patch_size = 0;

  const MatrixFreeConnect<dim, number> * const mf_connect = nullptr;

  mutable std::vector<CellIterator> cell_iterators_scratchpad;
};



// --------------------------------   PatchWorker   --------------------------------

template<int dim, typename number>
PatchWorker<dim, number>::PatchWorker(const PatchInfo<dim> & patch_info_in)
  : patch_info(&patch_info_in),
    patch_size(UniversalInfo<dim>::n_cells(patch_info_in.get_additional_data().patch_variant))
{
  AssertThrow(patch_info_in.get_additional_data().patch_variant != TPSS::PatchVariant::invalid,
              ExcInvalidState());
  typename PatchInfo<dim>::PartitionData subdomain_partition_data;
  compute_partition_data(subdomain_partition_data, patch_info_in.get_internal_data());
  const bool partition_data_is_valid =
    subdomain_partition_data.is_compatible(patch_info_in.subdomain_partition_data);
  (void)partition_data_is_valid;
  Assert(partition_data_is_valid, ExcMessage("The PartitionData does not fit the InternalData."));
}


template<int dim, typename number>
PatchWorker<dim, number>::PatchWorker(const PatchInfo<dim> &                 patch_info_in,
                                      const MatrixFreeConnect<dim, number> & mf_connect_in)
  : patch_info(&patch_info_in),
    patch_size(UniversalInfo<dim>::n_cells(patch_info_in.get_additional_data().patch_variant)),
    mf_connect(&mf_connect_in)
{
  AssertThrow(patch_info_in.get_additional_data().patch_variant != TPSS::PatchVariant::invalid,
              ExcInvalidState());
  typename PatchInfo<dim>::PartitionData subdomain_partition_data;
  compute_partition_data(subdomain_partition_data, patch_info_in.get_internal_data());
  const bool partition_data_is_valid =
    subdomain_partition_data.is_compatible(patch_info_in.subdomain_partition_data);
  (void)partition_data_is_valid;
  Assert(partition_data_is_valid, ExcMessage("The PartitionData does not fit the InternalData."));
}


template<int dim, typename number>
PatchWorker<dim, number>::PatchWorker(PatchInfo<dim> & patch_info_in)
  : patch_info(&patch_info_in),
    patch_size(UniversalInfo<dim>::n_cells(patch_info_in.get_additional_data().patch_variant))
{
  AssertThrow(patch_info_in.get_additional_data().patch_variant != TPSS::PatchVariant::invalid,
              ExcInvalidState());

  /**
   * If the given patch_info_in is correctly initialized there is no work
   */
  typename PatchInfo<dim>::PartitionData subdomain_partition_data;
  compute_partition_data(subdomain_partition_data, patch_info_in.get_internal_data());
  if(subdomain_partition_data.is_compatible(patch_info_in.subdomain_partition_data))
    return;

  /**
   * Initialize the vectorized infrastructure for the patches stored
   * in the PatchInfo. Partitioning into boundary/interior and
   * complete/incomplete patches is presented by @p
   * PatchInfo::subdomain_partition_data and the associated strides
   * through with respect to the flat data array are set in
   * PatchInfo::patch_starts
   */
  patch_info_in.patch_starts.clear();
  patch_info_in.at_boundary_mask.clear();
  patch_info_in.subdomain_partition_data.clear();
  partition_patches(patch_info_in);
  const auto & partition_data = patch_info_in.subdomain_partition_data;
  Assert(partition_data.n_colors() > 0, ExcMessage("At least one color."));
  for(unsigned color = 0; color < partition_data.n_colors(); ++color)
    has_valid_state(partition_data, color);
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_physical_subdomains() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  // const auto n_physical_subdomains_per_partition = [this](const unsigned int partition,
  //                                                         const unsigned int color) {
  //   const auto & partition_data = patch_info->subdomain_partition_data;
  //   const auto & range          = partition_data.get_patch_range(partition, color);

  //   // TODO
  //   unsigned int n_subdomains = 0;
  //   for(unsigned id = range.first; id < range.second; ++id)
  //     n_subdomains += n_lanes_filled(id);
  //   return n_subdomains;
  //   // return (range.second - range.first) * macro_size;
  // };

  // unsigned     n_subdomains   = 0;
  // const auto & partition_data = patch_info->subdomain_partition_data;
  // const auto   n_colors       = partition_data.n_colors();
  // for(unsigned color = 0; color < n_colors; ++color)
  // {
  //   const auto n_partitions = partition_data.n_partitions(color);
  //   for(unsigned partition = 0; partition < n_partitions; ++partition)
  //     n_subdomains += n_physical_subdomains_per_partition(partition, color);
  // }
  const auto &       subdomain_data = patch_info->get_internal_data()->n_physical_subdomains_total;
  const unsigned int n_subdomains   = subdomain_data.n_interior + subdomain_data.n_boundary;

  return n_subdomains;
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_lanes_filled(const unsigned int patch_id) const
{
  return n_lanes_filled_impl(patch_id);
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_lanes_filled_impl(const unsigned int patch_id) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch_id, patch_info->subdomain_partition_data.n_subdomains());
  const auto & patch_starts = patch_info->patch_starts;
  const auto   start        = patch_starts[patch_id];
  const auto   end          = patch_starts[patch_id + 1];
  Assert(start < end, ExcMessage("Empty set."));
  const unsigned int n_physical_cells = end - start;
  AssertDimension(n_physical_cells % patch_size, 0);
  const unsigned int n_physical_subdomains = n_physical_cells / patch_size;
  Assert(n_physical_subdomains > 0, ExcMessage("No lanes filled."));

  return n_physical_subdomains;
}

template<int dim, typename number>
inline const typename PatchInfo<dim>::PartitionData &
PatchWorker<dim, number>::get_partition_data() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  return patch_info->subdomain_partition_data;
}

template<int dim, typename number>
inline std::array<unsigned int, GeometryInfo<dim>::faces_per_cell>
PatchWorker<dim, number>::get_at_boundary_masks_flat(const unsigned int patch) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch, patch_info->subdomain_partition_data.n_subdomains());
  std::array<unsigned int, GeometryInfo<dim>::faces_per_cell> at_bdry_mask;
  std::copy_n(patch_info->at_boundary_mask.data() + GeometryInfo<dim>::faces_per_cell * patch,
              GeometryInfo<dim>::faces_per_cell,
              at_bdry_mask.begin());
  return at_bdry_mask;
}

template<int dim, typename number>
inline std::array<std::bitset<PatchWorker<dim, number>::macro_size>,
                  GeometryInfo<dim>::faces_per_cell>
PatchWorker<dim, number>::get_at_boundary_masks(const unsigned int patch) const
{
  std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> at_bdry_mask;
  const auto & at_bdry_mask_flat = get_at_boundary_masks_flat(patch);
  std::copy(at_bdry_mask_flat.cbegin(), at_bdry_mask_flat.cend(), at_bdry_mask.begin());
  return at_bdry_mask;
}

template<int dim, typename number>
inline std::vector<
  std::array<std::pair<unsigned int, unsigned int>, PatchWorker<dim, number>::macro_size>>
PatchWorker<dim, number>::get_batch_collection(unsigned int patch_id) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  Assert(mf_connect != nullptr, ExcMessage("No MatrixFreeConnect set. Check constructor."));
  AssertIndexRange(patch_id, patch_info->subdomain_partition_data.n_subdomains());

  const auto & patch_starts = patch_info->patch_starts;
  std::vector<std::array<std::pair<unsigned int, unsigned int>, macro_size>> collection(patch_size);
  const auto & batch_and_lane = mf_connect->batch_and_lane;
  auto         batch_pair     = batch_and_lane.cbegin() + patch_starts[patch_id];
  const auto   n_lanes_filled = this->n_lanes_filled(patch_id);
  for(unsigned int lane = 0; lane < n_lanes_filled; ++lane)
    for(unsigned int cell_no = 0; cell_no < patch_size; ++batch_pair, ++cell_no)
      collection[cell_no][lane] = *batch_pair;
  //: fill non-physical lanes by mirroring first lane
  for(unsigned int lane = n_lanes_filled; lane < macro_size; ++lane)
    for(auto & macro_cell : collection)
      macro_cell[lane] = macro_cell[0];

  return collection;
}


template<int dim, typename number>
inline std::vector<
  std::array<typename PatchWorker<dim, number>::CellIterator, PatchWorker<dim, number>::macro_size>>
PatchWorker<dim, number>::get_cell_collection(unsigned int patch_id) const
{
  Assert(patch_info, ExcMessage("Patch info not set."));
  const auto n_lanes_filled = this->n_lanes_filled(patch_id);
  /// fill the empty vectorization lanes by copying the first lane
  const auto get_cell_position_filled = [&](const auto cell_no, const auto lane) {
    AssertIndexRange(lane, this->macro_size);
    if(lane < n_lanes_filled)
      return this->get_cell_position(patch_id, cell_no, lane);
    else
      return this->get_cell_position(patch_id, cell_no, 0);
  };

  std::vector<std::array<CellIterator, macro_size>> cell_collect(patch_size);
  for(auto cell_no = 0U; cell_no < cell_collect.size(); ++cell_no)
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      const auto cell_position    = get_cell_position_filled(cell_no, lane);
      cell_collect[cell_no][lane] = patch_info->get_cell_iterator(cell_position);
    }
  return cell_collect;
}


template<int dim, typename number>
inline std::vector<ArrayView<const typename PatchWorker<dim, number>::CellIterator>>
PatchWorker<dim, number>::get_cell_collection_views(unsigned int patch_id) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch_id, patch_info->subdomain_partition_data.n_subdomains());

  const auto get_views = [&](const auto & begin) {
    std::vector<ArrayView<const CellIterator>> views;
    for(unsigned int m = 0; m < n_lanes_filled(patch_id); ++m)
    {
      const auto first = begin + m * patch_size;
      views.emplace_back(ArrayView<const CellIterator>(first, patch_size));
    }
    return views;
  };

  if(!patch_info->get_internal_data()->cell_iterators.empty()) // cached
  {
    const auto & cell_iterators = patch_info->get_internal_data()->cell_iterators;
    const auto & patch_starts   = patch_info->patch_starts;
    const auto   begin          = cell_iterators.data() + patch_starts[patch_id];
    return get_views(begin);
  }

  cell_iterators_scratchpad.clear();
  for(auto lane = 0U; lane < n_lanes_filled(patch_id); ++lane)
    for(auto cell_no = 0U; cell_no < patch_size; ++cell_no)
    {
      const auto cell_position = get_cell_position(patch_id, cell_no, lane);
      cell_iterators_scratchpad.emplace_back(patch_info->get_cell_iterator(cell_position));
    }
  return get_views(cell_iterators_scratchpad.data());
}


template<int dim, typename number>
inline void
PatchWorker<dim, number>::compute_partition_data(
  typename PatchInfo<dim>::PartitionData &      partition_data,
  const typename PatchInfo<dim>::InternalData * internal_data,
  std::vector<unsigned int> *                   patch_starts) const
{
  const unsigned int n_colors = internal_data->n_physical_subdomains.size();
  partition_data.initialize(n_colors);
  unsigned int n_subdomains_before = 0;
  unsigned int start               = 0;
  if(patch_starts)
    patch_starts->clear();

  for(unsigned int color = 0; color < n_colors; ++color)
  {
    auto & partitions = partition_data.partitions[color];
    partitions.resize(4 + 1);
    const auto subdomain_data = internal_data->n_physical_subdomains.at(color);
    partitions[0]             = n_subdomains_before;

    // Partition locally owned subdomains.
    const unsigned int n_interior_subdomains_owned =
      subdomain_data.n_interior - subdomain_data.n_interior_ghost;
    const unsigned int n_boundary_subdomains_owned =
      subdomain_data.n_boundary - subdomain_data.n_boundary_ghost;
    const unsigned int n_subdomains_owned =
      n_interior_subdomains_owned + n_boundary_subdomains_owned;
    const unsigned int n_remaining_owned = (n_subdomains_owned % macro_size);
    {
      const unsigned int n_complete_batches   = (n_subdomains_owned / macro_size);
      partitions[1]                           = partitions[0] + n_complete_batches; // complete
      const bool         has_incomplete       = ((n_subdomains_owned % macro_size) != 0);
      const unsigned int n_incomplete_batches = (has_incomplete ? 1 : 0);
      partitions[2]                           = partitions[1] + n_incomplete_batches; // incomplete
    }

    // Parttition subdomains with ghosts.
    const unsigned int n_subdomains_ghost =
      subdomain_data.n_interior_ghost + subdomain_data.n_boundary_ghost;
    const unsigned int n_remaining_ghost = (n_subdomains_ghost % macro_size);
    {
      const unsigned int n_complete_batches   = (n_subdomains_ghost / macro_size);
      partitions[3]                           = partitions[2] + n_complete_batches; // complete
      const bool         has_incomplete       = ((n_subdomains_ghost % macro_size) != 0);
      const unsigned int n_incomplete_batches = (has_incomplete ? 1 : 0);
      partitions[4]                           = partitions[3] + n_incomplete_batches; // incomplete
    }

    n_subdomains_before = partitions.back();
    has_valid_state(partition_data, color);

    // Submit the (vectorized) data access points in @p patch_starts.
    if(patch_starts)
    {
      const auto &       additional_data = patch_info->get_additional_data();
      const unsigned int patch_size = UniversalInfo<dim>::n_cells(additional_data.patch_variant);

      // First, the data access with respect to locally owned subdomains is set.
      {
        // complete range
        const unsigned int stride_comp    = patch_size * macro_size;
        const auto         complete_range = partition_data.get_patch_range(0, color);
        for(unsigned int pp = complete_range.first; pp < complete_range.second; ++pp)
        {
          patch_starts->emplace_back(start);
          start += stride_comp;
        }

        // incomplete range
        const auto         incomplete_range     = partition_data.get_patch_range(1, color);
        const unsigned int n_incomplete_batches = incomplete_range.second - incomplete_range.first;
        Assert(n_incomplete_batches == 0 || n_incomplete_batches == 1,
               ExcMessage("Has more than one incomplete batch."));
        if(n_incomplete_batches == 1)
        {
          patch_starts->emplace_back(start);
          const auto stride_incomplete = patch_size * n_remaining_owned;
          start += stride_incomplete;
        }
      }

      // Analogously, the data access with respect to subdomains with ghost cells is set.
      {
        // complete range
        const unsigned int stride_comp    = patch_size * macro_size;
        const auto         complete_range = partition_data.get_patch_range(2, color);
        for(unsigned int pp = complete_range.first; pp < complete_range.second; ++pp)
        {
          patch_starts->emplace_back(start);
          start += stride_comp;
        }

        // incomplete range
        const auto         incomplete_range     = partition_data.get_patch_range(3, color);
        const unsigned int n_incomplete_batches = incomplete_range.second - incomplete_range.first;
        Assert(n_incomplete_batches == 0 || n_incomplete_batches == 1,
               ExcMessage("Has more than one incomplete batch."));
        if(n_incomplete_batches == 1)
        {
          patch_starts->emplace_back(start);
          const auto stride_incomplete = patch_size * n_remaining_ghost;
          start += stride_incomplete;
        }
      }
    }
  }

  if(patch_starts)
  {
    AssertDimension(start, internal_data->n_cells_plain());
    AssertDimension(patch_starts->size(), partition_data.n_subdomains());
    patch_starts->emplace_back(start); // endpoint required by n_lanes_filled()
  }
}


template<int dim, typename number>
inline void
PatchWorker<dim, number>::has_valid_state(const typename PatchInfo<dim>::PartitionData & partitions,
                                          const unsigned int                             color)
{
  const auto n_partitions = partitions.n_partitions(color);
  AssertDimension(n_partitions, 4);
  for(unsigned int p = 0; p < n_partitions; ++p)
  {
    const auto range = partitions.get_patch_range(p, color);
    (void)range;
    Assert(range.first >= 0, ExcMessage("Negative range."));
    Assert(range.first <= range.second, ExcMessage("Lower bound exceeds upper bound."));
  }
  { // incomplete + locally owned
    const auto     incomplete_range = partitions.get_patch_range(1, color);
    const unsigned n_elements       = incomplete_range.second - incomplete_range.first;
    (void)n_elements;
    Assert(n_elements == 0 || n_elements == 1,
           ExcMessage("Incomplete range has more than one batch."));
  }
  { // incomplete + ghost
    const auto     incomplete_range = partitions.get_patch_range(3, color);
    const unsigned n_elements       = incomplete_range.second - incomplete_range.first;
    (void)n_elements;
    Assert(n_elements == 0 || n_elements == 1,
           ExcMessage("Incomplete range has more than one batch."));
  }
}



} // end namespace TPSS

#include "patch_worker.templates.h"

#endif // end inclusion guard
