#ifndef PATCH_WORKER_H
#define PATCH_WORKER_H

#include "patch_info.h"



namespace TPSS
{
// TODO revise description
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
  static constexpr unsigned int macro_size = dealii::VectorizedArray<number>::size();
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

  std::vector<FaceInfoLocal<dim>>
  fill_face_infos(const unsigned patch_id, const unsigned int lane) const;

  std::array<unsigned int, GeometryInfo<dim>::faces_per_cell>
  get_at_boundary_masks_flat(const unsigned int patch) const;

  std::array<std::bitset<PatchWorker<dim, number>::macro_size>, GeometryInfo<dim>::faces_per_cell>
  get_at_boundary_masks(const unsigned int patch) const;

  std::array<std::array<types::boundary_id, macro_size>, GeometryInfo<dim>::faces_per_cell>
  get_boundary_ids(const unsigned int patch) const;

  /**
   * Returns the collection of cell iterators describing the physical patch
   * identified by macro patch @p patch_id and vectorization lane @p lane. Cells
   * are lexicographically ordered.
   */
  std::vector<CellIterator>
  get_cell_collection(const unsigned int patch_id, const unsigned int lane) const;

  /**
   * Returns the collection of macro cells describing the (regular) macro patch
   * @p patch_id subject to a lexicographical ordering.
   */
  std::vector<std::array<CellIterator, macro_size>>
  get_cell_collection(const unsigned int patch_id) const;

  unsigned int
  get_cell_position(const unsigned int patch_id,
                    const unsigned int cell_no,
                    const unsigned int lane) const;

  const typename PatchInfo<dim>::PartitionData &
  get_partition_data() const;

  std::pair<unsigned int, unsigned int>
  get_owned_range(const unsigned int color,
                  const RangeVariant range_variant = RangeVariant::all) const;

  std::pair<unsigned int, unsigned int>
  get_ghost_range(const unsigned int color,
                  const RangeVariant range_variant = RangeVariant::all) const;

  /*
   * Return whether this macro patch is located in the interior of the domain.
   */
  bool
  is_interior(const unsigned int patch) const;

  unsigned int
  n_cells_per_subdomain() const;

  /**
   * Return the number of macro subdomains. A macro subdomain consists of a
   * vectorized batch of physical subdomains. How many vectorization lanes are
   * filled with physical subdomains can be queried by @p n_lanes_filled().
   */
  unsigned int
  n_subdomains() const;

  /**
   * Return the number of physical subdomains.
   */
  unsigned int
  n_physical_subdomains() const;

  /*
   * Return the number of vectorization lanes of macro patch @p patch_id which
   * represent physical subdomins. If @p n_lanes_filled is less than the
   * vectorization length remaining lanes are artificially filled.
   */
  unsigned int
  n_lanes_filled(const unsigned int patch_id) const;

protected:
  /**
   * This method partitions the physical subdomains (that is not vectorized),
   * contained in PatchInfo, into interior/boundary, incomplete/complete groups,
   * of vectorized batches of physical subdomains, so-called macro patches.
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

  const TPSS::PatchVariant patch_variant;

  mutable std::vector<CellIterator> cell_iterators_scratchpad;
};



// --------------------------------   PatchWorker   --------------------------------



template<int dim, typename number>
PatchWorker<dim, number>::PatchWorker(const PatchInfo<dim> & patch_info_in)
  : patch_info(&patch_info_in), patch_variant(patch_info_in.get_additional_data().patch_variant)
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
  : patch_info(&patch_info_in), patch_variant(patch_info_in.get_additional_data().patch_variant)
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
  patch_info_in.subdomain_partition_data.clear();
  partition_patches(patch_info_in);
  const auto & partition_data = patch_info_in.subdomain_partition_data;
  Assert(partition_data.n_colors() > 0, ExcMessage("At least one color."));
  for(unsigned color = 0; color < partition_data.n_colors(); ++color)
    has_valid_state(partition_data, color);
}


template<int dim, typename number>
bool
PatchWorker<dim, number>::is_interior(const unsigned int patch) const
{
  const auto bdry_masks        = get_at_boundary_masks(patch);
  const bool is_interior_patch = std::all_of(bdry_masks.cbegin(),
                                             bdry_masks.cend(),
                                             [](const auto & mask) { return mask.none(); });
  return is_interior_patch;
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_cells_per_subdomain() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  const auto n_cells = TPSS::UniversalInfo<dim>::n_cells(patch_variant);
  return n_cells;
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_subdomains() const
{
  return get_partition_data().n_subdomains();
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_physical_subdomains() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  const auto & subdomain_data = patch_info->get_internal_data()->subdomain_quantities_accumulated;
  const unsigned int n_subdomains = subdomain_data.n_interior + subdomain_data.n_boundary;

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
  AssertIndexRange(patch_id, get_partition_data().n_subdomains());
  const auto & patch_starts = patch_info->patch_starts;
  const auto   start        = patch_starts[patch_id];
  const auto   end          = patch_starts[patch_id + 1];
  Assert(start < end, ExcMessage("Empty set."));
  const unsigned int n_physical_cells = end - start;
  const auto         patch_size       = n_cells_per_subdomain();
  AssertDimension(n_physical_cells % patch_size, 0);
  const unsigned int n_physical_subdomains = n_physical_cells / patch_size;
  Assert(n_physical_subdomains > 0, ExcMessage("No lanes filled."));

  return n_physical_subdomains;
}


template<int dim, typename number>
inline std::vector<FaceInfoLocal<dim>>
PatchWorker<dim, number>::fill_face_infos(const unsigned int patch_id,
                                          const unsigned int lane) const
{
  const auto &                    cell_collection = get_cell_collection(patch_id, lane);
  std::vector<FaceInfoLocal<dim>> face_infos;
  for(auto cell_no = 0U; cell_no < cell_collection.size(); ++cell_no)
    face_infos.emplace_back(cell_no, cell_collection);
  return face_infos;
}


template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::get_cell_position(const unsigned int patch_id,
                                            const unsigned int cell_no,
                                            const unsigned int lane) const
{
  Assert(patch_info, ExcMessage("PatchInfo not initialized."));
  AssertIndexRange(patch_id, get_partition_data().n_subdomains());
  const auto patch_size = n_cells_per_subdomain();
  AssertIndexRange(cell_no, patch_size);
  AssertIndexRange(lane, n_lanes_filled(patch_id));
  return patch_info->patch_starts[patch_id] + lane * patch_size + cell_no;
}


template<int dim, typename number>
inline std::pair<unsigned int, unsigned int>
PatchWorker<dim, number>::get_owned_range(const unsigned int color,
                                          const RangeVariant range_variant) const
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


template<int dim, typename number>
inline std::pair<unsigned int, unsigned int>
PatchWorker<dim, number>::get_ghost_range(const unsigned int color,
                                          const RangeVariant range_variant) const
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



template<int dim, typename number>
inline const typename PatchInfo<dim>::PartitionData &
PatchWorker<dim, number>::get_partition_data() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  return patch_info->subdomain_partition_data;
}


template<int dim, typename number>
inline std::array<std::bitset<PatchWorker<dim, number>::macro_size>,
                  GeometryInfo<dim>::faces_per_cell>
PatchWorker<dim, number>::get_at_boundary_masks(const unsigned int patch_id) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch_id, get_partition_data().n_subdomains());

  const auto get_mask =
    [](auto && macro_cell, const unsigned int dimension, const unsigned int face_no_1d) {
      std::bitset<macro_size> bitset_mask;
      const auto              face_no = 2 * dimension + face_no_1d;
      for(unsigned int lane = 0; lane < macro_size; ++lane)
        bitset_mask[lane] = macro_cell[lane]->face(face_no)->at_boundary();
      return bitset_mask;
    };

  // TODO patch face contains interior cell face and cell face at physical boundary
  std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> at_bdry_masks;
  const auto & cell_collection = get_cell_collection(patch_id);
  for(unsigned int d = 0; d < dim; ++d)
  {
    at_bdry_masks[d * 2]     = get_mask(cell_collection.front(), d, /*face_no*/ 0);
    at_bdry_masks[d * 2 + 1] = get_mask(cell_collection.back(), d, /*face_no*/ 1);
  }

  return at_bdry_masks;
}


template<int dim, typename number>
inline std::array<std::array<types::boundary_id, PatchWorker<dim, number>::macro_size>,
                  GeometryInfo<dim>::faces_per_cell>
PatchWorker<dim, number>::get_boundary_ids(const unsigned int patch) const
{
  std::array<std::array<types::boundary_id, macro_size>, GeometryInfo<dim>::faces_per_cell>
    bdry_ids;

  /// if patch is interior fill with internal boundary ids
  if(is_interior(patch))
  {
    for(auto & macro_id : bdry_ids)
      for(auto & id : macro_id)
        id = numbers::internal_face_boundary_id;
    return bdry_ids;
  }

  /// if patch at boundary query boundary ids from cell iterators
  const auto local_face_no = [](const auto dimension, const auto face_no_1d) {
    /// two faces per dimension, namely endpoints of intervals
    return dimension * 2 + face_no_1d;
  };
  const auto fill_bdry_ids = [&](const auto & macro_cell, const auto face_no_1d) {
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      const auto & cell = macro_cell[lane];
      for(auto d = 0U; d < dim; ++d)
      {
        const auto face_no      = local_face_no(d, face_no_1d);
        bdry_ids[face_no][lane] = cell->face(face_no)->boundary_id();
      }
    }
  };

  const auto & cell_collection = get_cell_collection(patch);
  /// fill faces 0, 2, 4, ..., i.e. faces with face_no_1d = 0
  const auto & first_macro_cell = cell_collection.front();
  fill_bdry_ids(first_macro_cell, 0);
  /// fill faces 1, 3, 5, ..., i.e. faces with face_no_1d = 1
  const auto & last_macro_cell = cell_collection.back();
  fill_bdry_ids(last_macro_cell, 1);

  return bdry_ids;
}


template<int dim, typename number>
inline std::vector<typename PatchWorker<dim, number>::CellIterator>
PatchWorker<dim, number>::get_cell_collection(const unsigned int patch_id,
                                              const unsigned int lane) const
{
  Assert(patch_info, ExcMessage("Patch info is not set."));
  AssertIndexRange(lane, n_lanes_filled(patch_id));
  AssertIndexRange(patch_id, get_partition_data().n_subdomains());

  std::vector<PatchWorker<dim, number>::CellIterator> collection;
  const auto                                          patch_size = n_cells_per_subdomain();
  for(auto cell_no = 0U; cell_no < patch_size; ++cell_no)
  {
    const auto cell_position = get_cell_position(patch_id, cell_no, lane);
    collection.emplace_back(patch_info->get_cell_iterator(cell_position));
  }
  return collection;
}


template<int dim, typename number>
inline std::vector<
  std::array<typename PatchWorker<dim, number>::CellIterator, PatchWorker<dim, number>::macro_size>>
PatchWorker<dim, number>::get_cell_collection(const unsigned int patch_id) const
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

  const auto                                        patch_size = n_cells_per_subdomain();
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
inline void
PatchWorker<dim, number>::compute_partition_data(
  typename PatchInfo<dim>::PartitionData &      partition_data,
  const typename PatchInfo<dim>::InternalData * internal_data,
  std::vector<unsigned int> *                   patch_starts) const
{
  const unsigned int n_colors = internal_data->n_colors();
  partition_data.initialize(n_colors);
  unsigned int n_subdomains_before = 0;
  unsigned int start               = 0;
  if(patch_starts)
    patch_starts->clear();

  for(unsigned int color = 0; color < n_colors; ++color)
  {
    auto & partitions = partition_data.partitions[color];
    partitions.resize(4 + 1);
    const auto subdomain_data = internal_data->subdomain_quantities.at(color);
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
