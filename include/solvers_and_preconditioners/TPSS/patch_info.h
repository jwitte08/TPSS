#ifndef PATCHINFO_H
#define PATCHINFO_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "TPSS.h"
#include "generic_functionalities.h"
#include "time_info.h"

#include <array>
#include <fstream>
#include <memory>

namespace TPSS
{
template<int dim>
class PatchInfo
{
public:
  using CellIterator  = typename dealii::DoFHandler<dim>::level_cell_iterator;
  using PatchIterator = typename std::vector<std::vector<CellIterator>>::const_iterator;

  struct AdditionalData;

  struct GhostPatch;

  struct InternalData;

  struct SubdomainData
  {
    unsigned int n_interior       = 0;
    unsigned int n_boundary       = 0;
    unsigned int n_interior_ghost = 0;
    unsigned int n_boundary_ghost = 0;
  };

  struct PartitionData;

  PatchInfo() = default;

  PatchInfo(const PatchInfo<dim> &) = delete;

  ~PatchInfo() = default;

  PatchInfo<dim> &
  operator=(const PatchInfo<dim> &) = delete;

  void
  initialize(const dealii::DoFHandler<dim> * dof_handler, const AdditionalData additional_data);

  void
  clear();

  bool
  empty() const;

  /**
   * Read access to the internal data.
   */
  const InternalData *
  get_internal_data() const;

  /**
   * Read access to the additional data
   */
  const AdditionalData &
  get_additional_data() const;

  std::pair<int, int>
  get_cell_level_and_index(const unsigned int cell_position) const
  {
    AssertIndexRange(cell_position, n_cells_plain());
    if(level_and_index_is_cached)
    {
      AssertDimension(n_cells_plain(), get_internal_data()->cell_level_and_index_pairs.size());
      return (get_internal_data()->cell_level_and_index_pairs)[cell_position];
    }
    AssertDimension(n_cells_plain(), get_internal_data()->cell_iterators.size());
    const auto & cell = (get_internal_data()->cell_iterators)[cell_position];
    return std::make_pair<int, int>(cell->level(), cell->index());
  }

  CellIterator
  get_cell_iterator(const unsigned int cell_position) const
  {
    AssertIndexRange(cell_position, n_cells_plain());
    if(iterator_is_cached)
    {
      AssertDimension(n_cells_plain(), get_internal_data()->cell_iterators.size());
      return (get_internal_data()->cell_iterators)[cell_position];
    }
    const auto & tria                   = get_triangulation();
    const auto [cell_level, cell_index] = get_cell_level_and_index(cell_position);
    // // TODO we should not need dof handler here
    // const auto dof_handler = get_internal_data()->dof_handler;
    // Assert(dof_handler, ExcMessage("DoFHandler not set."));
    return CellIterator(&tria, cell_level, cell_index, nullptr);
  }

  const Triangulation<dim> &
  get_triangulation() const
  {
    const auto tria = get_internal_data()->triangulation;
    Assert(tria, ExcMessage("Triangulation not set."));
    return *tria;
  }

  unsigned int
  n_cells_plain() const
  {
    return get_internal_data()->n_cells_plain();
  }

  /**
   * An array of strides to access the CellIterator collections
   * (=patch) within the flat array @p cell_iterators. In other words
   * two consecutive elements in @p patch_starts represent a half-open
   * range of CellIterators, containing all cells to construct a
   * (macro) patch.
   */
  std::vector<unsigned int> patch_starts;

  /**
   * An array that stores how many vectorization lanes of a macro patch contain meaningful data
   */
  std::vector<unsigned int> n_lanes_filled;

  /**
   * A flat array that stores the information if each face within a
   * (macro) face is at the physical boundary. Faces are ordered in
   * the standard deal.II way, see GeometryInfo. The integer must be
   * reinterpreted as std::bitset of length @p macro_size.
   * Lexicographical:   face number   <   patch id
   */
  std::vector<unsigned int> at_boundary_mask;

  /**
   * Container storing the partitioning into colored batches of
   * interior/boundary, incomplete/complete subdomains.
   */
  PartitionData subdomain_partition_data;

  /**
   * An array to store timer output.
   */
  std::vector<TimeInfo> time_data;

private:
  void
  count_physical_subdomains()
  {
    internal_data.n_physical_subdomains_total.n_interior =
      std::accumulate(internal_data.n_physical_subdomains.cbegin(),
                      internal_data.n_physical_subdomains.cend(),
                      0,
                      [](const auto sum, const auto & data) { return sum + data.n_interior; });
    internal_data.n_physical_subdomains_total.n_boundary =
      std::accumulate(internal_data.n_physical_subdomains.cbegin(),
                      internal_data.n_physical_subdomains.cend(),
                      0,
                      [](const auto sum, const auto & data) { return sum + data.n_boundary; });
    internal_data.n_physical_subdomains_total.n_interior_ghost =
      std::accumulate(internal_data.n_physical_subdomains.cbegin(),
                      internal_data.n_physical_subdomains.cend(),
                      0,
                      [](const auto sum, const auto & data) {
                        return sum + data.n_interior_ghost;
                      });
    internal_data.n_physical_subdomains_total.n_boundary_ghost =
      std::accumulate(internal_data.n_physical_subdomains.cbegin(),
                      internal_data.n_physical_subdomains.cend(),
                      0,
                      [](const auto sum, const auto & data) {
                        return sum + data.n_boundary_ghost;
                      });
  }

  static std::vector<types::global_dof_index>
  get_face_conflicts(const PatchIterator & patch)
  {
    std::vector<types::global_dof_index> conflicts;
    const auto &                         cell_collection = *patch;

    for(const auto & cell : cell_collection)
      for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        const bool neighbor_has_same_level = (cell->neighbor_level(face_no) == cell->level());
        const bool neighbor_doesnt_exist   = (cell->neighbor_level(face_no) == -1);
        const bool non_adaptive            = neighbor_has_same_level || neighbor_doesnt_exist;
        (void)non_adaptive;
        Assert(non_adaptive, ExcNotImplemented());
        conflicts.emplace_back(cell->face(face_no)->index());
      }
    return conflicts;
  }

  /**
   * Gathering the locally owned and ghost cells attached to a common
   * vertex as the collection of cell iterators (patch). The
   * successive distribution of the collections with ghost cells
   * follows the logic:
   *
   * 1.) if one mpi-proc owns more than half of the cells (locally owned)
   * of the vertex patch the mpi-proc takes ownership
   *
   * 2.) for the remaining ghost patches the mpi-proc with the cell of
   * the lowest CellId (see dealii::Triangulation) takes ownership
   */
  std::vector<std::vector<CellIterator>>
  gather_vertex_patches(const DoFHandler<dim> & dof_handler,
                        const AdditionalData &  additional_data) const;

  void
  initialize_cell_patches(const dealii::DoFHandler<dim> * dof_handler,
                          const AdditionalData            additional_data);

  void
  initialize_vertex_patches(const dealii::DoFHandler<dim> * dof_handler,
                            const AdditionalData            additional_data);

  // void
  // write_visual_data(
  //   const dealii::DoFHandler<dim> &                            dof_handler,
  //   const std::vector<std::pair<unsigned int, unsigned int>> & reordered_colors) const;

  template<int regular_size>
  void
  submit_patches(const std::vector<PatchIterator> & patch_iterators)
  {
    // First, submit all interior patches and subsequently all boundary patches
    const auto & internal_submit = [this](const std::vector<PatchIterator> & patch_iterators) {
      unsigned int               n_interior_subdomains_regular = 0;
      std::vector<PatchIterator> boundary_patch_regular;
      for(const auto & patch : patch_iterators)
      {
        const bool patch_at_boundary =
          std::any_of(patch->cbegin(), patch->cend(), IteratorFilters::AtBoundary{});
        if(patch->size() == regular_size) // regular
        {
          if(patch_at_boundary)
          {
            boundary_patch_regular.push_back(patch);
          }
          else
          {
            ++n_interior_subdomains_regular;
            for(const auto & cell : *patch)
              internal_data.cell_iterators.emplace_back(cell);
          }
        }
        else // irregular
          Assert(false, ExcNotImplemented());
      }

      for(const auto it : boundary_patch_regular)
      {
        for(const auto & cell : *it)
          internal_data.cell_iterators.emplace_back(cell);
      }

      SubdomainData local_data;
      local_data.n_interior = n_interior_subdomains_regular;
      local_data.n_boundary = boundary_patch_regular.size();
      return local_data;
    };
    // We separate the patch iterators into locally owned subdomains
    // and those with ghost cells. First, we submit locally owned and
    // subsequently subdomains with ghosts. Each group is separated
    // into interior (first) and boundary (second) subdomains,
    // respectively.
    const auto   my_subdomain_id   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto & is_ghost_on_level = [my_subdomain_id](const auto & cell) {
      const bool is_owned      = cell->level_subdomain_id() == my_subdomain_id;
      const bool is_artificial = cell->level_subdomain_id() == numbers::artificial_subdomain_id;
      return !is_owned && !is_artificial;
    };
    std::vector<PatchIterator> owned_patch_iterators, ghost_patch_iterators;
    for(const auto & patch : patch_iterators)
    {
      const bool patch_is_ghost = std::any_of(patch->cbegin(), patch->cend(), is_ghost_on_level);
      if(patch_is_ghost)
        ghost_patch_iterators.emplace_back(patch);
      else
        owned_patch_iterators.emplace_back(patch);
    }
    AssertDimension(owned_patch_iterators.size() + ghost_patch_iterators.size(),
                    patch_iterators.size());
    const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
    if(!is_mpi_parallel)
      AssertDimension(ghost_patch_iterators.size(), 0);

    const auto    owned_subdomain_data = internal_submit(owned_patch_iterators);
    const auto    ghost_subdomain_data = internal_submit(ghost_patch_iterators);
    SubdomainData subdomain_data;
    subdomain_data.n_interior = owned_subdomain_data.n_interior + ghost_subdomain_data.n_interior;
    subdomain_data.n_boundary = owned_subdomain_data.n_boundary + ghost_subdomain_data.n_boundary;
    subdomain_data.n_interior_ghost = ghost_subdomain_data.n_interior;
    subdomain_data.n_boundary_ghost = ghost_subdomain_data.n_boundary;
    internal_data.n_physical_subdomains.emplace_back(subdomain_data);
  }

  ConditionalOStream pcout{std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0};

  /**
   * A struct storing CellIterators shaping patches
   */
  InternalData internal_data;

  /**
   * Specific information like coloring schemes, etc.
   */
  AdditionalData additional_data;

  bool iterator_is_cached        = false;
  bool level_and_index_is_cached = false;
};



template<int dim>
struct PatchInfo<dim>::AdditionalData
{
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::invalid;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;
  unsigned int          level            = -1;
  std::function<
    std::vector<std::vector<PatchIterator>>(const std::vector<std::vector<CellIterator>> & patches,
                                            const AdditionalData additional_data)>
    coloring_func;
  std::function<void(const DoFHandler<dim> *                  dof_handler,
                     const AdditionalData                     additional_data,
                     std::vector<std::vector<CellIterator>> & cell_collections)>
    manual_gathering_func;
  std::function<void(const DoFHandler<dim> & dof_handler_in,
                     const std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>> &
                       colored_iterators,
                     const std::string)>
                        visualize_coloring;
  TPSS::CachingStrategy caching_strategy = TPSS::CachingStrategy::Cached;
  bool                  print_details    = false; // DEBUG
};



template<int dim>
struct PatchInfo<dim>::GhostPatch
{
  GhostPatch(const unsigned int proc, const CellId & cell_id);

  void
  submit_id(const unsigned int proc, const CellId & cell_id);

  std::string
  str() const;

  std::map<unsigned, std::vector<CellId>> proc_to_cell_ids;
};



/**
 * This helper struct stores the information on the distribution of
 * CellIterator collections (=patch) into partitions of the same
 * constitution for each color. In other words, patch properties are
 * uniquely determined by the half-open patch range (namely a
 * partition).
 *
 * It is an abstract container providing access and information of the
 * distribution. The actual distribution and assignment of the patch
 * properties is treated by worker classes, see PatchWorker. Moreover,
 * the actual CellIterator storage is encapsulated in PatchInfo's @p
 * internal_data.
 */
template<int dim>
struct PatchInfo<dim>::PartitionData
{
  PartitionData() = default;

  PartitionData(const PartitionData &) = default;

  ~PartitionData() = default;

  PartitionData &
  operator=(const PartitionData &) = default;

  void
  initialize(const unsigned int n_colors);

  void
  clear();

  std::size_t
  n_colors() const;

  std::size_t
  n_partitions(const unsigned int color = 0) const;

  std::size_t
  n_subdomains() const;

  std::size_t
  n_subdomains(const unsigned int color) const;

  /**
   * Returns the total, half-open patch range including all colors.
   */
  std::pair<unsigned int, unsigned int>
  get_patch_range() const;

  /**
   * Returns the half-open patch range associated to the color @p
   * color.
   */
  std::pair<unsigned int, unsigned int>
  get_patch_range(const unsigned int color) const;

  /**
   * Returns the half-open patch range of partition @p partition
   * regarding the color @p color.
   */
  std::pair<unsigned int, unsigned int>
  get_patch_range(const unsigned int partition, const unsigned int color) const;

  /**
   * Checks the compatibility of @p other against the current
   * partition data. This identifies if data might be reused.
   */
  bool
  is_compatible(const PartitionData & other) const;

  /**
   * Actual storage of a set of partitions for each color. Two
   * consecutive integers of the inner vector represent a half-open
   * interval, the so-called patch range identified by a partition id.
   */
  std::vector<std::vector<unsigned int>> partitions;
};



/**
 * This helper struct contains all data to construct (macro) patches
 * and the distribution into partitions of certain properties for each
 * color. Up to now it only contains data for regular patches to
 * obtain tensor product structures.
 * A (macro) patch is characterized as the contiguous range of a starting
 * and ending CellIterator in the flat array @p cell_iterators.
 */
template<int dim>
struct PatchInfo<dim>::InternalData
{
  InternalData() = default;

  InternalData(const InternalData &) = delete;

  ~InternalData() = default;

  InternalData &
  operator=(const InternalData &) = delete;

  void
  clear();

  /*
   * Experimental. If we have extracted all informations on dofs and mesh cells
   * we are able to delete the @p cell_iterators field.
   */
  void
  compress() const;

  bool
  empty() const;

  bool
  empty_on_all() const;

  unsigned int
  n_cells_plain() const
  {
    return std::max(cell_iterators.size(), cell_level_and_index_pairs.size());
  }

  unsigned int level = numbers::invalid_unsigned_int;

  /**
   * Flat array that stores all CellIterators for the construction of
   * (macro) patches. The successive alignment of CellIterators is
   * such that all interior subdomains are stored first, then the ones
   * at the physical boundary
   *
   * Lexicographical:  cell number  <  lane  <  patch id  <   color
   */
  mutable std::vector<CellIterator> cell_iterators;

  /**
   * Flat array that stores all CellIterators for the construction of
   * (macro) patches. The successive alignment of CellIterators is
   * such that all interior subdomains are stored first, then the ones
   * at the physical boundary
   *
   * Lexicographical:  cell number  <  lane  <  patch id  <   color
   */
  mutable std::vector<std::pair<int, int>> cell_level_and_index_pairs;

  /**
   * Numbers of physical subdomains for each color.
   */
  std::vector<SubdomainData> n_physical_subdomains;

  /**
   * Numbers of physical subdomains (accumulated over colors)
   */
  SubdomainData n_physical_subdomains_total;

  /**
   * Underlying triangulation cell iterators are based on.
   */
  const Triangulation<dim> * triangulation = nullptr;
  // TODO we should not need dof handler !
  const DoFHandler<dim> * dof_handler = nullptr;
};



template<int dim, typename number>
struct MatrixFreeConnect
{
  /**
   * The underlying MatrixFree object used to map matrix-free infrastructure
   * to the patch distribution stored in PatchInfo
   */
  const dealii::MatrixFree<dim, number> * mf_storage = nullptr;

  /**
   * The batch index and vectorization lane pairs identifying cells
   * stored in the underlying MatrixFree object and corresponding to
   * cells stored in the field @p cell_iterators.
   */
  std::vector<std::pair<unsigned int, unsigned int>> batch_and_lane;
};



// --------------------------------   PatchInfo   --------------------------------



template<int dim>
inline void
PatchInfo<dim>::clear()
{
  patch_starts.clear();
  at_boundary_mask.clear();
  subdomain_partition_data.clear();

  internal_data.clear();
  additional_data = AdditionalData{};
  time_data.clear();
}

template<int dim>
bool
PatchInfo<dim>::empty() const
{
  return internal_data.empty();
}


template<int dim>
inline const typename PatchInfo<dim>::InternalData *
PatchInfo<dim>::get_internal_data() const
{
  return &internal_data;
}

template<int dim>
inline const typename PatchInfo<dim>::AdditionalData &
PatchInfo<dim>::get_additional_data() const
{
  return additional_data;
}

// --------------------------------   PatchInfo::GhostPatch   --------------------------------

template<int dim>
PatchInfo<dim>::GhostPatch::GhostPatch(const unsigned int proc, const CellId & cell_id)
{
  submit_id(proc, cell_id);
}

template<int dim>
inline void
PatchInfo<dim>::GhostPatch::submit_id(const unsigned int proc, const CellId & cell_id)
{
  const auto member = proc_to_cell_ids.find(proc);
  if(member != proc_to_cell_ids.cend())
  {
    member->second.emplace_back(cell_id);
    Assert(!(member->second.empty()), ExcMessage("at least one element"));
  }
  else
  {
    const auto status = proc_to_cell_ids.emplace(proc, std::vector<CellId>{cell_id});
    (void)status;
    Assert(status.second, ExcMessage("failed to insert key-value-pair"));
  }
}

template<int dim>
inline std::string
PatchInfo<dim>::GhostPatch::str() const
{
  std::ostringstream oss;
  oss << "{";
  const auto size = proc_to_cell_ids.size();
  unsigned   i    = 0;
  for(auto key_value = proc_to_cell_ids.cbegin(); key_value != proc_to_cell_ids.cend();
      ++key_value, ++i)
    oss << "(" << key_value->first << ", " << vector_to_string(key_value->second)
        << ((i + 1) < size ? "), " : ")}");
  return oss.str();
}

// --------------------------------   PatchInfo::PartitionData   --------------------------------

template<int dim>
inline void
PatchInfo<dim>::PartitionData::initialize(const unsigned int n_colors)
{
  clear();
  partitions.resize(n_colors);
}

template<int dim>
inline void
PatchInfo<dim>::PartitionData::clear()
{
  partitions.clear();
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_colors() const
{
  return partitions.size();
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_partitions(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  return (partitions[color].size() > 0) ? partitions[color].size() - 1 : 0;
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_subdomains() const
{
  // Assert(!partitions.empty(), ExcMessage("TODO should not be empty."));
  return partitions.empty() ? 0 : (partitions.back().empty()) ? 0 : partitions.back().back();
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_subdomains(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  return partitions[color].empty() ? 0 : partitions[color].back();
}

template<int dim>
inline std::pair<unsigned int, unsigned int>
PatchInfo<dim>::PartitionData::get_patch_range() const
{
  Assert(!partitions.empty(), dealii::ExcInvalidState());
  Assert(!partitions.front().empty(), dealii::ExcInvalidState());
  Assert(!partitions.back().empty(), dealii::ExcInvalidState());
  return std::make_pair(partitions.front().front(), partitions.back().back());
}

template<int dim>
inline std::pair<unsigned int, unsigned int>
PatchInfo<dim>::PartitionData::get_patch_range(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  Assert(!partitions[color].empty(), dealii::ExcInvalidState());
  return std::make_pair(partitions[color].front(), partitions[color].back());
}

template<int dim>
inline std::pair<unsigned int, unsigned int>
PatchInfo<dim>::PartitionData::get_patch_range(const unsigned int partition,
                                               const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  AssertIndexRange(partition, n_partitions());
  return std::make_pair(partitions[color][partition], partitions[color][partition + 1]);
}

template<int dim>
inline bool
PatchInfo<dim>::PartitionData::is_compatible(const PartitionData & other) const
{
  return (partitions == other.partitions);
}

// --------------------------------   PatchInfo::InternalData   --------------------------------

template<int dim>
inline void
PatchInfo<dim>::InternalData::clear()
{
  level = numbers::invalid_unsigned_int;
  n_physical_subdomains.clear();
  n_physical_subdomains_total = SubdomainData{};
  cell_iterators.clear();
  cell_level_and_index_pairs.clear();
  triangulation = nullptr;
  dof_handler   = nullptr;
}

template<int dim>
inline void
PatchInfo<dim>::InternalData::compress() const
{
  cell_iterators.clear();
}

template<int dim>
inline bool
PatchInfo<dim>::InternalData::empty() const
{
  return cell_iterators.empty() && cell_level_and_index_pairs.empty();
}


template<int dim>
inline bool
PatchInfo<dim>::InternalData::empty_on_all() const
{
  const auto n_iterators_mpimax = Utilities::MPI::max(cell_iterators.size(), MPI_COMM_WORLD);
  const auto n_pairs_mpimax =
    Utilities::MPI::max(cell_level_and_index_pairs.size(), MPI_COMM_WORLD);
  return (n_iterators_mpimax == 0) && (n_pairs_mpimax);
}



} // end namespace TPSS

#include "patch_info.templates.h"

#endif // end inclusion guard
