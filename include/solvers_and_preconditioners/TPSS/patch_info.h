#ifndef PATCHINFO_H
#define PATCHINFO_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_out.h>

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
/// TODO description...
/// TODO conceptional PatchInfo should be a struct to store patch
/// distributions and interfacing should be done by PatchWorker (and its
/// derivatives)...
template<int dim>
class PatchInfo
{
public:
  /// TODO CellIterator does not need to have underlying DoFAccessor type
  /// TODO CellIterator should handle active cells if PatchInfo does so
  using cell_iterator_type     = typename dealii::Triangulation<dim>::cell_iterator;
  using dof_cell_iterator_type = typename dealii::DoFHandler<dim>::level_cell_iterator;
  using CellIterator           = dof_cell_iterator_type;                                 // DEPR
  using PatchIterator = typename std::vector<std::vector<CellIterator>>::const_iterator; // DEPR

  struct AdditionalData;

  struct GhostPatch;

  struct InternalData;

  struct SubdomainData;

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
  get_cell_level_and_index(const unsigned int cell_position) const;

  const std::vector<std::pair<int, int>> &
  get_cell_level_and_index_pairs() const;

  const Triangulation<dim> &
  get_triangulation() const;

  unsigned int
  n_cells_plain() const;

  /**
   * An array of strides to access the CellIterator collections
   * (=patch) within the flat array @p cell_iterators. In other words
   * two consecutive elements in @p patch_starts represent a half-open
   * range of CellIterators, containing all cells to construct a
   * macro patch.
   */
  std::vector<unsigned int> patch_starts;

  /**
   * An array that stores how many vectorization lanes of a macro patch contain meaningful data
   */
  std::vector<unsigned int> n_lanes_filled;

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
  count_physical_subdomains();

  static std::vector<types::global_dof_index>
  get_face_conflicts(const PatchIterator & patch);

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
  submit_patches(const std::vector<PatchIterator> & patch_iterators);

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
  unsigned int          level            = numbers::invalid_unsigned_int;
  std::function<
    std::vector<std::vector<PatchIterator>>(const std::vector<std::vector<CellIterator>> & patches,
                                            const AdditionalData additional_data)>
    coloring_func;
  std::function<void(const DoFHandler<dim> *                  dof_handler,
                     const AdditionalData                     additional_data,
                     std::vector<std::vector<CellIterator>> & cell_collections)>
    patch_distribution_func;
  std::function<void(const DoFHandler<dim> & dof_handler_in,
                     const std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>> &
                       colored_iterators,
                     const std::string)>
       visualize_coloring;
  bool use_tbb       = false;
  bool print_details = false; // DEBUG
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

  unsigned int
  n_colors() const;

  unsigned int
  n_partitions(const unsigned int color = 0) const;

  unsigned int
  n_subdomains() const;

  unsigned int
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
   * Checks the compatibility of this partition data against @p other. Helpful
   * to decide if data might be re-used.
   */
  bool
  is_compatible(const PartitionData & other) const;

  /**
   * The set of partitions for each color. The outer vector represents
   * colors. Two consecutive integers of the inner vector represent a half-open
   * interval, so-called patch range.
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

  bool
  empty() const;

  bool
  empty_on_all() const;

  unsigned int
  n_cells_plain() const;

  unsigned int
  n_colors() const;

  unsigned int level = numbers::invalid_unsigned_int;

  /**
   * Temporary array storing all cells to build regular patches of cells as
   * CellIterator. During the initialization this array is compressed to @p
   * cell_level_and_index_pairs.
   */
  /// TODO this field becomes redundant if we rely on cell_level_and_index_pairs
  std::vector<CellIterator> cell_iterators;

  /**
   * This flat array stores all cells required to build regular patches of cells. The
   * successive alignment of patches of cells is such that all interior
   * subdomains are stored first, then the ones at the physical boundary. The
   * array has following lexicographical ordering:
   *
   *    cell_no  <  patch id  <  color
   *
   * Each cell within a regular patch @p patch_id of color @p color is identified by its
   * local number @p cell_no. The local cell numbering is again lexicographic.
   */
  std::vector<std::pair<int, int>> cell_level_and_index_pairs;

  /**
   * Numbers of physical subdomains for each color.
   */
  std::vector<SubdomainData> subdomain_quantities;

  /**
   * Numbers of physical subdomains (accumulated over colors)
   */
  SubdomainData subdomain_quantities_accumulated;

  /**
   * Underlying triangulation cell iterators are based on.
   */
  const Triangulation<dim> * triangulation = nullptr;

  // TODO we should not have to cache a DoFHandler !!!
  const DoFHandler<dim> * dof_handler = nullptr; // DEPR
};



template<int dim>
struct PatchInfo<dim>::SubdomainData
{
  unsigned int n_interior       = 0;
  unsigned int n_boundary       = 0;
  unsigned int n_interior_ghost = 0;
  unsigned int n_boundary_ghost = 0;
};



template<int dim>
struct FaceInfoLocal
{
  using cell_iterator_type = typename PatchInfo<dim>::cell_iterator_type;

  FaceInfoLocal(const unsigned int                      my_cell_no,
                const std::vector<cell_iterator_type> & cell_collection_in);

  static constexpr unsigned int
  n_faces();

  static constexpr unsigned int
  n_faces_1d();

  static constexpr unsigned int
  face_no(const unsigned int face_no_1d, const unsigned int dimension);

  static constexpr std::pair<unsigned int, unsigned int>
  face_no_1d_and_dimension(const unsigned int face_no);

  bool
  at_lower_neighbor(const unsigned int face_no) const;

  bool
  at_patch_boundary(const unsigned int face_no) const;

  unsigned int
  get_adjacent_cell_no(const unsigned int face_no) const;

  std::vector<unsigned int>
  get_adjacent_cell_numbers() const;

  std::vector<unsigned int>
  get_face_numbers_at_patch_boundary() const;

  std::vector<unsigned int>
  get_face_numbers_lower_neighbor() const;

  const unsigned int                  cell_number;
  const unsigned int                  n_cells;
  std::array<unsigned int, n_faces()> face_to_cell_number;
};



// --------------------------------   FaceInfoLocal   --------------------------------



template<int dim>
FaceInfoLocal<dim>::FaceInfoLocal(const unsigned int                      my_cell_no,
                                  const std::vector<cell_iterator_type> & cell_collection_in)
  : cell_number(my_cell_no), n_cells(cell_collection_in.size())
{
  AssertIndexRange(my_cell_no, cell_collection_in.size());
  const auto & my_cell = cell_collection_in[my_cell_no];

  /// If this cell @p my_cell shares its face @p face_no with an adjacent cell
  /// within the patch of cells @p cell_collection_in the local cell number of
  /// the neighbor is returned. Otherwise invalid_unsigned_int is returned.
  const auto find_adjacent_cell_no = [&](const auto face_no) {
    const auto my_neighbor_cell_index = my_cell->neighbor_index(face_no);
    if(my_neighbor_cell_index == -1) // at physical boundary
      return numbers::invalid_unsigned_int;
    for(auto cell_no = 0U; cell_no < cell_collection_in.size(); ++cell_no)
      if(cell_no != my_cell_no)
      {
        const auto & other_cell = cell_collection_in[cell_no];
        if(my_neighbor_cell_index == other_cell->index())
          return cell_no;
      }
    return numbers::invalid_unsigned_int;
  };

  //: fill the face_no to adjacent cell_no map @p face_to_cell_number
  for(auto face_no = 0U; face_no < n_faces(); ++face_no)
    face_to_cell_number[face_no] = find_adjacent_cell_no(face_no);
}


template<int dim>
inline constexpr unsigned int
FaceInfoLocal<dim>::n_faces()
{
  return GeometryInfo<dim>::faces_per_cell;
}


template<int dim>
inline constexpr unsigned int
FaceInfoLocal<dim>::n_faces_1d()
{
  return 2;
}


template<int dim>
inline constexpr unsigned int
FaceInfoLocal<dim>::face_no(const unsigned int face_no_1d, const unsigned int dimension)
{
  AssertIndexRange(face_no_1d, n_faces_1d());
  AssertIndexRange(dimension, dim);
  return dimension * n_faces_1d() + face_no_1d;
}


template<int dim>
inline constexpr std::pair<unsigned int, unsigned int>
FaceInfoLocal<dim>::face_no_1d_and_dimension(const unsigned int face_no)
{
  return std::make_pair(face_no % n_faces_1d(), face_no / n_faces_1d());
}


template<int dim>
inline bool
FaceInfoLocal<dim>::at_lower_neighbor(const unsigned int face_no) const
{
  AssertIndexRange(face_no, n_faces());
  const bool is_patch_interior_face    = !at_patch_boundary(face_no);
  const bool neighbor_cell_no_is_lower = face_to_cell_number[face_no] < cell_number;
  return is_patch_interior_face && neighbor_cell_no_is_lower;
}


template<int dim>
inline bool
FaceInfoLocal<dim>::at_patch_boundary(const unsigned int face_no) const
{
  AssertIndexRange(face_no, n_faces());
  return face_to_cell_number[face_no] == numbers::invalid_unsigned_int;
}


template<int dim>
inline unsigned int
FaceInfoLocal<dim>::get_adjacent_cell_no(const unsigned int face_no) const
{
  AssertIndexRange(face_no, n_faces());
  return face_to_cell_number[face_no];
}


template<int dim>
inline std::vector<unsigned int>
FaceInfoLocal<dim>::get_adjacent_cell_numbers() const
{
  std::vector<unsigned int> cell_numbers;
  for(auto face_no = 0U; face_no < n_faces(); ++face_no)
    if(get_adjacent_cell_no(face_no) != numbers::invalid_unsigned_int)
      cell_numbers.emplace_back(get_adjacent_cell_no(face_no));
  return cell_numbers;
}


template<int dim>
inline std::vector<unsigned int>
FaceInfoLocal<dim>::get_face_numbers_at_patch_boundary() const
{
  std::vector<unsigned int> face_numbers;
  for(auto face_no = 0U; face_no < n_faces(); ++face_no)
    if(at_patch_boundary(face_no))
      face_numbers.emplace_back(face_no);
  return face_numbers;
}


template<int dim>
inline std::vector<unsigned int>
FaceInfoLocal<dim>::get_face_numbers_lower_neighbor() const
{
  std::vector<unsigned int> face_numbers;
  for(auto face_no = 0U; face_no < n_faces(); ++face_no)
    if(at_lower_neighbor(face_no))
      face_numbers.emplace_back(face_no);
  return face_numbers;
}



// --------------------------------   PatchInfo   --------------------------------



template<int dim>
inline void
PatchInfo<dim>::clear()
{
  patch_starts.clear();
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
inline const typename PatchInfo<dim>::AdditionalData &
PatchInfo<dim>::get_additional_data() const
{
  return additional_data;
}


template<int dim>
inline std::pair<int, int>
PatchInfo<dim>::get_cell_level_and_index(const unsigned int cell_position) const
{
  AssertIndexRange(cell_position, n_cells_plain());
  AssertDimension(n_cells_plain(), get_internal_data()->cell_level_and_index_pairs.size());
  return (get_internal_data()->cell_level_and_index_pairs)[cell_position];
}


template<int dim>
inline const std::vector<std::pair<int, int>> &
PatchInfo<dim>::get_cell_level_and_index_pairs() const
{
  return get_internal_data()->cell_level_and_index_pairs;
}


template<int dim>
inline const typename PatchInfo<dim>::InternalData *
PatchInfo<dim>::get_internal_data() const
{
  return &internal_data;
}


template<int dim>
inline const Triangulation<dim> &
PatchInfo<dim>::get_triangulation() const
{
  const auto tria = get_internal_data()->triangulation;
  Assert(tria, ExcMessage("Triangulation not set."));
  return *tria;
}


template<int dim>
inline unsigned int
PatchInfo<dim>::n_cells_plain() const
{
  return get_internal_data()->n_cells_plain();
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
inline unsigned int
PatchInfo<dim>::PartitionData::n_colors() const
{
  return partitions.size();
}


template<int dim>
inline unsigned int
PatchInfo<dim>::PartitionData::n_partitions(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  return (partitions[color].size() > 0) ? partitions[color].size() - 1 : 0;
}


template<int dim>
inline unsigned int
PatchInfo<dim>::PartitionData::n_subdomains() const
{
  return partitions.empty() ? 0 : (partitions.back().empty() ? 0 : partitions.back().back());
}


template<int dim>
inline unsigned int
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
  subdomain_quantities.clear();
  subdomain_quantities_accumulated = SubdomainData{};
  cell_iterators.clear();
  cell_level_and_index_pairs.clear();
  triangulation = nullptr;
  dof_handler   = nullptr;
}


template<int dim>
inline bool
PatchInfo<dim>::InternalData::empty() const
{
  return n_cells_plain() == 0;
}


template<int dim>
inline bool
PatchInfo<dim>::InternalData::empty_on_all() const
{
  const auto n_cells_plain_mpimax = Utilities::MPI::max(n_cells_plain(), MPI_COMM_WORLD);
  return n_cells_plain_mpimax == 0;
}


template<int dim>
inline unsigned int
PatchInfo<dim>::InternalData::n_cells_plain() const
{
  return std::max(cell_iterators.size(), cell_level_and_index_pairs.size());
}


template<int dim>
inline unsigned int
PatchInfo<dim>::InternalData::n_colors() const
{
  return subdomain_quantities.size();
}



} // end namespace TPSS

#include "patch_info.templates.h"

#endif // end inclusion guard
