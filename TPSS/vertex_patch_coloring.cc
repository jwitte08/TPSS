
/*
 * user-defined red-black coloring of vertex patches on a hyper-rectangle
 *
 *  Created on: Sep 17, 2019
 *      Author: witte
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "solvers_and_preconditioners/TPSS/patch_info.h"

using namespace dealii;

template<typename T>
std::string
vector_to_string(const std::vector<T> & vector)
{
  std::ostringstream oss;
  oss << "{";
  for(unsigned i = 0; i < vector.size(); ++i)
    oss << vector[i] << ((i + 1) < vector.size() ? ", " : "}");
  return oss.str();
}

template<typename T>
std::string
set_to_string(const std::set<T> & set)
{
  std::vector<T> set_as_vector(set.cbegin(), set.cend());
  return vector_to_string(set_as_vector);
}

struct GhostPatch
{
  GhostPatch(const unsigned int proc, const CellId & cell_id)
  {
    submit_id(proc, cell_id);
  }

  void
  submit_id(const unsigned int proc, const CellId & cell_id)
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
      Assert(status.second, ExcMessage("failed to insert key-value-pair"));
    }
  }

  std::string
  str() const
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

  std::map<unsigned, std::vector<CellId>> proc_to_cell_ids;
};

template<int dim>
std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>
gather(const DoFHandler<dim> & dof_handler, const unsigned int level)
{
  // LAMBDA checks if a vertex is at the physical boundary
  auto && is_boundary_vertex = [](const typename DoFHandler<dim>::level_cell_iterator & cell,
                                  const unsigned int                                    vertex_id) {
    return std::any_of(std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       [&cell](const auto & face_no) { return cell->at_boundary(face_no); });
  };
  constexpr unsigned int regular_vpatch_size = 1 << dim;
  const auto &           tria                = dof_handler.get_triangulation();
  std::ostringstream     oss;
  oss << "process " << tria.locally_owned_subdomain() << " reports:\n";
  const auto locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                     IteratorFilters::LocallyOwnedLevelCell());

  /**
   * A mapping @p global_to_local_map between the global vertex and
   * the pair containing the number of locally owned cells and the
   * number of all cells (including ghosts) is constructed
   */
  std::map<unsigned int, std::pair<unsigned int, unsigned int>> global_to_local_map;
  for(const auto & cell : locally_owned_range_mg)
  {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      if(!is_boundary_vertex(cell, v))
      {
        const unsigned int global_index = cell->vertex_index(v);
        const auto         element      = global_to_local_map.find(global_index);
        if(element != global_to_local_map.cend())
        {
          ++(element->second.first);
          ++(element->second.second);
        }
        else
        {
          const auto n_cells_pair = std::pair<unsigned, unsigned>{1, 1};
          const auto status =
            global_to_local_map.insert(std::make_pair(global_index, n_cells_pair));
          Assert(status.second, ExcMessage("failed to insert key-value-pair"))
        }
      }
  }

  /**
   * Ghost patches are stored as the mapping @p global_to_ghost_id
   * between the global vertex index and GhostPatch. The number of
   * cells, book-kept in @p global_to_local_map, is updated taking the
   * ghost cells into account.
   */
  // TODO: is_ghost_on_level() missing
  const auto not_locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                     [](const auto & cell) { return !(cell->is_locally_owned_on_level()); });
  std::map<unsigned int, GhostPatch> global_to_ghost_id;
  for(const auto & cell : not_locally_owned_range_mg)
  {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      const unsigned int global_index = cell->vertex_index(v);
      const auto         element      = global_to_local_map.find(global_index);
      if(element != global_to_local_map.cend())
      {
        ++(element->second.second);
        const unsigned int subdomain_id_ghost = cell->level_subdomain_id();
        const auto         ghost              = global_to_ghost_id.find(global_index);
        if(ghost != global_to_ghost_id.cend())
          ghost->second.submit_id(subdomain_id_ghost, cell->id());
        else
        {
          const auto status =
            global_to_ghost_id.emplace(global_index, GhostPatch(subdomain_id_ghost, cell->id()));
          Assert(status.second, ExcMessage("failed to insert key-value-pair"));
        }
      }
    }
  }

  // oss << "ghost patches (raw):" << std::endl;
  // for(const auto key_value : global_to_ghost_id)
  //   oss << key_value.first /*<< ", " << key_value.second.str()*/ << "   ";
  // oss << std::endl;
  // oss << "global-to-local mapping (raw):" << std::endl;
  // for(const auto key_value : global_to_local_map)
  //   oss << key_value.first << ", (" << key_value.second.first << "," << key_value.second.second
  //       << ")   ";
  // oss << std::endl;

  { // ASSIGN GHOSTS
    const unsigned int my_subdomain_id = tria.locally_owned_subdomain();
    /**
     * logic: if the mpi-proc owns more than half of the cells within
     *        a ghost patch he takes ownership
     */
    {
      //: (1) add subdomain_ids of locally owned cells to GhostPatches
      for(const auto & cell : locally_owned_range_mg)
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const unsigned global_index = cell->vertex_index(v);
          const auto     ghost        = global_to_ghost_id.find(global_index);
          //: checks if the global vertex has ghost cells attached
          if(ghost != global_to_ghost_id.end())
            ghost->second.submit_id(my_subdomain_id, cell->id());
        }

      std::set<unsigned> to_be_owned;
      std::set<unsigned> to_be_erased;
      for(const auto key_value : global_to_ghost_id)
      {
        const unsigned int global_index     = key_value.first;
        const auto &       proc_to_cell_ids = key_value.second.proc_to_cell_ids;

        const auto & get_proc_with_most_cellids = [](const auto & lhs, const auto & rhs) {
          const std::vector<CellId> & cell_ids_lhs = lhs.second;
          const std::vector<CellId> & cell_ids_rhs = rhs.second;
          Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
          Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
          return (cell_ids_lhs.size() < cell_ids_rhs.size());
        };

        const auto         most                       = std::max_element(proc_to_cell_ids.cbegin(),
                                           proc_to_cell_ids.cend(),
                                           get_proc_with_most_cellids);
        const unsigned int subdomain_id_most          = most->first;
        const unsigned int n_locally_owned_cells_most = most->second.size();
        const auto         member                     = global_to_local_map.find(global_index);
        Assert(member != global_to_local_map.cend(), ExcMessage("must be listed as patch"));
        const unsigned int n_cells = member->second.second;
        if(my_subdomain_id == subdomain_id_most)
        {
          AssertDimension(member->second.first, n_locally_owned_cells_most);
          if(2 * n_locally_owned_cells_most > n_cells)
            to_be_owned.insert(global_index);
        }
        else
        {
          if(2 * n_locally_owned_cells_most > n_cells)
            to_be_erased.insert(global_index);
        }
      }

      for(const unsigned global_index : to_be_owned)
      {
        auto & my_patch = global_to_local_map[global_index];
        my_patch.first  = my_patch.second;
        global_to_ghost_id.erase(global_index);
      }
      for(const unsigned global_index : to_be_erased)
      {
        global_to_local_map.erase(global_index);
        global_to_ghost_id.erase(global_index);
      }

      // oss << "ghost patches (more than half):" << std::endl;
      // for(const auto key_value : global_to_ghost_id)
      //   oss << key_value.first /*<< ", " << key_value.second.str()*/ << "   ";
      // oss << std::endl;
      // oss << "global-to-local mapping (more than half):" << std::endl;
      // for(const auto key_value : global_to_local_map)
      //   oss << key_value.first << ", (" << key_value.second.first << "," <<
      //   key_value.second.second
      //       << ")   ";
      // oss << std::endl;
    }

    /**
     * logic: the owner of the cell with the lowest CellId takes ownership
     */
    {
      //: (2) determine mpi-proc with the minimal CellId for all GhostPatches
      std::set<unsigned> to_be_owned;
      for(const auto key_value : global_to_ghost_id)
      {
        const unsigned int global_index     = key_value.first;
        const auto &       proc_to_cell_ids = key_value.second.proc_to_cell_ids;

        const auto & get_proc_with_min_cellid = [](const auto & lhs, const auto & rhs) {
          std::vector<CellId> cell_ids_lhs = lhs.second;
          Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
          std::sort(cell_ids_lhs.begin(), cell_ids_lhs.end());
          const auto          min_cell_id_lhs = cell_ids_lhs.front();
          std::vector<CellId> cell_ids_rhs    = rhs.second;
          Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
          std::sort(cell_ids_rhs.begin(), cell_ids_rhs.end());
          const auto min_cell_id_rhs = cell_ids_rhs.front();
          return min_cell_id_lhs < min_cell_id_rhs;
        };

        const auto min = std::min_element(proc_to_cell_ids.cbegin(),
                                          proc_to_cell_ids.cend(),
                                          get_proc_with_min_cellid);

        const unsigned int subdomain_id_min = min->first;
        if(my_subdomain_id == subdomain_id_min)
          to_be_owned.insert(global_index);
      }

      //: (3) set owned GhostPatches in global_to_local_map and delete all remaining
      for(const unsigned global_index : to_be_owned)
      {
        auto & my_patch = global_to_local_map[global_index];
        my_patch.first  = my_patch.second;
        global_to_ghost_id.erase(global_index);
      }
      for(const auto key_value : global_to_ghost_id)
      {
        const unsigned int global_index = key_value.first;
        global_to_local_map.erase(global_index);
      }
    }

    // oss << "ghost patches (final):" << std::endl;
    // for(const auto key_value : global_to_ghost_id)
    //   oss << key_value.first /*<< ", " << key_value.second.str()*/ << "   ";
    // oss << std::endl;
    // oss << "global-to-local mapping (final):" << std::endl;
    // for(const auto key_value : global_to_local_map)
    //   oss << key_value.first << ", (" << key_value.second.first << "," << key_value.second.second
    //       << ")   ";
    // oss << std::endl;
  }

  /**
   * Enumerate the patches contained in @p global_to_local_map by
   * replacing the former number of locally owned cells in terms of a
   * consecutive numbering. The local numbering is required for
   * gathering the level cell iterators into a collection @
   * cell_collections according to the global vertex index.
   */
  unsigned int local_index = 0;
  for(auto & key_value : global_to_local_map)
    key_value.second.first = local_index++;
  const unsigned n_subdomains = global_to_local_map.size();
  AssertDimension(n_subdomains, local_index);
  std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>
    cell_collections; // temporary vector gathering all
  cell_collections.resize(n_subdomains);
  for(auto & cell : dof_handler.mg_cell_iterators_on_level(level))
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      const unsigned int global_index = cell->vertex_index(v);
      const auto         element      = global_to_local_map.find(global_index);
      if(element != global_to_local_map.cend())
      {
        const unsigned int local_index = element->second.first;
        const unsigned int patch_size  = element->second.second;
        auto &             collection  = cell_collections[local_index];
        if(collection.empty())
          collection.resize(patch_size);
        if(patch_size == regular_vpatch_size) // regular patch
          collection[regular_vpatch_size - 1 - v] = cell;
        else // irregular patch
          AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
      }
    }

  std::cout << oss.str() << "\n\n";
  return cell_collections;
}


template<int dim>
Point<dim, unsigned int>
get_integer_coords(const CellId cell_id)
{
  // Get child indices
  std::vector<unsigned int> child_indices;
  std::string               cell_id_str = cell_id.to_string();
  while(cell_id_str.size() > 4)
  {
    child_indices.insert(child_indices.begin(), Utilities::string_to_int(&(cell_id_str.back())));
    cell_id_str.pop_back();
  }

  // Initialize global coordinate with coarse cell coordinate
  Point<dim, unsigned int> global_coord;
  const unsigned int       coarse_id = cell_id.to_binary<dim>()[0];
  {
    const std::bitset<dim> bit_indices(coarse_id);
    for(unsigned int d = 0; d < dim; ++d)
      global_coord(d) = bit_indices[d];
  }

  // Compute local coordinate and add to global
  for(auto c : child_indices)
  {
    Point<dim, unsigned int> local_coord;
    {
      const std::bitset<dim> bit_indices(c);
      for(unsigned int d = 0; d < dim; ++d)
        local_coord(d) = bit_indices[d];
    }

    global_coord *= 2;
    global_coord += local_coord;
  }

  return global_coord;
}



template<int dim>
std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>>
get_coloring(
  const std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>> & uncolored_patches)
{
  using PatchIterator = typename TPSS::PatchInfo<dim>::PatchIterator;
  std::vector<std::vector<PatchIterator>> patch_iterators;
  const unsigned int                      n_colors = 2 * (1 << dim);
  patch_iterators.resize(n_colors);
  for(PatchIterator patch = uncolored_patches.cbegin(); patch != uncolored_patches.cend(); ++patch)
  {
    // determine shift of the lowest leftmost cell
    const auto & shift = [](const PatchIterator & patch, const unsigned direction) {
      AssertIndexRange(direction, dim);
      const unsigned stride                 = 1 << direction;
      const auto     cell_left              = (*patch)[0];
      const auto     cell_right             = (*patch)[stride];
      const auto     coord_left             = get_integer_coords<dim>(cell_left->id());
      const auto     coord_right            = get_integer_coords<dim>(cell_right->id());
      const auto     patch_coord_from_left  = coord_left(direction) / 2;
      const auto     patch_coord_from_right = coord_right(direction) / 2;
      return patch_coord_from_left != patch_coord_from_right;
    };
    std::bitset<dim> shift_mask;
    for(unsigned int d = 0; d < dim; ++d)
      shift_mask[d] = (shift(patch, d));
    AssertIndexRange(shift_mask.to_ulong(), 1 << dim);

    // (1 << dim) layers of red-black colorings (layer is determined by the shift)
    const auto                     cell        = patch->front();
    const Point<dim, unsigned int> coordinates = get_integer_coords<dim>(cell->id());
    unsigned int                   sum         = 0;
    for(unsigned int d = 0; d < dim; ++d)
      sum += (coordinates(d) + static_cast<unsigned>(shift_mask[d])) / 2;
    const unsigned red_or_black = sum % 2;
    const unsigned color        = 2 * shift_mask.to_ulong() + red_or_black;
    AssertIndexRange(color, (1 << dim) * 2);

    patch_iterators[color].emplace_back(patch);
  }
  return patch_iterators;
}



struct TestParameter
{
  unsigned int n_refinements = 3;
};



template<int dim>
void
test(const TestParameter & prms)
{
  // create tria and distribute dofs
  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::limit_level_difference_at_vertices,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global(prms.n_refinements);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_DGQ<dim>(0));
  dof_handler.distribute_mg_dofs();
  const unsigned int global_level = tria.n_global_levels() - 1;

  // gather vertex patches
  const auto patch_collection = gather(dof_handler, global_level);

  // color vertex patches
  const auto patch_iterators = get_coloring<dim>(patch_collection);

  for(unsigned color = 0; color < patch_iterators.size(); ++color)
  {
    // collect output data
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    Vector<double> subdomain(tria.n_active_cells());
    for(unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain", DataOut<dim>::type_cell_data);

    LinearAlgebra::distributed::Vector<double> marker;
    IndexSet                                   locally_relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, global_level, locally_relevant_dofs);
    marker.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    marker                        = 0;
    const auto & patch_collection = patch_iterators[color];
    for(const auto & patch : patch_collection)
      for(const auto & cell : (*patch))
      {
        const auto            active_cell = typename DoFHandler<dim>::active_cell_iterator{&tria,
                                                                                cell->level(),
                                                                                cell->index(),
                                                                                &dof_handler};
        std::vector<unsigned> dof_indices(dof_handler.get_fe().dofs_per_cell);
        active_cell->get_dof_indices(dof_indices);
        const auto active_cell_index = dof_indices.front(); // active_cell->active_cell_index();
        marker(active_cell_index)    = 1;
      }
    marker.compress(VectorOperation::add);
    data_out.add_data_vector(marker, "coloring", DataOut<dim>::type_dof_data);

    // write paraview files
    std::string color_name = "_color" + Utilities::int_to_string(color, 2);
    data_out.build_patches();
    {
      std::ofstream file(
        "data-active-" + Utilities::int_to_string(dim) + "d-" +
        Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4) + color_name +
        ".vtu");
      data_out.write_vtu(file);
    }
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::vector<std::string> filenames;
      for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
        filenames.push_back("data-active-" + Utilities::int_to_string(dim) + "d-" +
                            Utilities::int_to_string(i, 4) + color_name + ".vtu");

      std::ofstream master_output("data-active-" + Utilities::int_to_string(dim) + "d" +
                                  color_name + ".pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  }
}



int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  TestParameter prms;
  if(argc > 1)
    prms.n_refinements = std::atoi(argv[1]);
  test<2>(prms);
  test<3>(prms);
}
