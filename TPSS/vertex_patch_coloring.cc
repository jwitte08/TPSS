
/*
 * user-defined red-black coloring of vertex patches on a hyper-rectangle
 *
 *  Created on: Sep 17, 2019
 *      Author: witte
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
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

// // LATER
// using IndexIteratorPair = typename std::pair<unsigned,typename DoFHandler<dim>::level_cell_iterator>;
//   const auto & compare_func = [](const IndexIteratorPair & lhs, const IndexIteratorPair & rhs){return std::less<unsigned>(lhs.first,rhs.first);};
//   std::set<IndexIteratorPair,decltype(compare_func)> global_indices_with_ghost;

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

  std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>
                         cell_collections; // temporary vector gathering all
  constexpr unsigned int regular_vpatch_size = 1 << dim;
  const auto &           tria                = dof_handler.get_triangulation();

  std::map <unsigned int, std::pair<unsigned int, unsigned int>> global_to_local_map;
  std::ostringstream oss;
  oss << "process " << tria.locally_owned_subdomain() << " reports:\n";
  const auto locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
  		     IteratorFilters::LocallyOwnedLevelCell());
  unsigned int local_index = 0;
  oss << "loop over locally owned\n";
  for (const auto & cell : locally_owned_range_mg)
  {
      oss << "cell " << cell->index()
	  << " is locally owned " << cell->is_locally_owned_on_level()
	  << ", is ghosted " << cell->is_ghost()
	  << ", is artificial " << cell->is_artificial()
	  << std::endl;
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
	if(!is_boundary_vertex(cell, v))
	  {
	    const unsigned int global_index = cell->vertex_index(v);
	    const auto element = global_to_local_map.find (global_index);
	    if (element != global_to_local_map.cend())
	      ++(element->second.second);
	    else
	      {
		const auto local_index_pair = std::pair<unsigned, unsigned>{local_index++,1};
		const auto status = global_to_local_map.insert(std::make_pair(global_index, local_index_pair));
		Assert (status.second, ExcMessage("failed to insert key-value-pair"))
	      }
	  }
  }
    
  const unsigned int my_subdomain_id = tria.locally_owned_subdomain();
  // TODO balance vertex patches with ghosts better
  const auto partial_ghost_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
  		     [](const auto & cell){return cell->is_ghost();},
		     [&,my_subdomain_id](const auto & cell){return tria.locally_owned_subdomain() < cell->level_subdomain_id();});
  oss << "loop over ghosts\n";
  std::set<unsigned> global_indices_with_ghost;
  for (const auto & cell : partial_ghost_range_mg)
  {
      oss << "cell " << cell->index()
	  << " is locally owned " << cell->is_locally_owned_on_level()
	  << ", is ghosted " << cell->is_ghost()
	  << ", is artificial " << cell->is_artificial()
	  << std::endl;
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
	  {
	    const unsigned int global_index = cell->vertex_index(v);
	    const auto element = global_to_local_map.find (global_index);
	    if (element != global_to_local_map.cend())
	      ++(element->second.second);
	  }
  }

  // TODO see TODO balance ...
  oss << "erase irregular patches:\n";
      const auto & is_invalid_patch = [](const auto & key_value){
      const unsigned int n_cells = key_value.second.second;
      const bool is_invalid = (n_cells != regular_vpatch_size); // TODO non-irregular cases?
      return is_invalid;
      };
  while (std::any_of (global_to_local_map.begin(),global_to_local_map.end(), is_invalid_patch))
    for (const auto key_value : global_to_local_map)
      {
	const unsigned int n_cells = key_value.second.second;
	const unsigned int global_index = key_value.first;
	if (n_cells != regular_vpatch_size)
	  global_to_local_map.erase(global_index);
      }
  unsigned int new_local_index = 0;
  for (auto & key_value : global_to_local_map)
    key_value.second.first = new_local_index++;
  
  oss << "print global-to-local-mapping\n";
  for (const auto key_value : global_to_local_map)
    oss << key_value.first << ", ("
	<< key_value.second.first << "," << key_value.second.second << ")   ";
  std::cout << oss.str() << std::endl;

  // *** we gather CellIterators into patches (cell_collections)
  const unsigned n_subdomains = global_to_local_map.size();
  AssertDimension(n_subdomains,new_local_index);
  cell_collections.clear();
  cell_collections.resize(n_subdomains);
  for(auto & cell : dof_handler.mg_cell_iterators_on_level(level))
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        const unsigned int global_index = cell->vertex_index(v);
	const auto element = global_to_local_map.find (global_index);
	if (element != global_to_local_map.cend())
	  {
	    const unsigned int local_index = element->second.first;
	    auto & collection = cell_collections[local_index];
	    const unsigned patch_size = element->second.second;
	    if(collection.empty())
	      collection.resize(patch_size);
	    if(patch_size == regular_vpatch_size) // regular patch
	      collection[regular_vpatch_size - 1 - v] = cell;
	    else // irregular patch
	      Assert (false, ExcMessage("TODO"));
	  }
      }
  return cell_collections;

  // // PRE-PROCESSING
  // std::vector<unsigned int> cell_count(tria.n_vertices(), 0);
  // std::vector<unsigned int> vloc_map(tria.n_vertices(), -1);
  // unsigned int              vg_max       = 0;
  // unsigned int              n_subdomains = 0;

  // // *** map each interior vertex (vg, i.e. global index) onto a
  // // *** local index (vloc) and count the amount of cells
  // // *** belonging to each vertex
  // for(auto & cell : dof_handler.mg_cell_iterators_on_level(level))
  //   for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
  //     if(!is_boundary_vertex(cell, v))
  //     {
  //       const unsigned int vg = cell->vertex_index(v);
  //       vg_max                = std::max(vg_max, vg);
  //       if(vloc_map[vg] == static_cast<unsigned int>(-1))
  //         vloc_map[vg] = n_subdomains++;
  //       ++cell_count[vg];
  //     }
  // std::pair<unsigned int, unsigned int> vg_range(
  //   0, vg_max + 1); // half-open range of global vertex indices to be considered
  // cell_count.resize(vg_range.second);
  // vloc_map.resize(vg_range.second);

  // // *** count the amount of regular and irregular patches
  // unsigned int n_regular_patches =
  //   std::count_if(cell_count.cbegin(), cell_count.cend(), [](const auto & n) {
  //     return regular_vpatch_size == n;
  //   });
  // unsigned int n_irregular_patches =
  //   std::count_if(cell_count.cbegin(), cell_count.cbegin() + vg_max + 1, [](const auto & n) {
  //     return (regular_vpatch_size != n) && (0 != n);
  //   });
  // (void)n_regular_patches, (void)n_irregular_patches;
  // AssertDimension(n_subdomains, n_regular_patches + n_irregular_patches);
  // // TODO treat irregular patches
  // Assert(n_irregular_patches == 0, ExcNotImplemented());

  // // *** we gather CellIterators into patches (cell_collections)
  // cell_collections.clear();
  // cell_collections.resize(n_subdomains);
  // for(auto & cell : dof_handler.mg_cell_iterators_on_level(level))
  //   for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
  //     if(!is_boundary_vertex(cell, v))
  //     {
  //       const unsigned int vg   = cell->vertex_index(v);
  //       const unsigned int vloc = vloc_map[vg];
  //       Assert(vloc != static_cast<unsigned int>(-1), ExcInternalError());
  //       auto &     collection = cell_collections[vloc];
  //       const auto patch_size = cell_count[vg];

  //       if(collection.empty())
  //         collection.resize(patch_size);
  //       if(patch_size == regular_vpatch_size) // regular patch
  //         collection[regular_vpatch_size - 1 - v] = cell;
  //       else // irregular patch
  //         Assert(false, ExcNotImplemented());
  //     }
  // return cell_collections;
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
  using PatchIterator                           = typename TPSS::PatchInfo<dim>::PatchIterator;
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
    AssertIndexRange(color, (1<<dim)*2);

    patch_iterators[color].emplace_back(patch);
  }
  return patch_iterators;
}



struct TestParameter
{
  unsigned int n_refinements = 1;
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
  dof_handler.distribute_dofs(FE_Q<dim>(1));
  dof_handler.distribute_mg_dofs();
  const unsigned int global_level = tria.n_global_levels() - 1;

  // std::cout << tria.locally_owned_subdomain() << " n_vertices: " << tria.n_vertices() << std::endl;
  // std::cout << tria.locally_owned_subdomain() << " n_used_vertices: " << tria.n_used_vertices() << std::endl;

  // const auto locally_owned_range_mg =
  //   filter_iterators(dof_handler.mg_cell_iterators_on_level(global_level),
  // 		     IteratorFilters::LocallyOwnedCell(),
  // 		     IteratorFilters::AtBoundary());
  // std::vector<unsigned> vertices;
  // for (const auto & cell : locally_owned_range_mg)
  //   for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
  //   {
  //     vertices.push_back(cell->vertex_index(v));
  //   }
  // std::sort(vertices.begin(),vertices.end());
  // const auto last = std::unique(vertices.begin(),vertices.end());
  // vertices.erase(last, vertices.end());
  // std::ostringstream oss;
  // oss << tria.locally_owned_subdomain() << " n_vertices: " << tria.n_vertices() << std::endl;
  // for (unsigned v : vertices)
  //   oss << v << " ";
  // oss << "\n\n";
  // std::cout << oss.str();
  // // for(const auto & cell : filtered_iterators_range)

  // gather vertex patches
  const auto patch_collection = gather(dof_handler, global_level);

  // // color vertex patches
  // const auto patch_iterators = get_coloring<dim>(patch_collection);
  
  // // collect output data
  // DataOut<dim> data_out;
  // data_out.attach_dof_handler(dof_handler);

  // Vector<double> subdomain(tria.n_active_cells());
  // for(unsigned int i = 0; i < subdomain.size(); ++i)
  //   subdomain(i) = tria.locally_owned_subdomain();
  // data_out.add_data_vector(subdomain, "subdomain");

  // // mark each single patch
  // std::vector<Vector<double>> patch_markers;
  // for(const auto & patch : patch_collection)
  // {
  //   Vector<double> patch_marker;
  //   patch_marker.reinit(tria.n_active_cells());
  //   std::fill(patch_marker.begin(), patch_marker.end(), 0);
  //   for(const auto & cell : patch)
  //   {
  //     const auto active_cell          = typename DoFHandler<dim>::active_cell_iterator{&tria,
  //                                                                             cell->level(),
  //                                                                             cell->index(),
  //                                                                             &dof_handler};
  //     const auto active_cell_index    = active_cell->active_cell_index();
  //     patch_marker(active_cell_index) = 1;
  //   }
  //   patch_markers.emplace_back(patch_marker);
  // }
  // unsigned int patch_no = 0;
  // for(const auto & patch_marker : patch_markers)
  // {
  //   std::string patch_name = "patch" + Utilities::int_to_string(patch_no++, 4);
  //   data_out.add_data_vector(patch_marker, patch_name);
  // }

  // // for each color mark all patches
  // std::vector<Vector<double>> colored_markers;
  // for(unsigned color = 0; color < patch_iterators.size(); ++color)
  // {
  //   Vector<double> marker;
  //   marker.reinit(tria.n_active_cells());
  //   std::fill(marker.begin(), marker.end(), 0);
  //   const auto & patch_collection = patch_iterators[color];
  //   for(const auto & patch : patch_collection)
  //     for(const auto & cell : (*patch))
  //     {
  //       const auto active_cell       = typename DoFHandler<dim>::active_cell_iterator{&tria,
  //                                                                               cell->level(),
  //                                                                               cell->index(),
  //                                                                               &dof_handler};
  //       const auto active_cell_index = active_cell->active_cell_index();
  //       marker(active_cell_index)    = 1;
  //     }
  //   colored_markers.emplace_back(marker);
  // }
  // unsigned int color = 0;
  // AssertDimension(colored_markers.size(), 2 * (1 << dim));
  // for(const auto & marker : colored_markers)
  // {
  //   std::string color_name = "color" + Utilities::int_to_string(color++, 2);
  //   data_out.add_data_vector(marker, color_name);
  // }

  // // write paraview files
  // data_out.build_patches();
  // {
  //   std::ofstream file(
  //     "data-active-" + Utilities::int_to_string(dim) + "d-" +
  //     Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4) + ".vtu");
  //   data_out.write_vtu(file);
  // }
  // if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  // {
  //   std::vector<std::string> filenames;
  //   for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
  //     filenames.push_back("data-active-" + Utilities::int_to_string(dim) + "d-" +
  //                         Utilities::int_to_string(i, 4) + ".vtu");

  //   std::ofstream master_output("data-active-" + Utilities::int_to_string(dim) + "d.pvtu");
  //   data_out.write_pvtu_record(master_output, filenames);
  // }
}



int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  TestParameter prms;
  if(argc > 1)
    prms.n_refinements = std::atoi(argv[1]);
  test<2>(prms);
  // test<3>();
}
