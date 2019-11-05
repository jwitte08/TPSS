
/*
 * user-defined coloring algorithms:
 *
 * cell-based: red-black on hyper-rectangle
 *
 * vertex patch: multiple red-black on hyper-rectangle
 *
 *  Created on: Sep 20, 2019
 *      Author: witte
 */

#ifndef TPSS_COLORING_H
#define TPSS_COLORING_H

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

#include "mesh.h"

using namespace dealii;

template<int dim>
Point<dim, unsigned int>
get_integer_coords(const CellId cell_id)
{
  // Get child indices
  std::vector<unsigned int> child_indices;
  std::string               cell_id_str = cell_id.to_string();
  // // TODO better solution than this hack
  // std::cout << cell_id_str << std::endl;

  const auto pos_of_colon = cell_id_str.find(':');
  Assert(pos_of_colon > 0, ExcInternalError());
  // const auto pos_of_underscore = cell_id_str.find('_');
  // Assert(pos_of_underscore != std::string::npos, ExcMessage("No underscore found."));
  // const auto root_cell_str = cell_id_str.substr(0, pos_of_underscore);
  // const auto root_cell_idx = std::stoul(root_cell_str);
  while(cell_id_str.size() > (pos_of_colon + 1))
  {
    child_indices.insert(child_indices.begin(), Utilities::string_to_int(&(cell_id_str.back())));
    cell_id_str.pop_back();
  }

  // Initialize global coordinate with coarse cell coordinate
  Point<dim, unsigned int> global_coord;
  const unsigned int       root_cell_index = cell_id.to_binary<dim>()[0];
  AssertThrow(
    root_cell_index < (1 << dim),
    ExcMessage(
      "TODO Algorithm not implemented for root meshes which differ from a single cell or vertex patch."));
  {
    const std::bitset<dim> bit_indices(root_cell_index);
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
struct RedBlackColoring
{
  using CellIterator   = typename TPSS::PatchInfo<dim>::CellIterator;
  using PatchIterator  = typename TPSS::PatchInfo<dim>::PatchIterator;
  using AdditionalData = typename TPSS::PatchInfo<dim>::AdditionalData;

  std::array<unsigned int, dim> n_subdivisions;

  RedBlackColoring() = delete;

  RedBlackColoring(const MeshParameter & mesh_prms) : n_subdivisions(get_subdivisions(mesh_prms))
  {
  }



  std::array<unsigned int, dim>
  get_subdivisions(const MeshParameter & mesh_prms)
  {
    std::array<unsigned int, dim> n_subdivisions;
    const bool                    is_hypercube =
      MeshParameter::GeometryVariant::Cube == mesh_prms.geometry_variant ||
      MeshParameter::GeometryVariant::CubeDistorted == mesh_prms.geometry_variant;
    if(is_hypercube)
    {
      Assert(mesh_prms.n_repetitions > 0, ExcMessage("At least one (isotropic) repetition."));
      n_subdivisions.fill(mesh_prms.n_repetitions);
    }
    else
      AssertThrow(false, ExcMessage("TODO"));
    return n_subdivisions;
  }



  std::vector<std::vector<PatchIterator>>
  operator()(const std::vector<std::vector<CellIterator>> & patches,
             const AdditionalData                           additional_data)
  {
    std::vector<std::vector<PatchIterator>> coloring;
    if(additional_data.patch_variant == TPSS::PatchVariant::cell)
    {
      coloring = std::move(cell_coloring_impl(patches));
    }
    else if(additional_data.patch_variant == TPSS::PatchVariant::vertex)
    {
      coloring = std::move(vertex_patch_coloring_impl(patches));
    }
    else
      AssertThrow(false, ExcNotImplemented());
    return coloring;
  }



  std::vector<std::vector<PatchIterator>>
  cell_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const
  {
    std::vector<std::vector<PatchIterator>> colored_patches;
    colored_patches.resize(2);

    for(auto patch = patches.cbegin(); patch != patches.cend(); ++patch)
    {
      AssertDimension(patch->size(), 1);
      const auto & cell = patch->front();

      // Get integer coordinates
      Point<dim, unsigned int> cell_int_coords = get_integer_coords<dim>(cell->id());

      // If integer coordinates sum to an even
      // number give color 0, else give color 1
      unsigned int color = 0;
      unsigned int sum   = 0;
      for(unsigned int d = 0; d < dim; ++d)
        sum += cell_int_coords(d);
      if(sum % 2 == 1)
        color = 1;

      // Add patch to coloring
      colored_patches[color].push_back(patch);
    }
    return colored_patches;
  }



  std::vector<std::vector<PatchIterator>>
  vertex_patch_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const
  {
    std::vector<std::vector<PatchIterator>> colored_patches;
    const unsigned int                      n_colors = 2 * (1 << dim);
    colored_patches.resize(n_colors);
    for(PatchIterator patch = patches.cbegin(); patch != patches.cend(); ++patch)
    {
      // determine shift of the lower left cell
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

      colored_patches[color].emplace_back(patch);
    }
    return colored_patches;
  }



  static void
  visualize_coloring(const DoFHandler<dim> & dof_handler_in,
                     const std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>> &
                                       colored_iterators,
                     const std::string prefix)
  {
    for(unsigned color = 0; color < colored_iterators.size(); ++color)
    {
      // collect output data
      const auto &       tria         = dof_handler_in.get_triangulation();
      const unsigned int global_level = tria.n_global_levels() - 1;
      DoFHandler<dim>    dof_handler(tria);
      dof_handler.distribute_dofs(FE_DGQ<dim>(0)); // require one (dof) value per cell !!!
      dof_handler.distribute_mg_dofs();
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      // print the subdomain_ids
      Vector<double> subdomain(tria.n_active_cells());
      for(unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = tria.locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain", DataOut<dim>::type_cell_data);

      // initialize the ghosted dof vector
      LinearAlgebra::distributed::Vector<double> marker;
      IndexSet                                   locally_relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                    global_level,
                                                    locally_relevant_dofs);
      marker.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
      marker                        = 0;
      const auto & patch_collection = colored_iterators[color];
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
          prefix + "data-active-" + Utilities::int_to_string(dim) + "d-" +
          Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4) +
          color_name + ".vtu");
        data_out.write_vtu(file);
      }
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          filenames.push_back(prefix + "data-active-" + Utilities::int_to_string(dim) + "d-" +
                              Utilities::int_to_string(i, 4) + color_name + ".vtu");

        std::ofstream master_output(prefix + "data-active-" + Utilities::int_to_string(dim) + "d" +
                                    color_name + ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
      }
    }
  }
};

#endif /* TPSS_COLORING_H */
