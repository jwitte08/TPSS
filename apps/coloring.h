#ifndef APPS_COLORING_H
#define APPS_COLORING_H

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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "solvers_and_preconditioners/TPSS/patch_info.h"

#include "mesh.h"



namespace Impl
{
template<int dim>
void
visualize_coloring(
  const dealii::DoFHandler<dim> &                                                dof_handler_in,
  const std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>> & colored_iterators,
  const std::string                                                              prefix)
{
  for(unsigned color = 0; color < colored_iterators.size(); ++color)
  {
    // collect output data
    const auto &            tria         = dof_handler_in.get_triangulation();
    const unsigned int      global_level = tria.n_global_levels() - 1;
    dealii::DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(FE_DGQ<dim>(0)); // require one (dof) value per cell !!!
    dof_handler.distribute_mg_dofs();
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    // print the subdomain_ids
    dealii::Vector<double> subdomain(tria.n_active_cells());
    for(unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain", DataOut<dim>::type_cell_data);

    // initialize the ghosted dof vector
    dealii::LinearAlgebra::distributed::Vector<double> marker;
    IndexSet                                           locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                          global_level,
                                                          locally_relevant_dofs);
    marker.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    marker                        = 0;
    const auto & patch_collection = colored_iterators[color];
    for(const auto & patch : patch_collection)
      for(const auto & cell : (*patch))
      {
        const auto active_cell = typename dealii::DoFHandler<dim>::active_cell_iterator{
          &tria, cell->level(), cell->index(), &dof_handler};
        std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);
        active_cell->get_dof_indices(dof_indices);
        const auto active_cell_index = dof_indices.front();
        marker(active_cell_index)    = 1;
      }
    marker.compress(dealii::VectorOperation::add);
    data_out.add_data_vector(marker, "coloring", dealii::DataOut<dim>::type_dof_data);

    // write paraview files
    std::string color_name = "_color" + dealii::Utilities::int_to_string(color, 2);
    data_out.build_patches();
    {
      std::ofstream file(
        prefix + "data-active-" + dealii::Utilities::int_to_string(dim) + "d-" +
        dealii::Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                                         4) +
        color_name + ".vtu");
      data_out.write_vtu(file);
    }
    if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::vector<std::string> filenames;
      for(unsigned int i = 0; i < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
        filenames.push_back(prefix + "data-active-" + dealii::Utilities::int_to_string(dim) + "d-" +
                            dealii::Utilities::int_to_string(i, 4) + color_name + ".vtu");

      std::ofstream master_output(prefix + "data-active-" + dealii::Utilities::int_to_string(dim) +
                                  "d" + color_name + ".pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  }
}
} // namespace Impl



template<int dim>
struct IntegerCoordinate
{
  IntegerCoordinate() = delete;

  IntegerCoordinate(const MeshParameter & mesh_prms) : integer_coordinate_root(mesh_prms)
  {
  }

  Point<dim, unsigned int>
  operator()(const dealii::CellId cell_id) const
  {
    // Extract child indices from CellId
    std::vector<unsigned int> child_indices;
    std::string               cell_id_str  = cell_id.to_string();
    const auto                pos_of_colon = cell_id_str.find(':');
    Assert(pos_of_colon > 0, ExcInternalError());
    while(cell_id_str.size() > (pos_of_colon + 1))
    {
      child_indices.insert(child_indices.begin(), Utilities::string_to_int(&(cell_id_str.back())));
      cell_id_str.pop_back();
    }

    // Compute integer coordinates of root cells
    dealii::Point<dim, unsigned int> global_coord;
    const unsigned int               root_cell_index = cell_id.to_binary<dim>()[0];
    global_coord                                     = integer_coordinate_root(root_cell_index);

    // Compute global integer coordinates by traversing through children cells.  We assume the
    // refinement is uniform, namely dividing each dimension by two.
    for(auto c : child_indices)
    {
      global_coord *= 2;
      Point<dim, unsigned int> local_coord;
      {
        const std::bitset<dim> bit_indices(c);
        for(unsigned int d = 0; d < dim; ++d)
          local_coord(d) = bit_indices[d];
      }
      global_coord += local_coord;
    }

    return global_coord;
  }

  const IntegerCoordinateRoot<dim> integer_coordinate_root;
};


template<int dim>
struct ColoringBase
{
  using CellIterator   = typename TPSS::PatchInfo<dim>::CellIterator;
  using PatchIterator  = typename TPSS::PatchInfo<dim>::PatchIterator;
  using AdditionalData = typename TPSS::PatchInfo<dim>::AdditionalData;

  IntegerCoordinate<dim> get_integer_coordinate;

  ColoringBase(const MeshParameter & mesh_prms) : get_integer_coordinate(mesh_prms)
  {
  }

  virtual ~ColoringBase() = default;

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



  virtual std::vector<std::vector<PatchIterator>>
  cell_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const = 0;



  virtual std::vector<std::vector<PatchIterator>>
  vertex_patch_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const = 0;



  static void
  visualize_coloring(const dealii::DoFHandler<dim> & dof_handler_in,
                     const std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>> &
                                       colored_iterators,
                     const std::string prefix)
  {
    Impl::visualize_coloring<dim>(dof_handler_in, colored_iterators, prefix);
  }
};



/**
 * Red-black coloring for a uniformly refined hypercube.
 *
 * (1) Standard red-black coloring for cells.
 *
 * (2) Shifted red-black coloring for vertex patches: We choose a "starting"
 * vertex patch in one corner of the hypercube. Each vertex patch is a 2x2x...x2
 * mesh by itself. Therefore, we have 2^dim possibilities to shift this vertex
 * patch by zero or one (in positive direction) in any dimension. Interpreting a
 * vertex patch as a macro cell, for each shifted starting vertex patch we have
 * a standard red-black coloring. In total this results in (2*2^dim) colors.
 */
template<int dim>
struct RedBlackColoring : public ColoringBase<dim>
{
  using Base = ColoringBase<dim>;
  using typename Base::AdditionalData;
  using typename Base::CellIterator;
  using typename Base::PatchIterator;

  RedBlackColoring(const MeshParameter & mesh_prms) : ColoringBase<dim>(mesh_prms)
  {
  }



  std::vector<std::vector<PatchIterator>>
  cell_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const override
  {
    std::vector<std::vector<PatchIterator>> colored_patches;
    colored_patches.resize(2);

    for(auto patch = patches.cbegin(); patch != patches.cend(); ++patch)
    {
      AssertDimension(patch->size(), 1);
      const auto & cell = patch->front();

      // Get integer coordinates
      dealii::Point<dim, unsigned int> cell_int_coords = this->get_integer_coordinate(cell->id());

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
  vertex_patch_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const override
  {
    std::vector<std::vector<PatchIterator>> colored_patches;
    const unsigned int                      n_colors = 2 * (1 << dim);
    colored_patches.resize(n_colors);
    for(PatchIterator patch = patches.cbegin(); patch != patches.cend(); ++patch)
    {
      // determine shift of the lower left cell
      const auto & shift = [this](const PatchIterator & patch, const unsigned direction) {
        AssertIndexRange(direction, dim);
        const unsigned stride                 = 1 << direction;
        const auto     cell_left              = (*patch)[0];
        const auto     cell_right             = (*patch)[stride];
        const auto     coord_left             = this->get_integer_coordinate(cell_left->id());
        const auto     coord_right            = this->get_integer_coordinate(cell_right->id());
        const auto     patch_coord_from_left  = coord_left(direction) / 2;
        const auto     patch_coord_from_right = coord_right(direction) / 2;
        return patch_coord_from_left != patch_coord_from_right;
      };
      std::bitset<dim> shift_mask;
      for(unsigned int d = 0; d < dim; ++d)
        shift_mask[d] = (shift(patch, d));
      AssertIndexRange(shift_mask.to_ulong(), 1 << dim);

      // (1 << dim) layers of red-black colorings (layer is determined by the shift)
      const auto                             cell        = patch->front();
      const dealii::Point<dim, unsigned int> coordinates = this->get_integer_coordinate(cell->id());
      unsigned int                           sum         = 0;
      for(unsigned int d = 0; d < dim; ++d)
        sum += (coordinates(d) + static_cast<unsigned>(shift_mask[d])) / 2;
      const unsigned red_or_black = sum % 2;
      const unsigned color        = 2 * shift_mask.to_ulong() + red_or_black;
      AssertIndexRange(color, (1 << dim) * 2);

      colored_patches[color].emplace_back(patch);
    }
    return colored_patches;
  }



  using Base::visualize_coloring;
};



/**
 * The coloring used for H^1-conforming finite elements on a uniformly refined
 * hypercube.
 *
 * (1) Coloring for cells: TODO (if needed)
 *
 * (2) Coloring for vertex patches: We choose a "starting" vertex patch in one
 * corner of the hypercube. Each vertex patch is a 2x2x...x2 mesh by
 * itself. Therefore, we have 2^dim possibilities to shift this vertex patch by
 * zero or one (in positive direction) in any dimension. Starting with any of
 * those shifted vertex patches, there is an obvious tiling of vertex patches
 * avoiding any space in-between two adjacent. In total this results in 2^dim
 * colors.
 */
template<int dim>
struct TiledColoring : public ColoringBase<dim>
{
  using Base = ColoringBase<dim>;
  using typename Base::AdditionalData;
  using typename Base::CellIterator;
  using typename Base::PatchIterator;

  TiledColoring(const MeshParameter & mesh_prms) : ColoringBase<dim>(mesh_prms)
  {
  }

  std::vector<std::vector<PatchIterator>>
  vertex_patch_coloring_impl(const std::vector<std::vector<CellIterator>> & patches) const override
  {
    std::vector<std::vector<PatchIterator>> colored_patches;
    const unsigned int                      n_colors = (1 << dim);
    colored_patches.resize(n_colors);
    for(PatchIterator patch = patches.cbegin(); patch != patches.cend(); ++patch)
    {
      // determine shift of the lower left cell
      const auto & shift = [this](const PatchIterator & patch, const unsigned direction) {
        AssertIndexRange(direction, dim);
        const unsigned stride                 = 1 << direction;
        const auto     cell_left              = (*patch)[0];
        const auto     cell_right             = (*patch)[stride];
        const auto     coord_left             = this->get_integer_coordinate(cell_left->id());
        const auto     coord_right            = this->get_integer_coordinate(cell_right->id());
        const auto     patch_coord_from_left  = coord_left(direction) / 2;
        const auto     patch_coord_from_right = coord_right(direction) / 2;
        return patch_coord_from_left != patch_coord_from_right;
      };

      // (1 << dim) layers of non-overlapping vertex patches (layer is
      // determined by the shift)
      std::bitset<dim> shift_mask;
      for(unsigned int d = 0; d < dim; ++d)
        shift_mask[d] = (shift(patch, d));
      const unsigned color = shift_mask.to_ulong();

      colored_patches[color].emplace_back(patch);
    }
    return colored_patches;
  }



  std::vector<std::vector<PatchIterator>>
  cell_coloring_impl(const std::vector<std::vector<CellIterator>> & /*patches*/) const override
  {
    AssertThrow(false, ExcMessage("Is not implemented."));
  }

  using Base::visualize_coloring;
};



#endif /* APPS_COLORING_H */
