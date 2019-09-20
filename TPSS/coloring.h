
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
struct RedBlackColoring
{
  using CellIterator   = typename TPSS::PatchInfo<dim>::CellIterator;
  using PatchIterator  = typename TPSS::PatchInfo<dim>::PatchIterator;
  using AdditionalData = typename TPSS::PatchInfo<dim>::AdditionalData;

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

      colored_patches[color].emplace_back(patch);
    }
    return colored_patches;
  }
};
