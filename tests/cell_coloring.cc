// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// check function MGTools::max_level_for_coarse_mesh()

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
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
std::vector<std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>>
get_coloring(const DoFHandler<dim> & dof, const unsigned int level)
{
  std::vector<std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>> coloring(2);

  for(auto & cell : dof.mg_cell_iterators_on_level(level))
    if(cell->is_locally_owned_on_level())
    {
      // This is for cell patches, so each patch is only 1 cell
      std::vector<typename DoFHandler<dim>::level_cell_iterator> patch;
      patch.push_back(cell);

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
      coloring[color].push_back(patch);
    }

  return coloring;
}


template<int dim>
void
test()
{
  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::limit_level_difference_at_vertices,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global(1);

  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(FE_Q<dim>(1));
  dof.distribute_mg_dofs();


  // Returns the coloring
  for(unsigned int level = 0; level < tria.n_global_levels(); ++level)
  {
    std::vector<std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>> coloring =
      get_coloring(dof, level);
    // supress warning
    (void)coloring;
  }

  // Output to screen to verify global coordinates
  //  for (unsigned int level=0; level<tria.n_global_levels(); ++level)
  //  {
  //    std::cout << level <<  ": " << std::endl;

  //    for (auto &cell : dof.mg_cell_iterators_on_level(level))
  //      if (cell->is_locally_owned_on_level())
  //      {
  //        std::cout << cell->id().to_string() << "     ";

  //        Point<dim,unsigned int> cell_int_coords =
  //        get_integer_coords<dim>(cell->id(),tria.n_global_levels()); std::cout << "(" <<
  //        cell_int_coords(0) << ", " << cell_int_coords(1) << ")" << std::endl;
  //      }

  //    std::cout << std::endl;
  //  }

  // Plot things
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof);

  Vector<double> subdomain(tria.n_active_cells());
  for(unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  Vector<double> xcoord(tria.n_active_cells());
  Vector<double> ycoord(tria.n_active_cells());
  Vector<double> coord_sum(tria.n_active_cells());
  Vector<double> cell_coloring(tria.n_active_cells());
  for(auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned())
    {
      xcoord(cell->active_cell_index()) = get_integer_coords<dim>(cell->id())(0);
      ycoord(cell->active_cell_index()) = get_integer_coords<dim>(cell->id())(1);

      coord_sum(cell->active_cell_index()) =
        xcoord(cell->active_cell_index()) + ycoord(cell->active_cell_index());

      // Output coloring based on the logic
      Point<dim, unsigned int> cell_int_coords = get_integer_coords<dim>(cell->id());
      unsigned int             color           = 0;
      unsigned int             sum             = 0;
      for(unsigned int d = 0; d < dim; ++d)
        sum += cell_int_coords(d);
      if(sum % 2 == 1)
        color = 1;
      cell_coloring(cell->active_cell_index()) = color;
    }
  data_out.add_data_vector(xcoord, "xcoord");
  data_out.add_data_vector(ycoord, "ycoord");
  data_out.add_data_vector(coord_sum, "coord_sum");
  data_out.add_data_vector(cell_coloring, "coloring");

  data_out.build_patches();

  {
    std::ofstream file(
      "data-active-" + Utilities::int_to_string(dim) + "d-" +
      Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 4) + ".vtu");
    data_out.write_vtu(file);
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
      filenames.push_back("data-active-" + Utilities::int_to_string(dim) + "d-" +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::ofstream master_output("data-active-" + Utilities::int_to_string(dim) + "d.pvtu");
    data_out.write_pvtu_record(master_output, filenames);
  }
}



int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<2>();
  test<3>();
}
