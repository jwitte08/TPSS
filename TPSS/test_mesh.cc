
/*
 * test different mesh variants
 *
 * distorted cube: randomly distorts the interior vertices
 *
 *  Created on: Oct 05, 2019
 *      Author: witte
 */

#include "mesh.h"

using namespace dealii;

template<int dim>
void
test_distorted_cube(Triangulation<dim> & tria, const MeshParameter & prm_in)
{
  const bool         is_rank_0 = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  ConditionalOStream pcout(std::cout, is_rank_0);
  MeshParameter      prm = prm_in;

  prm.n_refinements = 0; // only root mesh !
  pcout << create_distorted_cube(tria, prm);
  pcout << std::endl;
  // print

  prm.n_refinements = prm_in.n_refinements;
  pcout << create_distorted_cube(tria, prm);
  // print again
}

int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MeshParameter prm;
  prm.n_refinements = 2;
  prm.n_repetitions = 2;
  prm.distortion    = 0.1;
  if(argc > 1)
    prm.n_refinements = std::atoi(argv[1]);
  if(argc > 2)
    prm.n_repetitions = std::atoi(argv[2]);
  if(argc > 3)
    prm.distortion = std::atof(argv[3]);

  constexpr int                             dim = 2;
  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::limit_level_difference_at_vertices,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  test_distorted_cube<dim>(tria, prm);
  return 0;
}
