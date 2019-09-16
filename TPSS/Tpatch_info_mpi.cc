
/*
 * Test the (colored) distribution of cell patches.
 * TODO: other variants than user-colored MCP
 *
 *  Created on: Sep 12, 2019
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

using namespace dealii;

struct TestParameter
{
  unsigned              n_refinements    = 1;
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::multiplicative;
  bool                  manual_coloring  = true;
};



template<int dim, int fe_degree>
void
test(const TestParameter & prm)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    Triangulation<dim>::limit_level_difference_at_vertices,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  DoFHandler<dim>                     dof_handler;
  std::shared_ptr<FiniteElement<dim>> fe;

  //: create triangulation
  {
    const double left = 0.0, right = 1.0;
    GridGenerator::subdivided_hyper_cube(triangulation, 1, left, right);
    triangulation.refine_global(prm.n_refinements);
  }

  //: distribute dofs
  {
    fe = std::make_shared<FE_DGQ<dim>>(fe_degree);
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();
  }

  pcout << "Initializing patch info ...  \n\n";
  TPSS::PatchInfo<dim>                          patch_info;
  typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
  additional_data.patch_variant    = prm.patch_variant;
  additional_data.smoother_variant = prm.smoother_variant;
  const auto level                 = triangulation.n_levels() - 1;
  additional_data.level            = level;
  additional_data.print_details    = true;
  Laplace::ManualColoringCP<dim> user_coloring_cp;
  additional_data.manual_coloring_func_cp = std::ref(user_coloring_cp);
  patch_info.initialize(&dof_handler, additional_data);
  pcout << " ... done!\n\n";

  pcout << "Testing cell patch distribution ...  \n\n";
  std::vector<int> cindices;
  const auto       cell_iterators = patch_info.get_internal_data()->cell_iterators;
  for(const auto & cell : cell_iterators)
    cindices.emplace_back(cell->index());
  const auto n_cell_patches = cindices.size();
  std::sort(cindices.begin(), cindices.end());
  std::unique(cindices.begin(), cindices.end());
  AssertThrow(n_cell_patches == cindices.size(), ExcMessage("Duplicates!"));
  // const unsigned int n_active_cells_all = triangulation.n_cells(level); // fails on level 3
  const unsigned int n_active_cells_all = triangulation.n_global_active_cells();
  const unsigned int n_cell_patches_all = Utilities::MPI::sum(n_cell_patches, MPI_COMM_WORLD);
  std::ostringstream oss;
  oss << n_cell_patches_all << " != " << n_active_cells_all;
  AssertThrow(n_cell_patches_all == n_active_cells_all, ExcMessage(oss.str()));
  pcout << " ... success!\n\n";
}

int
main(int argc, char * argv[])
{
  constexpr int                    max_threads = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  TestParameter prms;
  for(unsigned n_refinements : {1, 3, 4})
  {
    prms.n_refinements = n_refinements;
    // if(argc > 1)
    // 	prms.n_refinements = std::atoi(argv[1]);
    test<2, 1>(prms);
    test<3, 1>(prms);
  }

  return 0;
}
