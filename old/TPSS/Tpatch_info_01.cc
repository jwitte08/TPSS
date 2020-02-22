
/*
 * PatchInfo - Test: cell patch identification
 *
 *  Created on: Aug 07, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

template<int dim, int fe_degree>
struct Test
{
  Test(ConditionalOStream & pcout_in, const Laplace::Parameter & parameter)
    : pcout(pcout_in), laplace_problem(pcout, parameter)
  {
    laplace_problem.setup();
  }

  void
  operator()(const TPSS::PatchVariant patch_variant)
  {
    // *** set references
    const auto & triangulation = laplace_problem.triangulation;
    const auto & dof_handler   = laplace_problem.dof_handler;

    pcout << "Initializing patch info ...  \n\n";
    TPSS::PatchInfo<dim>                          patch_info;
    typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
    additional_data.patch_variant    = patch_variant;
    additional_data.smoother_variant = TPSS::SmootherVariant::additive;
    const auto level                 = triangulation.n_levels() - 1;
    additional_data.level            = level;
    pcout << " ... done!\n\n";

    pcout << "Testing cell patch identification ...  \n\n";
    patch_info.initialize(&dof_handler, additional_data);
    std::vector<int> cindices;
    const auto       cell_iterators = patch_info.get_internal_data()->cell_iterators;
    for(const auto & cell : cell_iterators)
      cindices.emplace_back(cell->index());
    const auto n_cell_patches = cindices.size();
    std::sort(cindices.begin(), cindices.end());
    std::unique(cindices.begin(), cindices.end());
    AssertThrow(n_cell_patches == cindices.size(), ExcMessage("Duplicates!"));
    AssertThrow(n_cell_patches == triangulation.n_cells(level), ExcMessage("Missing cells!"));
    pcout << " ... success!\n\n";
  }

  ConditionalOStream &                             pcout;
  typename Laplace::MatrixOperator<dim, fe_degree> laplace_problem;
};

int
main(int argc, char * argv[])
{
  using namespace dealii;

  // *** init MPI (1 proc only!)
  constexpr int                    max_threads = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // *** test parameters
  Laplace::Parameter parameters;
  parameters.n_refines = 2;

  // *** run tests
  Test<2, 1> test_2d(pcout, parameters);
  test_2d(TPSS::PatchVariant::cell);
  Test<3, 1> test_3d(pcout, parameters);
  test_3d(TPSS::PatchVariant::cell);

  return 0;
}
