/*
 * PatchTransfer - Test: cell patch transfer
 *
 *  Created on: Aug 07, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

using namespace dealii;

template<int dim, int fe_degree>
struct Test
{
  Test(ConditionalOStream &       pcout_in,
       const Laplace::Parameter & parameters_in,
       const bool                 print_test_details = false)
    : pcout(pcout_in),
      test_details(print_test_details ? std::cout.rdbuf() : null_stream.rdbuf()),
      parameters(parameters_in),
      laplace_problem(pcout, parameters)
  {
    laplace_problem.setup();
  }

  void
  operator()()
  {
    using PATCH_TRANSFER = typename TPSS::PatchTransfer<dim, fe_degree, fe_degree + 1, 1, double>;

    // *** patch transfer
    const auto global_level      = laplace_problem.global_level;
    const auto mf_storage_global = laplace_problem.template build_mf_storage<double>(global_level);
    const auto fdss_storage =
      laplace_problem.template build_patch_storage<double>(global_level, mf_storage_global);
    const auto   patch_transfer = std::make_shared<PATCH_TRANSFER>(*fdss_storage);
    const auto & dof_handler    = laplace_problem.dof_handler;
    const auto   level          = laplace_problem.global_level;

    Vector<double> dof_vector(dof_handler.n_dofs(level));
    for(unsigned i = 0; i < dof_vector.size(); ++i)
      dof_vector(i) = rand() / RAND_MAX; //(double)i;

    const auto &   partition_data = fdss_storage->get_partition_data();
    const auto     n_subdomains   = partition_data.n_subdomains();
    Vector<double> dst(dof_handler.n_dofs(level));
    dst *= 0.;

    pcout << "Restrict & Prolongate = Identity ...  \n\n";
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);

      auto local_vector = patch_transfer->gather(dof_vector);
      patch_transfer->scatter_add(dst, local_vector);
    }

    dst -= dof_vector;
    print_row(test_details, 15, "error:", dst.l2_norm(), "\n\n");
    Assert(std::abs(dst.l2_norm()) < (1.e-12), ExcMessage("Patch transfer (cell patch) failed!"));
    pcout << " ... success! \n\n";
  }

  ConditionalOStream &                             pcout;
  NullOStream                                      null_stream;
  std::ostream                                     test_details;
  const Laplace::Parameter                         parameters;
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
  parameters.schwarz_smoother_data.patch_variant = TPSS::PatchVariant::cell;
  Test<2, 1>{pcout, parameters, true}();
  Test<2, 4>{pcout, parameters}();
  Test<3, 1>{pcout, parameters}();
  Test<3, 4>{pcout, parameters}();

  return 0;
}
