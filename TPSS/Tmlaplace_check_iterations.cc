
/*
 * Multi - Laplace (check iteration counts)
 *
 *  Created on: Jun 24, 2019
 *      Author: witte
 */

#include "Plaplace_fdss.h"
#include "ct_parameter.h"
#include "mlaplace_problem.h"

using namespace dealii;
using namespace MultiLaplace;



struct TestParameter
{
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;
  unsigned int          coarse_level     = 1;
  unsigned int          n_refines        = coarse_level + 2;
  bool                  print_details    = false;
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const TestParameter test_parameters)
{
  using MultiLaplaceOperator =
    typename MultiLaplace::MatrixOperator<dim, fe_degree, dim, value_type>;

  const bool         is_mpi_process0 = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  const bool         print_details   = is_mpi_process0 && test_parameters.print_details;
  ConditionalOStream pcout_internal(std::cout, print_details);
  Laplace::Parameter parameters;
  parameters.coarse_level                           = test_parameters.coarse_level;
  parameters.n_refines                              = test_parameters.n_refines;
  parameters.schwarz_smoother_data.patch_variant    = test_parameters.patch_variant;
  parameters.schwarz_smoother_data.smoother_variant = test_parameters.smoother_variant;
  parameters.solver_max_iterations                  = 200;
  const auto solver_variant =
    GlobParam::lookup_solver_variant_impl(test_parameters.smoother_variant);
  parameters.solver_variant       = solver_variant;
  parameters.precondition_variant = Laplace::Parameter::PreconditionVariant::MG;

  MultiLaplaceOperator mlaplace_problem(pcout_internal, parameters);
  mlaplace_problem.run();
  auto & laplace_problem = mlaplace_problem.laplace_problem;
  laplace_problem.run();

  ConditionalOStream pcout(std::cout, is_mpi_process0);
  const double       n_steps_block  = mlaplace_problem.pp_data[0].n_iterations_system.back();
  const double       n_steps_scalar = laplace_problem.pp_data.n_iterations_system.back();
  // pcout << "block: " << n_steps_block << " versus scalar: " << n_steps_scalar << "\n\n";
  std::ostringstream oss;
  oss << "number of iterations of the block-system (" << n_steps_block
      << ") is HIGHER than the number of iterations of the scalar system (" << n_steps_block << ")";
  AssertThrow(n_steps_block < (n_steps_scalar + 1.e-1), ExcMessage(oss.str()));
}

int
main(int argc, char * argv[])
{
  deallog.depth_console(0);

  // *** init TBB and MPI
  const unsigned int max_threads = 1; // serial
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  TestParameter test_ACP;
  test<2, 2>(test_ACP);
  test<2, 5>(test_ACP);
  test<3, 2>(test_ACP);
  test<3, 3>(test_ACP);

  TestParameter test_MCP;
  test_MCP.smoother_variant = TPSS::SmootherVariant::multiplicative;
  test<2, 2>(test_MCP);
  test<2, 5>(test_MCP);
  test<3, 2>(test_MCP);
  test<3, 3>(test_MCP);

  TestParameter test_AVP;
  test_AVP.patch_variant = TPSS::PatchVariant::vertex;
  test<2, 2>(test_AVP);
  test<2, 5>(test_AVP);
  test<3, 2>(test_AVP);
  test<3, 3>(test_AVP);

  TestParameter test_MVP;
  test_MVP.patch_variant    = TPSS::PatchVariant::vertex;
  test_MVP.smoother_variant = TPSS::SmootherVariant::multiplicative;
  test<2, 2>(test_MVP);
  test<2, 5>(test_MVP);
  test<3, 2>(test_MVP);
  test<3, 3>(test_MVP);

  return 0;
}
