
/*
 * Multi - Laplace (test FD integrators)
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
  unsigned int          n_refines        = coarse_level + 1;
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const TestParameter test_parameters)
{
  using MultiLaplaceOperator =
    typename MultiLaplace::MatrixOperator<dim, fe_degree, dim, value_type>;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  Laplace::Parameter parameters;
  parameters.coarse_level                           = test_parameters.coarse_level;
  parameters.n_refines                              = test_parameters.n_refines;
  parameters.schwarz_smoother_data.patch_variant    = test_parameters.patch_variant;
  parameters.schwarz_smoother_data.smoother_variant = test_parameters.smoother_variant;

  MultiLaplaceOperator mlaplace_problem(pcout, parameters);
  mlaplace_problem.create_triangulation_finest();
  mlaplace_problem.distribute_dofs();
  mlaplace_problem.prepare_system();
  mlaplace_problem.prepare_preconditioner_mg();
  const auto &       laplace_problem = mlaplace_problem.laplace_problem;
  const unsigned int level           = laplace_problem.global_level;

  const auto                            b = mlaplace_problem.system_rhs;
  typename MultiLaplaceOperator::VECTOR tmp1{b}, tmp2{b};
  // std::cout << "print b:\n\n";
  // b.print(std::cout);

  // *** test scalar Schwarz preconditioner
  auto &       tmp1_block0 = tmp1.block(0);
  const auto & b_block0    = b.block(0);
  const auto   Pl_scalar   = laplace_problem.mg_schwarz_precondition[level];
  Pl_scalar->vmult(tmp1_block0, b_block0);
  // std::cout << "print tmp1(block0):\n\n";
  // tmp1.print(std::cout);

  // *** test block Schwarz preconditioner
  const auto Pl_block = mlaplace_problem.mg_schwarz_precondition[level];
  Pl_block->vmult(tmp2, b);
  // std::cout << "print tmp2:\n\n";
  // tmp2.print(std::cout);

  // *** comparison
  const unsigned int n_components = b.n_blocks();
  for(unsigned comp = 0; comp < n_components; ++comp)
  {
    auto diff = tmp2.block(comp);
    diff -= tmp1.block(0);
    std::ostringstream ostrstream;
    ostrstream << "||tmp2.block(" << comp << ") - tmp1.block(0)||_l2 exceeds the error bound";
    AssertThrow(std::abs(diff.l2_norm()) < (1.e-12 * tmp1.block(0).l2_norm()),
                ExcMessage(ostrstream.str()));
  }
}

int
main(int argc, char * argv[])
{
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
