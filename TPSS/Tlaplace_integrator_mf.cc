
/*
 * test functionality of laplace_integrator.h ...
 * ... matrix-free operator !
 *
 *  Created on: Aug 16, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

template<int dim, int fe_degree>
struct Test
{
  static constexpr unsigned macro_size = VectorizedArray<double>::n_array_elements;

  Test(ConditionalOStream & pcout_in, const Laplace::Parameter & parameter)
    : laplace_problem(pcout_in, parameter)
  {
    laplace_problem.setup();
    laplace_problem.assemble_matrix();
  }

  void
  operator()()
  {
    std::cout << "Comparing sparse matrix and matrix-free operator ... \n\n";
    const auto & sparse_matrix = laplace_problem.sparse_matrix;
    const auto & mf_operator   = laplace_problem.system_matrix;

    auto     temp{laplace_problem.solution};
    unsigned i = 0;
    for(auto it = temp.begin(); it != temp.end(); ++it, ++i)
      *it = (double)rand() / RAND_MAX; //(double)i;
    const auto           src{temp};
    const Vector<double> src_copy(src.begin(), src.end());

    auto dst{src};
    mf_operator.vmult(dst, src);

    auto dst_sparse{src_copy};
    sparse_matrix.vmult(dst_sparse, src_copy);

    Vector<double> dst_copy(dst.begin(), dst.end());
    dst_copy -= dst_sparse;
    Assert(std::abs(dst_copy.l2_norm()) < (1.e-12 * dst.l2_norm()),
           ExcMessage("Sparse matrix and matrix-free operator don't match"));
    std::cout << " ... success!\n\n";
  }

  Laplace::MatrixOperator<dim, fe_degree, double> laplace_problem;
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
  Test<2, 1>{pcout, parameters}();
  Test<2, 4>{pcout, parameters}();
  Test<3, 1>{pcout, parameters}();
  Test<3, 2>{pcout, parameters}();

  return 0;
}
