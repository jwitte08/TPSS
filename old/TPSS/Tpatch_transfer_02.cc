/*
 * PatchTransfer - Test: patch transfer
 *
 *  Created on: May 20, 2019
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "mlaplace_problem.h"

using namespace dealii;

template<typename Number>
void
write_random_input(const ArrayView<Number> array)
{
  std::generate(array.begin(), array.end(), []() {
    Number val;
    val = (double)rand() / RAND_MAX;
    return val;
  });
}

template<int dim, int fe_degree>
struct Test
{
  using TRANSFER = typename TPSS::PatchTransferBlock<dim, fe_degree, fe_degree + 1, double>;
  static constexpr unsigned int n_static_components = dim;

  Test(ConditionalOStream & pcout_in, const Laplace::Parameter & parameters_in)
    : pcout(pcout_in), parameters(parameters_in), mlaplace_problem(pcout, parameters)
  {
    mlaplace_problem.distribute_dofs();
    mlaplace_problem.prepare_system();
  }

  void
  operator()()
  {
    // *** initialize patch transfer
    const auto level      = mlaplace_problem.laplace_problem.global_level;
    const auto mf_storage = mlaplace_problem.template build_mf_storage<double>(level);
    const auto patch_storage =
      mlaplace_problem.template build_patch_storage<double>(level, mf_storage);
    const auto patch_transfer = std::make_shared<TRANSFER>(*patch_storage);

    // *** check if we have the same number of DoFs in each component
    const auto & dof_handlers = mlaplace_problem.dof_handlers;
    const bool   same_number_of_dofs_per_level =
      std::adjacent_find(dof_handlers.cbegin(),
                         dof_handlers.cend(),
                         [level](const auto & dh1, const auto & dh2) {
                           return dh1->n_dofs(level) != dh2->n_dofs(level);
                         });
    Assert(same_number_of_dofs_per_level, ExcNotImplemented());

    // *** generate random input
    const unsigned int n_components = mf_storage->n_components();
    AssertDimension(n_components, n_static_components);
    LinearAlgebra::distributed::BlockVector<double> random_input(n_components);
    for(unsigned int b = 0; b < n_components; ++b)
    {
      auto & bvector = random_input.block(b);
      mf_storage->initialize_dof_vector(bvector, b);
      random_input.collect_sizes();
      const ArrayView<double> bview = make_array_view(bvector.begin(), bvector.end());
      write_random_input(bview);
      // bvector.print(std::cout);
      // random_input.print(std::cout);
    }

    pcout << "TEST: restriction and prolongation\n";
    auto                                   dst{random_input}; // first time !!!
    AlignedVector<VectorizedArray<double>> tmp_local;
    const auto &                           partition_data = patch_storage->get_partition_data();
    const auto                             n_subdomains   = partition_data.n_subdomains();
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);
      patch_transfer->reinit_local_vector(tmp_local);
      tmp_local.fill(make_vectorized_array<double>(0.));
      patch_transfer->gather_add(tmp_local, random_input); // restrict
      patch_transfer->scatter_add(dst, tmp_local);         // prolongate
    }                                                      // second time !!!

    // *** check if restrict + prolongate results in the identity
    dst -= random_input;
    dst -= random_input; // added 2 times random_input to dst !!!
    print_row(pcout, 15, "l2 error:", dst.l2_norm(), "\n\n");
    AssertThrow(std::abs(dst.l2_norm()) < (1.e-12), ExcMessage("Cell patch transfer failed."));
  }

  ConditionalOStream &                                                       pcout;
  const Laplace::Parameter                                                   parameters;
  typename MultiLaplace::MatrixOperator<dim, fe_degree, n_static_components> mlaplace_problem;
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
  Test<2, 1>{pcout, parameters}();
  Test<2, 4>{pcout, parameters}();
  Test<3, 1>{pcout, parameters}();
  Test<3, 4>{pcout, parameters}();

  return 0;
}
