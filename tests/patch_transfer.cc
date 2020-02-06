/*
 * Tests for patch_transfer.h:
 *
 *    (1) DG cell patch (scalar)
 *
 *  Created on: Feb 06, 2020
 *      Author: witte
 *  (based on Tpatch_transfer02.cc, Tpatch_transfer01.cc)
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include "poisson.h"

#include "test_utilities.h"

using namespace dealii;

template<typename T>
class TestPatchTransferDG : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();
  using PoissonProblem           = typename Poisson::ModelProblem<dim, fe_degree>;
  using PatchTransfer            = typename PoissonProblem::LEVEL_MATRIX::transfer_type;


  struct Params
  {
    unsigned int       n_refinements = 0;
    TPSS::PatchVariant patch_variant = TPSS::PatchVariant::cell;
  };


  void
  SetUp() override
  {
    ofs.open("patch_transfer.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  test()
  {
    RT::Parameter rt_parameters;
    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;
    rt_parameters.mesh.n_refinements        = params.n_refinements;

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;

    const auto poisson_problem = std::make_shared<PoissonProblem>(rt_parameters);
    poisson_problem->pcout     = pcout;
    poisson_problem->create_triangulation();
    poisson_problem->distribute_dofs();
    poisson_problem->prepare_linear_system(/*compute_rhs?*/ false);
    poisson_problem->prepare_multigrid();

    // *** patch transfer
    const auto global_level      = poisson_problem->level;
    const auto mf_storage_global = poisson_problem->template build_mf_storage<double>(global_level);
    const auto fdss_storage =
      poisson_problem->template build_patch_storage<double>(global_level, mf_storage_global);
    const auto   patch_transfer = std::make_shared<PatchTransfer>(*fdss_storage);
    const auto & dof_handler    = poisson_problem->dof_handler;

    Vector<double> dof_vector(dof_handler.n_dofs(global_level));
    fill_with_random_values(dof_vector);

    const auto &   partition_data = fdss_storage->get_partition_data();
    const auto     n_subdomains   = partition_data.n_subdomains();
    Vector<double> dst(dof_handler.n_dofs(global_level));
    dst *= 0.;

    *pcout << "Restrict & Prolongate = Identity ...  \n\n";
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);

      auto local_vector = patch_transfer->gather(dof_vector);
      patch_transfer->scatter_add(dst, local_vector);
    }

    Util::compare_vector(dst, dof_vector, *pcout);
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  Params                              params;
};

TYPED_TEST_SUITE_P(TestPatchTransferDG);

TYPED_TEST_P(TestPatchTransferDG, CellPatch)
{
  using Fixture = TestPatchTransferDG<TypeParam>;

  Fixture::params.n_refinements = 2U;
  Fixture::test();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferDG, CellPatch);

using TestParamsLinear = testing::Types<Util::NonTypeParams<2, 1>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferDG, TestParamsLinear);

using TestParamsHigherOrder = testing::Types<Util::NonTypeParams<2, 3>, Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestPatchTransferDG, TestParamsHigherOrder);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("patch_transfer.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
