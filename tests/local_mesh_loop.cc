/*
 * Tests for ...TODO...
 *
 *  Created on: Feb 12, 2021
 *      Author: witte
 */

#include <gtest/gtest.h>

#include "poisson_problem.h"

#include "test_utilities.h"

using namespace dealii;



using TestParamsLinear2D    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic2D = testing::Types<Util::NonTypeParams<2, 2>>;



////////// TestLocalMeshLoopBase



template<typename T>
class TestLocalMeshLoopBase : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();

  virtual void
  SetUp() override
  {
    ofs.open("local_mesh_loop.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    /// base mesh is a vertex patch
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_repetitions    = 2;
    rt_parameters.mesh.n_refinements    = 0U;
  }

  virtual void
  TearDown() override
  {
    ofs.close();
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
};



////////// TestLocalMeshLoop



template<typename T>
class TestLocalMeshLoop : public TestLocalMeshLoopBase<T>
{
protected:
  using Base = TestLocalMeshLoopBase<T>;

  static constexpr int dim       = Base::dim;
  static constexpr int fe_degree = Base::fe_degree;

  using Base::pcout;
  using Base::rt_parameters;

  virtual void
  SetUp() override final
  {
    Base::SetUp();
  }

  virtual void
  TearDown() override final
  {
    Base::TearDown();
  }

  template<TPSS::DoFLayout dof_layout>
  void
  test()
  {
    using PoissonProblem = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;
    using PatchTransfer  = typename TPSS::PatchTransfer<dim, double>;

    const auto poisson_problem = std::make_shared<PoissonProblem>(rt_parameters);
    poisson_problem->pcout     = pcout;
    poisson_problem->create_triangulation();
    poisson_problem->distribute_dofs();

    // *** patch transfer
    const auto global_level = poisson_problem->level;
    const auto mf_storage   = poisson_problem->template build_mf_storage<double>(global_level);
    const auto subdomain_handler =
      poisson_problem->template build_patch_storage<double>(global_level, mf_storage);
    const auto   patch_transfer = std::make_shared<PatchTransfer>(*subdomain_handler);
    const auto & dof_handler    = poisson_problem->dof_handler;

    const auto & partition_data = subdomain_handler->get_partition_data();
    const auto   n_subdomains   = partition_data.n_subdomains();

    const auto & patch_dof_worker = patch_transfer->get_patch_dof_worker();

    for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
    {
      *pcout << "patch: " << patch_index << std::endl;
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
      {
        *pcout << "lane: " << lane << std::endl;
        const auto & cell_collection = patch_dof_worker.get_cell_collection(patch_index, lane);
        for(const auto & cell : cell_collection)
          *pcout << cell->index() << " ";
        *pcout << std::endl;
      }
    }
  }
};

TYPED_TEST_SUITE_P(TestLocalMeshLoop);

TYPED_TEST_P(TestLocalMeshLoop, VertexPatchDGQ)
{
  using Fixture = TestLocalMeshLoop<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::template test<TPSS::DoFLayout::DGQ>();

  Fixture::rt_parameters.mesh.n_refinements = 1U;
  Fixture::template test<TPSS::DoFLayout::DGQ>();
}

REGISTER_TYPED_TEST_SUITE_P(TestLocalMeshLoop, VertexPatchDGQ);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLocalMeshLoop, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestLocalMeshLoop, TestParamsQuadratic2D);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("local_mesh_loop.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
