/*
 * Tests for patch_transfer.h:
 *
 *    (1) DG cell patch (scalar)
 *    (2) DG vertex patch (scalar)
 *    (3) DG cell patch (block)
 *    (4) DG vertex patch (block)
 *
 *  Created on: Feb 06, 2020
 *      Author: witte
 *  (based on Tpatch_transfer02.cc, Tpatch_transfer01.cc)
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include "linelasticity_problem.h"
#include "poisson_problem.h"

#include "test_utilities.h"

using namespace dealii;

template<typename T>
class TestPatchTransferDG : public testing::Test
{
protected:
  static constexpr int  dim        = T::template value<0>();
  static constexpr int  fe_degree  = T::template value<1>();
  static constexpr auto dof_layout = TPSS::DoFLayout::DGQ;
  using PoissonProblem             = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;
  using PatchTransfer              = typename PoissonProblem::LEVEL_MATRIX::transfer_type;


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
    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 3;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  test()
  {
    rt_parameters.mesh.n_refinements = params.n_refinements;

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;

    const auto poisson_problem = std::make_shared<PoissonProblem>(rt_parameters);
    poisson_problem->pcout     = pcout;
    poisson_problem->create_triangulation();
    poisson_problem->distribute_dofs();

    // *** patch transfer
    const auto global_level      = poisson_problem->level;
    const auto mf_storage_global = poisson_problem->template build_mf_storage<double>(global_level);
    const auto fdss_storage =
      poisson_problem->template build_patch_storage<double>(global_level, mf_storage_global);
    const auto   patch_transfer = std::make_shared<PatchTransfer>(*fdss_storage);
    const auto & dof_handler    = poisson_problem->dof_handler;

    LinearAlgebra::distributed::Vector<double> dof_vector(dof_handler.n_dofs(global_level));
    fill_with_random_values(dof_vector);

    const auto &                               partition_data = fdss_storage->get_partition_data();
    const auto                                 n_subdomains   = partition_data.n_subdomains();
    LinearAlgebra::distributed::Vector<double> dst(dof_handler.n_dofs(global_level));
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
  RT::Parameter                       rt_parameters;
};

TYPED_TEST_SUITE_P(TestPatchTransferDG);

TYPED_TEST_P(TestPatchTransferDG, CellPatch)
{
  using Fixture = TestPatchTransferDG<TypeParam>;

  Fixture::params.n_refinements = 2U;
  Fixture::test();
}

TYPED_TEST_P(TestPatchTransferDG, VertexPatch)
{
  using Fixture = TestPatchTransferDG<TypeParam>;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::params.n_refinements                = 0U;
  Fixture::params.patch_variant                = TPSS::PatchVariant::vertex;
  Fixture::test();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferDG, CellPatch, VertexPatch);

using TestParamsLinear2D = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsHigherOrder2D =
  testing::Types<Util::NonTypeParams<2, 3>, Util::NonTypeParams<2, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferDG, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestPatchTransferDG, TestParamsHigherOrder2D);

using TestParamsLinear3D = testing::Types<Util::NonTypeParams<3, 1>>;
using TestParamsHigherOrder3D =
  testing::Types<Util::NonTypeParams<3, 3>, Util::NonTypeParams<3, 4>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransferDG, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D, TestPatchTransferDG, TestParamsHigherOrder3D);



template<typename T>
class TestPatchTransferBlockDG : public testing::Test
{
protected:
  static constexpr int          dim                 = T::template value<0>();
  static constexpr int          fe_degree           = T::template value<1>();
  static constexpr unsigned int n_static_components = dim;
  using ElasticityProblem = typename LinElasticity::ModelProblem<dim, fe_degree>;
  using PatchTransfer     = typename ElasticityProblem::LEVEL_MATRIX::transfer_type;


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
    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  test()
  {
    rt_parameters.mesh.n_refinements = params.n_refinements;

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;

    const auto elasticity_problem = std::make_shared<ElasticityProblem>(*pcout, rt_parameters);
    elasticity_problem->create_triangulation();
    elasticity_problem->distribute_dofs();

    const auto level      = elasticity_problem->level;
    const auto mf_storage = elasticity_problem->template build_mf_storage<double>(level);
    const auto patch_storage =
      elasticity_problem->template build_patch_storage<double>(level, mf_storage);
    const auto patch_transfer = std::make_shared<PatchTransfer>(*patch_storage);

    // *** check if we have the same number of DoFs in each component
    const auto & dof_handlers = elasticity_problem->dof_handlers;
    const bool   same_number_of_dofs_per_level =
      std::adjacent_find(dof_handlers.cbegin(),
                         dof_handlers.cend(),
                         [level](const auto & dh1, const auto & dh2) {
                           return dh1->n_dofs(level) != dh2->n_dofs(level);
                         });
    ASSERT_TRUE(same_number_of_dofs_per_level)
      << "Not implemented for components with different numbers of degree of freedoms.";

    // *** generate random input
    const unsigned int n_components = mf_storage->n_components();
    ASSERT_EQ(n_components, n_static_components);
    LinearAlgebra::distributed::BlockVector<double> random_input(n_components);
    for(unsigned int b = 0; b < n_components; ++b)
    {
      auto & bvector = random_input.block(b);
      mf_storage->initialize_dof_vector(bvector, b);
      random_input.collect_sizes();
      const ArrayView<double> bview = make_array_view(bvector.begin(), bvector.end());
      fill_with_random_values(bview);
    }

    // *** check if restrict + prolongate equals the identity
    *pcout << "TEST: restriction and prolongation\n";
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
    dst *= 0.5;                                            // added 2 times random_input to dst !!!
    Util::compare_vector(dst, random_input, *pcout);
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  Params                              params;
  RT::Parameter                       rt_parameters;
};

TYPED_TEST_SUITE_P(TestPatchTransferBlockDG);

TYPED_TEST_P(TestPatchTransferBlockDG, CellPatch)
{
  using Fixture = TestPatchTransferBlockDG<TypeParam>;

  Fixture::params.n_refinements = 2U;
  Fixture::test();
}

TYPED_TEST_P(TestPatchTransferBlockDG, VertexPatch)
{
  using Fixture = TestPatchTransferBlockDG<TypeParam>;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::params.n_refinements                = 0U;
  Fixture::params.patch_variant                = TPSS::PatchVariant::vertex;
  Fixture::test();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferBlockDG, CellPatch, VertexPatch);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferBlockDG, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestPatchTransferBlockDG, TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransferBlockDG, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D, TestPatchTransferBlockDG, TestParamsHigherOrder3D);



template<typename T>
class TestPatchTransferVectorValued : public testing::Test
{
protected:
  static constexpr int          dim                 = T::template value<0>();
  static constexpr int          fe_degree           = T::template value<1>();
  static constexpr unsigned int n_static_components = dim;
  using PatchTransfer                               = TPSS::PatchTransfer<dim, double, fe_degree>;


  TestPatchTransferVectorValued()
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      mapping(1)
  {
  }


  void
  SetUp() override
  {
    ofs.open("patch_transfer.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);
    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  check_dg()
  {
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;
    fe                       = std::make_shared<FESystem<dim>>(FE_DGQ<dim>(fe_degree), dim);
    *pcout << fe->get_name() << std::endl;
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mg_level              = level;
      // additional_data.mapping_update_flags =
      // 	(update_gradients | update_JxW_values | update_quadrature_points);
      // additional_data.mapping_update_flags =
      // 	(update_gradients | update_JxW_values | update_quadrature_points);
      // additional_data.mapping_update_flags_inner_faces =
      //   (update_gradients | update_JxW_values | update_normal_vectors);
      // additional_data.mapping_update_flags_boundary_faces =
      //   (update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points);
      QGauss<1> quadrature(fe_degree + 1);
      mf_storage->reinit(mapping, dof_handler, constraints, quadrature, additional_data);
    }

    const auto patch_storage = std::make_shared<SubdomainHandler<dim, double>>();
    {
      typename SubdomainHandler<dim, double>::AdditionalData additional_data;
      fill_schwarz_smoother_data<dim, double>(additional_data,
                                              rt_parameters.multigrid.pre_smoother.schwarz);
      additional_data.level = level;
      patch_storage->reinit(mf_storage, additional_data);
    }

    const auto patch_transfer = std::make_shared<PatchTransfer>(*patch_storage);

    // // *** check if we have the same number of DoFs in each component
    // const auto & dof_handlers = elasticity_problem->dof_handlers;
    // const bool   same_number_of_dofs_per_level =
    //   std::adjacent_find(dof_handlers.cbegin(),
    //                      dof_handlers.cend(),
    //                      [level](const auto & dh1, const auto & dh2) {
    //                        return dh1->n_dofs(level) != dh2->n_dofs(level);
    //                      });
    // ASSERT_TRUE(same_number_of_dofs_per_level)
    //   << "Not implemented for components with different numbers of degree of freedoms.";

    // // *** generate random input
    // const unsigned int n_components = mf_storage->n_components();
    // ASSERT_EQ(n_components, n_static_components);
    // LinearAlgebra::distributed::BlockVector<double> random_input(n_components);
    // for(unsigned int b = 0; b < n_components; ++b)
    // {
    //   auto & bvector = random_input.block(b);
    //   mf_storage->initialize_dof_vector(bvector, b);
    //   random_input.collect_sizes();
    //   const ArrayView<double> bview = make_array_view(bvector.begin(), bvector.end());
    //   fill_with_random_values(bview);
    // }

    // // *** check if restrict + prolongate equals the identity
    // *pcout << "TEST: restriction and prolongation\n";
    // auto                                   dst{random_input}; // first time !!!
    // AlignedVector<VectorizedArray<double>> tmp_local;
    // const auto &                           partition_data = patch_storage->get_partition_data();
    // const auto                             n_subdomains   = partition_data.n_subdomains();
    // for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    // {
    //   patch_transfer->reinit(patch_id);
    //   patch_transfer->reinit_local_vector(tmp_local);
    //   tmp_local.fill(make_vectorized_array<double>(0.));
    //   patch_transfer->gather_add(tmp_local, random_input); // restrict
    //   patch_transfer->scatter_add(dst, tmp_local);         // prolongate
    // }                                                      // second time !!!
    // dst *= 0.5;                                            // added 2 times random_input to dst
    // !!! Util::compare_vector(dst, random_input, *pcout);
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;

  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim>                           dof_handler;
  std::shared_ptr<FiniteElement<dim>>       fe;
  const MappingQGeneric<dim>                mapping;
  AffineConstraints<double>                 constraints;
};

TYPED_TEST_SUITE_P(TestPatchTransferVectorValued);

TYPED_TEST_P(TestPatchTransferVectorValued, CellPatch)
{
  using Fixture = TestPatchTransferVectorValued<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::cell;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;
  Fixture::rt_parameters.mesh.n_refinements = 2U;
  Fixture::check_dg();
}

TYPED_TEST_P(TestPatchTransferVectorValued, VertexPatch)
{
  // using Fixture = TestPatchTransferVectorValued<TypeParam>;

  // // There is only one vertex patch possible such that each degree of freedom
  // // uniquely belongs to one patch
  // Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  // Fixture::rt_parameters.mesh.n_repetitions    = 2;
  // Fixture::params.n_refinements                = 0U;
  // Fixture::params.patch_variant                = TPSS::PatchVariant::vertex;
  // Fixture::test();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferVectorValued, CellPatch, VertexPatch);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferVectorValued, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D,
                               TestPatchTransferVectorValued,
                               TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransferVectorValued, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D,
                               TestPatchTransferVectorValued,
                               TestParamsHigherOrder3D);



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
