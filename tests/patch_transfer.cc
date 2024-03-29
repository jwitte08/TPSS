/*
 * Tests for patch_transfer.h:
 *
 *    (1a) DGQ cell patch (scalar)
 *    (1b) DGQ vertex patch (scalar)
 *    (1c) Q vertex patch (scalar)
 *    (2a) DGQ^dim cell patch (block)
 *    (2b) DGQ^dim vertex patch (block)
 *    (3a) DGQ^dim cell patch (vector-valued)
 *    (3b) DGQ^dim vertex patch (vector-valued)
 *    (3c) Q^dim vertex patch (vector-valued)
 *    (4a) DGP cell patch (scalar) (MPI)
 *    (4b) DGP vertex patch (scalar) (MPI)
 *    (4c) Q^dim x DGP vertex patch (block) (MPI)
 *    (5a) RAS Q vertex patch (scalar)
 *
 *  Created on: Feb 06, 2020
 *      Author: witte
 *  (based on Tpatch_transfer02.cc, Tpatch_transfer01.cc)
 */

#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_dgp.h>

#include <gtest/gtest.h>

#include "linelasticity_problem.h"
#include "poisson_problem.h"
#include "stokes_problem.h"

#include "test_utilities.h"

using namespace dealii;



using TestParamsConstant2D = testing::Types<Util::NonTypeParams<2, 0>>;
using TestParamsLinear2D   = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsHigherOrder2D =
  testing::Types<Util::NonTypeParams<2, 3>, Util::NonTypeParams<2, 4>>;

using TestParamsLinear3D = testing::Types<Util::NonTypeParams<3, 1>>;
using TestParamsHigherOrder3D =
  testing::Types<Util::NonTypeParams<3, 3>, Util::NonTypeParams<3, 4>>;
using TestParamsConstant3D = testing::Types<Util::NonTypeParams<3, 0>>;



/**
 * Comparison functor struct to compare two Points and return if one is
 * "less" than the other one. This can be used to use Point<dim> as a key in
 * std::map.
 *
 * Comparison is done through an artificial downstream direction that
 * tells directions apart through a factor of 1e-5. Once we got the
 * direction, we check for its value. In case the distance is exactly zero
 * (without an epsilon), we might still have the case that two points
 * combine in a particular way, e.g. the points (1.0, 1.0) and (1.0+1e-5,
 * 0.0). Thus, compare the points component by component in the second
 * step. Thus, points need to have identical floating point components to
 * be considered equal.
 */
template<int dim, typename Number = double>
struct ComparisonHelper
{
  /**
   * Comparison operator.
   *
   * Return true if @p lhs is considered less than @p rhs.
   */
  bool
  operator()(const Point<dim, Number> & lhs, const Point<dim, Number> & rhs) const
  {
    double downstream_size = 0;
    double weight          = 1.;
    for(unsigned int d = 0; d < dim; ++d)
    {
      downstream_size += (rhs[d] - lhs[d]) * weight;
      weight *= 1e-5;
    }
    if(downstream_size < 0)
      return false;
    else if(downstream_size > 0)
      return true;
    else
    {
      for(unsigned int d = 0; d < dim; ++d)
      {
        if(lhs[d] == rhs[d])
          continue;
        return lhs[d] < rhs[d];
      }
      return false;
    }
  }
};



////////// TestPatchTransferBase



template<typename T>
class TestPatchTransferBase : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();


  void
  SetUp() override
  {
    ofs.open("patch_transfer.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    /// base mesh is a vertex patch
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_repetitions    = 2;
    rt_parameters.mesh.n_refinements    = 0U;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
  bool                                do_visualize = false;
};



////////// TestPatchTransfer



template<typename T>
class TestPatchTransfer : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();


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


  template<TPSS::DoFLayout dof_layout>
  void
  test()
  {
    using PoissonProblem = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;
    using PatchTransfer  = typename PoissonProblem::LEVEL_MATRIX::transfer_type;

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

    LinearAlgebra::distributed::Vector<double> dof_vector;
    mf_storage_global->initialize_dof_vector(dof_vector);
    fill_with_random_values(dof_vector);

    const auto &                               partition_data = fdss_storage->get_partition_data();
    const auto                                 n_subdomains   = partition_data.n_subdomains();
    LinearAlgebra::distributed::Vector<double> dst;
    mf_storage_global->initialize_dof_vector(dst);
    dst *= 0.;

    /// DEBUG: this might be useful at a later point...
    // LinearAlgebra::distributed::Vector<double> weights;
    // {
    // mf_storage_global->initialize_dof_vector(weights);
    // weights *= 0;
    // const auto locally_owned_range_mg =
    //   filter_iterators(dof_handler.mg_cell_iterators_on_level(global_level),
    // 		       IteratorFilters::LocallyOwnedLevelCell());
    // const auto partitioner = weights.get_partitioner();
    // std::vector<types::global_dof_index> dof_indices(poisson_problem->fe->n_dofs_per_cell());
    // for(const auto & cell : locally_owned_range_mg)
    //   {
    // 	cell->get_active_or_mg_dof_indices(dof_indices);
    // 	for (auto i = 0U; i < dof_indices.size(); ++i)
    // 	  if(partitioner->in_local_range(dof_indices[i]))
    // 	    weights(dof_indices[i]) += 1;
    //   }
    // }

    *pcout << "Restrict & Prolongate = Identity ...  \n\n";
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);

      auto local_vector = patch_transfer->gather(dof_vector);
      patch_transfer->scatter_add(dst, local_vector);
    }

    /// DEBUG: this might be useful at a later point...
    // std::transform(dst.begin(), dst.end(), weights.begin(), dst.begin(), [](const auto value,
    // const auto weight){return value/weight;});

    if constexpr(dof_layout == TPSS::DoFLayout::Q)
    {
      AffineConstraints<double> level_constraints;
      IndexSet                  relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, global_level, relevant_dofs);
      level_constraints.reinit(relevant_dofs);
      level_constraints.add_lines(
        poisson_problem->mg_constrained_dofs->get_boundary_indices(global_level));
      level_constraints.close();
      level_constraints.set_zero(dst);
      level_constraints.set_zero(dof_vector);
    }

    Util::compare_vector(dst, dof_vector, *pcout);
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
};

TYPED_TEST_SUITE_P(TestPatchTransfer);

TYPED_TEST_P(TestPatchTransfer, CellPatchDG)
{
  using Fixture = TestPatchTransfer<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::cell;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::rt_parameters.mesh.n_refinements = 2U;
  Fixture::template test<TPSS::DoFLayout::DGQ>();
}

TYPED_TEST_P(TestPatchTransfer, VertexPatchDG)
{
  using Fixture = TestPatchTransfer<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::template test<TPSS::DoFLayout::DGQ>();
}

TYPED_TEST_P(TestPatchTransfer, VertexPatchQ)
{
  using Fixture = TestPatchTransfer<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::template test<TPSS::DoFLayout::Q>();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransfer, CellPatchDG, VertexPatchDG, VertexPatchQ);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransfer, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestPatchTransfer, TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransfer, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D, TestPatchTransfer, TestParamsHigherOrder3D);



////////// TestPatchTransferBlockDG



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



////////// TestPatchTransferVectorValued



template<typename T>
class TestPatchTransferVectorValued : public testing::Test
{
protected:
  static constexpr int          dim                 = T::template value<0>();
  static constexpr int          fe_degree           = T::template value<1>();
  static constexpr unsigned int n_static_components = dim;
  using PatchTransfer                               = TPSS::PatchTransfer<dim, double>;


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


  template<TPSS::DoFLayout dof_layout>
  void
  check_impl()
  {
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;

    if(dof_layout == TPSS::DoFLayout::Q)
      fe = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree), dim);
    else if(dof_layout == TPSS::DoFLayout::DGQ)
      fe = std::make_shared<FESystem<dim>>(FE_DGQ<dim>(fe_degree), dim);
    ASSERT_TRUE(fe) << "No valid finite element set.";
    *pcout << fe->get_name() << std::endl;

    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    if(dof_layout == TPSS::DoFLayout::Q)
    {
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
    }
    constraints.close();

    const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mg_level              = level;
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

    //: generate random input
    LinearAlgebra::distributed::Vector<double> dst;
    mf_storage->initialize_dof_vector(dst);
    // std::iota(dst.begin(), dst.end(), 0.);
    fill_with_random_values(dst); // first time !
    if(dof_layout == TPSS::DoFLayout::Q)
      constraints.set_zero(dst);
    const LinearAlgebra::distributed::Vector<double> src(dst);
    // dst *= 0.;

    //: restrict to and prolongate from each patch
    *pcout << "Restrict & Prolongate = Identity ...  \n\n";
    const auto & partition_data = patch_storage->get_partition_data();
    const auto   n_subdomains   = partition_data.n_subdomains();
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);
      const auto local_vector = patch_transfer->gather(src);
      patch_transfer->scatter_add(dst, local_vector); // second time !
    }

    //: compare vectors (we added 2 times src !)
    dst *= 0.5;
    Util::compare_vector(dst, src, *pcout);
  }


  void
  check_dg()
  {
    check_impl<TPSS::DoFLayout::DGQ>();
  }


  void
  check_q()
  {
    check_impl<TPSS::DoFLayout::Q>();
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

TYPED_TEST_P(TestPatchTransferVectorValued, CellPatchDG)
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

TYPED_TEST_P(TestPatchTransferVectorValued, VertexPatchDG)
{
  using Fixture = TestPatchTransferVectorValued<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::check_dg();
}

TYPED_TEST_P(TestPatchTransferVectorValued, VertexPatchQ)
{
  using Fixture = TestPatchTransferVectorValued<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::check_q();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferVectorValued,
                            CellPatchDG,
                            VertexPatchDG,
                            VertexPatchQ);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferVectorValued, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D,
                               TestPatchTransferVectorValued,
                               TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransferVectorValued, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D,
                               TestPatchTransferVectorValued,
                               TestParamsHigherOrder3D);



////////// TestPatchTransferBlockWithVector



template<typename T>
class TestPatchTransferBlockWithVector : public testing::Test
{
protected:
  static constexpr int dim         = T::template value<0>();
  static constexpr int fe_degree   = T::template value<1>();
  static constexpr int fe_degree_p = fe_degree - 1;
  using PatchTransferBlock         = TPSS::PatchTransferBlock<dim, double>;


  TestPatchTransferBlockWithVector()
    : triangulation(Triangulation<dim>::maximum_smoothing),
      dof_handlers(2),
      constraints(2),
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


  template<typename E>
  std::vector<const E *>
  to_vector_of_ptrs(const std::vector<E> & vec)
  {
    std::vector<const E *> vec_of_ptrs;
    std::transform(vec.cbegin(),
                   vec.cend(),
                   std::back_inserter(vec_of_ptrs),
                   [](const auto & elem) { return &elem; });
    return vec_of_ptrs;
  }


  void
  check()
  {
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;

    std::vector<bool> velocity_mask(dim, true);
    velocity_mask.push_back(false);

    //: initialize velocity dof_handler & constraints
    {
      auto & dof_handler = dof_handlers[0];
      auto & constraints = this->constraints[0];
      fe_v               = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree), dim);
      dof_handler.initialize(triangulation, *fe_v);
      dof_handler.distribute_mg_dofs();
      DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
      constraints.close();
    }

    //: initialize pressure dof_handler
    {
      auto & dof_handler = dof_handlers[1];
      fe_p               = std::make_shared<FE_DGQ<dim>>(fe_degree_p);
      dof_handler.initialize(triangulation, *fe_p);
      dof_handler.distribute_mg_dofs();
    }

    /// Initialize MatrixFree storage (DoFHandlers with block-structure are not
    /// supported, thus we introduced separate dof_handlers for the velocity and
    /// pressure).
    const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mg_level              = level;
      QGauss<1> quadrature(fe_degree + 1);
      mf_storage->reinit(mapping,
                         to_vector_of_ptrs(dof_handlers),
                         to_vector_of_ptrs(constraints),
                         quadrature,
                         additional_data);
    }

    const auto patch_storage = std::make_shared<SubdomainHandler<dim, double>>();
    {
      typename SubdomainHandler<dim, double>::AdditionalData additional_data;
      fill_schwarz_smoother_data<dim, double>(additional_data,
                                              rt_parameters.multigrid.pre_smoother.schwarz);
      additional_data.level = level;
      patch_storage->reinit(mf_storage, additional_data);
    }

    const auto patch_transfer = std::make_shared<PatchTransferBlock>(*patch_storage);

    //: initialize dof_handler with block structure
    fe = std::make_shared<FESystem<dim>>(*fe_v, 1, *fe_p, 1);
    *pcout << fe->get_name() << std::endl;
    DoFHandler<dim> dof_handler;
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    //: generate random input
    const std::vector<unsigned int>            block_component{0U, 1U};
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    BlockVector<double> dst(dofs_per_block);
    fill_with_random_values(dst); // first time !
    const auto & constraints_velocity = constraints.front();
    constraints_velocity.set_zero(dst.block(0));
    const BlockVector<double> src(dst);

    //: restrict to and prolongate from each patch
    *pcout << "Restrict & Prolongate = Identity ...  \n\n";
    const auto & partition_data = patch_storage->get_partition_data();
    const auto   n_subdomains   = partition_data.n_subdomains();
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);
      const auto local_vector = patch_transfer->gather(src);
      patch_transfer->scatter_add(dst, local_vector); // second time !
    }

    //: compare vectors (we added 2 times src !)
    dst *= 0.5;
    Util::compare_vector(dst, src, *pcout);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;

  Triangulation<dim>                     triangulation;
  std::vector<DoFHandler<dim>>           dof_handlers;
  std::shared_ptr<FiniteElement<dim>>    fe;
  std::shared_ptr<FiniteElement<dim>>    fe_v;
  std::shared_ptr<FiniteElement<dim>>    fe_p;
  std::vector<AffineConstraints<double>> constraints;
  const MappingQGeneric<dim>             mapping;
};

TYPED_TEST_SUITE_P(TestPatchTransferBlockWithVector);

TYPED_TEST_P(TestPatchTransferBlockWithVector, VertexPatch)
{
  using Fixture = TestPatchTransferBlockWithVector<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::check();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferBlockWithVector, VertexPatch);

/// linear is not possible: MatrixFree does not support DGQ(0) pressure !
using TestParamsQuadratic2D = testing::Types<Util::NonTypeParams<2, 2>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D,
                               TestPatchTransferBlockWithVector,
                               TestParamsQuadratic2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D,
                               TestPatchTransferBlockWithVector,
                               TestParamsHigherOrder2D);

/// linear is not possible: MatrixFree does not support DGQ(0) pressure !
using TestParamsQuadratic3D = testing::Types<Util::NonTypeParams<3, 2>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic3D,
                               TestPatchTransferBlockWithVector,
                               TestParamsQuadratic3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D,
                               TestPatchTransferBlockWithVector,
                               TestParamsHigherOrder3D);



////////// TestPatchTransferDGP



template<typename T>
class TestPatchTransferDGP : public testing::Test
{
protected:
  static constexpr int dim         = T::template value<0>();
  static constexpr int fe_degree   = T::template value<1>();
  static constexpr int fe_degree_p = fe_degree;
  static constexpr int fe_degree_v = fe_degree_p + 2;


  TestPatchTransferDGP()
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  {
  }


  void
  SetUp() override
  {
    ofs.open("patch_transfer.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);
    rt_parameters.mesh.n_subdivisions.resize(dim, 1);
    rt_parameters.mesh.n_subdivisions.at(0) = 2U;
    rt_parameters.mesh.n_refinements        = 0U;
    rt_parameters.mesh.geometry_variant     = MeshParameter::GeometryVariant::CuboidSubdivided;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  check()
  {
    //: generate mesh
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;

    //: initialize dof_handler
    DoFHandler<dim>                     dof_handler;
    std::shared_ptr<FiniteElement<dim>> fe;
    fe = std::make_shared<FE_DGP<dim>>(fe_degree);
    *pcout << fe->get_name() << std::endl;
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    //: distribute subdomains
    TPSS::PatchInfo<dim> patch_info;
    {
      typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
      const auto schwarz_data          = rt_parameters.multigrid.pre_smoother.schwarz;
      additional_data.patch_variant    = schwarz_data.patch_variant;
      additional_data.smoother_variant = schwarz_data.smoother_variant;
      additional_data.level            = level;
      patch_info.initialize(&dof_handler, additional_data); // raw
      TPSS::PatchWorker<dim, double>{patch_info};           // vectorized
    }

    //: distribute dofs on subdomains
    QGauss<1>                                                         quadrature(fe_degree + 1);
    internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info;
    shape_info.reinit(quadrature, *fe, /*base_element_index*/ 0);
    TPSS::DoFInfo<dim, double> dof_info;
    {
      typename TPSS::DoFInfo<dim, double>::AdditionalData additional_data;
      additional_data.level = level;
      dof_info.initialize(&dof_handler, &patch_info, &shape_info, additional_data);
    }

    const auto   patch_transfer   = std::make_shared<TPSS::PatchTransfer<dim, double>>(dof_info);
    const auto & patch_dof_worker = patch_transfer->get_patch_dof_worker();

    //: generate random input
    LinearAlgebra::distributed::Vector<double> dst;
    patch_dof_worker.initialize_dof_vector(dst);
    // std::iota(dst.begin(), dst.end(), 0.);
    fill_with_random_values(dst); // first time !
    const LinearAlgebra::distributed::Vector<double> src(dst);
    // dst *= 0.;

    //: restrict to and prolongate from each patch
    *pcout << "Restrict & Prolongate = Identity ...  \n\n";
    src.update_ghost_values();
    dst.zero_out_ghosts();
    const auto & partition_data = patch_dof_worker.get_partition_data();
    const auto   n_subdomains   = partition_data.n_subdomains();
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);
      const auto local_vector = patch_transfer->gather(src);
      patch_transfer->scatter_add(dst, local_vector); // second time !
    }
    dst.compress(VectorOperation::add);

    //: compare vectors (we added 2 times src !)
    dst *= 0.5;
    Util::compare_vector(dst, src, *pcout);
  }


  void
  check_velocity_pressure()
  {
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;

    //: initialize velocity dof_handler & constraints
    std::shared_ptr<FiniteElement<dim>> fe_v;
    DoFHandler<dim>                     dof_handler_v;
    AffineConstraints<double>           constraints_v;
    {
      fe_v = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree_v), dim);
      *pcout << "Finite element (velocity): " << fe_v->get_name() << std::endl;
      dof_handler_v.initialize(triangulation, *fe_v);
      dof_handler_v.distribute_mg_dofs();

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler_v, locally_relevant_dofs);
      constraints_v.reinit(locally_relevant_dofs);
      DoFTools::make_zero_boundary_constraints(dof_handler_v, 0, constraints_v);
      constraints_v.close();
    }

    //: initialize pressure dof_handler
    std::shared_ptr<FiniteElement<dim>> fe_p;
    DoFHandler<dim>                     dof_handler_p;
    {
      fe_p = std::make_shared<FE_DGP<dim>>(fe_degree_p);
      *pcout << "Finite element (pressure): " << fe_p->get_name() << std::endl;
      dof_handler_p.initialize(triangulation, *fe_p);
      dof_handler_p.distribute_mg_dofs();
    }

    //: distribute subdomains
    TPSS::PatchInfo<dim> patch_info;
    {
      typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
      const auto schwarz_data          = rt_parameters.multigrid.pre_smoother.schwarz;
      additional_data.patch_variant    = schwarz_data.patch_variant;
      additional_data.smoother_variant = schwarz_data.smoother_variant;
      additional_data.level            = level;
      patch_info.initialize(&dof_handler_v, additional_data); // raw
      TPSS::PatchWorker<dim, double>{patch_info};             // vectorized
    }

    //: distribute velocity + pressure dofs on subdomains
    QGauss<1>                                                         quadrature(fe_degree_v + 1);
    std::vector<TPSS::DoFInfo<dim, double>>                           dof_infos(2U);
    internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info_v;
    {
      shape_info_v.reinit(quadrature, *fe_v, /*base_element_index*/ 0);
      typename TPSS::DoFInfo<dim, double>::AdditionalData additional_data;
      additional_data.level = level;
      dof_infos[0].initialize(&dof_handler_v, &patch_info, &shape_info_v, additional_data);
    }
    internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info_p;
    {
      shape_info_p.reinit(quadrature, *fe_p, /*base_element_index*/ 0);
      typename TPSS::DoFInfo<dim, double>::AdditionalData additional_data;
      additional_data.level = level;
      dof_infos[1].initialize(&dof_handler_p, &patch_info, &shape_info_p, additional_data);
    }

    const auto patch_transfer = std::make_shared<TPSS::PatchTransferBlock<dim, double>>(dof_infos);

    LinearAlgebra::distributed::BlockVector<double> dst;
    patch_transfer->initialize_dof_vector(dst);
    for(auto b = 0U; b < dst.n_blocks(); ++b)
      fill_with_random_values(dst.block(b)); // first time !
    constraints_v.set_zero(dst.block(0));
    const LinearAlgebra::distributed::BlockVector<double> src(dst);

    //: restrict to and prolongate from each patch
    src.update_ghost_values();
    dst.zero_out_ghosts();
    *pcout << "Restrict & Prolongate = Identity ...  \n\n";
    const auto & partition_data = patch_transfer->get_patch_dof_worker(0).get_partition_data();
    const auto   n_subdomains   = partition_data.n_subdomains();
    for(unsigned patch_id = 0; patch_id < n_subdomains; ++patch_id)
    {
      patch_transfer->reinit(patch_id);
      const auto local_vector = patch_transfer->gather(src);
      patch_transfer->scatter_add(dst, local_vector); // second time !
    }
    dst.compress(VectorOperation::add);

    //: compare vectors (we added 2 times src !)
    dst *= 0.5;
    Util::compare_vector(dst, src, *pcout);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;

  parallel::distributed::Triangulation<dim> triangulation;
};

TYPED_TEST_SUITE_P(TestPatchTransferDGP);

TYPED_TEST_P(TestPatchTransferDGP, CellPatchMPI)
{
  using Fixture = TestPatchTransferDGP<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::cell;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::check();

  Fixture::rt_parameters.mesh.n_refinements = 2U;
  Fixture::check();
}

TYPED_TEST_P(TestPatchTransferDGP, VertexPatchMPI)
{
  using Fixture = TestPatchTransferDGP<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::check();
}

TYPED_TEST_P(TestPatchTransferDGP, VertexPatchVelocityPressureMPI)
{
  using Fixture = TestPatchTransferDGP<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  // There is only one vertex patch possible such that each degree of freedom
  // uniquely belongs to one patch
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0U;
  Fixture::check_velocity_pressure();
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferDGP,
                            CellPatchMPI,
                            VertexPatchMPI,
                            VertexPatchVelocityPressureMPI);

INSTANTIATE_TYPED_TEST_SUITE_P(Constant2D, TestPatchTransferDGP, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferDGP, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestPatchTransferDGP, TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Constant3D, TestPatchTransferDGP, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransferDGP, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D, TestPatchTransferDGP, TestParamsHigherOrder3D);



template<typename T>
class TestPatchTransferget_dof_indices : public TestPatchTransferBase<T>
{
protected:
  using Base = TestPatchTransferBase<T>;
  using Base::dim;
  using Base::fe_degree;
  using Base::pcout;
  using Base::rt_parameters;

  void
  check_get_dof_indices_lane_component(std::shared_ptr<const FiniteElement<dim>> fe)
  {
    Base::rt_parameters.multigrid.post_smoother.schwarz =
      Base::rt_parameters.multigrid.pre_smoother.schwarz;

    //: generate mesh
    parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;

    //: initialize dof_handler
    DoFHandler<dim> dof_handler;
    *pcout << fe->get_name() << std::endl;
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    //: distribute subdomains
    TPSS::PatchInfo<dim> patch_info;
    {
      typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
      const auto schwarz_data          = rt_parameters.multigrid.pre_smoother.schwarz;
      additional_data.patch_variant    = schwarz_data.patch_variant;
      additional_data.smoother_variant = schwarz_data.smoother_variant;
      additional_data.level            = level;
      patch_info.initialize(&dof_handler, additional_data); // raw
      TPSS::PatchWorker<dim, double>{patch_info};           // vectorized
    }

    //: distribute dofs on subdomains
    QGauss<1>                                                         quadrature(fe_degree + 1);
    internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info;
    shape_info.reinit(quadrature, *fe, /*base_element_index*/ 0);
    TPSS::DoFInfo<dim, double> dof_info;
    {
      typename TPSS::DoFInfo<dim, double>::AdditionalData additional_data;
      additional_data.level = level;
      dof_info.initialize(&dof_handler, &patch_info, &shape_info, additional_data);
    }

    TPSS::PatchTransfer<dim, double> patch_transfer(dof_info);
    const auto &                     patch_dof_worker = patch_transfer.get_patch_dof_worker();

    ASSERT_EQ(fe->n_base_elements(), 1U);
    const auto n_components = fe->n_components();

    const auto n_subdomains = patch_dof_worker.get_partition_data().n_subdomains();
    for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
    {
      patch_transfer.reinit(patch_index);
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
      {
        const auto &              view_total = patch_transfer.get_dof_indices(lane);
        std::vector<unsigned int> dof_indices_total(view_total.begin(), view_total.end());

        std::vector<unsigned int> dof_indices;
        for(auto comp = 0U; comp < n_components; ++comp)
        {
          const auto & view = patch_transfer.get_dof_indices(lane, comp);
          std::copy(view.begin(), view.end(), std::back_inserter(dof_indices));
        }

        std::sort(dof_indices_total.begin(), dof_indices_total.end());
        std::sort(dof_indices.begin(), dof_indices.end());
        ASSERT_TRUE(dof_indices_total == dof_indices);
      }
    }
  }
};

TYPED_TEST_SUITE_P(TestPatchTransferget_dof_indices);

TYPED_TEST_P(TestPatchTransferget_dof_indices, DGQMPI)
{
  constexpr auto dim       = TestFixture::dim;
  constexpr auto fe_degree = TestFixture::fe_degree;

  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  const auto fe = std::make_shared<FE_DGQ<dim>>(fe_degree);

  /// cell patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::cell;

  /// 1
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 2
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);

  /// vertex patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::vertex;

  /// 3
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 4
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);
}

TYPED_TEST_P(TestPatchTransferget_dof_indices, DGQvectorvaluedMPI)
{
  constexpr auto dim       = TestFixture::dim;
  constexpr auto fe_degree = TestFixture::fe_degree;

  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  const auto fe = std::make_shared<FESystem<dim>>(FE_DGQ<dim>(fe_degree), dim);

  /// cell patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::cell;

  /// 1
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 2
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);

  /// vertex patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::vertex;

  /// 3
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 4
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);
}

// !!! TODO test fails with mpirun
TYPED_TEST_P(TestPatchTransferget_dof_indices, Qvectorvalued)
{
  constexpr auto dim       = TestFixture::dim;
  constexpr auto fe_degree = TestFixture::fe_degree;

  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  const auto fe = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree), dim);

  /// vertex patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::vertex;

  /// 3
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 4
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);
}

TYPED_TEST_P(TestPatchTransferget_dof_indices, DGPvectorvaluedMPI)
{
  constexpr auto dim       = TestFixture::dim;
  constexpr auto fe_degree = TestFixture::fe_degree;

  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  const auto fe = std::make_shared<FESystem<dim>>(FE_DGP<dim>(fe_degree), dim);

  /// cell patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::cell;

  /// 1
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 2
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);

  /// vertex patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::vertex;

  /// 3
  TestFixture::check_get_dof_indices_lane_component(fe);
  /// 4
  TestFixture::rt_parameters.mesh.n_refinements = 2U;
  TestFixture::check_get_dof_indices_lane_component(fe);
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferget_dof_indices,
                            DGQMPI,
                            DGQvectorvaluedMPI,
                            Qvectorvalued,
                            DGPvectorvaluedMPI);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestPatchTransferget_dof_indices, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D,
                               TestPatchTransferget_dof_indices,
                               TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear3D, TestPatchTransferget_dof_indices, TestParamsLinear3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D,
                               TestPatchTransferget_dof_indices,
                               TestParamsHigherOrder3D);



template<typename T>
class TestPatchTransferrscatter : public TestPatchTransferBase<T>
{
protected:
  using Base = TestPatchTransferBase<T>;
  using Base::dim;
  using Base::fe_degree;
  using Base::pcout;
  using Base::rt_parameters;

  void
  check_rscatter(std::shared_ptr<const FiniteElement<dim>> fe)
  {
    Base::rt_parameters.multigrid.post_smoother.schwarz =
      Base::rt_parameters.multigrid.pre_smoother.schwarz;

    //: generate mesh
    parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    *pcout << create_mesh(triangulation, rt_parameters.mesh) << std::endl;
    const unsigned int level = triangulation.n_global_levels() - 1;

    //: initialize dof_handler
    DoFHandler<dim> dof_handler;
    *pcout << fe->get_name() << std::endl;
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    ASSERT_EQ(fe->n_base_elements(), 1U);
    const auto n_components = fe->n_components();
    ASSERT_EQ(n_components, 1U);

    //: initialize zero-boundary constraints
    AffineConstraints<double> zero_constraints;
    if(TPSS::get_dof_layout(*fe) == TPSS::DoFLayout::Q)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
      zero_constraints.reinit(relevant_dofs);
      DoFTools::make_zero_boundary_constraints(dof_handler, 0U, zero_constraints);
    }
    zero_constraints.close();

    //: distribute subdomains
    TPSS::PatchInfo<dim> patch_info;
    {
      typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
      const auto schwarz_data          = rt_parameters.multigrid.pre_smoother.schwarz;
      additional_data.patch_variant    = schwarz_data.patch_variant;
      additional_data.smoother_variant = schwarz_data.smoother_variant;
      additional_data.level            = level;
      patch_info.initialize(&dof_handler, additional_data); // raw
      TPSS::PatchWorker<dim, double>{patch_info};           // vectorized
    }

    //: distribute dofs on subdomains
    QGauss<1>                                                         quadrature(fe_degree + 1);
    internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info;
    shape_info.reinit(quadrature, *fe, /*base_element_index*/ 0);
    TPSS::DoFInfo<dim, double> dof_info;
    {
      typename TPSS::DoFInfo<dim, double>::AdditionalData additional_data;
      additional_data.level                       = level;
      additional_data.compute_ras_boolean_weights = true;
      dof_info.initialize(&dof_handler, &patch_info, &shape_info, additional_data);
    }

    TPSS::PatchTransfer<dim, double> patch_transfer(dof_info);
    const auto &                     patch_dof_worker = patch_transfer.get_patch_dof_worker();

    LinearAlgebra::distributed::Vector<double> vec;
    patch_transfer.initialize_dof_vector(vec);
    LinearAlgebra::distributed::Vector<double> one_vector;
    patch_transfer.initialize_dof_vector(one_vector);
    std::fill(one_vector.begin(), one_vector.end(), 1.);
    zero_constraints.distribute(one_vector);

    std::vector<LinearAlgebra::distributed::Vector<double>> visual_vecs;

    const auto write_gnuplot_restricted_dof_info =
      [&](
        std::ostream &                                                       out,
        const std::map<types::global_dof_index, Point<dim>> &                support_points,
        const std::map<types::global_dof_index, std::vector<unsigned int>> & dof_to_patch_indices) {
        AssertDimension(support_points.size(), dof_to_patch_indices.size());
        using point_map_t = std::map<Point<dim>, std::vector<unsigned int>, ComparisonHelper<dim>>;

        point_map_t point_map;

        // generate mapping from support points to associated patch indices (unique)
        for(const auto & [dof_index, support_point] : support_points)
        {
          const auto & patch_indices = dof_to_patch_indices.at(dof_index);
          std::copy(patch_indices.cbegin(),
                    patch_indices.cend(),
                    std::back_inserter(point_map[support_point]));
        }

        // print the newly created map:
        for(typename point_map_t::iterator it = point_map.begin(); it != point_map.end(); ++it)
        {
          // const auto & v = std::vector<unsigned int>(it->second.cbegin(), it->second.cend());
          const auto & v = it->second;
          if(!v.empty())
          {
            out << it->first << " \"";
            for(unsigned int i = 0; i < v.size(); ++i)
            {
              if(i > 0)
                out << ",";
              out << v[i];
            }

            out << "\"\n";
          }
        }
        out << std::flush;
      };

    const auto n_subdomains = patch_dof_worker.get_partition_data().n_subdomains();
    for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
    {
      patch_transfer.reinit(patch_index);
      AlignedVector<VectorizedArray<double>> local_one_vector;
      patch_transfer.reinit_local_vector(local_one_vector);
      local_one_vector.fill(make_vectorized_array(1.));

      patch_transfer.rscatter_add(vec, local_one_vector);

      /// DEBUG visualize in paraview
      if(dim == 2 && Base::do_visualize)
      {
        for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
        {
          LinearAlgebra::distributed::Vector<double> vvec;
          patch_transfer.initialize_dof_vector(vvec);

          auto one_on_lane  = make_vectorized_array<double>(0.);
          one_on_lane[lane] = 1.;
          AlignedVector<VectorizedArray<double>> local_one_vector_on_lane;
          patch_transfer.reinit_local_vector(local_one_vector_on_lane);
          local_one_vector_on_lane.fill(one_on_lane);

          patch_transfer.rscatter_add(vvec, local_one_vector_on_lane);
          visual_vecs.emplace_back(std::move(vvec));
        }
      }
    }

    Util::compare_vector(vec, one_vector, *pcout);

    /// DEBUG visualize restricted dof distribution
    if(dim == 2 && Base::do_visualize)
    {
      std::map<types::global_dof_index, std::vector<unsigned int>> dof_to_patch_indices;
      for(types::global_dof_index i = 0; i < vec.size(); ++i)
        dof_to_patch_indices[i] = std::vector<unsigned int>{};

      for(auto patch = 0U; patch < visual_vecs.size(); ++patch)
      {
        const auto & vec = visual_vecs.at(patch);
        for(types::global_dof_index i = 0; i < vec.size(); ++i)
        {
          if(vec(i) > 0.5)
            dof_to_patch_indices[i].push_back(patch);
        }
      }

      std::ostringstream oss;
      oss << "gnuplot_RAS_" << TPSS::str_dof_layout(dof_handler.get_fe()) << Base::fe_degree;
      std::ofstream                                 out(oss.str() + ".gpl");
      std::map<types::global_dof_index, Point<dim>> support_points;
      out << "set terminal svg size 800,800 enhanced font \"Helvetica,14\"\n"
          << "set output \"" << oss.str() << ".svg\"\n"
          << "set size square\n"
          << "set view equal xy\n"
          << "unset xtics\n"
          << "unset ytics\n"
          << "unset grid\n"
          << "unset border\n"
          << "set lmargin 1\n"
          << "set rmargin 1\n"
          << "set tmargin 2.5\n"
          << "set bmargin 1\n"
          << "plot '-' using 1:2 with lines notitle, "
          << "'-' with labels point pt 2 offset 1,1 notitle" << std::endl;
      GridOut().write_gnuplot(dof_handler.get_triangulation(), out);
      out << "e" << std::endl;
      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);
      write_gnuplot_restricted_dof_info(out, support_points, dof_to_patch_indices);
      out << "e" << std::endl;
    }
  }
};

TYPED_TEST_SUITE_P(TestPatchTransferrscatter);

TYPED_TEST_P(TestPatchTransferrscatter, Q)
{
  constexpr auto dim       = TestFixture::dim;
  constexpr auto fe_degree = TestFixture::fe_degree;

  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  const auto fe = std::make_shared<FE_Q<dim>>(fe_degree);

  /// vertex patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::vertex;
  TestFixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  TestFixture::rt_parameters.mesh.n_repetitions    = 4U;

  /// 3
  TestFixture::do_visualize = true;
  TestFixture::check_rscatter(fe);
  /// 4
  TestFixture::do_visualize                     = false;
  TestFixture::rt_parameters.mesh.n_refinements = 1U;
  TestFixture::check_rscatter(fe);
}

TYPED_TEST_P(TestPatchTransferrscatter, DGQ)
{
  constexpr auto dim       = TestFixture::dim;
  constexpr auto fe_degree = TestFixture::fe_degree;

  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  const auto fe = std::make_shared<FE_DGQ<dim>>(fe_degree);

  /// vertex patch
  TestFixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant =
    TPSS::PatchVariant::vertex;
  TestFixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  TestFixture::rt_parameters.mesh.n_repetitions    = 4U;

  /// 3
  TestFixture::do_visualize = true;
  TestFixture::check_rscatter(fe);
  /// 4
  TestFixture::do_visualize                     = false;
  TestFixture::rt_parameters.mesh.n_refinements = 1U;
  TestFixture::check_rscatter(fe);
}

REGISTER_TYPED_TEST_SUITE_P(TestPatchTransferrscatter, Q, DGQ);

INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestPatchTransferrscatter, TestParamsQuadratic2D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder2D, TestPatchTransferrscatter, TestParamsHigherOrder2D);

INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic3D, TestPatchTransferrscatter, TestParamsQuadratic3D);
INSTANTIATE_TYPED_TEST_SUITE_P(HigherOrder3D, TestPatchTransferrscatter, TestParamsHigherOrder3D);



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
