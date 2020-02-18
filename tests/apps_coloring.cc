
/*
 * Visualizes the red-black coloring of multiplicative Schwarz smoothers defined
 * in 'apps/coloring.h'.
 *
 *  Created on: Nov 28, 2019
 *      Author: witte
 */

#include <gtest/gtest.h>

#include <deal.II/base/utilities.h>

#include "solvers_and_preconditioners/TPSS/patch_info.h"

#include "coloring.h"
#include "mesh.h"

using namespace dealii;

template<int dim = 2, typename ColoringType = RedBlackColoring<dim>>
struct TestColoring
{
  static constexpr int fe_degree = 1;

  TestColoring()
  {
    mesh_prms.geometry_variant = MeshParameter::GeometryVariant::Cube;
    mesh_prms.n_refinements    = 1;
    mesh_prms.n_repetitions    = 3;
  }

  struct Params
  {
    TPSS::PatchVariant patch_variant        = TPSS::PatchVariant::cell;
    bool               visualize_subdomains = true;
  };

  void
  visualize_coloring()
  {
    parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    DoFHandler<dim>                     dof_handler;
    std::shared_ptr<FiniteElement<dim>> fe;

    //: create triangulation
    create_mesh(triangulation, mesh_prms);

    //: distribute dofs
    fe = std::make_shared<FE_DGQ<dim>>(fe_degree);
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();

    TPSS::PatchInfo<dim>                          patch_info;
    typename TPSS::PatchInfo<dim>::AdditionalData additional_data;
    additional_data.patch_variant    = params.patch_variant;
    additional_data.smoother_variant = TPSS::SmootherVariant::multiplicative;
    const auto level                 = triangulation.n_global_levels() - 1;
    additional_data.level            = level;
    ColoringType user_coloring(mesh_prms);
    additional_data.coloring_func      = std::ref(user_coloring);
    additional_data.visualize_coloring = std::ref(ColoringType::visualize_coloring);
    patch_info.initialize(&dof_handler, additional_data);
  }

  Params        params;
  MeshParameter mesh_prms;
};



TEST(RedBlackColoring, TwoDimensionsCube)
{
  TestColoring test;

  /// cell
  test.params.patch_variant = TPSS::PatchVariant::cell;
  test.visualize_coloring();

  /// vertex patch
  test.params.patch_variant = TPSS::PatchVariant::vertex;
  test.visualize_coloring();
}

TEST(RedBlackColoring, TwoDimensionsCuboidSubdivided)
{
  TestColoring test;
  test.mesh_prms.geometry_variant = MeshParameter::GeometryVariant::CuboidSubdivided;
  test.mesh_prms.n_subdivisions.resize(/*dim*/ 2);
  test.mesh_prms.n_subdivisions[0] = 4;
  test.mesh_prms.n_subdivisions[1] = 3; // 4 x 3 root mesh !

  /// cell
  test.params.patch_variant = TPSS::PatchVariant::cell;
  test.visualize_coloring();

  /// vertex patch
  test.params.patch_variant = TPSS::PatchVariant::vertex;
  test.visualize_coloring();
}

TEST(TiledColoring, TwoDimensionsCube)
{
  TestColoring<2, TiledColoring<2>> test;

  /// vertex patch
  test.params.patch_variant = TPSS::PatchVariant::vertex;
  test.visualize_coloring();
}

TEST(TiledColoring, TwoDimensionsCuboidSubdivided)
{
  TestColoring<2, TiledColoring<2>> test;
  test.mesh_prms.geometry_variant = MeshParameter::GeometryVariant::CuboidSubdivided;
  test.mesh_prms.n_subdivisions.resize(/*dim*/ 2);
  test.mesh_prms.n_subdivisions[0] = 3;
  test.mesh_prms.n_subdivisions[1] = 2; // 3 x 2 root mesh !

  /// vertex patch
  test.params.patch_variant = TPSS::PatchVariant::vertex;
  test.visualize_coloring();
}

int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_coloring.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
