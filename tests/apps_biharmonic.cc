
/*
 * Test functionality of Biharmonic::ModelProblem.
 *
 *  Created on: Mar 13, 2021
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "biharmonic_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace Biharmonic;



using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
using TestParamsCubic     = testing::Types<Util::NonTypeParams<2, 3>>;



template<typename T>
class TestModelProblem : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();


  void
  SetUp() override
  {
    ofs.open("apps_biharmonic.log", std::ios_base::app);
    is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    is_quadratic  = fe_degree == 2;
    pcout_owned   = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::vertex;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  test_compute_nondivfree_shape_functions()
  {
    equation_data.variant = EquationData::Variant::ClampedStreamPoiseuilleNoSlip;

    Biharmonic::ModelProblem<dim, fe_degree> biharmonic_problem(rt_parameters, equation_data);
    biharmonic_problem.pcout = pcout_owned;
    biharmonic_problem.make_grid();
    biharmonic_problem.setup_system();

    auto & stokes_problem        = *(biharmonic_problem.stokes_problem);
    stokes_problem.pcout         = pcout_owned;
    stokes_problem.triangulation = biharmonic_problem.triangulation;
    if(fe_stokes)
      stokes_problem.fe = fe_stokes;
    stokes_problem.setup_system();

    const auto [trafomatrix_rt_to_gradp, trafomatrix_rt_to_constp] =
      biharmonic_problem.compute_nondivfree_shape_functions();

    const auto & dof_handler_velocity = stokes_problem.dof_handler_velocity;
    *pcout_owned << "finite element (velocity): " << dof_handler_velocity.get_fe().get_name()
                 << std::endl;

    using ::MW::ScratchData;

    using ::MW::DoF::CopyData;

    const auto face_worker = [&](const auto &         cell,
                                 const unsigned int & f,
                                 const unsigned int & sf,
                                 const auto &         ncell,
                                 const unsigned int & nf,
                                 const unsigned int & nsf,
                                 ScratchData<dim> &   scratch_data,
                                 CopyData &           copy_data) {
      auto & phi = scratch_data.fe_interface_values;
      phi.reinit(cell, f, sf, ncell, nf, nsf);

      const auto & lphi = phi.get_fe_face_values(0U);
      const auto & rphi = phi.get_fe_face_values(1U);

      auto & cell_data = copy_data.cell_data.emplace_back(2U, phi.n_quadrature_points);

      auto & ldof_indices = cell_data.dof_indices;
      ldof_indices.resize(lphi.dofs_per_cell);
      cell->get_active_or_mg_dof_indices(ldof_indices);

      auto & rdof_indices = cell_data.dof_indices;
      rdof_indices.resize(rphi.dofs_per_cell);
      ncell->get_active_or_mg_dof_indices(rdof_indices);

      ASSERT_EQ(lphi.dofs_per_cell, rphi.dofs_per_cell);
      ASSERT_EQ(trafomatrix_rt_to_constp.n(), lphi.dofs_per_cell);
      ASSERT_EQ(trafomatrix_rt_to_constp.m(), GeometryInfo<dim>::faces_per_cell);
      for(auto q = 0U; q < phi.n_quadrature_points; ++q)
      {
        Tensor<1, dim> lvalue, rvalue;
        for(auto i = 0U; i < lphi.dofs_per_cell; ++i)
          for(auto c = 0U; c < dim; ++c)
          {
            lvalue[c] += trafomatrix_rt_to_constp(f, i) * lphi.shape_value_component(i, q, c);
            rvalue[c] += trafomatrix_rt_to_constp(nf, i) * rphi.shape_value_component(i, q, c);
          }
        const auto & n          = phi.normal(q);
        cell_data.matrix(0U, q) = lvalue * n;
        cell_data.matrix(1U, q) = rvalue * n;
      }
    };

    const auto copier = [&](const CopyData & copy_data) {
      for(const auto & cd : copy_data.cell_data)
      {
        ASSERT_EQ(cd.matrix.m(), 2U);
        Vector<double> lvalues_times_normal(cd.matrix.n()), rvalues_times_normal(cd.matrix.n());
        for(auto q = 0U; q < cd.matrix.n(); ++q)
        {
          lvalues_times_normal[q] = cd.matrix(0U, q);
          rvalues_times_normal[q] = cd.matrix(1U, q);
        }
        compare_vector(lvalues_times_normal, rvalues_times_normal);
      }
    };

    const UpdateFlags update_flags_v = update_values | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags_v =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

    ScratchData<dim> scratch_data(stokes_problem.mapping,
                                  dof_handler_velocity.get_fe(),
                                  fe_degree + 1,
                                  update_flags_v,
                                  interface_update_flags_v);

    CopyData copy_data;

    ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1U);
    MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                          dof_handler_velocity.end(),
                          /*cell_worker*/ nullptr,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_interior_faces_once,
                          /*boundary_worker*/ nullptr,
                          face_worker);

    /// DEBUG
    if(is_quadratic && dof_handler_velocity.n_dofs() < 100)
    {
      //: visualize "interior" non-div-free test functions
      std::vector<types::global_dof_index> global_dof_indices_v(trafomatrix_rt_to_gradp.n());
      Vector<double>                       phi_K_i(dof_handler_velocity.n_dofs());
      unsigned int                         Ki = 0; // global shape function index
      for(const auto & cell : dof_handler_velocity.active_cell_iterators())
      {
        cell->get_active_or_mg_dof_indices(global_dof_indices_v);
        for(auto i = 0U; i < trafomatrix_rt_to_gradp.m(); ++i)
        {
          phi_K_i *= 0.;
          for(auto j = 0U; j < trafomatrix_rt_to_gradp.n(); ++j)
            phi_K_i(global_dof_indices_v[j]) = trafomatrix_rt_to_gradp(i, j);

          std::vector<std::string> names(dim, "shape_function");
          const std::string        prefix         = "tildev_interior_global";
          const std::string        suffix         = "phi" + Utilities::int_to_string(Ki++, 3);
          const auto               n_subdivisions = 10U;
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);
          visualize_dof_vector(dof_handler_velocity,
                               phi_K_i,
                               names,
                               prefix,
                               suffix,
                               n_subdivisions,
                               data_component_interpretation,
                               stokes_problem.mapping);
        }
      }

      //: visualize "interface" non-div-free test functions
      {
        Pressure::InterfaceHandler<dim> interface_handler;
        interface_handler.reinit(dof_handler_velocity.get_triangulation());

        std::vector<types::global_dof_index> global_dof_indices(trafomatrix_rt_to_constp.n());
        std::vector<types::global_dof_index> global_ndof_indices(trafomatrix_rt_to_constp.n());
        Vector<double>                       phi_K_i(dof_handler_velocity.n_dofs());
        unsigned int                         Ki = 0; // global shape function index
        for(const auto & cell : dof_handler_velocity.active_cell_iterators())
        {
          cell->get_active_or_mg_dof_indices(global_dof_indices);

          for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
          {
            const bool there_is_no_neighbor = cell->neighbor_index(face_no) == -1;

            if(there_is_no_neighbor)
              continue;

            const auto & ncell    = cell->neighbor(face_no);
            const auto   nface_no = cell->neighbor_face_no(face_no);

            const Pressure::InterfaceId interface_id{cell->id(), ncell->id()};
            const bool                  this_interface_isnt_contained =
              interface_handler.template get_interface_index<false>(interface_id) ==
              numbers::invalid_unsigned_int;

            if(this_interface_isnt_contained)
              continue;

            ncell->get_active_or_mg_dof_indices(global_ndof_indices);

            auto nonunique_dof_indices = global_dof_indices;
            std::copy(global_ndof_indices.begin(),
                      global_ndof_indices.end(),
                      std::back_inserter(nonunique_dof_indices));
            std::sort(nonunique_dof_indices.begin(), nonunique_dof_indices.end());

            std::vector<types::global_dof_index> unique_dof_indices;
            std::unique_copy(nonunique_dof_indices.begin(),
                             nonunique_dof_indices.end(),
                             std::back_inserter(unique_dof_indices));

            std::vector<unsigned int> counts;
            std::transform(unique_dof_indices.begin(),
                           unique_dof_indices.end(),
                           std::back_inserter(counts),
                           [&](const auto index) {
                             return std::count(nonunique_dof_indices.begin(),
                                               nonunique_dof_indices.end(),
                                               index);
                           });

            *pcout_owned << vector_to_string(counts) << std::endl;

            std::map<types::global_dof_index, unsigned int> dof_index_to_count;
            for(auto i = 0U; i < counts.size(); ++i)
              dof_index_to_count.emplace(unique_dof_indices[i], counts[i]);

            phi_K_i *= 0.;
            for(auto j = 0U; j < trafomatrix_rt_to_constp.n(); ++j)
              phi_K_i(global_dof_indices[j]) =
                trafomatrix_rt_to_constp(face_no, j) /
                static_cast<double>(dof_index_to_count[global_dof_indices[j]]);
            for(auto j = 0U; j < trafomatrix_rt_to_constp.n(); ++j)
              phi_K_i(global_ndof_indices[j]) =
                trafomatrix_rt_to_constp(nface_no, j) /
                static_cast<double>(dof_index_to_count[global_ndof_indices[j]]);

            std::vector<std::string> names(dim, "shape_function");
            const std::string        prefix         = "tildev_face_global";
            const std::string        suffix         = "phi" + Utilities::int_to_string(Ki++, 3);
            const auto               n_subdivisions = 10U;
            std::vector<DataComponentInterpretation::DataComponentInterpretation>
              data_component_interpretation(
                dim, DataComponentInterpretation::component_is_part_of_vector);
            visualize_dof_vector(dof_handler_velocity,
                                 phi_K_i,
                                 names,
                                 prefix,
                                 suffix,
                                 n_subdivisions,
                                 data_component_interpretation,
                                 stokes_problem.mapping);
          }
        }
      }
    }
  }


  template<typename VectorType>
  void
  compare_vector(const VectorType & vec, const VectorType & other) const
  {
    Util::compare_vector(vec, other, *pcout_owned);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  std::shared_ptr<FiniteElement<dim>> fe_stokes;

  bool is_first_proc;
  bool is_quadratic;
};



TYPED_TEST_SUITE_P(TestModelProblem);

TYPED_TEST_P(TestModelProblem, compute_nondivfree_shape_functions_RTmoments)
{
  using Fixture = TestModelProblem<TypeParam>;

  Fixture::fe_stokes =
    std::make_shared<FESystem<Fixture::dim>>(FE_RaviartThomas<Fixture::dim>(Fixture::fe_degree - 1),
                                             1,
                                             FE_DGQLegendre<Fixture::dim>(Fixture::fe_degree - 1),
                                             1);

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::test_compute_nondivfree_shape_functions();

  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::test_compute_nondivfree_shape_functions();
}

TYPED_TEST_P(TestModelProblem, compute_nondivfree_shape_functions_RTnodal)
{
  using Fixture = TestModelProblem<TypeParam>;

  Fixture::fe_stokes =
    std::make_shared<FESystem<Fixture::dim>>(FE_RaviartThomasNodal_new<Fixture::dim>(
                                               Fixture::fe_degree - 1),
                                             1,
                                             FE_DGQLegendre<Fixture::dim>(Fixture::fe_degree - 1),
                                             1);

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::test_compute_nondivfree_shape_functions();

  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::test_compute_nondivfree_shape_functions();
}

REGISTER_TYPED_TEST_SUITE_P(TestModelProblem,
                            compute_nondivfree_shape_functions_RTmoments,
                            compute_nondivfree_shape_functions_RTnodal);

INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestModelProblem, TestParamsQuadratic);
INSTANTIATE_TYPED_TEST_SUITE_P(Cubic2D, TestModelProblem, TestParamsCubic);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_biharmonic.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
