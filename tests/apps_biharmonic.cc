
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

    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_repetitions    = 1;
    rt_parameters.mesh.n_refinements    = 0;

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


  void
  test_prolongation_sf_to_rt()
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

    const auto & dofh_v  = stokes_problem.dof_handler_velocity;
    const auto & dofh_sf = biharmonic_problem.dof_handler;

    auto & curl_phi_i = stokes_problem.system_solution.block(0);
    auto & phi_i      = biharmonic_problem.system_u;

    constexpr unsigned int n_subdivisions = 10;

    const auto visualize_curl_phi = [&](const unsigned int i, const auto & phi_i) {
      const std::string prefix = "curl_phi";
      const std::string suffix = "phi" + Utilities::int_to_string(i, 3);

      std::ostringstream oss;
      oss << prefix << "_"
          << "Q" << Utilities::int_to_string(dofh_sf.get_fe().degree) << "_" << suffix << ".vtu";

      std::ofstream ofs(oss.str());

      DataOut<dim> data_out;

      StreamVelocityPP<dim> stream_velocity_pp;

      data_out.attach_dof_handler(dofh_sf);

      data_out.add_data_vector(phi_i, stream_velocity_pp);

      data_out.build_patches(biharmonic_problem.mapping,
                             n_subdivisions,
                             DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      data_out.write_vtu(ofs);
    };

    for(auto i = 0U; i < dofh_sf.get_fe().dofs_per_cell; ++i)
    {
      phi_i *= 0.;
      phi_i[i] = 1.;

      biharmonic_problem.prolongate_sf_to_velocity(curl_phi_i, phi_i);

      /// visualization as curl of stream function phi_i
      visualize_curl_phi(i, phi_i);

      /// visualization as prolongation into velocity ansatz
      std::vector<std::string> names(dim, "shape_function");
      const std::string        prefix = "curl_phi_prolongated";
      const std::string        suffix = "phi" + Utilities::int_to_string(i, 3);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);

      visualize_dof_vector(dofh_v,
                           curl_phi_i,
                           names,
                           prefix,
                           suffix,
                           n_subdivisions,
                           data_component_interpretation,
                           stokes_problem.mapping);

      EXPECT_NEAR(biharmonic_problem.template compute_stream_function_error<false>(),
                  0.,
                  Util::numeric_eps<double>);
    }
  }


  void
  test_vpprolongation_sf_to_rt()
  {
    using namespace TPSS;

    rt_parameters.solver.variant              = "cg";
    rt_parameters.solver.n_iterations_max     = 10000;
    rt_parameters.solver.control_variant      = SolverParameter::ControlVariant::relative;
    rt_parameters.solver.abs_tolerance        = 1.e-14;
    rt_parameters.solver.rel_tolerance        = 1.e-08;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::None;

    // equation_data.variant = EquationData::Variant::ClampedStreamPoiseuilleNoSlip;
    equation_data.variant = EquationData::Variant::ClampedStreamNoSlip;

    constexpr unsigned int n_q_points_1d = fe_degree + 1;
    QGauss<1>              quad_1d(n_q_points_1d);

    Biharmonic::ModelProblem<dim, fe_degree> biharmonic_problem(rt_parameters, equation_data);
    biharmonic_problem.pcout = pcout_owned;
    biharmonic_problem.make_grid();
    biharmonic_problem.setup_system();
    biharmonic_problem.assemble_system();
    biharmonic_problem.solve();
    biharmonic_problem.solve_pressure();

    PatchInfo<dim> patch_info;
    {
      using additional_data_type = typename PatchInfo<dim>::AdditionalData;
      additional_data_type additional_data;
      additional_data.patch_variant    = PatchVariant::vertex;
      additional_data.smoother_variant = SmootherVariant::additive;
      additional_data.level            = biharmonic_problem.max_level();
      ASSERT_EQ(rt_parameters.multigrid.pre_smoother.schwarz.patch_variant, PatchVariant::vertex);
      ASSERT_EQ(rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant,
                SmootherVariant::additive);
      additional_data.coloring_func = std::ref(*biharmonic_problem.user_coloring);
      patch_info.initialize(&biharmonic_problem.dof_handler, additional_data);
      PatchWorker<dim, double>{patch_info};
    }

    dealii::internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info;
    shape_info.reinit(quad_1d, biharmonic_problem.dof_handler.get_fe());

    DoFInfo<dim, double> dof_info;
    {
      using additional_data_type = typename DoFInfo<dim, double>::AdditionalData;
      additional_data_type additional_data;
      additional_data.level = biharmonic_problem.max_level();
      dof_info.initialize(&biharmonic_problem.dof_handler,
                          &patch_info,
                          &shape_info,
                          additional_data);
    }

    auto & stokes_problem        = *(biharmonic_problem.stokes_problem);
    stokes_problem.pcout         = pcout_owned;
    stokes_problem.triangulation = biharmonic_problem.triangulation;
    if(fe_stokes)
      stokes_problem.fe = fe_stokes;
    stokes_problem.setup_system();
    stokes_problem.dof_handler_velocity.distribute_mg_dofs();
    stokes_problem.assemble_system();

    dealii::internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<double>> shape_info_v;
    shape_info_v.reinit(quad_1d, stokes_problem.dof_handler_velocity.get_fe());

    DoFInfo<dim, double> dof_info_v;
    {
      using additional_data_type = typename DoFInfo<dim, double>::AdditionalData;
      additional_data_type additional_data;
      additional_data.level = stokes_problem.max_level();
      dof_info_v.initialize(&stokes_problem.dof_handler_velocity,
                            &patch_info,
                            &shape_info_v,
                            additional_data);
    }

    const auto & dofh_v  = stokes_problem.dof_handler_velocity;
    const auto & dofh_sf = biharmonic_problem.dof_handler;

    PatchTransfer<dim, double> patch_transfer_sf(dof_info);
    PatchTransfer<dim, double> patch_transfer_v(dof_info_v);

    const FullMatrix<double> & cell_prolongation_matrix =
      biharmonic_problem.compute_prolongation_sf_to_velocity();

    FullMatrix<double> prolongation_matrix;
    FullMatrix<double> prolongation_matrix_orth;
    FullMatrix<double> prolongation_matrix_orth_face;

    unsigned int patch_index = 0;
    unsigned int lane        = 0;

    patch_transfer_sf.reinit(patch_index);
    patch_transfer_v.reinit(patch_index);

    const auto & g2l_sf = patch_transfer_sf.get_global_to_local_dof_indices(lane);
    const auto & g2l_v  = patch_transfer_v.get_global_to_local_dof_indices(lane);

    const auto g2l_if_impl = [&](const auto & map, const types::global_dof_index i) {
      const auto it = map.find(i);
      return it == map.end() ? numbers::invalid_unsigned_int : it->second;
    };

    const auto & g2l_if_sf = [&](const types::global_dof_index i) {
      return g2l_if_impl(g2l_sf, i);
    };

    const auto & g2l_if_v = [&](const types::global_dof_index i) { return g2l_if_impl(g2l_v, i); };

    const unsigned int n_patch_dofs_sf = g2l_sf.size();
    const unsigned int n_patch_dofs_v  = g2l_v.size();
    const unsigned int n_dofs_per_cell_p =
      stokes_problem.dof_handler_pressure.get_fe().dofs_per_cell;

    const auto & [rt_to_gradp, rt_to_constp] =
      biharmonic_problem.compute_nondivfree_shape_functions();

    Pressure::InterfaceHandler<dim> interface_handler;
    interface_handler.reinit(dofh_v.get_triangulation());
    AssertDimension(interface_handler.n_interfaces(), 1 << dim);

    const auto get_active_interface_indices = [&](const auto & cell) {
      return Stokes::Velocity::SIPG::MW::get_active_interface_indices_impl(interface_handler, cell);
    };

    prolongation_matrix.reinit(n_patch_dofs_v, n_patch_dofs_sf);
    prolongation_matrix_orth.reinit(n_patch_dofs_v, (n_dofs_per_cell_p - 1) * (1 << dim));
    prolongation_matrix_orth_face.reinit(n_patch_dofs_v, interface_handler.n_interfaces());

    const auto & cell_collection_sf =
      patch_transfer_sf.get_patch_dof_worker().get_cell_collection(patch_index, lane);
    const auto & cell_collection_v =
      patch_transfer_v.get_patch_dof_worker().get_cell_collection(patch_index, lane);

    std::vector<types::global_dof_index> global_dof_indices_sf(cell_prolongation_matrix.n());
    std::vector<types::global_dof_index> global_dof_indices_v(cell_prolongation_matrix.m());

    for(auto cell_no = 0U; cell_no < cell_collection_sf.size(); ++cell_no)
    {
      const auto & cell_sf = cell_collection_sf[cell_no];
      const auto & cell_v  = cell_collection_v[cell_no];

      cell_sf->get_active_or_mg_dof_indices(global_dof_indices_sf);
      cell_v->get_active_or_mg_dof_indices(global_dof_indices_v);

      const auto & [active_face_nos, global_face_nos] = get_active_interface_indices(cell_v);

      for(auto ci = 0U; ci < global_dof_indices_v.size(); ++ci)
      {
        const auto         i  = global_dof_indices_v[ci];
        const unsigned int ii = g2l_if_v(i);
        if(ii != numbers::invalid_unsigned_int)
        {
          for(auto cj = 0U; cj < global_dof_indices_sf.size(); ++cj)
          {
            const auto         j  = global_dof_indices_sf[cj];
            const unsigned int jj = g2l_if_sf(j);
            if(jj != numbers::invalid_unsigned_int)
              prolongation_matrix(ii, jj) = cell_prolongation_matrix(ci, cj);
          }

          for(auto cj = 0U; cj < rt_to_gradp.m(); ++cj)
          {
            const unsigned int jj            = cell_no * rt_to_gradp.m() + cj;
            prolongation_matrix_orth(ii, jj) = rt_to_gradp(cj, ci);
          }

          for(auto cj = 0U; cj < active_face_nos.size(); ++cj)
          {
            const unsigned int face_no            = active_face_nos[cj];
            const unsigned int jj                 = global_face_nos[cj];
            prolongation_matrix_orth_face(ii, jj) = rt_to_constp(face_no, ci);
          }
        }
      }
    }

    auto & curl_phi_i = stokes_problem.system_solution.block(0);
    auto & phi_i      = biharmonic_problem.system_u;

    constexpr unsigned int n_subdivisions = 10;

    const auto visualize_curl_phi = [&](const unsigned int i, const auto & phi_i) {
      const std::string prefix = "curl_phi";
      const std::string suffix = "phi" + Utilities::int_to_string(i, 3);

      std::ostringstream oss;
      oss << prefix << "_"
          << "Q" << Utilities::int_to_string(dofh_sf.get_fe().degree) << "_" << suffix << ".vtu";

      std::ofstream ofs(oss.str());

      DataOut<dim> data_out;

      StreamVelocityPP<dim> stream_velocity_pp;

      data_out.attach_dof_handler(dofh_sf);

      data_out.add_data_vector(phi_i, stream_velocity_pp);

      data_out.build_patches(biharmonic_problem.mapping,
                             n_subdivisions,
                             DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      data_out.write_vtu(ofs);
    };

    for(auto i = 0U; i < n_patch_dofs_sf; ++i)
    {
      curl_phi_i *= 0.;
      phi_i *= 0.;

      AlignedVector<VectorizedArray<double>> local_phi_i;
      patch_transfer_sf.reinit_local_vector(local_phi_i);
      local_phi_i[i][lane] = 1.;

      AlignedVector<VectorizedArray<double>> local_curl_phi_i;
      patch_transfer_v.reinit_local_vector(local_curl_phi_i);

      for(auto j = 0U; j < n_patch_dofs_v; ++j)
        local_curl_phi_i[j][lane] = prolongation_matrix(j, i);

      /// visualization as curl of stream function phi_i
      patch_transfer_sf.scatter(phi_i, local_phi_i);
      visualize_curl_phi(i, phi_i);

      /// visualization as prolongation into velocity ansatz
      std::vector<std::string> names(dim, "shape_function");
      const std::string        prefix = "curl_phi_prolongated";
      const std::string        suffix = "phi" + Utilities::int_to_string(i, 3);

      patch_transfer_v.scatter(curl_phi_i, local_curl_phi_i);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);

      visualize_dof_vector(dofh_v,
                           curl_phi_i,
                           names,
                           prefix,
                           suffix,
                           n_subdivisions,
                           data_component_interpretation,
                           stokes_problem.mapping);

      EXPECT_NEAR(biharmonic_problem.template compute_stream_function_error<false>(),
                  0.,
                  Util::numeric_eps<double>);
    }

    /// Prolongate the local coefficient vector phi = (1,2,3,...)^T.
    {
      curl_phi_i *= 0.;
      phi_i *= 0.;

      AlignedVector<VectorizedArray<double>> local_phi;
      patch_transfer_sf.reinit_local_vector(local_phi);
      for(auto i = 0U; i < n_patch_dofs_sf; ++i)
        local_phi[i][lane] = static_cast<double>(i);

      AlignedVector<VectorizedArray<double>> local_curl_phi;
      patch_transfer_v.reinit_local_vector(local_curl_phi);

      for(auto j = 0U; j < n_patch_dofs_v; ++j)
        for(auto k = 0U; k < n_patch_dofs_sf; ++k)
          local_curl_phi[j][lane] += local_phi[k][lane] * prolongation_matrix(j, k);

      patch_transfer_sf.scatter(phi_i, local_phi);
      patch_transfer_v.scatter(curl_phi_i, local_curl_phi);

      EXPECT_NEAR(biharmonic_problem.template compute_stream_function_error<false>(),
                  0.,
                  Util::numeric_eps<double>);
    }

    Stokes::ProlongationStream<dim, double> prolongation_stream(dofh_sf.get_fe(), dofh_v.get_fe());

    prolongation_matrix.print_formatted(std::cout);
    std::cout << std::endl;
    table_to_fullmatrix(prolongation_stream.prolongation_matrix.as_table(), lane)
      .print_formatted(std::cout);

    /// Compare system right-hand sides.
    {
      if(pcout_owned->is_active())
      {
        biharmonic_problem.system_rhs.print(pcout_owned->get_stream());
        stokes_problem.system_rhs.block(0).print(pcout_owned->get_stream());
      }

      const auto local_rhs_sf = patch_transfer_sf.gather(biharmonic_problem.system_rhs);

      const auto local_rhs_v = patch_transfer_v.gather(stokes_problem.system_rhs.block(0));

      AlignedVector<VectorizedArray<double>> restricted_local_rhs_v;
      patch_transfer_sf.reinit_local_vector(restricted_local_rhs_v);
      for(auto j = 0U; j < prolongation_matrix.n(); ++j)
        for(auto i = 0U; i < prolongation_matrix.m(); ++i)
          restricted_local_rhs_v[j] += prolongation_matrix(i, j) * local_rhs_v[i];

      *pcout_owned << "stream: " << vector_to_string(alignedvector_to_vector(local_rhs_sf, lane))
                   << std::endl;
      *pcout_owned << "divfree: "
                   << vector_to_string(alignedvector_to_vector(restricted_local_rhs_v, lane))
                   << std::endl;

      AlignedVector<VectorizedArray<double>> other_restricted_local_rhs_v;
      patch_transfer_sf.reinit_local_vector(other_restricted_local_rhs_v);
      const ArrayView<const VectorizedArray<double>> view_local_rhs_v(local_rhs_v.begin(),
                                                                      local_rhs_v.size());
      const ArrayView<VectorizedArray<double>> view_other_v(other_restricted_local_rhs_v.begin(),
                                                            other_restricted_local_rhs_v.size());
      prolongation_stream.dual_restrict(view_other_v, view_local_rhs_v);

      *pcout_owned << "divfree(other): "
                   << vector_to_string(alignedvector_to_vector(other_restricted_local_rhs_v, lane))
                   << std::endl;

      AlignedVector<VectorizedArray<double>> restricted_local_rhs_v_orth;
      restricted_local_rhs_v_orth.resize(prolongation_matrix_orth.n());
      for(auto j = 0U; j < prolongation_matrix_orth.n(); ++j)
        for(auto i = 0U; i < prolongation_matrix_orth.m(); ++i)
          restricted_local_rhs_v_orth[j] += prolongation_matrix_orth(i, j) * local_rhs_v[i];

      *pcout_owned << "orthcirc: "
                   << vector_to_string(alignedvector_to_vector(restricted_local_rhs_v_orth, lane))
                   << std::endl;

      AlignedVector<VectorizedArray<double>> restricted_local_rhs_v_orth_face;
      restricted_local_rhs_v_orth_face.resize(prolongation_matrix_orth_face.n());
      for(auto j = 0U; j < prolongation_matrix_orth_face.n(); ++j)
        for(auto i = 0U; i < prolongation_matrix_orth_face.m(); ++i)
          restricted_local_rhs_v_orth_face[j] +=
            prolongation_matrix_orth_face(i, j) * local_rhs_v[i];

      *pcout_owned << "orthface: "
                   << vector_to_string(
                        alignedvector_to_vector(restricted_local_rhs_v_orth_face, lane))
                   << std::endl;
    }
  }


  void
  debug_and_visualize_raviart_thomas()
  {
    equation_data.variant = EquationData::Variant::ClampedStreamPoiseuilleNoSlip;

    Biharmonic::ModelProblem<dim, fe_degree> biharmonic_problem(rt_parameters, equation_data);
    biharmonic_problem.pcout = pcout_owned;
    biharmonic_problem.make_grid();
    biharmonic_problem.setup_system();

    auto & stokes_problem        = *(biharmonic_problem.stokes_problem);
    using StokesProblem          = typename std::decay_t<decltype(stokes_problem)>;
    stokes_problem.pcout         = pcout_owned;
    stokes_problem.triangulation = biharmonic_problem.triangulation;
    if(fe_stokes)
      stokes_problem.fe = fe_stokes;
    stokes_problem.setup_system();

    const auto & dofh_v  = stokes_problem.dof_handler_velocity;
    const auto & fe_v    = dofh_v.get_fe();
    const auto & mapping = stokes_problem.mapping;

    const auto n_dofs_per_cell_v = fe_v.dofs_per_cell;

    ASSERT_EQ(n_dofs_per_cell_v, dofh_v.n_dofs()) << "Not a single cell...";

    /// Display RT shape functions in ParaView.
    for(auto i = 0U; i < n_dofs_per_cell_v; ++i)
    {
      Vector<double> phi_i(n_dofs_per_cell_v);
      phi_i[i] = 1.;

      std::vector<std::string> names(dim, "shape_function");
      const std::string prefix = "RT" + Utilities::int_to_string(StokesProblem::fe_degree_v, 1);
      const std::string suffix = "phi" + Utilities::int_to_string(i, 3);
      const auto        n_subdivisions = 10U;
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);
      visualize_dof_vector(dofh_v,
                           phi_i,
                           names,
                           prefix,
                           suffix,
                           n_subdivisions,
                           data_component_interpretation,
                           mapping);
    }
  }


  void
  raviartthomas_compare_shape_data()
  {
    const auto fe_rt_old = std::make_shared<FE_RaviartThomas<dim>>(fe_degree - 1);

    const auto fe_rt_new = std::make_shared<FE_RaviartThomas_new<dim>>(fe_degree - 1);

    const unsigned int n_q_points_1d = fe_degree + 1;

    QGauss<1>                                        quad_1d(n_q_points_1d);
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    fe_rt_new->fill_shape_info(shape_info, quad_1d);

    /// interior dofs/node functionals run faster over component than moment
    /// of the node functional for FE_RaviartThomas. for FE_RaviartThomas_new
    /// it is vice versa
    const unsigned int n_face_dofs =
      GeometryInfo<dim>::faces_per_cell * fe_rt_new->n_dofs_per_face();
    const unsigned int        n_interior_dofs          = fe_rt_new->n_dofs_per_cell() - n_face_dofs;
    const unsigned int        n_interior_dofs_per_comp = n_interior_dofs / dim;
    std::vector<unsigned int> o2n(n_interior_dofs);
    for(auto i = 0U; i < n_interior_dofs_per_comp; ++i)
      for(auto comp = 0U; comp < dim; ++comp)
        o2n[i * dim + comp] = comp * n_interior_dofs_per_comp + i;

    *pcout_owned << "(n_q x n_dofs) shape_value matrix:" << fe_rt_new->get_name() << " vs. "
                 << fe_rt_old->get_name() << std::endl;

    const unsigned int n_dofs_per_comp = fe_rt_new->n_dofs_per_cell() / dim;

    QGauss<dim> quad(n_q_points_1d);

    /// reference shape values
    FullMatrix<double> dim_shape_values(quad.size(), fe_rt_new->n_dofs_per_cell());

    for(auto comp = 0U; comp < dim; ++comp)
      for(auto j = 0U; j < fe_rt_new->n_dofs_per_cell(); ++j)
        for(auto q = 0U; q < quad.size(); ++q)
          /// for each shape function only one vector component is non-zero
          dim_shape_values(q, j) += fe_rt_new->shape_value_component(j, quad.point(q), comp);

    const auto fill_shape_values = [&](const unsigned int d, const unsigned int c) {
      const auto &       shape_data = shape_info.get_shape_data(d, c);
      const auto         n          = shape_data.fe_degree + 1;
      FullMatrix<double> shape_values(shape_data.n_q_points_1d, n);
      for(auto q = 0U; q < shape_data.n_q_points_1d; ++q)
        for(auto j = 0U; j < n; ++j)
          shape_values(q, j) = shape_data.shape_values[j * shape_data.n_q_points_1d + q];
      return shape_values;
    };

    /// shape values as tensor product
    const auto & shape_values_kplus1 = fill_shape_values(0, 0);

    const auto & shape_values_k = fill_shape_values(1, 0);

    const auto & l2h = shape_info.lexicographic_numbering;

    FullMatrix<double> tensor_shape_values(dim_shape_values.m(), dim_shape_values.n());

    FullMatrix<double> one(IdentityMatrix(1U)); // dummy

    for(auto comp = 0U; comp < dim; ++comp)
    {
      const unsigned int offset_comp = comp * n_dofs_per_comp;

      const FullMatrix<double> & S_2 =
        dim > 2 ? (comp == 2 ? shape_values_kplus1 : shape_values_k) : one;
      const FullMatrix<double> & S_1 =
        dim > 1 ? (comp == 1 ? shape_values_kplus1 : shape_values_k) : one;
      const FullMatrix<double> & S_0 =
        dim > 0 ? (comp == 0 ? shape_values_kplus1 : shape_values_k) : one;
      const FullMatrix<double> & prod =
        Tensors::kronecker_product(S_2, Tensors::kronecker_product(S_1, S_0));

      for(auto q = 0U; q < quad.size(); ++q)
        for(auto j = 0U; j < n_dofs_per_comp; ++j)
          tensor_shape_values(q, l2h[offset_comp + j]) = prod(q, j);
    }

    compare_matrix(tensor_shape_values, dim_shape_values);
  }


  void
  raviartthomas_compare_restriction()
  {
    const auto fe_rt_old = std::make_shared<FE_RaviartThomas<dim>>(fe_degree - 1);

    const auto fe_rt_new = std::make_shared<FE_RaviartThomas_new<dim>>(fe_degree - 1);

    const unsigned int                               n_q_points_1d = fe_degree + 1;
    QGauss<1>                                        quad_1d(n_q_points_1d);
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    fe_rt_new->fill_shape_info(shape_info, quad_1d);

    /// interior dofs/node functionals run faster over component than moment
    /// of the node functional for FE_RaviartThomas. for FE_RaviartThomas_new
    /// it is vice versa
    const unsigned int n_face_dofs =
      GeometryInfo<dim>::faces_per_cell * fe_rt_new->n_dofs_per_face();
    const unsigned int        n_interior_dofs          = fe_rt_new->n_dofs_per_cell() - n_face_dofs;
    const unsigned int        n_interior_dofs_per_comp = n_interior_dofs / dim;
    std::vector<unsigned int> o2n(n_interior_dofs);
    for(auto i = 0U; i < n_interior_dofs_per_comp; ++i)
      for(auto comp = 0U; comp < dim; ++comp)
        o2n[i * dim + comp] = comp * n_interior_dofs_per_comp + i;

    *pcout_owned << "comparing restriction matrices: " << fe_rt_new->get_name() << " vs. "
                 << fe_rt_old->get_name() << std::endl;
    for(auto child = 0U; child < GeometryInfo<dim>::max_children_per_cell; ++child)
    {
      const auto &       restriction_matrix     = fe_rt_new->get_restriction_matrix(child);
      const auto &       restriction_matrix_old = fe_rt_old->get_restriction_matrix(child);
      FullMatrix<double> ordered_rmatrix_old(restriction_matrix_old);
      {
        for(auto io = 0U; io < n_interior_dofs; ++io)
          for(auto jo = 0U; jo < n_interior_dofs; ++jo)
            ordered_rmatrix_old(n_face_dofs + o2n[io], n_face_dofs + o2n[jo]) =
              restriction_matrix_old(n_face_dofs + io, n_face_dofs + jo);
      }

      *pcout_owned << "restriction for child " << child << ":" << std::endl;
      compare_matrix(restriction_matrix, ordered_rmatrix_old);
    }
  }


  void
  raviartthomas_compare_generalized_support_points()
  {
    const auto fe_rt_old = std::make_shared<FE_RaviartThomas<dim>>(fe_degree - 1);

    const auto fe_rt_new = std::make_shared<FE_RaviartThomas_new<dim>>(fe_degree - 1);

    const auto & gsp_new = fe_rt_new->get_generalized_support_points();

    const auto & gsp_old = fe_rt_old->get_generalized_support_points();

    ASSERT_EQ(gsp_new, gsp_old);

    for(auto q = 0U; q < gsp_new.size(); ++q)
    {
      const auto & point_new = gsp_new[q];
      const auto & point_old = gsp_old[q];
      *pcout_owned << q << " : " << point_new << " vs. " << point_old << std::endl;
      EXPECT_TRUE(has_nearly_zero_abs(point_new.distance_square(point_old)))
        << "point " << q << " mismatches";
    }

    const FullMatrix<double> M_new = FETools::compute_node_matrix(*fe_rt_new);
    const FullMatrix<double> M_old = FETools::compute_node_matrix(*fe_rt_old);
    compare_matrix(M_new, M_old);
  }


  void
  raviartthomas_convert_generalized_support_point_values_to_dof_values()
  {
    const auto fe_rt_new = std::make_shared<FE_RaviartThomas_new<dim>>(fe_degree - 1);

    /// Computing the node value matrix after initializing the finite element
    /// has to result in the identity matrix. compute_node_matrix() makes us of
    /// convert_generalized_support_point_values_to_dof_values() to compute the
    /// node values... Of course we assume that the node values are correctly
    /// computed during construction.
    const FullMatrix<double> node_value_matrix = FETools::compute_node_matrix(*fe_rt_new);

    FullMatrix<double> id(IdentityMatrix(node_value_matrix.n()));
    compare_matrix(node_value_matrix, id);
  }


  template<typename VectorType>
  void
  compare_vector(const VectorType & vec, const VectorType & other) const
  {
    Util::compare_vector(vec, other, *pcout_owned);
  }


  void
  compare_matrix(const FullMatrix<double> & matrix, const FullMatrix<double> & other) const
  {
    Util::compare_matrix(matrix, other, *pcout_owned);
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

  const auto & blub = FE_RaviartThomas_new<Fixture::dim>(Fixture::fe_degree - 1);

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



TYPED_TEST_P(TestModelProblem, compute_prolongation_sf_to_rt)
{
  using Fixture = TestModelProblem<TypeParam>;

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;

  Fixture::fe_stokes =
    std::make_shared<FESystem<Fixture::dim>>(FE_RaviartThomas_new<Fixture::dim>(Fixture::fe_degree -
                                                                                1),
                                             1,
                                             FE_DGQLegendre<Fixture::dim>(Fixture::fe_degree - 1),
                                             1);

  Fixture::test_prolongation_sf_to_rt();
}



TYPED_TEST_P(TestModelProblem, compute_vpprolongation_sf_to_rt)
{
  using Fixture = TestModelProblem<TypeParam>;

  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;

  Fixture::fe_stokes =
    std::make_shared<FESystem<Fixture::dim>>(FE_RaviartThomas_new<Fixture::dim>(Fixture::fe_degree -
                                                                                1),
                                             1,
                                             FE_DGQLegendre<Fixture::dim>(Fixture::fe_degree - 1),
                                             1);

  Fixture::test_vpprolongation_sf_to_rt();
}



// TYPED_TEST_P(TestModelProblem, compare_rhs)
// {
//   using Fixture = TestModelProblem<TypeParam>;

//   Fixture::fe_stokes =
//     std::make_shared<FESystem<Fixture::dim>>(FE_RaviartThomas_new<Fixture::dim>(Fixture::fe_degree
//     -
//                                                                                 1),
//                                              1,
//                                              FE_DGQLegendre<Fixture::dim>(Fixture::fe_degree -
//                                              1), 1);

//   Fixture::compare_rhs();
// }



TYPED_TEST_P(TestModelProblem, debug_and_visualize_RTmoments)
{
  using Fixture = TestModelProblem<TypeParam>;

  Fixture::fe_stokes =
    std::make_shared<FESystem<Fixture::dim>>(FE_RaviartThomas_new<Fixture::dim>(Fixture::fe_degree -
                                                                                1),
                                             1,
                                             FE_DGQLegendre<Fixture::dim>(Fixture::fe_degree - 1),
                                             1);

  Fixture::debug_and_visualize_raviart_thomas();
}



TYPED_TEST_P(TestModelProblem, raviartthomas_compare_shape_data)
{
  using Fixture = TestModelProblem<TypeParam>;
  Fixture::raviartthomas_compare_shape_data();
}



TYPED_TEST_P(TestModelProblem, raviartthomas_compare_restriction)
{
  using Fixture = TestModelProblem<TypeParam>;
  Fixture::raviartthomas_compare_restriction();
}



TYPED_TEST_P(TestModelProblem, raviartthomas_compare_generalized_support_points)
{
  using Fixture = TestModelProblem<TypeParam>;
  Fixture::raviartthomas_compare_generalized_support_points();
}


TYPED_TEST_P(TestModelProblem, raviartthomas_convert_generalized_support_point_values_to_dof_values)
{
  using Fixture = TestModelProblem<TypeParam>;
  Fixture::raviartthomas_convert_generalized_support_point_values_to_dof_values();
}



REGISTER_TYPED_TEST_SUITE_P(TestModelProblem,
                            compute_nondivfree_shape_functions_RTmoments,
                            compute_nondivfree_shape_functions_RTnodal,
                            compute_prolongation_sf_to_rt,
                            compute_vpprolongation_sf_to_rt,
                            debug_and_visualize_RTmoments,
                            raviartthomas_compare_shape_data,
                            raviartthomas_compare_restriction,
                            raviartthomas_compare_generalized_support_points,
                            raviartthomas_convert_generalized_support_point_values_to_dof_values/*,
												  compare_rhs*/);

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
