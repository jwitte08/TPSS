/*
 * streamfunction_compute_pressure.cc
 *
 *  Created on: Sep 25, 2020
 *      Author: witte
 */

#include "biharmonic_problem.h"
#include "ct_parameter.h"
#include "stokes.h"
#include "utilities.h"



int
main(int argc, char * argv[])
{
  try
  {
    using namespace dealii;

    const auto atoi_if = [&](auto & prm, const int index) {
      Util::ConditionalAtoi(argc, argv)(prm, index);
    };
    // const auto atof_if = [&](auto & prm, const int index) {
    //   Util::ConditionalAtof(argc, argv)(prm, index);
    // };

    //: default
    unsigned int test_index  = 0;
    unsigned int debug_depth = 3;
    double       damping     = 0.;
    unsigned int pde_index   = 3;
    // unsigned int test_index_stokes = 0;
    // unsigned int pde_index_stokes  = 3;
    unsigned int n_cycles = 1;

    //: parse arguments
    atoi_if(test_index, 1);
    atoi_if(n_cycles, 2);
    atoi_if(pde_index, 3);
    // atof_if(ip_factor, 3);
    // atoi_if(debug_depth, 4);
    // atof_if(damping, 5);

    deallog.depth_console(debug_depth);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    constexpr int dim              = CT::DIMENSION_;
    constexpr int fe_degree_stream = CT::FE_DEGREE_;
    // constexpr int  fe_degree_pressure = fe_degree_stream - 1;
    constexpr auto patch_variant    = CT::PATCH_VARIANT_;
    constexpr auto smoother_variant = CT::SMOOTHER_VARIANT_;

    // 0: direct solver
    // 1: CG solver (no preconditioner)
    // 2: CG solver (GMG preconditioner without smoothing)
    // 3: CG solver (GMG preconditioner with symm. Gauss-Seidel smoothing)
    // 4: CG solver (GMG preconditioner with Schwarz smoothing)
    constexpr unsigned int test_index_max = 4;
    AssertThrow(test_index <= test_index_max, ExcMessage("test_index is not valid"));

    RT::Parameter prms;
    {
      //: discretization
      prms.n_cycles              = n_cycles;
      prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
      prms.mesh.n_refinements    = 1;
      prms.mesh.n_repetitions    = 2;

      //: solver
      prms.solver.variant              = test_index == 0 ? "direct" : "cg";
      prms.solver.rel_tolerance        = 1.e-8;
      prms.solver.precondition_variant = test_index >= 2 ?
                                           SolverParameter::PreconditionVariant::GMG :
                                           SolverParameter::PreconditionVariant::None;
      prms.solver.n_iterations_max = 200;

      //: multigrid
      if(damping == 0.)
        damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
      prms.multigrid.coarse_level                 = 0;
      prms.multigrid.coarse_grid.solver_variant   = CoarseGridParameter::SolverVariant::DirectSVD;
      prms.multigrid.coarse_grid.iterative_solver = "cg";
      prms.multigrid.coarse_grid.accuracy         = 1.e-12;
      const SmootherParameter::SmootherVariant smoother_variant[test_index_max + 1] = {
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::GaussSeidel,
        SmootherParameter::SmootherVariant::Schwarz};
      prms.multigrid.pre_smoother.variant                      = smoother_variant[test_index];
      prms.multigrid.pre_smoother.n_smoothing_steps            = 2;
      prms.multigrid.pre_smoother.schwarz.patch_variant        = CT::PATCH_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.smoother_variant     = CT::SMOOTHER_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
      prms.multigrid.pre_smoother.schwarz.damping_factor       = damping;
      prms.multigrid.post_smoother                             = prms.multigrid.pre_smoother;
      prms.multigrid.post_smoother.schwarz.reverse_smoothing   = true;
    }

    Biharmonic::EquationData equation_data;
    AssertThrow(pde_index < Biharmonic::EquationData::n_variants,
                ExcMessage("This equation is not implemented."));
    equation_data.variant = static_cast<Biharmonic::EquationData::Variant>(pde_index);

    Biharmonic::ModelProblem<dim, fe_degree_stream> biharmonic_problem(prms, equation_data);
    auto pcout               = std::make_shared<ConditionalOStream>(std::cout, true);
    biharmonic_problem.pcout = pcout;

    // Stokes::StokesFlow options_stokes;
    // if(damping == 0.)
    //   damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    // options_stokes.setup(test_index_stokes, damping);
    // options_stokes.prms.n_cycles = prms.n_cycles;
    // options_stokes.prms.mesh     = prms.mesh;

    // Stokes::EquationData equation_data_stokes;
    // equation_data_stokes.variant = static_cast<Stokes::EquationData::Variant>(pde_index_stokes);
    // equation_data_stokes.use_cuthill_mckee = false;
    // if(options_stokes.prms.solver.variant == "GMRES_GMG" ||
    //    options_stokes.prms.solver.variant == "CG_GMG")
    //   equation_data_stokes.local_kernel_size = 1U;
    // if(options_stokes.prms.solver.variant == "UMFPACK")
    //   equation_data_stokes.force_mean_value_constraint = true;

    // Stokes::ModelProblem<dim, fe_degree_pressure, Stokes::Method::RaviartThomas> stokes_problem(
    //   options_stokes.prms, equation_data_stokes);

    std::cout << std::endl;
    // !!!
    biharmonic_problem.run();
    auto & stokes_problem = *biharmonic_problem.stokes_problem;
    // biharmonic_problem.make_grid();
    // biharmonic_problem.setup_system();
    // biharmonic_problem.solve_pressure();
    // biharmonic_problem.stokes_problem->output_results(numbers::invalid_unsigned_int);

    // std::cout << std::endl;
    // stokes_problem.make_grid();
    // stokes_problem.setup_system();

    // Triangulation<dim> unit_triangulation(Triangulation<dim>::maximum_smoothing);
    // {
    //   MeshParameter mesh_prms;
    //   mesh_prms.geometry_variant = MeshParameter::GeometryVariant::Cube;
    //   mesh_prms.n_refinements    = 0U;
    //   mesh_prms.n_repetitions    = 1U;
    //   create_mesh(unit_triangulation, mesh_prms);
    //   AssertDimension(unit_triangulation.n_active_cells(), 1U);
    // }

    // const auto &    fe_v = stokes_problem.dof_handler_velocity.get_fe();
    // DoFHandler<dim> unit_dofh_v;
    // const auto &    fe_p = stokes_problem.dof_handler_pressure.get_fe();
    // DoFHandler<dim> unit_dofh_p;
    // unit_dofh_v.initialize(unit_triangulation, fe_v);
    // unit_dofh_p.initialize(unit_triangulation, fe_p);

    // const auto n_q_points_1d     = stokes_problem.n_q_points_1d;
    // const auto n_dofs_per_cell_v = fe_v.dofs_per_cell;
    // const auto n_dofs_per_cell_p = fe_p.dofs_per_cell;

    // /// Display RT shape functions in ParaView.
    // {
    //   AssertDimension(n_dofs_per_cell_v, unit_dofh_v.n_dofs()); // one cell
    //   for(auto i = 0U; i < n_dofs_per_cell_v; ++i)
    //   {
    //     Vector<double> phi_i(n_dofs_per_cell_v);
    //     phi_i[i] = 1.;

    //     std::vector<std::string> names(dim, "shape_function");
    //     const std::string        prefix         = "RT";
    //     const std::string        suffix         = "phi" + Utilities::int_to_string(i, 3);
    //     const auto               n_subdivisions = 10U;
    //     std::vector<DataComponentInterpretation::DataComponentInterpretation>
    //       data_component_interpretation(dim,
    //                                     DataComponentInterpretation::component_is_part_of_vector);
    //     visualize_dof_vector(unit_dofh_v,
    //                          phi_i,
    //                          names,
    //                          prefix,
    //                          suffix,
    //                          n_subdivisions,
    //                          data_component_interpretation,
    //                          stokes_problem.mapping);
    //   }
    // }



    Triangulation<dim> unit_triangulation(Triangulation<dim>::maximum_smoothing);
    {
      MeshParameter mesh_prms;
      mesh_prms.geometry_variant = MeshParameter::GeometryVariant::Cube;
      mesh_prms.n_refinements    = 0U;
      mesh_prms.n_repetitions    = 1U;
      create_mesh(unit_triangulation, mesh_prms);
      AssertDimension(unit_triangulation.n_active_cells(), 1U);
    }

    const auto &    fe_v = stokes_problem.dof_handler_velocity.get_fe();
    DoFHandler<dim> unit_dofh_v;
    const auto &    fe_p = stokes_problem.dof_handler_pressure.get_fe();
    DoFHandler<dim> unit_dofh_p;
    unit_dofh_v.initialize(unit_triangulation, fe_v);
    unit_dofh_p.initialize(unit_triangulation, fe_p);

    // const auto n_q_points_1d     = stokes_problem.n_q_points_1d;
    const auto n_dofs_per_cell_v = fe_v.dofs_per_cell;
    // const auto n_dofs_per_cell_p = fe_p.dofs_per_cell;

    /// Display RT shape functions in ParaView.
    {
      AssertDimension(n_dofs_per_cell_v, unit_dofh_v.n_dofs()); // one cell
      for(auto i = 0U; i < n_dofs_per_cell_v; ++i)
      {
        Vector<double> phi_i(n_dofs_per_cell_v);
        phi_i[i] = 1.;

        std::vector<std::string> names(dim, "shape_function");
        const std::string        prefix         = "RT";
        const std::string        suffix         = "phi" + Utilities::int_to_string(i, 3);
        const auto               n_subdivisions = 10U;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);
        visualize_dof_vector(unit_dofh_v,
                             phi_i,
                             names,
                             prefix,
                             suffix,
                             n_subdivisions,
                             data_component_interpretation,
                             stokes_problem.mapping);
      }
    }

    {
      AssertThrow(fe_v.has_generalized_support_points(), ExcMessage("Check the finite element."));
      const auto &                support_points = fe_v.get_generalized_support_points();
      std::vector<Vector<double>> support_values(support_points.size(), Vector<double>(dim));

      const auto & fe_sf = *biharmonic_problem.finite_element; // stream functions

      const auto compute_curl_phi_j = [&](const unsigned int j, const auto & x_q) {
        AssertThrow(dim == 2, ExcMessage("Only valid in 2D."));
        Vector<double> curl(dim);
        const auto &   grad = fe_sf.shape_grad(j, x_q);
        curl[0]             = grad[1];
        curl[1]             = -grad[0];
        return curl;
      };

      LAPACKFullMatrix<double> node_values(fe_v.dofs_per_cell, fe_sf.dofs_per_cell);
      std::vector<double>      node_values_of_curl_phi_j(fe_v.dofs_per_cell);
      for(auto j = 0U; j < fe_sf.dofs_per_cell; ++j)
      {
        std::transform(support_points.cbegin(),
                       support_points.cend(),
                       support_values.begin(),
                       [&](const auto & x_q) { return compute_curl_phi_j(j, x_q); });
        fe_v.convert_generalized_support_point_values_to_dof_values(support_values,
                                                                    node_values_of_curl_phi_j);

        for(auto i = 0U; i < fe_v.dofs_per_cell; ++i)
          node_values(i, j) = node_values_of_curl_phi_j[i];
      }

      const auto & [V, invSigma, UT]               = compute_inverse_svd(node_values);
      const auto &             inverse_node_values = merge_lapack_decomposition(V, invSigma, UT);
      LAPACKFullMatrix<double> trafomatrix(fe_v.dofs_per_cell, fe_sf.dofs_per_cell);
      inverse_node_values.transpose(trafomatrix);
      trafomatrix.print_formatted(std::cout);
    }



    // AssertDimension(dim, 2U);
    // const auto n_faces_per_cell  = GeometryInfo<dim>::faces_per_cell;
    // const auto n_interior_dofs_v = fe_v.dofs_per_quad;
    // const auto n_face_dofs_v     = n_faces_per_cell * fe_v.dofs_per_face;
    // AssertDimension(n_interior_dofs_v + n_face_dofs_v, n_dofs_per_cell_v);
    // const auto n_interior_nodes_by_pressure = n_dofs_per_cell_p - 1;

    // LAPACKFullMatrix<double> trafomatrix_rt_to_gradp(n_interior_nodes_by_pressure,
    //                                                  n_interior_dofs_v);

    // LAPACKFullMatrix<double> trafomatrix_rt_to_constp(n_faces_per_cell, n_face_dofs_v);

    // /// Compute the application of the current RT shape functions to the interior RT
    // /// node functionals, which are generated by the gradients of the pressure shape
    // /// functions (the constant pressure mode is neglected!).
    // {
    //   QGauss<dim>       quad(n_q_points_1d);
    //   const UpdateFlags update_flags =
    //     update_values | update_gradients | update_quadrature_points | update_JxW_values;
    //   FEValues<dim> fe_values_p(stokes_problem.mapping, fe_p, quad, update_flags);
    //   fe_values_p.reinit(unit_dofh_p.begin_active()); // one reference cell

    //   FEValues<dim> fe_values_v(stokes_problem.mapping, fe_v, quad, update_flags);
    //   fe_values_v.reinit(unit_dofh_v.begin_active()); // one reference cell

    //   const auto interior_node_functional = [&](const unsigned int   node_index_p,
    //                                             const Vector<double> v) {
    //     const auto n_q_points  = fe_values_p.n_quadrature_points;
    //     const auto dof_index_p = node_index_p + 1; // skip constant mode
    //     AssertIndexRange(dof_index_p, fe_values_p.dofs_per_cell);
    //     AssertDimension(v.size(), fe_values_v.dofs_per_cell);

    //     const auto & compute_v_q = [&](const unsigned int q) {
    //       Tensor<1, dim> value;
    //       for(unsigned int j = 0; j < n_dofs_per_cell_v; ++j)
    //         value += v[j] * MW::compute_vvalue(fe_values_v, j, q);
    //       return value;
    //     };

    //     double node_value = 0.;
    //     for(unsigned int q = 0; q < n_q_points; ++q)
    //     {
    //       const Tensor<1, dim> & grad_q = fe_values_p.shape_grad(dof_index_p, q);
    //       const Tensor<1, dim> & v_q    = compute_v_q(q);
    //       const auto &           dx     = fe_values_p.JxW(q);

    //       node_value += v_q * grad_q * dx;
    //     }

    //     return node_value;
    //   };

    //   LAPACKFullMatrix<double> node_value_weights(n_interior_nodes_by_pressure,
    //   n_interior_dofs_v); for(auto i = 0U; i < node_value_weights.m(); ++i)   // node functionals
    //     for(auto j = 0U; j < node_value_weights.n(); ++j) // interior shape funcs
    //     {
    //       Vector<double> phi_j(n_dofs_per_cell_v);
    //       const auto     dof_index = n_face_dofs_v + j; // shift to interior dofs
    //       phi_j[dof_index]         = 1.;
    //       node_value_weights(i, j) = interior_node_functional(i, phi_j);
    //     }

    //   const auto & [V, invSigma, UT] = compute_inverse_svd(node_value_weights);
    //   /// "Inverse" has to be understood in the sense of an inverse SVD.
    //   const auto & inverse_node_value_weights = merge_lapack_decomposition(V, invSigma, UT);
    //   inverse_node_value_weights.transpose(trafomatrix_rt_to_gradp);

    //   /// Display the "new shape functions" \tilde{v}_i
    //   for(auto i = 0U; i < n_interior_nodes_by_pressure; ++i)
    //   {
    //     Vector<double> phi_i(n_dofs_per_cell_v);
    //     for(auto j = 0U; j < n_interior_dofs_v; ++j)
    //     {
    //       const auto dof_index = n_face_dofs_v + j; // shift to interior dofs
    //       phi_i(dof_index)     = trafomatrix_rt_to_gradp(i, j);
    //     }

    //     std::vector<std::string> names(dim, "shape_function");
    //     const std::string        prefix         = "tildev_interior";
    //     const std::string        suffix         = "phi" + Utilities::int_to_string(i, 3);
    //     const auto               n_subdivisions = 10U;
    //     std::vector<DataComponentInterpretation::DataComponentInterpretation>
    //       data_component_interpretation(dim,
    //                                     DataComponentInterpretation::component_is_part_of_vector);
    //     visualize_dof_vector(unit_dofh_v,
    //                          phi_i,
    //                          names,
    //                          prefix,
    //                          suffix,
    //                          n_subdivisions,
    //                          data_component_interpretation,
    //                          stokes_problem.mapping);
    //   }
    // }

    // /// Compute the application of the given RT shape functions to the RT node
    // /// functional associated to a given face, which is generated by the constant function.
    // {
    //   QGauss<dim - 1>   quad(n_q_points_1d);
    //   const UpdateFlags update_flags =
    //     update_values | update_normal_vectors | update_quadrature_points | update_JxW_values;
    //   FEFaceValues<dim> fe_values_v(stokes_problem.mapping, fe_v, quad, update_flags);

    //   const auto face_node_functional = [&](const unsigned int face_no, const Vector<double> v) {
    //     const auto cell_v = unit_dofh_v.begin_active(); // one reference cell
    //     fe_values_v.reinit(cell_v, face_no);

    //     const auto   n_q_points = fe_values_v.n_quadrature_points;
    //     const auto & normals    = fe_values_v.get_normal_vectors();
    //     AssertDimension(v.size(), fe_values_v.dofs_per_cell);

    //     const auto & compute_v_q = [&](const unsigned int q) {
    //       Tensor<1, dim> value;
    //       for(unsigned int j = 0; j < n_dofs_per_cell_v; ++j)
    //         value += v[j] * MW::compute_vvalue(fe_values_v, j, q);
    //       return value;
    //     };

    //     double node_value = 0.;
    //     for(unsigned int q = 0; q < n_q_points; ++q)
    //     {
    //       const double orientation = GeometryInfo<dim>::unit_normal_orientation[face_no];
    //       /// TODO is this the correct way? with orientation we have normal
    //       /// continuity, but node functionals are not w.r.t. outward pointing
    //       /// normal, therefore we have to take care while computing pressure
    //       /// constants...
    //       const Tensor<1, dim> & n_q = orientation * normals[q];
    //       const Tensor<1, dim> & v_q = compute_v_q(q);
    //       const auto &           dx  = fe_values_v.JxW(q);

    //       node_value += v_q * n_q * dx;
    //     }

    //     return node_value;
    //   };

    //   LAPACKFullMatrix<double> node_value_weights(n_faces_per_cell, n_face_dofs_v);
    //   for(auto i = 0U; i < node_value_weights.m(); ++i)   // node functionals
    //     for(auto j = 0U; j < node_value_weights.n(); ++j) // face shape funcs
    //     {
    //       Vector<double> phi_j(n_dofs_per_cell_v);
    //       phi_j[j]                 = 1.;
    //       node_value_weights(i, j) = face_node_functional(/*face_no*/ i, phi_j);
    //     }

    //   const auto & [V, invSigma, UT]          = compute_inverse_svd(node_value_weights);
    //   const auto & inverse_node_value_weights = merge_lapack_decomposition(V, invSigma, UT);
    //   inverse_node_value_weights.transpose(trafomatrix_rt_to_constp);

    //   /// Display the "new shape functions" \tilde{v}_i
    //   for(auto i = 0U; i < n_faces_per_cell; ++i)
    //   {
    //     Vector<double> phi_i(n_dofs_per_cell_v);
    //     for(auto j = 0U; j < n_face_dofs_v; ++j)
    //       phi_i(j) = trafomatrix_rt_to_constp(i, j);

    //     std::vector<std::string> names(dim, "shape_function");
    //     const std::string        prefix         = "tildev_face";
    //     const std::string        suffix         = "phi" + Utilities::int_to_string(i, 3);
    //     const auto               n_subdivisions = 10U;
    //     std::vector<DataComponentInterpretation::DataComponentInterpretation>
    //       data_component_interpretation(dim,
    //                                     DataComponentInterpretation::component_is_part_of_vector);
    //     visualize_dof_vector(unit_dofh_v,
    //                          phi_i,
    //                          names,
    //                          prefix,
    //                          suffix,
    //                          n_subdivisions,
    //                          data_component_interpretation,
    //                          stokes_problem.mapping);
    //   }
    // }

    // {
    //   const auto & dof_handler_velocity = stokes_problem.dof_handler_velocity;
    //   const auto & dof_handler_pressure = stokes_problem.dof_handler_pressure;

    //   const auto & triangulation = stokes_problem.dof_handler_velocity.get_triangulation();
    //   const auto   n_cells       = triangulation.n_global_active_cells();

    //   using InterfaceId = typename std::pair<CellId, CellId>;

    //   std::vector<InterfaceId> interface_ids;
    //   std::set<CellId>         marked_cells;

    //   const auto & first_cell = triangulation.begin_active();
    //   interface_ids.emplace_back(first_cell->id(), first_cell->id());
    //   marked_cells.emplace(first_cell->id());

    //   while(marked_cells.size() < triangulation.n_global_active_cells())
    //     for(auto & cell : triangulation.active_cell_iterators())
    //     {
    //       const bool cell_is_marked = marked_cells.find(cell->id()) != marked_cells.cend();

    //       for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
    //       {
    //         const bool there_is_no_neighbor = cell->neighbor_index(face_no) == -1;
    //         if(there_is_no_neighbor)
    //           continue;

    //         const auto & ncell           = cell->neighbor(face_no);
    //         const bool   ncell_is_marked = marked_cells.find(ncell->id()) != marked_cells.cend();

    //         if(cell_is_marked && ncell_is_marked)
    //           continue;

    //         else if(cell_is_marked && !ncell_is_marked) // cell -> ncell
    //         {
    //           // {
    //           //   const auto & [dummy_, has_been_inserted] =
    //           //     interface_ids.emplace(cell->id(), ncell->id());
    //           //   Assert(has_been_inserted, ExcMessage("Insertion failed."));
    //           // }
    //           interface_ids.emplace_back(cell->id(), ncell->id());
    //           {
    //             const auto & [dummy_, has_been_inserted] = marked_cells.emplace(ncell->id());
    //             Assert(has_been_inserted, ExcMessage("Insertion failed."));
    //           }
    //         }

    //         else if(!cell_is_marked && ncell_is_marked) // ncell -> cell
    //         {
    //           // {
    //           //   const auto & [dummy_, has_been_inserted] =
    //           //     interface_ids.emplace(ncell->id(), cell->id());
    //           //   Assert(has_been_inserted, ExcMessage("Insertion failed."));
    //           // }
    //           interface_ids.emplace_back(ncell->id(), cell->id());
    //           {
    //             const auto & [dummy_, has_been_inserted] = marked_cells.emplace(cell->id());
    //             Assert(has_been_inserted, ExcMessage("Insertion failed."));
    //           }
    //         }
    //       }
    //     } // end loop over cells

    //   std::sort(interface_ids.begin(), interface_ids.end(), [](const auto & lhs, const auto &
    //   rhs) {
    //     return lhs.second < rhs.second;
    //   });

    //   AssertDimension(marked_cells.size(), n_cells);
    //   AssertDimension(marked_cells.size(), interface_ids.size());
    //   // DEBUG
    //   std::cout << "marked cells: " << std::endl;
    //   for(const auto & cell_id : marked_cells)
    //     std::cout << cell_id << " ";
    //   std::cout << std::endl;
    //   std::cout << "marked interfaces: " << std::endl;
    //   for(const auto & interface_id : interface_ids)
    //     std::cout << interface_id << std::endl;

    //   DynamicSparsityPattern dsp(n_cells);

    //   const auto & get_interface_it = [&](const InterfaceId & id) {
    //     const bool is_contained =
    //       std::binary_search(interface_ids.cbegin(), interface_ids.cend(), id);
    //     if(is_contained)
    //       return std::lower_bound(interface_ids.cbegin(), interface_ids.cend(), id);
    //     const InterfaceId flipped_id = {id.second, id.first};
    //     const bool        is_flipped =
    //       std::binary_search(interface_ids.cbegin(), interface_ids.cend(), flipped_id);
    //     if(is_flipped)
    //       return std::lower_bound(interface_ids.cbegin(), interface_ids.cend(), flipped_id);
    //     return interface_ids.cend();
    //   };

    //   const auto & get_interface_index = [&](const InterfaceId & id) -> unsigned int {
    //     const bool id_isnt_contained = interface_ids.cend() == get_interface_it(id);
    //     if(id_isnt_contained)
    //       return numbers::invalid_unsigned_int;
    //     return std::distance(interface_ids.cbegin(), get_interface_it(id));
    //   };

    //   const auto & get_cell_index = [&](const CellId & id) {
    //     const auto it = std::lower_bound(interface_ids.cbegin(),
    //                                      interface_ids.cend(),
    //                                      InterfaceId{id, id},
    //                                      [](const auto & lhs, const auto & rhs) {
    //                                        return lhs.second < rhs.second;
    //                                      });
    //     return std::distance(interface_ids.cbegin(), it);
    //   };

    //   const auto & get_cell_index_pair =
    //     [&](const InterfaceId & id) -> std::pair<unsigned int, unsigned int> {
    //     return {get_cell_index(id.first), get_cell_index(id.second)};
    //   };

    //   const auto interface_index_of_fixed_cell =
    //     get_interface_index({first_cell->id(), first_cell->id()});
    //   const auto fixed_cell_index = get_cell_index(first_cell->id());
    //   std::cout << "fixed cell index " << fixed_cell_index << " interface_index_of_fixed_cell "
    //             << interface_index_of_fixed_cell << std::endl;
    //   /// I am not sure if the constraints are still valid in case these indices
    //   /// do not coincide.
    //   AssertDimension(fixed_cell_index, interface_index_of_fixed_cell);

    //   AffineConstraints constraints_on_interface;
    //   constraints_on_interface.add_line(interface_index_of_fixed_cell);
    //   constraints_on_interface.set_inhomogeneity(interface_index_of_fixed_cell, 1.);
    //   constraints_on_interface.close();

    //   AffineConstraints constraints_on_cell;
    //   // constraints_on_cell.add_line(fixed_cell_index);
    //   // constraints_on_cell.set_inhomogeneity(fixed_cell_index, 0.);
    //   constraints_on_cell.close();

    //   for(const auto & id : interface_ids)
    //   {
    //     // DEBUG
    //     std::cout << "interface index (row): " << get_interface_index(id) << " ";
    //     const auto [left_index, right_index] = get_cell_index_pair(id);
    //     std::cout << "left cell index (column): " << left_index << " ";
    //     std::cout << "right cell index (column): " << right_index << " ";
    //     std::cout << std::endl;

    //     const auto e                 = get_interface_index(id);
    //     const auto [K_left, K_right] = get_cell_index_pair(id);
    //     dsp.add(e, K_left);
    //     dsp.add(e, K_right);
    //   }

    //   constraints_on_interface.condense(dsp);

    //   SparsityPattern sparsity_pattern;
    //   sparsity_pattern.copy_from(dsp);

    //   SparseMatrix<double> constant_pressure_matrix;
    //   constant_pressure_matrix.reinit(sparsity_pattern);
    //   Vector<double> right_hand_side(n_cells);

    //   std::vector<types::global_dof_index> constant_pressure_dof_indices(n_cells);
    //   Vector<double> & discrete_pressure = stokes_problem.system_solution.block(1);

    //   // TODO do we have to condense() matrix and rhs vector due to the inhomogeneity
    //   {
    //     using ::MW::ScratchData;

    //     using ::MW::Mixed::CopyData;

    //     using CellIterator = typename ::MW::IteratorSelector<dim, /*is_multigrid*/ false>::type;

    //     const auto                     component_range = std::make_pair<unsigned int>(0, dim);
    //     Stokes::FunctionExtractor<dim> load_function_velocity(stokes_problem.load_function.get(),
    //                                                           component_range);

    //     Function<dim> * load_function = &load_function_velocity;

    //     auto cell_worker =
    //       [&](const CellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    //         CellIterator cellP(&dof_handler_pressure.get_triangulation(),
    //                            cell->level(),
    //                            cell->index(),
    //                            &dof_handler_pressure);


    //         copy_data.cell_rhs_test = 0.;

    //         FEValues<dim> & phi = scratch_data.fe_values;
    //         phi.reinit(cell);

    //         cellP->get_active_or_mg_dof_indices(copy_data.local_dof_indices_test);

    //         const unsigned int cell_index = get_cell_index(cell->id());
    //         AssertDimension(copy_data.local_dof_indices_ansatz.size(), 1U);
    //         copy_data.local_dof_indices_ansatz[0U] = cell_index;

    //         std::vector<Tensor<1, dim>> load_values;
    //         {
    //           Assert(load_function, ExcMessage("load_function is not set."));
    //           AssertDimension(load_function->n_components, dim);
    //           const auto & q_points = phi.get_quadrature_points();
    //           std::transform(q_points.cbegin(),
    //                          q_points.cend(),
    //                          std::back_inserter(load_values),
    //                          [&](const auto & x_q) {
    //                            Tensor<1, dim> value;
    //                            for(auto c = 0U; c < dim; ++c)
    //                              value[c] = load_function->value(x_q, c);
    //                            return value;
    //                          });
    //         }

    //         const auto & compute_v_i = [&](const unsigned int i, const unsigned int q) {
    //           AssertIndexRange(i, n_interior_nodes_by_pressure);
    //           Tensor<1, dim> value;
    //           for(auto j = 0U; j < n_interior_dofs_v; ++j)
    //           {
    //             const auto jj = n_face_dofs_v + j; // shift to interior dofs
    //             for(auto d = 0U; d < dim; ++d)
    //               value[d] += trafomatrix_rt_to_gradp(i, j) * phi.shape_value_component(jj, q,
    //               d);
    //           }
    //           return value;
    //         };

    //         double integral_iq;
    //         for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
    //         {
    //           const auto & f  = load_values[q];
    //           const auto & dx = phi.JxW(q);

    //           for(auto i = 0U; i < n_interior_nodes_by_pressure; ++i)
    //           {
    //             const auto & v_i = compute_v_i(i, q);
    //             integral_iq      = v_i * f * dx;

    //             AssertIndexRange(i + 1, copy_data.cell_rhs_test.size());
    //             copy_data.cell_rhs_test(i + 1) += integral_iq; // skip constant mode
    //           }
    //         }
    //       };

    //     AffineConstraints<double> empty_constraints;
    //     empty_constraints.close();

    //     const auto copier = [&](const CopyData & copy_data) {
    //       empty_constraints.template distribute_local_to_global<Vector<double>>(
    //         copy_data.cell_rhs_test, copy_data.local_dof_indices_test, discrete_pressure);

    //       constant_pressure_dof_indices[copy_data.local_dof_indices_ansatz[0]] =
    //         copy_data.local_dof_indices_test[0];
    //     };

    //     const UpdateFlags update_flags =
    //       update_values | update_quadrature_points | update_JxW_values;

    //     ScratchData<dim> scratch_data(stokes_problem.mapping,
    //                                   stokes_problem.dof_handler_velocity.get_fe(),
    //                                   n_q_points_1d,
    //                                   update_flags);

    //     CopyData copy_data(n_dofs_per_cell_p, 1U);

    //     MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
    //                           dof_handler_velocity.end(),
    //                           cell_worker,
    //                           copier,
    //                           scratch_data,
    //                           copy_data,
    //                           MeshWorker::assemble_own_cells);

    //     discrete_pressure.print(std::cout);
    //     std::cout << vector_to_string(constant_pressure_dof_indices) << std::endl;
    //   }

    //   {
    //     using ::MW::Mixed::ScratchData;

    //     using ::MW::Mixed::CopyData;

    //     using CellIterator = typename ::MW::IteratorSelector<dim, /*is_multigrid*/ false>::type;

    //     const auto & dof_handler_velocity = stokes_problem.dof_handler_velocity;
    //     const auto & dof_handler_pressure = stokes_problem.dof_handler_pressure;

    //     const auto                     component_range = std::make_pair<unsigned int>(0, dim);
    //     Stokes::FunctionExtractor<dim> load_function_velocity(stokes_problem.load_function.get(),
    //                                                           component_range);

    //     Vector<double> & discrete_pressure = stokes_problem.system_solution.block(1);
    //     Function<dim> *  load_function     = &load_function_velocity;

    //     const auto & compute_v_face = [&](const auto &       phi,
    //                                       const unsigned int face_no,
    //                                       const unsigned int q) {
    //       AssertIndexRange(face_no, GeometryInfo<dim>::faces_per_cell);
    //       Tensor<1, dim> value;
    //       for(auto j = 0U; j < n_face_dofs_v; ++j)
    //       {
    //         for(auto d = 0U; d < dim; ++d)
    //           value[d] += trafomatrix_rt_to_constp(face_no, j) * phi.shape_value_component(j, q,
    //           d);
    //       }
    //       return value;
    //     };

    //     auto cell_worker =
    //       [&](const CellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    //         // const unsigned int cell_index = get_cell_index(cell->id());

    //         AssertDimension(copy_data.local_dof_indices_test.size(),
    //                         GeometryInfo<dim>::faces_per_cell);
    //         std::fill(copy_data.local_dof_indices_test.begin(),
    //                   copy_data.local_dof_indices_test.end(),
    //                   0);
    //         AssertDimension(copy_data.cell_rhs_test.size(), GeometryInfo<dim>::faces_per_cell);
    //         // copy_data.local_dof_indices_ansatz[0U] = cell_index;
    //         // AssertDimension(copy_data.local_dof_indices_ansatz.size(), 1U);
    //         copy_data.cell_rhs_test = 0.;
    //         // copy_data.cell_matrix   = 0.;

    //         for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
    //         {
    //           const bool this_is_no_interface = cell->neighbor_index(face_no) == -1;
    //           if(this_is_no_interface)
    //             continue;

    //           const auto         ncell           = cell->neighbor(face_no);
    //           const unsigned int interface_index = get_interface_index({cell->id(),
    //           ncell->id()}); const bool         this_interface_isnt_contained =
    //             interface_index == numbers::invalid_unsigned_int;
    //           if(this_interface_isnt_contained)
    //             continue;

    //           copy_data.local_dof_indices_test[face_no] = interface_index;

    //           FEValues<dim> & phi = scratch_data.fe_values_test;
    //           phi.reinit(cell);

    //           std::vector<Tensor<1, dim>> load_values;
    //           {
    //             Assert(load_function, ExcMessage("load_function is not set."));
    //             AssertDimension(load_function->n_components, dim);
    //             const auto & q_points = phi.get_quadrature_points();
    //             std::transform(q_points.cbegin(),
    //                            q_points.cend(),
    //                            std::back_inserter(load_values),
    //                            [&](const auto & x_q) {
    //                              Tensor<1, dim> value;
    //                              for(auto c = 0U; c < dim; ++c)
    //                                value[c] = load_function->value(x_q, c);
    //                              return value;
    //                            });
    //           }
    //           AssertDimension(load_values.size(), phi.n_quadrature_points);

    //           double residual = 0.;
    //           for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
    //           {
    //             {
    //               const auto & f      = load_values[q];
    //               const auto & dx     = phi.JxW(q);
    //               const auto & v_face = compute_v_face(phi, face_no, q);
    //               // std::cout << "v_face " << v_face << " f " << f << std::endl;
    //               residual += v_face * f * dx;
    //             }
    //           }
    //           std::cout << "interface index " << interface_index << " cell index "
    //                     << get_cell_index(cell->id()) << " cell index pair "
    //                     << get_cell_index_pair({cell->id(), ncell->id()}) << " res " << residual
    //                     << std::endl;

    //           copy_data.cell_rhs_test[face_no] = residual;
    //         }
    //       };

    //     /// This integrator assumes that the coefficients of the discrete pressure
    //     /// associated to the constant modes are set to zero!
    //     auto face_worker = [&](const auto &         cell,
    //                            const unsigned int & face_no,
    //                            const unsigned int & sface_no,
    //                            const auto &         ncell,
    //                            const unsigned int & nface_no,
    //                            const unsigned int & nsface_no,
    //                            ScratchData<dim> &   scratch_data,
    //                            CopyData &           copy_data) {
    //       CellIterator cellP(&dof_handler_pressure.get_triangulation(),
    //                          cell->level(),
    //                          cell->index(),
    //                          &dof_handler_pressure);


    //       const auto         ncellP          = cellP->neighbor(face_no);
    //       const unsigned int interface_index = get_interface_index({cellP->id(), ncellP->id()});
    //       const bool         this_interface_isnt_contained =
    //         interface_index == numbers::invalid_unsigned_int;
    //       if(this_interface_isnt_contained)
    //         return;

    //       FEInterfaceValues<dim> & phiV = scratch_data.fe_interface_values_test;
    //       phiV.reinit(cell, face_no, sface_no, ncell, nface_no, nsface_no);

    //       FEInterfaceValues<dim> & phiP = scratch_data.fe_interface_values_ansatz;
    //       phiP.reinit(cellP, face_no, sface_no, ncellP, nface_no, nsface_no);

    //       const unsigned int n_interface_dofs_p  = phiP.n_current_interface_dofs();
    //       const unsigned int n_interface_dofs_v  = phiV.n_current_interface_dofs();
    //       const auto &       joint_dof_indices_p = phiP.get_interface_dof_indices();
    //       // const auto &       joint_dof_indices_v = phiV.get_interface_dof_indices();
    //       // for(auto i = 0U; i < n_interface_dofs_v; ++i)
    //       //   std::cout << "interface dof " << joint_dof_indices_v[i] << " cell dofs "
    //       //             << phiV.interface_dof_to_dof_indices(i)[0] << " "
    //       //             << phiV.interface_dof_to_dof_indices(i)[1] << std::endl;

    //       const unsigned int cell_index  = get_cell_index(cell->id());
    //       const unsigned int ncell_index = get_cell_index(ncell->id());

    //       copy_data.face_data.emplace_back();
    //       CopyData::FaceData & copy_data_face = copy_data.face_data.back();

    //       copy_data_face.cell_matrix.reinit(GeometryInfo<dim>::faces_per_cell, 2U);
    //       copy_data_face.joint_dof_indices_test.resize(GeometryInfo<dim>::faces_per_cell, 0U);
    //       copy_data_face.joint_dof_indices_ansatz.resize(2U, 0U);

    //       copy_data_face.joint_dof_indices_test[face_no] = interface_index;
    //       copy_data_face.joint_dof_indices_ansatz[0U]    = cell_index;
    //       copy_data_face.joint_dof_indices_ansatz[1U]    = ncell_index;

    //       const auto & compute_v_face =
    //         [&](const auto & phi, const unsigned int face_no, const unsigned int q) {
    //           AssertIndexRange(face_no, GeometryInfo<dim>::faces_per_cell);
    //           Tensor<1, dim> value;
    //           for(auto j = 0U; j < n_face_dofs_v; ++j)
    //           {
    //             for(auto d = 0U; d < dim; ++d)
    //               value[d] +=
    //                 trafomatrix_rt_to_constp(face_no, j) * phi.shape_value_component(j, q, d);
    //           }
    //           return value;
    //         };

    //       std::vector<double> joint_dof_values_p;
    //       for(const auto i : joint_dof_indices_p)
    //         joint_dof_values_p.push_back(discrete_pressure(i));

    //       /// Note that the constant mode which isn't fixed (either on cell or on
    //       /// ncell) currently has a zero value. Therefore, it is possible to loop
    //       /// over the whole set of dof indices.
    //       const auto & compute_jump_pn = [&](const unsigned int q) {
    //         double               value = 0.;
    //         const Tensor<1, dim> n     = phiP.normal(q);
    //         for(auto j = 0U; j < n_interface_dofs_p; ++j) // skip constant mode
    //           value += joint_dof_values_p[j] * phiP.jump(j, q);
    //         return value * n;
    //       };

    //       const auto & phiV_left = phiV.get_fe_face_values(0U);
    //       // phiV_left.reinit(cell, face_no);

    //       double alpha_left  = 0.;
    //       double alpha_right = 0.;
    //       double pn_dot_v    = 0.;
    //       for(unsigned int q = 0; q < phiP.n_quadrature_points; ++q)
    //       {
    //         const auto &           dx      = phiV.JxW(q);
    //         const Tensor<1, dim> & v_face  = compute_v_face(phiV_left, face_no, q);
    //         const Tensor<1, dim> & jump_pn = compute_jump_pn(q);
    //         const Tensor<1, dim> & n_left  = phiP.normal(q);
    //         const Tensor<1, dim> & n_right = -n_left;

    //         pn_dot_v += jump_pn * v_face * dx;

    //         alpha_left += -1. * n_left * v_face * dx;
    //         alpha_right += -1. * n_right * v_face * dx;
    //       }
    //       std::cout << "interface index " << interface_index << " pn_dot_v " << pn_dot_v
    //                 << std::endl;
    //       std::cout << "interface index " << interface_index << " alpha_left " << alpha_left
    //                 << std::endl;
    //       std::cout << "interface index " << interface_index << " alpha_right " << alpha_right
    //                 << std::endl;

    //       copy_data.cell_rhs_test[face_no] += pn_dot_v;
    //       copy_data_face.cell_matrix(face_no, 0U) = alpha_left;
    //       copy_data_face.cell_matrix(face_no, 1U) = alpha_right;
    //     };

    //     const auto copier = [&](const CopyData & copy_data) {
    //       constraints_on_interface.template distribute_local_to_global<Vector<double>>(
    //         copy_data.cell_rhs_test, copy_data.local_dof_indices_test, right_hand_side);

    //       for(const auto & cdf : copy_data.face_data)
    //         constraints_on_interface.template distribute_local_to_global<SparseMatrix<double>>(
    //           cdf.cell_matrix,
    //           cdf.joint_dof_indices_test,
    //           constraints_on_cell,
    //           cdf.joint_dof_indices_ansatz,
    //           constant_pressure_matrix);
    //     };

    //     const UpdateFlags update_flags =
    //       update_values | update_quadrature_points | update_JxW_values;
    //     const UpdateFlags update_flags_pressure = update_default;
    //     const UpdateFlags interface_update_flags =
    //       update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
    //     const UpdateFlags interface_update_flags_pressure =
    //       update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

    //     ScratchData<dim> scratch_data(stokes_problem.mapping,
    //                                   stokes_problem.dof_handler_velocity.get_fe(),
    //                                   stokes_problem.dof_handler_pressure.get_fe(),
    //                                   n_q_points_1d,
    //                                   update_flags,
    //                                   update_flags_pressure,
    //                                   interface_update_flags,
    //                                   interface_update_flags_pressure);

    //     CopyData copy_data(GeometryInfo<dim>::faces_per_cell, 1U);

    //     MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
    //                           dof_handler_velocity.end(),
    //                           cell_worker,
    //                           copier,
    //                           scratch_data,
    //                           copy_data,
    //                           MeshWorker::assemble_own_cells |
    //                             MeshWorker::assemble_own_interior_faces_once,
    //                           nullptr,
    //                           face_worker);
    //   }

    //   right_hand_side.print(std::cout);
    //   constant_pressure_matrix.print_formatted(std::cout);

    //   constraints_on_interface.condense(constant_pressure_matrix, right_hand_side);

    //   right_hand_side.print(std::cout);
    //   constant_pressure_matrix.print_formatted(std::cout);

    //   Vector<double> pressure_solution(n_cells);

    //   SparseDirectUMFPACK A_direct;
    //   A_direct.template initialize<SparseMatrix<double>>(constant_pressure_matrix);
    //   A_direct.vmult(pressure_solution, right_hand_side);

    //   constraints_on_interface.distribute(pressure_solution);
    //   pressure_solution.print(std::cout);

    //   for(auto cell_index = 0U; cell_index < n_cells; ++cell_index)
    //   {
    //     const auto dof_index         = constant_pressure_dof_indices[cell_index];
    //     discrete_pressure(dof_index) = pressure_solution[cell_index];
    //   }

    //   discrete_pressure.print(std::cout);

    //   stokes_problem.correct_mean_value_pressure();
    //   discrete_pressure.print(std::cout);
    // }

    // stokes_problem.output_results(numbers::invalid_unsigned_int);
  }

  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }

  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
