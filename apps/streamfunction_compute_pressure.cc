/*
 * streamfunction_compute_pressure.cc
 *
 *  Created on: Sep 25, 2020
 *      Author: witte
 */

#include "app_utilities.h"
#include "biharmonic_problem.h"
#include "ct_parameter.h"
#include "stokes.h"



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
    unsigned int test_index        = 0;
    unsigned int debug_depth       = 3;
    double       damping           = 0.;
    unsigned int pde_index         = 3;
    unsigned int test_index_stokes = 0;
    unsigned int pde_index_stokes  = 4;

    //: parse arguments
    atoi_if(test_index, 1);
    // atoi_if(pde_index, 2);
    // atof_if(ip_factor, 3);
    // atoi_if(debug_depth, 4);
    // atof_if(damping, 5);

    deallog.depth_console(debug_depth);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    constexpr int  dim                = CT::DIMENSION_;
    constexpr int  fe_degree_stream   = CT::FE_DEGREE_;
    constexpr int  fe_degree_pressure = fe_degree_stream - 1;
    constexpr auto patch_variant      = CT::PATCH_VARIANT_;
    constexpr auto smoother_variant   = CT::SMOOTHER_VARIANT_;

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
      prms.n_cycles              = 1;
      prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
      prms.mesh.n_refinements    = 1;
      prms.mesh.n_repetitions    = 1;

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
      prms.multigrid.coarse_grid.solver_variant   = CoarseGridParameter::SolverVariant::FullSVD;
      prms.multigrid.coarse_grid.iterative_solver = "cg";
      prms.multigrid.coarse_grid.accuracy         = 1.e-12;
      const SmootherParameter::SmootherVariant smoother_variant[test_index_max + 1] = {
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::GaussSeidel,
        SmootherParameter::SmootherVariant::Schwarz};
      prms.multigrid.pre_smoother.variant                    = smoother_variant[test_index];
      prms.multigrid.pre_smoother.n_smoothing_steps          = 2;
      prms.multigrid.pre_smoother.schwarz.patch_variant      = CT::PATCH_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.smoother_variant   = CT::SMOOTHER_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.manual_coloring    = true;
      prms.multigrid.pre_smoother.schwarz.damping_factor     = damping;
      prms.multigrid.post_smoother                           = prms.multigrid.pre_smoother;
      prms.multigrid.post_smoother.schwarz.reverse_smoothing = true;
    }

    Biharmonic::EquationData equation_data;
    AssertThrow(pde_index < Biharmonic::EquationData::n_variants,
                ExcMessage("This equation is not implemented."));
    equation_data.variant = static_cast<Biharmonic::EquationData::Variant>(pde_index);

    Biharmonic::ModelProblem<dim, fe_degree_stream> biharmonic_problem(prms, equation_data);
    auto pcout               = std::make_shared<ConditionalOStream>(std::cout, true);
    biharmonic_problem.pcout = pcout;

    Stokes::StokesFlow options_stokes;
    if(damping == 0.)
      damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    options_stokes.setup(test_index_stokes, damping);
    options_stokes.prms.n_cycles = prms.n_cycles;
    options_stokes.prms.mesh     = prms.mesh;

    Stokes::EquationData equation_data_stokes;
    equation_data_stokes.variant = static_cast<Stokes::EquationData::Variant>(pde_index_stokes);
    equation_data_stokes.use_cuthill_mckee = false;
    if(options_stokes.prms.solver.variant == "GMRES_GMG" ||
       options_stokes.prms.solver.variant == "CG_GMG")
      equation_data_stokes.local_kernel_size = 1U;
    if(options_stokes.prms.solver.variant == "UMFPACK")
      equation_data_stokes.force_mean_value_constraint = true;

    Stokes::ModelProblem<dim, fe_degree_pressure, Stokes::Method::RaviartThomas> stokes_problem(
      options_stokes.prms, equation_data_stokes);

    std::cout << std::endl;
    biharmonic_problem.make_grid();
    biharmonic_problem.setup_system();

    std::cout << std::endl;
    stokes_problem.make_grid();
    stokes_problem.setup_system();

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

    const auto n_q_points_1d     = stokes_problem.n_q_points_1d;
    const auto n_dofs_per_cell_v = fe_v.dofs_per_cell;
    const auto n_dofs_per_cell_p = fe_p.dofs_per_cell;

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

    AssertDimension(dim, 2U);
    const auto n_faces_per_cell  = GeometryInfo<dim>::faces_per_cell;
    const auto n_interior_dofs_v = fe_v.dofs_per_quad;
    const auto n_face_dofs_v     = n_faces_per_cell * fe_v.dofs_per_face;
    AssertDimension(n_interior_dofs_v + n_face_dofs_v, n_dofs_per_cell_v);
    const auto n_interior_nodes_by_pressure = n_dofs_per_cell_p - 1;

    LAPACKFullMatrix<double> trafomatrix_rt_to_gradp(n_interior_nodes_by_pressure,
                                                     n_interior_dofs_v);

    LAPACKFullMatrix<double> trafomatrix_rt_to_constp(n_faces_per_cell, n_face_dofs_v);

    /// Compute the application of the current RT shape functions to the interior RT
    /// node functionals, which are generated by the gradients of the pressure shape
    /// functions (the constant pressure mode is neglected!).
    {
      QGauss<dim>       quad(n_q_points_1d);
      const UpdateFlags update_flags =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      FEValues<dim> fe_values_p(stokes_problem.mapping, fe_p, quad, update_flags);
      fe_values_p.reinit(unit_dofh_p.begin_active()); // one reference cell

      FEValues<dim> fe_values_v(stokes_problem.mapping, fe_v, quad, update_flags);
      fe_values_v.reinit(unit_dofh_v.begin_active()); // one reference cell

      const auto interior_node_functional = [&](const unsigned int   node_index_p,
                                                const Vector<double> v) {
        const auto n_q_points  = fe_values_p.n_quadrature_points;
        const auto dof_index_p = node_index_p + 1; // skip constant mode
        AssertIndexRange(dof_index_p, fe_values_p.dofs_per_cell);
        AssertDimension(v.size(), fe_values_v.dofs_per_cell);

        const auto & compute_v_q = [&](const unsigned int q) {
          Tensor<1, dim> value;
          for(unsigned int j = 0; j < n_dofs_per_cell_v; ++j)
            value += v[j] * MW::compute_vvalue(fe_values_v, j, q);
          return value;
        };

        double node_value = 0.;
        for(unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<1, dim> & grad_q = fe_values_p.shape_grad(dof_index_p, q);
          const Tensor<1, dim> & v_q    = compute_v_q(q);
          const auto &           dx     = fe_values_p.JxW(q);

          node_value += v_q * grad_q * dx;
        }

        return node_value;
      };

      LAPACKFullMatrix<double> node_value_weights(n_interior_nodes_by_pressure, n_interior_dofs_v);
      for(auto i = 0U; i < node_value_weights.m(); ++i)   // node functionals
        for(auto j = 0U; j < node_value_weights.n(); ++j) // interior shape funcs
        {
          Vector<double> phi_j(n_dofs_per_cell_v);
          const auto     dof_index = n_face_dofs_v + j; // shift to interior dofs
          phi_j[dof_index]         = 1.;
          node_value_weights(i, j) = interior_node_functional(i, phi_j);
        }

      // LAPACKFullMatrix<double> copy(node_value_weights);
      // copy.print_formatted(std::cout);
      // LAPACKFullMatrix<double> copy2(node_value_weights);

      // const auto & [U, Sigma, VT] = compute_svd(copy);
      // Sigma.print_formatted(std::cout);

      const auto & [V, invSigma, UT] = compute_inverse_svd(node_value_weights);
      // invSigma.print_formatted(std::cout);

      /// Note that inverse means in the sense of an inverse SVD.
      const auto & inverse_node_value_weights = merge_lapack_decomposition(V, invSigma, UT);
      inverse_node_value_weights.transpose(trafomatrix_rt_to_gradp);

      // LAPACKFullMatrix<double> AVinvSigmaUT(copy2.m(), UT.n());
      // copy2.mmult(AVinvSigmaUT, VinvSigmaUT);
      // AVinvSigmaUT.print_formatted(std::cout);

      /// Display the "new shape functions" \tilde{v}_i
      for(auto i = 0U; i < n_interior_nodes_by_pressure; ++i)
      {
        Vector<double> phi_i(n_dofs_per_cell_v);
        for(auto j = 0U; j < n_interior_dofs_v; ++j)
        {
          const auto dof_index = n_face_dofs_v + j; // shift to interior dofs
          phi_i(dof_index)     = trafomatrix_rt_to_gradp(i, j);
        }

        std::vector<std::string> names(dim, "shape_function");
        const std::string        prefix         = "tildev_interior";
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

    /// Compute the application of the given RT shape functions to the RT node
    /// functional associated to a given face, which is generated by the constant function.
    {
      QGauss<dim - 1>   quad(n_q_points_1d);
      const UpdateFlags update_flags =
        update_values | update_normal_vectors | update_quadrature_points | update_JxW_values;
      FEFaceValues<dim> fe_values_v(stokes_problem.mapping, fe_v, quad, update_flags);

      const auto face_node_functional = [&](const unsigned int face_no, const Vector<double> v) {
        const auto cell_v = unit_dofh_v.begin_active(); // one reference cell
        fe_values_v.reinit(cell_v, face_no);

        const auto   n_q_points = fe_values_v.n_quadrature_points;
        const auto & normals    = fe_values_v.get_normal_vectors();
        AssertDimension(v.size(), fe_values_v.dofs_per_cell);

        const auto & compute_v_q = [&](const unsigned int q) {
          Tensor<1, dim> value;
          for(unsigned int j = 0; j < n_dofs_per_cell_v; ++j)
            value += v[j] * MW::compute_vvalue(fe_values_v, j, q);
          return value;
        };

        double node_value = 0.;
        for(unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<1, dim> & n_q = normals[q];
          const Tensor<1, dim> & v_q = compute_v_q(q);
          const auto &           dx  = fe_values_v.JxW(q);

          node_value += v_q * n_q * dx;
        }

        return node_value;
      };

      LAPACKFullMatrix<double> node_value_weights(n_faces_per_cell, n_face_dofs_v);
      for(auto i = 0U; i < node_value_weights.m(); ++i)   // node functionals
        for(auto j = 0U; j < node_value_weights.n(); ++j) // face shape funcs
        {
          Vector<double> phi_j(n_dofs_per_cell_v);
          phi_j[j]                 = 1.;
          node_value_weights(i, j) = face_node_functional(/*face_no*/ i, phi_j);
        }

      const auto & [V, invSigma, UT]          = compute_inverse_svd(node_value_weights);
      const auto & inverse_node_value_weights = merge_lapack_decomposition(V, invSigma, UT);
      inverse_node_value_weights.transpose(trafomatrix_rt_to_constp);

      /// Display the "new shape functions" \tilde{v}_i
      for(auto i = 0U; i < n_faces_per_cell; ++i)
      {
        Vector<double> phi_i(n_dofs_per_cell_v);
        for(auto j = 0U; j < n_face_dofs_v; ++j)
          phi_i(j) = trafomatrix_rt_to_constp(i, j);

        std::vector<std::string> names(dim, "shape_function");
        const std::string        prefix         = "tildev_face";
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
    }
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
