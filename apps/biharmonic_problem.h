#ifndef APPS_BIHARMONICPROBLEM_H_
#define APPS_BIHARMONICPROBLEM_H_

/* ---------------------------------------------------------------------
 * based on step-47
 *
 * Copyright (C) 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Natasha Sharma, University of Texas at El Paso,
 *          Guido Kanschat, University of Heidelberg
 *          Timo Heister, Clemson University
 *          Wolfgang Bangerth, Colorado State University
 *          Zhuroan Wang, Colorado State University
 */


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/meshworker/mesh_loop.h>


#include <cmath>
#include <fstream>
#include <iostream>


#include "biharmonic_integrator.h"
#include "equation_data.h"
#include "mesh.h"
#include "rt_parameter.h"



namespace Biharmonic
{
using namespace dealii;

template<int dim>
class ModelProblem
{
public:
  static_assert(dim == 2, "only implemented in 2D");

  ModelProblem(const unsigned int fe_degree);

  void
  run();

  void
  make_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  compute_errors();
  void
  output_results(const unsigned int iteration) const;

  unsigned int
  max_level() const
  {
    return triangulation.n_global_levels() - 1;
  }

  RT::Parameter                       rt_parameters;
  std::shared_ptr<ConditionalOStream> pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  MappingQ<dim>                             mapping;
  FE_Q<dim>                                 fe;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<double>                 constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};



template<int dim>
ModelProblem<dim>::ModelProblem(const unsigned int fe_degree)
  : pcout(
      std::make_shared<ConditionalOStream>(std::cout,
                                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    mapping(1),
    fe(fe_degree)
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("One process only."));
}



template<int dim>
void
ModelProblem<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(1);

  *pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
         << "Total number of cells: " << triangulation.n_cells() << std::endl;
}



template<int dim>
void
ModelProblem<dim>::setup_system()
{
  dof_handler.clear();
  dof_handler.initialize(triangulation, fe);
  dof_handler.distribute_mg_dofs();

  *pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           ExactSolution::Solution<dim>(),
                                           constraints);
  constraints.close();


  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints, true);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template<int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe,
              const unsigned int         quadrature_degree,
              const UpdateFlags          update_flags,
              const UpdateFlags          interface_update_flags)
    : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags),
      fe_interface_values(mapping, fe, QGauss<dim - 1>(quadrature_degree), interface_update_flags)
  {
  }


  ScratchData(const ScratchData<dim> & scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags()),
      fe_interface_values(scratch_data.fe_values.get_mapping(),
                          scratch_data.fe_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {
  }

  FEValues<dim>          fe_values;
  FEInterfaceValues<dim> fe_interface_values;
};



struct CopyData
{
  CopyData(const unsigned int dofs_per_cell)
    : cell_matrix(dofs_per_cell, dofs_per_cell),
      cell_rhs(dofs_per_cell),
      local_dof_indices(dofs_per_cell)
  {
  }


  CopyData(const CopyData &) = default;


  struct FaceData
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };

  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<FaceData>                face_data;
};



template<int dim>
void
ModelProblem<dim>::assemble_system()
{
  using Iterator = typename DoFHandler<dim>::active_cell_iterator;

  auto cell_worker = [&](const Iterator &   cell,
                         ScratchData<dim> & scratch_data,
                         CopyData &         copy_data) {
    copy_data.cell_matrix = 0;
    copy_data.cell_rhs    = 0;

    FEValues<dim> & fe_values = scratch_data.fe_values;
    fe_values.reinit(cell);

    cell->get_dof_indices(copy_data.local_dof_indices);

    const ExactSolution::RightHandSide<dim> right_hand_side;

    const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;

    for(unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points; ++qpoint)
    {
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const Tensor<2, dim> hessian_i = fe_values.shape_hessian(i, qpoint);

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const Tensor<2, dim> hessian_j = fe_values.shape_hessian(j, qpoint);

          copy_data.cell_matrix(i, j) += scalar_product(hessian_i,   // nabla^2 phi_i(x)
                                                        hessian_j) * // nabla^2 phi_j(x)
                                         fe_values.JxW(qpoint);      // dx
        }

        copy_data.cell_rhs(i) += fe_values.shape_value(i, qpoint) * // phi_i(x)
                                 right_hand_side.value(fe_values.quadrature_point(qpoint)) * // f(x)
                                 fe_values.JxW(qpoint);                                      // dx
      }
    }
  };


  auto face_worker = [&](const Iterator &     cell,
                         const unsigned int & f,
                         const unsigned int & sf,
                         const Iterator &     ncell,
                         const unsigned int & nf,
                         const unsigned int & nsf,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) {
    FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
    fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

    copy_data.face_data.emplace_back();
    CopyData::FaceData & copy_data_face = copy_data.face_data.back();

    copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

    const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();
    copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

    const unsigned int p = fe.degree;
    const double       gamma_over_h =
      std::max((1.0 * p * (p + 1) /
                cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f])),
               (1.0 * p * (p + 1) /
                ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf])));

    for(unsigned int qpoint = 0; qpoint < fe_interface_values.n_quadrature_points; ++qpoint)
    {
      const auto & n = fe_interface_values.normal(qpoint);

      for(unsigned int i = 0; i < n_interface_dofs; ++i)
      {
        const double av_hessian_i_dot_n_dot_n =
          (fe_interface_values.average_hessian(i, qpoint) * n * n);
        const double jump_grad_i_dot_n = (fe_interface_values.jump_gradient(i, qpoint) * n);

        for(unsigned int j = 0; j < n_interface_dofs; ++j)
        {
          const double av_hessian_j_dot_n_dot_n =
            (fe_interface_values.average_hessian(j, qpoint) * n * n);
          const double jump_grad_j_dot_n = (fe_interface_values.jump_gradient(j, qpoint) * n);

          copy_data_face.cell_matrix(i, j) += (-av_hessian_i_dot_n_dot_n       // - {grad^2 v n n }
                                                 * jump_grad_j_dot_n           // [grad u n]
                                               - av_hessian_j_dot_n_dot_n      // - {grad^2 u n n }
                                                   * jump_grad_i_dot_n         // [grad v n]
                                               +                               // +
                                               gamma_over_h *                  // gamma/h
                                                 jump_grad_i_dot_n *           // [grad v n]
                                                 jump_grad_j_dot_n) *          // [grad u n]
                                              fe_interface_values.JxW(qpoint); // dx
        }
      }
    }
  };


  auto boundary_worker = [&](const Iterator &     cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
    FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
    fe_interface_values.reinit(cell, face_no);
    const auto & q_points = fe_interface_values.get_quadrature_points();

    copy_data.face_data.emplace_back();
    CopyData::FaceData & copy_data_face = copy_data.face_data.back();

    const unsigned int n_dofs        = fe_interface_values.n_current_interface_dofs();
    copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

    copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

    const std::vector<double> &         JxW     = fe_interface_values.get_JxW_values();
    const std::vector<Tensor<1, dim>> & normals = fe_interface_values.get_normal_vectors();


    const ExactSolution::Solution<dim> exact_solution;
    std::vector<Tensor<1, dim>>        exact_gradients(q_points.size());
    exact_solution.gradient_list(q_points, exact_gradients);


    // Positively, because we now only deal with one cell adjacent to the
    // face (as we are on the boundary), the computation of the penalty
    // factor $\gamma$ is substantially simpler:
    const unsigned int p = fe.degree;
    const double       gamma_over_h =
      (1.0 * p * (p + 1) /
       cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]));

    for(unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
    {
      const auto & n = normals[qpoint];

      for(unsigned int i = 0; i < n_dofs; ++i)
      {
        const double av_hessian_i_dot_n_dot_n =
          (fe_interface_values.average_hessian(i, qpoint) * n * n);
        const double jump_grad_i_dot_n = (fe_interface_values.jump_gradient(i, qpoint) * n);

        for(unsigned int j = 0; j < n_dofs; ++j)
        {
          const double av_hessian_j_dot_n_dot_n =
            (fe_interface_values.average_hessian(j, qpoint) * n * n);
          const double jump_grad_j_dot_n = (fe_interface_values.jump_gradient(j, qpoint) * n);

          copy_data_face.cell_matrix(i, j) += (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n}
                                                 * jump_grad_j_dot_n      //   [grad u n]
                                                                          //
                                               - av_hessian_j_dot_n_dot_n // - {grad^2 u n n}
                                                   * jump_grad_i_dot_n    //   [grad v n]
                                                                          //
                                               + gamma_over_h             //  gamma/h
                                                   * jump_grad_i_dot_n    // [grad v n]
                                                   * jump_grad_j_dot_n    // [grad u n]
                                               ) *
                                              JxW[qpoint]; // dx
        }

        copy_data.cell_rhs(i) += (-av_hessian_i_dot_n_dot_n *       // - {grad^2 v n n }
                                    (exact_gradients[qpoint] * n)   //   (grad u_exact . n)
                                  +                                 // +
                                  gamma_over_h                      //  gamma/h
                                    * jump_grad_i_dot_n             // [grad v n]
                                    * (exact_gradients[qpoint] * n) // (grad u_exact . n)
                                  ) *
                                 JxW[qpoint]; // dx
      }
    }
  };

  auto copier = [&](const CopyData & copy_data) {
    constraints.distribute_local_to_global(copy_data.cell_matrix,
                                           copy_data.cell_rhs,
                                           copy_data.local_dof_indices,
                                           system_matrix,
                                           system_rhs);

    for(auto & cdf : copy_data.face_data)
    {
      constraints.distribute_local_to_global(cdf.cell_matrix, cdf.joint_dof_indices, system_matrix);
    }
  };


  const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
  ScratchData<dim>   scratch_data(mapping,
                                fe,
                                n_gauss_points,
                                update_values | update_gradients | update_hessians |
                                  update_quadrature_points | update_JxW_values,
                                update_values | update_gradients | update_hessians |
                                  update_quadrature_points | update_JxW_values |
                                  update_normal_vectors);
  CopyData           copy_data(dof_handler.get_fe().dofs_per_cell);
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}



template<int dim>
void
ModelProblem<dim>::solve()
{
  *pcout << "   Solving system..." << std::endl;

  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);

  constraints.distribute(solution);
}



// The next function evaluates the error between the computed solution
// and the exact solution (which is known here because we have chosen
// the right hand side and boundary values in a way so that we know
// the corresponding solution). In the first two code blocks below,
// we compute the error in the $L_2$ norm and the $H^1$ semi-norm.
template<int dim>
void
ModelProblem<dim>::compute_errors()
{
  {
    Vector<float> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      ExactSolution::Solution<dim>(),
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    const double error_norm =
      VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::L2_norm);
    *pcout << "   Error in the L2 norm       :     " << error_norm << std::endl;
  }

  {
    Vector<float> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      ExactSolution::Solution<dim>(),
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::H1_seminorm);
    const double error_norm =
      VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::H1_seminorm);
    *pcout << "   Error in the H1 seminorm       : " << error_norm << std::endl;
  }

  // Now also compute an approximation to the $H^2$ seminorm error. The actual
  // $H^2$ seminorm would require us to integrate second derivatives of the
  // solution $u_h$, but given the Lagrange shape functions we use, $u_h$ of
  // course has kinks at the interfaces between cells, and consequently second
  // derivatives are singular at interfaces. As a consequence, we really only
  // integrating over the interiors of the cells and ignore the interface
  // contributions. This is *not* an equivalent norm to the energy norm for
  // the problem, but still gives us an idea of how fast the error converges.
  {
    const QGauss<dim>            quadrature_formula(fe.degree + 2);
    ExactSolution::Solution<dim> exact_solution;
    Vector<double>               error_per_cell(triangulation.n_active_cells());

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_hessians | update_quadrature_points |
                              update_JxW_values);

    FEValuesExtractors::Scalar scalar(0);
    const unsigned int         n_q_points = quadrature_formula.size();

    std::vector<SymmetricTensor<2, dim>> exact_hessians(n_q_points);
    std::vector<Tensor<2, dim>>          hessians(n_q_points);
    for(auto & cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[scalar].get_function_hessians(solution, hessians);
      exact_solution.hessian_list(fe_values.get_quadrature_points(), exact_hessians);

      double local_error = 0;
      for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        local_error +=
          ((exact_hessians[q_point] - hessians[q_point]).norm_square() * fe_values.JxW(q_point));
      }
      error_per_cell[cell->active_cell_index()] = std::sqrt(local_error);
    }

    const double error_norm = error_per_cell.l2_norm();
    *pcout << "   Error in the broken H2 seminorm: " << error_norm << std::endl;
  }
}



template<int dim>
void
ModelProblem<dim>::output_results(const unsigned int iteration) const
{
  *pcout << "   Writing graphical output..." << std::endl;

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output_vtu(("output_" + Utilities::int_to_string(iteration, 6) + ".vtu").c_str());
  data_out.write_vtu(output_vtu);
}



template<int dim>
void
ModelProblem<dim>::run()
{
  make_grid();

  const unsigned int n_cycles = 4;
  for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    *pcout << "Cycle: " << cycle << " of " << n_cycles << std::endl;

    triangulation.refine_global(1);
    setup_system();

    assemble_system();
    solve();

    output_results(cycle);

    compute_errors();
    *pcout << std::endl;
  }
}



} // namespace Biharmonic

#endif // APPS_BIHARMONICPROBLEM_H_
