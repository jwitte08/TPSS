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
#include "multigrid.h"
#include "postprocess.h"
#include "rt_parameter.h"



namespace Biharmonic
{
using namespace dealii;

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
  CopyData(const unsigned int dofs_per_cell,
           const unsigned int level_in = numbers::invalid_unsigned_int)
    : level(level_in),
      cell_matrix(dofs_per_cell, dofs_per_cell),
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

  unsigned int                         level;
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<FaceData>                face_data;
};



template<int dim>
class ModelProblem
{
public:
  using VECTOR = Vector<double>;

  using MG_TRANSFER           = MGTransferPrebuilt<VECTOR>;
  using GAUSS_SEIDEL_SMOOTHER = PreconditionSOR<SparseMatrix<double>>;
  using GMG_PRECONDITIONER    = PreconditionMG<dim, VECTOR, MG_TRANSFER>;

  static_assert(dim == 2, "only implemented in 2D");

  ModelProblem(const RT::Parameter & rt_parameters_in, const unsigned int fe_degree);

  void
  run();

  void
  make_grid();

  void
  setup_system();

  template<typename IteratorType>
  void
  cell_worker_impl(const IteratorType & cell,
                   ScratchData<dim> &   scratch_data,
                   CopyData &           copy_data) const;

  template<typename IteratorType>
  void
  face_worker_impl(const IteratorType & cell,
                   const unsigned int & f,
                   const unsigned int & sf,
                   const IteratorType & ncell,
                   const unsigned int & nf,
                   const unsigned int & nsf,
                   ScratchData<dim> &   scratch_data,
                   CopyData &           copy_data) const;

  template<typename IteratorType>
  void
  boundary_worker_impl(const IteratorType & cell,
                       const unsigned int & face_no,
                       ScratchData<dim> &   scratch_data,
                       CopyData &           copy_data) const;

  void
  assemble_system();

  void
  prepare_multigrid();

  const GMG_PRECONDITIONER &
  prepare_preconditioner_mg()
  {
    prepare_multigrid();
    AssertThrow(multigrid, ExcNotInitialized());

    preconditioner_mg = std::make_shared<GMG_PRECONDITIONER>(dof_handler, *multigrid, mg_transfer);
    return *preconditioner_mg;
  }

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner);

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

  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const
  {
    *pcout << Util::parameter_to_fstring(description, value);
  }

  void
  print_informations() const
  {
    print_parameter("Finite element:", fe.get_name());
    *pcout << rt_parameters.to_string();
    *pcout << std::endl;
  }

  RT::Parameter                       rt_parameters;
  std::shared_ptr<ConditionalOStream> pcout;
  mutable PostProcessData             pp_data;

  parallel::distributed::Triangulation<dim> triangulation;
  MappingQ<dim>                             mapping;
  FE_Q<dim>                                 fe;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<double>                 constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  VECTOR               system_u;
  VECTOR               system_rhs;

  // *** multigrid
  std::shared_ptr<MGConstrainedDoFs>  mg_constrained_dofs;
  MG_TRANSFER                         mg_transfer;
  MGLevelObject<SparsityPattern>      mg_sparsity_patterns;
  MGLevelObject<SparseMatrix<double>> mg_matrices;
  // mutable std::shared_ptr<ColoringBase<dim>> user_coloring;
  // std::shared_ptr<const MG_SMOOTHER_SCHWARZ> mg_schwarz_smoother_pre;
  // std::shared_ptr<const MG_SMOOTHER_SCHWARZ> mg_schwarz_smoother_post;
  std::shared_ptr<const MGSmootherIdentity<VECTOR>> mg_smoother_identity;
  std::shared_ptr<const mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER, VECTOR>>
                                                 mg_smoother_gauss_seidel;
  const MGSmootherBase<VECTOR> *                 mg_smoother_pre;
  const MGSmootherBase<VECTOR> *                 mg_smoother_post;
  CoarseGridSolver<SparseMatrix<double>, VECTOR> coarse_grid_solver;
  MGCoarseGridSVD<double, VECTOR>                coarse_grid_svd;
  const MGCoarseGridBase<VECTOR> *               mg_coarse_grid;
  mg::Matrix<VECTOR>                             mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>>             multigrid;

  std::shared_ptr<GMG_PRECONDITIONER> preconditioner_mg;
};



template<int dim>
ModelProblem<dim>::ModelProblem(const RT::Parameter & rt_parameters_in,
                                const unsigned int    fe_degree)
  : rt_parameters(rt_parameters_in),
    pcout(
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
  *pcout << create_mesh(triangulation, rt_parameters.mesh);
  // this->level = triangulation.n_global_levels() - 1;
  pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());
  pp_data.n_dimensions = dim;

  // GridGenerator::hyper_cube(triangulation, 0., 1.);
  // triangulation.refine_global(1);

  // *pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
  //        << "Total number of cells: " << triangulation.n_cells() << std::endl;
}



template<int dim>
void
ModelProblem<dim>::setup_system()
{
  constraints.clear();
  system_u.reinit(0);
  system_rhs.reinit(0);
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
  const bool             direct_solver_used = (rt_parameters.solver.variant == "direct");
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       direct_solver_used ? true : false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  system_u.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template<int dim>
template<typename IteratorType>
void
ModelProblem<dim>::cell_worker_impl(const IteratorType & cell,
                                    ScratchData<dim> &   scratch_data,
                                    CopyData &           copy_data) const
{
  copy_data.cell_matrix = 0;
  copy_data.cell_rhs    = 0;

  FEValues<dim> & fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);

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
}


template<int dim>
template<typename IteratorType>
void
ModelProblem<dim>::face_worker_impl(const IteratorType & cell,
                                    const unsigned int & f,
                                    const unsigned int & sf,
                                    const IteratorType & ncell,
                                    const unsigned int & nf,
                                    const unsigned int & nsf,
                                    ScratchData<dim> &   scratch_data,
                                    CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
  fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

  const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();
  copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

  const unsigned int p            = fe.degree;
  const double       gamma_over_h = std::max(
    (1.0 * p * (p + 1) / cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f])),
    (1.0 * p * (p + 1) / ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf])));

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

        copy_data_face.cell_matrix(i, j) += (-av_hessian_i_dot_n_dot_n       // - {grad^2 v n n
                                                                             //
                                               * jump_grad_j_dot_n           // [grad u n]
                                             - av_hessian_j_dot_n_dot_n      // - {grad^2 u n n
                                                                             //
                                                 * jump_grad_i_dot_n         // [grad v n]
                                             +                               // +
                                             gamma_over_h *                  // gamma/h
                                               jump_grad_i_dot_n *           // [grad v n]
                                               jump_grad_j_dot_n) *          // [grad u n]
                                            fe_interface_values.JxW(qpoint); // dx
      }
    }
  }
}



template<int dim>
template<typename IteratorType>
void
ModelProblem<dim>::boundary_worker_impl(const IteratorType & cell,
                                        const unsigned int & face_no,
                                        ScratchData<dim> &   scratch_data,
                                        CopyData &           copy_data) const
{
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
                                             + (2. * gamma_over_h)      //  gamma/h
                                                 * jump_grad_i_dot_n    // [grad v n]
                                                 * jump_grad_j_dot_n    // [grad u n]
                                             ) *
                                            JxW[qpoint]; // dx
      }

      copy_data.cell_rhs(i) += (-av_hessian_i_dot_n_dot_n *       // - {grad^2 v n n }
                                  (exact_gradients[qpoint] * n)   //   (grad u_exact . n)
                                +                                 // +
                                (2. * gamma_over_h)               //  gamma/h
                                  * jump_grad_i_dot_n             // [grad v n]
                                  * (exact_gradients[qpoint] * n) // (grad u_exact . n)
                                ) *
                               JxW[qpoint]; // dx
    }
  }
}



template<int dim>
void
ModelProblem<dim>::assemble_system()
{
  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    cell_worker_impl(cell, scratch_data, copy_data);
  };
  auto face_worker = [&](const auto &         cell,
                         const unsigned int & f,
                         const unsigned int & sf,
                         const auto &         ncell,
                         const unsigned int & nf,
                         const unsigned int & nsf,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) {
    face_worker_impl(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
  };
  auto boundary_worker = [&](const auto &         cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
    boundary_worker_impl(cell, face_no, scratch_data, copy_data);
  };
  const auto copier = [&](const CopyData & copy_data) {
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
ModelProblem<dim>::prepare_multigrid()
{
  // *** clear multigrid infrastructure
  multigrid.reset();
  mg_matrix_wrapper.reset();
  coarse_grid_solver.clear();
  mg_smoother_post = nullptr;
  mg_smoother_pre  = nullptr;
  // mg_schwarz_smoother_pre.reset();
  // mg_schwarz_smoother_post.reset();
  mg_transfer.clear();
  // mg_matrices.clear_elements();
  mg_constrained_dofs.reset();

  // *** setup multigrid data
  const unsigned mg_level_min = rt_parameters.multigrid.coarse_level;
  const unsigned mg_level_max = max_level();
  pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);

  // *** initialize multigrid constraints
  mg_constrained_dofs = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs->initialize(dof_handler);
  mg_constrained_dofs->make_zero_boundary_constraints(dof_handler, std::set<types::boundary_id>{0});

  // *** initialize level matrices A_l
  // system_matrix.print_formatted(std::cout);
  mg_matrices.resize(mg_level_min, mg_level_max);
  mg_sparsity_patterns.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(level), dof_handler.n_dofs(level));
    MGTools::make_sparsity_pattern(dof_handler, dsp, level);
    mg_sparsity_patterns[level].copy_from(dsp);
    mg_matrices[level].reinit(mg_sparsity_patterns[level]);

    // assemble
    AffineConstraints<double> level_constraints;
    IndexSet                  relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        cell_worker_impl(cell, scratch_data, copy_data);
      };
    auto face_worker = [&](const auto &         cell,
                           const unsigned int & f,
                           const unsigned int & sf,
                           const auto &         ncell,
                           const unsigned int & nf,
                           const unsigned int & nsf,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      face_worker_impl(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
    };
    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      boundary_worker_impl(cell, face_no, scratch_data, copy_data);
    };
    const auto copier = [&](const CopyData & copy_data) {
      AssertDimension(copy_data.level, level);
      level_constraints.distribute_local_to_global(copy_data.cell_matrix,
                                                   copy_data.local_dof_indices,
                                                   mg_matrices[copy_data.level]);

      for(auto & cdf : copy_data.face_data)
      {
        level_constraints.distribute_local_to_global(cdf.cell_matrix,
                                                     cdf.joint_dof_indices,
                                                     mg_matrices[copy_data.level]);
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
    CopyData           copy_data(dof_handler.get_fe().dofs_per_cell, level);
    MeshWorker::mesh_loop(dof_handler.begin_mg(level),
                          dof_handler.end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);

    // mg_matrices[level].print_formatted(std::cout);
  }

  // *** initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  // *** initialize Schwarz smoother S_l
  // prepare_mg_smoothers();

  // *** set pre- and post-smoother
  // pp_data.n_colors_system.push_back(n_colors_system());
  switch(rt_parameters.multigrid.pre_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      mg_smoother_identity = std::make_shared<const MGSmootherIdentity<VECTOR>>();
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::GaussSeidel:
    {
      auto tmp = std::make_shared<mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER, VECTOR>>();
      tmp->initialize(mg_matrices);
      tmp->set_steps(rt_parameters.multigrid.pre_smoother.n_smoothing_steps);
      tmp->set_symmetric(true);
      mg_smoother_gauss_seidel = tmp;
      mg_smoother_pre          = mg_smoother_gauss_seidel.get();
    }
    break;
    case SmootherParameter::SmootherVariant::Schwarz:
      AssertThrow(false, ExcMessage("TODO !!!"));
      // AssertThrow(mg_schwarz_smoother_pre);
      // mg_smoother_pre      = mg_schwarz_smoother_pre.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }
  switch(rt_parameters.multigrid.post_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_post = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::GaussSeidel:
      AssertThrow(mg_smoother_gauss_seidel, ExcMessage("Not initialized."));
      mg_smoother_post = mg_smoother_gauss_seidel.get();
      break;
    case SmootherParameter::SmootherVariant::Schwarz:
      AssertThrow(false, ExcMessage("TODO !!!"));
      // AssertThrow(mg_schwarz_smoother_post);
      // mg_smoother_post      = mg_schwarz_smoother_post.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }

  // *** initialize coarse grid solver
  // coarse_grid_solver.initialize(mg_matrices[mg_level_min], rt_parameters.multigrid.coarse_grid);
  // mg_coarse_grid = &coarse_grid_solver;
  FullMatrix<double> coarse_matrix(mg_matrices[mg_level_min].m());
  coarse_matrix.copy_from(mg_matrices[mg_level_min]);
  coarse_grid_svd.initialize(coarse_matrix);
  // coarse_matrix.print_formatted(std::cout);
  mg_coarse_grid = &coarse_grid_svd;

  mg_matrix_wrapper.initialize(mg_matrices);
  multigrid = std::make_shared<Multigrid<VECTOR>>(mg_matrix_wrapper,
                                                  *mg_coarse_grid,
                                                  mg_transfer,
                                                  *mg_smoother_pre,
                                                  *mg_smoother_post,
                                                  mg_level_min,
                                                  mg_level_max);
}



template<int dim>
template<typename PreconditionerType>
void
ModelProblem<dim>::iterative_solve_impl(const PreconditionerType & preconditioner)
{
  // ReductionControl       solver_control;
  // solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  // solver_control.set_reduction(rt_parameters.solver.rel_tolerance);
  // solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  // solver_control.log_history(true);
  // solver_control.log_result(true);
  // solver_control.enable_history_data();

  IterationNumberControl solver_control;
  solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  solver_control.log_history(true);
  solver_control.log_result(true);
  solver_control.enable_history_data();

  SolverSelector<VECTOR> iterative_solver;
  iterative_solver.set_control(solver_control);
  iterative_solver.select(rt_parameters.solver.variant);
  iterative_solver.solve(system_matrix, system_u, system_rhs, preconditioner);
  constraints.distribute(system_u);

  const auto [n_frac, reduction_rate] = compute_fractional_steps(solver_control);
  pp_data.average_reduction_system.push_back(reduction_rate);
  pp_data.n_iterations_system.push_back(n_frac);
}



template<int dim>
void
ModelProblem<dim>::solve()
{
  *pcout << "   Solving system..." << std::endl;

  if(rt_parameters.solver.variant == "direct")
  {
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(system_u, system_rhs);
    return;
  }

  else
  {
    switch(rt_parameters.solver.precondition_variant)
    {
      case SolverParameter::PreconditionVariant::None:
        iterative_solve_impl(PreconditionIdentity{});
        break;

      case SolverParameter::PreconditionVariant::GMG:
        prepare_preconditioner_mg();
        iterative_solve_impl(*preconditioner_mg);
        break;

      default:
        AssertThrow(false, ExcNotImplemented());
    }
  }
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
                                      system_u,
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
                                      system_u,
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
      fe_values[scalar].get_function_hessians(system_u, hessians);
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
  data_out.add_data_vector(system_u, "u");
  data_out.build_patches();

  std::ofstream output_vtu(("output_" + Utilities::int_to_string(iteration, 6) + ".vtu").c_str());
  data_out.write_vtu(output_vtu);
}



template<int dim>
void
ModelProblem<dim>::run()
{
  print_informations();
  make_grid();

  const unsigned int n_cycles = rt_parameters.n_cycles;
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
