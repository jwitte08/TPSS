#ifndef APPS_STOKESPROBLEM_H_
#define APPS_STOKESPROBLEM_H_

/**
 * Stokes...
 *
 * Created on: May 18, 2020
 *     Author: witte
 */


/* ---------------------------------------------------------------------
 * based on step-56
 *
 * Copyright (C) 2016 - 2020 by the deal.II authors
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

 * Author: Ryan Grove, Clemson University
 *         Timo Heister, Clemson University
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/sparse_ilu.h>

// We need to include the following file to do timings:
#include <deal.II/base/timer.h>

// // This includes the files necessary for us to use geometric Multigrid
// #include <deal.II/multigrid/multigrid.h>
// #include <deal.II/multigrid/mg_transfer.h>
// #include <deal.II/multigrid/mg_tools.h>
// #include <deal.II/multigrid/mg_coarse.h>
// #include <deal.II/multigrid/mg_smoother.h>
// #include <deal.II/multigrid/mg_matrix.h>

#include <fstream>
#include <iostream>


#include "equation_data.h"
#include "multigrid.h"
#include "postprocess.h"
#include "rt_parameter.h"



namespace Stokes
{
using namespace dealii;

template<class PreconditionerAType, class PreconditionerSType>
class BlockSchurPreconditioner : public Subscriptor
{
public:
  BlockSchurPreconditioner(const BlockSparseMatrix<double> & system_matrix,
                           const SparseMatrix<double> &      schur_complement_matrix,
                           const PreconditionerAType &       preconditioner_A,
                           const PreconditionerSType &       preconditioner_S,
                           const bool                        do_solve_A);

  void
  vmult(BlockVector<double> & dst, const BlockVector<double> & src) const;

  mutable unsigned int n_iterations_A;
  mutable unsigned int n_iterations_S;

private:
  const BlockSparseMatrix<double> & system_matrix;
  const SparseMatrix<double> &      schur_complement_matrix;
  const PreconditionerAType &       preconditioner_A;
  const PreconditionerSType &       preconditioner_S;

  const bool do_solve_A;
};

template<class PreconditionerAType, class PreconditionerSType>
BlockSchurPreconditioner<PreconditionerAType, PreconditionerSType>::BlockSchurPreconditioner(
  const BlockSparseMatrix<double> & system_matrix,
  const SparseMatrix<double> &      schur_complement_matrix,
  const PreconditionerAType &       preconditioner_A,
  const PreconditionerSType &       preconditioner_S,
  const bool                        do_solve_A)
  : n_iterations_A(0),
    n_iterations_S(0),
    system_matrix(system_matrix),
    schur_complement_matrix(schur_complement_matrix),
    preconditioner_A(preconditioner_A),
    preconditioner_S(preconditioner_S),
    do_solve_A(do_solve_A)
{
}

template<class PreconditionerAType, class PreconditionerSType>
void
BlockSchurPreconditioner<PreconditionerAType, PreconditionerSType>::vmult(
  BlockVector<double> &       dst,
  const BlockVector<double> & src) const
{
  Vector<double> utmp(src.block(0));

  // First solve with the approximation for S
  {
    SolverControl            solver_control(1000, 1e-6 * src.block(1).l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    dst.block(1) = 0.0;
    cg.solve(schur_complement_matrix, dst.block(1), src.block(1), preconditioner_S);

    n_iterations_S += solver_control.last_step();
    dst.block(1) *= -1.0;
  }

  // Second, apply the top right block (B^T)
  {
    system_matrix.block(0, 1).vmult(utmp, dst.block(1));
    utmp *= -1.0;
    utmp += src.block(0);
  }

  // Finally, either solve with the top left block
  // or just apply one preconditioner sweep
  if(do_solve_A == true)
  {
    SolverControl            solver_control(10000, utmp.l2_norm() * 1e-4);
    SolverCG<Vector<double>> cg(solver_control);

    dst.block(0) = 0.0;
    cg.solve(system_matrix.block(0, 0), dst.block(0), utmp, preconditioner_A);

    n_iterations_A += solver_control.last_step();
  }
  else
  {
    preconditioner_A.vmult(dst.block(0), utmp);
    n_iterations_A += 1;
  }
}



template<int dim, int fe_degree_p>
class ModelProblem
{
  static_assert(dim == 2, "only 2D");

public:
  ModelProblem(const RT::Parameter & rt_parameters_in, const EquationData & equation_data_in);

  void
  run();

  bool
  make_grid();

  bool
  make_grid(const unsigned int n_refinements);

  void
  make_grid_impl(const MeshParameter & mesh_prms);

  void
  setup_system();

  void
  assemble_system();

  void
  assemble_multigrid();

  void
  solve();

  void
  compute_errors();

  void
  output_results(const unsigned int refinement_cycle) const;

  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const
  {
    *pcout << Util::parameter_to_fstring(description, value);
  }

  void
  print_informations() const
  {
    *pcout << equation_data.to_string();
    *pcout << std::endl;
    print_parameter("Finite element:", fe.get_name());
    *pcout << rt_parameters.to_string();
    *pcout << std::endl;
  }

  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  mutable PostProcessData             pp_data;
  mutable PostProcessData             pp_data_pressure;

  Triangulation<dim> triangulation;
  FESystem<dim>      velocity_fe;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;
  DoFHandler<dim>    velocity_dof_handler;

  AffineConstraints<double> constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  SparseMatrix<double>      pressure_mass_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;

  MGLevelObject<SparsityPattern>      mg_sparsity_patterns;
  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;
};



template<int dim, int fe_degree_p>
ModelProblem<dim, fe_degree_p>::ModelProblem(const RT::Parameter & rt_parameters_in,
                                             const EquationData &  equation_data_in)
  : pcout(
      std::make_shared<ConditionalOStream>(std::cout,
                                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    rt_parameters(rt_parameters_in),
    equation_data(equation_data_in),
    triangulation(Triangulation<dim>::maximum_smoothing),
    // Finite element for the velocity only:
    velocity_fe(FE_Q<dim>(fe_degree_p + 1), dim),
    // Finite element for the whole system:
    fe(velocity_fe, 1, FE_Q<dim>(fe_degree_p), 1),
    dof_handler(triangulation),
    velocity_dof_handler(triangulation)
{
}



template<int dim, int fe_degree_p>
bool
ModelProblem<dim, fe_degree_p>::make_grid()
{
  make_grid_impl(rt_parameters.mesh);
  return true;
}



template<int dim, int fe_degree_p>
bool
ModelProblem<dim, fe_degree_p>::make_grid(const unsigned int n_refinements)
{
  MeshParameter mesh_prms = rt_parameters.mesh;
  mesh_prms.n_refinements = n_refinements;

  //: estimate number of dofs (velocity + pressure)
  AssertDimension(fe.n_base_elements(), 2); // velocity + pressure
  const auto & fe_v = fe.base_element(0);
  AssertDimension(fe_v.n_components(), dim);          // velocity
  AssertDimension(fe_v.element_multiplicity(0), dim); // dim times FE_Q
  const auto   n_dofs_est_v = dim * estimate_n_dofs(fe_v.base_element(0), mesh_prms);
  const auto & fe_p         = fe.base_element(1);
  AssertDimension(fe_p.n_components(), 1); // pressure
  const auto n_dofs_est_p = estimate_n_dofs(fe_p, mesh_prms);
  const auto n_dofs_est   = n_dofs_est_v + n_dofs_est_p;
  if(rt_parameters.exceeds_dof_limits(n_dofs_est))
    return false;

  make_grid_impl(mesh_prms);
  return true;
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::make_grid_impl(const MeshParameter & mesh_prms)
{
  triangulation.clear();
  *pcout << create_mesh(triangulation, mesh_prms) << std::endl;
  pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());
  pp_data.n_dimensions = dim;
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::setup_system()
{
  system_matrix.clear();
  pressure_mass_matrix.clear();

  dof_handler.distribute_dofs(fe);

  std::vector<unsigned int> block_component(2);
  block_component[0] = 0;
  block_component[1] = 1;

  // Velocities start at component 0:
  const FEValuesExtractors::Vector velocities(0);

  // ILU behaves better if we apply a reordering to reduce fillin. There
  // is no advantage in doing this for the other solvers.
  if(rt_parameters.solver.variant == "FGMRES_ILU")
  {
    DoFRenumbering::Cuthill_McKee(dof_handler);
  }

  // This ensures that all velocities DoFs are enumerated before the
  // pressure unknowns. This allows us to use blocks for vectors and
  // matrices and allows us to get the same DoF numbering for
  // dof_handler and velocity_dof_handler.
  DoFRenumbering::block_wise(dof_handler);

  if(rt_parameters.solver.variant == "FGMRES_GMG")
  {
    velocity_dof_handler.distribute_dofs(velocity_fe);
    velocity_dof_handler.distribute_mg_dofs();

    std::set<types::boundary_id> zero_boundary_ids;
    zero_boundary_ids.insert(0);

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(velocity_dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(velocity_dof_handler, zero_boundary_ids);
    const unsigned int n_levels = triangulation.n_levels();

    mg_interface_matrices.resize(0, n_levels - 1);
    mg_matrices.resize(0, n_levels - 1);
    mg_sparsity_patterns.resize(0, n_levels - 1);

    for(unsigned int level = 0; level < n_levels; ++level)
    {
      DynamicSparsityPattern csp(velocity_dof_handler.n_dofs(level),
                                 velocity_dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(velocity_dof_handler, csp, level);
      mg_sparsity_patterns[level].copy_from(csp);

      mg_matrices[level].reinit(mg_sparsity_patterns[level]);
      mg_interface_matrices[level].reinit(mg_sparsity_patterns[level]);
    }
  }

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
      dof_handler, 0, ZeroBoundary::Solution<dim>(), constraints, fe.component_mask(velocities));

    // As discussed in the introduction, we need to fix one degree of freedom
    // of the pressure variable to ensure solvability of the problem. We do
    // this here by marking the first pressure dof, which has index n_u as a
    // constrained dof.
    if(rt_parameters.solver.variant == "UMFPACK")
      constraints.add_line(n_u);

    constraints.close();
  }

  {
    BlockDynamicSparsityPattern csp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, csp, constraints, false);
    sparsity_pattern.copy_from(csp);
  }
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);

  print_parameter("Number of degrees of freedom (velocity):", n_u);
  print_parameter("Number of degrees of freedom (pressure):", n_p);
  print_parameter("Number of degrees of freedom (total):", n_u + n_p);
  pp_data.n_dofs_global.push_back(system_matrix.m());
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::assemble_system()
{
  system_matrix = 0;
  system_rhs    = 0;

  // If true, we will assemble the pressure mass matrix in the (1,1) block:
  const bool assemble_pressure_mass_matrix =
    (rt_parameters.solver.variant == "UMFPACK") ? false : true;

  QGauss<dim> quadrature_formula(fe_degree_p + 2);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values |
                            update_gradients);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const ZeroBoundary::RightHandSide<dim> right_hand_side;
  std::vector<Vector<double>>            rhs_values(n_q_points, Vector<double>(dim + 1));

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<double>                  phi_p(dofs_per_cell);

  for(const auto & cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    local_matrix = 0;
    local_rhs    = 0;

    right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);

    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
        div_phi_u[k]     = fe_values[velocities].divergence(k, q);
        phi_p[k]         = fe_values[pressure].value(k, q);
      }

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for(unsigned int j = 0; j <= i; ++j)
        {
          local_matrix(i, j) +=
            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) - div_phi_u[i] * phi_p[j] -
             phi_p[i] * div_phi_u[j] + (assemble_pressure_mass_matrix ? phi_p[i] * phi_p[j] : 0)) *
            fe_values.JxW(q);
        }

        const unsigned int component_i = fe.system_to_component_index(i).first;
        local_rhs(i) += fe_values.shape_value(i, q) * rhs_values[q](component_i) * fe_values.JxW(q);
      }
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = i + 1; j < dofs_per_cell; ++j)
        local_matrix(i, j) = local_matrix(j, i);

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
      local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
  }

  if(rt_parameters.solver.variant != "UMFPACK")
  {
    pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
    pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
    system_matrix.block(1, 1) = 0;
  }

  if(rt_parameters.solver.variant == "FGMRES_GMG")
    assemble_multigrid();
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::assemble_multigrid()
{
  mg_matrices = 0.;

  QGauss<dim> quadrature_formula(fe_degree_p + 2);

  FEValues<dim> fe_values(velocity_fe,
                          quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values |
                            update_gradients);

  const unsigned int dofs_per_cell = velocity_fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);

  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);

  std::vector<AffineConstraints<double>> boundary_constraints(triangulation.n_levels());
  std::vector<AffineConstraints<double>> boundary_interface_constraints(triangulation.n_levels());
  for(unsigned int level = 0; level < triangulation.n_levels(); ++level)
  {
    boundary_constraints[level].add_lines(mg_constrained_dofs.get_refinement_edge_indices(level));
    boundary_constraints[level].add_lines(mg_constrained_dofs.get_boundary_indices(level));
    boundary_constraints[level].close();

    IndexSet idx = mg_constrained_dofs.get_refinement_edge_indices(level) &
                   mg_constrained_dofs.get_boundary_indices(level);

    boundary_interface_constraints[level].add_lines(idx);
    boundary_interface_constraints[level].close();
  }

  // This iterator goes over all cells (not just active)
  for(const auto & cell : velocity_dof_handler.cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;

    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int k = 0; k < dofs_per_cell; ++k)
        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int j = 0; j <= i; ++j)
        {
          cell_matrix(i, j) += (symgrad_phi_u[i] * symgrad_phi_u[j]) * fe_values.JxW(q);
        }
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = i + 1; j < dofs_per_cell; ++j)
        cell_matrix(i, j) = cell_matrix(j, i);

    cell->get_mg_dof_indices(local_dof_indices);

    boundary_constraints[cell->level()].distribute_local_to_global(cell_matrix,
                                                                   local_dof_indices,
                                                                   mg_matrices[cell->level()]);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        if(!mg_constrained_dofs.at_refinement_edge(cell->level(), local_dof_indices[i]) ||
           mg_constrained_dofs.at_refinement_edge(cell->level(), local_dof_indices[j]))
          cell_matrix(i, j) = 0;

    boundary_interface_constraints[cell->level()].distribute_local_to_global(
      cell_matrix, local_dof_indices, mg_interface_matrices[cell->level()]);
  }
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::solve()
{
  constraints.set_zero(solution);

  if(rt_parameters.solver.variant == "UMFPACK")
  {
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);
    constraints.distribute(solution);

    pp_data.average_reduction_system.push_back(0.);
    pp_data.n_iterations_system.push_back(0.);
    print_parameter("Average reduction (solver):", "direct solver");
    print_parameter("Number of iterations (solver):", "---");
    return;
  }

  // Here we must make sure to solve for the residual with "good enough"
  // accuracy
  SolverControl solver_control(system_matrix.m(), 1e-10 * system_rhs.l2_norm());
  unsigned int  n_iterations_A;
  unsigned int  n_iterations_S;

  // This is used to pass whether or not we want to solve for A inside
  // the preconditioner.  One could change this to false to see if
  // there is still convergence and if so does the program then run
  // faster or slower
  const bool use_expensive = true;

  SolverFGMRES<BlockVector<double>> solver(solver_control);

  if(rt_parameters.solver.variant == "FGMRES_ILU")
  {
    std::cout << "   Computing preconditioner..." << std::endl << std::flush;

    SparseILU<double> A_preconditioner;
    A_preconditioner.initialize(system_matrix.block(0, 0));

    SparseILU<double> S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix);

    const BlockSchurPreconditioner<SparseILU<double>, SparseILU<double>> preconditioner(
      system_matrix, pressure_mass_matrix, A_preconditioner, S_preconditioner, use_expensive);

    {
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
      n_iterations_A = preconditioner.n_iterations_A;
      n_iterations_S = preconditioner.n_iterations_S;
    }
  }
  else
  {
    // Transfer operators between levels
    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(velocity_dof_handler);

    // Setup coarse grid solver
    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from(mg_matrices[0]);
    MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
    coarse_grid_solver.initialize(coarse_matrix);

    using Smoother = PreconditionSOR<SparseMatrix<double>>;
    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);

    // Multigrid, when used as a preconditioner for CG, needs to be a
    // symmetric operator, so the smoother must be symmetric
    mg_smoother.set_symmetric(true);

    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);

    // Now, we are ready to set up the V-cycle operator and the multilevel
    // preconditioner.
    Multigrid<Vector<double>> mg(
      mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> A_Multigrid(
      velocity_dof_handler, mg, mg_transfer);

    SparseILU<double> S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix, SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<
      PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
      SparseILU<double>>
      preconditioner(
        system_matrix, pressure_mass_matrix, A_Multigrid, S_preconditioner, use_expensive);


    {
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
      n_iterations_A = preconditioner.n_iterations_A;
      n_iterations_S = preconditioner.n_iterations_S;
    }
  }

  constraints.distribute(solution);

  std::cout << std::endl
            << "\tNumber of FGMRES iterations: " << solver_control.last_step() << std::endl
            << "\tTotal number of iterations used for approximation of A inverse: "
            << n_iterations_A << std::endl
            << "\tTotal number of iterations used for approximation of S inverse: "
            << n_iterations_S << std::endl
            << std::endl;

  const double mean_pressure =
    VectorTools::compute_mean_value(dof_handler, QGauss<dim>(fe_degree_p + 2), solution, dim);
  solution.block(1).add(-mean_pressure);
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::compute_errors()
{
  const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);

  Vector<float> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    ZeroBoundary::Solution<dim>(),
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree_p + 2),
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  const double Velocity_L2_error =
    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
  print_parameter("Velocity error in the L2 norm:", Velocity_L2_error);
  pp_data.L2_error.push_back(Velocity_L2_error);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    ZeroBoundary::Solution<dim>(),
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree_p + 2),
                                    VectorTools::L2_norm,
                                    &pressure_mask);
  const double Pressure_L2_error =
    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
  print_parameter("Pressure error in the L2 norm:", Pressure_L2_error);
  pp_data_pressure.L2_error.push_back(Pressure_L2_error);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    ZeroBoundary::Solution<dim>(),
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree_p + 2),
                                    VectorTools::H1_norm,
                                    &velocity_mask);
  const double Velocity_H1_error =
    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_norm);
  print_parameter("Velocity error in the H1 seminorm:", Velocity_H1_error);
  pp_data.H1semi_error.push_back(Velocity_H1_error);
}


// @sect4{ModelProblem::output_results}

// This function generates graphical output like it is done in step-22.
template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::output_results(const unsigned int refinement_cycle) const
{
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  std::ofstream output("solution-" + Utilities::int_to_string(refinement_cycle, 2) + ".vtk");
  data_out.write_vtk(output);
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::run()
{
  print_informations();

  const unsigned int n_cycles = rt_parameters.n_cycles;
  for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    *pcout << "Cycle: " << cycle + 1 << " of " << n_cycles << std::endl;

    const unsigned int n_refinements = rt_parameters.mesh.n_refinements + cycle;
    if(!make_grid(n_refinements))
    {
      *pcout << "NO MESH CREATED AT CYCLE " << cycle << " !!!\n\n";
      continue;
    }

    setup_system();

    assemble_system();

    solve();

    compute_errors();

    output_results(cycle);

    Utilities::System::MemoryStats mem;
    Utilities::System::get_memory_stats(mem);
    print_parameter("Memory used (VM Peak)", mem.VmPeak);

    *pcout << std::endl;
  }
}
} // namespace Stokes

#endif // inclusion guard
