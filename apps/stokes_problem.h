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

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/meshworker/mesh_loop.h>

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
#include "stokes_integrator.h"


namespace Stokes
{
using namespace dealii;

template<int dim, int fe_degree_p = 2, typename Number = double>
class BlockSparseMatrixAugmented : public BlockSparseMatrix<Number>
{
public:
  using value_type  = Number;
  using matrix_type = BlockSparseMatrix<Number>;
  // using local_integrator_type = C0IP::FD::MatrixIntegrator<dim, fe_degree_p, Number>;

  // void
  // initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
  //            const EquationData                             equation_data_in)
  // {
  //   mf_storage = mf_storage_in;
  //   local_integrator_type::initialize(equation_data_in);
  // }

  // std::shared_ptr<const MatrixFree<dim, Number>>
  // get_matrix_free() const
  // {
  //   AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  //   return mf_storage;
  // }

  // using matrix_type::vmult;

  // void
  // vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const
  // {
  //   AssertDimension(dst.size(), matrix_type::m());
  //   AssertDimension(src.size(), matrix_type::n());
  //   Vector<Number> v(matrix_type::n()); // src
  //   std::copy(src.cbegin(), src.cend(), v.begin());
  //   Vector<Number> w(matrix_type::m()); // dst
  //   matrix_type::vmult(w, v);           // w = A v
  //   std::copy(w.begin(), w.end(), dst.begin());
  // }

  // std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
};

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
    Vector<double>                       cell_rhs;
  };

  unsigned int                         level;
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<FaceData>                face_data;
};



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
  setup_system();

  void
  setup_system_velocity();

  void
  assemble_system();

  void
  assemble_system_velocity();

  void
  assemble_multigrid();

  void
  solve();

  void
  compute_errors();

  void
  output_results(const unsigned int refinement_cycle) const;

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
    *pcout << equation_data.to_string();
    *pcout << std::endl;
    print_parameter("Finite element:", fe.get_name());
    print_parameter("Finite element (velocity):", fe_velocity->get_name());
    *pcout << rt_parameters.to_string();
    *pcout << std::endl;
  }

  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  std::shared_ptr<Function<dim>>      analytical_solution;
  std::shared_ptr<Function<dim>>      load_function;
  mutable PostProcessData             pp_data;
  mutable PostProcessData             pp_data_pressure;

  Triangulation<dim>                  triangulation;
  MappingQ<dim>                       mapping;
  std::shared_ptr<FiniteElement<dim>> fe_velocity;
  FESystem<dim>                       fe;
  DoFHandler<dim>                     dof_handler;
  DoFHandler<dim>                     dof_handler_velocity;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_velocity;

  BlockSparsityPattern                         sparsity_pattern;
  BlockSparseMatrixAugmented<dim, fe_degree_p> system_matrix;
  SparseMatrix<double>                         pressure_mass_matrix;
  SparsityPattern                              sparsity_pattern_velocity;
  SparseMatrix<double>                         system_matrix_velocity;

  BlockVector<double> system_solution;
  BlockVector<double> system_rhs;

  MGLevelObject<SparsityPattern>      mg_sparsity_patterns;
  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;

private:
  template<typename IteratorType, bool is_multigrid = false>
  void
  cell_worker_impl(const IteratorType & cell,
                   ScratchData<dim> &   scratch_data,
                   CopyData &           copy_data) const;

  template<typename IteratorType, bool is_multigrid = false>
  void
  cell_worker_velocity_impl(const IteratorType & cell,
                            ScratchData<dim> &   scratch_data,
                            CopyData &           copy_data) const
  {
    copy_data.cell_matrix = 0.;
    copy_data.cell_rhs    = 0.;

    FEValues<dim> & phi = scratch_data.fe_values;
    phi.reinit(cell);
    cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);

    const unsigned int               dofs_per_cell = phi.get_fe().dofs_per_cell;
    const FEValuesExtractors::Vector velocities(0);
    /// symgrad_phi_{d,c} = 0.5 (\partial_d phi_{i;c} + \partial_c phi_{i;d})
    const auto symgrad_phi = [&](const unsigned int i, const unsigned int q) {
      SymmetricTensor<2, dim> symgrad_of_phi;
      for(auto d = 0U; d < dim; ++d)
        for(auto c = d; c < dim; ++c)
          symgrad_of_phi[d][c] =
            0.5 * (phi.shape_grad_component(i, q, c)[d] + phi.shape_grad_component(i, q, d)[c]);
      return symgrad_of_phi;
    };

    for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
    {
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const SymmetricTensor<2, dim> symgrad_phi_i =
          symgrad_phi(i, q); // phi[velocities].symmetric_gradient(i, q);
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const SymmetricTensor<2, dim> symgrad_phi_j = phi[velocities].symmetric_gradient(j, q);

          copy_data.cell_matrix(i, j) += 2. *
                                         scalar_product(symgrad_phi_i,   // symgrad phi_i(x)
                                                        symgrad_phi_j) * // symgrad phi_j(x)
                                         phi.JxW(q);                     // dx
        }
      }
    }
  }

  template<typename IteratorType, bool is_multigrid = false>
  void
  face_worker_velocity_impl(const IteratorType & cell,
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

    const auto   h  = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
    const auto   nh = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
    const double gamma_over_h = 0.5 * SIPG::compute_penalty_impl(fe_degree_p + 1, h, nh);

    /// average_symgrad(phi) = 0.5 ({{ \partial_d phi_{i;c} }} + {{ \partial_c phi_{i;d} }})
    const auto average_symgrad_phi = [&](const unsigned int i, const unsigned int q) {
      SymmetricTensor<2, dim> av_symgrad_of_phi;
      for(auto d = 0U; d < dim; ++d)
        for(auto c = d; c < dim; ++c)
          av_symgrad_of_phi[d][c] = 0.5 * (fe_interface_values.average_gradient(i, q, c)[d] +
                                           fe_interface_values.average_gradient(i, q, d)[c]);
      return av_symgrad_of_phi;
    };

    /// jump(phi) = [[ phi ]] = phi^+ - phi^-
    const auto jump_phi = [&](const unsigned int i, const unsigned int q) {
      Tensor<1, dim> jump_phi;
      for(auto c = 0; c < dim; ++c)
        jump_phi[c] = fe_interface_values.jump(i, q, c);
      return jump_phi;
    };

    /// jump_cross_normal(phi) = [[ phi ]] (x) n
    const auto jump_phi_cross_normal = [&](const unsigned int i, const unsigned int q) {
      const Tensor<1, dim> & n = fe_interface_values.normal(q);
      return outer_product(jump_phi(i, q), n);
    };

    double integral_ijq = 0.;
    for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
    {
      // const auto & n = fe_interface_values.normal(q);

      for(unsigned int i = 0; i < n_interface_dofs; ++i)
      {
        const auto & av_symgrad_phi_i   = average_symgrad_phi(i, q);
        const auto & jump_phi_i_cross_n = jump_phi_cross_normal(i, q);

        for(unsigned int j = 0; j < n_interface_dofs; ++j)
        {
          const auto & av_symgrad_phi_j   = average_symgrad_phi(j, q);
          const auto & jump_phi_j_cross_n = jump_phi_cross_normal(j, q);

          integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
          integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
          integral_ijq += gamma_over_h * jump_phi(j, q) * jump_phi(i, q);
          integral_ijq *= 2. * fe_interface_values.JxW(q);

          copy_data_face.cell_matrix(i, j) += integral_ijq;
        }
      }
    }
  }

  template<typename IteratorType, bool is_multigrid = false>
  void
  boundary_worker_velocity_impl(const IteratorType & cell,
                                const unsigned int & f,
                                ScratchData<dim> &   scratch_data,
                                CopyData &           copy_data) const
  {
    FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
    fe_interface_values.reinit(cell, f);

    copy_data.face_data.emplace_back();
    CopyData::FaceData & copy_data_face = copy_data.face_data.back();

    copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

    const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();
    copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

    const auto   h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
    const double gamma_over_h = SIPG::compute_penalty_impl(fe_degree_p + 1, h, h);

    /// average_symgrad(phi) = 0.5 ({{ \partial_d phi_{i;c} }} + {{ \partial_c phi_{i;d} }})
    const auto average_symgrad_phi = [&](const unsigned int i, const unsigned int q) {
      SymmetricTensor<2, dim> av_symgrad_of_phi;
      for(auto d = 0U; d < dim; ++d)
        for(auto c = d; c < dim; ++c)
          av_symgrad_of_phi[d][c] = 0.5 * (fe_interface_values.average_gradient(i, q, c)[d] +
                                           fe_interface_values.average_gradient(i, q, d)[c]);
      return av_symgrad_of_phi;
    };

    /// jump(phi) = [[ phi ]] = phi^+ - phi^-
    const auto jump_phi = [&](const unsigned int i, const unsigned int q) {
      Tensor<1, dim> jump_phi;
      for(auto c = 0; c < dim; ++c)
        jump_phi[c] = fe_interface_values.jump(i, q, c);
      return jump_phi;
    };

    /// jump_cross_normal(phi) = [[ phi ]] (x) n
    const auto jump_phi_cross_normal = [&](const unsigned int i, const unsigned int q) {
      const Tensor<1, dim> & n = fe_interface_values.normal(q);
      return outer_product(jump_phi(i, q), n);
    };

    double integral_ijq = 0.;
    for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
    {
      // const auto & n = fe_interface_values.normal(q);

      for(unsigned int i = 0; i < n_interface_dofs; ++i)
      {
        const auto & av_symgrad_phi_i   = average_symgrad_phi(i, q);
        const auto & jump_phi_i_cross_n = jump_phi_cross_normal(i, q);

        for(unsigned int j = 0; j < n_interface_dofs; ++j)
        {
          const auto & av_symgrad_phi_j   = average_symgrad_phi(j, q);
          const auto & jump_phi_j_cross_n = jump_phi_cross_normal(j, q);

          integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
          integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
          integral_ijq += gamma_over_h * jump_phi(j, q) * jump_phi(i, q);
          integral_ijq *= 2. * fe_interface_values.JxW(q);

          copy_data_face.cell_matrix(i, j) += integral_ijq;
        }
      }
    }
  }

  void
  make_grid_impl(const MeshParameter & mesh_prms);
};



template<int dim, int fe_degree_p>
ModelProblem<dim, fe_degree_p>::ModelProblem(const RT::Parameter & rt_parameters_in,
                                             const EquationData &  equation_data_in)
  : pcout(
      std::make_shared<ConditionalOStream>(std::cout,
                                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    rt_parameters(rt_parameters_in),
    equation_data(equation_data_in),
    analytical_solution(std::make_shared<DivergenceFree::Solution<dim>>()),
    load_function(std::make_shared<DivergenceFree::Load<dim>>()),
    triangulation(Triangulation<dim>::maximum_smoothing),
    mapping(1),
    // Finite element for the velocity only:
    fe_velocity(std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree_p + 1), dim)),
    // Finite element for the whole system:
    fe(*fe_velocity, 1, FE_Q<dim>(fe_degree_p), 1),
    dof_handler(triangulation),
    dof_handler_velocity(triangulation)
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
ModelProblem<dim, fe_degree_p>::setup_system_velocity()
{
  system_matrix_velocity.clear();
  dof_handler_velocity.initialize(triangulation, *fe_velocity);
  // dof_handler_velocity.distribute_dofs(*fe_velocity);
  dof_handler_velocity.distribute_mg_dofs();

  constraints_velocity.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_velocity, constraints_velocity);
  const FEValuesExtractors::Vector velocities(0);
  VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                           0,
                                           DivergenceFree::SolutionVelocity<dim>{},
                                           constraints_velocity);
  constraints_velocity.close();

  DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler_velocity,
                                       dsp,
                                       constraints_velocity,
                                       rt_parameters.solver.variant == "UMFPACK" ? true : false);
  sparsity_pattern_velocity.copy_from(dsp);
  system_matrix_velocity.reinit(sparsity_pattern_velocity);

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler_velocity);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_velocity,
                                                     equation_data.dirichlet_boundary_ids);

  mg_interface_matrices.resize(0, max_level());
  mg_matrices.resize(0, max_level());
  mg_sparsity_patterns.resize(0, max_level());

  for(unsigned int level = 0; level <= max_level(); ++level)
  {
    DynamicSparsityPattern csp(dof_handler_velocity.n_dofs(level),
                               dof_handler_velocity.n_dofs(level));
    MGTools::make_flux_sparsity_pattern(dof_handler_velocity, csp, level);
    mg_sparsity_patterns[level].copy_from(csp);

    mg_matrices[level].reinit(mg_sparsity_patterns[level]);
    mg_interface_matrices[level].reinit(mg_sparsity_patterns[level]);
  }
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::setup_system()
{
  system_matrix.clear();
  pressure_mass_matrix.clear();

  dof_handler.distribute_dofs(fe);

  // ILU behaves better if we apply a reordering to reduce fillin. There
  // is no advantage in doing this for the other solvers.
  if(rt_parameters.solver.variant == "FGMRES_ILU")
  {
    DoFRenumbering::Cuthill_McKee(dof_handler);
  }

  // This ensures that all velocities DoFs are enumerated before the
  // pressure unknowns. This allows us to use blocks for vectors and
  // matrices and allows us to get the same DoF numbering for
  // dof_handler and dof_handler_velocity.
  DoFRenumbering::block_wise(dof_handler);

  // if(rt_parameters.solver.variant == "FGMRES_GMG") // TODO !!!
  setup_system_velocity();

  std::vector<unsigned int>                  block_component{0U, 1U};
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    const FEValuesExtractors::Vector velocities(0);
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids)
      VectorTools::interpolate_boundary_values(
        dof_handler, boundary_id, *analytical_solution, constraints, fe.component_mask(velocities));

    // As discussed in the introduction, we need to fix one degree of freedom
    // of the pressure variable to ensure solvability of the problem. We do
    // this here by marking the first pressure dof, which has index n_u as a
    // constrained dof.
    if(rt_parameters.solver.variant == "UMFPACK")
      constraints.add_line(n_u);

    constraints.close();
  }

  BlockDynamicSparsityPattern csp(dofs_per_block, dofs_per_block);
  DoFTools::make_sparsity_pattern(dof_handler, csp, constraints, false);
  sparsity_pattern.copy_from(csp);
  system_matrix.reinit(sparsity_pattern);
  system_solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);

  print_parameter("Number of degrees of freedom (velocity):", n_u);
  print_parameter("Number of degrees of freedom (pressure):", n_p);
  print_parameter("Number of degrees of freedom (total):", n_u + n_p);
  pp_data.n_dofs_global.push_back(system_matrix.m());
}



template<int dim, int fe_degree_p>
template<typename IteratorType, bool is_multigrid>
void
ModelProblem<dim, fe_degree_p>::cell_worker_impl(const IteratorType & cell,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  // If true, we will assemble the pressure mass matrix in the (1,1) block:
  const bool assemble_pressure_mass_matrix =
    (rt_parameters.solver.variant == "UMFPACK") ? false : true;

  copy_data.cell_matrix = 0.;
  copy_data.cell_rhs    = 0.;

  FEValues<dim> & fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);
  cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);
  const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = fe_values.n_quadrature_points;

  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<double>                  phi_p(dofs_per_cell);

  load_function->vector_value_list(fe_values.get_quadrature_points(), rhs_values);

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
        copy_data.cell_matrix(i, j) +=
          (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) - div_phi_u[i] * phi_p[j] -
           phi_p[i] * div_phi_u[j] + (assemble_pressure_mass_matrix ? phi_p[i] * phi_p[j] : 0)) *
          fe_values.JxW(q);
      }

      const unsigned int component_i = fe.system_to_component_index(i).first;
      copy_data.cell_rhs(i) +=
        fe_values.shape_value(i, q) * rhs_values[q](component_i) * fe_values.JxW(q);
    }
  }

  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    for(unsigned int j = i + 1; j < dofs_per_cell; ++j)
      copy_data.cell_matrix(i, j) = copy_data.cell_matrix(j, i);
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::assemble_system()
{
  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    cell_worker_impl(cell, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    constraints.template distribute_local_to_global<BlockSparseMatrix<double>, BlockVector<double>>(
      copy_data.cell_matrix,
      copy_data.cell_rhs,
      copy_data.local_dof_indices,
      system_matrix,
      system_rhs);
  };

  const unsigned int n_gauss_points = fe_degree_p + 2;
  ScratchData<dim>   scratch_data(mapping,
                                fe,
                                n_gauss_points,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values | update_normal_vectors);
  CopyData           copy_data(dof_handler.get_fe().dofs_per_cell);
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells);

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
ModelProblem<dim, fe_degree_p>::assemble_system_velocity()
{
  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    cell_worker_velocity_impl(cell, scratch_data, copy_data);
  };

  auto face_worker = [&](const auto &         cell,
                         const unsigned int & f,
                         const unsigned int & sf,
                         const auto &         ncell,
                         const unsigned int & nf,
                         const unsigned int & nsf,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) {
    face_worker_velocity_impl(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
  };

  auto boundary_worker = [&](const auto &         cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
    boundary_worker_velocity_impl(cell, face_no, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
      copy_data.cell_matrix, copy_data.local_dof_indices, system_matrix_velocity);

    for(auto & cdf : copy_data.face_data)
    {
      constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
        cdf.cell_matrix, cdf.joint_dof_indices, system_matrix_velocity);
    }
  };

  const unsigned int n_gauss_points = fe_degree_p + 2;
  ScratchData<dim>   scratch_data(mapping,
                                *fe_velocity,
                                n_gauss_points,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values | update_normal_vectors);
  CopyData           copy_data(dof_handler_velocity.get_fe().dofs_per_cell);
  MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                        dof_handler_velocity.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::assemble_multigrid()
{
  mg_matrices = 0.;

  QGauss<dim> quadrature_formula(fe_degree_p + 2);

  FEValues<dim> fe_values(*fe_velocity,
                          quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values |
                            update_gradients);

  const unsigned int dofs_per_cell = fe_velocity->dofs_per_cell;
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
  for(const auto & cell : dof_handler_velocity.cell_iterators())
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
  constraints.set_zero(system_solution);

  if(rt_parameters.solver.variant == "UMFPACK")
  {
    SparseDirectUMFPACK A_direct;
    A_direct.template initialize<BlockSparseMatrix<double>>(system_matrix);
    A_direct.vmult(system_solution, system_rhs);
    constraints.distribute(system_solution);

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
      solver.solve(system_matrix, system_solution, system_rhs, preconditioner);
      n_iterations_A = preconditioner.n_iterations_A;
      n_iterations_S = preconditioner.n_iterations_S;
    }
  }
  else
  {
    // Transfer operators between levels
    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler_velocity);

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
      dof_handler_velocity, mg, mg_transfer);

    SparseILU<double> S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix, SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<
      PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
      SparseILU<double>>
      preconditioner(
        system_matrix, pressure_mass_matrix, A_Multigrid, S_preconditioner, use_expensive);


    {
      solver.solve(system_matrix, system_solution, system_rhs, preconditioner);
      n_iterations_A = preconditioner.n_iterations_A;
      n_iterations_S = preconditioner.n_iterations_S;
    }
  }

  constraints.distribute(system_solution);

  std::cout << std::endl
            << "\tNumber of FGMRES iterations: " << solver_control.last_step() << std::endl
            << "\tTotal number of iterations used for approximation of A inverse: "
            << n_iterations_A << std::endl
            << "\tTotal number of iterations used for approximation of S inverse: "
            << n_iterations_S << std::endl
            << std::endl;

  const double mean_pressure = VectorTools::compute_mean_value(dof_handler,
                                                               QGauss<dim>(fe_degree_p + 2),
                                                               system_solution,
                                                               dim);
  system_solution.block(1).add(-mean_pressure);
}



template<int dim, int fe_degree_p>
void
ModelProblem<dim, fe_degree_p>::compute_errors()
{
  const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);

  Vector<float> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree_p + 2),
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  const double Velocity_L2_error =
    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
  print_parameter("Velocity error in the L2 norm:", Velocity_L2_error);
  pp_data.L2_error.push_back(Velocity_L2_error);

  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree_p + 2),
                                    VectorTools::L2_norm,
                                    &pressure_mask);
  const double Pressure_L2_error =
    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
  print_parameter("Pressure error in the L2 norm:", Pressure_L2_error);
  pp_data_pressure.L2_error.push_back(Pressure_L2_error);

  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
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
  data_out.add_data_vector(system_solution,
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

    // TODO !!!
    {
      assemble_system_velocity();
      Vector<double> dst(system_matrix_velocity.m());
      Vector<double> dst_ref(system_matrix_velocity.m());
      Vector<double> src(system_matrix_velocity.m());
      fill_with_random_values(src);
      const auto & matrix_ref = system_matrix.block(0, 0);
      matrix_ref.vmult(dst_ref, src);
      system_matrix_velocity.vmult(dst, src);

      // matrix_ref.print_formatted(std::cout);
      // system_matrix_velocity.print_formatted(std::cout);
      dst -= dst_ref;
      std::cout << "diff: " << dst.l2_norm() << std::endl;
    }

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
