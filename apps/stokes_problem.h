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
#include <deal.II/base/timer.h>
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
#include <deal.II/lac/trilinos_precondition.h>

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

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/sparse_ilu.h>


#include <fstream>
#include <iostream>


#include "coloring.h"
#include "equation_data.h"
#include "multigrid.h"
#include "postprocess.h"
#include "rt_parameter.h"
#include "solvers_and_preconditioners/TPSS/move_to_deal_ii.h"
#include "stokes_integrator.h"

namespace Stokes
{
using namespace dealii;


//   /// DEPRECATED
//   template<TPSS::DoFLayout dof_layout_v, bool is_multigrid>
// MeshWorker::AssembleFlags
// get_assemble_flags_impl()
// {
//   const auto assemble_flags_conforming = MeshWorker::assemble_own_cells;
//   const auto assemble_flags_dg =
//     MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
//                      MeshWorker::assemble_own_interior_faces_once;
//   if(dof_layout_v == TPSS::DoFLayout::Q)
//     return assemble_flags_conforming;
//   else if(dof_layout_v == TPSS::DoFLayout::DGQ)
//     return assemble_flags_dg;
//   AssertThrow(false, ExcMessage("FE is not supported."));
//   return MeshWorker::assemble_nothing;
// }



/**
 * TODO...
 */
template<int dim,
         int fe_degree_p,
         typename Number                = double,
         TPSS::DoFLayout dof_layout_v   = TPSS::DoFLayout::Q,
         int             fe_degree_v    = fe_degree_p + 1,
         LocalAssembly   local_assembly = LocalAssembly::Tensor>
class BlockSparseMatrixAugmented
  : public BlockSparseMatrix<Number>,
    public VelocityPressure::FD::
      MatrixIntegrator<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>
{
public:
  using value_type            = Number;
  using matrix_type           = BlockSparseMatrix<Number>;
  using local_integrator_type = VelocityPressure::FD::
    MatrixIntegrator<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in)
  {
    mf_storage = mf_storage_in;
    local_integrator_type::initialize(equation_data_in);
  }

  void
  clear()
  {
    fullmatrix.reset();
    mf_storage.reset();
    matrix_type::clear();
  }

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const
  {
    AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
    return mf_storage;
  }

  using matrix_type::vmult;

  void
  vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const
  {
    AssertDimension(dst.size(), matrix_type::m());
    AssertDimension(src.size(), matrix_type::n());
    std::vector<typename matrix_type::size_type> n_rows_per_block;
    AssertDimension(matrix_type::n_block_rows(), matrix_type::n_block_cols());
    for(auto b = 0U; b < matrix_type::n_block_rows(); ++b)
      n_rows_per_block.emplace_back(matrix_type::block(b, b).m());
    BlockVector<Number> v(n_rows_per_block); // src
    std::copy(src.cbegin(), src.cend(), v.begin());
    BlockVector<Number> w(n_rows_per_block); // dst
    matrix_type::vmult(w, v);                // w = A v
    std::copy(w.begin(), w.end(), dst.begin());
  }

  operator const FullMatrix<Number> &() const
  {
    if(!fullmatrix)
    {
      const auto & tmp = Tensors::matrix_to_table(*this);
      fullmatrix       = std::make_shared<FullMatrix<Number>>(table_to_fullmatrix(tmp));
    }
    return *fullmatrix;
  }

  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  mutable std::shared_ptr<FullMatrix<Number>>    fullmatrix;
};



template<typename Number>
class BlockSparseMatrixFiltered : public BlockSparseMatrix<Number>
{
public:
  using matrix_type = BlockSparseMatrix<Number>;

  void
  vmult(BlockVector<Number> & dst, const BlockVector<Number> & src) const
  {
    matrix_type::vmult(dst, src);

    if(constant_pressure_mode)
    {
      auto & dst_pressure = dst.block(1);
      AssertDimension(constant_pressure_mode->size(), dst_pressure.size());
      const Number inner_product_value = (*constant_pressure_mode) * dst_pressure;
      dst_pressure.add(-inner_product_value, *constant_pressure_mode);
    }
  }

  const Vector<double> * constant_pressure_mode = nullptr;
};



/**
 * TODO...
 */
template<int dim,
         int fe_degree,
         typename Number            = double,
         TPSS::DoFLayout dof_layout = TPSS::DoFLayout::Q>
class SparseMatrixAugmented
  : public SparseMatrix<Number>,
    public Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree, Number, dof_layout>
{
public:
  using value_type  = Number;
  using matrix_type = SparseMatrix<Number>;
  using local_integrator_type =
    Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree, Number, dof_layout>;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in)
  {
    mf_storage = mf_storage_in;
    local_integrator_type::initialize(equation_data_in);
  }

  void
  clear()
  {
    fullmatrix.reset();
    mf_storage.reset();
    matrix_type::clear();
  }

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const
  {
    AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
    return mf_storage;
  }

  using matrix_type::vmult;

  void
  vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const
  {
    AssertDimension(dst.size(), matrix_type::m());
    AssertDimension(src.size(), matrix_type::n());
    Vector<Number> v(matrix_type::n()); // src
    std::copy(src.cbegin(), src.cend(), v.begin());
    Vector<Number> w(matrix_type::m()); // dst
    matrix_type::vmult(w, v);           // w = A v
    std::copy(w.begin(), w.end(), dst.begin());
  }

  operator const FullMatrix<Number> &() const
  {
    if(!fullmatrix)
    {
      const auto & tmp = Tensors::matrix_to_table(*this);
      fullmatrix       = std::make_shared<FullMatrix<Number>>(table_to_fullmatrix(tmp));
    }
    return *fullmatrix;
  }

  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  mutable std::shared_ptr<FullMatrix<Number>>    fullmatrix;
};



/**
 * TODO...
 */
template<class PreconditionerAType, class PreconditionerSType>
class BlockSchurPreconditioner : public Subscriptor
{
public:
  BlockSchurPreconditioner(const BlockSparseMatrix<double> & system_matrix,
                           const SparseMatrix<double> &      schur_complement_matrix,
                           const PreconditionerAType &       preconditioner_A,
                           const PreconditionerSType &       preconditioner_S,
                           const bool                        do_solve_A);

  std::string
  get_summary() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Summary of BlockSchurPreconditioner:", "//////////");
    oss << Util::parameter_to_fstring("Accum. number of iterations (~ A^-1):", n_iterations_A);
    oss << Util::parameter_to_fstring("Accum. number of iterations (~ S^-1):", n_iterations_S);
    return oss.str();
  }

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
    SolverControl            solver_control(9999, utmp.l2_norm() * 1e-4); // !!!
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



/**
 * TODO...
 */
template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
struct MGCollectionVelocity
{
  static constexpr int n_q_points_1d = fe_degree + 1 + (dof_layout == TPSS::DoFLayout::RT ? 1 : 0);

  using VECTOR = Vector<double>;
  using MATRIX = SparseMatrixAugmented<dim, fe_degree, double, dof_layout>;

  using MG_TRANSFER           = MGTransferPrebuilt<VECTOR>;
  using GAUSS_SEIDEL_SMOOTHER = PreconditionSOR<MATRIX>;
  using PATCH_MATRIX          = typename MATRIX::local_integrator_type::matrix_type;
  using MG_SMOOTHER_SCHWARZ   = MGSmootherSchwarz<dim, MATRIX, PATCH_MATRIX, VECTOR>;

  MGCollectionVelocity(const MGParameter & mg_prms_in, const EquationData & equation_data_in);

  void
  clear();

  void
  prepare_multigrid(const unsigned int                       mg_level_max,
                    const std::shared_ptr<ColoringBase<dim>> user_coloring);

  void
  prepare_schwarz_smoothers(const std::shared_ptr<ColoringBase<dim>> user_coloring);

  void
  assemble_multigrid();

  const PreconditionMG<dim, VECTOR, MG_TRANSFER> &
  get_preconditioner() const;

  MGParameter  parameters;
  EquationData equation_data;

  const DoFHandler<dim> * dof_handler;
  const Mapping<dim> *    mapping;
  /// DEBUG
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  std::shared_ptr<MGConstrainedDoFs>                mg_constrained_dofs;
  MG_TRANSFER                                       mg_transfer;
  Table<2, DoFTools::Coupling>                      cell_integrals_mask;
  Table<2, DoFTools::Coupling>                      face_integrals_mask;
  MGLevelObject<SparsityPattern>                    mg_sparsity_patterns;
  MGLevelObject<MATRIX>                             mg_matrices;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>        mg_schwarz_smoother_pre;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>        mg_schwarz_smoother_post;
  std::shared_ptr<const MGSmootherIdentity<VECTOR>> mg_smoother_identity;
  std::shared_ptr<const mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER, VECTOR>>
                                     mg_smoother_gauss_seidel;
  const MGSmootherBase<VECTOR> *     mg_smoother_pre;
  const MGSmootherBase<VECTOR> *     mg_smoother_post;
  CoarseGridSolver<MATRIX, VECTOR>   coarse_grid_solver;
  const MGCoarseGridBase<VECTOR> *   mg_coarse_grid;
  mg::Matrix<VECTOR>                 mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>> multigrid;

  mutable std::shared_ptr<const PreconditionMG<dim, VECTOR, MG_TRANSFER>> preconditioner_mg;
};

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
MGCollectionVelocity<dim, fe_degree, dof_layout>::MGCollectionVelocity(
  const MGParameter &  mg_prms_in,
  const EquationData & equation_data_in)
  : parameters(mg_prms_in), equation_data(equation_data_in), dof_handler(nullptr), mapping(nullptr)
{
}

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
void
MGCollectionVelocity<dim, fe_degree, dof_layout>::clear()
{
  preconditioner_mg.reset();
  multigrid.reset();
  mg_matrix_wrapper.reset();
  mg_coarse_grid = nullptr;
  coarse_grid_solver.clear();
  mg_smoother_post = nullptr;
  mg_smoother_pre  = nullptr;
  mg_smoother_gauss_seidel.reset();
  mg_smoother_identity.reset();
  mg_schwarz_smoother_post.reset();
  mg_schwarz_smoother_pre.reset();
  mg_matrices.clear_elements();
  mg_transfer.clear();
  mg_constrained_dofs.reset();
}

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
void
MGCollectionVelocity<dim, fe_degree, dof_layout>::prepare_multigrid(
  const unsigned int                       mg_level_max,
  const std::shared_ptr<ColoringBase<dim>> user_coloring)
{
  // *** clear multigrid infrastructure
  clear();

  // *** setup multigrid data
  const unsigned mg_level_min = parameters.coarse_level;

  // *** initialize multigrid constraints
  mg_constrained_dofs = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs->initialize(*dof_handler);
  if(dof_layout == TPSS::DoFLayout::Q)
    mg_constrained_dofs->make_zero_boundary_constraints(
      *dof_handler, equation_data.dirichlet_boundary_ids_velocity);
  else if(dof_layout == TPSS::DoFLayout::RT) // !!!
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      mg_constrained_dofs->make_no_normal_flux_constraints(*dof_handler, boundary_id, 0U);

  // *** initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  mg_sparsity_patterns.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    DynamicSparsityPattern dsp(dof_handler->n_dofs(level), dof_handler->n_dofs(level));
    MGTools::make_flux_sparsity_pattern(
      *dof_handler, dsp, level, cell_integrals_mask, face_integrals_mask);
    mg_sparsity_patterns[level].copy_from(dsp);
    mg_matrices[level].reinit(mg_sparsity_patterns[level]);
  }

  /// assemble the velocity system A_l on each level l.
  assemble_multigrid();

  // *** initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(*dof_handler);

  // *** initialize Schwarz smoother S_l
  switch(parameters.pre_smoother.variant)
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
      tmp->set_steps(parameters.pre_smoother.n_smoothing_steps);
      tmp->set_symmetric(true);
      mg_smoother_gauss_seidel = tmp;
      mg_smoother_pre          = mg_smoother_gauss_seidel.get();
    }
    break;
    case SmootherParameter::SmootherVariant::Schwarz:
      prepare_schwarz_smoothers(user_coloring);
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }
  switch(parameters.post_smoother.variant)
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
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Not initialized"));
      mg_smoother_post = mg_schwarz_smoother_post.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }

  // *** initialize coarse grid solver
  coarse_grid_solver.initialize(mg_matrices[mg_level_min], parameters.coarse_grid);
  mg_coarse_grid = &coarse_grid_solver;

  // *** initialize geometric multigrid method
  mg_matrix_wrapper.initialize(mg_matrices);
  multigrid = std::make_shared<Multigrid<VECTOR>>(mg_matrix_wrapper,
                                                  *mg_coarse_grid,
                                                  mg_transfer,
                                                  *mg_smoother_pre,
                                                  *mg_smoother_post,
                                                  mg_level_min,
                                                  mg_level_max);
}

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
void
MGCollectionVelocity<dim, fe_degree, dof_layout>::prepare_schwarz_smoothers(
  const std::shared_ptr<ColoringBase<dim>> user_coloring)
{
  Assert(parameters.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz,
         ExcMessage("Invalid smoothing variant."));
  for(auto level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
    AssertThrow(mg_matrices[level].mf_storage, ExcMessage("mf_storage is not initialized."));

  //: pre-smoother
  {
    const auto                                   mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    typename MG_SMOOTHER_SCHWARZ::AdditionalData additional_data;
    if(parameters.pre_smoother.schwarz.userdefined_coloring)
    {
      Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
      additional_data.coloring_func = std::ref(*user_coloring);
    }
    additional_data.parameters = parameters.pre_smoother;
    additional_data.foreach_dofh.resize(1);
    additional_data.foreach_dofh[0].dirichlet_ids = equation_data.dirichlet_boundary_ids_velocity;
    mgss->initialize(mg_matrices, additional_data);
    mg_schwarz_smoother_pre = mgss;
  }

  //: post-smoother (so far only shallow copy!)
  {
    const auto mgss_post = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    typename MG_SMOOTHER_SCHWARZ::AdditionalData additional_data;
    if(parameters.pre_smoother.schwarz.userdefined_coloring)
    {
      Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
      additional_data.coloring_func = std::ref(*user_coloring);
    }
    additional_data.parameters = parameters.post_smoother;
    mgss_post->initialize(*mg_schwarz_smoother_pre, additional_data);
    mg_schwarz_smoother_post = mgss_post;
  }
}

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
void
MGCollectionVelocity<dim, fe_degree, dof_layout>::assemble_multigrid()
{
  AssertDimension(mg_matrices.min_level(), parameters.coarse_level);
  Assert(mg_constrained_dofs, ExcMessage("mg_constrained_dofs is uninitialized."));

  /// TODO discard terms for Hdiv conforming SIPG?
  constexpr bool use_sipg_method =
    dof_layout == TPSS::DoFLayout::DGQ || dof_layout == TPSS::DoFLayout::RT;
  constexpr bool use_conf_method = dof_layout == TPSS::DoFLayout::Q;

  using Velocity::SIPG::MW::CopyData;
  using Velocity::SIPG::MW::ScratchData;
  using MatrixIntegrator  = Velocity::SIPG::MW::MatrixIntegrator<dim, true>;
  using LevelCellIterator = typename MatrixIntegrator::IteratorType;
  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  for(unsigned int level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
  {
    AffineConstraints<double> level_constraints;
    if(dof_layout == TPSS::DoFLayout::Q || dof_layout == TPSS::DoFLayout::RT) // !!!
      level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    {
      //: initialize matrix-free storage (dummy, required to setup TPSS)
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.mg_level = level;
      QGauss<1>  quadrature(n_q_points_1d);
      const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
      mf_storage->reinit(*mapping, *dof_handler, level_constraints, quadrature, additional_data);

      //: initialize FD::MatrixIntegrator (see SparseMatrixAugmented)
      mg_matrices[level].initialize(mf_storage, equation_data);
    }

    auto cell_worker =
      [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & f,
                           const unsigned int & sf,
                           const auto &         ncell,
                           const unsigned int & nf,
                           const unsigned int & nsf,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
    };

    const auto copier = [&](const CopyData & copy_data) {
      level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix, copy_data.local_dof_indices_test, mg_matrices[level]);

      for(auto & cdf : copy_data.face_data)
      {
        level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix, cdf.joint_dof_indices_test, mg_matrices[level]);
      }
    };

    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                               update_quadrature_points | update_JxW_values |
                                               update_normal_vectors;
    ScratchData<dim> scratch_data(*mapping,
                                  dof_handler->get_fe(),
                                  dof_handler->get_fe(),
                                  n_q_points_1d,
                                  update_flags,
                                  update_flags,
                                  interface_update_flags,
                                  interface_update_flags);

    CopyData copy_data(dof_handler->get_fe().dofs_per_cell);

    if(use_conf_method)
      MeshWorker::mesh_loop(dof_handler->begin_mg(level),
                            dof_handler->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method)
      MeshWorker::mesh_loop(dof_handler->begin_mg(level),
                            dof_handler->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker,
                            face_worker);
    else
      AssertThrow(false, ExcMessage("This FEM is not implemented"));
  }
}

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
const PreconditionMG<dim,
                     typename MGCollectionVelocity<dim, fe_degree, dof_layout>::VECTOR,
                     typename MGCollectionVelocity<dim, fe_degree, dof_layout>::MG_TRANSFER> &
MGCollectionVelocity<dim, fe_degree, dof_layout>::get_preconditioner() const
{
  AssertThrow(multigrid, ExcMessage("multigrid is uninitialized."));
  preconditioner_mg = std::make_shared<PreconditionMG<dim, VECTOR, MG_TRANSFER>>(*dof_handler,
                                                                                 *multigrid,
                                                                                 mg_transfer);
  return *preconditioner_mg;
}



/**
 * TODO...
 */
template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v   = TPSS::DoFLayout::Q,
         int             fe_degree_v    = fe_degree_p + 1,
         LocalAssembly   local_assembly = LocalAssembly::Tensor>
struct MGCollectionVelocityPressure
{
  static constexpr int n_q_points_1d =
    fe_degree_v + 1 + (dof_layout_v == TPSS::DoFLayout::RT ? 1 : 0);

  using VECTOR = BlockVector<double>;
  using MATRIX =
    BlockSparseMatrixAugmented<dim, fe_degree_p, double, dof_layout_v, fe_degree_v, local_assembly>;

  // TODO !!! MGTransferBlockMatrixFree (based on distributed::BlockVector)
  using MG_TRANSFER           = MGTransferPrebuilt<VECTOR>;
  using GAUSS_SEIDEL_SMOOTHER = PreconditionSOR<MATRIX>;
  using PATCH_MATRIX          = typename MATRIX::local_integrator_type::matrix_type;
  using MG_SMOOTHER_SCHWARZ   = MGSmootherSchwarz<dim, MATRIX, PATCH_MATRIX, VECTOR>;

  MGCollectionVelocityPressure(const MGParameter &  mg_prms_in,
                               const EquationData & equation_data_in);

  void
  clear();

  /// DEPRECATED
  // MeshWorker::AssembleFlags
  // get_mg_assemble_flags() const;

  void
  prepare_multigrid(const unsigned int                               mg_level_max,
                    const std::shared_ptr<ColoringBase<dim>>         user_coloring,
                    const MGLevelObject<AffineConstraints<double>> & mg_constraints_pressure);

  void
  prepare_schwarz_smoothers(const std::shared_ptr<ColoringBase<dim>> user_coloring);

  void
  assemble_multigrid(const MGLevelObject<AffineConstraints<double>> & mg_constraints_pressure);

  const PreconditionMG<dim, VECTOR, MG_TRANSFER> &
  get_preconditioner() const;

  MGParameter  parameters;
  EquationData equation_data;

  const DoFHandler<dim> * dof_handler;
  const DoFHandler<dim> * dof_handler_velocity;
  const DoFHandler<dim> * dof_handler_pressure;
  const Mapping<dim> *    mapping;

  std::shared_ptr<MGConstrainedDoFs>                mg_constrained_dofs;
  MG_TRANSFER                                       mg_transfer;
  Table<2, DoFTools::Coupling>                      cell_integrals_mask;
  Table<2, DoFTools::Coupling>                      face_integrals_mask;
  MGLevelObject<BlockSparsityPattern>               mg_sparsity_patterns;
  MGLevelObject<MATRIX>                             mg_matrices;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>        mg_schwarz_smoother_pre;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>        mg_schwarz_smoother_post;
  std::shared_ptr<const MGSmootherIdentity<VECTOR>> mg_smoother_identity;
  std::shared_ptr<const mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER, VECTOR>>
                                     mg_smoother_gauss_seidel;
  const MGSmootherBase<VECTOR> *     mg_smoother_pre;
  const MGSmootherBase<VECTOR> *     mg_smoother_post;
  CoarseGridSolver<MATRIX, VECTOR>   coarse_grid_solver;
  const MGCoarseGridBase<VECTOR> *   mg_coarse_grid;
  mg::Matrix<VECTOR>                 mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>> multigrid;

  mutable std::shared_ptr<const PreconditionMG<dim, VECTOR, MG_TRANSFER>> preconditioner_mg;
};

template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::
  MGCollectionVelocityPressure(const MGParameter &  mg_prms_in,
                               const EquationData & equation_data_in)
  : parameters(mg_prms_in),
    equation_data(equation_data_in),
    dof_handler_velocity(nullptr),
    dof_handler_pressure(nullptr),
    mapping(nullptr)
{
}

template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::clear()
{
  preconditioner_mg.reset();
  multigrid.reset();
  mg_matrix_wrapper.reset();
  mg_coarse_grid = nullptr;
  coarse_grid_solver.clear();
  mg_smoother_post = nullptr;
  mg_smoother_pre  = nullptr;
  mg_smoother_gauss_seidel.reset();
  mg_smoother_identity.reset();
  mg_schwarz_smoother_post.reset();
  mg_schwarz_smoother_pre.reset();
  mg_matrices.clear_elements();
  mg_transfer.clear();
  mg_constrained_dofs.reset();
}

// /// DEPRECATED
// template<int             dim,
//          int             fe_degree_p,
//          TPSS::DoFLayout dof_layout_v,
//          int             fe_degree_v,
//          LocalAssembly   local_assembly>
// MeshWorker::AssembleFlags
// MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::
//   get_mg_assemble_flags() const
// {
//   return get_assemble_flags_impl<dof_layout_v, true>();
// }

template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::
  assemble_multigrid(const MGLevelObject<AffineConstraints<double>> & mg_constraints_pressure)
{
  AssertDimension(mg_matrices.min_level(), parameters.coarse_level);
  Assert(mg_constrained_dofs, ExcMessage("mg_constrained_dofs is uninitialized."));
  Assert(dof_handler_velocity, ExcMessage("Did you set dof_handler_velocity?"));
  AssertThrow(TPSS::get_dof_layout(dof_handler->get_fe().base_element(0)) == dof_layout_v,
              ExcMessage("Velocity block of dof_handler and dof_layout_v are incompatible."));
  AssertThrow(TPSS::get_dof_layout(dof_handler_velocity->get_fe().base_element(0)) == dof_layout_v,
              ExcMessage("dof_handler_velocity and dof_layout_v are incompatible."));
  Assert(dof_handler_pressure, ExcMessage("Did you set dof_handler_pressure?"));

  constexpr bool use_sipg_method =
    dof_layout_v == TPSS::DoFLayout::DGQ || dof_layout_v == TPSS::DoFLayout::RT;
  constexpr bool use_conf_method = dof_layout_v == TPSS::DoFLayout::Q;

  for(unsigned int level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
  {
    /// As long as the DoF numbering of dof_handler_velocity and the velocity
    /// block of dof_handler is aligned, we might use the complete set of
    /// level_constraints as constraints for the velocity block.q
    AffineConstraints<double> level_constraints;
    if(mg_constrained_dofs->have_boundary_indices())
      level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    const auto & level_constraints_pressure = mg_constraints_pressure[level];
    // // DEBUG
    // const auto   entries = *(level_constraints_pressure.get_constraint_entries(0U));
    // for(const auto & [column, value] : entries)
    //   std::cout << " column: " << column << " value: " << value << std::endl;

    /// Initialize a (dummy) matrix-free storage for each level. This is
    /// required to initialize SubdomainHandlers for each level.
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.mg_level = level;
      std::vector<const DoFHandler<dim> *>           dof_handler_per_block{dof_handler_velocity,
                                                                 dof_handler_pressure};
      std::vector<const AffineConstraints<double> *> constraints_per_block{
        &level_constraints, &level_constraints_pressure};
      QGauss<1>  quadrature(n_q_points_1d);
      const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
      mf_storage->reinit(
        *mapping, dof_handler_per_block, constraints_per_block, quadrature, additional_data);

      //: initialize FD::MatrixIntegrator (see SparseMatrixAugmented)
      mg_matrices[level].initialize(mf_storage, equation_data);
    }

    /// Assemble velocity-velocity block first.
    {
      using Velocity::SIPG::MW::CopyData;
      using Velocity::SIPG::MW::ScratchData;
      using MatrixIntegrator  = Velocity::SIPG::MW::MatrixIntegrator<dim, true>;
      using LevelCellIterator = typename MatrixIntegrator::IteratorType;

      MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

      auto cell_worker =
        [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
          matrix_integrator.cell_worker(cell, scratch_data, copy_data);
        };

      auto face_worker = [&](const auto &         cell,
                             const unsigned int & f,
                             const unsigned int & sf,
                             const auto &         ncell,
                             const unsigned int & nf,
                             const unsigned int & nsf,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
        matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
      };

      auto boundary_worker = [&](const auto &         cell,
                                 const unsigned int & face_no,
                                 ScratchData<dim> &   scratch_data,
                                 CopyData &           copy_data) {
        matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
      };

      const auto copier = [&](const CopyData & copy_data) {
        level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          copy_data.cell_matrix, copy_data.local_dof_indices_test, mg_matrices[level].block(0, 0));

        for(auto & cdf : copy_data.face_data)
        {
          level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
            cdf.cell_matrix, cdf.joint_dof_indices_test, mg_matrices[level].block(0, 0));
        }
      };

      const UpdateFlags update_flags =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      const UpdateFlags interface_update_flags = update_values | update_gradients |
                                                 update_quadrature_points | update_JxW_values |
                                                 update_normal_vectors;
      ScratchData<dim> scratch_data(*mapping,
                                    dof_handler_velocity->get_fe(),
                                    dof_handler_velocity->get_fe(),
                                    n_q_points_1d,
                                    update_flags,
                                    update_flags,
                                    interface_update_flags,
                                    interface_update_flags);

      CopyData copy_data(dof_handler_velocity->get_fe().dofs_per_cell);

      if(use_conf_method)
        MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                              dof_handler_velocity->end_mg(level),
                              cell_worker,
                              copier,
                              scratch_data,
                              copy_data,
                              MeshWorker::assemble_own_cells);
      else if(use_sipg_method)
        MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                              dof_handler_velocity->end_mg(level),
                              cell_worker,
                              copier,
                              scratch_data,
                              copy_data,
                              MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                MeshWorker::assemble_own_interior_faces_once,
                              boundary_worker,
                              face_worker);
      else
        AssertThrow(false, ExcMessage("This FEM is not supported"));
    }

    /// Assemble pressure-pressure block.
    {
      /// This block is zero!
    }

    /// Assemble velocity-pressure and pressure-velocity blocks
    {
      using VelocityPressure::MW::Mixed::CopyData;
      using VelocityPressure::MW::Mixed::ScratchData;
      using MatrixIntegrator  = VelocityPressure::MW::Mixed::MatrixIntegrator<dim, true>;
      using LevelCellIterator = typename MatrixIntegrator::IteratorType;

      MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, nullptr, equation_data);
      const auto &     triangulation = dof_handler_pressure->get_triangulation();
      auto             cell_worker =
        [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
          LevelCellIterator cell_ansatz(&triangulation,
                                        cell->level(),
                                        cell->index(),
                                        dof_handler_pressure);
          matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
        };

      auto face_worker = [&](const LevelCellIterator & cell,
                             const unsigned int &      f,
                             const unsigned int &      sf,
                             const LevelCellIterator & ncell,
                             const unsigned int &      nf,
                             const unsigned int &      nsf,
                             ScratchData<dim> &        scratch_data,
                             CopyData &                copy_data) {
        LevelCellIterator cell_ansatz(&dof_handler_pressure->get_triangulation(),
                                      cell->level(),
                                      cell->index(),
                                      dof_handler_pressure);
        LevelCellIterator ncell_ansatz(&dof_handler_pressure->get_triangulation(),
                                       ncell->level(),
                                       ncell->index(),
                                       dof_handler_pressure);
        matrix_integrator.face_worker(
          cell, cell_ansatz, f, sf, ncell, ncell_ansatz, nf, nsf, scratch_data, copy_data);
      };

      auto boundary_worker = [&](const LevelCellIterator & cell,
                                 const unsigned int &      face_no,
                                 ScratchData<dim> &        scratch_data,
                                 CopyData &                copy_data) {
        LevelCellIterator cell_ansatz(&dof_handler_pressure->get_triangulation(),
                                      cell->level(),
                                      cell->index(),
                                      dof_handler_pressure);
        matrix_integrator.boundary_worker(cell, cell_ansatz, face_no, scratch_data, copy_data);
      };

      const auto copier = [&](const CopyData & copy_data) {
        level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          copy_data.cell_matrix,
          copy_data.local_dof_indices_test,
          level_constraints_pressure,
          copy_data.local_dof_indices_ansatz,
          mg_matrices[level].block(0, 1));
        level_constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
          copy_data.cell_matrix_flipped,
          copy_data.local_dof_indices_ansatz,
          level_constraints,
          copy_data.local_dof_indices_test,
          mg_matrices[level].block(1, 0));

        for(auto & cdf : copy_data.face_data)
        {
          AssertDimension(cdf.cell_rhs_test.size(), 0);
          AssertDimension(cdf.cell_rhs_ansatz.size(), 0);
          level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
            cdf.cell_matrix,
            cdf.joint_dof_indices_test,
            level_constraints_pressure,
            cdf.joint_dof_indices_ansatz,
            mg_matrices[level].block(0, 1));
          level_constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
            cdf.cell_matrix_flipped,
            cdf.joint_dof_indices_ansatz,
            level_constraints,
            cdf.joint_dof_indices_test,
            mg_matrices[level].block(1, 0));
        }
      };

      const UpdateFlags update_flags_velocity =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      const UpdateFlags update_flags_pressure =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      const UpdateFlags interface_update_flags_velocity =
        update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
      const UpdateFlags interface_update_flags_pressure =
        update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

      ScratchData<dim> scratch_data(*mapping,
                                    dof_handler_velocity->get_fe(),
                                    dof_handler_pressure->get_fe(),
                                    n_q_points_1d,
                                    update_flags_velocity,
                                    update_flags_pressure,
                                    interface_update_flags_velocity,
                                    interface_update_flags_pressure);

      CopyData copy_data(dof_handler_velocity->get_fe().dofs_per_cell,
                         dof_handler_pressure->get_fe().dofs_per_cell);

      if(use_conf_method)
        MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                              dof_handler_velocity->end_mg(level),
                              cell_worker,
                              copier,
                              scratch_data,
                              copy_data,
                              MeshWorker::assemble_own_cells);
      else if(use_sipg_method)
        MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                              dof_handler_velocity->end_mg(level),
                              cell_worker,
                              copier,
                              scratch_data,
                              copy_data,
                              MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                MeshWorker::assemble_own_interior_faces_once,
                              boundary_worker,
                              face_worker);
      else
        AssertThrow(false, ExcMessage("This FEM is not implemented."));
    }
  }
}

template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::
  prepare_schwarz_smoothers(const std::shared_ptr<ColoringBase<dim>> user_coloring)
{
  Assert(parameters.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz,
         ExcMessage("Invalid smoothing variant."));
  for(auto level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
    AssertThrow(mg_matrices[level].mf_storage, ExcMessage("mf_storage is not initialized."));

  //: pre-smoother
  {
    const auto                                   mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    typename MG_SMOOTHER_SCHWARZ::AdditionalData additional_data;
    if(parameters.pre_smoother.schwarz.userdefined_coloring)
    {
      Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
      additional_data.coloring_func = std::ref(*user_coloring);
    }
    additional_data.parameters = parameters.pre_smoother;
    additional_data.foreach_dofh.resize(2);
    additional_data.foreach_dofh[0].dirichlet_ids = equation_data.dirichlet_boundary_ids_velocity;
    additional_data.foreach_dofh[1].dirichlet_ids = equation_data.dirichlet_boundary_ids_pressure;
    // additional_data.foreach_dofh[1].force_no_boundary_condition = true;
    mgss->initialize(mg_matrices, additional_data);
    mg_schwarz_smoother_pre = mgss;
  }

  //: post-smoother (so far only shallow copy!)
  {
    const auto mgss_post = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    typename MG_SMOOTHER_SCHWARZ::AdditionalData additional_data;
    if(parameters.pre_smoother.schwarz.userdefined_coloring)
    {
      Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
      additional_data.coloring_func = std::ref(*user_coloring);
    }
    additional_data.parameters = parameters.post_smoother;
    mgss_post->initialize(*mg_schwarz_smoother_pre, additional_data);
    mg_schwarz_smoother_post = mgss_post;
  }
}

template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::
  prepare_multigrid(const unsigned int                               mg_level_max,
                    const std::shared_ptr<ColoringBase<dim>>         user_coloring,
                    const MGLevelObject<AffineConstraints<double>> & mg_constraints_pressure)
{
  Assert(dof_handler, ExcMessage("Did you set dof_handler?"));

  // *** clear multigrid infrastructure
  clear();

  // *** setup multigrid data
  const unsigned mg_level_min = parameters.coarse_level;

  // *** initialize multigrid constraints
  mg_constrained_dofs = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs->initialize(*dof_handler);
  const FEValuesExtractors::Vector velocities(0);
  if(dof_layout_v == TPSS::DoFLayout::Q)
    mg_constrained_dofs->make_zero_boundary_constraints(
      *dof_handler,
      equation_data.dirichlet_boundary_ids_velocity,
      dof_handler->get_fe().component_mask(velocities));

  AssertDimension(mg_level_min, mg_constraints_pressure.min_level());
  AssertDimension(mg_level_max, mg_constraints_pressure.max_level());
  if(equation_data.force_mean_value_constraint)
    for(auto level = mg_level_min; level <= mg_level_max; ++level)
    {
      AssertThrow(false, ExcMessage("TODO mean value constraint for MG matrices..."));
      //     const auto & level_constraints_pressure = mg_constraints_pressure[level];
      //     Assert(dof_handler_velocity, ExcMessage("dof_handler_velocity is uninitialized."));
      //     const types::global_dof_index n_dofs_velocity = dof_handler_velocity->n_dofs(level);
      //     const types::global_dof_index first_dof_index = n_dofs_velocity;
      //     AssertDimension(level_constraints_pressure.n_constraints(), 1U);
      //     AssertThrow(level_constraints_pressure.is_constrained(0U),
      //                 ExcMessage("Did you set a mean value free constraint?"));
      //     const auto mean_value_free_entries =
      //     *(level_constraints_pressure.get_constraint_entries(0U));

      //     /// shift dofs of the pressure block by n_dofs_velocity
      //     AffineConstraints<double> level_constraints;
      //     level_constraints.add_line(first_dof_index);
      //     for(const auto & [column, value] : mean_value_free_entries)
      //       level_constraints.add_entry(first_dof_index, first_dof_index + column, value);

      //     /// TODO this call throws an exception: internal IndexSets have no
      //     /// compatible size mg_constrained_dofs->add_user_constraints(level,
      //     /// level_constraints);
    }

  // *** initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  mg_sparsity_patterns.resize(mg_level_min, mg_level_max);
  std::vector<std::vector<types::global_dof_index>> level_to_dofs_per_block(
    mg_level_max + 1, std::vector<types::global_dof_index>(2, numbers::invalid_dof_index));
  MGTools::count_dofs_per_block(*dof_handler, level_to_dofs_per_block);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    const auto &                dofs_per_block = level_to_dofs_per_block[level];
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    MGTools::make_flux_sparsity_pattern(
      *dof_handler, dsp, level, cell_integrals_mask, face_integrals_mask);
    mg_sparsity_patterns[level].copy_from(dsp);
    mg_matrices[level].reinit(mg_sparsity_patterns[level]);
  }
  //: assemble the velocity system A_l on each level l.
  assemble_multigrid(mg_constraints_pressure);

  // *** initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(*dof_handler);

  // *** initialize Schwarz smoother S_l
  switch(parameters.pre_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      mg_smoother_identity = std::make_shared<const MGSmootherIdentity<VECTOR>>();
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::GaussSeidel:
    {
      // auto tmp = std::make_shared<mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER,
      // VECTOR>>();
      // tmp->initialize(mg_matrices);
      // tmp->set_steps(parameters.pre_smoother.n_smoothing_steps);
      // tmp->set_symmetric(true);
      // mg_smoother_gauss_seidel = tmp;
      // mg_smoother_pre          = mg_smoother_gauss_seidel.get();
    }
    break;
    case SmootherParameter::SmootherVariant::Schwarz:
      prepare_schwarz_smoothers(user_coloring);
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }
  switch(parameters.post_smoother.variant)
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
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Not initialized"));
      mg_smoother_post = mg_schwarz_smoother_post.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }

  // *** initialize coarse grid solver
  coarse_grid_solver.initialize(mg_matrices[mg_level_min], parameters.coarse_grid);
  mg_coarse_grid = &coarse_grid_solver;

  // *** initialize geometric multigrid method
  mg_matrix_wrapper.initialize(mg_matrices);
  multigrid = std::make_shared<Multigrid<VECTOR>>(mg_matrix_wrapper,
                                                  *mg_coarse_grid,
                                                  mg_transfer,
                                                  *mg_smoother_pre,
                                                  *mg_smoother_post,
                                                  mg_level_min,
                                                  mg_level_max);
}

template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
const PreconditionMG<dim,
                     typename MGCollectionVelocityPressure<dim,
                                                           fe_degree_p,
                                                           dof_layout_v,
                                                           fe_degree_v,
                                                           local_assembly>::VECTOR,
                     typename MGCollectionVelocityPressure<dim,
                                                           fe_degree_p,
                                                           dof_layout_v,
                                                           fe_degree_v,
                                                           local_assembly>::MG_TRANSFER> &
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>::
  get_preconditioner() const
{
  AssertThrow(multigrid, ExcMessage("multigrid is uninitialized."));
  preconditioner_mg = std::make_shared<PreconditionMG<dim, VECTOR, MG_TRANSFER>>(*dof_handler,
                                                                                 *multigrid,
                                                                                 mg_transfer);
  return *preconditioner_mg;
}



/**
 * The base class defines compile time parameters depending on the choice of the
 * finite element method.
 */
template<Method method, int dim, int fe_degree_p>
struct ModelProblemBase
{
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::TaylorHood, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::Q;
  using fe_type_v                               = FE_Q<dim>;
  using fe_type_p                               = FE_Q<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Tensor;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::TaylorHoodDGQ, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::DGQ;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::Q;
  using fe_type_v                               = FE_DGQ<dim>;
  using fe_type_p                               = FE_Q<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Tensor;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::Qkplus2_DGPk, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGP;
  using fe_type_v                               = FE_Q<dim>;
  using fe_type_p                               = FE_DGP<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 2;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Cut;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::DGQkplus2_DGPk, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::DGQ;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGP;
  using fe_type_v                               = FE_DGQ<dim>;
  using fe_type_p                               = FE_DGP<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Cut;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::RaviartThomas, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::RT;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGQ;
  // using fe_type_v                               = FE_RaviartThomasNodal_new<dim>;
  using fe_type_v                               = FE_RaviartThomas<dim>;
  using fe_type_p                               = FE_DGQLegendre<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Cut;
};


/**
 * This is the main class which combines the typical building blocks of a
 * (deal.II) finite element method.
 *
 * TODO Description...
 */
template<int dim, int fe_degree_p, Method method = Method::TaylorHood>
class ModelProblem : public ModelProblemBase<method, dim, fe_degree_p>
{
  static_assert(dim == 2, "only 2D");

  using Base = ModelProblemBase<method, dim, fe_degree_p>;

public:
  using Base::dof_layout_p;
  using Base::dof_layout_v;
  using Base::fe_type_p;
  using Base::fe_type_v;
  using Base::local_assembly;

  static constexpr int fe_degree_v = Base::fe_degree_v;
  static constexpr int n_q_points_1d =
    fe_degree_v + 1 + (dof_layout_v == TPSS::DoFLayout::RT ? 1 : 0);

  ModelProblem(const RT::Parameter & rt_parameters_in, const EquationData & equation_data_in);

  void
  run();

  bool
  make_grid();

  bool
  make_grid(const unsigned int n_refinements);

  void
  setup_system();

  std::shared_ptr<Vector<double>>
  compute_mass_foreach_pressure_dof() const;

  std::shared_ptr<Vector<double>>
  compute_mass_foreach_pressure_dof(const unsigned int level) const;

  void
  setup_system_velocity(const bool do_cuthill_mckee);

  void
  setup_system_pressure(const bool do_cuthill_mckee);

  /// DEPRECATED
  // MeshWorker::AssembleFlags
  // get_assemble_flags() const;

  void
  assemble_system();

  void
  assemble_system_step56();

  void
  assemble_system_velocity_pressure();

  void
  assemble_system_velocity();

  void
  prepare_multigrid_velocity();

  void
  prepare_multigrid_velocity_pressure();

  void
  solve();

  void
  correct_mean_value_pressure();

  std::shared_ptr<Vector<double>>
  compute_L2_error_velocity() const;

  std::shared_ptr<Vector<double>>
  compute_L2_error_pressure() const;

  std::shared_ptr<Vector<double>>
  compute_H1semi_error_velocity() const;

  void
  compute_errors();

  void
  output_results(const unsigned int refinement_cycle) const;

  unsigned int
  max_level() const;

  unsigned int
  n_colors_system();

  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const;

  void
  print_informations() const;

  const FiniteElement<dim> &
  get_fe_velocity() const;

  const FiniteElement<dim> &
  get_fe_pressure() const;

  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  std::shared_ptr<Function<dim>>      analytical_solution;
  std::shared_ptr<Function<dim>>      load_function;
  mutable PostProcessData             pp_data;
  mutable PostProcessData             pp_data_pressure;

  Triangulation<dim>                  triangulation;
  MappingQ<dim>                       mapping;
  std::shared_ptr<FiniteElement<dim>> fe;
  DoFHandler<dim>                     dof_handler;
  DoFHandler<dim>                     dof_handler_velocity;
  DoFHandler<dim>                     dof_handler_pressure;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> zero_constraints_velocity;
  AffineConstraints<double> mean_value_constraints;
  AffineConstraints<double> constraints_velocity;
  AffineConstraints<double> constraints_pressure;

  Table<2, DoFTools::Coupling>      cell_integrals_mask;
  Table<2, DoFTools::Coupling>      face_integrals_mask;
  BlockSparsityPattern              sparsity_pattern;
  BlockSparseMatrixFiltered<double> system_matrix; // !!!
  SparsityPattern                   sparsity_pattern_pressure;
  SparseMatrix<double>              pressure_mass_matrix;

  BlockVector<double> system_solution;
  BlockVector<double> system_delta_x;
  BlockVector<double> system_rhs;
  Vector<double>      constant_pressure_mode;

  //: multigrid
  mutable std::shared_ptr<ColoringBase<dim>>           user_coloring;
  MGCollectionVelocity<dim, fe_degree_v, dof_layout_v> mgc_velocity;
  MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v, fe_degree_v, local_assembly>
    mgc_velocity_pressure;

private:
  std::shared_ptr<FiniteElement<dim>>
  generate_fe() const;

  bool
  check_finite_elements() const;

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner, const std::string solver_variant);

  void
  make_grid_impl(const MeshParameter & mesh_prms);

  template<bool is_multigrid = false>
  std::shared_ptr<Vector<double>>
  compute_mass_foreach_pressure_dof_impl(
    const unsigned int level = numbers::invalid_unsigned_int) const;
};



template<int dim, int fe_degree_p, Method method>
ModelProblem<dim, fe_degree_p, method>::ModelProblem(const RT::Parameter & rt_parameters_in,
                                                     const EquationData &  equation_data_in)
  : pcout(
      std::make_shared<ConditionalOStream>(std::cout,
                                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    rt_parameters(rt_parameters_in),
    equation_data(equation_data_in),
    analytical_solution([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::DivFree)
        return std::make_shared<DivergenceFree::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipNormal)
        return std::make_shared<DivergenceFree::NoSlipNormal::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeBell)
        return std::make_shared<DivergenceFree::GaussianBell::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleNoSlip)
        return std::make_shared<DivergenceFree::Poiseuille::NoSlip::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlip)
        return std::make_shared<DivergenceFree::NoSlip::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleInhom)
        return std::make_shared<DivergenceFree::Poiseuille::Inhom::Solution<dim>>();
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    load_function([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::DivFree)
        return std::make_shared<DivergenceFree::Load<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipNormal)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeBell)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleNoSlip)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlip)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution); // !!!
      // return std::make_shared<DivergenceFree::NoSlip::Load<dim>>(); // !!!
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleInhom)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    triangulation(Triangulation<dim>::maximum_smoothing),
    mapping(1),
    // Finite element for the whole system:
    fe(generate_fe()),
    dof_handler(triangulation),
    /// TODO !!! coloring depends on the discrete pressure space as well
    user_coloring([&]() -> std::shared_ptr<ColoringBase<dim>> {
      if constexpr(dof_layout_v == TPSS::DoFLayout::Q)
        return std::make_shared<TiledColoring<dim>>(rt_parameters_in.mesh);
      else if(dof_layout_v == TPSS::DoFLayout::DGQ)
        return std::make_shared<RedBlackColoring<dim>>(rt_parameters_in.mesh);
      return std::shared_ptr<ColoringBase<dim>>();
    }()),
    mgc_velocity(rt_parameters_in.multigrid, equation_data_in),
    mgc_velocity_pressure(rt_parameters_in.multigrid, equation_data_in)
{
  Assert(check_finite_elements(), ExcMessage("Check default finite elements and dof_layout."));
}



template<int dim, int fe_degree_p, Method method>
unsigned int
ModelProblem<dim, fe_degree_p, method>::max_level() const
{
  return triangulation.n_global_levels() - 1;
}



template<int dim, int fe_degree_p, Method method>
unsigned int
ModelProblem<dim, fe_degree_p, method>::n_colors_system()
{
  auto & mgc = mgc_velocity;
  if(mgc.mg_schwarz_smoother_pre)
    return mgc.mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
  return numbers::invalid_unsigned_int;
}



template<int dim, int fe_degree_p, Method method>
template<typename T>
void
ModelProblem<dim, fe_degree_p, method>::print_parameter(const std::string & description,
                                                        const T &           value) const
{
  *pcout << Util::parameter_to_fstring(description, value);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::print_informations() const
{
  *pcout << equation_data.to_string();
  *pcout << std::endl;
  print_parameter("Finite element:", fe->get_name());
  *pcout << rt_parameters.to_string();
  *pcout << std::endl;
}



/// DEPRECATED
// template<int dim, int fe_degree_p, Method method>
// MeshWorker::AssembleFlags
// ModelProblem<dim, fe_degree_p, method>::get_assemble_flags() const
// {
//   return get_assemble_flags_impl<dof_layout_v, false>();
// }



template<int dim, int fe_degree_p, Method method>
const FiniteElement<dim> &
ModelProblem<dim, fe_degree_p, method>::get_fe_velocity() const
{
  std::vector<bool> velocity_mask(dim, true);
  velocity_mask.push_back(false);
  const auto & fe_velocity = fe->get_sub_fe(velocity_mask);
  return fe_velocity;
}


template<int dim, int fe_degree_p, Method method>
const FiniteElement<dim> &
ModelProblem<dim, fe_degree_p, method>::get_fe_pressure() const
{
  std::vector<bool> pressure_mask(dim, false);
  pressure_mask.push_back(true);
  const auto & fe_pressure = fe->get_sub_fe(pressure_mask);
  return fe_pressure;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<FiniteElement<dim>>
ModelProblem<dim, fe_degree_p, method>::generate_fe() const
{
  if constexpr(dof_layout_v == TPSS::DoFLayout::RT)
    return std::make_shared<FESystem<dim>>(typename Base::fe_type_v(fe_degree_v),
                                           1,
                                           typename Base::fe_type_p(fe_degree_p),
                                           1);
  else
    return std::make_shared<FESystem<dim>>(FESystem<dim>(typename Base::fe_type_v(fe_degree_v),
                                                         dim),
                                           1,
                                           typename Base::fe_type_p(fe_degree_p),
                                           1);
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::check_finite_elements() const
{
  //: check fe
  AssertDimension(fe->n_base_elements(), 2); // velocity + pressure
  const auto & fe_v = fe->base_element(0);
  AssertDimension(fe_v.n_components(), dim); // velocity
  AssertDimension(fe_v.element_multiplicity(0), dof_layout_v == TPSS::DoFLayout::RT ? 1 : dim);
  AssertThrow(TPSS::get_dof_layout(fe_v.base_element(0)) == dof_layout_v,
              ExcMessage("velocity part of fe and dof_layout are incompatible."));
  const auto & fe_p = fe->base_element(1);
  AssertDimension(fe_p.n_components(), 1); // pressure

  return true;
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::make_grid()
{
  make_grid_impl(rt_parameters.mesh);
  return true;
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::make_grid(const unsigned int n_refinements)
{
  MeshParameter mesh_prms = rt_parameters.mesh;
  mesh_prms.n_refinements = n_refinements;

  //: estimate number of dofs (velocity + pressure)
  AssertDimension(fe->n_base_elements(), 2); // velocity + pressure
  const auto & fe_v = fe->base_element(0);
  AssertDimension(fe_v.n_components(), dim); // velocity
  const auto n_dofs_est_v = [&]() {
    if(fe_v.element_multiplicity(0) == dim)
      return dim * estimate_n_dofs(fe_v.base_element(0), mesh_prms);
    return estimate_n_dofs(fe_v, mesh_prms);
  }();
  const auto & fe_p = fe->base_element(1);
  AssertDimension(fe_p.n_components(), 1); // pressure
  const auto n_dofs_est_p = estimate_n_dofs(fe_p, mesh_prms);
  const auto n_dofs_est   = n_dofs_est_v + n_dofs_est_p;
  if(rt_parameters.exceeds_dof_limits(n_dofs_est))
    return false;

  make_grid_impl(mesh_prms);
  return true;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::make_grid_impl(const MeshParameter & mesh_prms)
{
  triangulation.clear();
  *pcout << create_mesh(triangulation, mesh_prms) << std::endl;
  pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());
  pp_data.n_dimensions = dim;
}



template<int dim, int fe_degree_p, Method method>
template<bool is_multigrid>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_mass_foreach_pressure_dof_impl(
  const unsigned int level) const
{
  const auto n_dofs_pressure = [this, level]() {
    if(is_multigrid)
      return dof_handler_pressure.n_dofs(level);
    else
      return dof_handler_pressure.n_dofs();
  }();
  std::shared_ptr<Vector<double>> mass_foreach_dof_ptr; // book-keeping
  mass_foreach_dof_ptr = std::make_shared<Vector<double>>(n_dofs_pressure);

  AffineConstraints<double> constraints_pressure;
  constraints_pressure.clear();
  constraints_pressure.close();
  DynamicSparsityPattern dsp(n_dofs_pressure);
  if(is_multigrid)
    MGTools::make_sparsity_pattern(dof_handler_pressure, dsp, level);
  else
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp, constraints_pressure);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> pressure_mass_matrix;
  pressure_mass_matrix.reinit(sparsity_pattern);

  using Pressure::MW::CopyData;
  using Pressure::MW::ScratchData;
  using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, is_multigrid>;

  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    matrix_integrator.cell_mass_worker(cell, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
      copy_data.cell_matrix, copy_data.local_dof_indices, pressure_mass_matrix);
  };

  const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
  const UpdateFlags interface_update_flags = update_default;

  ScratchData<dim> scratch_data(
    mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);
  CopyData copy_data(dof_handler_pressure.get_fe().dofs_per_cell);
  if(is_multigrid)
    MeshWorker::mesh_loop(dof_handler_pressure.begin_mg(level),
                          dof_handler_pressure.end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  else
    MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
                          dof_handler_pressure.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);

  Vector<double> & mass_foreach_dof = *mass_foreach_dof_ptr;
  Vector<double>   constant_one_vector(n_dofs_pressure);
  std::fill(constant_one_vector.begin(), constant_one_vector.end(), 1.);
  pressure_mass_matrix.vmult(mass_foreach_dof, constant_one_vector);

  AssertThrow(mass_foreach_dof(0) > 0., ExcMessage("First DoF has no positive mass."));
  return mass_foreach_dof_ptr;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_mass_foreach_pressure_dof() const
{
  return compute_mass_foreach_pressure_dof_impl<false>();
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_mass_foreach_pressure_dof(
  const unsigned int level) const
{
  return compute_mass_foreach_pressure_dof_impl<true>(level);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system_velocity(const bool do_cuthill_mckee)
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  //: distribute DoFs and initialize MGCollection
  dof_handler_velocity.initialize(triangulation, get_fe_velocity());
  if(do_cuthill_mckee)
    DoFRenumbering::Cuthill_McKee(dof_handler_velocity);

  /// homogeneous boundary conditions for the solution update
  zero_constraints_velocity.clear();
  if(dof_layout_v == TPSS::DoFLayout::Q)
  {
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      DoFTools::make_zero_boundary_constraints(dof_handler_velocity,
                                               boundary_id,
                                               zero_constraints_velocity);
  }
  else if(dof_layout_v == TPSS::DoFLayout::RT)
  {
    Functions::ZeroFunction<dim> zero_velocity(dim);
    /// We use dof_handler by purpose here bypassing the assertion in
    /// project_boundary_values_div_conforming(), since the underlying finite
    /// element is of type FE_System and therefore not checked (otherwise the
    /// function presumes a finite element of type FE_RaviartThomas!).
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      VectorToolsFix::project_boundary_values_div_conforming(
        dof_handler_velocity, 0U, zero_velocity, boundary_id, zero_constraints_velocity, mapping);
  }
  zero_constraints_velocity.close();

  /// inhomogeneous boundary conditions for the actual solution
  constraints_velocity.clear();
  if(dof_layout_v == TPSS::DoFLayout::Q)
  {
    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
    std::map<types::boundary_id, const Function<dim, double> *> boundary_id_to_function;
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      boundary_id_to_function.emplace(boundary_id, &analytical_solution_velocity);
    VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                             boundary_id_to_function,
                                             constraints_velocity);
  }
  else if(dof_layout_v == TPSS::DoFLayout::RT)
  {
    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
    /// We use dof_handler by purpose here bypassing the assertion in
    /// project_boundary_values_div_conforming(), since the underlying finite
    /// element is of type FE_System and therefore not checked (otherwise the
    /// function presumes a finite element of type FE_RaviartThomas!).
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      VectorToolsFix::project_boundary_values_div_conforming(dof_handler_velocity,
                                                             0U,
                                                             analytical_solution_velocity,
                                                             boundary_id,
                                                             constraints_velocity,
                                                             mapping); // !!!
  }
  constraints_velocity.close();

  // //: setup system matrix (DEBUG)
  // if(do_system_matrix)
  // {
  //   Assert(constraints_velocity.n_constraints() != 0,
  //          ExcMessage("Did you initialize the velocity constraints?"));
  //   mgc_velocity.system_matrix.clear();
  //   DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs());
  //   DoFTools::make_flux_sparsity_pattern(dof_handler_velocity,
  //                                        dsp,
  //                                        constraints_velocity,
  //                                        rt_parameters.solver.variant == "UMFPACK" ? true :
  //                                        false);
  //   mgc_velocity.sparsity_pattern.copy_from(dsp);
  //   mgc_velocity.system_matrix.reinit(mgc_velocity.sparsity_pattern);
  // }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system_pressure(const bool do_cuthill_mckee)
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));
  dof_handler_pressure.initialize(triangulation, get_fe_pressure());

  if(do_cuthill_mckee)
  {
    const bool cuthill_mckee_is_compatible = dof_layout_v == dof_layout_p;
    AssertThrow(
      cuthill_mckee_is_compatible,
      ExcMessage(
        "The concatenation of the Cuthill-Mckee renumberings on the velocity and pressure dofs each does not result in the same Cuthill-McKee renumbering applied to the whole velocity-pressure block system. They should be compatible if the velocity components and pressure functions have the same dof layout."));
    DoFRenumbering::Cuthill_McKee(dof_handler_pressure);
  }

  /// Restrict the pressure to be from the mean-value free L2-space, that is
  /// L^2_0 (\Omega) following the notation in Guido's mixed FEM script.
  ///
  /// Therefore, we first compute the (unconstrained) pressure mass matrix and
  /// then apply the coefficient vector associated to the constant one function
  /// to obtain the weights for each degree of freedom. By means of these
  /// weights we impose a mean value free constraint.
  constraints_pressure.clear();
  if(equation_data.force_mean_value_constraint)
  {
    const auto   mass_foreach_dof_ptr = compute_mass_foreach_pressure_dof();
    const auto & mass_foreach_dof     = *mass_foreach_dof_ptr;

    const bool is_dgq_legendre =
      dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
    const bool is_legendre_type = is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP;
    const auto n_dofs_pressure  = dof_handler_pressure.n_dofs();
    {
      const double mass_of_first_dof = mass_foreach_dof(0);
      constraints_pressure.add_line(0U);
      const auto         n_dofs_per_cell = get_fe_pressure().dofs_per_cell;
      const unsigned int stride          = is_legendre_type ? n_dofs_per_cell : 1U;
      const unsigned int start           = is_legendre_type ? stride : 1U;
      for(auto i = start; i < n_dofs_pressure; i += stride)
        constraints_pressure.add_entry(0U, i, -mass_foreach_dof(i) / mass_of_first_dof);
    }
  }
  constraints_pressure.close();
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system()
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  dof_handler.initialize(triangulation, *fe);

  // ILU behaves better if we apply a reordering to reduce fillin. There
  // is no advantage in doing this for the other solvers.
  if(equation_data.use_cuthill_mckee)
  {
    const bool cuthill_mckee_is_compatible = dof_layout_v == dof_layout_p;
    const bool cuthill_mckee_pays_off =
      rt_parameters.solver.variant == "FGMRES_ILU" && cuthill_mckee_is_compatible;
    AssertThrow(
      cuthill_mckee_pays_off,
      ExcMessage(
        "For the current solver setting a Cuthill-McKee renumbering seems not be worthwile."));
    DoFRenumbering::Cuthill_McKee(dof_handler);
  }

  // This ensures that all velocity DoFs are enumerated before the pressure
  // unknowns. This allows us to use blocks for vectors and matrices and allows
  // us to get the same DoF numbering for dof_handler and its unmerged
  // counterparts dof_handler_velocity and dof_handler_pressure.
  DoFRenumbering::block_wise(dof_handler);
  std::vector<unsigned int>                  block_component{0U, 1U};
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_dofs_velocity = dofs_per_block[0];
  const unsigned int n_dofs_pressure = dofs_per_block[1];

  setup_system_velocity(equation_data.use_cuthill_mckee);

  setup_system_pressure(equation_data.use_cuthill_mckee);

  /// No-slip boundary conditions (velocity)
  zero_constraints.clear();
  if(dof_layout_v == TPSS::DoFLayout::Q)
  {
    const FEValuesExtractors::Vector velocities(0);
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               boundary_id,
                                               zero_constraints,
                                               fe->component_mask(velocities));
  }
  else if(dof_layout_v == TPSS::DoFLayout::RT)
  {
    Functions::ZeroFunction<dim> zero_velocity(dim);
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      VectorToolsFix::project_boundary_values_div_conforming(
        dof_handler, 0U, zero_velocity, boundary_id, zero_constraints, mapping);
  }
  zero_constraints.close();

  {
    mean_value_constraints.clear();

    /// Mean value condition (pressure) ...
    if(equation_data.force_mean_value_constraint)
    {
      const types::global_dof_index first_dof_index = n_dofs_velocity;
      AssertDimension(constraints_pressure.n_constraints(), 1U);
      AssertThrow(constraints_pressure.is_constrained(0U),
                  ExcMessage("Did you set a mean value free constraint?"));
      const auto mean_value_free_entries = *(constraints_pressure.get_constraint_entries(0U));

      /// shift dofs of the pressure block by n_dofs_velocity
      mean_value_constraints.add_line(first_dof_index);
      for(const auto & [column, value] : mean_value_free_entries)
        mean_value_constraints.add_entry(first_dof_index, first_dof_index + column, value);
    }
    /// ... or mean value filter
    else
    {
      constant_pressure_mode.reinit(n_dofs_pressure);
      const bool is_dgq_legendre =
        dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
      if(is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP)
      {
        const auto n_dofs_per_cell = dof_handler_pressure.get_fe().dofs_per_cell;
        AssertDimension(n_dofs_pressure % n_dofs_per_cell, 0);
        for(auto i = 0U; i < n_dofs_pressure; i += n_dofs_per_cell)
          constant_pressure_mode[i] = 1.;
      }
      else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
      {
        constant_pressure_mode = 1.;
      }
      else
        AssertThrow(false, ExcMessage("This pressure dof layout is not supported."));
    }

    mean_value_constraints.close();
  }

  // // 2!!! clear all constraints
  // zero_constraints.clear();
  // zero_constraints.close();
  // zero_constraints_velocity.clear();
  // zero_constraints_velocity.close();
  // constraints_velocity.clear();
  // constraints_velocity.close();

  zero_constraints.merge(mean_value_constraints);

  cell_integrals_mask.reinit(dim + 1, dim + 1);
  face_integrals_mask.reinit(dim + 1, dim + 1);
  for(auto i = 0U; i < dim + 1; ++i)
    for(auto j = 0U; j < dim + 1; ++j)
    {
      cell_integrals_mask(i, j) = DoFTools::Coupling::always;
      if(dof_layout_v == TPSS::DoFLayout::DGQ || dof_layout_v == TPSS::DoFLayout::RT)
        face_integrals_mask(i, j) = DoFTools::Coupling::always;
      else if(dof_layout_v == TPSS::DoFLayout::Q)
        face_integrals_mask(i, j) = DoFTools::Coupling::none;
      else
        AssertThrow(false, ExcMessage("This dof layout is not supported."));
    }

  system_matrix.clear();
  pressure_mass_matrix.clear();
  BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, cell_integrals_mask, face_integrals_mask);
  zero_constraints.condense(dsp);
  sparsity_pattern.copy_from(dsp);
  if(constant_pressure_mode.size() != 0U)
    system_matrix.constant_pressure_mode = &constant_pressure_mode; // !!!

  system_matrix.reinit(sparsity_pattern);
  system_solution.reinit(dofs_per_block);
  system_delta_x.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);

  zero_constraints.set_zero(system_solution);
  zero_constraints.set_zero(system_delta_x);
  constraints_velocity.distribute(system_solution.block(0));     // particular velocity solution!
  zero_constraints_velocity.distribute(system_delta_x.block(0)); // hom velocity solution !!!

  print_parameter("Number of degrees of freedom (velocity):", n_dofs_velocity);
  print_parameter("Number of degrees of freedom (pressure):", n_dofs_pressure);
  print_parameter("Number of degrees of freedom (total):", n_dofs_velocity + n_dofs_pressure);
  *pcout << std::endl;

  pp_data.n_dofs_global.push_back(n_dofs_velocity);
  pp_data_pressure.n_dofs_global.push_back(n_dofs_pressure);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system_step56()
{
  using VelocityPressure::MW::CopyData;
  using VelocityPressure::MW::ScratchData;
  using MatrixIntegrator = VelocityPressure::MW::MatrixIntegrator<dim, /*is_multigrid*/ false>;

  const auto *     particular_solution = &system_solution;
  MatrixIntegrator matrix_integrator(load_function.get(),
                                     analytical_solution.get(),
                                     particular_solution,
                                     equation_data);

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    matrix_integrator.cell_worker(cell, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    zero_constraints
      .template distribute_local_to_global<BlockSparseMatrix<double>, BlockVector<double>>(
        copy_data.cell_matrix,
        copy_data.cell_rhs,
        copy_data.local_dof_indices,
        system_matrix,
        system_rhs);
  };

  ScratchData<dim> scratch_data(mapping,
                                *fe,
                                n_q_points_1d,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values | update_normal_vectors);
  CopyData         copy_data(dof_handler.get_fe().dofs_per_cell);
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system_velocity_pressure()
{
  constexpr bool use_sipg_method    = dof_layout_v == TPSS::DoFLayout::DGQ;
  constexpr bool use_hdiv_ip_method = dof_layout_v == TPSS::DoFLayout::RT;
  constexpr bool use_conf_method    = dof_layout_v == TPSS::DoFLayout::Q;

  /// Assemble the velocity block, here block(0,0).
  {
    using Velocity::SIPG::MW::CopyData;
    using Velocity::SIPG::MW::ScratchData;
    using MatrixIntegrator = Velocity::SIPG::MW::MatrixIntegrator<dim, false>;

    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> load_function_velocity(load_function.get(), component_range);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);

    const auto * particular_solution_velocity =
      (use_conf_method || use_hdiv_ip_method) ? &(system_solution.block(0)) : nullptr;

    MatrixIntegrator matrix_integrator(&load_function_velocity,
                                       &analytical_solution_velocity,
                                       particular_solution_velocity,
                                       equation_data);

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & f,
                           const unsigned int & sf,
                           const auto &         ncell,
                           const unsigned int & nf,
                           const unsigned int & nsf,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      if(use_sipg_method)
        matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
      else if(use_hdiv_ip_method)
        // matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
        matrix_integrator.face_worker_tangential(
          cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This velocity dof layout is not supported."));
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      if(use_sipg_method)
        matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
      else if(use_hdiv_ip_method)
        // matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
        matrix_integrator.boundary_worker_tangential(cell, face_no, scratch_data, copy_data);
      // matrix_integrator.boundary_worker_tangential_old(cell, face_no, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This velocity dof layout is not supported."));
    };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints_velocity
        .template distribute_local_to_global<SparseMatrix<double>, Vector<double>>(
          copy_data.cell_matrix,
          copy_data.cell_rhs_test,
          copy_data.local_dof_indices_test,
          system_matrix.block(0, 0),
          system_rhs.block(0));

      for(auto & cdf : copy_data.face_data)
      {
        if(!use_hdiv_ip_method)
          AssertDimension(cdf.cell_rhs_test.size(), 0);

        if(cdf.cell_rhs_test.size() == 0)
          zero_constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
            cdf.cell_matrix, cdf.joint_dof_indices_test, system_matrix.block(0, 0));
        else
        {
          zero_constraints_velocity
            .template distribute_local_to_global<SparseMatrix<double>, Vector<double>>(
              cdf.cell_matrix,
              cdf.cell_rhs_test,
              cdf.joint_dof_indices_test,
              system_matrix.block(0, 0),
              system_rhs.block(0));
        }
      }
    };

    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                               update_quadrature_points | update_JxW_values |
                                               update_normal_vectors;

    ScratchData<dim> scratch_data(mapping,
                                  dof_handler_velocity.get_fe(),
                                  dof_handler_velocity.get_fe(),
                                  n_q_points_1d,
                                  update_flags,
                                  update_flags,
                                  interface_update_flags,
                                  interface_update_flags);

    CopyData copy_data(dof_handler_velocity.get_fe().dofs_per_cell);

    if(use_conf_method)
      MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                            dof_handler_velocity.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method || use_hdiv_ip_method)
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
    else
      AssertThrow(false, ExcMessage("This FEM is not implemented."));
  }

  /// Assemble the pressure block, here block(1,1).
  {
    using Pressure::MW::CopyData;
    using Pressure::MW::ScratchData;
    using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, false>;

    const auto             component_range = std::make_pair<unsigned int>(dim, dim + 1);
    FunctionExtractor<dim> load_function_pressure(load_function.get(), component_range);

    MatrixIntegrator matrix_integrator(&load_function_pressure, nullptr, nullptr, equation_data);

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    const auto copier = [&](const CopyData & copy_data) {
      constraints_pressure
        .template distribute_local_to_global<SparseMatrix<double>, Vector<double>>(
          copy_data.cell_matrix,
          copy_data.cell_rhs,
          copy_data.local_dof_indices,
          system_matrix.block(1, 1),
          system_rhs.block(1));
    };

    const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_default;

    ScratchData<dim> scratch_data(
      mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);

    CopyData copy_data(dof_handler_pressure.get_fe().dofs_per_cell);

    MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
                          dof_handler_pressure.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

  /// Assemble the mixed velocity-pressure block, that is test functions are
  /// from velocity space and ansatz functions from pressure space. The
  /// "flipped" pressure-velocity block is assembled as well.
  {
    using VelocityPressure::MW::Mixed::CopyData;
    using VelocityPressure::MW::Mixed::ScratchData;
    using MatrixIntegrator = VelocityPressure::MW::Mixed::MatrixIntegrator<dim, false>;
    using CellIterator     = typename MatrixIntegrator::IteratorType;

    const auto * particular_solution_velocity =
      (use_conf_method || use_hdiv_ip_method) ? &(system_solution.block(0)) : nullptr;

    const auto             component_range_velocity = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(),
                                                        component_range_velocity);

    MatrixIntegrator matrix_integrator(particular_solution_velocity,
                                       /*particular_solution_pressure*/ nullptr,
                                       &analytical_solution_velocity,
                                       /*&analytical_solution_pressure*/ nullptr,
                                       equation_data);

    auto cell_worker =
      [&](const CellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                                 cell->level(),
                                 cell->index(),
                                 &dof_handler_pressure);
        matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & f,
                           const unsigned int & sf,
                           const auto &         ncell,
                           const unsigned int & nf,
                           const unsigned int & nsf,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                               cell->level(),
                               cell->index(),
                               &dof_handler_pressure);
      CellIterator ncell_ansatz(&dof_handler_pressure.get_triangulation(),
                                ncell->level(),
                                ncell->index(),
                                &dof_handler_pressure);
      if(use_sipg_method)
        matrix_integrator.face_worker(
          cell, cell_ansatz, f, sf, ncell, ncell_ansatz, nf, nsf, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This FEM is not supported."));
    };

    auto boundary_worker = [&](const CellIterator & cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                               cell->level(),
                               cell->index(),
                               &dof_handler_pressure);
      if(use_sipg_method)
        matrix_integrator.boundary_worker(cell, cell_ansatz, face_no, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This FEM is not supported."));
    };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix,
        copy_data.local_dof_indices_test,
        constraints_pressure,
        copy_data.local_dof_indices_ansatz,
        system_matrix.block(0, 1));
      constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix_flipped,
        copy_data.local_dof_indices_ansatz,
        zero_constraints_velocity,
        copy_data.local_dof_indices_test,
        system_matrix.block(1, 0));

      zero_constraints_velocity.template distribute_local_to_global<Vector<double>>(
        copy_data.cell_rhs_test, copy_data.local_dof_indices_test, system_rhs.block(0));
      constraints_pressure.template distribute_local_to_global<Vector<double>>(
        copy_data.cell_rhs_ansatz, copy_data.local_dof_indices_ansatz, system_rhs.block(1));

      for(auto & cdf : copy_data.face_data)
      {
        zero_constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix,
          cdf.joint_dof_indices_test,
          constraints_pressure,
          cdf.joint_dof_indices_ansatz,
          system_matrix.block(0, 1));
        constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix_flipped,
          cdf.joint_dof_indices_ansatz,
          zero_constraints_velocity,
          cdf.joint_dof_indices_test,
          system_matrix.block(1, 0));

        /// For Hdiv-IP there might be liftings from the velocity written to the
        /// pressure RHS.
        AssertDimension(cdf.cell_rhs_test.size(), 0);
        if(dof_layout_v != TPSS::DoFLayout::RT)
          AssertDimension(cdf.cell_rhs_ansatz.size(), 0);

        if(cdf.cell_rhs_test.size() != 0)
          zero_constraints_velocity.template distribute_local_to_global<Vector<double>>(
            cdf.cell_rhs_test, cdf.joint_dof_indices_test, system_rhs.block(0));
        if(cdf.cell_rhs_ansatz.size() != 0)
          constraints_pressure.template distribute_local_to_global<Vector<double>>(
            cdf.cell_rhs_ansatz, cdf.joint_dof_indices_ansatz, system_rhs.block(1));
      }
    };

    const UpdateFlags update_flags_velocity =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags update_flags_pressure =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags_velocity =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
    const UpdateFlags interface_update_flags_pressure =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

    ScratchData<dim> scratch_data(mapping,
                                  dof_handler_velocity.get_fe(),
                                  dof_handler_pressure.get_fe(),
                                  n_q_points_1d,
                                  update_flags_velocity,
                                  update_flags_pressure,
                                  interface_update_flags_velocity,
                                  interface_update_flags_pressure);

    CopyData copy_data(dof_handler_velocity.get_fe().dofs_per_cell,
                       dof_handler_pressure.get_fe().dofs_per_cell);

    if(use_conf_method || use_hdiv_ip_method)
      MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                            dof_handler_velocity.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method)
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
    else
      AssertThrow(false, ExcMessage("This FEM is not supported."));
  }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system()
{
  assemble_system_velocity_pressure();

  if(rt_parameters.solver.variant == "FGMRES_ILU" ||
     rt_parameters.solver.variant == "FGMRES_GMGvelocity")
  {
    pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));

    using Pressure::MW::CopyData;
    using Pressure::MW::ScratchData;
    using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, false>;

    MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_mass_worker(cell, scratch_data, copy_data);
      };

    const auto copier = [&](const CopyData & copy_data) {
      constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix, copy_data.local_dof_indices, pressure_mass_matrix);
    };

    const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_default;

    ScratchData<dim> scratch_data(
      mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);
    CopyData copy_data(dof_handler_pressure.get_fe().dofs_per_cell);
    MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
                          dof_handler_pressure.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system_velocity()
{
  AssertThrow(mgc_velocity.system_matrix.m() != 0,
              ExcMessage("The velocity system matrix does not seem to be initialized."));

  using Velocity::SIPG::MW::CopyData;
  using Velocity::SIPG::MW::ScratchData;
  using MatrixIntegrator = Velocity::SIPG::MW::MatrixIntegrator<dim, false>;
  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    matrix_integrator.cell_worker(cell, scratch_data, copy_data);
  };

  auto face_worker = [&](const auto &         cell,
                         const unsigned int & f,
                         const unsigned int & sf,
                         const auto &         ncell,
                         const unsigned int & nf,
                         const unsigned int & nsf,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) {
    matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
  };

  auto boundary_worker = [&](const auto &         cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
    matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
      copy_data.cell_matrix, copy_data.local_dof_indices_test, mgc_velocity.system_matrix);

    for(auto & cdf : copy_data.face_data)
    {
      constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
        cdf.cell_matrix, cdf.joint_dof_indices_test, mgc_velocity.system_matrix);
    }
  };

  ScratchData<dim> scratch_data(mapping,
                                get_fe_velocity(),
                                n_q_points_1d,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values,
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values | update_normal_vectors);
  CopyData         copy_data(dof_handler_velocity.get_fe().dofs_per_cell);
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



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::prepare_multigrid_velocity()
{
  dof_handler_velocity.distribute_mg_dofs();

  mgc_velocity.dof_handler = &dof_handler_velocity;
  mgc_velocity.mapping     = &mapping;
  mgc_velocity.cell_integrals_mask.reinit(dim, dim);
  mgc_velocity.face_integrals_mask.reinit(dim, dim);
  for(auto i = 0U; i < dim; ++i)
    for(auto j = 0U; j < dim; ++j)
    {
      mgc_velocity.cell_integrals_mask(i, j) = cell_integrals_mask(i, j);
      mgc_velocity.face_integrals_mask(i, j) = face_integrals_mask(i, j);
    }

  mgc_velocity.prepare_multigrid(max_level(), user_coloring);

  const unsigned int mg_level_max = max_level();
  const unsigned int mg_level_min = rt_parameters.multigrid.coarse_level;
  pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);
  pp_data.n_colors_system.push_back(n_colors_system());
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::prepare_multigrid_velocity_pressure()
{
  const unsigned int mg_level_max = max_level();
  const unsigned int mg_level_min = rt_parameters.multigrid.coarse_level;

  dof_handler.distribute_mg_dofs();
  /// This aligns dof numbering of dof_handler's first block and
  /// dof_handler_velocity on each level!
  for(auto level = 0U /*!!! mg_level_min?*/; level <= max_level(); ++level)
    DoFRenumbering::block_wise(dof_handler, level);
  dof_handler_velocity.distribute_mg_dofs();
  dof_handler_pressure.distribute_mg_dofs();

  mgc_velocity_pressure.dof_handler          = &dof_handler;
  mgc_velocity_pressure.dof_handler_velocity = &dof_handler_velocity;
  mgc_velocity_pressure.dof_handler_pressure = &dof_handler_pressure;
  mgc_velocity_pressure.mapping              = &mapping;
  mgc_velocity_pressure.cell_integrals_mask  = cell_integrals_mask;
  mgc_velocity_pressure.face_integrals_mask  = face_integrals_mask;


  /// TODO I am not sure if it is possible to impose mean value constraints on
  /// all level matrices in dealii and if this makes sense at all...
  ///
  /// One way could be to use a mean_value_filter() (see deal.II) instead of a
  /// distribute() call. This would still require an assembly w.r.t. the mean
  /// value constraint!?
  MGLevelObject<AffineConstraints<double>> mg_constraints_pressure(mg_level_min, mg_level_max);
  for(auto level = mg_level_min; level <= mg_level_max; ++level)
  {
    // const auto   mass_foreach_dof_ptr = compute_mass_foreach_pressure_dof(level);
    // const auto & mass_foreach_dof     = *mass_foreach_dof_ptr;
    // const auto   n_dofs_pressure      = dof_handler_pressure.n_dofs(level);

    auto & level_constraints_pressure = mg_constraints_pressure[level];
    level_constraints_pressure.clear();
    if(equation_data.force_mean_value_constraint)
    {
      AssertThrow(false, ExcMessage("TODO..."));
    }
    level_constraints_pressure.close();
  }

  mgc_velocity_pressure.prepare_multigrid(mg_level_max, user_coloring, mg_constraints_pressure);

  pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);
  pp_data.n_colors_system.push_back(n_colors_system());
}



template<int dim, int fe_degree_p, Method method>
template<typename PreconditionerType>
void
ModelProblem<dim, fe_degree_p, method>::iterative_solve_impl(
  const PreconditionerType & preconditioner,
  const std::string          solver_variant)
{
  ReductionControl solver_control;
  solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  solver_control.set_reduction(rt_parameters.solver.rel_tolerance);
  solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  solver_control.log_history(true);
  solver_control.log_result(true);
  solver_control.enable_history_data();

  SolverSelector<BlockVector<double>> iterative_solver;
  iterative_solver.set_control(solver_control);
  iterative_solver.select(solver_variant);
  if(solver_variant == "gmres")
  {
    SolverGMRES<BlockVector<double>>::AdditionalData additional_data;
    additional_data.right_preconditioning = rt_parameters.solver.use_right_preconditioning;
    // additional_data.use_default_residual = false;
    additional_data.max_n_tmp_vectors = 100; // rt_parameters.solver.n_iterations_max;
    iterative_solver.set_data(additional_data);
  }
  iterative_solver.solve(system_matrix, system_delta_x, system_rhs, preconditioner);
  /// distribute() is needed to apply the mean value constraint (Dirichlet
  /// conditions of velocity have already been applied to system_solution)
  if(equation_data.force_mean_value_constraint)
    mean_value_constraints.distribute(system_delta_x);
  // zero_constraints.distribute(system_delta_x); // !!!
  system_solution += system_delta_x;

  const auto [n_frac, reduction_rate] = compute_fractional_steps(solver_control);
  pp_data.average_reduction_system.push_back(reduction_rate);
  pp_data.n_iterations_system.push_back(n_frac);
  print_parameter("Average reduction (solver):", reduction_rate);
  print_parameter("Number of iterations (solver):", n_frac);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::solve()
{
  if(rt_parameters.solver.variant == "UMFPACK")
  {
    SparseDirectUMFPACK A_direct;
    A_direct.template initialize<BlockSparseMatrix<double>>(system_matrix);
    A_direct.vmult(system_delta_x, system_rhs);
    /// distribute() is needed to apply the mean value constraint (Dirichlet
    /// conditions of velocity have already been applied to system_solution)
    Assert(equation_data.force_mean_value_constraint, ExcMessage("Use mean value constraint."));
    zero_constraints.distribute(system_delta_x); // !!! no-normal flux + mean value
    mean_value_constraints.distribute(system_delta_x);
    system_solution += system_delta_x;

    pp_data.average_reduction_system.push_back(0.);
    pp_data.n_iterations_system.push_back(0.);
    print_parameter("Average reduction (solver):", "direct solver");
    print_parameter("Number of iterations (solver):", "---");
    return;
  }

  // This is used to pass whether or not we want to solve for A inside
  // the preconditioner.  One could change this to false to see if
  // there is still convergence and if so does the program then run
  // faster or slower
  const bool use_expensive = true;

  if(rt_parameters.solver.variant == "FGMRES_ILU")
  {
    SparseILU<double> A_preconditioner;
    A_preconditioner.initialize(system_matrix.block(0, 0));

    SparseILU<double> S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix);

    const BlockSchurPreconditioner<typename std::decay<decltype(A_preconditioner)>::type,
                                   typename std::decay<decltype(S_preconditioner)>::type>
      preconditioner(
        system_matrix, pressure_mass_matrix, A_preconditioner, S_preconditioner, use_expensive);

    iterative_solve_impl(preconditioner, "fgmres");
    *pcout << preconditioner.get_summary() << std::endl;
  }

  else if(rt_parameters.solver.variant == "FGMRES_GMGvelocity")
  {
    prepare_multigrid_velocity();
    auto & A_preconditioner = mgc_velocity.get_preconditioner();

    SparseILU<double> S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix, SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<typename std::decay<decltype(A_preconditioner)>::type,
                                   SparseILU<double>>
      preconditioner(
        system_matrix, pressure_mass_matrix, A_preconditioner, S_preconditioner, use_expensive);

    iterative_solve_impl(preconditioner, "fgmres");
    *pcout << preconditioner.get_summary();
  }

  else if(rt_parameters.solver.variant == "GMRES_GMG")
  {
    prepare_multigrid_velocity_pressure();
    auto & preconditioner = mgc_velocity_pressure.get_preconditioner();

    iterative_solve_impl(preconditioner, "gmres");
  }

  else if(rt_parameters.solver.variant == "CG_GMG")
  {
    prepare_multigrid_velocity_pressure();
    auto & preconditioner = mgc_velocity_pressure.get_preconditioner();

    iterative_solve_impl(preconditioner, "cg");
  }

  else if(rt_parameters.solver.variant == "CG")
  {
    PreconditionIdentity preconditioner;
    rt_parameters.solver.n_iterations_max *= 100.;

    iterative_solve_impl(preconditioner, "cg");

    //   auto preconditioner_amg = std::make_shared<TrilinosWrappers::PreconditionAMG>();
    //   std::vector<std::vector<bool>> constant_modes;
    //   FEValuesExtractors::Vector     velocity_components(0);
    //   DoFTools::extract_constant_modes(dof_handler,
    // 				     dof_handler.get_fe().component_mask(
    //                                    velocity_components),
    //                                  constant_modes);

    //   TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    // amg_data.constant_modes = constant_modes;
    // amg_data.elliptic              = true;
    // amg_data.higher_order_elements = true;
    // amg_data.smoother_sweeps       = 2;
    // amg_data.aggregation_threshold = 0.02;
    // preconditioner_amg->initialize(system_matrix.block(0, 0),
    //                                amg_data);

    // SparseILU<double> S_preconditioner;
    // S_preconditioner.initialize(pressure_mass_matrix);

    // const BlockSchurPreconditioner<SparseILU<double>, SparseILU<double>> preconditioner(
    //   system_matrix, pressure_mass_matrix, *preconditioner_amg, S_preconditioner, use_expensive);

    // iterative_solve_impl(preconditioner, "fgmres");
    // *pcout << preconditioner.get_summary();
  }

  else
    AssertThrow(false, ExcMessage("Please, choose a valid solver variant."));

  /// Post processing of discrete solution
  const double mean_pressure =
    VectorTools::compute_mean_value(dof_handler, QGauss<dim>(n_q_points_1d), system_solution, dim);
  const bool is_dgq_legendre =
    dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
  const bool is_legendre_type = is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP;
  if(is_legendre_type)
  {
    const auto n_dofs_per_cell = get_fe_pressure().dofs_per_cell;
    const auto n_dofs_pressure = system_solution.block(1).size();
    AssertDimension(n_dofs_pressure % n_dofs_per_cell, 0);
    Vector<double> & dof_values_pressure = system_solution.block(1);
    for(auto i = 0U; i < n_dofs_pressure; i += n_dofs_per_cell)
      dof_values_pressure[i] -= mean_pressure;
  }
  else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
    system_solution.block(1).add(-mean_pressure);
  else
    AssertThrow(false, ExcMessage("This dof layout is not supported."));

  print_parameter("Mean of pressure corrected by:", -mean_pressure);
  *pcout << std::endl;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::correct_mean_value_pressure()
{
  const double mean_pressure =
    VectorTools::compute_mean_value(dof_handler, QGauss<dim>(n_q_points_1d), system_solution, dim);
  const bool is_dgq_legendre =
    dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
  const bool is_legendre_type = is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP;
  if(is_legendre_type)
  {
    const auto n_dofs_per_cell = get_fe_pressure().dofs_per_cell;
    const auto n_dofs_pressure = system_solution.block(1).size();
    AssertDimension(n_dofs_pressure % n_dofs_per_cell, 0);
    Vector<double> & dof_values_pressure = system_solution.block(1);
    for(auto i = 0U; i < n_dofs_pressure; i += n_dofs_per_cell)
      dof_values_pressure[i] -= mean_pressure;
  }
  else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
    system_solution.block(1).add(-mean_pressure);
  else
    AssertThrow(false, ExcMessage("This dof layout is not supported."));

  print_parameter("Mean of pressure corrected by:", -mean_pressure);
  *pcout << std::endl;
}


template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_L2_error_velocity() const
{
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
  const auto difference_per_cell = std::make_shared<Vector<double>>(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    *difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  return difference_per_cell;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_L2_error_pressure() const
{
  const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
  const auto difference_per_cell = std::make_shared<Vector<double>>(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    *difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::L2_norm,
                                    &pressure_mask);
  return difference_per_cell;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_H1semi_error_velocity() const
{
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
  const auto difference_per_cell = std::make_shared<Vector<double>>(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    *difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::H1_norm,
                                    &velocity_mask);
  return difference_per_cell;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::compute_errors()
{
  {
    const auto   difference_per_cell = compute_L2_error_velocity();
    const double Velocity_L2_error =
      VectorTools::compute_global_error(triangulation, *difference_per_cell, VectorTools::L2_norm);
    print_parameter("Velocity error in the L2 norm:", Velocity_L2_error);
    pp_data.L2_error.push_back(Velocity_L2_error);
  }

  {
    const auto   difference_per_cell = compute_L2_error_pressure();
    const double Pressure_L2_error =
      VectorTools::compute_global_error(triangulation, *difference_per_cell, VectorTools::L2_norm);
    print_parameter("Pressure error in the L2 norm:", Pressure_L2_error);
    pp_data_pressure.L2_error.push_back(Pressure_L2_error);
  }

  {
    const auto   difference_per_cell = compute_H1semi_error_velocity();
    const double Velocity_H1_error =
      VectorTools::compute_global_error(triangulation, *difference_per_cell, VectorTools::H1_norm);
    print_parameter("Velocity error in the H1 seminorm:", Velocity_H1_error);
    pp_data.H1semi_error.push_back(Velocity_H1_error);
  }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::output_results(const unsigned int refinement_cycle) const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  /// discrete solution (velocity, pressure): (u, p)
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");
  data_out.add_data_vector(system_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  /// discrete right hand side (velocity, pressure): (Fu, Fp)
  std::vector<std::string> rhs_names(dim, "rhs_velocity");
  rhs_names.emplace_back("rhs_pressure");
  data_out.add_data_vector(system_rhs,
                           rhs_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  const auto L2_error_v = compute_L2_error_velocity();
  data_out.add_data_vector(*L2_error_v, "velocity_L2_error", DataOut<dim>::type_cell_data);

  const auto L2_error_p = compute_L2_error_pressure();
  data_out.add_data_vector(*L2_error_p, "pressure_L2_error", DataOut<dim>::type_cell_data);

  const auto H1semi_error_v = compute_H1semi_error_velocity();
  data_out.add_data_vector(*H1semi_error_v, "velocity_H1semi_error", DataOut<dim>::type_cell_data);

  data_out.build_patches();

  std::ofstream output("stokes_" + equation_data.sstr_equation_variant() + "_" +
                       Utilities::int_to_string(refinement_cycle, 3) + ".vtk");
  data_out.write_vtk(output);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::run()
{
  print_informations();

  const unsigned int n_cycles = rt_parameters.n_cycles;
  for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    *pcout << "Cycle: " << cycle + 1 << " of " << n_cycles << std::endl;

    const unsigned int n_refinements = rt_parameters.mesh.n_refinements + cycle;
    if(!make_grid(n_refinements))
    {
      *pcout << "NO MESH CREATED AT CYCLE " << cycle << " !\n\n";
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
