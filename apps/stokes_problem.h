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
#include "stokes_integrator.h"


namespace Stokes
{
using namespace dealii;

/**
 * TODO...
 */
template<int dim, int fe_degree_p = 2, typename Number = double>
class BlockSparseMatrixAugmented
  : public BlockSparseMatrix<Number>,
    public VelocityPressure::FD::MatrixIntegrator<dim, fe_degree_p, Number>
{
public:
  using value_type  = Number;
  using matrix_type = BlockSparseMatrix<Number>;
  using local_integrator_type =
    VelocityPressure::FD::MatrixIntegrator<dim, fe_degree_p, Number>; // TODO !!!

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in)
  {
    mf_storage = mf_storage_in;
    local_integrator_type::initialize(equation_data_in);
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

  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
};



/**
 * TODO...
 */
template<int dim, int fe_degree, typename Number = double>
class SparseMatrixAugmented : public SparseMatrix<Number>,
                              public Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree, Number>
{
public:
  using value_type            = Number;
  using matrix_type           = SparseMatrix<Number>;
  using local_integrator_type = Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree, Number>;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in)
  {
    mf_storage = mf_storage_in;
    local_integrator_type::initialize(equation_data_in);
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

  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
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



/**
 * TODO...
 */
template<int dim, int fe_degree, TPSS::DoFLayout dof_layout>
struct MGCollectionVelocity
{
  static constexpr int n_q_points_1d = fe_degree + 1;

  using VECTOR = Vector<double>;
  using MATRIX = SparseMatrixAugmented<dim, fe_degree, double>;

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
  mg_constrained_dofs->make_zero_boundary_constraints(*dof_handler,
                                                      equation_data.dirichlet_boundary_ids);

  // *** initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  mg_sparsity_patterns.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    DynamicSparsityPattern dsp(dof_handler->n_dofs(level), dof_handler->n_dofs(level));
    MGTools::make_flux_sparsity_pattern(*dof_handler, dsp, level);
    mg_sparsity_patterns[level].copy_from(dsp);
    mg_matrices[level].reinit(mg_sparsity_patterns[level]);
  }
  //: assemble the velocity system A_l on each level l.
  assemble_multigrid();

  // *** initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(*dof_handler);

  // *** initialize Schwarz smoother S_l
  switch(parameters.pre_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      mg_smoother_identity = std::make_shared<const MGSmootherIdentity<Vector<double>>>();
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::GaussSeidel:
    {
      auto tmp = std::make_shared<mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER, Vector<double>>>();
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
  multigrid = std::make_shared<Multigrid<Vector<double>>>(mg_matrix_wrapper,
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
    if(parameters.pre_smoother.schwarz.manual_coloring)
    {
      Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
      additional_data.coloring_func = std::ref(*user_coloring);
    }
    additional_data.parameters = parameters.pre_smoother;
    additional_data.dirichlet_ids.emplace_back(equation_data.dirichlet_boundary_ids);
    mgss->initialize(mg_matrices, additional_data);
    mg_schwarz_smoother_pre = mgss;
  }

  //: post-smoother (so far only shallow copy!)
  {
    const auto mgss_post = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    typename MG_SMOOTHER_SCHWARZ::AdditionalData additional_data;
    if(parameters.pre_smoother.schwarz.manual_coloring)
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

  using Velocity::SIPG::MW::CopyData;
  using Velocity::SIPG::MW::ScratchData;
  using MatrixIntegrator  = Velocity::SIPG::MW::MatrixIntegrator<dim, true>;
  using LevelCellIterator = typename MatrixIntegrator::IteratorType;
  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  for(unsigned int level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
  {
    AffineConstraints<double> level_constraints;
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
      if(dof_layout == TPSS::DoFLayout::DGQ)
        matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      if(dof_layout == TPSS::DoFLayout::DGQ)
        matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
    };

    const auto copier = [&](const CopyData & copy_data) {
      level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix, copy_data.local_dof_indices, mg_matrices[level]);

      for(auto & cdf : copy_data.face_data)
      {
        level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix, cdf.joint_dof_indices, mg_matrices[level]);
      }
    };

    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                               update_quadrature_points | update_JxW_values |
                                               update_normal_vectors;
    ScratchData<dim> scratch_data(
      *mapping, dof_handler->get_fe(), n_q_points_1d, update_flags, interface_update_flags);

    CopyData copy_data(dof_handler->get_fe().dofs_per_cell);

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
template<int dim, int fe_degree_p, TPSS::DoFLayout dof_layout_v>
struct MGCollectionVelocityPressure
{
  static constexpr int fe_degree_v   = fe_degree_p + 1;
  static constexpr int n_q_points_1d = fe_degree_v + 1;

  using VECTOR = BlockVector<double>;
  using MATRIX = BlockSparseMatrixAugmented<dim, fe_degree_p, double>;

  // TODO !!! MGTransferBlockMatrixFree (based on distributed::BlockVector)
  using MG_TRANSFER           = MGTransferPrebuilt<VECTOR>;
  using GAUSS_SEIDEL_SMOOTHER = PreconditionSOR<MATRIX>;
  using PATCH_MATRIX          = typename MATRIX::local_integrator_type::matrix_type;
  using MG_SMOOTHER_SCHWARZ   = MGSmootherSchwarz<dim, MATRIX, PATCH_MATRIX, VECTOR>;

  MGCollectionVelocityPressure(const MGParameter &  mg_prms_in,
                               const EquationData & equation_data_in);

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
  const DoFHandler<dim> * dof_handler_velocity;
  const DoFHandler<dim> * dof_handler_pressure;
  const Mapping<dim> *    mapping;

  std::shared_ptr<MGConstrainedDoFs>                mg_constrained_dofs;
  MG_TRANSFER                                       mg_transfer;
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

template<int dim, int fe_degree_p, TPSS::DoFLayout dof_layout_v>
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v>::MGCollectionVelocityPressure(
  const MGParameter &  mg_prms_in,
  const EquationData & equation_data_in)
  : parameters(mg_prms_in),
    equation_data(equation_data_in),
    dof_handler_velocity(nullptr),
    dof_handler_pressure(nullptr),
    mapping(nullptr)
{
}

template<int dim, int fe_degree_p, TPSS::DoFLayout dof_layout_v>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v>::clear()
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

template<int dim, int fe_degree_p, TPSS::DoFLayout dof_layout_v>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v>::assemble_multigrid()
{
  AssertDimension(mg_matrices.min_level(), parameters.coarse_level);
  Assert(mg_constrained_dofs, ExcMessage("mg_constrained_dofs is uninitialized."));
  Assert(dof_handler_velocity, ExcMessage("Did you set dof_handler_velocity?"));
  AssertThrow(TPSS::get_dof_layout(dof_handler->get_fe().base_element(0)) == dof_layout_v,
              ExcMessage("Velocity block of dof_handler and dof_layout_v are incompatible."));
  AssertThrow(TPSS::get_dof_layout(dof_handler_velocity->get_fe().base_element(0)) == dof_layout_v,
              ExcMessage("dof_handler_velocity and dof_layout_v are incompatible."));
  Assert(dof_handler_pressure, ExcMessage("Did you set dof_handler_pressure?"));

  for(unsigned int level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
  {
    /// As long as the DoF numbering of dof_handler_velocity and the velocity
    /// block of dof_handler is aligned, we might use the complete set of
    /// level_constraints as constraints for the velocity block.q
    AffineConstraints<double> level_constraints;
    level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    AffineConstraints<double> level_constraints_pressure;
    level_constraints_pressure.close();

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
        if(dof_layout_v == TPSS::DoFLayout::DGQ) // TODO !!!
          matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
      };

      auto boundary_worker = [&](const auto &         cell,
                                 const unsigned int & face_no,
                                 ScratchData<dim> &   scratch_data,
                                 CopyData &           copy_data) {
        if(dof_layout_v == TPSS::DoFLayout::DGQ) // TODO !!!
          matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
      };

      const auto copier = [&](const CopyData & copy_data) {
        level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          copy_data.cell_matrix, copy_data.local_dof_indices, mg_matrices[level].block(0, 0));

        for(auto & cdf : copy_data.face_data)
        {
          AssertThrow(false, ExcMessage("TODO DGQ case"));
          level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
            cdf.cell_matrix, cdf.joint_dof_indices, mg_matrices[level].block(0, 0));
        }
      };

      const UpdateFlags update_flags =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      const UpdateFlags interface_update_flags = update_values | update_gradients |
                                                 update_quadrature_points | update_JxW_values |
                                                 update_normal_vectors;
      ScratchData<dim> scratch_data(*mapping,
                                    dof_handler_velocity->get_fe(),
                                    n_q_points_1d,
                                    update_flags,
                                    interface_update_flags);

      CopyData copy_data(dof_handler_velocity->get_fe().dofs_per_cell);

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

      MatrixIntegrator matrix_integrator(nullptr, nullptr, equation_data);
      const auto &     triangulation = dof_handler_pressure->get_triangulation();
      auto             cell_worker =
        [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
          LevelCellIterator cell_ansatz(&triangulation,
                                        cell->level(),
                                        cell->index(),
                                        dof_handler_pressure);
          matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
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
      };

      const UpdateFlags update_flags_velocity =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;
      const UpdateFlags update_flags_pressure =
        update_values | update_gradients | update_quadrature_points | update_JxW_values;

      ScratchData<dim> scratch_data(*mapping,
                                    dof_handler_velocity->get_fe(),
                                    dof_handler_pressure->get_fe(),
                                    n_q_points_1d,
                                    update_flags_velocity,
                                    update_flags_pressure);
      CopyData         copy_data(dof_handler_velocity->get_fe().dofs_per_cell,
                         dof_handler_pressure->get_fe().dofs_per_cell);
      MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                            dof_handler_velocity->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    }
  }
}

template<int dim, int fe_degree_p, TPSS::DoFLayout dof_layout_v>
void
MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v>::prepare_multigrid(
  const unsigned int                       mg_level_max,
  const std::shared_ptr<ColoringBase<dim>> user_coloring)
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
  mg_constrained_dofs->make_zero_boundary_constraints(*dof_handler,
                                                      equation_data.dirichlet_boundary_ids,
                                                      dof_handler->get_fe().component_mask(
                                                        velocities));

  // *** initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  mg_sparsity_patterns.resize(mg_level_min, mg_level_max);
  std::vector<std::vector<types::global_dof_index>> level_to_dofs_per_block(
    mg_level_max + 1, std::vector<types::global_dof_index>(2, numbers::invalid_dof_index));
  // level_to_dofs_per_block.resize(mg_level_max+1);
  MGTools::count_dofs_per_block(*dof_handler, level_to_dofs_per_block);
  // AssertDimension(mg_level_max - mg_level_min + 1, level_to_dofs_per_block.size());
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    const auto &                dofs_per_block = level_to_dofs_per_block[level];
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    MGTools::make_sparsity_pattern(*dof_handler, dsp, level); // TODO !!! DGQ case
    mg_sparsity_patterns[level].copy_from(dsp);
    mg_matrices[level].reinit(mg_sparsity_patterns[level]);
  }
  //: assemble the velocity system A_l on each level l.
  assemble_multigrid();

  // *** initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(*dof_handler);

  // // *** initialize Schwarz smoother S_l
  // switch(parameters.pre_smoother.variant)
  // {
  //   case SmootherParameter::SmootherVariant::None:
  //     mg_smoother_identity = std::make_shared<const MGSmootherIdentity<Vector<double>>>();
  //     AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
  //     mg_smoother_pre = mg_smoother_identity.get();
  //     break;
  //   case SmootherParameter::SmootherVariant::GaussSeidel:
  //   {
  //     auto tmp = std::make_shared<mg::SmootherRelaxation<GAUSS_SEIDEL_SMOOTHER,
  //     Vector<double>>>(); tmp->initialize(mg_matrices);
  //     tmp->set_steps(parameters.pre_smoother.n_smoothing_steps);
  //     tmp->set_symmetric(true);
  //     mg_smoother_gauss_seidel = tmp;
  //     mg_smoother_pre          = mg_smoother_gauss_seidel.get();
  //   }
  //   break;
  //   case SmootherParameter::SmootherVariant::Schwarz:
  //     prepare_schwarz_smoothers(user_coloring);
  //     AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Not initialized."));
  //     mg_smoother_pre = mg_schwarz_smoother_pre.get();
  //     break;
  //   default:
  //     AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  // }
  // switch(parameters.post_smoother.variant)
  // {
  //   case SmootherParameter::SmootherVariant::None:
  //     AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
  //     mg_smoother_post = mg_smoother_identity.get();
  //     break;
  //   case SmootherParameter::SmootherVariant::GaussSeidel:
  //     AssertThrow(mg_smoother_gauss_seidel, ExcMessage("Not initialized."));
  //     mg_smoother_post = mg_smoother_gauss_seidel.get();
  //     break;
  //   case SmootherParameter::SmootherVariant::Schwarz:
  //     AssertThrow(mg_schwarz_smoother_post, ExcMessage("Not initialized"));
  //     mg_smoother_post = mg_schwarz_smoother_post.get();
  //     break;
  //   default:
  //     AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  // }

  // // *** initialize coarse grid solver
  // coarse_grid_solver.initialize(mg_matrices[mg_level_min], parameters.coarse_grid);
  // mg_coarse_grid = &coarse_grid_solver;

  // // *** initialize geometric multigrid method
  // mg_matrix_wrapper.initialize(mg_matrices);
  // multigrid = std::make_shared<Multigrid<Vector<double>>>(mg_matrix_wrapper,
  //                                                         *mg_coarse_grid,
  //                                                         mg_transfer,
  //                                                         *mg_smoother_pre,
  //                                                         *mg_smoother_post,
  //                                                         mg_level_min,
  //                                                         mg_level_max);
}



/**
 * TODO...
 */
enum class Method
{
  TaylorHood,
  /*only for testing*/ TaylorHoodDGQ,
  TaylorHoodBlock
};

/**
 * TODO...
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
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::TaylorHoodDGQ, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::DGQ;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::Q;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::TaylorHoodBlock, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::Q;
};



/**
 * TODO...
 */
template<int dim, int fe_degree_p, Method method = Method::TaylorHood>
class ModelProblem : public ModelProblemBase<method, dim, fe_degree_p>
{
  static_assert(dim == 2, "only 2D");

  using Base = ModelProblemBase<method, dim, fe_degree_p>;

public:
  static constexpr int fe_degree_v   = fe_degree_p + 1;
  static constexpr int n_q_points_1d = fe_degree_v + 1;
  using Base::dof_layout_v;

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
  setup_system_velocity(const bool do_system_matrix = true);

  void
  setup_system_pressure();

  void
  assemble_system();

  void
  assemble_system_step56();

  void
  assemble_system_velocity_pressure();

  void
  assemble_system_velocity();

  void
  assemble_multigrid_velocity();

  void
  prepare_multigrid_velocity();

  void
  prepare_multigrid_velocity_pressure();

  void
  solve();

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

  void
  set_finite_elements(const std::shared_ptr<FiniteElement<dim>> fe_velocity_in);

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
  std::shared_ptr<FiniteElement<dim>> fe;
  DoFHandler<dim>                     dof_handler;
  DoFHandler<dim>                     dof_handler_velocity;
  DoFHandler<dim>                     dof_handler_pressure;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> zero_constraints_velocity;
  AffineConstraints<double> constraints_velocity;
  AffineConstraints<double> constraints_pressure;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  SparseMatrix<double>      pressure_mass_matrix;

  BlockVector<double> system_solution;
  BlockVector<double> system_delta_x;
  BlockVector<double> system_rhs;

  //: multigrid
  mutable std::shared_ptr<ColoringBase<dim>>                   user_coloring;
  MGCollectionVelocity<dim, fe_degree_p + 1, dof_layout_v>     mgc_velocity;
  MGCollectionVelocityPressure<dim, fe_degree_p, dof_layout_v> mgc_velocity_pressure;

private:
  std::shared_ptr<FiniteElement<dim>>
  generate_fe(const FiniteElement<dim> & fe_velocity);

  bool
  check_finite_elements() const;

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner, const std::string solver_variant);

  void
  make_grid_impl(const MeshParameter & mesh_prms);
};



template<int dim, int fe_degree_p, Method method>
ModelProblem<dim, fe_degree_p, method>::ModelProblem(const RT::Parameter & rt_parameters_in,
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
    fe_velocity(dof_layout_v == TPSS::DoFLayout::DGQ ?
                  std::make_shared<FESystem<dim>>(FE_DGQ<dim>(fe_degree_p + 1), dim) :
                  std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree_p + 1), dim)),
    // Finite element for the whole system:
    fe(generate_fe(*fe_velocity)),
    dof_handler(triangulation),
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
  equation_data.assemble_pressure_mass_matrix =
    (rt_parameters.solver.variant == "UMFPACK") ? false : true;
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
  print_parameter("Finite element (velocity):", fe_velocity->get_name());
  *pcout << rt_parameters.to_string();
  *pcout << std::endl;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::set_finite_elements(
  const std::shared_ptr<FiniteElement<dim>> fe_velocity_in)
{
  fe_velocity = fe_velocity_in;
  generate_fe(*fe_velocity_in);
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<FiniteElement<dim>>
ModelProblem<dim, fe_degree_p, method>::generate_fe(const FiniteElement<dim> & fe_velocity)
{
  return std::make_shared<FESystem<dim>>(fe_velocity,
                                         1,
                                         fe_degree_p == 0 ?
                                           static_cast<const FiniteElement<dim> &>(FE_DGQ<dim>(0)) :
                                           static_cast<const FiniteElement<dim> &>(
                                             FE_Q<dim>(fe_degree_p)),
                                         1);
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::check_finite_elements() const
{
  //: check fe
  AssertDimension(fe->n_base_elements(), 2); // velocity + pressure
  const auto & fe_v = fe->base_element(0);
  AssertDimension(fe_v.n_components(), dim);          // velocity
  AssertDimension(fe_v.element_multiplicity(0), dim); // dim times FE_Q
  AssertThrow(TPSS::get_dof_layout(fe_v.base_element(0)) == dof_layout_v,
              ExcMessage("velocity part of fe and dof_layout are incompatible."));
  const auto & fe_p = fe->base_element(1);
  AssertDimension(fe_p.n_components(), 1); // pressure

  //: check fe_velocity
  AssertThrow(TPSS::get_dof_layout(fe_velocity->base_element(0)) == dof_layout_v,
              ExcMessage("fe_velocity and dof_layout are incompatible."));

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
  AssertDimension(fe_v.n_components(), dim);          // velocity
  AssertDimension(fe_v.element_multiplicity(0), dim); // dim times FE_Q
  const auto   n_dofs_est_v = dim * estimate_n_dofs(fe_v.base_element(0), mesh_prms);
  const auto & fe_p         = fe->base_element(1);
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
void
ModelProblem<dim, fe_degree_p, method>::setup_system_velocity(const bool do_system_matrix)
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  //: distribute DoFs and initialize MGCollection
  dof_handler_velocity.initialize(triangulation, *fe_velocity);

  //: setup system matrix (DEBUG)
  if(do_system_matrix)
  {
    Assert(constraints_velocity.n_constraints() != 0,
           ExcMessage("Did you initialize the velocity constraints?"));
    mgc_velocity.system_matrix.clear();
    DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler_velocity,
                                         dsp,
                                         constraints_velocity,
                                         rt_parameters.solver.variant == "UMFPACK" ? true : false);
    mgc_velocity.sparsity_pattern.copy_from(dsp);
    mgc_velocity.system_matrix.reinit(mgc_velocity.sparsity_pattern);
  }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system_pressure()
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  std::vector<bool> pressure_mask(dim, false);
  pressure_mask.push_back(true);
  const auto & fe_pressure = fe->get_sub_fe(pressure_mask);
  dof_handler_pressure.initialize(triangulation, fe_pressure);
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
  if(rt_parameters.solver.variant == "FGMRES_ILU")
    DoFRenumbering::Cuthill_McKee(dof_handler);

  // This ensures that all velocities DoFs are enumerated before the
  // pressure unknowns. This allows us to use blocks for vectors and
  // matrices and allows us to get the same DoF numbering for
  // dof_handler and dof_handler_velocity.
  DoFRenumbering::block_wise(dof_handler);
  std::vector<unsigned int>                  block_component{0U, 1U};
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_dofs_velocity = dofs_per_block[0];
  const unsigned int n_dofs_pressure = dofs_per_block[1];

  {
    zero_constraints.clear();
    const FEValuesExtractors::Vector velocities(0);

    if(rt_parameters.solver.variant == "FGMRES_ILU") // inhomog. constraints !
    {
      for(const auto boundary_id : equation_data.dirichlet_boundary_ids)
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 *analytical_solution,
                                                 zero_constraints,
                                                 fe->component_mask(velocities));
    }
    else
    {
      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      for(const auto boundary_id : equation_data.dirichlet_boundary_ids)
        DoFTools::make_zero_boundary_constraints(dof_handler,
                                                 boundary_id,
                                                 zero_constraints,
                                                 fe->component_mask(velocities));
    }

    {
      constraints_velocity.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_velocity);
      std::map<types::boundary_id, const Function<dim, double> *> boundary_id_to_function;
      for(const auto boundary_id : equation_data.dirichlet_boundary_ids)
        boundary_id_to_function.emplace(boundary_id, analytical_solution.get());
      VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_id_to_function,
                                               constraints_velocity,
                                               fe->component_mask(velocities));
      constraints_velocity.close();
    }

    // As discussed in the introduction, we need to fix one degree of freedom
    // of the pressure variable to ensure solvability of the problem. We do
    // this here by marking the first pressure dof, which has index n_dofs_velocity as a
    // constrained dof.
    constraints_pressure.clear();
    if(rt_parameters.solver.variant == "UMFPACK")
    {
      zero_constraints.add_line(n_dofs_velocity);
      constraints_pressure.add_line(0);
    }
    constraints_pressure.close();

    zero_constraints.close();
  }

  setup_system_velocity(false);

  setup_system_pressure();

  system_matrix.clear();
  pressure_mass_matrix.clear();
  BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, zero_constraints, false); // TODO !!! DGQ case
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  system_solution.reinit(dofs_per_block);
  zero_constraints.set_zero(system_solution);
  constraints_velocity.distribute(system_solution.block(0)); // particular velocity solution!
  system_delta_x.reinit(dofs_per_block);
  zero_constraints.set_zero(system_delta_x);
  system_rhs.reinit(dofs_per_block);

  print_parameter("Number of degrees of freedom (velocity):", n_dofs_velocity);
  print_parameter("Number of degrees of freedom (pressure):", n_dofs_pressure);
  print_parameter("Number of degrees of freedom (total):", n_dofs_velocity + n_dofs_pressure);
  *pcout << std::endl;
  pp_data.n_dofs_global.push_back(system_matrix.m());
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system_step56()
{
  using VelocityPressure::MW::CopyData;
  using VelocityPressure::MW::ScratchData;
  using MatrixIntegrator = VelocityPressure::MW::MatrixIntegrator<dim, /*is_multigrid*/ false>;

  const auto * particular_solution =
    rt_parameters.solver.variant == "FGMRES_ILU" ? nullptr : &system_solution;
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
  /// Assemble the velocity block, here block(0,0).
  {
    using Velocity::SIPG::MW::CopyData;
    using Velocity::SIPG::MW::ScratchData;
    using MatrixIntegrator = Velocity::SIPG::MW::MatrixIntegrator<dim, false>;

    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> load_function_velocity(load_function.get(), component_range);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
    const auto *           particular_solution_velocity =
      rt_parameters.solver.variant == "FGMRES_ILU" ? nullptr : &(system_solution.block(0));
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
      if(dof_layout_v == TPSS::DoFLayout::DGQ) // TODO !!!
        matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      if(dof_layout_v == TPSS::DoFLayout::DGQ) // TODO !!!
        matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
    };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints.template distribute_local_to_global<SparseMatrix<double>, Vector<double>>(
        copy_data.cell_matrix,
        copy_data.cell_rhs,
        copy_data.local_dof_indices,
        system_matrix.block(0, 0),
        system_rhs.block(0));

      for(auto & cdf : copy_data.face_data) // TODO !!!
      {
        if(cdf.cell_rhs.size() == 0) // only filled on cells at the boundary
          zero_constraints.template distribute_local_to_global<SparseMatrix<double>>(
            cdf.cell_matrix, cdf.joint_dof_indices, system_matrix.block(0, 0));
        else
          zero_constraints
            .template distribute_local_to_global<SparseMatrix<double>, Vector<double>>(
              cdf.cell_matrix,
              cdf.cell_rhs,
              cdf.joint_dof_indices,
              system_matrix.block(0, 0),
              system_rhs.block(0));
      }
    };

    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                               update_quadrature_points | update_JxW_values |
                                               update_normal_vectors;

    ScratchData<dim> scratch_data(
      mapping, dof_handler_velocity.get_fe(), n_q_points_1d, update_flags, interface_update_flags);
    CopyData copy_data(dof_handler_velocity.get_fe().dofs_per_cell);
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
      rt_parameters.solver.variant == "FGMRES_ILU" ? nullptr : &(system_solution.block(0));
    const auto * particular_solution_pressure =
      rt_parameters.solver.variant == "FGMRES_ILU" ? nullptr : &(system_solution.block(1));
    MatrixIntegrator matrix_integrator(particular_solution_velocity,
                                       particular_solution_pressure,
                                       equation_data);

    auto cell_worker =
      [&](const CellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                                 cell->level(),
                                 cell->index(),
                                 &dof_handler_pressure);
        matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
      };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix,
        copy_data.local_dof_indices_test,
        constraints_pressure,
        copy_data.local_dof_indices_ansatz,
        system_matrix.block(0, 1));
      constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix_flipped,
        copy_data.local_dof_indices_ansatz,
        zero_constraints,
        copy_data.local_dof_indices_test,
        system_matrix.block(1, 0));

      zero_constraints.template distribute_local_to_global<Vector<double>>(
        copy_data.cell_rhs_test, copy_data.local_dof_indices_test, system_rhs.block(0));
      constraints_pressure.template distribute_local_to_global<Vector<double>>(
        copy_data.cell_rhs_ansatz, copy_data.local_dof_indices_ansatz, system_rhs.block(1));
    };

    const UpdateFlags update_flags_velocity =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags update_flags_pressure =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;

    ScratchData<dim> scratch_data(mapping,
                                  dof_handler_velocity.get_fe(),
                                  dof_handler_pressure.get_fe(),
                                  n_q_points_1d,
                                  update_flags_velocity,
                                  update_flags_pressure);
    CopyData         copy_data(dof_handler_velocity.get_fe().dofs_per_cell,
                       dof_handler_pressure.get_fe().dofs_per_cell);
    MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                          dof_handler_velocity.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system()
{
  /// Currently, the FGMRES_ILU path is only supported by the old assembly (step-56)
  if(rt_parameters.solver.variant == "FGMRES_ILU")
    assemble_system_step56();

  /// All remaining cases use the assembly where velocity and pressure blocks
  /// are assembled separately. This allows to use FEInterfaceValues for DG
  /// elements.
  else
    assemble_system_velocity_pressure();

  if(rt_parameters.solver.variant != "UMFPACK")
  {
    pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
    pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
    system_matrix.block(1, 1) = 0.;
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
      copy_data.cell_matrix, copy_data.local_dof_indices, mgc_velocity.system_matrix);

    for(auto & cdf : copy_data.face_data)
    {
      constraints_velocity.template distribute_local_to_global<SparseMatrix<double>>(
        cdf.cell_matrix, cdf.joint_dof_indices, mgc_velocity.system_matrix);
    }
  };

  ScratchData<dim> scratch_data(mapping,
                                *fe_velocity,
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
  dof_handler.distribute_mg_dofs();
  /// This aligns dof numbering of dof_handler's first block and
  /// dof_handler_velocity on each level!
  for(auto level = 0U; level <= max_level(); ++level)
    DoFRenumbering::block_wise(dof_handler, level);
  dof_handler_velocity.distribute_mg_dofs();
  dof_handler_pressure.distribute_mg_dofs();

  mgc_velocity_pressure.dof_handler          = &dof_handler;
  mgc_velocity_pressure.dof_handler_velocity = &dof_handler_velocity;
  mgc_velocity_pressure.dof_handler_pressure = &dof_handler_pressure;
  mgc_velocity_pressure.mapping              = &mapping;

  mgc_velocity_pressure.prepare_multigrid(max_level(), user_coloring);

  const unsigned int mg_level_max = max_level();
  const unsigned int mg_level_min = rt_parameters.multigrid.coarse_level;
  pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);
  pp_data.n_colors_system.push_back(n_colors_system());
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_multigrid_velocity()
{
  mgc_velocity.assemble_multigrid();
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
  iterative_solver.solve<BlockSparseMatrix<double>, PreconditionerType>(system_matrix,
                                                                        system_delta_x,
                                                                        system_rhs,
                                                                        preconditioner);
  if(rt_parameters.solver.variant == "FGMRES_ILU") // inhomog. constraints !
  {
    zero_constraints.distribute(system_delta_x);
    system_solution = 0.;
  }
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

  /// Currently, this method only converges with inhomog. constraints !!!
  if(rt_parameters.solver.variant == "FGMRES_ILU")
  {
    SparseILU<double> A_preconditioner;
    A_preconditioner.initialize(system_matrix.block(0, 0));

    SparseILU<double> S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix);

    const BlockSchurPreconditioner<SparseILU<double>, SparseILU<double>> preconditioner(
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

  else
    AssertThrow(false, ExcMessage("Please, choose a valid solver variant."));

  /// Post processing of discrete solution
  const double mean_pressure = VectorTools::compute_mean_value(dof_handler,
                                                               QGauss<dim>(fe_degree_p + 2),
                                                               system_solution,
                                                               dim);
  system_solution.block(1).add(-mean_pressure);
  print_parameter("Mean of pressure corrected by:", -mean_pressure);
  *pcout << std::endl;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::compute_errors()
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



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::output_results(const unsigned int refinement_cycle) const
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
