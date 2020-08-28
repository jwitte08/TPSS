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
#include "coloring.h"
#include "equation_data.h"
#include "mesh.h"
#include "multigrid.h"
#include "postprocess.h"
#include "rt_parameter.h"



namespace Biharmonic
{
using namespace dealii;

template<int dim, int fe_degree = 2, typename Number = double>
class SparseMatrixAugmented : public SparseMatrix<Number>,
                              public C0IP::FD::MatrixIntegrator<dim, fe_degree, Number>
{
public:
  using value_type            = Number;
  using matrix_type           = SparseMatrix<Number>;
  using local_integrator_type = C0IP::FD::MatrixIntegrator<dim, fe_degree, Number>;

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



template<int dim>
using ScratchData = C0IP::MW::ScratchData<dim>;

using CopyData = C0IP::MW::CopyData;



template<int dim, int fe_degree = 2>
class ModelProblem
{
public:
  using VECTOR = Vector<double>;
  using MATRIX = SparseMatrixAugmented<dim, fe_degree, double>;

  using MG_TRANSFER           = MGTransferPrebuilt<VECTOR>;
  using GAUSS_SEIDEL_SMOOTHER = PreconditionSOR<MATRIX>;
  using PATCH_MATRIX          = Tensors::TensorProductMatrix<dim, VectorizedArray<double>>;
  using MG_SMOOTHER_SCHWARZ   = MGSmootherSchwarz<dim, MATRIX, PATCH_MATRIX, VECTOR>;
  using GMG_PRECONDITIONER    = PreconditionMG<dim, VECTOR, MG_TRANSFER>;

  static_assert(dim == 2, "only implemented in 2D");

  ModelProblem(const RT::Parameter & rt_parameters_in,
               const EquationData &  equation_data_in = EquationData{});

  void
  run();

  bool
  make_grid();

  bool
  make_grid(const unsigned int n_refinements);

  void
  setup_system();

  void
  assemble_system();

  void
  prepare_multigrid();

  void
  prepare_schwarz_smoothers();

  const GMG_PRECONDITIONER &
  prepare_preconditioner_mg()
  {
    prepare_multigrid();
    AssertThrow(multigrid, ExcNotInitialized());

    preconditioner_mg = std::make_shared<GMG_PRECONDITIONER>(dof_handler, *multigrid, mg_transfer);
    return *preconditioner_mg;
  }

  void
  solve();

  double
  compute_stream_function_error();

  void
  compute_errors();

  void
  output_results(const unsigned int iteration) const;

  unsigned int
  max_level() const
  {
    return triangulation.n_global_levels() - 1;
  }

  unsigned int
  n_colors_system()
  {
    if(mg_schwarz_smoother_pre)
      return mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
    return numbers::invalid_unsigned_int;
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
    *pcout << rt_parameters.to_string();
    *pcout << std::endl;
  }

  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  std::shared_ptr<Function<dim>>      analytical_solution;
  std::shared_ptr<Function<dim>>      analytical_velocity; // Stokes
  std::shared_ptr<Function<dim>>      load_function;
  std::shared_ptr<Function<dim>>      load_function_stokes;
  std::shared_ptr<ConditionalOStream> pcout;
  mutable PostProcessData             pp_data;
  mutable PostProcessData             pp_data_stokes;

  parallel::distributed::Triangulation<dim> triangulation;
  MappingQ<dim>                             mapping;
  FE_Q<dim>                                 fe;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<double>                 constraints;
  AffineConstraints<double>                 zero_constraints;

  SparsityPattern sparsity_pattern;
  MATRIX          system_matrix;
  VECTOR          system_u;
  VECTOR          system_delta_u;
  VECTOR          system_rhs;

  // *** multigrid
  std::shared_ptr<MGConstrainedDoFs>                mg_constrained_dofs;
  MG_TRANSFER                                       mg_transfer;
  MGLevelObject<SparsityPattern>                    mg_sparsity_patterns;
  MGLevelObject<MATRIX>                             mg_matrices;
  mutable std::shared_ptr<ColoringBase<dim>>        user_coloring;
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

  std::shared_ptr<GMG_PRECONDITIONER> preconditioner_mg;

private:
  void
  make_grid_impl(const MeshParameter & mesh_prms);

  template<bool is_stream_function>
  void
  assemble_system_impl();

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner);
};



template<int dim, int fe_degree>
ModelProblem<dim, fe_degree>::ModelProblem(const RT::Parameter & rt_parameters_in,
                                           const EquationData &  equation_data_in)
  : rt_parameters(rt_parameters_in),
    equation_data(equation_data_in),
    analytical_solution([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::ClampedHom)
        return std::make_shared<Clamped::Homogeneous::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedBell)
        return std::make_shared<Clamped::GaussianBells::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlip)
        return std::make_shared<Clamped::StreamFunction::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuille)
        return std::make_shared<Clamped::Poiseuille::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<Clamped::NoSlipNormal::Solution<dim>>();
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    analytical_velocity([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::ClampedHom)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedBell)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlip)
        return std::make_shared<Stokes::DivergenceFree::NoSlip::SolutionVelocity<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuille)
        return std::make_shared<Stokes::DivergenceFree::Poiseuille::SolutionVelocity<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<Stokes::DivergenceFree::NoSlipNormal::SolutionVelocity<dim>>();
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    load_function([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::ClampedHom)
        return std::make_shared<Clamped::Homogeneous::Load<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedBell)
        return std::make_shared<Clamped::GaussianBells::Load<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlip)
        return std::make_shared<Clamped::StreamFunction::Load<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuille)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<ManufacturedLoad<dim, Clamped::NoSlipNormal::Solution<dim>>>();
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    load_function_stokes([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::ClampedHom)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedBell)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlip)
        return std::make_shared<Stokes::ManufacturedLoad<dim>>(
          std::make_shared<Stokes::DivergenceFree::NoSlip::Solution<dim>>());
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuille)
        return std::make_shared<Stokes::ManufacturedLoad<dim>>(
          std::make_shared<Stokes::DivergenceFree::Poiseuille::Solution<dim>>());
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<Stokes::ManufacturedLoad<dim>>(
          std::make_shared<Stokes::DivergenceFree::NoSlipNormal::Solution<dim>>());
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    pcout(
      std::make_shared<ConditionalOStream>(std::cout,
                                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    mapping(1),
    fe(fe_degree),
    user_coloring(std::make_shared<RedBlackColoring<dim>>(rt_parameters_in.mesh))
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("One process only."));
  AssertThrow(rt_parameters.multigrid.pre_smoother.schwarz.patch_variant ==
                TPSS::PatchVariant::vertex,
              ExcMessage("Vertex patches only."));

  if(equation_data.is_stream_function)
  {
    Assert(load_function_stokes, ExcMessage("load_function_stokes is required."));
    AssertDimension(load_function_stokes->n_components, dim);
  }
  else
  {
    Assert(load_function, ExcMessage("load_function is required."));
    AssertDimension(load_function->n_components, 1U);
  }
}



template<int dim, int fe_degree>
bool
ModelProblem<dim, fe_degree>::make_grid()
{
  make_grid_impl(rt_parameters.mesh);
  return true;
}



template<int dim, int fe_degree>
bool
ModelProblem<dim, fe_degree>::make_grid(const unsigned int n_refinements)
{
  MeshParameter mesh_prms = rt_parameters.mesh;
  mesh_prms.n_refinements = n_refinements;

  const auto n_dofs_est = estimate_n_dofs(fe, mesh_prms);
  if(rt_parameters.exceeds_dof_limits(n_dofs_est))
    return false;

  make_grid_impl(mesh_prms);
  return true;
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::make_grid_impl(const MeshParameter & mesh_prms)
{
  triangulation.clear();
  *pcout << create_mesh(triangulation, mesh_prms) << std::endl;
  pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());
  pp_data.n_dimensions = dim;
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::setup_system()
{
  dof_handler.clear();
  dof_handler.initialize(triangulation, fe);
  dof_handler.distribute_mg_dofs();

  print_parameter("Number of degrees of freedom:", dof_handler.n_dofs());

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler, 0, *analytical_solution, constraints);
  constraints.close();
  zero_constraints.clear();
  for(const auto boundary_id : equation_data.dirichlet_boundary_ids)
    DoFTools::make_zero_boundary_constraints(dof_handler, boundary_id, zero_constraints);
  zero_constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  const bool             direct_solver_used = (rt_parameters.solver.variant == "direct");
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       direct_solver_used ? true : false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  pp_data.n_dofs_global.push_back(system_matrix.m());

  system_u.reinit(0);
  system_delta_u.reinit(0);
  system_rhs.reinit(0);
  system_u.reinit(dof_handler.n_dofs());
  constraints.distribute(system_u); // set boundary values (particular solution)
  system_delta_u.reinit(dof_handler.n_dofs());
  zero_constraints.distribute(system_delta_u); // set zero boundary values (homogen. solution)
  system_rhs.reinit(dof_handler.n_dofs());
}



template<int dim, int fe_degree>
template<bool is_stream>
void
ModelProblem<dim, fe_degree>::assemble_system_impl()
{
  const auto * load_function_ptr = is_stream ? load_function_stokes.get() : load_function.get();

  using MatrixIntegrator = typename C0IP::MW::
    MatrixIntegrator<dim, /*is_multigrid*/ false, /*stream function?*/ is_stream>;

  MatrixIntegrator matrix_integrator(load_function_ptr,
                                     analytical_solution.get(),
                                     &system_u,
                                     equation_data);

  auto cell_worker =
    [&](const auto & cell, C0IP::MW::ScratchData<dim> & scratch_data, CopyData & copy_data) {
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

  /// The copier has to use homogeneous boundary constraints for dof transfer
  /// since we would like to assemble a system matrix for the homogeneous
  /// solution @p system_delta_u. The particular solution u0 with the correct
  /// heterogeneous boundary values is incorporated in the right hand side. This
  /// is like using Newton's method on a linear system, thus, resulting in one
  /// Newton step with @p system_delta_u being the Newton update and @p system_u
  /// the initial value.
  const auto copier = [&](const CopyData & copy_data) {
    zero_constraints.template distribute_local_to_global<SparseMatrix<double>, VECTOR>(
      copy_data.cell_matrix,
      copy_data.cell_rhs,
      copy_data.local_dof_indices,
      system_matrix,
      system_rhs);

    for(auto & cdf : copy_data.face_data)
    {
      if(cdf.cell_rhs.size() == 0) // only filled on cells at the boundary
        zero_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix, cdf.joint_dof_indices, system_matrix);
      else
        zero_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix, cdf.cell_rhs, cdf.joint_dof_indices, system_matrix, system_rhs);
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



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::assemble_system()
{
  if(equation_data.is_stream_function)
    assemble_system_impl<true>();
  else
    assemble_system_impl<false>();
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::prepare_schwarz_smoothers()
{
  //: pre-smoother
  Assert(rt_parameters.multigrid.pre_smoother.variant ==
           SmootherParameter::SmootherVariant::Schwarz,
         ExcMessage("Invalid smoothing variant."));
  AssertDimension(mg_matrices.max_level(), max_level());
  for(unsigned int level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
    AssertThrow(mg_matrices[level].mf_storage, ExcMessage("mf_storage is not initialized."));

  const auto                                   mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
  typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data;
  mgss_data.coloring_func = std::ref(*user_coloring);
  mgss_data.parameters    = rt_parameters.multigrid.pre_smoother;
  // mgss_data.dirichlet_ids.emplace_back(equation_data.dirichlet_boundary_ids);
  mgss->initialize(mg_matrices, mgss_data);
  mg_schwarz_smoother_pre = mgss;

  //: post-smoother (so far only shallow copy!)
  {
    typename SubdomainHandler<dim, double>::AdditionalData sd_handler_data;
    rt_parameters.template fill_schwarz_smoother_data<dim, double>(sd_handler_data,
                                                                   /*is_pre?*/ false);

    const auto mgss_post = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data_post;
    mgss_data_post.coloring_func = std::ref(*user_coloring);
    mgss_data_post.parameters    = rt_parameters.multigrid.post_smoother;
    mgss_post->initialize(*mg_schwarz_smoother_pre, mgss_data_post);
    mg_schwarz_smoother_post = mgss_post;
  }
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::prepare_multigrid()
{
  // *** clear multigrid infrastructure
  multigrid.reset();
  mg_matrix_wrapper.reset();
  coarse_grid_solver.clear();
  mg_smoother_post = nullptr;
  mg_smoother_pre  = nullptr;
  mg_schwarz_smoother_pre.reset();
  mg_schwarz_smoother_post.reset();
  mg_transfer.clear();
  mg_matrices.clear_elements();
  mg_constrained_dofs.reset();

  // *** setup multigrid data
  const unsigned mg_level_min = rt_parameters.multigrid.coarse_level;
  const unsigned mg_level_max = max_level();
  pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);

  // *** initialize multigrid constraints
  mg_constrained_dofs = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs->initialize(dof_handler);
  mg_constrained_dofs->make_zero_boundary_constraints(dof_handler,
                                                      equation_data.dirichlet_boundary_ids);

  // *** initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  mg_sparsity_patterns.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(level), dof_handler.n_dofs(level));
    MGTools::make_flux_sparsity_pattern(dof_handler, dsp, level);
    mg_sparsity_patterns[level].copy_from(dsp);
    mg_matrices[level].reinit(mg_sparsity_patterns[level]);

    // assemble
    AffineConstraints<double> level_constraints;
    IndexSet                  relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    using MatrixIntegrator = C0IP::MW::MatrixIntegrator<dim, /*is_multigrid*/ true>;
    MatrixIntegrator matrix_integrator(nullptr, analytical_solution.get(), nullptr, equation_data);
    using LevelCellIterator = typename MatrixIntegrator::IteratorType;

    auto cell_worker =
      [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const LevelCellIterator & cell,
                           const unsigned int &      f,
                           const unsigned int &      sf,
                           const LevelCellIterator & ncell,
                           const unsigned int &      nf,
                           const unsigned int &      nsf,
                           ScratchData<dim> &        scratch_data,
                           CopyData &                copy_data) {
      matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const LevelCellIterator & cell,
                               const unsigned int &      face_no,
                               ScratchData<dim> &        scratch_data,
                               CopyData &                copy_data) {
      matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
    };

    const auto copier = [&](const CopyData & copy_data) {
      AssertDimension(copy_data.level, level);
      level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
        copy_data.cell_matrix, copy_data.local_dof_indices, mg_matrices[copy_data.level]);

      for(auto & cdf : copy_data.face_data)
      {
        level_constraints.template distribute_local_to_global<SparseMatrix<double>>(
          cdf.cell_matrix, cdf.joint_dof_indices, mg_matrices[copy_data.level]);
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

    /// initialize matrix-free storage (dummy, required to setup TPSS) + local
    /// matrix integrator
    {
      typename MatrixFree<dim, double>::AdditionalData mf_features;
      mf_features.mg_level = level;
      QGauss<1>  quadrature(fe_degree + 1);
      const auto mf_storage_ = std::make_shared<MatrixFree<dim, double>>();
      mf_storage_->reinit(mapping, dof_handler, level_constraints, quadrature, mf_features);
      mg_matrices[level].initialize(mf_storage_, equation_data);
    }
  }

  // *** initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  // *** initialize Schwarz smoother S_l
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
      prepare_schwarz_smoothers();
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
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
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Not initialized"));
      mg_smoother_post = mg_schwarz_smoother_post.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }

  pp_data.n_colors_system.push_back(n_colors_system());

  // *** initialize coarse grid solver
  coarse_grid_solver.initialize(mg_matrices[mg_level_min], rt_parameters.multigrid.coarse_grid);
  mg_coarse_grid = &coarse_grid_solver;

  mg_matrix_wrapper.initialize(mg_matrices);
  multigrid = std::make_shared<Multigrid<VECTOR>>(mg_matrix_wrapper,
                                                  *mg_coarse_grid,
                                                  mg_transfer,
                                                  *mg_smoother_pre,
                                                  *mg_smoother_post,
                                                  mg_level_min,
                                                  mg_level_max);
}



template<int dim, int fe_degree>
template<typename PreconditionerType>
void
ModelProblem<dim, fe_degree>::iterative_solve_impl(const PreconditionerType & preconditioner)
{
  ReductionControl solver_control;
  solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  solver_control.set_reduction(rt_parameters.solver.rel_tolerance);
  solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  solver_control.log_history(true);
  solver_control.log_result(true);
  solver_control.enable_history_data();

  /// DEBUG
  // IterationNumberControl solver_control;
  // solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  // solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  // solver_control.log_history(true);
  // solver_control.log_result(true);
  // solver_control.enable_history_data();

  SolverSelector<VECTOR> iterative_solver;
  iterative_solver.set_control(solver_control);
  iterative_solver.select(rt_parameters.solver.variant);
  iterative_solver.solve(system_matrix, system_delta_u, system_rhs, preconditioner);
  system_u += system_delta_u;

  const auto [n_frac, reduction_rate] = compute_fractional_steps(solver_control);
  pp_data.average_reduction_system.push_back(reduction_rate);
  pp_data.n_iterations_system.push_back(n_frac);
  print_parameter("Average reduction (solver):", reduction_rate);
  print_parameter("Number of iterations (solver):", n_frac);
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::solve()
{
  print_parameter("Solving system", "...");

  if(rt_parameters.solver.variant == "direct")
  {
    SparseDirectUMFPACK A_direct;
    A_direct.template initialize<SparseMatrix<double>>(system_matrix);
    A_direct.vmult(system_delta_u, system_rhs);
    system_u += system_delta_u;
    pp_data.average_reduction_system.push_back(0.);
    pp_data.n_iterations_system.push_back(0.);
    print_parameter("Average reduction (solver):", "direct solver");
    print_parameter("Number of iterations (solver):", "---");
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



template<int dim, int fe_degree>
double
ModelProblem<dim, fe_degree>::compute_stream_function_error()
{
  AssertThrow(analytical_velocity, ExcMessage("analytical_velocity isn't initialized."));
  AssertDimension(analytical_velocity->n_components, dim);

  using ::MW::ScratchData;

  using ::MW::CopyData;

  using ::MW::compute_vcurl;

  AffineConstraints empty_constraints;
  empty_constraints.close();

  Vector<double> norm_per_cell(triangulation.n_active_cells());

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    auto & phi = scratch_data.fe_values;
    phi.reinit(cell);

    const unsigned int n_dofs_per_cell = phi.get_fe().dofs_per_cell;
    const unsigned int n_q_points      = phi.n_quadrature_points;
    const auto &       q_points        = phi.get_quadrature_points();

    std::vector<Tensor<1, dim>> velocity_values;
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(velocity_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> u_q;
                     for(auto c = 0U; c < dim; ++c)
                       u_q[c] = analytical_velocity->value(x_q, c);
                     return u_q;
                   });

    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
    cell->get_active_or_mg_dof_indices(local_dof_indices);
    std::vector<double> stream_function_dof_values;
    std::transform(local_dof_indices.cbegin(),
                   local_dof_indices.cend(),
                   std::back_inserter(stream_function_dof_values),
                   [this](const auto & i) { return system_u(i); });

    double local_error = 0.;
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      Tensor<1, dim> uh_q;
      for(unsigned int i = 0; i < n_dofs_per_cell; ++i)
      {
        const auto & alpha_i    = stream_function_dof_values[i];
        const auto & curl_phi_i = compute_vcurl(phi, i, q);
        uh_q += alpha_i * curl_phi_i;
      }

      const auto & u_q = velocity_values[q];
      local_error += (uh_q - u_q) * (uh_q - u_q) * phi.JxW(q);
    }

    AssertDimension(copy_data.cell_rhs.size(), 1U);
    AssertDimension(copy_data.local_dof_indices.size(), 1U);
    AssertIndexRange(cell->index(), norm_per_cell.size());

    copy_data.local_dof_indices[0] = cell->index();
    copy_data.cell_rhs(0)          = std::sqrt(local_error);
  };

  const auto copier = [&](const CopyData & copy_data) {
    /// We first store cell-wise errors to avoid data races in the mesh_loop()
    /// call and then accumulate the global error, instead of directly copying
    /// all local errors to one global error field.
    AssertDimension(copy_data.cell_rhs.size(), 1U);
    AssertDimension(copy_data.local_dof_indices.size(), 1U);
    const auto cell_index     = copy_data.local_dof_indices[0];
    norm_per_cell(cell_index) = copy_data.cell_rhs(0);
  };

  const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
  const UpdateFlags  update_flags =
    update_values | update_gradients | update_quadrature_points | update_JxW_values;
  const UpdateFlags interface_update_flags  = update_default;
  const auto        n_error_values_per_cell = 1U;

  ScratchData<dim> scratch_data(
    mapping, dof_handler.get_fe(), n_gauss_points, update_flags, interface_update_flags);

  CopyData copy_data(n_error_values_per_cell);

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells);

  return VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::L2_norm);
}



// The next function evaluates the error between the computed solution
// and the exact solution (which is known here because we have chosen
// the right hand side and boundary values in a way so that we know
// the corresponding solution). In the first two code blocks below,
// we compute the error in the $L_2$ norm and the $H^1$ semi-norm.
template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::compute_errors()
{
  if(equation_data.is_stream_function)
  {
    const double l2_velocity_error = compute_stream_function_error();
    print_parameter("L2 velocity error (stream function):", l2_velocity_error);
    pp_data_stokes.L2_error.push_back(l2_velocity_error);
  }

  {
    Vector<double> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      system_u,
                                      *analytical_solution,
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    const double error_norm =
      VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::L2_norm);
    print_parameter("Error in the L2 norm:", error_norm);
    pp_data.L2_error.push_back(error_norm);
  }

  {
    Vector<double> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      system_u,
                                      *analytical_solution,
                                      norm_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::H1_seminorm);
    const double error_norm =
      VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::H1_seminorm);
    print_parameter("Error in the H1 seminorm:", error_norm);
    pp_data.H1semi_error.push_back(error_norm);
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
    const QGauss<dim> quadrature_formula(fe.degree + 2);
    Vector<double>    error_per_cell(triangulation.n_active_cells());

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
      analytical_solution->hessian_list(fe_values.get_quadrature_points(), exact_hessians);

      double local_error = 0;
      for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        local_error +=
          ((exact_hessians[q_point] - hessians[q_point]).norm_square() * fe_values.JxW(q_point));
      }
      error_per_cell[cell->active_cell_index()] = std::sqrt(local_error);
    }

    const double error_norm = error_per_cell.l2_norm();
    print_parameter("Error in the broken H2 seminorm:", error_norm);
    pp_data.H2semi_error.push_back(error_norm);
  }
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::output_results(const unsigned int iteration) const
{
  print_parameter("Writing graphical output", "...");

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(system_u, "solution");
  data_out.build_patches();

  std::ofstream output_vtk(("biharm_" + equation_data.sstr_equation_variant() + "_" +
                            Utilities::int_to_string(iteration, 6) + ".vtk")
                             .c_str());
  data_out.write_vtk(output_vtk);
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::run()
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

    output_results(cycle);

    compute_errors();

    *pcout << std::endl;
  }
}



} // namespace Biharmonic

#endif // APPS_BIHARMONICPROBLEM_H_
