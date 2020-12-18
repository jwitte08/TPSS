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
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_hierarchical.h>
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
#include "stokes.h"


namespace Biharmonic
{
using namespace dealii;

/**
 * This class augments the (trilinos) sparse matrix struct to suit the TPSS
 * interface and adds some convenience functions.
 */
template<int dim, int fe_degree = 2, typename Number = double>
class SparseMatrixAugmented : public TrilinosWrappers::SparseMatrix,
                              public C0IP::FD::MatrixIntegrator<dim, fe_degree, Number>
{
public:
  using value_type            = Number;
  using matrix_type           = TrilinosWrappers::SparseMatrix;
  using local_integrator_type = C0IP::FD::MatrixIntegrator<dim, fe_degree, Number>;
  using vector_type           = LinearAlgebra::distributed::Vector<Number>;

  void
  initialize(const TrilinosWrappers::SparsityPattern & dsp,
             const IndexSet &                          locally_owned_dof_indices,
             const IndexSet &                          ghosted_dof_indices,
             const MPI_Comm &                          mpi_communicator);

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in);

  void
  initialize_dof_vector(vector_type & vec) const;

  void
  initialize_dof_vector_mf(vector_type & vec) const;

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const;

  using matrix_type::vmult;

  void
  vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const;

  void
  clear();

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
  std::shared_ptr<const MatrixFree<dim, Number>>     mf_storage;
};



/**
 * (1) Biharmonic model problem with clamped boundary conditions. The problem
 * is discretized by means of the C^0 interior penalty method (C0IP).
 *
 * (2) Stream function formulation for Stokes model problems discretized by
 * means of H^div-conforming interior penalty methods.
 */
template<int dim, int fe_degree = 2>
class ModelProblem
{
  static_assert(dim == 2, "The model problem is implemented for 2D only.");
  static_assert(
    fe_degree >= 2,
    "The C0IP formulation for the biharmonic problem is reasonable for finite elements of polynomial degree two or larger.");

public:
  using VECTOR = LinearAlgebra::distributed::Vector<double>;
  using MATRIX = SparseMatrixAugmented<dim, fe_degree, double>;

  using MG_TRANSFER         = MGTransferMatrixFree<dim, double>;
  using PATCH_MATRIX        = Tensors::TensorProductMatrix<dim, VectorizedArray<double>>;
  using MG_SMOOTHER_SCHWARZ = MGSmootherSchwarz<dim, MATRIX, PATCH_MATRIX, VECTOR>;
  using GMG_PRECONDITIONER  = PreconditionMG<dim, VECTOR, MG_TRANSFER>;

  static constexpr unsigned int n_q_points_1d    = fe_degree + 1;
  static constexpr unsigned int fe_degree_static = fe_degree;

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
  prepare_preconditioner_mg();

  const TrilinosWrappers::PreconditionAMG &
  prepare_preconditioner_amg();

  const TrilinosWrappers::PreconditionBlockwiseDirect &
  prepare_preconditioner_blockdirect();

  void
  solve();

  void
  solve_pressure();

  void
  compute_discretization_errors();

  void
  output_results(const unsigned int iteration) const;

  template<typename OtherNumber>
  std::shared_ptr<const MatrixFree<dim, OtherNumber>>
  build_mf_storage() const;

  unsigned int
  max_level() const;

  unsigned int
  n_mg_levels() const;

  unsigned int
  n_colors_system() const;

  void
  print_informations() const;

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
  std::shared_ptr<FiniteElement<dim>>       finite_element;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<double>                 constraints;
  AffineConstraints<double>                 zero_constraints;

  MATRIX            system_matrix;
  const UpdateFlags update_flags;
  const UpdateFlags update_flags_interface;
  VECTOR            system_u;
  VECTOR            system_delta_u;
  VECTOR            system_rhs;

  std::shared_ptr<MGConstrainedDoFs>                mg_constrained_dofs;
  MG_TRANSFER                                       mg_transfer;
  MGLevelObject<MATRIX>                             mg_matrices;
  mutable std::shared_ptr<ColoringBase<dim>>        user_coloring;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>        mg_schwarz_smoother_pre;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>        mg_schwarz_smoother_post;
  std::shared_ptr<const MGSmootherIdentity<VECTOR>> mg_smoother_identity;
  std::shared_ptr<const MGSmootherPrecondition<TrilinosWrappers::SparseMatrix,
                                               TrilinosWrappers::PreconditionSSOR,
                                               VECTOR>>
                                                           mg_smoother_gauss_seidel;
  const MGSmootherBase<VECTOR> *                           mg_smoother_pre;
  const MGSmootherBase<VECTOR> *                           mg_smoother_post;
  CoarseGridSolver<TrilinosWrappers::SparseMatrix, VECTOR> coarse_grid_solver;
  const MGCoarseGridBase<VECTOR> *                         mg_coarse_grid;
  mg::Matrix<VECTOR>                                       mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>>                       multigrid;

  std::shared_ptr<GMG_PRECONDITIONER> preconditioner_mg;

  std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_amg;

  std::shared_ptr<TrilinosWrappers::PreconditionBlockwiseDirect> preconditioner_blockdirect;

  static constexpr unsigned int               fe_degree_pressure = fe_degree - 1;
  Stokes::StokesFlow<dim, fe_degree_pressure> options_stokes;
  Stokes::EquationData                        equation_data_stokes;
  std::shared_ptr<Stokes::ModelProblem<dim, fe_degree_pressure, Stokes::Method::RaviartThomas>>
    stokes_problem;

  unsigned int proc_no;

private:
  void
  make_grid_impl(const MeshParameter & mesh_prms);

  template<bool is_stream_function>
  void
  assemble_system_impl();

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner);

  std::shared_ptr<SolverControl>
  get_solver_control() const;

  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const;

  /**
   * Computes the energy (semi)norm induced by the C^0 interior penalty bilinear
   * form. To be precise, we actually compute the mesh-dependent seminorm from
   * equation (3.5) in BrennerSung05 which is equivalent to the energy norm
   * (there given in equation (4.19)) on the finite element space. For smooth
   * functions the mesh-dependent seminorm is equivalent to the H^2-seminorm.
   */
  double
  compute_energy_error() const;

  std::array<LAPACKFullMatrix<double>, 2>
  compute_nondivfree_shape_functions() const;

  double
  compute_stream_function_error();

  std::shared_ptr<Vector<double>>
  compute_L2_error_pressure() const;
};



////////////////////////////// Defintions

template<int dim, int fe_degree, typename Number>
void
SparseMatrixAugmented<dim, fe_degree, Number>::initialize(
  const TrilinosWrappers::SparsityPattern & dsp,
  const IndexSet &                          locally_owned_dof_indices,
  const IndexSet &                          ghosted_dof_indices,
  const MPI_Comm &                          mpi_communicator)
{
  Assert(dsp.is_compressed(), ExcMessage("The sparsity pattern isn't compressed."));
  matrix_type::reinit(dsp);
  partitioner = std::make_shared<const Utilities::MPI::Partitioner>(locally_owned_dof_indices,
                                                                    ghosted_dof_indices,
                                                                    mpi_communicator);
}


template<int dim, int fe_degree, typename Number>
void
SparseMatrixAugmented<dim, fe_degree, Number>::initialize(
  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
  const EquationData                             equation_data_in)
{
  mf_storage = mf_storage_in;
  local_integrator_type::initialize(equation_data_in);
}


template<int dim, int fe_degree, typename Number>
void
SparseMatrixAugmented<dim, fe_degree, Number>::initialize_dof_vector(vector_type & vec) const
{
  Assert(partitioner, ExcMessage("Did you initialize partitioner?"));
  vec.reinit(partitioner);
}


template<int dim, int fe_degree, typename Number>
void
SparseMatrixAugmented<dim, fe_degree, Number>::initialize_dof_vector_mf(vector_type & vec) const
{
  Assert(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  mf_storage->initialize_dof_vector(vec);
}


template<int dim, int fe_degree, typename Number>
std::shared_ptr<const MatrixFree<dim, Number>>
SparseMatrixAugmented<dim, fe_degree, Number>::get_matrix_free() const
{
  AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  return mf_storage;
}


template<int dim, int fe_degree, typename Number>
void
SparseMatrixAugmented<dim, fe_degree, Number>::vmult(const ArrayView<Number>       dst,
                                                     const ArrayView<const Number> src) const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("No MPI support"));
  AssertDimension(dst.size(), matrix_type::m());
  AssertDimension(src.size(), matrix_type::n());
  vector_type v(matrix_type::n()); // src
  std::copy(src.cbegin(), src.cend(), v.begin());
  vector_type w(matrix_type::m()); // dst
  matrix_type::vmult(w, v);        // w = A v
  std::copy(w.begin(), w.end(), dst.begin());
}


template<int dim, int fe_degree, typename Number>
void
SparseMatrixAugmented<dim, fe_degree, Number>::clear()
{
  mf_storage.reset();
  partitioner.reset();
  matrix_type::clear();
}



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
        return std::make_shared<Clamped::NoSlip::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleNoSlip)
        return std::make_shared<Clamped::Poiseuille::NoSlip::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<Clamped::NoSlipNormal::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleInhom)
        return std::make_shared<Clamped::Poiseuille::Inhom::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedHomPoly)
        return std::make_shared<Clamped::Homogeneous::Poly::Solution<dim>>();
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
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleNoSlip)
        return std::make_shared<
          Stokes::DivergenceFree::Poiseuille::NoSlip::SolutionVelocity<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<Stokes::DivergenceFree::NoSlipNormal::SolutionVelocity<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleInhom)
        return std::make_shared<Stokes::DivergenceFree::Poiseuille::Inhom::SolutionVelocity<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedHomPoly)
        return nullptr;
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
        return std::make_shared<Clamped::NoSlip::Load<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleNoSlip)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<ManufacturedLoad<dim, Clamped::NoSlipNormal::Solution<dim>>>();
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleInhom)
        return nullptr;
      else if(equation_data_in.variant == EquationData::Variant::ClampedHomPoly)
        return std::make_shared<Clamped::Homogeneous::Poly::Load<dim>>();
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
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleNoSlip)
        return std::make_shared<Stokes::ManufacturedLoad<dim>>(
          std::make_shared<Stokes::DivergenceFree::Poiseuille::NoSlip::Solution<dim>>());
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamNoSlipNormal)
        return std::make_shared<Stokes::ManufacturedLoad<dim>>(
          std::make_shared<Stokes::DivergenceFree::NoSlipNormal::Solution<dim>>());
      else if(equation_data_in.variant == EquationData::Variant::ClampedStreamPoiseuilleInhom)
        return std::make_shared<Stokes::ManufacturedLoad<dim>>(
          std::make_shared<Stokes::DivergenceFree::Poiseuille::Inhom::Solution<dim>>());
      else if(equation_data_in.variant == EquationData::Variant::ClampedHomPoly)
        return nullptr;
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
    finite_element(std::make_shared<FE_Q<dim>>(fe_degree)),
    update_flags(update_values | update_hessians | update_quadrature_points | update_JxW_values),
    update_flags_interface(update_values | update_gradients | update_hessians |
                           update_quadrature_points | update_JxW_values | update_normal_vectors),
    user_coloring([&]() -> std::shared_ptr<ColoringBase<dim>> {
      const bool is_AVP = rt_parameters.multigrid.pre_smoother.schwarz.is_additive_vertex_patch();
      Assert(rt_parameters.multigrid.pre_smoother.schwarz.is_additive_vertex_patch() ==
               rt_parameters.multigrid.post_smoother.schwarz.is_additive_vertex_patch(),
             ExcMessage("Pre- and postsmoother differ."));
      const bool is_multithreaded = MultithreadInfo::n_threads() > 1;
      if(is_AVP && is_multithreaded)
        return std::make_shared<TiledColoring<dim>>(rt_parameters_in.mesh);
      return std::make_shared<RedBlackColoring<dim>>(rt_parameters_in.mesh);
    }()),
    proc_no(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
{
  AssertThrow(rt_parameters.multigrid.pre_smoother.schwarz.patch_variant ==
                TPSS::PatchVariant::vertex,
              ExcMessage("Model problem is designed for Schwarz methods on vertex patches."));

  if(equation_data.is_stream_function())
  {
    Assert(load_function_stokes, ExcMessage("load_function_stokes is required."));

    const auto damping = rt_parameters.multigrid.pre_smoother.schwarz.damping_factor;
    options_stokes.setup(0U, damping);
    options_stokes.prms.n_cycles = rt_parameters.n_cycles;
    options_stokes.prms.mesh     = rt_parameters.mesh;

    std::map<EquationData::Variant, Stokes::EquationData::Variant> biharm_to_stokes_variant = {
      {EquationData::Variant::ClampedStreamNoSlip, Stokes::EquationData::Variant::DivFreeNoSlip},
      {EquationData::Variant::ClampedStreamPoiseuilleNoSlip,
       Stokes::EquationData::Variant::DivFreePoiseuilleNoSlip},
      {EquationData::Variant::ClampedStreamNoSlipNormal,
       Stokes::EquationData::Variant::DivFreeNoSlipNormal},
      {EquationData::Variant::ClampedStreamPoiseuilleInhom,
       Stokes::EquationData::Variant::DivFreePoiseuilleInhom}};

    equation_data_stokes.variant           = biharm_to_stokes_variant.at(equation_data.variant);
    equation_data_stokes.use_cuthill_mckee = false;
    if(options_stokes.prms.solver.variant == "GMRES_GMG" ||
       options_stokes.prms.solver.variant == "CG_GMG")
      equation_data_stokes.local_kernel_size = 1U;
    if(options_stokes.prms.solver.variant == "UMFPACK")
      equation_data_stokes.force_mean_value_constraint = true;

    stokes_problem = std::make_shared<
      Stokes::ModelProblem<dim, fe_degree_pressure, Stokes::Method::RaviartThomas>>(
      options_stokes.prms, equation_data_stokes);
    stokes_problem->pcout = pcout;
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

  const auto n_dofs_est = estimate_n_dofs(*finite_element, mesh_prms);
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
template<typename OtherNumber>
std::shared_ptr<const MatrixFree<dim, OtherNumber>>
ModelProblem<dim, fe_degree>::build_mf_storage() const
{
  typename MatrixFree<dim, OtherNumber>::AdditionalData mf_features;
  mf_features.tasks_parallel_scheme            = MatrixFree<dim, OtherNumber>::AdditionalData::none;
  mf_features.mapping_update_flags             = update_flags;
  mf_features.mapping_update_flags_inner_faces = update_flags_interface;
  mf_features.mapping_update_flags_boundary_faces = update_flags_interface;

  QGauss<1> quadrature(n_q_points_1d);

  const auto mf_storage = std::make_shared<MatrixFree<dim, OtherNumber>>();
  mf_storage->reinit(mapping, dof_handler, constraints, quadrature, mf_features);

  return mf_storage;
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::setup_system()
{
  dof_handler.clear();
  dof_handler.initialize(triangulation, *finite_element);
  dof_handler.distribute_mg_dofs();

  print_parameter("Number of degrees of freedom:", dof_handler.n_dofs());

  const auto & locally_owned_dof_indices = dof_handler.locally_owned_dofs();
  IndexSet     locally_relevant_dof_indices;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dof_indices);

  constraints.clear();
  constraints.reinit(locally_relevant_dof_indices);
  if(dof_handler.get_fe().has_support_points())
  {
    print_parameter("Interpolating boundary values", "...");
    VectorTools::interpolate_boundary_values(dof_handler, 0, *analytical_solution, constraints);
  }
  else
  {
    print_parameter("Projecting boundary values", "...");
    std::map<types::boundary_id, const Function<dim> *> boundary_id_to_function;
    for(const auto id : equation_data.dirichlet_boundary_ids)
      boundary_id_to_function.emplace(id, analytical_solution.get());
    VectorTools::project_boundary_values(dof_handler,
                                         boundary_id_to_function,
                                         QGauss<dim - 1>(fe_degree + 1),
                                         constraints);
  }
  constraints.close();

  zero_constraints.clear();
  zero_constraints.reinit(locally_relevant_dof_indices);
  for(const auto boundary_id : equation_data.dirichlet_boundary_ids)
    DoFTools::make_zero_boundary_constraints(dof_handler, boundary_id, zero_constraints);
  zero_constraints.close();

  TrilinosWrappers::SparsityPattern dsp(locally_owned_dof_indices,
                                        locally_owned_dof_indices,
                                        locally_relevant_dof_indices,
                                        MPI_COMM_WORLD);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints, false);
  dsp.compress();

  // system_matrix.initialize(build_mf_storage<double>(), equation_data);
  system_matrix.initialize(dsp,
                           locally_owned_dof_indices,
                           locally_relevant_dof_indices,
                           MPI_COMM_WORLD);
  pp_data.n_dofs_global.push_back(system_matrix.m());

  system_u.reinit(0);
  system_delta_u.reinit(0);
  system_rhs.reinit(0);

  system_matrix.initialize_dof_vector(system_u);
  constraints.distribute(system_u); // set boundary values (particular solution)

  system_matrix.initialize_dof_vector(system_delta_u);
  zero_constraints.distribute(system_delta_u); // set zero boundary values (homo. solution)

  system_matrix.initialize_dof_vector(system_rhs);
}



template<int dim, int fe_degree>
template<bool is_stream>
void
ModelProblem<dim, fe_degree>::assemble_system_impl()
{
  if(is_stream)
  {
    AssertThrow(false, ExcMessage("TODO MPI..."));
    // using Stokes::Velocity::SIPG::MW::ScratchData;

    // using Stokes::Velocity::SIPG::MW::CopyData;

    // const auto                     velocity_components = std::make_pair<unsigned int>(0, dim);
    // Stokes::FunctionExtractor<dim> load_function_stream(load_function_stokes.get(),
    //                                                     velocity_components);

    // Stokes::FunctionExtractor<dim> analytical_solution_velocity(analytical_velocity.get(),
    //                                                             velocity_components);

    // using MatrixIntegrator =
    //   typename Stokes::Velocity::SIPG::MW::MatrixIntegrator<dim, /*is_multigrid*/ false>;

    // MatrixIntegrator matrix_integrator(&load_function_stream,
    //                                    &analytical_solution_velocity,
    //                                    &system_u,
    //                                    equation_data_stokes);

    // auto cell_worker =
    //   [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    //     matrix_integrator.cell_worker_stream(cell, scratch_data, copy_data);
    //   };

    // auto face_worker = [&](const auto &         cell,
    //                        const unsigned int & f,
    //                        const unsigned int & sf,
    //                        const auto &         ncell,
    //                        const unsigned int & nf,
    //                        const unsigned int & nsf,
    //                        ScratchData<dim> &   scratch_data,
    //                        CopyData &           copy_data) {
    //   matrix_integrator.face_worker_stream(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
    // };

    // auto boundary_worker = [&](const auto &         cell,
    //                            const unsigned int & face_no,
    //                            ScratchData<dim> &   scratch_data,
    //                            CopyData &           copy_data) {
    //   matrix_integrator.boundary_worker_stream(cell, face_no, scratch_data, copy_data);
    // };

    // /// The copier has to use homogeneous boundary constraints for dof transfer
    // /// since we would like to assemble a system matrix for the homogeneous
    // /// solution @p system_delta_u. The particular solution u0 with the correct
    // /// heterogeneous boundary values is incorporated in the right hand side. This
    // /// is like using Newton's method on a linear system, thus, resulting in one
    // /// Newton step with @p system_delta_u being the Newton update and @p system_u
    // /// the initial value.
    // const auto copier = [&](const CopyData & copy_data) {
    //   zero_constraints.template distribute_local_to_global<SparseMatrix<double>, VECTOR>(
    //     copy_data.cell_matrix,
    //     copy_data.cell_rhs_test,
    //     copy_data.local_dof_indices_test,
    //     system_matrix,
    //     system_rhs);

    //   for(auto & cdf : copy_data.face_data)
    //   {
    //     if(cdf.cell_rhs_test.size() == 0) // only filled on cells at the boundary
    //       zero_constraints.template distribute_local_to_global<SparseMatrix<double>>(
    //         cdf.cell_matrix, cdf.joint_dof_indices_test, system_matrix);
    //     else
    //       zero_constraints.template distribute_local_to_global<SparseMatrix<double>>(
    //         cdf.cell_matrix,
    //         cdf.cell_rhs_test,
    //         cdf.joint_dof_indices_test,
    //         system_matrix,
    //         system_rhs);
    //   }
    // };

    // UpdateFlags update_flags = update_values | update_gradients | update_hessians |
    //                            update_quadrature_points | update_JxW_values;
    // UpdateFlags        interface_update_flags = update_flags | update_normal_vectors;
    // const unsigned int n_gauss_points         = dof_handler.get_fe().degree + 1;

    // ScratchData<dim> scratch_data(mapping,
    //                               *finite_element,
    //                               *finite_element,
    //                               n_gauss_points,
    //                               update_flags,
    //                               update_flags,
    //                               interface_update_flags,
    //                               interface_update_flags);

    // CopyData copy_data(dof_handler.get_fe().dofs_per_cell, dof_handler.get_fe().dofs_per_cell);

    // MeshWorker::mesh_loop(dof_handler.begin_active(),
    //                       dof_handler.end(),
    //                       cell_worker,
    //                       copier,
    //                       scratch_data,
    //                       copy_data,
    //                       MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
    //                         MeshWorker::assemble_own_interior_faces_once,
    //                       boundary_worker,
    //                       face_worker);
  }

  else
  {
    using C0IP::MW::ScratchData;

    using C0IP::MW::CopyData;

    using MatrixIntegrator =
      typename C0IP::MW::MatrixIntegrator<dim, /*multigrid?*/ false, /*stream function?*/ false>;

    system_u.update_ghost_values();
    system_rhs.zero_out_ghosts();

    MatrixIntegrator matrix_integrator(load_function.get(),
                                       analytical_solution.get(),
                                       &system_u,
                                       equation_data);

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & face_no,
                           const unsigned int & sface_no,
                           const auto &         ncell,
                           const unsigned int & nface_no,
                           const unsigned int & nsface_no,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      matrix_integrator.face_worker(
        cell, face_no, sface_no, ncell, nface_no, nsface_no, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
    };

    /// The copier has to transfer dof values with respect to homogeneous
    /// boundary constraints when assembling a system matrix 'A' (@p
    /// system_matrix) for the homogeneous solution '\Delta u'(@p
    /// system_delta_u). Subtracting the matrix-vector product 'A u_0' for a
    /// particular solution u_0 from the right-hand side vector 'b' lifts
    /// heterogeneous boundary values. This is like using Newton's method on the
    /// residual of the linear system 'F(u) = b - A u': the problem gets solved
    /// in one Newton step by correcting a particular solution 'u_0' (@p
    /// system_u) as initial guess with the Newton update '\Delta u' (@p
    /// system_delta_u).
    const auto copier = [&](const CopyData & copy_data) {
      for(const auto & cell_data : copy_data.cell_data)
      {
        AssertDimension(copy_data.cell_data.size(), 1U);
        zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cell_data.matrix, cell_data.rhs, cell_data.dof_indices, system_matrix, system_rhs);
      }

      for(auto & cdf : copy_data.face_data)
      {
        zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cdf.matrix, cdf.rhs, cdf.dof_indices, system_matrix, system_rhs);
      }
    };

    ScratchData<dim> scratch_data(
      mapping, dof_handler.get_fe(), n_q_points_1d, update_flags, update_flags_interface);

    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once |
                            MeshWorker::assemble_ghost_faces_once,
                          boundary_worker,
                          face_worker);

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::assemble_system()
{
  if(equation_data.is_stream_function())
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
    AssertThrow(mg_matrices[level].mf_storage, ExcMessage("mf_storage isn't initialized."));

  typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data;
  mgss_data.coloring_func = std::ref(*user_coloring);
  mgss_data.use_tbb       = rt_parameters.use_tbb;
  mgss_data.parameters    = rt_parameters.multigrid.pre_smoother;

  const auto mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
  mgss->initialize(mg_matrices, mgss_data);
  mg_schwarz_smoother_pre = mgss;

  //: post-smoother (so far only shallow copy!)
  {
    auto mgss_data_post       = mgss_data;
    mgss_data_post.parameters = rt_parameters.multigrid.post_smoother;

    const auto mgss_post = std::make_shared<MG_SMOOTHER_SCHWARZ>();
    mgss_post->initialize(*mg_schwarz_smoother_pre, mgss_data_post);
    mg_schwarz_smoother_post = mgss_post;
  }
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::prepare_multigrid()
{
  //: clear multigrid infrastructure
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

  //: setup multigrid data
  const unsigned mg_level_min = rt_parameters.multigrid.coarse_level;
  const unsigned mg_level_max = max_level();

  //: initialize multigrid constraints
  mg_constrained_dofs = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs->initialize(dof_handler);
  mg_constrained_dofs->make_zero_boundary_constraints(dof_handler,
                                                      equation_data.dirichlet_boundary_ids);

  //: initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    ////////// setup
    const IndexSet & locally_owned_dof_indices = dof_handler.locally_owned_mg_dofs(level);
    IndexSet         locally_relevant_dof_indices;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, locally_relevant_dof_indices);

    AffineConstraints<double> level_constraints;
    level_constraints.reinit(locally_relevant_dof_indices);
    level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    TrilinosWrappers::SparsityPattern dsp(locally_owned_dof_indices,
                                          locally_owned_dof_indices,
                                          locally_relevant_dof_indices,
                                          MPI_COMM_WORLD);
    MGTools::make_flux_sparsity_pattern(dof_handler, dsp, level);
    dsp.compress();

    mg_matrices[level].initialize(dsp,
                                  locally_owned_dof_indices,
                                  locally_relevant_dof_indices,
                                  MPI_COMM_WORLD);
    {
      /// initialize dummy matrix-free storage required to setup TPSS
      typename MatrixFree<dim, double>::AdditionalData mf_features;
      mf_features.mg_level = level;
      QGauss<1>  quadrature(n_q_points_1d);
      const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
      mf_storage->reinit(mapping, dof_handler, level_constraints, quadrature, mf_features);

      /// initialize local integrator required to setup TPSS
      mg_matrices[level].initialize(mf_storage, equation_data);
    }

    ////////// assemble
    using C0IP::MW::ScratchData;

    using C0IP::MW::CopyData;

    using MatrixIntegrator = C0IP::MW::MatrixIntegrator<dim, /*is_multigrid*/ true>;

    using LevelCellIterator = typename MatrixIntegrator::IteratorType;

    MatrixIntegrator matrix_integrator(nullptr, analytical_solution.get(), nullptr, equation_data);

    auto cell_worker =
      [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const LevelCellIterator & cell,
                           const unsigned int &      face_no,
                           const unsigned int &      sface_no,
                           const LevelCellIterator & ncell,
                           const unsigned int &      nface_no,
                           const unsigned int &      nsface_no,
                           ScratchData<dim> &        scratch_data,
                           CopyData &                copy_data) {
      matrix_integrator.face_worker(
        cell, face_no, sface_no, ncell, nface_no, nsface_no, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const LevelCellIterator & cell,
                               const unsigned int &      face_no,
                               ScratchData<dim> &        scratch_data,
                               CopyData &                copy_data) {
      matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
    };

    const auto copier = [&](const CopyData & copy_data) {
      for(const auto & cell_data : copy_data.cell_data)
      {
        AssertDimension(copy_data.cell_data.size(), 1U);
        level_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cell_data.matrix, cell_data.dof_indices, mg_matrices[level]);
      }

      for(auto & cdf : copy_data.face_data)
      {
        level_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cdf.matrix, cdf.dof_indices, mg_matrices[level]);
      }
    };

    ScratchData<dim> scratch_data(
      mapping, dof_handler.get_fe(), n_q_points_1d, update_flags, update_flags_interface);

    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_mg(level),
                          dof_handler.end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once |
                            MeshWorker::assemble_ghost_faces_once,
                          boundary_worker,
                          face_worker);

    mg_matrices[level].compress(VectorOperation::add);
  }

  //: initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  //: initialize Schwarz smoother S_l
  switch(rt_parameters.multigrid.pre_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      mg_smoother_identity = std::make_shared<const MGSmootherIdentity<VECTOR>>();
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::GaussSeidel:
    {
      typename TrilinosWrappers::PreconditionSSOR::AdditionalData ssor_features;
      // ssor_features.n_sweeps = rt_parameters.multigrid.pre_smoother.n_smoothing_steps;
      ssor_features.omega = rt_parameters.multigrid.pre_smoother.damping_factor;
      auto tmp            = std::make_shared<MGSmootherPrecondition<TrilinosWrappers::SparseMatrix,
                                                         TrilinosWrappers::PreconditionSSOR,
                                                         VECTOR>>();
      tmp->initialize(mg_matrices, ssor_features);
      tmp->set_steps(rt_parameters.multigrid.pre_smoother.n_smoothing_steps);
      mg_smoother_gauss_seidel = tmp;
      mg_smoother_pre          = mg_smoother_gauss_seidel.get();
      break;
    }
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

  /// initialize coarse grid solver
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
const typename ModelProblem<dim, fe_degree>::GMG_PRECONDITIONER &
ModelProblem<dim, fe_degree>::prepare_preconditioner_mg()
{
  prepare_multigrid();
  AssertThrow(multigrid, ExcNotInitialized());

  preconditioner_mg = std::make_shared<GMG_PRECONDITIONER>(dof_handler, *multigrid, mg_transfer);
  return *preconditioner_mg;
}


template<int dim, int fe_degree>
const TrilinosWrappers::PreconditionAMG &
ModelProblem<dim, fe_degree>::prepare_preconditioner_amg()
{
  preconditioner_amg = std::make_shared<TrilinosWrappers::PreconditionAMG>();
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_features;
  amg_features.elliptic              = true;
  amg_features.higher_order_elements = true;
  amg_features.smoother_sweeps       = 2;
  amg_features.aggregation_threshold = 0.0002;
  preconditioner_amg->initialize(static_cast<const TrilinosWrappers::SparseMatrix &>(system_matrix),
                                 amg_features);
  return *preconditioner_amg;
}


template<int dim, int fe_degree>
const TrilinosWrappers::PreconditionBlockwiseDirect &
ModelProblem<dim, fe_degree>::prepare_preconditioner_blockdirect()
{
  preconditioner_blockdirect = std::make_shared<TrilinosWrappers::PreconditionBlockwiseDirect>();
  TrilinosWrappers::PreconditionBlockwiseDirect::AdditionalData blockdirect_features;
  blockdirect_features.overlap = 0U;
  preconditioner_blockdirect->initialize(
    static_cast<const TrilinosWrappers::SparseMatrix &>(system_matrix), blockdirect_features);
  return *preconditioner_blockdirect;
}


template<int dim, int fe_degree>
unsigned int
ModelProblem<dim, fe_degree>::max_level() const
{
  return triangulation.n_global_levels() - 1;
}


template<int dim, int fe_degree>
unsigned int
ModelProblem<dim, fe_degree>::n_mg_levels() const
{
  if(mg_matrices.min_level() != mg_matrices.max_level())
  {
    const unsigned mg_level_min = rt_parameters.multigrid.coarse_level;
    const unsigned mg_level_max = max_level();
    AssertDimension(mg_level_max - mg_level_min + 1,
                    mg_matrices.max_level() - mg_matrices.min_level() + 1);
    return mg_level_max - mg_level_min + 1;
  }
  return 0;
}


template<int dim, int fe_degree>
unsigned int
ModelProblem<dim, fe_degree>::n_colors_system() const
{
  if(mg_schwarz_smoother_pre)
    return mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
  if(mg_schwarz_smoother_post)
    return mg_schwarz_smoother_post->get_subdomain_handler()->get_partition_data().n_colors();
  return 0;
}


template<int dim, int fe_degree>
template<typename T>
void
ModelProblem<dim, fe_degree>::print_parameter(const std::string & description,
                                              const T &           value) const
{
  *pcout << Util::parameter_to_fstring(description, value);
}


template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::print_informations() const
{
  *pcout << equation_data.to_string();
  *pcout << std::endl;
  print_parameter("Finite element:", finite_element->get_name());
  *pcout << rt_parameters.to_string();
  *pcout << std::endl;
}


template<int dim, int fe_degree>
std::shared_ptr<SolverControl>
ModelProblem<dim, fe_degree>::get_solver_control() const
{
  auto solver_control = [&]() -> std::shared_ptr<SolverControl> {
    if(rt_parameters.solver.control_variant == SolverParameter::ControlVariant::relative)
    {
      auto control = std::make_shared<ReductionControl>();
      control->set_reduction(rt_parameters.solver.rel_tolerance);
      return control;
    }
    else if(rt_parameters.solver.control_variant == SolverParameter::ControlVariant::absolute)
      return std::make_shared<SolverControl>();
    else
      AssertThrow(false, ExcMessage("ControlVariant isn't supported."));
    return nullptr;
  }();
  AssertThrow(solver_control, ExcMessage("ControlVariant isn't supported."));
  solver_control->set_max_steps(rt_parameters.solver.n_iterations_max);
  solver_control->set_tolerance(rt_parameters.solver.abs_tolerance);
  solver_control->log_history(true);
  solver_control->log_result(true);
  solver_control->enable_history_data();

  /// DEBUG
  // IterationNumberControl solver_control;
  // solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  // solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  // solver_control.log_history(true);
  // solver_control.log_result(true);
  // solver_control.enable_history_data();

  return solver_control;
}



template<int dim, int fe_degree>
template<typename PreconditionerType>
void
ModelProblem<dim, fe_degree>::iterative_solve_impl(const PreconditionerType & preconditioner)
{
  auto solver_control = get_solver_control();

  SolverSelector<VECTOR> iterative_solver;
  iterative_solver.set_control(*solver_control);
  iterative_solver.select(rt_parameters.solver.variant);
  iterative_solver.solve(system_matrix, system_delta_u, system_rhs, preconditioner);
  system_u += system_delta_u;

  auto reduction_control = dynamic_cast<ReductionControl *>(solver_control.get());
  if(reduction_control)
  {
    const auto [n_frac, reduction_rate] = compute_fractional_steps(*reduction_control);
    pp_data.average_reduction_system.push_back(reduction_rate);
    pp_data.n_iterations_system.push_back(n_frac);
    print_parameter("Average reduction (solver):", reduction_rate);
    print_parameter("Number of iterations (solver):", n_frac);
  }
  else
  {
    pp_data.average_reduction_system.push_back(solver_control->average_reduction());
    pp_data.n_iterations_system.push_back(solver_control->last_step());
    print_parameter("Average reduction (solver):", solver_control->average_reduction());
    print_parameter("Number of iterations (solver):", solver_control->last_step());
  }
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::solve()
{
  print_parameter("Solving system", "...");

  if(rt_parameters.solver.variant == "direct")
  {
    auto solver_control = get_solver_control();

    TrilinosWrappers::SolverDirect::AdditionalData features;
    features.output_solver_details = true;

    TrilinosWrappers::SolverDirect solver(*solver_control, features);
    solver.solve(system_matrix, system_delta_u, system_rhs);

    system_u += system_delta_u;

    pp_data.average_reduction_system.push_back(0.);
    pp_data.n_iterations_system.push_back(0.);

    print_parameter("Average reduction (solver):", "direct (trilinos)");
    print_parameter("Number of iterations (solver):", "direct (trilinos)");

    // if(/*CG + additive Schwarz preconditioner*/)
    // {
    //   const auto & prec = prepare_preconditioner_blockdirect();
    //   pp_data.average_reduction_system.push_back(0.);
    //   pp_data.n_iterations_system.push_back(0.);
    //   print_parameter("Average reduction (solver):", "trilinos pCG");
    //   print_parameter("Number of iterations (solver):", "---");
    //   auto                                       solver_control = get_solver_control();
    //   TrilinosWrappers::SolverCG::AdditionalData cg_features(true);
    //   TrilinosWrappers::SolverCG                 solver(*solver_control, cg_features);
    //   solver.solve(system_matrix, system_delta_u, system_rhs, prec);
    //   system_u += system_delta_u;
    //   // pp_data.average_reduction_system.push_back(solver_control->average_reduction());
    //   // pp_data.n_iterations_system.push_back(solver_control->last_step());
    //   // print_parameter("Average reduction (solver):", solver_control->average_reduction());
    //   // print_parameter("Number of iterations (solver):", solver_control->last_step());
    //   return;
    // }
  }

  else // iterative solver
  {
    switch(rt_parameters.solver.precondition_variant)
    {
      case SolverParameter::PreconditionVariant::None:
      {
        // const auto & prec = prepare_preconditioner_amg();
        iterative_solve_impl(PreconditionIdentity{});
        break;
      }

      case SolverParameter::PreconditionVariant::GMG:
      {
        prepare_preconditioner_mg();
        iterative_solve_impl(*preconditioner_mg);
        break;
      }

      default:
        AssertThrow(false, ExcNotImplemented());
    }
  }

  pp_data.n_colors_system.push_back(n_colors_system());
  pp_data.n_mg_levels.push_back(n_mg_levels());
}



template<int dim, int fe_degree>
std::array<LAPACKFullMatrix<double>, 2>
ModelProblem<dim, fe_degree>::compute_nondivfree_shape_functions() const
{
  std::array<LAPACKFullMatrix<double>, 2> shape_function_weights;
  auto & [trafomatrix_rt_to_gradp, trafomatrix_rt_to_constp] = shape_function_weights;

  AssertDimension(dim, 2U);

  const auto & fe_v          = stokes_problem->dof_handler_velocity.get_fe();
  const auto & fe_p          = stokes_problem->dof_handler_pressure.get_fe();
  const auto & mapping       = stokes_problem->mapping;
  const auto   n_q_points_1d = stokes_problem->n_q_points_1d;

  const auto n_dofs_per_cell_v            = fe_v.dofs_per_cell;
  const auto n_dofs_per_cell_p            = fe_p.dofs_per_cell;
  const auto n_faces_per_cell             = GeometryInfo<dim>::faces_per_cell;
  const auto n_interior_nodes_by_pressure = n_dofs_per_cell_p - 1;

  Triangulation<dim> unit_triangulation(Triangulation<dim>::maximum_smoothing);
  {
    MeshParameter mesh_prms;
    mesh_prms.geometry_variant = MeshParameter::GeometryVariant::Cube;
    mesh_prms.n_refinements    = 0U;
    mesh_prms.n_repetitions    = 1U;
    create_mesh(unit_triangulation, mesh_prms);
    AssertDimension(unit_triangulation.n_active_cells(), 1U);
  }

  DoFHandler<dim> unit_dofh_v;
  DoFHandler<dim> unit_dofh_p;
  unit_dofh_v.initialize(unit_triangulation, fe_v);
  unit_dofh_p.initialize(unit_triangulation, fe_p);

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
  //                          mapping);
  //   }
  // }

  trafomatrix_rt_to_gradp.reinit(n_interior_nodes_by_pressure, n_dofs_per_cell_v);

  /**
   * Compute the application of the current RT shape functions to the interior
   * RT node functionals, which are generated by the gradients of the pressure
   * shape functions (the constant pressure mode is neglected!).
   */
  {
    QGauss<dim>       quad(n_q_points_1d);
    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;

    FEValues<dim> fe_values_p(mapping, fe_p, quad, update_flags);
    fe_values_p.reinit(unit_dofh_p.begin_active()); // one reference cell

    FEValues<dim> fe_values_v(mapping, fe_v, quad, update_flags);
    fe_values_v.reinit(unit_dofh_v.begin_active()); // one reference cell

    /// Node functionals generated by grad p.
    const auto interior_node_functional = [&](const unsigned int   node_index_p,
                                              const Vector<double> velocity) {
      const auto n_q_points  = fe_values_p.n_quadrature_points;
      const auto dof_index_p = node_index_p + 1; // skip constant mode
      AssertIndexRange(dof_index_p, fe_values_p.dofs_per_cell);
      AssertDimension(velocity.size(), fe_values_v.dofs_per_cell);

      const auto & compute_v_q = [&](const unsigned int q) {
        Tensor<1, dim> value;
        for(unsigned int j = 0; j < n_dofs_per_cell_v; ++j)
          value += velocity[j] * MW::compute_vvalue(fe_values_v, j, q);
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

    LAPACKFullMatrix<double> node_value_weights(n_interior_nodes_by_pressure, n_dofs_per_cell_v);
    for(auto i = 0U; i < node_value_weights.m(); ++i)   // node functionals
      for(auto j = 0U; j < node_value_weights.n(); ++j) // RT shape funcs
      {
        Vector<double> phi_j(n_dofs_per_cell_v);
        phi_j[j] = 1.;

        node_value_weights(i, j) = interior_node_functional(i, phi_j);
      }

    const auto & [V, invSigma, UT] = compute_inverse_svd(node_value_weights);
    /// "Inverse" has to be understood in the sense of an inverse SVD.
    const auto & inverse_node_value_weights = merge_lapack_decomposition(V, invSigma, UT);
    inverse_node_value_weights.transpose(trafomatrix_rt_to_gradp);

    /// Display the "new shape functions" \tilde{v}_i
    for(auto i = 0U; i < n_interior_nodes_by_pressure; ++i)
    {
      Vector<double> phi_i(n_dofs_per_cell_v);
      for(auto j = 0U; j < n_dofs_per_cell_v; ++j)
        phi_i(j) = trafomatrix_rt_to_gradp(i, j);

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
                           mapping);
    }
  }

  trafomatrix_rt_to_constp.reinit(n_faces_per_cell, n_dofs_per_cell_v);

  /**
   * Compute the application of the given RT shape functions to the RT node
   * functional associated to a given face, which is generated by the constant
   * function.
   */
  {
    QGauss<dim - 1>   quad(n_q_points_1d);
    const UpdateFlags update_flags =
      update_values | update_normal_vectors | update_quadrature_points | update_JxW_values;

    FEFaceValues<dim> fe_values_v(mapping, fe_v, quad, update_flags);

    const auto face_node_functional = [&](const unsigned int face_no, const Vector<double> v) {
      const auto & unit_cell = unit_dofh_v.begin_active(); // one reference cell
      fe_values_v.reinit(unit_cell, face_no);

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
        const double           orientation = GeometryInfo<dim>::unit_normal_orientation[face_no];
        const Tensor<1, dim> & n_q         = orientation * normals[q];
        const Tensor<1, dim> & v_q         = compute_v_q(q);
        const auto &           dx          = fe_values_v.JxW(q);

        node_value += v_q * n_q * dx;
      }

      return node_value;
    };

    LAPACKFullMatrix<double> node_value_weights(n_faces_per_cell, n_dofs_per_cell_v);
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
      for(auto j = 0U; j < n_dofs_per_cell_v; ++j)
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
                           mapping);
    }
  }

  return shape_function_weights;
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::solve_pressure()
{
  AssertThrow(false, ExcMessage("TODO MPI..."));
  // print_parameter("Solving pressure system", "...");

  // AssertThrow(stokes_problem, ExcMessage("FEM for Stokes equations is uninitialized."));

  // // TODO !!! share triangulations as shared_ptr
  // stokes_problem->triangulation.clear();
  // stokes_problem->triangulation.copy_triangulation(this->triangulation);
  // stokes_problem->setup_system();

  // const auto & dof_handler_velocity = stokes_problem->dof_handler_velocity;
  // const auto & dof_handler_pressure = stokes_problem->dof_handler_pressure;
  // const bool   is_dgq_legendre =
  //   dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
  // AssertThrow(is_dgq_legendre,
  //             ExcMessage("Implementation is based on a DGQ pressure ansatz of Legendre-type"));

  // const auto [trafomatrix_rt_to_gradp, trafomatrix_rt_to_constp] =
  //   compute_nondivfree_shape_functions();

  // FullMatrix<double> shape_to_test_functions_interior(trafomatrix_rt_to_gradp.m(),
  //                                                     trafomatrix_rt_to_gradp.n());
  // shape_to_test_functions_interior = trafomatrix_rt_to_gradp;

  // // DEBUG
  // // shape_to_test_functions_interior.print_formatted(std::cout);
  // // trafomatrix_rt_to_gradp.print_formatted(std::cout);

  // FullMatrix<double> shape_to_test_functions_interface(trafomatrix_rt_to_constp.m(),
  //                                                      trafomatrix_rt_to_constp.n());
  // shape_to_test_functions_interface = trafomatrix_rt_to_constp;

  // // DEBUG
  // // shape_to_test_functions_interface.print_formatted(std::cout);
  // // trafomatrix_rt_to_constp.print_formatted(std::cout);

  // Pressure::InterfaceHandler<dim> interface_handler;
  // interface_handler.reinit(dof_handler_velocity);
  // const auto n_interface_nodes = interface_handler.n_interfaces();

  // AssertThrow(
  //   interface_handler.get_fixed_cell_index() == interface_handler.get_fixed_interface_index(),
  //   ExcMessage(
  //     "I am worried about the constraints in case the fixed cell and interface index do not
  //     coincide."));

  // AffineConstraints constraints_on_interface;
  // const auto        interface_index_of_fixed_cell =
  // interface_handler.get_fixed_interface_index();
  // constraints_on_interface.add_line(interface_index_of_fixed_cell);
  // constraints_on_interface.set_inhomogeneity(interface_index_of_fixed_cell, 1.);
  // constraints_on_interface.close();

  // AffineConstraints constraints_on_cell;
  // constraints_on_cell.close();

  // DynamicSparsityPattern dsp(n_interface_nodes);
  // for(const auto & id : interface_handler.interface_ids)
  // {
  //   const auto e                 = interface_handler.get_interface_index(id);
  //   const auto [K_left, K_right] = interface_handler.get_cell_index_pair(id);
  //   dsp.add(e, K_left);
  //   dsp.add(e, K_right);

  //   // DEBUG
  //   // std::cout << "interface index (row): " << interface_handler.get_interface_index(id) << "
  //   ";
  //   // const auto [left_index, right_index] = interface_handler.get_cell_index_pair(id);
  //   // std::cout << "left cell index (column): " << left_index << " ";
  //   // std::cout << "right cell index (column): " << right_index << " ";
  //   // std::cout << std::endl;
  // }

  // constraints_on_interface.condense(dsp);

  // SparsityPattern sparsity_pattern;
  // sparsity_pattern.copy_from(dsp);

  // SparseMatrix<double> constant_pressure_matrix;
  // constant_pressure_matrix.reinit(sparsity_pattern);
  // Vector<double> right_hand_side(n_interface_nodes);

  // std::vector<types::global_dof_index> constant_pressure_dof_indices(n_interface_nodes);
  // Vector<double> &                     discrete_pressure =
  // stokes_problem->system_solution.block(1);

  // const auto n_q_points_1d = stokes_problem->n_q_points_1d;

  // /**
  //  * First, we compute all pressure coefficients except the constant mode
  //  * coefficient. To this end, we construct a basis of cell-wise non-div-free
  //  * test functions vv_j associated to the interior RT node functionals N_i as follows:
  //  * as node functional generating polynomials from Q_k-1,k x Q_k,k-1 we
  //  * substitute grad p_i, where p_i are the pressure shape functions, such that
  //  *
  //  *    N_i(vv_j) = \int_K vv_j * (grad p_i) dx   =!=   \delta_ij   (1)
  //  *
  //  * We see that the gradient of the constant pressure mode is zero and, thus,
  //  * not contributing to the basis of non-div-free test functions. The new basis
  //  * functions vv_j are a linear combination of the RT basis v_k given, that is
  //  * vv_j = \sum_k \alpha_jk v_k. Equation (1) is converted one-to-one to the
  //  * reference cell and reference functions, thus, it suffices to compute the
  //  * transformation matrix A = (\alpha_jk)_jk, here @p
  //  * shape_to_test_functions_interior, on the reference cell.
  //  *
  //  * Why does this determine the non-constant pressure modes? Given the stream
  //  * function velocity u_h the discretization of the first Stokes equation reads
  //  *
  //  *    a_h(u_h, v) - (p, div v) = (f, v) + BDRY   (2)
  //  *
  //  * Integration by parts of the pressure term, (p, vv_j * n)_\face = 0 \forall
  //  * vv_j and substituting \beta_j * p_j as pressure (\beta_j is the pressure
  //  * coefficient) results in
  //  *
  //  *    -\beta_j * (grad p_j, vv_i) = (f, vv_i) + BDRY_i - a_h(u_h, vv_i),   (3)
  //  *
  //  * where the LHS (grad p_j, vv_i) = N_j(vv_i) = \delta_ij by definition of
  //  * vv_i, thus, the non-constant pressure coefficients are computed as follows
  //  *
  //  *    -\beta_i = (f, vv_i) + BDRY_i - a_h(u_h, vv_i).   (4)
  //  *
  //  * The RHS is the residual of the discretized Stokes equations for the
  //  * discrete stream function velocity u_h.
  //  */
  // {
  //   const bool is_dgq_legendre =
  //     dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
  //   AssertThrow(
  //     is_dgq_legendre,
  //     ExcMessage(
  //       "For this reconstruction method we assume that the pressure shape functions are of
  //       Legendre-type."));
  //   AssertThrow(
  //     TPSS::get_dof_layout(dof_handler_velocity.get_fe()) == TPSS::DoFLayout::RT,
  //     ExcMessage(
  //       "For this reconstruction method we assume that the velocity finite elements are of
  //       Raviart-Thomas-type."));

  //   using Stokes::Velocity::SIPG::MW::ScratchData;

  //   using Stokes::Velocity::SIPG::MW::CopyData;

  //   using Stokes::Velocity::SIPG::MW::MatrixIntegrator;

  //   using CellIterator = typename MatrixIntegrator<dim>::IteratorType;

  //   const auto                     component_range = std::make_pair<unsigned int>(0, dim);
  //   Stokes::FunctionExtractor<dim> load_function_velocity(stokes_problem->load_function.get(),
  //                                                         component_range);
  //   Stokes::FunctionExtractor<dim> analytical_velocity(stokes_problem->analytical_solution.get(),
  //                                                      component_range);

  //   MatrixIntegrator<dim> matrix_integrator(&load_function_velocity,
  //                                           &analytical_velocity,
  //                                           &system_u,
  //                                           equation_data_stokes,
  //                                           &interface_handler);

  //   auto cell_worker =
  //     [&](const CellIterator & cell, ScratchData<dim, true> & scratch_data, CopyData & copy_data)
  //     {
  //       CellIterator cell_stream_function(&dof_handler.get_triangulation(),
  //                                         cell->level(),
  //                                         cell->index(),
  //                                         &dof_handler);
  //       CellIterator cell_pressure(&dof_handler_pressure.get_triangulation(),
  //                                  cell->level(),
  //                                  cell->index(),
  //                                  &dof_handler_pressure);
  //       matrix_integrator.cell_residual_worker(
  //         cell, cell_stream_function, cell_pressure, scratch_data, copy_data);
  //     };

  //   const auto face_worker = [&](const CellIterator &     cell,
  //                                const unsigned int &     f,
  //                                const unsigned int &     sf,
  //                                const CellIterator &     ncell,
  //                                const unsigned int &     nf,
  //                                const unsigned int &     nsf,
  //                                ScratchData<dim, true> & scratch_data,
  //                                CopyData &               copy_data) {
  //     CellIterator cell_stream(&dof_handler.get_triangulation(),
  //                              cell->level(),
  //                              cell->index(),
  //                              &dof_handler);
  //     CellIterator cell_pressure(&dof_handler_pressure.get_triangulation(),
  //                                cell->level(),
  //                                cell->index(),
  //                                &dof_handler_pressure);
  //     CellIterator ncell_stream(&dof_handler.get_triangulation(),
  //                               ncell->level(),
  //                               ncell->index(),
  //                               &dof_handler);
  //     CellIterator ncell_pressure(&dof_handler_pressure.get_triangulation(),
  //                                 ncell->level(),
  //                                 ncell->index(),
  //                                 &dof_handler_pressure);
  //     matrix_integrator.face_residual_worker_tangential(cell,
  //                                                       cell_stream,
  //                                                       cell_pressure,
  //                                                       f,
  //                                                       sf,
  //                                                       ncell,
  //                                                       ncell_stream,
  //                                                       ncell_pressure,
  //                                                       nf,
  //                                                       nsf,
  //                                                       scratch_data,
  //                                                       copy_data);
  //   };

  //   const auto boundary_worker = [&](const CellIterator &     cell,
  //                                    const unsigned int &     face_no,
  //                                    ScratchData<dim, true> & scratch_data,
  //                                    CopyData &               copy_data) {
  //     CellIterator cell_stream(&dof_handler.get_triangulation(),
  //                              cell->level(),
  //                              cell->index(),
  //                              &dof_handler);
  //     CellIterator cell_pressure(&dof_handler_pressure.get_triangulation(),
  //                                cell->level(),
  //                                cell->index(),
  //                                &dof_handler_pressure);
  //     matrix_integrator.boundary_residual_worker_tangential(
  //       cell, cell_stream, cell_pressure, face_no, scratch_data, copy_data);
  //   };

  //   AffineConstraints<double> empty_constraints;
  //   empty_constraints.close();

  //   const auto copier = [&](const CopyData & copy_data) {
  //     empty_constraints.template distribute_local_to_global<Vector<double>>(
  //       copy_data.cell_rhs_test, copy_data.local_dof_indices_test, discrete_pressure);

  //     /// Book-keeping the (global) dof indices of each constant mode per cell
  //     AssertDimension(copy_data.local_dof_indices_ansatz.size(), 2U);
  //     const auto cell_index                     = copy_data.local_dof_indices_ansatz.back();
  //     const auto dof_index                      = copy_data.local_dof_indices_ansatz.front();
  //     constant_pressure_dof_indices[cell_index] = dof_index;

  //     for(const auto & cdf : copy_data.face_data)
  //       empty_constraints.template distribute_local_to_global<Vector<double>>(
  //         cdf.cell_rhs_test, cdf.joint_dof_indices_test, discrete_pressure);
  //   };

  //   const UpdateFlags update_flags_v =
  //     update_values | update_gradients | update_quadrature_points | update_JxW_values;
  //   const UpdateFlags update_flags_sf          = update_flags_v | update_hessians;
  //   const UpdateFlags interface_update_flags_v = update_values | update_gradients |
  //                                                update_quadrature_points | update_JxW_values |
  //                                                update_normal_vectors;
  //   const UpdateFlags interface_update_flags_sf = interface_update_flags_v | update_hessians;

  //   ScratchData<dim, true> scratch_data(mapping,
  //                                       dof_handler_velocity.get_fe(),
  //                                       dof_handler.get_fe(),
  //                                       n_q_points_1d,
  //                                       shape_to_test_functions_interior,
  //                                       update_flags_v,
  //                                       update_flags_sf,
  //                                       interface_update_flags_v,
  //                                       interface_update_flags_sf);

  //   CopyData copy_data(shape_to_test_functions_interior.m(), dof_handler.get_fe().dofs_per_cell);

  //   MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
  //                         dof_handler_velocity.end(),
  //                         cell_worker,
  //                         copier,
  //                         scratch_data,
  //                         copy_data,
  //                         MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
  //                           MeshWorker::assemble_own_interior_faces_once,
  //                         boundary_worker,
  //                         face_worker);

  //   // // DEBUG
  //   // discrete_pressure.print(std::cout);
  //   // std::cout << vector_to_string(constant_pressure_dof_indices) << std::endl;
  // }

  // /**
  //  * Second, it remains to determine the constant pressure mode on each cell to
  //  * globally determine the discrete pressure. As discussed before, the constant
  //  * pressure modes do not contribute to the cell-interior non-div-free velocity
  //  * functions, thus, they are determined by the degrees of freedom on
  //  * interfaces.
  //  *
  //  * The (#cells - 1) remaining test functions are constructed by the method
  //  * described in [Caussignac'87]... TODO
  //  */
  // {
  //   {
  //     using Stokes::Velocity::SIPG::MW::ScratchData;

  //     using Stokes::Velocity::SIPG::MW::CopyData;

  //     using Stokes::Velocity::SIPG::MW::MatrixIntegrator;

  //     using CellIterator = typename MatrixIntegrator<dim>::IteratorType;

  //     const auto                     component_range = std::make_pair<unsigned int>(0, dim);
  //     Stokes::FunctionExtractor<dim> load_function_velocity(stokes_problem->load_function.get(),
  //                                                           component_range);
  //     Stokes::FunctionExtractor<dim>
  //     analytical_velocity(stokes_problem->analytical_solution.get(),
  //                                                        component_range);

  //     MatrixIntegrator<dim> matrix_integrator(&load_function_velocity,
  //                                             &analytical_velocity,
  //                                             &system_u,
  //                                             equation_data_stokes,
  //                                             &interface_handler);

  //     const auto cell_worker = [&](const CellIterator &     cell,
  //                                  ScratchData<dim, true> & scratch_data,
  //                                  CopyData &               copy_data) {
  //       CellIterator cell_stream_function(&dof_handler.get_triangulation(),
  //                                         cell->level(),
  //                                         cell->index(),
  //                                         &dof_handler);
  //       matrix_integrator.cell_residual_worker_interface(cell,
  //                                                        cell_stream_function,
  //                                                        scratch_data,
  //                                                        copy_data);
  //     };

  //     const auto face_worker = [&](const CellIterator &     cell,
  //                                  const unsigned int &     f,
  //                                  const unsigned int &     sf,
  //                                  const CellIterator &     ncell,
  //                                  const unsigned int &     nf,
  //                                  const unsigned int &     nsf,
  //                                  ScratchData<dim, true> & scratch_data,
  //                                  CopyData &               copy_data) {
  //       CellIterator cell_stream(&dof_handler.get_triangulation(),
  //                                cell->level(),
  //                                cell->index(),
  //                                &dof_handler);
  //       CellIterator ncell_stream(&dof_handler.get_triangulation(),
  //                                 ncell->level(),
  //                                 ncell->index(),
  //                                 &dof_handler);
  //       matrix_integrator.face_residual_worker_tangential_interface(
  //         cell, cell_stream, f, sf, ncell, ncell_stream, nf, nsf, scratch_data, copy_data);
  //     };

  //     const auto boundary_worker = [&](const CellIterator &     cell,
  //                                      const unsigned int &     face_no,
  //                                      ScratchData<dim, true> & scratch_data,
  //                                      CopyData &               copy_data) {
  //       CellIterator cell_stream(&dof_handler.get_triangulation(),
  //                                cell->level(),
  //                                cell->index(),
  //                                &dof_handler);
  //       matrix_integrator.boundary_residual_worker_tangential_interface(
  //         cell, cell_stream, face_no, scratch_data, copy_data);
  //     };

  //     const auto copier = [&](const CopyData & copy_data) {
  //       constraints_on_interface.template distribute_local_to_global<Vector<double>>(
  //         copy_data.cell_rhs_test, copy_data.local_dof_indices_test, right_hand_side);

  //       for(const auto & cdf : copy_data.face_data)
  //       {
  //         constraints_on_interface.template distribute_local_to_global<Vector<double>>(
  //           cdf.cell_rhs_test, cdf.joint_dof_indices_test, right_hand_side);
  //       }
  //     };

  //     const UpdateFlags update_flags_v =
  //       update_values | update_gradients | update_quadrature_points | update_JxW_values;
  //     const UpdateFlags update_flags_sf          = update_flags_v | update_hessians;
  //     const UpdateFlags interface_update_flags_v = update_values | update_gradients |
  //                                                  update_quadrature_points | update_JxW_values |
  //                                                  update_normal_vectors;
  //     const UpdateFlags interface_update_flags_sf = interface_update_flags_v | update_hessians;

  //     ScratchData<dim, true> scratch_data(mapping,
  //                                         dof_handler_velocity.get_fe(),
  //                                         dof_handler.get_fe(),
  //                                         n_q_points_1d,
  //                                         shape_to_test_functions_interface,
  //                                         update_flags_v,
  //                                         update_flags_sf,
  //                                         interface_update_flags_v,
  //                                         interface_update_flags_sf);

  //     CopyData copy_data(shape_to_test_functions_interface.m(),
  //     dof_handler.get_fe().dofs_per_cell);

  //     MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
  //                           dof_handler_velocity.end(),
  //                           cell_worker,
  //                           copier,
  //                           scratch_data,
  //                           copy_data,
  //                           MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
  //                             MeshWorker::assemble_own_interior_faces_once,
  //                           boundary_worker,
  //                           face_worker);
  //   }

  //   {
  //     using Pressure::Interface::MW::ScratchData;

  //     using Pressure::Interface::MW::CopyData;

  //     using Pressure::Interface::MW::MatrixIntegrator;

  //     using CellIterator = typename MatrixIntegrator<dim>::IteratorType;

  //     const auto                     component_range = std::make_pair<unsigned int>(0, dim);
  //     Stokes::FunctionExtractor<dim> load_function_velocity(stokes_problem->load_function.get(),
  //                                                           component_range);

  //     MatrixIntegrator<dim> matrix_integrator(&load_function_velocity,
  //                                             nullptr,
  //                                             &discrete_pressure,
  //                                             &interface_handler,
  //                                             equation_data_stokes);

  //     const auto face_worker = [&](const CellIterator & cell,
  //                                  const unsigned int & f,
  //                                  const unsigned int & sf,
  //                                  const CellIterator & ncell,
  //                                  const unsigned int & nf,
  //                                  const unsigned int & nsf,
  //                                  ScratchData<dim> &   scratch_data,
  //                                  CopyData &           copy_data) {
  //       CellIterator cellP(&dof_handler_pressure.get_triangulation(),
  //                          cell->level(),
  //                          cell->index(),
  //                          &dof_handler_pressure);
  //       CellIterator ncellP(&dof_handler_pressure.get_triangulation(),
  //                           ncell->level(),
  //                           ncell->index(),
  //                           &dof_handler_pressure);
  //       matrix_integrator.face_worker(
  //         cell, cellP, f, sf, ncell, ncellP, nf, nsf, scratch_data, copy_data);
  //     };

  //     const auto copier = [&](const CopyData & copy_data) {
  //       for(const auto & cdf : copy_data.face_data)
  //       {
  //         constraints_on_interface.template distribute_local_to_global<Vector<double>>(
  //           cdf.cell_rhs_test, cdf.joint_dof_indices_test, right_hand_side);

  //         constraints_on_interface.template distribute_local_to_global<SparseMatrix<double>>(
  //           cdf.cell_matrix,
  //           cdf.joint_dof_indices_test,
  //           constraints_on_cell,
  //           cdf.joint_dof_indices_ansatz,
  //           constant_pressure_matrix);
  //       }
  //     };

  //     const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
  //     const UpdateFlags update_flags_pressure = update_default;
  //     const UpdateFlags interface_update_flags =
  //       update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
  //     const UpdateFlags interface_update_flags_pressure =
  //       update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

  //     ScratchData<dim> scratch_data(mapping,
  //                                   dof_handler_velocity.get_fe(),
  //                                   dof_handler_pressure.get_fe(),
  //                                   n_q_points_1d,
  //                                   shape_to_test_functions_interface,
  //                                   update_flags,
  //                                   update_flags_pressure,
  //                                   interface_update_flags,
  //                                   interface_update_flags_pressure);

  //     CopyData copy_data(GeometryInfo<dim>::faces_per_cell, 1U);

  //     MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
  //                           dof_handler_velocity.end(),
  //                           nullptr /*cell_worker*/,
  //                           copier,
  //                           scratch_data,
  //                           copy_data,
  //                           /*MeshWorker::assemble_own_cells |*/
  //                           MeshWorker::assemble_own_interior_faces_once,
  //                           nullptr,
  //                           face_worker);
  //   }

  //   constraints_on_interface.condense(constant_pressure_matrix, right_hand_side);

  //   // // DEBUG
  //   // right_hand_side.print(std::cout);
  //   // constant_pressure_matrix.print_formatted(std::cout);

  //   const auto     n_cells = constant_pressure_matrix.n();
  //   Vector<double> constant_mode_solution(n_cells);

  //   SparseDirectUMFPACK A_direct;
  //   A_direct.template initialize<SparseMatrix<double>>(constant_pressure_matrix);
  //   A_direct.vmult(constant_mode_solution, right_hand_side);

  //   constraints_on_interface.distribute(constant_mode_solution);

  //   for(auto cell_index = 0U; cell_index < n_cells; ++cell_index)
  //   {
  //     const auto dof_index         = constant_pressure_dof_indices[cell_index];
  //     discrete_pressure(dof_index) = constant_mode_solution[cell_index];
  //   }

  //   stokes_problem->correct_mean_value_pressure();
  // }
}



template<int dim, int fe_degree>
double
ModelProblem<dim, fe_degree>::compute_stream_function_error()
{
  AssertThrow(false, ExcMessage("TODO MPI..."));
  // AssertThrow(analytical_velocity, ExcMessage("analytical_velocity isn't initialized."));
  // AssertDimension(analytical_velocity->n_components, dim);

  // using ::MW::ScratchData;

  // using ::MW::CopyData;

  // using ::MW::compute_vcurl;

  // AffineConstraints empty_constraints;
  // empty_constraints.close();

  // Vector<double> norm_per_cell(triangulation.n_active_cells());

  // auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData &
  // copy_data) {
  //   auto & phi = scratch_data.fe_values;
  //   phi.reinit(cell);

  //   const unsigned int n_dofs_per_cell = phi.get_fe().dofs_per_cell;
  //   const unsigned int n_q_points      = phi.n_quadrature_points;
  //   const auto &       q_points        = phi.get_quadrature_points();

  //   std::vector<Tensor<1, dim>> velocity_values;
  //   std::transform(q_points.cbegin(),
  //                  q_points.cend(),
  //                  std::back_inserter(velocity_values),
  //                  [this](const auto & x_q) {
  //                    Tensor<1, dim> u_q;
  //                    for(auto c = 0U; c < dim; ++c)
  //                      u_q[c] = analytical_velocity->value(x_q, c);
  //                    return u_q;
  //                  });

  //   std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
  //   cell->get_active_or_mg_dof_indices(local_dof_indices);
  //   std::vector<double> stream_function_dof_values;
  //   std::transform(local_dof_indices.cbegin(),
  //                  local_dof_indices.cend(),
  //                  std::back_inserter(stream_function_dof_values),
  //                  [this](const auto & i) { return system_u(i); });

  //   double local_error = 0.;
  //   for(unsigned int q = 0; q < n_q_points; ++q)
  //   {
  //     Tensor<1, dim> uh_q;
  //     for(unsigned int i = 0; i < n_dofs_per_cell; ++i)
  //     {
  //       const auto & alpha_i    = stream_function_dof_values[i];
  //       const auto & curl_phi_i = compute_vcurl(phi, i, q);
  //       uh_q += alpha_i * curl_phi_i;
  //     }

  //     const auto & u_q = velocity_values[q];
  //     local_error += (uh_q - u_q) * (uh_q - u_q) * phi.JxW(q);
  //   }

  //   AssertDimension(copy_data.cell_rhs.size(), 1U);
  //   AssertDimension(copy_data.local_dof_indices.size(), 1U);
  //   AssertIndexRange(cell->index(), norm_per_cell.size());

  //   copy_data.local_dof_indices[0] = cell->index();
  //   copy_data.cell_rhs(0)          = std::sqrt(local_error);
  // };

  // const auto copier = [&](const CopyData & copy_data) {
  //   /// We first store cell-wise errors to avoid data races in the mesh_loop()
  //   /// call and then accumulate the global error, instead of directly copying
  //   /// all local errors to one global error field.
  //   AssertDimension(copy_data.cell_rhs.size(), 1U);
  //   AssertDimension(copy_data.local_dof_indices.size(), 1U);
  //   const auto cell_index     = copy_data.local_dof_indices[0];
  //   norm_per_cell(cell_index) = copy_data.cell_rhs(0);
  // };

  // const unsigned int n_gauss_points = dof_handler.get_fe().degree + 2;
  // const UpdateFlags  update_flags =
  //   update_values | update_gradients | update_quadrature_points | update_JxW_values;
  // const UpdateFlags interface_update_flags  = update_default;
  // const auto        n_error_values_per_cell = 1U;

  // ScratchData<dim> scratch_data(
  //   mapping, dof_handler.get_fe(), n_gauss_points, update_flags, interface_update_flags);

  // CopyData copy_data(n_error_values_per_cell);

  // MeshWorker::mesh_loop(dof_handler.begin_active(),
  //                       dof_handler.end(),
  //                       cell_worker,
  //                       copier,
  //                       scratch_data,
  //                       copy_data,
  //                       MeshWorker::assemble_own_cells);

  // return VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::L2_norm);
}



template<int dim, int fe_degree>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree>::compute_L2_error_pressure() const
{
  AssertThrow(false, ExcMessage("TODO MPI..."));
  // AssertThrow(stokes_problem, ExcMessage("stokes_problem is not initialized"));

  // const auto component_range_pressure = std::make_pair<unsigned int>(dim, dim + 1);
  // Stokes::FunctionExtractor<dim> analytical_solution_pressure(
  //   stokes_problem->analytical_solution.get(), component_range_pressure);
  // const auto & dof_handler_p     = stokes_problem->dof_handler_pressure;
  // const auto & discrete_pressure = stokes_problem->system_solution.block(1);

  // const auto difference_per_cell =
  // std::make_shared<Vector<double>>(triangulation.n_active_cells());
  // VectorTools::integrate_difference(dof_handler_p,
  //                                   discrete_pressure,
  //                                   analytical_solution_pressure,
  //                                   *difference_per_cell,
  //                                   QGauss<dim>(stokes_problem->n_q_points_1d + 2),
  //                                   VectorTools::L2_norm);
  // return difference_per_cell;
}



template<int dim, int fe_degree>
double
ModelProblem<dim, fe_degree>::compute_energy_error() const
{
  using ::MW::ScratchData;

  using ::MW::Cell::CopyData;

  Vector<double> error_per_cell(triangulation.n_active_cells());

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    auto & cell_data          = copy_data.cell_data.emplace_back(1U);
    cell_data.cell_indices[0] = cell->active_cell_index();

    FEValuesExtractors::Scalar scalar(0);
    auto &                     fe_values = scratch_data.fe_values;
    fe_values.reinit(cell);

    std::vector<Tensor<2, dim>> hessians(fe_values.n_quadrature_points);
    fe_values[scalar].get_function_hessians(system_u, hessians);

    std::vector<SymmetricTensor<2, dim>> exact_hessians(fe_values.n_quadrature_points);
    analytical_solution->hessian_list(fe_values.get_quadrature_points(), exact_hessians);

    double local_error = 0;
    for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    {
      const auto & hess_uh = hessians[q];
      const auto & hess_u  = exact_hessians[q];

      local_error += (hess_u - hess_uh).norm_square() * fe_values.JxW(q);
    }

    cell_data.values = local_error;
  };

  auto face_worker = [&](const auto &         cell,
                         const unsigned int & face_no,
                         const unsigned int & sface_no,
                         const auto &         ncell,
                         const unsigned int & nface_no,
                         const unsigned int & nsface_no,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) {
    auto & face_data          = copy_data.face_data.emplace_back(2U);
    face_data.cell_indices[0] = cell->active_cell_index();
    face_data.cell_indices[1] = ncell->active_cell_index();

    FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
    fe_interface_values.reinit(cell, face_no, sface_no, ncell, nface_no, nsface_no);

    const unsigned int  n_interface_dofs  = fe_interface_values.n_current_interface_dofs();
    const auto &        joint_dof_indices = fe_interface_values.get_interface_dof_indices();
    std::vector<double> dof_values;
    std::transform(joint_dof_indices.cbegin(),
                   joint_dof_indices.cend(),
                   std::back_inserter(dof_values),
                   [&](const auto dof_index) { return system_u(dof_index); });

    const auto h  = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
    const auto nh = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nface_no]);
    const double gamma_over_h =
      0.5 * equation_data.ip_factor * C0IP::compute_penalty_impl(fe_degree, h, nh);

    const auto quadrature_points = fe_interface_values.get_quadrature_points();

    double local_error = 0.;
    for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
    {
      const auto & n = fe_interface_values.normal(q);

      double jump_grad_uh_dot_n = 0.;
      for(unsigned int i = 0; i < n_interface_dofs; ++i)
        jump_grad_uh_dot_n += dof_values[i] * fe_interface_values.jump_gradient(i, q) * n;

      /// assuming u is smooth the jump of its gradient is zero
      local_error += jump_grad_uh_dot_n * jump_grad_uh_dot_n * fe_interface_values.JxW(q);
    }
    local_error *= gamma_over_h;

    /// split the error at the interface even among cells
    face_data.values[0] = 0.5 * local_error;
    face_data.values[1] = 0.5 * local_error;
  };

  auto boundary_worker = [&](const auto &         cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
    auto & face_data          = copy_data.face_data.emplace_back(2U);
    face_data.cell_indices[0] = cell->active_cell_index();
    face_data.cell_indices[1] = cell->active_cell_index();

    FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
    fe_interface_values.reinit(cell, face_no);

    const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();

    const auto &        joint_dof_indices = fe_interface_values.get_interface_dof_indices();
    std::vector<double> dof_values;
    std::transform(joint_dof_indices.cbegin(),
                   joint_dof_indices.cend(),
                   std::back_inserter(dof_values),
                   [&](const auto dof_index) { return system_u(dof_index); });

    const auto   h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
    const double gamma_over_h =
      equation_data.ip_factor * C0IP::compute_penalty_impl(fe_degree, h, h);

    const auto quadrature_points = fe_interface_values.get_quadrature_points();

    double local_error = 0.;
    for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
    {
      const auto & n            = fe_interface_values.normal(q);
      const auto & x_q          = quadrature_points[q];
      const double grad_u_dot_n = analytical_solution->gradient(x_q) * n;

      double grad_uh_dot_n = 0.;
      for(unsigned int i = 0; i < n_interface_dofs; ++i)
        grad_uh_dot_n += dof_values[i] * fe_interface_values.jump_gradient(i, q) * n;

      local_error += (grad_uh_dot_n - grad_u_dot_n) * (grad_uh_dot_n - grad_u_dot_n) *
                     fe_interface_values.JxW(q);
    }
    local_error *= gamma_over_h;

    face_data.values[0] = 0.5 * local_error;
    face_data.values[1] = 0.5 * local_error;
  };

  const auto copier = [&](const CopyData & copy_data) {
    for(const auto & cell_data : copy_data.cell_data)
      error_per_cell(cell_data.cell_indices[0]) += cell_data.values(0);

    for(const auto & cdf : copy_data.face_data)
    {
      error_per_cell(cdf.cell_indices[0]) += cdf.values(0);
      error_per_cell(cdf.cell_indices[1]) += cdf.values(1);
    }
  };

  ScratchData<dim> scratch_data(
    mapping, dof_handler.get_fe(), n_q_points_1d + 1, update_flags, update_flags_interface);

  CopyData copy_data;

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once |
                          MeshWorker::assemble_ghost_faces_once,
                        boundary_worker,
                        face_worker);

  const double locally_owned_error_square = error_per_cell.l1_norm();

  const double global_error =
    std::sqrt(Utilities::MPI::sum(locally_owned_error_square, MPI_COMM_WORLD));

  return global_error;
}


// The next function evaluates the error between the computed solution
// and the exact solution (which is known here because we have chosen
// the right hand side and boundary values in a way so that we know
// the corresponding solution). In the first two code blocks below,
// we compute the error in the $L_2$ norm and the $H^1$ semi-norm.
template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::compute_discretization_errors()
{
  if(equation_data.is_stream_function())
  {
    /// Velocity - L2
    {
      AssertThrow(stokes_problem, ExcMessage("stokes_problem isnt initialized"));
      const double l2_velocity_error = compute_stream_function_error();
      print_parameter("L2 velocity error (stream function):", l2_velocity_error);
      stokes_problem->pp_data.L2_error.push_back(l2_velocity_error);
    }

    /// Pressure - L2
    {
      const auto   error_per_cell = compute_L2_error_pressure();
      const double l2_pressure_error =
        VectorTools::compute_global_error(triangulation, *error_per_cell, VectorTools::L2_norm);
      print_parameter("L2 pressure error (stream function):", l2_pressure_error);
      stokes_problem->pp_data_pressure.L2_error.push_back(l2_pressure_error);
    }
  }

  {
    Vector<double> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      system_u,
                                      *analytical_solution,
                                      norm_per_cell,
                                      QGauss<dim>(finite_element->degree + 2),
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
                                      QGauss<dim>(finite_element->degree + 2),
                                      VectorTools::H1_seminorm);
    const double error_norm =
      VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::H1_seminorm);
    print_parameter("Error in the H1 seminorm:", error_norm);
    pp_data.H1semi_error.push_back(error_norm);
  }

  {
    const double error_norm = compute_energy_error();
    print_parameter("Error in energy norm (resp. |.|_h):", error_norm);
    pp_data.H2semi_error.push_back(error_norm);
  }
}



template<int dim, int fe_degree>
void
ModelProblem<dim, fe_degree>::output_results(const unsigned int iteration) const
{
  print_parameter("Writing graphical output", "...");

  const std::string filename = "biharm_" + equation_data.sstr_equation_variant() + "_" +
                               Utilities::int_to_string(iteration, 3) + ".vtk";
  std::ofstream output_vtk(filename);

  StreamVelocityPP<dim> stream_velocity_pp;
  PressurePP<dim>       pressure_pp;
  DataOut<dim>          data_out;

  // TODO requires ParaView 5.5.0
  // DataOutBase::VtkFlags flags;
  // flags.write_higher_order_cells = true;
  // data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(system_u, "solution");
  if(equation_data.is_stream_function())
  {
    data_out.add_data_vector(system_u, stream_velocity_pp);

    Assert(stokes_problem, ExcMessage("stokes_problem isnt initialized"));
    const auto & dof_handler_p     = stokes_problem->dof_handler_pressure;
    const auto & discrete_pressure = stokes_problem->system_solution.block(1);
    data_out.add_data_vector(dof_handler_p, discrete_pressure, pressure_pp);
  }

  data_out.build_patches(fe_degree);

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
    {
      std::ostringstream oss;
      oss << "Starting cycle " << cycle + 1 << " of " << n_cycles;
      print_parameter(oss.str(), "...");
    }

    const unsigned int n_refinements = rt_parameters.mesh.n_refinements + cycle;
    if(!make_grid(n_refinements))
    {
      print_parameter("No mesh created", "...");
      *pcout << std::endl << std::endl;
      continue;
    }

    setup_system();

    assemble_system();

    solve();

    if(equation_data.is_stream_function())
    {
      solve_pressure();
      // stokes_problem->output_results(cycle);
    }

    // output_results(cycle);

    compute_discretization_errors();

    *pcout << std::endl;
  }
}



} // namespace Biharmonic

#endif // APPS_BIHARMONICPROBLEM_H_
