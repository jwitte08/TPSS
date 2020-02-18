/**
 * poisson.h
 *
 * standard FEM poisson problem
 *
 *  Created on: Feb 03, 2020
 *      Author: witte
 */

#ifndef POISSONPROBLEM_H_
#define POISSONPROBLEM_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_selector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/matrix_helper.h"
#include "solvers_and_preconditioners/preconditioner/schwarz_preconditioner.h"

#include "coloring.h"
#include "equation_data.h"
#include "laplace_integrator.h"
#include "mesh.h"
#include "postprocess.h"
#include "rt_parameter.h"
#include "utilities.h"
#include "vectorization_helper.h"

using namespace dealii;

namespace Poisson
{
namespace CFEM
{
template<int dim, int fe_degree, typename Number = double, int n_patch_dofs = -1>
struct ModelProblem : public Subscriptor
{
  static constexpr unsigned int fe_order  = fe_degree + 1;
  static constexpr unsigned int n_qpoints = fe_degree + 1;

  using value_type    = Number;
  using VECTOR        = typename LinearAlgebra::distributed::Vector<Number>;
  using SYSTEM_MATRIX = Laplace::CFEM::MF::Operator<dim, fe_degree, Number>;

  using value_type_mg = Number;
  using LEVEL_MATRIX  = Laplace::CFEM::CombinedOperator<dim, fe_degree, value_type_mg>;
  using MG_TRANSFER   = MGTransferMatrixFree<dim, value_type_mg>;
  using TP_MATRIX     = TensorProductMatrixSymmetricSum<dim, VectorizedArray<Number>, n_patch_dofs>;
  using PATCH_MATRIX  = ConstrainedMatrix<TP_MATRIX>;
  using SCHWARZ_PRECONDITIONER = SchwarzPreconditioner<dim, LEVEL_MATRIX, VECTOR, PATCH_MATRIX>;
  using SCHWARZ_SMOOTHER       = SchwarzSmoother<dim, LEVEL_MATRIX, SCHWARZ_PRECONDITIONER, VECTOR>;
  using MG_SMOOTHER_SCHWARZ    = MGSmootherSchwarz<dim, LEVEL_MATRIX, PATCH_MATRIX, VECTOR>;
  using GMG_PRECONDITIONER     = PreconditionMG<dim, VECTOR, MG_TRANSFER>;

  // *** parameters and auxiliary structs
  RT::Parameter                               rt_parameters;
  mutable std::shared_ptr<ConditionalOStream> pcout;
  mutable PostProcessData                     pp_data;

  // *** FEM fundamentals
  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim>                           dof_handler;
  std::shared_ptr<FiniteElement<dim>>       fe;
  const MappingQGeneric<dim>                mapping;
  AffineConstraints<Number>                 constraints;

  // *** PDE information
  std::shared_ptr<Function<dim>> analytical_solution;
  std::shared_ptr<Function<dim>> load_function;
  types::boundary_id             dirichlet_id = 0;

  // *** linear algebra
  unsigned int                                   level;
  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  SYSTEM_MATRIX                                  system_matrix;
  VECTOR                                         system_u;
  VECTOR                                         system_delta_u;
  VECTOR                                         system_rhs;
  ReductionControl                               solver_control;
  SolverSelector<VECTOR>                         iterative_solver;

  // *** multigrid
  MGConstrainedDoFs                          mg_constrained_dofs;
  MGLevelObject<LEVEL_MATRIX>                mg_matrices;
  MG_TRANSFER                                mg_transfer;
  mutable TiledColoring<dim>                 user_coloring;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ> mg_schwarz_smoother_pre;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ> mg_schwarz_smoother_post;
  const MGSmootherBase<VECTOR> *             mg_smoother_pre;
  const MGSmootherBase<VECTOR> *             mg_smoother_post;
  CoarseGridSolver<LEVEL_MATRIX, VECTOR>     coarse_grid_solver;
  const MGCoarseGridBase<VECTOR> *           mg_coarse_grid;
  mg::Matrix<VECTOR>                         mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>>         multigrid;

  // *** preconditioners
  std::shared_ptr<GMG_PRECONDITIONER> preconditioner_mg;
  PreconditionIdentity                preconditioner_id;


  ModelProblem(const RT::Parameter & rt_parameters_in)
    : rt_parameters(rt_parameters_in),
      pcout(std::make_shared<ConditionalOStream>(std::cout,
                                                 Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                                                   0)),
      triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe(std::make_shared<FE_Q<dim>>(fe_degree)),
      mapping(fe_degree),
      analytical_solution(std::make_shared<Laplace::Solution<dim>>()),
      load_function(std::make_shared<Laplace::ManufacturedLoad<dim>>(analytical_solution)),
      level(static_cast<unsigned int>(-1)),
      user_coloring(rt_parameters_in.mesh),
      mg_smoother_pre(nullptr),
      mg_smoother_post(nullptr),
      mg_coarse_grid(nullptr)
  {
  }

  ~ModelProblem() = default;

  unsigned
  n_colors_system()
  {
    if(!mg_schwarz_smoother_pre)
      return 0;

    AssertThrow(rt_parameters.multigrid.pre_smoother.variant ==
                  SmootherParameter::SmootherVariant::Schwarz,
                ExcMessage("Pre-smoother isn't of Schwarz-type."));
    return mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
  }


  // /*
  //  * Prints the accumulated timings of the Schwarz pre- and post-smoothers on
  //  * the finest level.
  //  */
  // void
  // print_schwarz_preconditioner_times()
  // {
  //   Assert(mg_schwarz_smoother_pre, ExcMessage("MG Schwarz smoother isn't initialized."));
  //   Assert(mg_schwarz_smoother_post, ExcMessage("MG Schwarz smoother isn't initialized."));
  //   const auto precondition_pre  = mg_schwarz_smoother_pre->get_preconditioner();
  //   const auto time_data_pre     = precondition_pre->get_time_data();
  //   const auto precondition_post = mg_schwarz_smoother_post->get_preconditioner();
  //   const auto time_data_post    = precondition_post->get_time_data();
  //   AssertDimension(time_data_pre.size(), time_data_post.size());
  //   for(unsigned t = 0; t < time_data_pre.size(); ++t)
  //   {
  //     const auto description = time_data_pre[t].description;
  //     const auto t_max       = Utilities::MPI::max(time_data_pre[t].time, MPI_COMM_WORLD) +
  //                        Utilities::MPI::max(time_data_post[t].time, MPI_COMM_WORLD);
  //     print_parameter(description, t_max);
  //   }
  // }


  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const
  {
    *pcout << Util::parameter_to_fstring(description, value);
  }


  void
  print_informations() const
  {
    AssertThrow(fe, ExcMessage("Finite element is not initialized."));
    print_parameter("Finite element:", fe->get_name());
    *pcout << rt_parameters.to_string();
    *pcout << std::endl;
  }


  template<typename OtherNumber>
  std::shared_ptr<const MatrixFree<dim, OtherNumber>>
  build_mf_storage() const
  {
    typename MatrixFree<dim, OtherNumber>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, OtherNumber>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);

    QGauss<1> quadrature(n_qpoints);

    const auto mf_storage = std::make_shared<MatrixFree<dim, OtherNumber>>();
    mf_storage->reinit(mapping, dof_handler, constraints, quadrature, additional_data);

    return mf_storage;
  }


  template<typename OtherNumber>
  std::shared_ptr<const MatrixFree<dim, OtherNumber>>
  build_mf_storage(const unsigned int level) const
  {
    AssertIndexRange(level, triangulation.n_global_levels());

    typename MatrixFree<dim, OtherNumber>::AdditionalData additional_data;
    additional_data.mg_level              = level;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, OtherNumber>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);

    /// TODO check if this is more efficient than using
    /// MGConstrainedDoFs::get_level_constraints() in case of using MPI
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
    AffineConstraints<double> level_constraints;
    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
    level_constraints.close();

    QGauss<1> quadrature(n_qpoints);

    const auto mf_storage = std::make_shared<MatrixFree<dim, OtherNumber>>();
    mf_storage->reinit(mapping, dof_handler, level_constraints, quadrature, additional_data);

    return mf_storage;
  }


  template<typename OtherNumber>
  std::shared_ptr<const SubdomainHandler<dim, OtherNumber>>
  build_patch_storage(const unsigned                                            level,
                      const std::shared_ptr<const MatrixFree<dim, OtherNumber>> mf_storage,
                      const bool is_pre_smoother = true) const
  {
    AssertIndexRange(level, triangulation.n_global_levels());

    typename SubdomainHandler<dim, OtherNumber>::AdditionalData fdss_additional_data;
    fdss_additional_data.level = level;
    if(rt_parameters.multigrid.pre_smoother.schwarz.manual_coloring)
      fdss_additional_data.coloring_func = std::ref(user_coloring);
    rt_parameters.template fill_schwarz_smoother_data<dim, OtherNumber>(fdss_additional_data,
                                                                        is_pre_smoother);

    const auto patch_storage = std::make_shared<SubdomainHandler<dim, OtherNumber>>();
    patch_storage->reinit(mf_storage, fdss_additional_data);
    return patch_storage;
  }


  bool
  create_triangulation(const unsigned n_refinements)
  {
    triangulation.clear();
    this->level = static_cast<unsigned int>(-1);

    MeshParameter mesh_prms = rt_parameters.mesh;
    mesh_prms.n_refinements = n_refinements;

    AssertThrow(
      !rt_parameters.exceeds_dof_limits(numbers::invalid_dof_index),
      ExcMessage(
        "There is no DoF estimation implemented for standard finite elements. Therefore, the dof limit functionality is disabled."));

    /// create the triangulation and store few informations
    *pcout << create_mesh(triangulation, mesh_prms);
    this->level = triangulation.n_global_levels() - 1;
    pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());
    pp_data.n_dimensions = dim;

    return true;
  }


  /*
   * Convenience function creating the mesh based on RT::Parameter.
   */
  bool
  create_triangulation()
  {
    return create_triangulation(rt_parameters.mesh.n_refinements);
  }


  void
  distribute_dofs()
  {
    dof_handler.clear();
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();
  }


  void
  compute_rhs(const MatrixFree<dim, Number> * mf_storage, const Function<dim> & load_function)
  {
    system_rhs = 0.;
    system_u   = 0.;
    constraints.distribute(system_u);
    system_u.update_ghost_values();

    FEEvaluation<dim, fe_degree> phi(*mf_storage);
    for(unsigned int cell = 0; cell < mf_storage->n_cell_batches(); ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(system_u);
      phi.evaluate(false, true);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        Point<dim, VectorizedArray<Number>> point_batch = phi.quadrature_point(q);
        const auto &                        load_value = VHelper::value(load_function, point_batch);
        phi.submit_value(load_value, q);
        phi.submit_gradient(-phi.get_gradient(q), q);
      }
      phi.integrate(true, true);
      phi.distribute_local_to_global(system_rhs);
    }
    system_rhs.compress(VectorOperation::add);
  }


  void
  prepare_linear_system(const bool do_compute_rhs = true)
  {
    /// initialize constraints (strong B.C.)
    constraints.clear();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, dirichlet_id, *analytical_solution, constraints);
    constraints.close();

    auto mf_storage = build_mf_storage<Number>();
    system_matrix.initialize(mf_storage);
    pp_data.n_dofs_global.push_back(system_matrix.m());
    mf_storage->initialize_dof_vector(system_u);
    mf_storage->initialize_dof_vector(system_delta_u);
    if(do_compute_rhs)
    {
      mf_storage->initialize_dof_vector(system_rhs);
      compute_rhs(mf_storage.get(), *load_function);
    }
  }


  void
  prepare_mg_smoothers()
  {
    /// setup Schwarz-type pre-smoother
    if(rt_parameters.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      const auto                                   mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
      typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data;
      mgss_data.coloring_func = std::ref(user_coloring);
      mgss_data.parameters    = rt_parameters.multigrid.pre_smoother;
      std::set<types::boundary_id> dirichlet_boundary_ids;
      dirichlet_boundary_ids.insert(0);
      mgss_data.dirichlet_ids.emplace_back(dirichlet_boundary_ids);
      auto & shape_infos = mgss_data.shape_infos;
      shape_infos.reinit(1, dim);
      for(auto d = 0U; d < dim; ++d)
      {
        const auto mf_storage = mg_matrices[mg_matrices.max_level()].get_matrix_free();
        shape_infos(0, d)     = mf_storage->get_shape_info(0);
      }
      mgss->initialize(mg_matrices, mgss_data);
      mg_schwarz_smoother_pre = mgss;
    }
    else
      AssertThrow(false, ExcMessage("Smoothing variant not implemented. TODO"));

    /// setup Schwarz-type post-smoother
    if(rt_parameters.multigrid.post_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("MG Schwarz pre-smoother not initialized."));
      typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
      const auto schwarz_data_post       = rt_parameters.multigrid.post_smoother.schwarz;
      precondition_data.relaxation       = schwarz_data_post.damping_factor;
      precondition_data.local_relaxation = schwarz_data_post.local_damping_factor;
      precondition_data.symmetrized      = schwarz_data_post.symmetrize_smoothing;
      precondition_data.reverse          = schwarz_data_post.reverse_smoothing;

      /// use pre-smoother as well as post-smoother
      if(rt_parameters.multigrid.pre_smoother == rt_parameters.multigrid.post_smoother)
      {
        // *pcout << "Using pre-smoother as post-smoother ... " << std::endl;
        // *pcout << Util::parameter_to_fstring("/// Post-smoother (OLD)", "");
        // *pcout << rt_parameters.multigrid.post_smoother.to_string();
        // rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
        // *pcout << Util::parameter_to_fstring("/// Post-smoother (NEW)", "");
        // *pcout << rt_parameters.multigrid.post_smoother.to_string();
        mg_schwarz_smoother_post = mg_schwarz_smoother_pre;
      }

      /// initialize (independent) post-smoother
      else
      {
        typename SubdomainHandler<dim, typename LEVEL_MATRIX::value_type>::AdditionalData
          sd_handler_data;
        rt_parameters.template fill_schwarz_smoother_data<dim, typename LEVEL_MATRIX::value_type>(
          sd_handler_data, false);
        sd_handler_data.level = mg_matrices.max_level();
        if(rt_parameters.multigrid.post_smoother.schwarz.manual_coloring)
          sd_handler_data.coloring_func = std::ref(user_coloring);
        const bool is_shallow_copyable =
          mg_schwarz_smoother_pre->get_preconditioner(level)->is_shallow_copyable(sd_handler_data);

        if(is_shallow_copyable)
        {
          const auto mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
          typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data;
          mgss_data.coloring_func = std::ref(user_coloring);
          mgss_data.parameters    = rt_parameters.multigrid.post_smoother;
          mgss->initialize(*mg_schwarz_smoother_pre, mgss_data);
          mg_schwarz_smoother_post = mgss;
        }
        else
          AssertThrow(false, ExcMessage("Shallow copy is not possible. TODO"));
      }
    }
    else
      AssertThrow(false, ExcMessage("Smoothing variant is not implemented. TODO"));
  }


  void
  prepare_multigrid()
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
    mg_constrained_dofs.clear();

    // *** setup multigrid data
    const unsigned mg_level_min = rt_parameters.multigrid.coarse_level;
    const unsigned mg_level_max = this->level;
    pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);

    // *** initialize multigrid constraints
    mg_constrained_dofs.initialize(dof_handler);
    std::set<types::boundary_id> mg_boundary_ids;
    mg_boundary_ids.insert(dirichlet_id);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, mg_boundary_ids);

    // *** initialize level matrices A_l
    mg_matrices.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
    {
      const auto mf_storage_level = build_mf_storage<value_type_mg>(level);
      mg_matrices[level].initialize(mf_storage_level, mg_constrained_dofs, level);
    }

    // *** initialize multigrid transfer R_l
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    // *** initialize Schwarz smoother S_l
    prepare_mg_smoothers();

    /// set pre-smoother
    pp_data.n_colors_system.push_back(n_colors_system());
    if(rt_parameters.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Is not initialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
    }
    else
      AssertThrow(false, ExcMessage("Pre-smoothing variant is not implemented. TODO"));

    /// set post-smoother
    if(rt_parameters.multigrid.post_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Is not initialized."));
      mg_smoother_post = mg_schwarz_smoother_post.get();
    }
    else
      AssertThrow(false, ExcMessage("Post-smoothing variant is not implemented. TODO"));

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


  const GMG_PRECONDITIONER &
  prepare_preconditioner_mg()
  {
    prepare_multigrid();
    AssertThrow(multigrid, ExcNotInitialized());

    preconditioner_mg = std::make_shared<GMG_PRECONDITIONER>(dof_handler, *multigrid, mg_transfer);
    return *preconditioner_mg;
  }


  /**
   * compute the average reduction rho over n iterations and the
   * fractional number of iterations to achieve the requested
   * reduction (relative stopping criterion)
   */
  std::pair<double, double>
  compute_fractional_steps(const ReductionControl & solver_control)
  {
    const double residual_0 = solver_control.initial_value();
    const double residual_n = solver_control.last_value();
    const int    n          = solver_control.last_step(); // number of iterations
    const double reduction  = solver_control.reduction(); // relative tolerance

    // *** average reduction: r_n = rho^n * r_0
    const double rho = std::pow(residual_n / residual_0, static_cast<double>(1. / n));

    /**
     * since r_n <= reduction * r_0 we can compute the fractional
     * number of iterations n_frac that is sufficient to achieve the
     * desired reduction:
     *    rho^n_frac = reduction   <=>   n_frac = log(reduction)/log(rho)
     */
    const double n_frac = std::log(reduction) / std::log(rho);

    return std::make_pair(n_frac, rho);
  }


  template<typename PreconditionerType>
  void
  solve(const PreconditionerType & preconditioner)
  {
    solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
    solver_control.set_reduction(rt_parameters.solver.rel_tolerance);
    solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
    solver_control.log_history(true);
    solver_control.log_result(true);
    solver_control.enable_history_data();

    iterative_solver.set_control(solver_control);
    iterative_solver.select(rt_parameters.solver.variant);
    constraints.set_zero(system_delta_u);
    iterative_solver.solve(system_matrix, system_delta_u, system_rhs, preconditioner);
    system_u += system_delta_u;

    double n_frac                    = 0.;
    double reduction_rate            = 0.;
    std::tie(n_frac, reduction_rate) = compute_fractional_steps(solver_control);
    pp_data.average_reduction_system.push_back(reduction_rate);
    pp_data.n_iterations_system.push_back(n_frac);
    pp_data.solve_time.push_back(-2712.1989);

    print_parameter("Average reduction (solver):", reduction_rate);
    print_parameter("Number of iterations (solver):", n_frac);
  }


  double
  compute_l2_error(const MatrixFree<dim, Number> * mf_storage,
                   const Function<dim> *           analytic_solution) const
  {
    double                                             global_error = 0;
    FEEvaluation<dim, fe_degree, n_qpoints, 1, Number> phi(*mf_storage);
    system_u.update_ghost_values();
    const auto & uh = system_u;
    for(unsigned int cell = 0; cell < mf_storage->n_macro_cells(); ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(uh);
      phi.evaluate(true, false);
      VectorizedArray<Number> local_error = 0.;
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        const auto value_u  = VHelper::value(*analytic_solution, phi.quadrature_point(q));
        const auto value_uh = phi.get_value(q);
        local_error += (value_uh - value_u) * (value_uh - value_u) * phi.JxW(q);
      }
      for(unsigned int v = 0; v < mf_storage->n_active_entries_per_cell_batch(cell); ++v)
      {
        global_error += local_error[v];
      }
    }
    global_error = Utilities::MPI::sum(global_error, MPI_COMM_WORLD);
    return std::sqrt(global_error);
  }


  void
  compute_discretization_errors() const
  {
    const auto   mf_storage = system_matrix.get_matrix_free();
    const double l2_error   = compute_l2_error(mf_storage.get(), analytical_solution.get());
    pp_data.L2_error.push_back(l2_error);
    print_parameter("||u - uh||_L2 =", l2_error);
  }


  void
  run()
  {
    pp_data = PostProcessData{};
    TimerOutput time(MPI_COMM_WORLD, *pcout, TimerOutput::summary, TimerOutput::wall_times);

    for(unsigned cycle = 0; cycle < rt_parameters.n_cycles; ++cycle)
    {
      print_parameter("//////////////////////////////", "//////////////////////////////");
      print_parameter("Poisson run cycle:", cycle);
      *pcout << std::endl;

      {
        TimerOutput::Scope time_section(time, "Create triangulation");
        const unsigned     n_refinements = rt_parameters.mesh.n_refinements + cycle;
        const bool         is_tria_valid = create_triangulation(n_refinements);
        if(!is_tria_valid)
          continue;
      }

      {
        TimerOutput::Scope time_section(time, "Distribute dofs");
        distribute_dofs();
      }

      {
        TimerOutput::Scope time_section(time, "Setup linear system");
        prepare_linear_system();
      }

      switch(rt_parameters.solver.precondition_variant)
      {
        case SolverParameter::PreconditionVariant::None:
        {
          {
            print_informations();
            TimerOutput::Scope time_section(time, "Solve linear system");
            solve(preconditioner_id);
          }
          break;
        }
        case SolverParameter::PreconditionVariant::GMG:
        {
          {
            TimerOutput::Scope time_section(time, "Setup MG preconditioner");
            prepare_preconditioner_mg();
          }
          {
            print_informations();
            TimerOutput::Scope time_section(time, "Solve");
            solve(*preconditioner_mg);
          }
          break;
        }
        default:
          AssertThrow(false, ExcNotImplemented());
      }

      compute_discretization_errors();
      visualize_dof_vector(dof_handler, system_u, "solution", 1, mapping);
      // print_schwarz_preconditioner_times();
    }
  }
};

} // end namespace CFEM
} // end namespace Poisson

#endif // POISSONPROBLEM_H_
