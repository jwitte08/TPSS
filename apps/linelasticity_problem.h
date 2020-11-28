/**
 * linelasticity_problem.h
 *
 * DG linear elasticity problem used to test Schwarz smoothers.
 *
 *  Created on: Jun 24, 2019
 *      Author: witte
 */

#ifndef TESTS_LINELASTICITYPROBLEM_H_
#define TESTS_LINELASTICITYPROBLEM_H_

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_selector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/meshworker/assembler.h>



#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/preconditioner/schwarz_preconditioner.h"



#include "coloring.h"
#include "equation_data.h"
#include "linelasticity_integrator.h"
#include "mesh.h"
#include "multigrid.h"
#include "postprocess.h"
#include "rt_parameter.h"
#include "vectorization_helper.h"



using namespace dealii;

namespace LinElasticity
{
template<int dim,
         int fe_degree,
         typename Number     = double,
         typename MatrixType = Tensors::BlockMatrixDiagonal<dim, VectorizedArray<Number>>>
struct ModelProblem : public Subscriptor
{
  static constexpr unsigned int n_components = dim;
  static constexpr unsigned int fe_order     = fe_degree + 1;

  using value_type    = Number;
  using VECTOR        = typename LinearAlgebra::distributed::BlockVector<Number>;
  using SYSTEM_MATRIX = LinElasticity::MF::Operator<dim, fe_degree, Number>;

  using value_type_mg          = Number;
  using LEVEL_MATRIX           = LinElasticity::CombinedOperator<dim, fe_degree, value_type_mg>;
  using MG_TRANSFER            = MGTransferMatrixFree<dim, value_type_mg>;
  using PATCH_MATRIX           = MatrixType;
  using SCHWARZ_PRECONDITIONER = SchwarzPreconditioner<dim, LEVEL_MATRIX, VECTOR, PATCH_MATRIX>;
  using SCHWARZ_SMOOTHER       = SchwarzSmoother<dim, LEVEL_MATRIX, SCHWARZ_PRECONDITIONER, VECTOR>;
  using MG_SMOOTHER_SCHWARZ    = MGSmootherSchwarz<dim, LEVEL_MATRIX, PATCH_MATRIX, VECTOR>;

  // *** parameters and auxiliary structs
  RT::Parameter           rt_parameters;
  ConditionalOStream &    pcout;
  mutable PostProcessData pp_data;

  // *** Matrix-based code (testing)
  SparsityPattern      sparsity;
  SparseMatrix<double> sparse_matrix;

  // *** FEM fundamentals
  parallel::distributed::Triangulation<dim>                  triangulation;
  std::array<std::shared_ptr<DoFHandler<dim>>, n_components> dof_handlers;
  std::shared_ptr<FiniteElement<dim>>                        fe;
  const MappingQGeneric<dim>                                 mapping;

  // *** PDE information
  const EquationData            equation_data;
  const AnalyticalSolution<dim> analytical_solution;
  const VolumeForce<dim>        volume_force;

  // *** linear algebra (system of equations)
  unsigned int                                   level;
  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  SYSTEM_MATRIX                                  system_matrix;
  VECTOR                                         system_u;
  VECTOR                                         system_rhs;
  ReductionControl                               solver_control;
  SolverSelector<VECTOR>                         iterative_solver;

  // *** multigrid
  MGLevelObject<LEVEL_MATRIX>                   mg_matrices;
  std::vector<MGConstrainedDoFs>                mg_constrained_dofs;
  MGTransferBlockMatrixFree<dim, value_type_mg> mg_transfer;
  RedBlackColoring<dim>                         red_black_coloring;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>    mg_schwarz_smoother_pre;
  std::shared_ptr<const MG_SMOOTHER_SCHWARZ>    mg_schwarz_smoother_post;
  const MGSmootherBase<VECTOR> *                mg_smoother_pre;
  const MGSmootherBase<VECTOR> *                mg_smoother_post;
  CoarseGridSolver<LEVEL_MATRIX, VECTOR>        coarse_grid_solver;
  const MGCoarseGridBase<VECTOR> *              mg_coarse_grid;
  mg::Matrix<VECTOR>                            mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>>            multigrid;

  // *** preconditioners
  std::shared_ptr<const PreconditionMG<dim, VECTOR, decltype(mg_transfer)>> preconditioner_mg;
  PreconditionIdentity                                                      preconditioner_id;

  ModelProblem(ConditionalOStream &  pcout_in,
               const RT::Parameter & rt_parameters_in,
               const EquationData &  equation_data_in = EquationData{})
    : rt_parameters(rt_parameters_in),
      pcout(pcout_in),
      triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe(std::make_shared<FE_DGQ<dim>>(fe_degree)),
      mapping(fe_degree),
      equation_data(equation_data_in),
      analytical_solution(equation_data),
      volume_force(equation_data),
      level(static_cast<unsigned int>(-1)),
      mg_constrained_dofs(dof_handlers.size()),
      mg_transfer(mg_constrained_dofs),
      red_black_coloring(rt_parameters_in.mesh),
      mg_smoother_pre(nullptr),
      mg_smoother_post(nullptr),
      mg_coarse_grid(nullptr)
  {
  }

  std::vector<const DoFHandler<dim> *>
  get_dof_handlers() const
  {
    std::vector<const DoFHandler<dim> *> dofhs;
    std::transform(dof_handlers.cbegin(),
                   dof_handlers.cend(),
                   std::back_inserter(dofhs),
                   [](const auto & p) { return p.get(); });
    return dofhs;
  }

  const DoFHandler<dim> &
  get_dof_handler(const unsigned int component)
  {
    AssertIndexRange(component, dim);
    return *(dof_handlers[component]);
  }

  std::shared_ptr<const SubdomainHandler<dim, value_type_mg>>
  get_subdomain_handler(const unsigned int level = numbers::invalid_unsigned_int) const
  {
    AssertThrow(mg_schwarz_smoother_pre, ExcMessage("MG Schwarz pre-smoother is uninitialized."));
    return mg_schwarz_smoother_pre->get_subdomain_handler(level);
  }

  void
  assemble_matrix()
  {
    // *** distribute dofs
    DoFHandler<dim> dof_handler{triangulation};
    FESystem<dim>   fe_system{*fe /*FE_DGQ<dim>(fe_degree)*/, dim};
    dof_handler.initialize(triangulation, fe_system);
    dof_handler.distribute_dofs(fe_system);
    dof_handler.distribute_mg_dofs();
    // dof_handler.initialize_local_block_info();

    // *** initialize sparse matrix
    Assert(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
           ExcMessage(
             "In order to use the serial sparse matrix format only 1 mpi process is allowed!"));
    const unsigned int     n_dofs     = dof_handler.n_dofs();
    const unsigned int     n_q_points = fe_degree + 1;
    DynamicSparsityPattern dsp(n_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity.copy_from(dsp);
    sparse_matrix.reinit(sparsity);

    MeshWorker::IntegrationInfoBox<dim> info_box;
    UpdateFlags update_flags = update_values | update_gradients | update_quadrature_points;
    info_box.add_update_flags_all(update_flags);
    info_box.initialize_gauss_quadrature(n_q_points, n_q_points, n_q_points);
    info_box.initialize(fe_system, mapping);

    MeshWorker::DoFInfo<dim>                                  dof_info(dof_handler);
    MeshWorker::Assembler::MatrixSimple<SparseMatrix<Number>> assembler;
    assembler.initialize(sparse_matrix);
    LinElasticity::MW::MatrixIntegrator<dim> integrator(equation_data);
    MeshWorker::integration_loop<dim, dim>(
      dof_handler.begin_active(), dof_handler.end(), dof_info, info_box, integrator, assembler);
  }

  template<typename OtherNumber>
  std::shared_ptr<const MatrixFree<dim, OtherNumber>>
  build_mf_storage(const unsigned int level = static_cast<unsigned>(-1))
  {
    using AddData     = typename MatrixFree<dim, OtherNumber>::AdditionalData;
    using TasksScheme = typename AddData::TasksParallelScheme;
    /// dummy constraints
    AffineConstraints<double> constraints_dummy;
    constraints_dummy.close();
    std::vector<const AffineConstraints<double> *> constraints_dummies;
    std::fill_n(std::back_inserter(constraints_dummies), n_components, &constraints_dummy);

    /// initialize quadrature
    unsigned int n_qpoints = fe_degree + 1;
    QGauss<1>    quadrature(n_qpoints);

    /// prepare additional data
    AddData    additional_data;
    const auto mapping_update_flags = dealii::update_gradients | dealii::update_JxW_values |
                                      dealii::update_quadrature_points | dealii::update_values;
    additional_data.mapping_update_flags                = mapping_update_flags;
    additional_data.mapping_update_flags_inner_faces    = mapping_update_flags;
    additional_data.mapping_update_flags_boundary_faces = mapping_update_flags;
    if(level != static_cast<unsigned>(-1))
      additional_data.mg_level = level;
    const auto tasks_scheme               = static_cast<TasksScheme>(0 /*none*/);
    additional_data.tasks_parallel_scheme = tasks_scheme;

    /// initialize matrix-free storage
    const auto mf_storage = std::make_shared<MatrixFree<dim, OtherNumber>>();
    mf_storage->reinit(
      mapping, get_dof_handlers(), constraints_dummies, quadrature, additional_data);

    return mf_storage;
  }

  template<typename OtherNumber>
  std::shared_ptr<const SubdomainHandler<dim, OtherNumber>>
  build_patch_storage(const unsigned                                            level,
                      const std::shared_ptr<const MatrixFree<dim, OtherNumber>> mf_storage,
                      const bool is_pre_smoother = true)
  {
    /// prepare additional data
    typename SubdomainHandler<dim, OtherNumber>::AdditionalData fdss_additional_data;
    fdss_additional_data.level = level;
    if(rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring)
      fdss_additional_data.coloring_func = std::ref(red_black_coloring);
    rt_parameters.template fill_schwarz_smoother_data<dim, OtherNumber>(fdss_additional_data,
                                                                        is_pre_smoother);

    /// initialize patch storage
    const auto patch_storage = std::make_shared<SubdomainHandler<dim, OtherNumber>>();
    patch_storage->reinit(mf_storage, fdss_additional_data);
    return patch_storage;
  }

  bool
  create_triangulation(const unsigned n_refinements) //, const bool print_details = false)
  {
    triangulation.clear();
    this->level = static_cast<unsigned int>(-1);

    if(false) // TODO check estimated dofs
      return false;

    MeshParameter mesh_prms = rt_parameters.mesh;
    mesh_prms.n_refinements = n_refinements;

    /// if the dof limits are exceeded we return without creating the mesh
    const auto n_dofs_est = n_components * estimate_n_dofs(*fe, mesh_prms);
    if(rt_parameters.exceeds_dof_limits(n_dofs_est))
      return false;

    /// create the triangulation and store few informations
    pcout << create_mesh(triangulation, mesh_prms) << std::endl;
    this->level = triangulation.n_global_levels() - 1;
    pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());

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
  distribute_dofs(const bool /*print_details*/ = false)
  {
    /// initialize one scalar dof handler
    const auto dofh_in = std::make_shared<DoFHandler<dim>>(triangulation);
    dofh_in->initialize(triangulation, *fe);
    dofh_in->distribute_mg_dofs();

    /// reset each component of the vector-valued dof handler @p dof_handlers
    for(auto & dofh : dof_handlers)
      dofh = dofh_in;
  }

  void
  compute_rhs(const bool /*print_details*/ = false)
  {
    const auto & data = *mf_storage;
    { // CELL INTEGRALS
      std::vector<std::shared_ptr<FEEvaluation<dim, fe_degree>>> phi;
      for(unsigned int comp = 0; comp < dim; ++comp)
        phi.push_back(std::make_shared<FEEvaluation<dim, fe_degree>>(data, comp));

      for(unsigned int cell = 0; cell < mf_storage->n_cell_batches(); ++cell)
        for(unsigned int comp = 0; comp < dim; ++comp)
        {
          auto & phi_comp        = *(phi[comp]);
          auto & system_rhs_comp = system_rhs.block(comp);
          phi_comp.reinit(cell);
          for(unsigned int q = 0; q < phi_comp.n_q_points; ++q)
          {
            VectorizedArray<double>             rhs_val     = VectorizedArray<double>();
            Point<dim, VectorizedArray<double>> point_batch = phi_comp.quadrature_point(q);
            for(unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
            {
              Point<dim> single_point;
              for(unsigned int d = 0; d < dim; ++d)
                single_point[d] = point_batch[d][v];
              rhs_val[v] = volume_force.value(single_point)[comp];
            }
            phi_comp.submit_value(rhs_val, q);
          }
          phi_comp.integrate_scatter(true, false, system_rhs_comp);
        }
    }

    { // FACE INTEGRALS
      std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> v;
      for(unsigned int comp = 0; comp < dim; ++comp)
        v.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(
          data, true, comp));

      const auto mu         = equation_data.mu;
      const auto lambda     = equation_data.lambda;
      const auto face_range = std::make_pair<unsigned, unsigned>(data.n_inner_face_batches(),
                                                                 data.n_inner_face_batches() +
                                                                   data.n_boundary_face_batches());
      for(unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        for(unsigned int comp = 0; comp < dim; ++comp)
          v[comp]->reinit(face);

        // *** compute penalty
        const VectorizedArray<Number> inverse_length_normal_to_face =
          std::abs((v[0]->get_normal_vector(0) * v[0]->inverse_jacobian(0))[dim - 1]);
        const VectorizedArray<Number> sigma =
          2. * inverse_length_normal_to_face * system_matrix.get_penalty_factor();
        const unsigned int n_qpoints = v[0]->n_q_points; // TODO assert isotropic quadrature
        Assert(std::all_of(v.cbegin(),
                           v.cend(),
                           [n_qpoints](const auto & phi) { return phi->n_q_points == n_qpoints; }),
               ExcMessage("Quadrature is not isotropic."));

        // *** integrate against test functions and derivates v (LINEAR STRAIN)
        for(unsigned int q = 0; q < n_qpoints; ++q)
        {
          const auto normal      = v[0]->get_normal_vector(q);
          const auto g_dirichlet = VHelper::value(analytical_solution, v[0]->quadrature_point(q));

          system_matrix.submit_value(v, 2. * mu * sigma * g_dirichlet, q);
          system_matrix.submit_symmetric_gradient(v,
                                                  -2. * mu * outer_product(g_dirichlet, normal),
                                                  q);
        }
        for(unsigned comp = 0; comp < dim; ++comp)
          v[comp]->integrate_scatter(true, true, system_rhs.block(comp));

        // *** integrate against test functions and derivates v (GRAD DIV)
        for(unsigned int q = 0; q < n_qpoints; ++q)
        {
          const auto normal      = v[0]->get_normal_vector(q);
          const auto g_dirichlet = VHelper::value(analytical_solution, v[0]->quadrature_point(q));

          system_matrix.submit_value(v, lambda * sigma * (g_dirichlet * normal) * normal, q);
          system_matrix.submit_divergence(v, -1. * lambda * g_dirichlet * normal, q);
        }
        for(unsigned comp = 0; comp < dim; ++comp)
          v[comp]->integrate_scatter(true, true, system_rhs.block(comp));
      }
    }
  }

  void
  prepare_system(const bool print_details = false, const bool do_compute_rhs = true)
  {
    // *** initialize system matrix
    this->mf_storage = build_mf_storage<Number>();
    system_matrix.initialize(mf_storage, equation_data);
    pp_data.n_dofs_global.push_back(system_matrix.m());

    // *** initialize solution vector
    system_u.reinit(n_components);
    for(unsigned int comp = 0; comp < n_components; ++comp)
      mf_storage->initialize_dof_vector(system_u.block(comp), comp);
    system_u.collect_sizes();

    // *** initialize (& compute) right hand side vector
    system_rhs.reinit(n_components);
    for(unsigned int comp = 0; comp < n_components; ++comp)
      mf_storage->initialize_dof_vector(system_rhs.block(comp), comp);
    system_rhs.collect_sizes();
    if(do_compute_rhs)
      compute_rhs(print_details);
  }

  void
  prepare_mg_smoothers()
  {
    /// setup Schwarz-type pre-smoother
    if(rt_parameters.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      const auto                                   mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
      typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data;
      mgss_data.coloring_func = std::ref(red_black_coloring);
      mgss_data.parameters    = rt_parameters.multigrid.pre_smoother;
      mgss->initialize(mg_matrices, mgss_data);
      mg_schwarz_smoother_pre = mgss;
    }
    else
      AssertThrow(false, ExcMessage("TODO ..."));

    /// setup Schwarz-type post-smoother
    if(rt_parameters.multigrid.post_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("MG Schwarz pre-smoother not initialized."));
      typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
      const auto schwarz_data_post  = rt_parameters.multigrid.post_smoother.schwarz;
      precondition_data.relaxation  = schwarz_data_post.damping_factor;
      precondition_data.symmetrized = schwarz_data_post.symmetrize_smoothing;
      precondition_data.reverse     = schwarz_data_post.reverse_smoothing;

      /// use pre-smoother as well as post-smoother
      if(rt_parameters.multigrid.pre_smoother == rt_parameters.multigrid.post_smoother)
      {
        mg_schwarz_smoother_post = mg_schwarz_smoother_pre;
      }

      /// initialize (independent) post-smoother
      else
      {
        typename SubdomainHandler<dim, typename LEVEL_MATRIX::value_type>::AdditionalData
          sdhandler_data;
        rt_parameters.template fill_schwarz_smoother_data<dim, typename LEVEL_MATRIX::value_type>(
          sdhandler_data, false);
        sdhandler_data.level = mg_matrices.max_level();
        const bool is_shallow_copyable =
          mg_schwarz_smoother_pre->get_preconditioner(level)->is_shallow_copyable(sdhandler_data);

        if(is_shallow_copyable)
        {
          const auto mgss = std::make_shared<MG_SMOOTHER_SCHWARZ>();
          typename MG_SMOOTHER_SCHWARZ::AdditionalData mgss_data;
          mgss_data.coloring_func = std::ref(red_black_coloring);
          mgss_data.parameters    = rt_parameters.multigrid.post_smoother;
          mgss->initialize(*mg_schwarz_smoother_pre, mgss_data);
          mg_schwarz_smoother_post = mgss;
        }
        else
          AssertThrow(false, ExcMessage("TODO ..."));
      }
    }
    else
      AssertThrow(false, ExcMessage("TODO ..."));
  }

  void
  prepare_multigrid(const bool /*print_details*/ = false)
  {
    // *** clear multigrid infrastructure
    multigrid.reset();
    mg_matrix_wrapper.reset();
    coarse_grid_solver.clear();
    mg_smoother_post = nullptr;
    mg_smoother_pre  = nullptr;
    mg_schwarz_smoother_pre.reset();
    mg_schwarz_smoother_post.reset();
    /// TODO Corrupt class design. clear() deletes the internal vector of
    /// transfer objects @p matrix_free_transfer_vector. however, the
    /// initializing function initialize_constraints() does not provide a
    /// resize() call, such that we access out-of-reach vector elements.
    // mg_transfer.clear();
    mg_matrices.clear_elements();

    // *** setup multigrid data
    const unsigned mg_level_min = rt_parameters.multigrid.coarse_level;
    const unsigned mg_level_max = level;
    pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);

    // *** initialize level matrices A_l
    mg_matrices.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
    {
      const auto mf_storage_level = build_mf_storage<value_type_mg>(level);
      mg_matrices[level].initialize(mf_storage_level, equation_data);
    }

    // *** initialize multigrid transfer R_l
    mg_constrained_dofs.resize(dof_handlers.size());
    for(unsigned int block = 0; block < mg_constrained_dofs.size(); ++block)
      mg_constrained_dofs[block].initialize(get_dof_handler(block));
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(get_dof_handlers());

    // *** initialize Schwarz smoother S_l
    prepare_mg_smoothers();

    /// set pre-smoother
    if(rt_parameters.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Schwarz pre-smoother is uninitialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
    }
    else
      AssertThrow(false, ExcMessage("TODO .. "));
    /// set post-smoother
    if(rt_parameters.multigrid.post_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Schwarz post-smoother is uninitialized."));
      mg_smoother_post = mg_schwarz_smoother_post.get();
    }
    else
      AssertThrow(false, ExcMessage("TODO .. "));

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


  void
  prepare_preconditioner_mg(const bool print_details = false)
  {
    prepare_multigrid(print_details);

    AssertThrow(multigrid, ExcMessage("Multigrid is uninitialized."));
    preconditioner_mg =
      std::make_shared<PreconditionMG<dim, VECTOR, decltype(mg_transfer)>>(get_dof_handlers(),
                                                                           *multigrid,
                                                                           mg_transfer);
  }


  void
  print_informations() const
  {
    AssertThrow(fe, ExcMessage("Finite element is not initialized."));
    pcout << equation_data.to_string();
    pcout << std::endl;
    print_parameter("Finite element (per component):", fe->get_name());
    pcout << rt_parameters.to_string();
    pcout << std::endl;
  }


  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const
  {
    pcout << Util::parameter_to_fstring(description, value);
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
  solve(const PreconditionerType & preconditioner, const bool print_details = false)
  {
    Timer time;
    time.restart();

    solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
    solver_control.set_reduction(rt_parameters.solver.rel_tolerance);
    solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
    solver_control.log_history(true);
    solver_control.log_result(true);
    solver_control.enable_history_data();

    iterative_solver.set_control(solver_control);
    iterative_solver.select(rt_parameters.solver.variant);
    iterative_solver.solve(system_matrix, system_u, system_rhs, preconditioner);
    time.stop();

    double n_frac                    = 0.;
    double reduction_rate            = 0.;
    std::tie(n_frac, reduction_rate) = compute_fractional_steps(solver_control);
    pp_data.average_reduction_system.push_back(reduction_rate);
    pp_data.n_iterations_system.push_back(n_frac);
    pp_data.solve_time.push_back(time.wall_time());

    if(print_details)
    {
      print_parameter("Average reduction (solver):", reduction_rate);
      print_parameter("Number of iterations (solver):", n_frac);
    }
  }

  void
  print_dof_info(const DoFHandler<dim> & dof_handler) const
  {
    print_row_variable(pcout, 2, "", 43, "Finite element:", fe->get_name());
    pcout << std::endl;

    print_row_variable(pcout, 5, "", 15, "level:", 15, "n_dofs:");
    for(unsigned l = rt_parameters.multigrid.coarse_level; l < triangulation.n_levels(); ++l)
      print_row_variable(pcout, 5, "", 15, l, 15, dof_handler.n_dofs(l));
    pcout << std::endl;
  }

  double
  compute_l2_error(const bool print_details = false) const
  {
    const auto & data = *(system_matrix.get_matrix_free());
    std::vector<std::shared_ptr<FEEvaluation<dim, fe_degree>>> phi;
    for(unsigned int comp = 0; comp < dim; ++comp)
      phi.push_back(std::make_shared<FEEvaluation<dim, fe_degree>>(data, comp));

    double global_error = 0.;
    for(unsigned int cell = 0; cell < data.n_macro_cells(); ++cell)
    {
      for(unsigned comp = 0; comp < dim; ++comp)
      {
        const auto & uh_comp = system_u.block(comp);
        phi[comp]->reinit(cell);
        phi[comp]->gather_evaluate(uh_comp, true, false);
      }
      auto local_error = make_vectorized_array<Number>(0.);
      for(unsigned int q = 0; q < phi[0]->n_q_points; ++q)
      {
        const auto value_u  = VHelper::value(analytical_solution, phi[0]->quadrature_point(q));
        const auto value_uh = system_matrix.get_value(phi, q);
        local_error += (value_uh - value_u) * (value_uh - value_u) * phi[0]->JxW(q);
      }
      for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
        global_error += local_error[v];
    }

    global_error = Utilities::MPI::sum(global_error, MPI_COMM_WORLD);
    global_error = std::sqrt(global_error);

    if(print_details)
    {
      pcout << " ||u - uh||_L2 = " << global_error << std::endl << std::endl;
    }

    return global_error;
  }

  void
  compute_discretization_errors(const bool print_details = false) const
  {
    const double l2_error = compute_l2_error(print_details);
    pp_data.L2_error.push_back(l2_error);
  }

  void
  print_convergence_table(const PostProcessData & pp_data)
  {
    ConvergenceTable convergence_table;
    AssertDimension(pp_data.L2_error.size(), pp_data.n_dofs_global.size());
    for(std::size_t t = 0; t < pp_data.L2_error.size(); ++t)
    {
      convergence_table.add_value("#DoFs", pp_data.n_dofs_global[t]);
      convergence_table.add_value("L2 error", pp_data.L2_error[t]);
    }

    convergence_table.set_precision("L2 error", 3);
    convergence_table.set_scientific("L2 error", true);

    convergence_table.evaluate_convergence_rates("L2 error", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("L2 error",
                                                 "#DoFs",
                                                 ConvergenceTable::reduction_rate_log2,
                                                 dim);

    if(pcout.is_active())
    {
      convergence_table.write_text(pcout.get_stream());
      pcout << "\n\n";
    }
  }

  void
  write_graphical_output()
  {
    Assert(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
           ExcMessage("Demanding one MPI process only!"));

    std::string  filename = "solution-" + Utilities::int_to_string(triangulation.n_levels() - 1, 2);
    DataOut<dim> data_out;
    // data_out.attach_dof_handler(dof_handler);

    const std::string prefix[] = {"x", "y", "z"};
    for(unsigned int comp = 0; comp < dim; ++comp)
      data_out.add_data_vector(get_dof_handler(comp),
                               system_u.block(comp),
                               prefix[comp] + "_solution");
    data_out.build_patches(mapping,
                           /*n_subdivisions*/ fe_degree,
                           DataOut<dim>::CurvedCellRegion::curved_inner_cells);

    std::fstream fstream;
    fstream.open(filename + ".vtu", std::ios_base::out);
    data_out.write_vtu(fstream);
    fstream.close();
  }

  void
  run(const bool print_details = true)
  {
    pp_data = PostProcessData{};
    TimerOutput time(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times);

    for(unsigned cycle = 0; cycle < rt_parameters.n_cycles; ++cycle)
    {
      print_parameter("//////////////////////////////", "//////////////////////////////");
      print_parameter("Linear elasticity run cycle:", cycle);
      pcout << std::endl;

      {
        TimerOutput::Scope time_section(time, "Create triangulation");
        const unsigned     n_refinements = rt_parameters.mesh.n_refinements + cycle;
        const bool         is_tria_valid = create_triangulation(n_refinements);
        if(!is_tria_valid)
        {
          print_parameter("Out of DoF limits!", "Try next...");
          continue;
        }
      }

      {
        TimerOutput::Scope time_section(time, "Distribute dofs");
        distribute_dofs(print_details);
      }

      {
        TimerOutput::Scope time_section(time, "Setup linear system");
        prepare_system(print_details);
      }

      switch(rt_parameters.solver.precondition_variant)
      {
        case SolverParameter::PreconditionVariant::None:
        {
          {
            print_informations();
            TimerOutput::Scope time_section(time, "Solve linear system");
            solve(preconditioner_id, print_details);
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
            solve(*preconditioner_mg, print_details);
          }
          break;
        }
        default:
          AssertThrow(false, ExcNotImplemented());
      }

      compute_discretization_errors(print_details);

      /// OPT write paraview files
      // write_graphical_output();
    }

    print_convergence_table(pp_data);
  }
}; // namespace LinElasticity

} // end namespace LinElasticity

#endif /* TESTS_LINELASTICITYPROBLEM_H_ */
