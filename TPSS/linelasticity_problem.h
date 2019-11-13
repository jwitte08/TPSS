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


// /* #include "solvers_and_preconditioners/preconditioner/preconditioner_base.h" */
// /* #  include "solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h" */

#include <deal.II/fe/fe_system.h>

#include "equation_data.h"
#include "laplace_problem.h"
#include "linelasticity_integrator.h"
#include "vectorization_helper.h"

using namespace dealii;



namespace LinElasticity
{
template<int dim, int fe_degree, typename Number = double, int n_patch_dofs = -1>
struct MatrixOperator : public Subscriptor
{
  static constexpr unsigned int n_components = dim;
  static constexpr unsigned int fe_order     = fe_degree + 1;

  using value_type    = Number;
  using VECTOR        = typename LinearAlgebra::distributed::BlockVector<Number>;
  using SYSTEM_MATRIX = LinElasticity::MF::Operator<dim, fe_degree, Number>;

  using value_type_mg = Number;
  using LEVEL_MATRIX  = LinElasticity::CombinedOperator<dim, fe_degree, value_type_mg>;
  using MG_TRANSFER   = MGTransferMatrixFree<dim, value_type_mg>;
  using PATCH_MATRIX  = Tensors::BlockMatrixDiagonal<dim, VectorizedArray<Number>, n_patch_dofs>;
  using SCHWARZ_PRECONDITIONER = SchwarzPreconditioner<dim, LEVEL_MATRIX, VECTOR, PATCH_MATRIX>;
  using SCHWARZ_SMOOTHER       = SchwarzSmoother<dim, LEVEL_MATRIX, SCHWARZ_PRECONDITIONER, VECTOR>;

  // *** parameters and auxiliary structs
  Laplace::MatrixOperator<dim, fe_degree, Number, n_patch_dofs> laplace_problem;
  Laplace::Parameter                                            parameters;
  ConditionalOStream &                                          pcout;
  mutable PostProcessData                                       pp_data;
  SparsityPattern                                               sparsity;
  SparseMatrix<double>                                          sparse_matrix;

  // *** FEM fundamentals
  parallel::distributed::Triangulation<dim>                  triangulation;
  std::array<std::shared_ptr<DoFHandler<dim>>, n_components> dof_handlers;
  std::shared_ptr<FiniteElement<dim>>                        fe;
  const MappingQGeneric<dim> &                               mapping;

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
  MGLevelObject<LEVEL_MATRIX>                                  mg_matrices;
  MGTransferBlockMatrixFree<dim, value_type_mg>                mg_transfer;
  MGLevelObject<std::shared_ptr<const SCHWARZ_PRECONDITIONER>> mg_schwarz_precondition;
  MGSmootherRelaxation<LEVEL_MATRIX, SCHWARZ_SMOOTHER, VECTOR> mg_schwarz_smoother;
  std::shared_ptr<const MGSmootherRelaxation<LEVEL_MATRIX, SCHWARZ_SMOOTHER, VECTOR>>
                                                  mg_schwarz_smoother_post;
  const MGSmootherBase<VECTOR> *                  mg_smoother_pre;
  const MGSmootherBase<VECTOR> *                  mg_smoother_post;
  SolverControl                                   coarse_control_exact;
  PreconditionIdentity                            coarse_preconditioner;
  SolverSelector<VECTOR>                          coarse_solver;
  std::shared_ptr<const MGCoarseGridBase<VECTOR>> mg_coarse_grid;
  mg::Matrix<VECTOR>                              mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>>              multigrid;

  // *** preconditioners
  std::shared_ptr<const PreconditionMG<dim, VECTOR, decltype(mg_transfer)>> preconditioner_mg;
  PreconditionIdentity                                                      preconditioner_id;

  MatrixOperator(ConditionalOStream &       pcout_in,
                 const Laplace::Parameter & parameters_in,
                 const EquationData &       equation_data_in = EquationData{})
    : laplace_problem(pcout_in, parameters_in, /*generate grid?*/ false),
      parameters(laplace_problem.parameters),
      pcout(pcout_in),
      triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe(std::make_shared<FE_DGQ<dim>>(fe_degree)),
      mapping(laplace_problem.mapping),
      equation_data(equation_data_in),
      analytical_solution(equation_data),
      volume_force(equation_data),
      level(static_cast<unsigned int>(-1)),
      mg_transfer(dof_handlers.size()),
      mg_smoother_pre(nullptr),
      mg_smoother_post(nullptr)
  {
    create_triangulation(parameters.n_refines, true);
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

  template<typename Number2>
  std::shared_ptr<const MatrixFree<dim, Number2>>
  build_mf_storage(const unsigned int level = static_cast<unsigned>(-1))
  {
    using AddData = typename MatrixFree<dim, Number2>::AdditionalData;

    AffineConstraints<double> constraints_dummy;
    constraints_dummy.close();
    std::vector<const AffineConstraints<double> *> constraints_dummies;
    std::fill_n(std::back_inserter(constraints_dummies), n_components, &constraints_dummy);
    const auto mf_storage = std::make_shared<MatrixFree<dim, Number2>>();

    unsigned int n_qpoints = fe_degree + 1;
    QGauss<1>    quadrature(n_qpoints);

    AddData    additional_data;
    const auto mapping_update_flags = dealii::update_gradients | dealii::update_JxW_values |
                                      dealii::update_quadrature_points | dealii::update_values;
    additional_data.mapping_update_flags                = mapping_update_flags;
    additional_data.mapping_update_flags_inner_faces    = mapping_update_flags;
    additional_data.mapping_update_flags_boundary_faces = mapping_update_flags;
    if(level != static_cast<unsigned>(-1))
      additional_data.mg_level = level;
    const auto tasks_scheme =
      static_cast<typename AddData::TasksParallelScheme>(parameters.mf_tasks_scheme_id);
    additional_data.tasks_parallel_scheme = tasks_scheme;

    mf_storage->reinit(
      mapping, get_dof_handlers(), constraints_dummies, quadrature, additional_data);
    return mf_storage;
  }

  template<typename Number2>
  std::shared_ptr<const SubdomainHandler<dim, Number2>>
  build_patch_storage(const unsigned                                        level,
                      const std::shared_ptr<const MatrixFree<dim, Number2>> mf_storage)
  {
    const auto & patch_storage = laplace_problem.build_patch_storage(level, mf_storage);
    return patch_storage;
  }

  bool
  create_triangulation(const unsigned n_refines, const bool print_details = false)
  {
    const unsigned int DOF_LIMIT_MAX_ = parameters.dof_limit_max;
    const unsigned int DOF_LIMIT_MIN_ = parameters.dof_limit_min;

    triangulation.clear();
    this->level    = -1;
    auto mesh_info = std::make_pair<bool, std::string>(false, "");

    if(parameters.geometry_variant == Laplace::Parameter::GeometryVariant::SubdividedCubeoid)
    {
      AssertDimension(parameters.n_subdivisions.size(), dim);
      std::array<unsigned int, dim> n_subdivisions;
      std::copy_n(parameters.n_subdivisions.cbegin(), dim, n_subdivisions.begin());

      // function
      const auto create_subdivided_rectangle = [](Triangulation<dim> &          tria,
                                                  const unsigned int            n_refinements,
                                                  std::array<unsigned int, dim> n_subdivisions) {
        constexpr double          h = 1.;
        std::vector<unsigned int> n_subdivs;
        std::copy(n_subdivisions.cbegin(), n_subdivisions.cend(), std::back_inserter(n_subdivs));
        Point<dim> xx0, xx1; // origin
        for(unsigned int d = 0; d < dim; ++d)
          xx1[d] = h * n_subdivisions[d];
        GridGenerator::subdivided_hyper_rectangle(tria, n_subdivs, xx0, xx1);
        tria.refine_global(n_refinements);

        std::ostringstream oss;
        oss << "domain: ";
        for(unsigned int d = 0; d < dim; ++d)
          oss << "(" << xx0[d] << ", " << xx1[d] << (d != (dim - 1) ? ") x " : ")\n");
        oss << "mesh: ";
        for(unsigned int d = 0; d < dim; ++d)
          oss << (n_subdivisions[d] * (1 << n_refinements)) << (d != (dim - 1) ? " x " : "\n");

        const auto info = std::make_pair<bool, std::string>(true, oss.str());
        return info;
      };

      mesh_info = create_subdivided_rectangle(triangulation, n_refines, n_subdivisions);
    }

    else if(parameters.geometry_variant == Laplace::Parameter::GeometryVariant::Cube)
    {
      // function
      const auto create_hyper_cube = [DOF_LIMIT_MIN_,
                                      DOF_LIMIT_MAX_](Triangulation<dim> & tria,
                                                      const unsigned int   n_refinements,
                                                      const unsigned int   n_repetitions) {
        constexpr unsigned n_dofs_per_cell_est = dim * Utilities::pow(fe_order, dim);
        const unsigned     n_cells_per_dim     = n_repetitions * (1 << n_refinements);
        const unsigned     n_cells_est         = Utilities::pow(n_cells_per_dim, dim);
        const unsigned     n_dofs_est          = n_cells_est * n_dofs_per_cell_est;
        if(n_dofs_est < DOF_LIMIT_MIN_ || DOF_LIMIT_MAX_ < n_dofs_est)
          return std::make_pair<bool, std::string>(false, "mesh exceeds limits!");

        const double left = 0.0, right = 1.0;
        GridGenerator::subdivided_hyper_cube(tria, n_repetitions, left, right);
        tria.refine_global(n_refinements);

        std::ostringstream oss;
        oss << "domain: ";
        for(unsigned int d = 0; d < dim; ++d)
          oss << "(" << left << ", " << right << (d != (dim - 1) ? ") x " : ")\n");
        oss << "mesh: ";
        for(unsigned int d = 0; d < dim; ++d)
          oss << n_cells_per_dim << (d != (dim - 1) ? " x " : "\n");

        const auto info = std::make_pair<bool, std::string>(true, oss.str());
        return info;
      };

      mesh_info = create_hyper_cube(triangulation, n_refines, parameters.n_cell_repetitions);
    }

    else
      AssertThrow(false, ExcNotImplemented());

    const bool triangulation_was_created = mesh_info.first;
    if(!triangulation_was_created) // invalid mesh
      return false;

    this->level = triangulation.n_levels() - 1;
    if(print_details)
      pcout << mesh_info.second << std::endl;
    return true; // valid mesh
  }

  void
  distribute_dofs(const bool print_details = false)
  {
    unsigned int component = 0;
    for(auto & dofh : dof_handlers)
    {
      dofh = std::make_shared<DoFHandler<dim>>(triangulation);
      dofh->initialize(triangulation, *fe);
      // dofh->distribute_dofs(*fe); //already done in initialize() call
      dofh->distribute_mg_dofs();
      if(print_details)
      {
        pcout << " ... distributed dofs of component " << component << std::endl;
        print_dof_info(*dofh);
      }
      ++component;
    }
  }

  void
  compute_rhs(const bool print_details = false)
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
            for(unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
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

    if(print_details)
      pcout << " ... assembled right hand side " << std::endl;
  }

  void
  prepare_system(const bool print_details = false, const bool do_compute_rhs = true)
  {
    // *** system matrix
    this->mf_storage = build_mf_storage<Number>();
    system_matrix.initialize(mf_storage, equation_data);
    pp_data.n_dofs_global.push_back(system_matrix.m());

    // TODO dirty workaround
    system_u.reinit(n_components);
    system_rhs.reinit(n_components);
    for(unsigned int comp = 0; comp < n_components; ++comp)
    {
      mf_storage->initialize_dof_vector(system_u.block(comp), comp);
      mf_storage->initialize_dof_vector(system_rhs.block(comp), comp);
    }
    system_u.collect_sizes();
    system_rhs.collect_sizes();

    // *** compute the linear system's right hand side
    if(do_compute_rhs)
      compute_rhs(print_details);
    if(print_details)
      pcout << " ... prepared system with " << mf_storage->n_components() << " components\n";
  }

  void
  prepare_multigrid(const bool print_details = false)
  {
    // *** setup multigrid data
    const unsigned mg_level_min = parameters.coarse_level;
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
    mg_transfer.build(get_dof_handlers());

    // *** initialize Schwarz smoother S_l
    const auto &                              schwarz_data = parameters.schwarz_smoother_data;
    typename SCHWARZ_SMOOTHER::AdditionalData dummy_data;
    mg_schwarz_smoother.initialize(mg_matrices,
                                   dummy_data); // insert A_l in MGSmootherRelaxation

    mg_schwarz_precondition.resize(mg_level_min, mg_level_max); // book-keeping
    for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
    {
      const auto mf_storage_on_level = mg_matrices[level].get_matrix_free();
      const auto patch_storage = build_patch_storage<value_type_mg>(level, mf_storage_on_level);

      // *** setup Schwarz preconditioner
      typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
      precondition_data.relaxation       = schwarz_data.damping_factor;
      precondition_data.local_relaxation = schwarz_data.local_damping_factor;
      precondition_data.symmetrized      = schwarz_data.symmetrize_smoothing;
      const auto schwarz_preconditioner  = std::make_shared<SCHWARZ_PRECONDITIONER>();
      schwarz_preconditioner->initialize(patch_storage, mg_matrices[level], precondition_data);
      mg_schwarz_precondition[level] = schwarz_preconditioner; // book-keeping

      // *** setup Schwarz smoother
      typename SCHWARZ_SMOOTHER::AdditionalData mg_smoother_data;
      mg_smoother_data.number_of_smoothing_steps = schwarz_data.number_of_smoothing_steps;
      mg_schwarz_smoother.smoothers[level].initialize(
        mg_matrices[level],
        schwarz_preconditioner,
        mg_smoother_data); // actual initialization of Schwarz smoother within
                           // MGSmootherRelaxation
    }
    mg_smoother_pre = &mg_schwarz_smoother;

    // *** initialize post Schwarz smoother
    if(parameters.mg_smoother_post_reversed)
    {
      const auto mg_schwarz_post =
        std::make_shared<MGSmootherRelaxation<LEVEL_MATRIX, SCHWARZ_SMOOTHER, VECTOR>>();
      mg_schwarz_post->initialize(mg_matrices, dummy_data);
      for(unsigned int level = mg_matrices.min_level(); level < mg_matrices.max_level() + 1;
          ++level)
      {
        const auto & smoother               = mg_schwarz_smoother.smoothers[level];
        const auto & schwarz_preconditioner = smoother.get_preconditioner();
        typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data =
          schwarz_preconditioner.get_additional_data();
        precondition_data.reverse              = true;
        const auto schwarz_preconditioner_copy = std::make_shared<SCHWARZ_PRECONDITIONER>();
        schwarz_preconditioner_copy->initialize(schwarz_preconditioner, precondition_data);
        const auto & smoother_data = smoother.get_additional_data();
        mg_schwarz_post->smoothers[level].initialize(mg_matrices[level],
                                                     schwarz_preconditioner_copy,
                                                     smoother_data);
      }
      this->mg_schwarz_smoother_post = mg_schwarz_post;
    }

    // *** set post smoother
    if(mg_schwarz_smoother_post)
      mg_smoother_post = mg_schwarz_smoother_post.get();
    else
      mg_smoother_post = &mg_schwarz_smoother;

    // *** initialize coarse grid solver
    const auto dofh0 = get_dof_handlers().front();
    coarse_control_exact.set_max_steps(dofh0->n_dofs() * n_components);
    // coarse_control_exact.set_reduction(1.e-12);
    coarse_control_exact.set_tolerance(1.e-12);
    coarse_control_exact.log_history(false);
    coarse_control_exact.log_result(false);
    coarse_solver.set_control(coarse_control_exact);
    coarse_solver.select("cg");
    auto mg_coarse =
      std::make_shared<MGCoarseGridIterativeSolver<VECTOR,
                                                   decltype(coarse_solver),
                                                   LEVEL_MATRIX,
                                                   decltype(coarse_preconditioner)>>();
    mg_coarse->initialize(coarse_solver, mg_matrices[mg_level_min], coarse_preconditioner);
    mg_coarse_grid = mg_coarse;

    mg_matrix_wrapper.initialize(mg_matrices);
    multigrid = std::make_shared<Multigrid<VECTOR>>(mg_matrix_wrapper,
                                                    *mg_coarse_grid,
                                                    mg_transfer,
                                                    *mg_smoother_pre,
                                                    *mg_smoother_post,
                                                    mg_level_min,
                                                    mg_level_max);

    if(print_details)
      pcout << " ... prepared multigrid with " << mf_storage->n_components() << " components\n";
  }

  void
  prepare_preconditioner_mg(const bool print_details = false)
  {
    prepare_multigrid(print_details);

    AssertThrow(multigrid, ExcNotInitialized());
    preconditioner_mg =
      std::make_shared<PreconditionMG<dim, VECTOR, decltype(mg_transfer)>>(get_dof_handlers(),
                                                                           *multigrid,
                                                                           mg_transfer);

    if(print_details)
      pcout << " ... prepared preconditioner\n";
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

    solver_control.set_max_steps(parameters.solver_max_iterations);
    solver_control.set_reduction(parameters.solver_reduction);
    solver_control.set_tolerance(1.e-14);
    solver_control.log_history(true);
    solver_control.log_result(true);
    solver_control.enable_history_data();

    iterative_solver.set_control(solver_control);
    iterative_solver.select(Laplace::Parameter::str_solver_variant[(int)parameters.solver_variant]);
    iterative_solver.solve(system_matrix, system_u, system_rhs, preconditioner);
    time.stop();

    const auto n_frac_and_reduction_rate = compute_fractional_steps(solver_control);
    pp_data.average_reduction_system.push_back(n_frac_and_reduction_rate.second);
    pp_data.n_iterations_system.push_back(n_frac_and_reduction_rate.first);
    pp_data.solve_time.push_back(time.wall_time());

    if(print_details)
    {
      print_row_variable(
        pcout, 2, "", 43, "Solver's reduction rate:", n_frac_and_reduction_rate.second);
      print_row_variable(
        pcout, 2, "", 43, "Number of fractional steps:", n_frac_and_reduction_rate.first);
      pcout << " ... solved system with the subsequent setup: \n\n";
      pcout << " CAUTION CAUTION CAUTION \n\n\n"
            << "Coarse-Grid Solver actually differs from the setup shown below:\n"
            << "exact, unprec. CG with 10^-12 tolerance is used!\n"
            << "\n\n\n CAUTION CAUTION CAUTION \n\n";
      parameters.print(pcout);
    }
  }

  void
  print_dof_info(const DoFHandler<dim> & dof_handler) const
  {
    print_row_variable(pcout, 2, "", 43, "Finite element:", fe->get_name());
    pcout << std::endl;

    print_row_variable(pcout, 5, "", 15, "level:", 15, "n_dofs:");
    for(unsigned l = parameters.coarse_level; l < triangulation.n_levels(); ++l)
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
  run(const bool print_details_in = false)
  {
    for(unsigned cycle = 0; cycle < parameters.n_cycles; ++cycle)
    {
      // const bool is_last_cycle = cycle == parameters.n_cycles - 1;
      // const bool print_details = print_details_in && is_last_cycle;
      const bool print_details = print_details_in;

      // if(cycle > 0)
      const bool is_tria_valid = create_triangulation(parameters.n_refines + cycle, print_details);
      if(!is_tria_valid)
        continue;
      pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());

      distribute_dofs(print_details);
      prepare_system(print_details);

      const auto str_preconditioner =
        parameters.str_precondition_variant(parameters.precondition_variant);
      print_row_variable(pcout, 2, "", 43, "Preconditioner:", str_preconditioner);
      switch(parameters.precondition_variant)
      {
        case Laplace::Parameter::PreconditionVariant::ID:
        {
          solve(preconditioner_id, print_details);
          break;
        }
        case Laplace::Parameter::PreconditionVariant::MG:
        {
          prepare_preconditioner_mg(print_details);
          solve(*preconditioner_mg, print_details);
          break;
        }
        default:
          AssertThrow(false, ExcNotImplemented());
      }

      compute_discretization_errors(print_details);
      if(parameters.write_fe_output)
        write_graphical_output();
    }
    print_convergence_table(pp_data);
  }
}; // namespace LinElasticity

} // end namespace LinElasticity

#endif /* TESTS_LINELASTICITYPROBLEM_H_ */
