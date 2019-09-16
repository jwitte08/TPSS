/*
 * mlaplace_problem.h
 *
 * DG Mlaplace problem used to test Schwarz smoothers.
 *
 *  Created on: Apr 28, 2019
 *      Author: witte
 */

#ifndef TESTS_MLAPLACEPROBLEM_H_
#define TESTS_MLAPLACEPROBLEM_H_

using namespace dealii;

#include "solvers_and_preconditioners/preconditioner/preconditioner_base.h"
#include "solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"

#include "laplace_problem.h"
#include "mlaplace_integrator.h"

namespace MultiLaplace
{
template<int dim, int fe_degree, int n_comps = 1, typename Number = double, int n_patch_dofs = -1>
struct MatrixOperator : public Subscriptor
{
  using VECTOR        = typename LinearAlgebra::distributed::BlockVector<Number>;
  using SYSTEM_MATRIX = MultiLaplace::MF::Operator<dim, fe_degree, Number>;

  using value_type_mg = Number;
  using LEVEL_MATRIX  = MultiLaplace::CombinedOperator<dim, fe_degree, value_type_mg>;
  using MG_TRANSFER   = MGTransferMatrixFree<dim, value_type_mg>;
  using PATCH_MATRIX  = Tensors::BlockMatrixDiagonal<dim, VectorizedArray<Number>, n_patch_dofs>;
  using SCHWARZ_PRECONDITIONER = SchwarzPreconditioner<dim, LEVEL_MATRIX, VECTOR, PATCH_MATRIX>;
  using SCHWARZ_SMOOTHER       = SchwarzSmoother<dim, LEVEL_MATRIX, SCHWARZ_PRECONDITIONER, VECTOR>;

  Laplace::MatrixOperator<dim, fe_degree, Number, n_patch_dofs> laplace_problem;
  Laplace::Parameter                                            parameters;
  ConditionalOStream &                                          pcout;

  parallel::distributed::Triangulation<dim> &           triangulation;
  std::array<std::shared_ptr<DoFHandler<dim>>, n_comps> dof_handlers;
  FiniteElement<dim> &                                  fe;
  const MappingQGeneric<dim> &                          mapping;
  // Laplace::ManualColoring<dim>                          make_manual_coloring;
  // Laplace::NonOverlappingVertexPatch<dim>               make_non_overlapping_vertex_patch;

  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  SYSTEM_MATRIX                                  system_matrix;
  VECTOR                                         system_u;
  VECTOR                                         system_rhs;
  ReductionControl                               solver_control;
  SolverSelector<VECTOR>                         iterative_solver;

  MGLevelObject<LEVEL_MATRIX>                                  mg_matrices;
  MGTransferBlockMatrixFree<dim, value_type_mg>                mg_transfer;
  MGLevelObject<std::shared_ptr<const SCHWARZ_PRECONDITIONER>> mg_schwarz_precondition;
  MGSmootherRelaxation<LEVEL_MATRIX, SCHWARZ_SMOOTHER, VECTOR> mg_schwarz_smoother;
  const MGSmootherBase<VECTOR> *                               mg_smoother_pre;
  const MGSmootherBase<VECTOR> *                               mg_smoother_post;
  ReductionControl                                             coarse_control_exact;
  PreconditionIdentity                                         coarse_preconditioner;
  SolverSelector<VECTOR>                                       coarse_solver;
  std::shared_ptr<const MGCoarseGridBase<VECTOR>>              mg_coarse_grid;

  mg::Matrix<VECTOR>                                                        mg_matrix_wrapper;
  std::shared_ptr<Multigrid<VECTOR>>                                        multigrid;
  std::shared_ptr<const PreconditionMG<dim, VECTOR, decltype(mg_transfer)>> preconditioner_mg;

  PreconditionIdentity preconditioner_id;

  mutable std::array<Laplace::PostProcessData, n_comps> pp_data;

  MatrixOperator(ConditionalOStream & pcout_in, const Laplace::Parameter & parameters_in)
    : laplace_problem(pcout_in, parameters_in),
      parameters(laplace_problem.parameters),
      pcout(pcout_in),
      triangulation(laplace_problem.triangulation),
      fe(*(laplace_problem.fe)),
      mapping(laplace_problem.mapping),
      mg_transfer(dof_handlers.size()),
      mg_smoother_pre(nullptr),
      mg_smoother_post(nullptr)
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

  template<typename Number2>
  std::shared_ptr<const MatrixFree<dim, Number2>>
  build_mf_storage(const unsigned int level = static_cast<unsigned>(-1))
  {
    using AddData = typename MatrixFree<dim, Number2>::AdditionalData;

    AffineConstraints<double> constraints_dummy;
    constraints_dummy.close();
    std::vector<const AffineConstraints<double> *> constraints_dummies;
    std::fill_n(std::back_inserter(constraints_dummies), n_comps, &constraints_dummy);
    const auto mf_storage = std::make_shared<MatrixFree<dim, Number2>>();

    unsigned int n_qpoints = fe_degree + 1;
    QGauss<1>    quadrature(n_qpoints);

    AddData    additional_data;
    const auto mapping_update_flags =
      dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points;
    additional_data.mapping_update_flags                = mapping_update_flags;
    additional_data.mapping_update_flags_inner_faces    = mapping_update_flags;
    additional_data.mapping_update_flags_boundary_faces = mapping_update_flags;
    if(level != static_cast<unsigned>(-1))
      additional_data.level_mg_handler = level;
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

  void
  create_triangulation(const unsigned n_refines, const bool print_details = false)
  {
    laplace_problem.create_triangulation(n_refines, print_details);
  }

  void
  create_triangulation_finest(const bool print_details = false)
  {
    create_triangulation(parameters.n_refines, print_details);
  }

  void
  distribute_dofs(const bool print_details = false)
  {
    unsigned int component = 0;
    for(auto & dofh : dof_handlers)
    {
      dofh = std::make_shared<DoFHandler<dim>>(triangulation);
      dofh->initialize(triangulation, fe);
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
    AssertThrow(mf_storage, ExcMessage("Not initialized."));
    laplace_problem.setup(/*build_rhs*/ true);
    for(unsigned int comp = 0; comp < n_comps; ++comp)
    {
      mf_storage->initialize_dof_vector(system_u.block(comp), comp);
      mf_storage->initialize_dof_vector(system_rhs.block(comp), comp);
      system_rhs.block(comp) = laplace_problem.system_rhs;
    }
    system_u.collect_sizes();
    system_rhs.collect_sizes();
    if(print_details)
      pcout << " ... RHS computed" << std::endl;
  }

  void
  prepare_system(const bool print_details = false)
  {
    // *** system matrix
    this->mf_storage = build_mf_storage<Number>();
    system_matrix.initialize(mf_storage, n_comps);

    // *** solution vector
    system_u.reinit(n_comps);

    // *** right hand side vector
    system_rhs.reinit(n_comps);
    compute_rhs(print_details);

    if(print_details)
      pcout << " ... prepared system with " << mf_storage->n_components() << " components\n";
  }

  void
  prepare_multigrid(const bool print_details = false)
  {
    const unsigned int global_level = laplace_problem.global_level;

    // *** setup multigrid data
    const unsigned mg_level_min = parameters.coarse_level;
    const unsigned mg_level_max = global_level;

    // *** initialize level matrices A_l
    mg_matrices.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
    {
      const auto mf_storage_level = build_mf_storage<value_type_mg>(level);
      mg_matrices[level].initialize(mf_storage_level, n_comps);
    }

    // *** initialize multigrid transfer R_l
    mg_transfer.build(get_dof_handlers());

    // *** initialize (Schwarz) smoother S_l
    const auto &                              schwarz_data = parameters.schwarz_smoother_data;
    typename SCHWARZ_SMOOTHER::AdditionalData dummy_data;
    mg_schwarz_smoother.initialize(mg_matrices,
                                   dummy_data); // insert level matrices A_l in MGSmootherRelaxation
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

      // *** setup smoother
      typename SCHWARZ_SMOOTHER::AdditionalData mg_smoother_data;
      mg_smoother_data.number_of_smoothing_steps = schwarz_data.number_of_smoothing_steps;
      mg_schwarz_smoother.smoothers[level].initialize(
        mg_matrices[level],
        schwarz_preconditioner,
        mg_smoother_data); // actual initialization of Schwarz smoother within
                           // MGSmootherRelaxation
    }
    mg_smoother_pre  = &mg_schwarz_smoother;
    mg_smoother_post = &mg_schwarz_smoother;

    // *** initialize coarse grid
    const auto dofh0 = get_dof_handlers().front();
    coarse_control_exact.set_max_steps(dofh0->n_dofs() * n_comps);
    coarse_control_exact.set_reduction(1.e-12);
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

  template<typename PreconditionerType>
  void
  solve(const PreconditionerType & preconditioner, const bool print_details = false)
  {
    solver_control.set_max_steps(parameters.solver_max_iterations);
    solver_control.set_reduction(parameters.solver_reduction);
    solver_control.set_tolerance(1.e-12);
    solver_control.log_history(true);
    solver_control.log_result(true);
    solver_control.enable_history_data();

    iterative_solver.set_control(solver_control);
    iterative_solver.select(Laplace::Parameter::str_solver_variant[(int)parameters.solver_variant]);
    iterative_solver.solve(system_matrix, system_u, system_rhs, preconditioner);

    const auto n_iterations_frac = laplace_problem.compute_n_iterations_fractional(solver_control);
    pp_data[0].average_reduction_system.push_back(solver_control.average_reduction());
    pp_data[0].n_iterations_system.push_back(n_iterations_frac);

    if(print_details)
    {
      pcout << " ... solved system with the subsequent setup: \n\n";
      parameters.print(pcout);
    }
  }

  void
  print_dof_info(const DoFHandler<dim> & dof_handler) const
  {
    print_row_variable(pcout, 2, "", 43, "Finite element:", fe.get_name());
    pcout << std::endl;

    print_row_variable(pcout, 5, "", 15, "level:", 15, "n_dofs:");
    for(unsigned l = parameters.coarse_level; l < triangulation.n_levels(); ++l)
      print_row_variable(pcout, 5, "", 15, l, 15, dof_handler.n_dofs(l));
    pcout << std::endl;
  }

  void
  compute_discretization_errors(const unsigned int component = 0) const
  {
    const auto & dof_handler       = *(dof_handlers[component]);
    const auto & discrete_solution = system_u.block(component);
    const auto & fe                = dof_handler.get_fe();
    auto &       pp_data           = this->pp_data[component];
    pp_data.n_dofs_global.push_back(dof_handler.n_dofs());

    // *** L2 error (first component)
    Vector<float> error_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      discrete_solution,
                                      Laplace::Solution<dim>{},
                                      error_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    const double L2_error =
      std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));
    pp_data.L2_error.push_back(L2_error);

    // *** H1semi error (first component)
    Vector<float> error_per_cell2(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      discrete_solution,
                                      Laplace::Solution<dim>{},
                                      error_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::H1_seminorm);
    const double H1semi_error =
      std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));
    pp_data.H1semi_error.push_back(H1semi_error);
  }

  void
  print_convergence_table(const Laplace::PostProcessData & pp_data)
  {
    ConvergenceTable convergence_table;
    AssertThrow(pp_data.L2_error.size() == pp_data.n_dofs_global.size(),
                ExcMessage("Mismatch of error container in PostProcessData"));
    AssertThrow(pp_data.H1semi_error.size() == pp_data.n_dofs_global.size(),
                ExcMessage("Mismatch of error container in PostProcessData"));
    for(std::size_t t = 0; t < pp_data.L2_error.size(); ++t)
    {
      convergence_table.add_value("#DoFs", pp_data.n_dofs_global[t]);
      convergence_table.add_value("L2 error", pp_data.L2_error[t]);
      convergence_table.add_value("H1semi error", pp_data.H1semi_error[t]);
    }

    convergence_table.set_precision("L2 error", 3);
    convergence_table.set_precision("H1semi error", 3);
    convergence_table.set_scientific("L2 error", true);
    convergence_table.set_scientific("H1semi error", true);

    convergence_table.evaluate_convergence_rates("L2 error", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("L2 error",
                                                 "#DoFs",
                                                 ConvergenceTable::reduction_rate_log2,
                                                 dim);
    convergence_table.evaluate_convergence_rates("H1semi error", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("H1semi error",
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
  run(const bool print_details_in = false)
  {
    for(unsigned cycle = 0; cycle < parameters.n_cycles; ++cycle)
    {
      const bool is_last_cycle = cycle == parameters.n_cycles - 1;
      const bool print_details = print_details_in && is_last_cycle;

      if(cycle > 0)
        create_triangulation(parameters.n_refines + cycle, print_details);

      distribute_dofs(print_details);
      prepare_system(print_details);

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
      }

      for(unsigned int comp = 0; comp < n_comps; ++comp)
        compute_discretization_errors(comp);
    }
    for(unsigned int comp = 0; comp < n_comps; ++comp)
      print_convergence_table(pp_data[comp]);
  }
};
} // namespace MultiLaplace

#endif /* TESTS_MLAPLACEPROBLEM_H_ */
