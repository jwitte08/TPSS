
/*
 * paper on Hermite-like polynomials
 *
 *  Created on: Jul 3, 2019
 *      Author: witte
 */

#include "Plaplace_fdss.h"
#include "ct_parameter.h"
#include "laplace_problem.h"

using namespace dealii;
using namespace Laplace;

struct TestParameter
{
  double cg_reduction         = 1.e-9;
  double coarsegrid_reduction = 1.e-5;
  double manual_coloring      = true;
  double n_smoothing_steps    = 1;
};

template<int dim, int fe_degree, typename value_type = double>
void
test(const std::shared_ptr<FiniteElement<dim>> finite_element,
     const TestParameter &                     prm = TestParameter{})
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  Parameter          parameters;

  // *** mesh and experimental parameters
  if(CT::PATCH_VARIANT_ == TPSS::PatchVariant::vertex)
  {
    parameters.coarse_level = 1;
    parameters.n_refines    = parameters.coarse_level + 1;
  }
  // parameters.geometry_variant = Parameter::GeometryVariant::PaperHermite;
  parameters.geometry_variant = Parameter::GeometryVariant::PaperHermiteAffine;
  parameters.n_cycles         = 10; // 30;

  // *** multigrid
  parameters.schwarz_smoother_data.manual_coloring           = prm.manual_coloring;
  parameters.schwarz_smoother_data.patch_variant             = CT::PATCH_VARIANT_;
  parameters.schwarz_smoother_data.smoother_variant          = CT::SMOOTHER_VARIANT_;
  parameters.schwarz_smoother_data.number_of_smoothing_steps = prm.n_smoothing_steps;
  constexpr int n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);
  parameters.mg_smoother_post_reversed     = true;
  parameters.mg_coarse_chebyshev_reduction = prm.coarsegrid_reduction;

  // *** iterative solver (cg)
  parameters.solver_variant = Parameter::SolverVariant::CG;
  // parameters.solver_variant        =
  // GlobParam::lookup_solver_variant_impl(CT::SMOOTHER_VARIANT_);
  parameters.solver_max_iterations = 100;
  parameters.solver_reduction      = prm.cg_reduction;

  // *** post process
  parameters.compute_errors = true;

  using LaplaceOperator = MatrixOperator<dim, fe_degree, value_type, n_patch_dofs_per_direction>;
  LaplaceOperator laplace_problem(pcout, parameters);
  laplace_problem.fe->= finite_element;
  pcout << "TEST: " << laplace_problem.fe->> get_name() << "\n\n";
  laplace_problem.run();

  const auto       pp_data = laplace_problem.pp_data;
  ConvergenceTable convergence_table;

  for(const auto item : pp_data.n_cells_global)
    convergence_table.add_value("cells", item);
  for(const auto item : pp_data.n_dofs_global)
    convergence_table.add_value("dofs", item);
  for(const auto item : pp_data.L2_error)
    convergence_table.add_value("cg_L2error", item);
  for(const auto item : pp_data.solve_time)
    convergence_table.add_value("cg_time", item);
  for(const auto item : pp_data.n_iterations_system)
    convergence_table.add_value("n_iter", item);
  for(const auto item : pp_data.average_reduction_system)
    convergence_table.add_value("reduction", item);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    convergence_table.set_scientific("cg_L2error", true);
    convergence_table.set_precision("cg_L2error", 3);
    convergence_table.evaluate_convergence_rates("cg_L2error",
                                                 "cells",
                                                 ConvergenceTable::reduction_rate_log2,
                                                 dim);
    convergence_table.set_scientific("reduction", true);
    convergence_table.set_precision("reduction", 3);
    convergence_table.set_scientific("cg_time", true);
    convergence_table.set_precision("cg_time", 3);

    std::fstream       fstream;
    std::ostringstream oss;
    oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_) << "_";
    if(prm.n_smoothing_steps != 1)
      oss << prm.n_smoothing_steps << "steps_";
    oss << finite_element->get_name() << ".txt";
    fstream.open(oss.str(), std::ios_base::app);
    // convergence_table.write_text(std::cout);
    ConditionalOStream pfout(fstream, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    laplace_problem.parameters.print(pfout);
    fstream << std::endl << std::endl;
    convergence_table.write_text(fstream);
    fstream.close();

    std::cout << "written to: " << oss.str() << std::endl << std::endl;
  }
}

int
main(int argc, char * argv[])
{
  // deallog.depth_console(3);

  // *** init TBB and MPI
  const unsigned int max_threads = 1; // numbers::invalid_unsigned_int; // serial
  MultithreadInfo::set_thread_limit(max_threads);
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static TEST parameters
  using value_type             = double;
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;

  unsigned int fe_type           = 0; // 0 = nodal, 1 = hermite-like
  unsigned int n_smoothing_steps = 1;
  if(argc > 1)
    fe_type = std::atoi(argv[1]);
  if(argc > 2)
    n_smoothing_steps = std::atoi(argv[2]);

  TestParameter prms;
  prms.n_smoothing_steps = n_smoothing_steps;
  if(fe_type == 0)
    test<dim, fe_degree>(std::make_shared<FE_DGQ<dim>>(fe_degree), prms);
  else if(fe_type == 1)
    test<dim, fe_degree>(std::make_shared<FE_DGQHermite<dim>>(fe_degree), prms);

  return 0;
}
