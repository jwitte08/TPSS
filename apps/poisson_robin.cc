
/*
 * sandbox poisson problem
 *
 *  Created on: Oct 30, 2019
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "ct_parameter.h"
#include "poisson_problem.h"
#include "utilities.h"

using namespace dealii;
using namespace Laplace;



std::string
write_ppdata_to_string(const PostProcessData & pp_data)
{
  std::ostringstream oss;
  ConvergenceTable   info_table;
  Assert(!pp_data.n_cells_global.empty(), ExcMessage("No cells to post process."));
  for(unsigned run = 0; run < pp_data.n_dofs_global.size(); ++run)
  {
    // info_table.add_value("n_levels", pp_data.n_mg_levels.at(run));
    // info_table.add_value("n_cells", pp_data.n_cells_global.at(run));
    info_table.add_value("n_dofs", pp_data.n_dofs_global.at(run));
    // info_table.add_value("n_colors", pp_data.n_colors_system.at(run));
    // info_table.add_value("n_iter", pp_data.n_iterations_system.at(run));
    info_table.add_value("reduction", pp_data.average_reduction_system.at(run));
    info_table.add_value("L2_error", pp_data.L2_error.at(run));
  }
  info_table.set_scientific("reduction", true);
  info_table.set_precision("reduction", 3);
  info_table.set_scientific("L2_error", true);
  info_table.set_precision("L2_error", 3);
  info_table.evaluate_convergence_rates("L2_error", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("L2_error",
                                        "n_dofs",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);

  info_table.write_text(oss);
  return oss.str();
}



int
main(int argc, char * argv[])
{
  const auto atoi_if = [&](auto & prm, const int index) {
    Util::ConditionalAtoi(argc, argv)(prm, index);
  };
  const auto atof_if = [&](auto & prm, const int index) {
    Util::ConditionalAtof(argc, argv)(prm, index);
  };

  const unsigned int solver_index_max                     = 1;
  const std::string  solver_variant[solver_index_max + 1] = {"cg", "gmres"};

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  constexpr int                    dim        = CT::DIMENSION_;
  constexpr int                    fe_degree  = CT::FE_DEGREE_;
  constexpr auto                   dof_layout = CT::DOF_LAYOUT_;
  constexpr int                    n_patch_dofs_1d_static =
    TPSS::UniversalInfo<dim>::n_dofs_1d(CT::PATCH_VARIANT_, dof_layout, fe_degree);

  //: default
  unsigned int solver_index    = 0;
  unsigned int use_ras_weights = 0;
  double       damping         = 0.;
  unsigned int n_cycles        = numbers::invalid_unsigned_int;

  //: parse runtime arguments
  atoi_if(solver_index, 1);
  atoi_if(use_ras_weights, 2);
  atof_if(damping, 3);
  atoi_if(n_cycles, 4);

  AssertThrow(solver_index <= solver_index_max, ExcMessage("Invalid solver index."));
  AssertThrow(use_ras_weights <= 1U, ExcMessage("Invalid integer value for use_ras_weights."));
  AssertThrow(damping <= 1., ExcMessage("No over-relaxation allowed."));

  RT::Parameter rt_parameters;
  {
    //: discretization
    rt_parameters.n_cycles              = n_cycles == numbers::invalid_unsigned_int ? 2 : n_cycles;
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = 0;
    rt_parameters.mesh.n_repetitions    = 2;
    rt_parameters.mesh.do_colorization  = true;

    //: solver
    rt_parameters.solver.variant              = solver_variant[solver_index];
    rt_parameters.solver.rel_tolerance        = 1.e-8;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::None;
    rt_parameters.solver.n_iterations_max     = 500;

    //: multigrid
    // if(damping == 0.)
    //   damping =
    //     TPSS::lookup_damping_factor(testprms.patch_variant, testprms.smoother_variant, dim);
    // rt_parameters.multigrid.coarse_level                 = 0;
    // rt_parameters.multigrid.coarse_grid.solver_variant   = testprms.coarse_grid_variant;
    // rt_parameters.multigrid.coarse_grid.iterative_solver = testprms.solver_variant;
    // rt_parameters.multigrid.coarse_grid.accuracy         = testprms.coarse_grid_accuracy;
    // rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    // rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = testprms.patch_variant;
    // rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     =
    // testprms.smoother_variant; rt_parameters.multigrid.pre_smoother.schwarz.use_ras_weights =
    // testprms.use_ras_weights; // !!!
    // rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true; std::cout << "damp:
    // " << damping << std::endl; rt_parameters.multigrid.pre_smoother.schwarz.damping_factor =
    // damping; rt_parameters.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(5,
    // fe_degree + 1); rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    // rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;
  }

  EquationData equation_data;
  // equation_data.neumann_boundary_ids = {0};
  for(types::boundary_id id = 0; id < GeometryInfo<dim>::faces_per_cell; ++id)
    equation_data.neumann_boundary_ids.insert(id);
  equation_data.dirichlet_boundary_ids.clear();
  for(types::boundary_id id = 0; id < GeometryInfo<dim>::faces_per_cell; ++id)
    equation_data.dirichlet_boundary_ids.insert(id);

  // const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  // std::shared_ptr<ConditionalOStream> pcout;
  // pcout = std::make_shared<ConditionalOStream>(std::cout, is_first_proc);

  using PoissonProblem =
    typename Poisson::ModelProblem<dim, fe_degree, dof_layout, double, n_patch_dofs_1d_static>;

  PoissonProblem poisson_problem(rt_parameters, equation_data);
  poisson_problem.run();

  *poisson_problem.pcout << "\n\n" << write_ppdata_to_string(poisson_problem.pp_data);

  return 0;
}
