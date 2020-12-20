#ifndef APPS_STOKES_H_
#define APPS_STOKES_H_

/**
 * Default settings and post-processing for Stokes flow.
 *
 * Created on: Aug 13, 2020
 *     Author: witte
 */

#include <deal.II/base/convergence_table.h>


#include "ct_parameter.h"
#include "stokes_problem.h"
#include "utilities.h"



namespace Stokes
{
template<int dimension_ = CT::DIMENSION_, int fe_degree_p_ = CT::FE_DEGREE_>
struct StokesFlow
{
  static constexpr auto dim         = dimension_;
  static constexpr auto fe_degree_p = fe_degree_p_;

  // static constexpr char const * skipper     = "o";
  // 0 : direct solver (UMFPACK)
  // 1 : flexible GMRES prec. by ILU (FGMRES_ILU)
  //     flexible GMRES prec. by Schur complement approximation ...
  // 2 : ... with GMG based on Gauss-Seidel smoothers for velocity (FGMRES_GMGvelocity)
  // 3 : ... with GMG based on Schwarz smoothers for velocity (FGMRES_GMGvelocity)
  //     GMRES prec. by GMG ...
  // 4 : ...based on Gauss-Seidel smoothers for velocity-pressure (GMRES_GMG)
  // 5 : ...based on Gauss-Seidel smoothers for velocity-pressure (GMRES_CG)
  // 6 : unpreconditioned CG (CG)
  static constexpr unsigned int test_index_max = 6;

  void
  setup(const unsigned int          test_index,
        const double                damping_factor,
        const TPSS::PatchVariant    patch_variant    = CT::PATCH_VARIANT_,
        const TPSS::SmootherVariant smoother_variant = CT::SMOOTHER_VARIANT_)
  {
    AssertThrow(test_index <= test_index_max, ExcMessage("test_index is not valid"));

    //: discretization
    prms.n_cycles              = 3;
    prms.dof_limits            = {1e1, 2e5};
    prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    prms.mesh.n_refinements    = 1;
    prms.mesh.n_repetitions    = 2;

    //: solver
    const SolverParameter::PreconditionVariant precondition_variant[test_index_max + 1] = {
      SolverParameter::PreconditionVariant::None,
      SolverParameter::PreconditionVariant::None,
      SolverParameter::PreconditionVariant::GMG,
      SolverParameter::PreconditionVariant::GMG,
      SolverParameter::PreconditionVariant::GMG,
      SolverParameter::PreconditionVariant::GMG,
      SolverParameter::PreconditionVariant::None};
    prms.solver.variant              = str_solver_variant[test_index];
    prms.solver.rel_tolerance        = 1.e-10; // !!!
    prms.solver.precondition_variant = precondition_variant[test_index];
    prms.solver.n_iterations_max     = 200;

    //: multigrid
    //:: coarse grid
    prms.multigrid.coarse_level               = 0;
    prms.multigrid.coarse_grid.solver_variant = CoarseGridParameter::SolverVariant::DirectSVD;
    /// The threshold is crucial, if we do not impose any mean value
    /// constraint on the coarse problem!
    prms.multigrid.coarse_grid.threshold_svd = 1.e-8;
    // prms.multigrid.coarse_grid.iterative_solver                                   = "cg";
    // prms.multigrid.coarse_grid.accuracy                                           = 1.e-12;

    //:: pre-smoother
    prms.multigrid.pre_smoother.variant                      = smoother_scheme[test_index];
    prms.multigrid.pre_smoother.n_smoothing_steps            = 2; // !!!
    prms.multigrid.pre_smoother.schwarz.patch_variant        = patch_variant;
    prms.multigrid.pre_smoother.schwarz.smoother_variant     = smoother_variant;
    prms.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    prms.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;

    //:: post_smoother
    prms.multigrid.post_smoother                           = prms.multigrid.pre_smoother;
    prms.multigrid.post_smoother.schwarz.reverse_smoothing = true;
  }

  const std::string str_solver_variant[test_index_max + 1] = {"UMFPACK",
                                                              "FGMRES_ILU",
                                                              "FGMRES_GMGvelocity",
                                                              "FGMRES_GMGvelocity",
                                                              "GMRES_GMG",
                                                              "CG_GMG",
                                                              "CG"};

  const SmootherParameter::SmootherVariant smoother_scheme[test_index_max + 1] = {
    SmootherParameter::SmootherVariant::None,
    SmootherParameter::SmootherVariant::None,
    SmootherParameter::SmootherVariant::GaussSeidel,
    SmootherParameter::SmootherVariant::Schwarz,
    SmootherParameter::SmootherVariant::Schwarz,
    SmootherParameter::SmootherVariant::Schwarz,
    SmootherParameter::SmootherVariant::None};

  RT::Parameter prms;
};



std::string
write_ppdata_to_string(const PostProcessData & pp_data, const PostProcessData & pp_data_pressure)
{
  std::ostringstream oss;
  ConvergenceTable   info_table;
  Assert(!pp_data.n_cells_global.empty(), ExcMessage("No cells to post process."));
  AssertDimension(pp_data.n_dofs_global.size(), pp_data_pressure.n_dofs_global.size());
  for(unsigned run = 0; run < pp_data.n_cells_global.size(); ++run)
  {
    // info_table.add_value("n_levels", pp_data.n_mg_levels.at(run));
    info_table.add_value("n_cells", pp_data.n_cells_global.at(run));
    const auto n_dofs_total =
      pp_data.n_dofs_global.at(run) + pp_data_pressure.n_dofs_global.at(run);
    info_table.add_value("n_dofs", n_dofs_total);
    info_table.add_value("n_dofs_u", pp_data.n_dofs_global.at(run));
    info_table.add_value("n_dofs_p", pp_data_pressure.n_dofs_global.at(run));
    // info_table.add_value("n_colors", pp_data.n_colors_system.at(run));
    info_table.add_value("n_iter", pp_data.n_iterations_system.at(run));
    info_table.add_value("reduction", pp_data.average_reduction_system.at(run));
    info_table.add_value("L2_error_u", pp_data.L2_error.at(run));
    info_table.add_value("L2_error_p", pp_data_pressure.L2_error.at(run));
    info_table.add_value("H1semi_error_u", pp_data.H1semi_error.at(run));
  }
  info_table.set_scientific("reduction", true);
  info_table.set_precision("reduction", 3);
  info_table.set_scientific("L2_error_u", true);
  info_table.set_precision("L2_error_u", 3);
  info_table.evaluate_convergence_rates("L2_error_u", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("L2_error_u",
                                        "n_dofs_u",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);
  info_table.set_scientific("H1semi_error_u", true);
  info_table.set_precision("H1semi_error_u", 3);
  info_table.evaluate_convergence_rates("H1semi_error_u", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("H1semi_error_u",
                                        "n_dofs_u",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);
  info_table.set_scientific("L2_error_p", true);
  info_table.set_precision("L2_error_p", 3);
  info_table.evaluate_convergence_rates("L2_error_p", ConvergenceTable::reduction_rate);
  info_table.evaluate_convergence_rates("L2_error_p",
                                        "n_dofs_p",
                                        ConvergenceTable::reduction_rate_log2,
                                        pp_data.n_dimensions);

  info_table.write_text(oss);
  return oss.str();
}

} // namespace Stokes

#endif
