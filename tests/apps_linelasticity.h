/*
 * Test the linear elasticity model problem in linelasticity_problem.h.
 *
 *  Created on: Nov 26, 2019
 *      Author: witte
 */

#ifndef TESTS_LINELASTICITY_H_
#define TESTS_LINELASTICITY_H_

#include <deal.II/base/utilities.h>

#include "linelasticity_problem.h"

using namespace dealii;
using namespace LinElasticity;

template<int dim, int fe_degree, typename value_type = double>
struct BasicSetup
{
  virtual void
  initialize()
  {
    TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;

    //: misc
    rt_parameters.n_cycles = 3;

    //: discretization
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = 4 - dim;
    rt_parameters.mesh.n_repetitions    = 2;

    //: multigrid
    const double damping_factor = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    rt_parameters.multigrid.coarse_level = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant =
      CoarseGridParameter::SolverVariant::Iterative;
    rt_parameters.multigrid.coarse_grid.iterative_solver = "cg";
    rt_parameters.multigrid.coarse_grid.accuracy         = 1.e-12;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping_factor;
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps            = 1;
    rt_parameters.multigrid.pre_smoother.schwarz.symmetrize_smoothing = false;
    rt_parameters.multigrid.pre_smoother.schwarz.reverse_smoothing    = false;
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.multigrid.post_smoother.schwarz.reverse_smoothing = true;

    //: solver
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
    rt_parameters.solver.rel_tolerance        = 1.e-8;
    rt_parameters.reset_solver_variant();
  }

  RT::Parameter rt_parameters;
};



template<int dim,
         int fe_degree,
         typename Number     = double,
         typename MatrixType = Tensors::BlockMatrixDiagonal<dim, VectorizedArray<Number>>>
struct TestLinElasticity : public BasicSetup<dim, fe_degree, Number>
{
  using Base = BasicSetup<dim, fe_degree, Number>;
  using Base::rt_parameters;

  void
  initialize() override
  {
    Base::initialize();

    rt_parameters.multigrid.pre_smoother.n_smoothing_steps  = 1;
    rt_parameters.multigrid.post_smoother.n_smoothing_steps = 1;
    rt_parameters.mesh.n_refinements                        = 1;
    rt_parameters.dof_limits                                = {1e3, 1e6};
    rt_parameters.n_cycles                                  = 10;
  }

  void
  run()
  {
    using LinElasticityOperator =
      typename LinElasticity::ModelProblem<dim, fe_degree, Number, MatrixType>;

    std::ofstream ofs("apps_linelasticity.log", std::ios_base::app);
    const bool    is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const auto    pcout         = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    equation_data.ip_factor = 1.;
    LinElasticityOperator linelasticity_problem(*pcout, rt_parameters, equation_data);
    linelasticity_problem.run(true);

    pp_data = linelasticity_problem.pp_data;
    ofs.close();
  }

  EquationData    equation_data;
  PostProcessData pp_data;
};

#endif // TESTS_LINELASTICITY_H_
