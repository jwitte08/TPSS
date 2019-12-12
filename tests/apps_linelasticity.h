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
  struct Params
  {
    TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::additive;
  };

  virtual void
  initialize()
  {
    //: misc
    rt_parameters.compressed = false;
    rt_parameters.n_cycles   = 3;

    //: discretization
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = 4 - dim;
    rt_parameters.mesh.n_repetitions    = 2;

    //: multigrid
    const double damping_factor =
      TPSS::lookup_damping_factor(params.patch_variant, params.smoother_variant, dim);
    rt_parameters.multigrid.coarse_level = 0;
    rt_parameters.multigrid.coarse_grid.solver_variant =
      CoarseGridParameter::SolverVariant::IterativeAcc;
    rt_parameters.multigrid.coarse_grid.iterative_solver = "cg";
    rt_parameters.multigrid.coarse_grid.accuracy         = 1.e-12;
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = params.patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = params.smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.manual_coloring      = true;
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
  Params        params;
};



template<int dim,
         int fe_degree,
         typename Number     = double,
         typename MatrixType = Tensors::BlockMatrixDiagonal<dim, VectorizedArray<Number>>>
struct TestLinElasticity : public BasicSetup<dim, fe_degree, Number>
{
  struct Params
  {
    TPSS::PatchVariant      patch_variant     = TPSS::PatchVariant::cell;
    TPSS::SmootherVariant   smoother_variant  = TPSS::SmootherVariant::additive;
    unsigned                n_smoothing_steps = 1;
    types::global_dof_index dof_limit_min     = 1;
    types::global_dof_index dof_limit_max     = 5e6;
    EquationData            equation_data;
  };

  using Base = BasicSetup<dim, fe_degree, Number>;
  using Base::rt_parameters;

  void
  base_initialize()
  {
    Base::params = {params.patch_variant, params.smoother_variant};
    Base::initialize();
  }

  void
  initialize() override
  {
    base_initialize();
    rt_parameters.multigrid.pre_smoother.n_smoothing_steps  = params.n_smoothing_steps;
    rt_parameters.multigrid.post_smoother.n_smoothing_steps = params.n_smoothing_steps;
    rt_parameters.mesh.n_refinements                        = 1;
    rt_parameters.dof_limits = {params.dof_limit_min, params.dof_limit_max};
    rt_parameters.n_cycles   = 10;
    // if(params.smoother_variant == TPSS::SmootherVariant::multiplicative)
    //   rt_parameters.solver.variant = "gmres";
    // params.equation_data.lambda = 20.; fails!!!
  }

  void
  run()
  {
    using LinElasticityOperator =
      typename LinElasticity::ModelProblem<dim, fe_degree, Number, MatrixType>;

    std::ofstream ofs("apps_linelasticity.log", std::ios_base::app);
    const bool    is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const auto    pcout         = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    LinElasticityOperator linelasticity_problem(*pcout, rt_parameters, params.equation_data);
    linelasticity_problem.run(true);

    pp_data = linelasticity_problem.pp_data;
    ofs.close();
  }

  Params          params;
  PostProcessData pp_data;
};

#endif // TESTS_LINELASTICITY_H_
