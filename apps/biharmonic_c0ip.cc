#include "biharmonic_problem.h"
#include "ct_parameter.h"

int
main(int argc, char * argv[])
{
  try
  {
    using namespace dealii;
    using namespace Biharmonic;

    deallog.depth_console(3);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    constexpr int                    dim       = CT::DIMENSION_;
    constexpr int                    fe_degree = CT::FE_DEGREE_;
    Assert(fe_degree >= 2,
           ExcMessage("The C0IP formulation for the biharmonic problem "
                      "only works if one uses elements of polynomial "
                      "degree at least 2."));

    RT::Parameter prms;
    // 0: direct solver
    // 1: CG solver (no preconditioner)
    // 2: CG solver (GMG preconditioner without smoothing)
    // 3: CG solver (GMG preconditioner with symm. Gauss-Seidel smoothing)
    constexpr unsigned int test_index_max = 3;
    unsigned int           test_index     = 0;
    if(argc > 1)
      test_index = std::atoi(argv[1]);
    AssertThrow(test_index <= test_index_max, ExcMessage("test_index is not valid"));
    unsigned int n_cycles = 1;
    if(argc > 2)
      n_cycles = std::atoi(argv[2]);
    {
      //: discretization
      prms.n_cycles = n_cycles;
      // prms.dof_limits            = {1e5, 2e7};
      prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
      prms.mesh.n_refinements    = 1;
      prms.mesh.n_repetitions    = 2;

      //: solver
      prms.solver.variant              = test_index == 0 ? "direct" : "cg";
      prms.solver.rel_tolerance        = 1.e-8;
      prms.solver.precondition_variant = test_index >= 2 ?
                                           SolverParameter::PreconditionVariant::GMG :
                                           SolverParameter::PreconditionVariant::None;
      prms.solver.n_iterations_max = 100;

      //: multigrid
      const double damping_factor =
        TPSS::lookup_damping_factor(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_, dim);
      prms.multigrid.coarse_level               = 0;
      prms.multigrid.coarse_grid.solver_variant = CoarseGridParameter::SolverVariant::IterativeAcc;
      prms.multigrid.coarse_grid.iterative_solver                                   = "cg";
      prms.multigrid.coarse_grid.accuracy                                           = 1.e-12;
      const SmootherParameter::SmootherVariant smoother_variant[test_index_max + 1] = {
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::None,
        SmootherParameter::SmootherVariant::GaussSeidel};
      prms.multigrid.pre_smoother.variant                  = smoother_variant[test_index];
      prms.multigrid.pre_smoother.n_smoothing_steps        = 2;
      prms.multigrid.pre_smoother.schwarz.patch_variant    = CT::PATCH_VARIANT_;
      prms.multigrid.pre_smoother.schwarz.smoother_variant = CT::SMOOTHER_VARIANT_;
      // prms.multigrid.pre_smoother.schwarz.manual_coloring      = true;
      prms.multigrid.pre_smoother.schwarz.damping_factor = damping_factor;
      // prms.multigrid.pre_smoother.schwarz.n_q_points_surrogate = std::min(5, fe_degree + 1);
      prms.multigrid.post_smoother                           = prms.multigrid.pre_smoother;
      prms.multigrid.post_smoother.schwarz.reverse_smoothing = true;
    }

    ModelProblem<dim> biharmonic_problem(prms, fe_degree);
    biharmonic_problem.run();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
