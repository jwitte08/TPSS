#include "biharmonic_problem.h"

int
main(int argc, char * argv[])
{
  try
  {
    using namespace dealii;
    using namespace Biharmonic;

    deallog.depth_console(3);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const unsigned int               fe_degree = 2;
    Assert(fe_degree >= 2,
           ExcMessage("The C0IP formulation for the biharmonic problem "
                      "only works if one uses elements of polynomial "
                      "degree at least 2."));

    RT::Parameter prms;
    prms.solver.variant          = "direct";
    prms.solver.variant          = "cg";
    prms.solver.n_iterations_max = 500;
    prms.solver.rel_tolerance    = 1.e-8;
    prms.n_cycles                = 1;
    // prms.multigrid.coarse_grid.solver_variant = CoarseGridParameter::SolverVariant::FullSVD;

    ModelProblem<2> biharmonic_problem(prms, fe_degree);
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
