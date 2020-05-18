#include "ct_parameter.h"
#include "stokes_problem.h"



int
main()
{
  try
  {
    using namespace Stokes;

    const int dim    = CT::DIMENSION_;
    const int degree = CT::FE_DEGREE_;
    Assert(degree >= 1, ExcMessage("at least linear elements for pressure"));
    // options for SolverVariant: UMFPACK FGMRES_ILU FGMRES_GMG

    RT::Parameter prms;

    EquationData equation_data;
    equation_data.solver_variant = EquationData::SolverVariant::UMFPACK;
    ModelProblem<dim, degree> flow_problem(prms, equation_data);

    flow_problem.run();
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
