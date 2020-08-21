/**
 * TODO...
 *
 * Created on: Aug 20, 2020
 *     Author: witte
 */



#include "stokes.h"



using namespace Stokes;

int
main(int argc, char * argv[])
{
  try
  {
    const auto atoi_if = [&](auto & prm, const int index) {
      if(index <= 0 || argc <= index)
        return;
      if(std::strcmp(argv[index], StokesFlow::skipper) == 0)
        return;
      prm = std::atoi(argv[index]);
    };
    const auto atof_if = [&](auto & prm, const int index) {
      if(index <= 0 || argc <= index)
        return;
      if(std::strcmp(argv[index], StokesFlow::skipper) == 0)
        return;
      prm = std::atof(argv[index]);
    };

    //: default
    unsigned int test_index                  = 0;
    unsigned int debug_depth                 = 0;
    double       damping                     = 0.;
    unsigned int force_mean_value_constraint = false;
    double       ip_factor                   = 0.;
    unsigned int n_cycles                    = 3;
    unsigned int local_solver_variant        = 0;

    //: parse arguments
    atoi_if(test_index, 1);
    atoi_if(ip_factor, 2);
    atoi_if(n_cycles, 3);
    atoi_if(local_solver_variant, 4);
    atoi_if(force_mean_value_constraint, 7);
    atoi_if(debug_depth, 5);
    atof_if(damping, 6);

    deallog.depth_console(debug_depth);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    StokesFlow     options;
    constexpr auto dim              = CT::DIMENSION_;
    constexpr auto fe_degree_p      = CT::FE_DEGREE_;
    constexpr auto patch_variant    = CT::PATCH_VARIANT_;
    constexpr auto smoother_variant = CT::SMOOTHER_VARIANT_;

    if(damping == 0.)
      damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);

    options.setup(test_index, damping);
    options.prms.n_cycles = n_cycles;

    EquationData equation_data;
    equation_data.variant           = EquationData::Variant::DivFreeHom; // !!!
    equation_data.use_cuthill_mckee = false;
    if(options.prms.solver.variant == "GMRES_GMG" || options.prms.solver.variant == "CG_GMG")
      equation_data.local_kernel_size = 1U;
    AssertThrow(force_mean_value_constraint == 0 || force_mean_value_constraint == 1,
                ExcMessage("Invalid."));
    equation_data.force_mean_value_constraint =
      options.prms.solver.variant == "UMFPACK" ?
        true :
        static_cast<bool>(force_mean_value_constraint); // !!!
    equation_data.ip_factor    = ip_factor;             // !!!
    equation_data.local_solver = static_cast<LocalSolver>(local_solver_variant);

    ModelProblem<dim, fe_degree_p, Method::RaviartThomas> stokes_problem(options.prms,
                                                                         equation_data); // !!!

    std::cout << std::endl;
    stokes_problem.run();

    std::cout << std::endl
              << std::endl
              << write_ppdata_to_string(stokes_problem.pp_data, stokes_problem.pp_data_pressure);
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
