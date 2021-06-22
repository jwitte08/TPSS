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
      Util::ConditionalAtoi(argc, argv)(prm, index);
    };
    const auto atof_if = [&](auto & prm, const int index) {
      Util::ConditionalAtof(argc, argv)(prm, index);
    };

    //: default
    unsigned int            test_index                  = 5; // CG_GMG
    unsigned int            debug_depth                 = 0;
    double                  damping                     = 0.;
    unsigned int            force_mean_value_constraint = false;
    double                  ip_factor                   = 1.;
    unsigned int            n_cycles                    = 3;
    unsigned int            local_solver_index          = 0;
    unsigned int            pde_index                   = 6; // NoSlipExp
    int                     n_threads_max               = 1;
    types::global_dof_index dof_limit_min               = 1e1;
    types::global_dof_index dof_limit_max               = 1e5;
    double                  local_stability_factor      = 1.;

    //: parse arguments
    atoi_if(test_index, 1);
    atoi_if(pde_index, 2);
    atoi_if(n_cycles, 3);
    atoi_if(dof_limit_min, 4);
    atoi_if(dof_limit_max, 5);
    atoi_if(force_mean_value_constraint, 6);
    atoi_if(local_solver_index, 7);
    atof_if(local_stability_factor, 8);
    atoi_if(debug_depth, 9);
    atof_if(damping, 10);
    atoi_if(n_threads_max, 11);

    deallog.depth_console(debug_depth);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                        argv,
                                                        n_threads_max == -1 ?
                                                          numbers::invalid_unsigned_int :
                                                          static_cast<unsigned int>(n_threads_max));

    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0U;
    const auto n_mpi_procs   = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    StokesFlow     options;
    constexpr auto dim              = CT::DIMENSION_;
    constexpr auto fe_degree_p      = CT::FE_DEGREE_;
    constexpr auto patch_variant    = CT::PATCH_VARIANT_;
    constexpr auto smoother_variant = CT::SMOOTHER_VARIANT_;

    if(damping == 0.)
    {
      if(test_index == 2U) // Gauss Seidel
        damping = n_mpi_procs == 1 ? 1. : 0.7;
      else
        damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    }

    options.setup(test_index, damping / local_stability_factor);
    options.prms.n_cycles   = n_cycles;
    options.prms.dof_limits = {dof_limit_min, dof_limit_max};

    EquationData equation_data;
    AssertThrow(pde_index < EquationData::n_variants,
                ExcMessage("This equation is not implemented."));
    equation_data.variant           = static_cast<EquationData::Variant>(pde_index);
    equation_data.use_cuthill_mckee = false;
    if(options.prms.solver.variant == "GMRES_GMG" || options.prms.solver.variant == "CG_GMG")
      equation_data.local_kernel_size = 1U;
    AssertThrow(force_mean_value_constraint == 0 || force_mean_value_constraint == 1,
                ExcMessage("Invalid."));
    equation_data.do_mean_value_constraint = force_mean_value_constraint;
    if(options.prms.solver.variant == "direct")
      equation_data.do_mean_value_constraint = true;
    equation_data.ip_factor    = ip_factor;
    equation_data.local_solver = static_cast<LocalSolver>(local_solver_index);

    const auto filename = get_filename(options.prms, equation_data);

    std::fstream fout;
    fout.open(filename + ".log", std::ios_base::out);

    const auto pcout = std::make_shared<ConditionalOStream>(fout, is_first_proc);

    using StokesProblem = ModelProblem<dim, fe_degree_p, Method::RaviartThomas>;
    StokesProblem stokes_problem(options.prms, equation_data);
    stokes_problem.pcout = pcout;

    *pcout << std::endl;
    stokes_problem.run();

    const auto results_as_string =
      write_ppdata_to_string(stokes_problem.pp_data, stokes_problem.pp_data_pressure);

    *pcout << std::endl << std::endl << results_as_string;

    fout.close();

    if(is_first_proc)
    {
      fout.open(filename + ".tab", std::ios_base::out);
      fout << results_as_string;
    }
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
