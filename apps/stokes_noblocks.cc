/**
 * TODO...
 *
 * Created on: Jan 30, 2021
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
    unsigned int test_index                  = 0;
    unsigned int debug_depth                 = 0;
    double       damping                     = 0.;
    unsigned int force_mean_value_constraint = false;
    double       ip_factor                   = 1.;
    unsigned int n_cycles                    = 3;
    unsigned int local_solver_variant        = 0;
    unsigned int pde_index                   = 4; // NoSlip

    //: parse arguments
    atoi_if(test_index, 1);
    atoi_if(ip_factor, 2);
    atoi_if(n_cycles, 3);
    atoi_if(pde_index, 4);
    atoi_if(debug_depth, 5);
    atof_if(damping, 6);
    atoi_if(force_mean_value_constraint, 7);
    atoi_if(local_solver_variant, 8);

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
    /// each side of the rectangular domain needs its own boundary_id (otherwise
    /// MGConstrainedDoFs::make_no_normal_zero_flux() is not supported)
    options.prms.mesh.do_colorization    = true;  // !!!
    options.prms.solver.rel_tolerance    = 1.e-8; // !!!
    options.prms.solver.n_iterations_max = 5000;  // !!!

    EquationData equation_data;
    AssertThrow(pde_index < EquationData::n_variants,
                ExcMessage("This equation is not implemented."));
    equation_data.variant           = static_cast<EquationData::Variant>(pde_index);
    equation_data.use_cuthill_mckee = false;
    if(options.prms.solver.variant == "GMRES_GMG" || options.prms.solver.variant == "CG_GMG")
      equation_data.local_kernel_size = 1U;
    AssertThrow(force_mean_value_constraint == 0 || force_mean_value_constraint == 1,
                ExcMessage("Invalid."));
    equation_data.force_mean_value_constraint = force_mean_value_constraint;
    if(options.prms.solver.variant == "direct")
      equation_data.force_mean_value_constraint = true;
    equation_data.ip_factor    = ip_factor;
    equation_data.local_solver = static_cast<LocalSolver>(local_solver_variant);
    if(options.prms.mesh.do_colorization)
      for(types::boundary_id id = 0; id < GeometryInfo<dim>::faces_per_cell; ++id)
        equation_data.dirichlet_boundary_ids_velocity.insert(id);

    typename NoBlocks::ModelProblem<dim, fe_degree_p, Method::RaviartThomas> stokes_problem(options.prms,
                                                                         equation_data);

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
