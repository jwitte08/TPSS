
/*
 * print graphical output of meshes
 *
 *  Created on: Aug 15, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "laplace_problem.h"

int
main(int argc, char * argv[])
{
  using namespace dealii;

  // *** init MPI (1 proc only!)
  constexpr int                    max_threads = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** test parameters
  using value_type             = double;
  constexpr unsigned dim       = 2;
  constexpr unsigned fe_degree = 2;

  Laplace::Parameter parameters;
  parameters.schwarz_smoother_data.patch_variant = TPSS::PatchVariant::cell;
  parameters.schwarz_smoother_data.smoother_variant =
    TPSS::SmootherVariant::additive; // multiplicative;
  parameters.schwarz_smoother_data.n_q_points_surrogate = 3;

  parameters.n_refines            = 1;
  parameters.allow_one_level_only = true;
  // BALL
  parameters.geometry_variant = Laplace::Parameter::GeometryVariant::Ball;
  // // DISTORTED CUBE
  // parameters.geometry_variant = Laplace::Parameter::GeometryVariant::CubeDistorted;
  // parameters.n_cell_repetitions = 2;
  // parameters.n_refines_distort = parameters.n_refines;
  // parameters.distortion_factor = 0.4;//0.25;
  // // CUBE
  // parameters.geometry_variant = Laplace::Parameter::GeometryVariant::Cube;

  ConditionalOStream pcout(std::cout /*fstream*/,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Laplace::MatrixOperator<dim, fe_degree> laplace_problem{pcout, parameters};
  laplace_problem.setup(true, true);
  laplace_problem.write_output();

  return 0;
}
