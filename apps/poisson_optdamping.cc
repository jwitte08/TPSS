
/*
 * sandbox poisson problem
 *
 *  Created on: Nov 30, 2020
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "poisson_problem.h"

using namespace dealii;
using namespace Laplace;



template<typename Number>
AlignedVector<Number>
filter_lane(const ArrayView<const Number> view, const unsigned int lane)
{
  AlignedVector<Number> out(view.size(), static_cast<Number>(0.));
  for(auto i = 0U; i < view.size(); ++i)
    out[i][lane] = view[i][lane];
  return out;
}



template<int dim, typename LevelMatrix>
struct SubmatrixHandler
{
  using scalar_value_type   = typename LevelMatrix::value_type;
  using value_type          = VectorizedArray<scalar_value_type>;
  using patch_transfer_type = typename LevelMatrix::transfer_type;

  SubmatrixHandler(const LevelMatrix &                              matrix_in,
                   const SubdomainHandler<dim, scalar_value_type> & sd_handler_in)
    : level_matrix(matrix_in),
      prolong(level_matrix.get_patch_transfer(sd_handler_in)),
      left_transfer(level_matrix.get_patch_transfer(sd_handler_in)),
      current_lane(0U)
  {
  }

  void
  reinit(const unsigned int left_index,
         const unsigned int right_index,
         const unsigned int lane_index)
  {
    AssertIndexRange(lane_index, VectorizedArray<scalar_value_type>::size());
    left_transfer->reinit(left_index);
    prolong->reinit(right_index);
    current_lane = lane_index;
  }

  types::global_dof_index
  m() const
  {
    return prolong->n_dofs_per_patch();
  }

  types::global_dof_index
  n() const
  {
    return left_transfer->n_dofs_per_patch();
  }

  void
  vmult(const ArrayView<value_type> dst_view, const ArrayView<const value_type> src_view) const
  {
    for(auto & elem : dst_view)
      elem = 0.;

    LinearAlgebra::distributed::Vector<scalar_value_type> dst, src;
    level_matrix.initialize_dof_vector(dst);
    level_matrix.initialize_dof_vector(src);
    const auto & filtered_src = filter_lane(src_view, current_lane);

    prolong->scatter_add(src, filtered_src);
    level_matrix.vmult(dst, src);
    left_transfer->gather_add(dst_view, dst);
  }

  const LevelMatrix &                  level_matrix;
  std::shared_ptr<patch_transfer_type> prolong;
  std::shared_ptr<patch_transfer_type> left_transfer;
  unsigned int                         current_lane;
};

int
main(int argc, char * argv[])
{
  const auto atoi_if = [&](auto & prm, const int index) {
    Util::ConditionalAtoi(argc, argv)(prm, index);
  };
  const auto atof_if = [&](auto & prm, const int index) {
    Util::ConditionalAtof(argc, argv)(prm, index);
  };

  constexpr int  dim              = CT::DIMENSION_;
  constexpr int  fe_degree        = CT::FE_DEGREE_;
  constexpr auto dof_layout       = CT::DOF_LAYOUT_;
  constexpr auto patch_variant    = CT::PATCH_VARIANT_;
  constexpr auto smoother_variant = CT::SMOOTHER_VARIANT_;

  //: default
  unsigned n_refinements = 0;
  unsigned n_repetitions = 3;
  double   damping       = 0.;
  int      n_threads_max = 1;

  //: parse runtime arguments
  atoi_if(n_refinements, 1);
  atoi_if(n_repetitions, 2);
  atof_if(damping, 3);
  atoi_if(n_threads_max, 4);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                      argv,
                                                      n_threads_max == -1 ?
                                                        numbers::invalid_unsigned_int :
                                                        static_cast<unsigned int>(n_threads_max));

  AssertThrow(damping <= 1., ExcMessage("No over-relaxation allowed."));

  RT::Parameter rt_parameters;
  {
    //: discretization
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = n_refinements;
    rt_parameters.mesh.n_repetitions    = n_repetitions;
    rt_parameters.use_tbb               = MultithreadInfo::n_threads() > 1;

    //: multigrid
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping;
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    rt_parameters.reset_damping_factor(dim);
  }

  using PoissonProblem = typename Poisson::ModelProblem<dim, fe_degree, dof_layout, double>;
  using vector_type    = typename PoissonProblem::VECTOR;

  const bool is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  auto       pcout         = std::make_shared<ConditionalOStream>(std::cout, is_first_proc);

  auto poisson_problem   = std::make_shared<PoissonProblem>(rt_parameters);
  poisson_problem->pcout = pcout;

  poisson_problem->create_triangulation();
  poisson_problem->distribute_dofs();
  poisson_problem->prepare_linear_system();
  poisson_problem->prepare_preconditioner_mg();

  AssertThrow(poisson_problem->mg_schwarz_smoother_pre, ExcMessage("Check runtime parameters."));

  const auto   level           = poisson_problem->mg_matrices.max_level();
  const auto & level_matrix    = poisson_problem->mg_matrices[level];
  const auto subdomain_handler = poisson_problem->mg_schwarz_smoother_pre->get_subdomain_handler();
  SubmatrixHandler submatrix_handler(level_matrix, *subdomain_handler); // R_j A_l R_j^T
  const auto &     patch_dof_worker = submatrix_handler.left_transfer->get_patch_dof_worker();

  const auto & local_solvers =
    *poisson_problem->mg_schwarz_smoother_pre->get_preconditioner(level)->get_local_solvers();

  for(auto j = 0U; j < patch_dof_worker.n_subdomains(); ++j)
  {
    const auto & local_solver = local_solvers[j];
    for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(j); ++lane)
    {
      const auto & Aj = table_to_fullmatrix(Tensors::matrix_to_table(local_solver), lane);
      if(pcout->is_active())
      {
        *pcout << "Aj:" << std::endl;
        Aj.print_formatted(pcout->get_stream());
      }

      submatrix_handler.reinit(j, j, lane);
      const auto & local_matrix =
        table_to_fullmatrix(Tensors::matrix_to_table(submatrix_handler), lane);
      if(pcout->is_active())
      {
        *pcout << "Rj A Rj^T:" << std::endl;
        local_matrix.print_formatted(pcout->get_stream());
      }
    }
  }

  return 0;
}
