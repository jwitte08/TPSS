
/*
 * sandbox poisson problem
 *
 *  Created on: Nov 30, 2020
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "biharmonic_problem.h"
#include "ct_parameter.h"
#include "utilities.h"

using namespace dealii;
using namespace Biharmonic;



template<typename Number>
AlignedVector<Number>
filter_lane(const ArrayView<const Number> view, const unsigned int lane)
{
  AlignedVector<Number> out(view.size(), static_cast<Number>(0.));
  for(auto i = 0U; i < view.size(); ++i)
    out[i][lane] = view[i][lane];
  return out;
}



template<typename MatrixType>
using MatrixWrapper =
  typename Util::MatrixWrapper<MatrixType, LinearAlgebra::distributed::Vector<double>>;



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
  constexpr auto patch_variant    = CT::PATCH_VARIANT_;
  constexpr auto smoother_variant = CT::SMOOTHER_VARIANT_;

  //: default
  unsigned n_refinements = 0;
  unsigned n_repetitions = 3;
  double   damping       = 1.;
  int      n_threads_max = 1;
  double   ip_factor     = 1.;

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

  using BiharmonicProblem = typename Biharmonic::ModelProblem<dim, fe_degree>;
  using vector_type       = typename BiharmonicProblem::VECTOR;

  RT::Parameter rt_parameters;
  {
    //: discretization
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = n_refinements;
    rt_parameters.mesh.n_repetitions    = n_repetitions;
    rt_parameters.use_tbb               = MultithreadInfo::n_threads() > 1;

    //: solver
    rt_parameters.solver.variant              = "cg";
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;

    //: multigrid
    rt_parameters.multigrid.pre_smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant        = patch_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant     = smoother_variant;
    rt_parameters.multigrid.pre_smoother.schwarz.userdefined_coloring = true;
    rt_parameters.multigrid.pre_smoother.schwarz.damping_factor       = damping;
    rt_parameters.multigrid.post_smoother = rt_parameters.multigrid.pre_smoother;
    // rt_parameters.reset_damping_factor(dim);
  }

  EquationData equation_data;
  equation_data.variant              = EquationData::Variant::ClampedBell;
  equation_data.local_solver_variant = LocalSolverVariant::Exact;
  equation_data.ip_factor            = ip_factor;

  auto biharmonic_problem = std::make_shared<BiharmonicProblem>(rt_parameters, equation_data);

  biharmonic_problem->make_grid();
  biharmonic_problem->setup_system();
  biharmonic_problem->assemble_system();
  biharmonic_problem->prepare_preconditioner_mg();

  AssertThrow(biharmonic_problem->mg_schwarz_smoother_pre, ExcMessage("Check runtime parameters."));

  const auto   level        = biharmonic_problem->mg_matrices.max_level();
  const auto & level_matrix = biharmonic_problem->mg_matrices[level];

  const auto subdomain_handler =
    biharmonic_problem->mg_schwarz_smoother_pre->get_subdomain_handler();
  const auto & dofh = subdomain_handler->get_dof_handler();

  TPSS::PatchTransfer<dim, double>        patch_transfer(*subdomain_handler);
  const TPSS::PatchDoFWorker<dim, double> patch_dof_worker(subdomain_handler->get_dof_info());
  const auto &                            level_precond =
    *biharmonic_problem->mg_schwarz_smoother_pre->get_preconditioner(level);
  const auto & local_solvers = *(level_precond.get_local_solvers());

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1U,
              ExcMessage("MPI not supported."));

  ////////// LAMBDAS
  const bool print_details =
    biharmonic_problem->triangulation.n_global_active_cells() < 20 && dim == 2;

  const auto & print_fullmatrix = [&](const FullMatrix<double> & matrix,
                                      const std::string &        description) {
    if(!print_details)
      return;
    std::cout << description << std::endl;
    matrix.print_formatted(std::cout);
    std::cout << std::endl;
  };

  const auto extract_interior_dofs = [&]() {
    /// accumulate "global" dof indices over all (physical) subdomains
    std::vector<unsigned int> dof_indices_accumulated;
    for(auto j = 0U; j < patch_dof_worker.n_subdomains(); ++j)
    {
      patch_transfer.reinit(j);
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(j); ++lane)
      {
        const auto & dof_indices_on_patch = patch_transfer.get_dof_indices(lane);
        std::copy(dof_indices_on_patch.cbegin(),
                  dof_indices_on_patch.cend(),
                  std::back_inserter(dof_indices_accumulated));
      }
    }

    /// neglect all duplicates which occur due to overlap
    std::sort(dof_indices_accumulated.begin(), dof_indices_accumulated.end());
    const auto end_of_unique =
      std::unique(dof_indices_accumulated.begin(), dof_indices_accumulated.end());

    return std::vector<unsigned int>(dof_indices_accumulated.begin(), end_of_unique);
  };

  using level_matrix_type = std::decay<decltype(level_matrix)>::type;
  MatrixWrapper<level_matrix_type> matrix_wrap(level_matrix);
  auto                             level_fullmatrix = matrix_wrap.as_fullmatrix();
  print_fullmatrix(level_fullmatrix, "level matrix A:");

  using level_precond_type = std::decay<decltype(level_precond)>::type;
  MatrixWrapper<level_precond_type> precond_wrap(level_precond);
  const auto &                      level_fullprecond = precond_wrap.as_fullmatrix();
  print_fullmatrix(level_fullprecond, "preconditioner B^{-1}:");

  FullMatrix<double> BinvA(level_fullmatrix.m());
  level_fullprecond.mmult(BinvA, level_fullmatrix);
  // print_fullmatrix(BinvA, "B^{-1}A:");

  FullMatrix<double> E(IdentityMatrix(level_fullmatrix.m()));
  E.add(-1., BinvA);
  print_fullmatrix(E, "error propagation matrix E=I-B^{-1}A:");

  /// neglect constrained dofs at the boundary if needed
  std::vector<unsigned int> interior_dofs;
  if(TPSS::get_dof_layout(dofh.get_fe()) == TPSS::DoFLayout::Q)
  {
    const auto n_dofs = level_fullmatrix.m();
    interior_dofs     = std::move(extract_interior_dofs());

    {
      const auto copyof_A = level_fullmatrix;
      level_fullmatrix.reinit(interior_dofs.size(), interior_dofs.size());
      level_fullmatrix.extract_submatrix_from(copyof_A, interior_dofs, interior_dofs);

      const auto copyof_E = E;
      E.reinit(interior_dofs.size(), interior_dofs.size());
      E.extract_submatrix_from(copyof_E, interior_dofs, interior_dofs);
    }
  }

  print_fullmatrix(level_fullmatrix, "level matrix A (no boundary!):");
  print_fullmatrix(E, "error propagation matrix E=I-B^{-1}A (no boundary!):");

  {
    const auto complex_eigenvalues = compute_eigenvalues(E);
    std::cout << "complex eigenvalues of E:" << std::endl;
    std::cout << vector_to_string(complex_eigenvalues) << std::endl;

    FullMatrix<double> AE(level_fullmatrix.m());
    level_fullmatrix.mmult(AE, E);
    FullMatrix<double> Q(level_fullmatrix.m());
    const auto &       geigenvalues = compute_generalized_eigenvalues_symm(AE, level_fullmatrix, Q);
    std::cout
      << "generalized eigenvalues for AEx = Ax, i.e. eigenvalues of error propagation E in A-induced inner product:"
      << std::endl;
    std::cout << "(NOTE From left to right means from high- to low-frequency modes)" << std::endl;
    std::cout << vector_to_string(geigenvalues) << std::endl;
    std::cout << std::endl;

    /// DEBUG
    // FullMatrix<double> tmp(level_fullmatrix.m());
    // FullMatrix<double> tmp2(level_fullmatrix.m());
    // AE.mmult(tmp, Q);
    // Q.Tmmult(tmp2, tmp);
    // tmp2.print_formatted(std::cout);
    // std::cout << std::endl;

    if(print_details && dim == 2)
    {
      std::cout << "visualize generalized eigenvectors Q..." << std::endl;
      {
        std::vector<LinearAlgebra::distributed::Vector<double>> eigenvectors;
        for(auto i = 0U; i < Q.n(); ++i)
        {
          eigenvectors.emplace_back();
          auto & v = eigenvectors.back();
          level_matrix.initialize_dof_vector(v);
          v = 0.;

          if(interior_dofs.empty())
            for(auto j = 0U; j < Q.m(); ++j)
              v(j) = Q(j, i);
          else
            for(auto j = 0U; j < Q.m(); ++j)
              v(interior_dofs[j]) = Q(j, i);

          visualize_dof_vector(
            dofh, v, "eigvecs", 10, MappingQGeneric<dim>(1), "_" + Utilities::int_to_string(i, 4));
        }
      }

      std::cout << "visualize A-mapped eigenvectors AQ..." << std::endl;
      {
        FullMatrix<double> AQ(level_fullmatrix.m());
        level_fullmatrix.mmult(AQ, Q);
        std::vector<LinearAlgebra::distributed::Vector<double>> eigenvectors;
        for(auto i = 0U; i < Q.n(); ++i)
        {
          eigenvectors.emplace_back();
          auto & v = eigenvectors.back();
          level_matrix.initialize_dof_vector(v);
          v = 0.;

          if(interior_dofs.empty())
            for(auto j = 0U; j < AQ.m(); ++j)
              v(j) = AQ(j, i);
          else
            for(auto j = 0U; j < AQ.m(); ++j)
              v(interior_dofs[j]) = AQ(j, i);

          visualize_dof_vector(
            dofh, v, "Aeigvecs", 10, MappingQGeneric<dim>(1), "_" + Utilities::int_to_string(i, 4));
        }
      }
    }
  }

  return 0;
}
