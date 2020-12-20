
/*
 * sandbox poisson problem
 *
 *  Created on: Nov 30, 2020
 *      Author: witte
 */

#include <deal.II/base/convergence_table.h>

#include "ct_parameter.h"
#include "poisson_problem.h"
#include "utilities.h"

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



template<typename MatrixType, typename PrecondType>
struct ErrorPropagationMatrix
{
  using value_type = typename MatrixType::value_type;

  ErrorPropagationMatrix(const MatrixType & matrix_in, const PrecondType & precond_in)
    : matrix(matrix_in), precond(precond_in)
  {
  }

  types::global_dof_index
  m() const
  {
    return matrix.m();
  }

  types::global_dof_index
  n() const
  {
    return matrix.n();
  }

  void
  vmult(const ArrayView<value_type> dst_view, const ArrayView<const value_type> src_view) const
  {
    LinearAlgebra::distributed::Vector<value_type> dst, src, Ax;
    matrix.initialize_dof_vector(dst);
    matrix.initialize_dof_vector(src);
    matrix.initialize_dof_vector(Ax);

    std::copy(src_view.cbegin(), src_view.cend(), src.begin());
    matrix.vmult(Ax, src);  // Ax
    precond.vmult(dst, Ax); // B^{-1}Ax
    dst.sadd(-1., 1., src); // (I-B^{-1}A)x
    std::copy(dst.begin(), dst.end(), dst_view.begin());
  }

  FullMatrix<value_type>
  as_fullmatrix()
  {
    return table_to_fullmatrix(Tensors::matrix_to_table(*this));
  }

  const MatrixType &  matrix;
  const PrecondType & precond;
};



template<typename MatrixType>
struct MatrixWrapper
{
  using value_type = typename MatrixType::value_type;

  MatrixWrapper(const MatrixType & matrix_in) : matrix(matrix_in)
  {
  }

  types::global_dof_index
  m() const
  {
    return matrix.m();
  }

  types::global_dof_index
  n() const
  {
    return matrix.n();
  }

  void
  vmult(const ArrayView<value_type> dst_view, const ArrayView<const value_type> src_view) const
  {
    LinearAlgebra::distributed::Vector<value_type> dst(dst_view.size());
    LinearAlgebra::distributed::Vector<value_type> src(src_view.size());

    std::copy(src_view.cbegin(), src_view.cend(), src.begin());
    matrix.vmult(dst, src);
    std::copy(dst.begin(), dst.end(), dst_view.begin());
  }

  FullMatrix<value_type>
  as_fullmatrix()
  {
    return table_to_fullmatrix(Tensors::matrix_to_table(*this));
  }

  const MatrixType & matrix;
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
  double   damping       = 1.;
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
    // rt_parameters.reset_damping_factor(dim);
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

  const auto   level        = poisson_problem->mg_matrices.max_level();
  const auto & level_matrix = poisson_problem->mg_matrices[level];

  const auto subdomain_handler = poisson_problem->mg_schwarz_smoother_pre->get_subdomain_handler();
  const auto &     dofh        = subdomain_handler->get_dof_handler();
  SubmatrixHandler submatrix_handler(level_matrix, *subdomain_handler); // R_j A_l R_j^T
  const auto &     patch_dof_worker = submatrix_handler.left_transfer->get_patch_dof_worker();
  const auto & level_precond = *poisson_problem->mg_schwarz_smoother_pre->get_preconditioner(level);
  const auto & local_solvers = *(level_precond.get_local_solvers());

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1U,
              ExcMessage("MPI not supported."));

  const bool print_details =
    poisson_problem->triangulation.n_global_active_cells() < 20 && dim == 2;

  const auto & print_fullmatrix = [&](const FullMatrix<double> & matrix,
                                      const std::string &        description) {
    if(!print_details)
      return;

    std::cout << description << std::endl;
    matrix.print_formatted(std::cout);
    std::cout << std::endl;
  };

  for(auto j = 0U; j < patch_dof_worker.n_subdomains(); ++j)
  {
    const auto & local_solver = local_solvers[j];
    for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(j); ++lane)
    {
      const auto & Aj = table_to_fullmatrix(Tensors::matrix_to_table(local_solver), lane);
      print_fullmatrix(Aj, "Aj:");

      submatrix_handler.reinit(j, j, lane);
      const auto & local_matrix =
        table_to_fullmatrix(Tensors::matrix_to_table(submatrix_handler), lane);
      print_fullmatrix(local_matrix, "Rj A Rj^T:");
    }
  }

  MatrixWrapper matrix_wrap(level_matrix);
  auto          level_fullmatrix = matrix_wrap.as_fullmatrix();
  print_fullmatrix(level_fullmatrix, "level matrix A:");

  MatrixWrapper precond_wrap(level_precond);
  const auto &  level_fullprecond = precond_wrap.as_fullmatrix();
  print_fullmatrix(level_fullprecond, "preconditioner B^{-1}:");

  FullMatrix<double> BinvA(level_fullmatrix.m());
  level_fullprecond.mmult(BinvA, level_fullmatrix);
  print_fullmatrix(BinvA, "B^{-1}A:");

  /// DEBUG
  // {
  //   FullMatrix<double> ABinvA(level_fullmatrix.m());
  //   level_fullmatrix.mmult(ABinvA, BinvA);
  //   FullMatrix<double> Q(level_fullmatrix.m());
  //   const auto & geigenvalues = compute_generalized_eigenvalues_symm(ABinvA, level_fullmatrix,
  //   Q); std::cout
  //     << "generalized eigenvalues for AB^{-1}Ax = Ax, i.e. eigenvalues of B^{-1}A in A-induced
  //     inner product:"
  //     << std::endl;
  //   std::cout << vector_to_string(geigenvalues) << std::endl;
  //   std::cout << std::endl;

  //   /// DEBUG
  //   // FullMatrix<double> tmp(level_fullmatrix.m());
  //   // FullMatrix<double> tmp2(level_fullmatrix.m());
  //   // ABinvA.mmult(tmp, Q);
  //   // Q.Tmmult(tmp2, tmp);
  //   // tmp2.print_formatted(std::cout);
  //   // std::cout << std::endl;
  // }

  FullMatrix<double> E(IdentityMatrix(level_fullmatrix.m()));
  E.add(-1., BinvA);
  print_fullmatrix(E, "error propagation matrix E=I-B^{-1}A:");

  /// neglect dofs at the boundary if needed
  std::vector<unsigned int> interior_dofs;
  if(TPSS::get_dof_layout(dofh.get_fe()) == TPSS::DoFLayout::Q)
  {
    const auto        n_dofs = level_fullmatrix.m();
    std::vector<bool> boundary_dof_masks(n_dofs);
    DoFTools::extract_boundary_dofs(dofh, ComponentMask(1U, true), boundary_dof_masks);

    for(auto i = 0U; i < n_dofs; ++i)
      if(!boundary_dof_masks[i])
        interior_dofs.push_back(i);

    {
      const auto copyof_A = level_fullmatrix;
      level_fullmatrix.reinit(interior_dofs.size(), interior_dofs.size());
      level_fullmatrix.extract_submatrix_from(copyof_A, interior_dofs, interior_dofs);

      const auto copyof_E = E;
      E.reinit(interior_dofs.size(), interior_dofs.size());
      E.extract_submatrix_from(copyof_E, interior_dofs, interior_dofs);
    }
  }

  {
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

    if(dim == 2)
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
          AssertDimension(Q.m(), v.size());
          AssertDimension(Q.n(), v.size());

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
          AssertDimension(AQ.m(), v.size());
          AssertDimension(AQ.n(), v.size());

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
