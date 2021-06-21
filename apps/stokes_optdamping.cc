/**
 * sandbox (simplified) Stokes problem
 *
 * Created on: Jun 03, 2021
 *     Author: witte
 */



#include "stokes.h"
#include "utilities.h"


using namespace Stokes;



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

  //: default
  unsigned     n_refinements       = 0;
  unsigned     n_repetitions       = 2;
  double       damping             = 1.;
  double       omega               = 1.; // local stability constant
  unsigned int local_solver_index  = 2;  // C0IP
  bool         do_mirror_p_v_block = true;
  unsigned int max_size            = 60;
  bool         skip_A              = false;
  unsigned int ksvd_rank           = 1;
  ///
  double       ip_factor     = 1.;
  int          n_threads_max = 1;
  unsigned int test_index    = 5; // GMG_CG
  unsigned int pde_index     = 6; // NoSlipExp

  //: parse runtime arguments
  atoi_if(n_refinements, 1);
  atoi_if(n_repetitions, 2);
  atof_if(damping, 3);
  atof_if(omega, 4);
  atof_if(local_solver_index, 5);
  atoi_if(do_mirror_p_v_block, 6);
  atoi_if(ksvd_rank, 7);
  atoi_if(max_size, 8);
  atoi_if(skip_A, 9);

  damping /= omega;
  AssertThrow(damping <= 1., ExcMessage("Over-relaxation is not allowed."));

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                      argv,
                                                      n_threads_max == -1 ?
                                                        numbers::invalid_unsigned_int :
                                                        static_cast<unsigned int>(n_threads_max));

  const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0U;
  const auto n_mpi_procs   = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  AssertThrow(n_mpi_procs == 1U, ExcMessage("MPI not supported."));

  StokesFlow       options;
  constexpr auto   dim              = CT::DIMENSION_;
  constexpr auto   fe_degree_sf     = CT::FE_DEGREE_;
  constexpr auto   fe_degree_p      = fe_degree_sf - 1;
  constexpr auto   patch_variant    = CT::PATCH_VARIANT_;
  constexpr auto   smoother_variant = CT::SMOOTHER_VARIANT_;
  constexpr bool   is_simplified    = false; // !!!
  constexpr double threshold_noise  = 1.e-10;

  const auto & make_filename = [&](const RT::Parameter & prms,
                                   const EquationData &  equation_data) -> std::string {
    std::ostringstream oss;

    const auto &      pre_schwarz = prms.multigrid.pre_smoother.schwarz;
    const std::string str_schwarz_variant =
      TPSS::getstr_schwarz_variant(pre_schwarz.patch_variant, pre_schwarz.smoother_variant);
    const unsigned int n_cells_1d = (1 << prms.mesh.n_refinements) * prms.mesh.n_repetitions;
    const double       damping    = pre_schwarz.damping_factor;

    oss << (is_simplified ? "simplestokes_optdamping" : "stokes_optdamping");
    oss << std::scientific << std::setprecision(2);
    oss << "_" << CT::DIMENSION_ << "D";
    oss << "_" << CT::FE_DEGREE_ << "deg";
    AssertDimension(CT::DIMENSION_, 2);
    oss << "_" << n_cells_1d << "x" << n_cells_1d << "mesh";
    oss << "_" << Util::damping_to_fstring(damping) << "damp";
    if(prms.multigrid.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz)
    {
      oss << "_" << str_schwarz_variant;
      oss << "_" << equation_data.sstr_local_solver();
      if(equation_data.local_solver == LocalSolver::C0IP_KSVD)
        for(const auto index : equation_data.ksvd_tensor_indices)
          oss << index;
    }

    return oss.str();
  };

  options.setup(test_index, damping);
  options.prms.multigrid.pre_smoother.schwarz.n_active_blocks =
    options.prms.multigrid.post_smoother.schwarz.n_active_blocks = 2;
  options.prms.multigrid.pre_smoother.n_smoothing_steps =
    options.prms.multigrid.post_smoother.n_smoothing_steps = 1;
  options.prms.multigrid.pre_smoother.schwarz.symmetrize_smoothing =
    options.prms.multigrid.post_smoother.schwarz.symmetrize_smoothing =
      (smoother_variant == TPSS::SmootherVariant::multiplicative ? true : false);
  options.prms.mesh.n_refinements = n_refinements;
  options.prms.mesh.n_repetitions = n_repetitions;

  EquationData equation_data;
  AssertThrow(pde_index < EquationData::n_variants,
              ExcMessage("This equation is not implemented."));
  equation_data.variant             = static_cast<EquationData::Variant>(pde_index);
  equation_data.ip_factor           = ip_factor;
  equation_data.local_solver        = static_cast<LocalSolver>(local_solver_index);
  equation_data.local_kernel_size   = equation_data.local_solver > LocalSolver::C0IP ? 0U : 1U;
  equation_data.ksvd_tensor_indices = [&]() -> std::set<unsigned int> {
    if(ksvd_rank == 1U)
      return {0U};
    else if(ksvd_rank == 2U)
      return {0U, 1U};
    else if(ksvd_rank == 12U)
      return {0U, 2U};
    else if(ksvd_rank == 3U)
      return {0U, 1U, 2U};
    else
      AssertThrow(false, ExcMessage("KSVD rank isn't supported."));
    return {};
  }();
  equation_data.n_lanczos_iterations = 4 + 1;
  if(equation_data.ksvd_tensor_indices == std::set<unsigned int>{0U, 1U})
    equation_data.addition_to_min_eigenvalue = 0.025;
  equation_data.skip_A = skip_A;

  const auto filename = make_filename(options.prms, equation_data);

  std::fstream fout;
  fout.open(filename + ".log", std::ios_base::out);

  const auto pcout = std::make_shared<ConditionalOStream>(fout, is_first_proc);

  // !!!
  using StokesProblem = ModelProblem<dim, fe_degree_p, Method::RaviartThomasStream, is_simplified>;
  // using StokesProblem = ModelProblem<dim, fe_degree_p, Method::RaviartThomas, is_simplified>;
  // !!!
  StokesProblem stokes_problem(options.prms, equation_data);
  stokes_problem.pcout = pcout;

  stokes_problem.make_grid();
  stokes_problem.setup_system();
  stokes_problem.assemble_system();
  const auto mgc = stokes_problem.make_multigrid_velocity_pressure();
  stokes_problem.print_informations();

  AssertThrow(mgc->mg_schwarz_smoother_pre, ExcMessage("Check runtime parameters."));

  const auto                                  level          = mgc->mg_matrices.max_level();
  const TrilinosWrappers::BlockSparseMatrix & level_matrix   = mgc->mg_matrices[level];
  const types::global_dof_index               n_level_dofs_v = level_matrix.block(0, 0).m();

  const auto mgss              = mgc->mg_schwarz_smoother_pre;
  const auto subdomain_handler = mgss->get_subdomain_handler(level);

  auto         patch_transfer     = mgc->mg_matrices[level].get_patch_transfer(*subdomain_handler);
  const auto & patch_dof_worker_v = patch_transfer->get_patch_dof_worker(0);

  const auto   level_precond  = mgc->mg_schwarz_smoother_pre->get_preconditioner(level);
  const auto & local_matrices = *(level_precond->get_local_solvers());

  ////////// LAMBDAS
  const auto & print_fullmatrix = [&](FullMatrix<double> & matrix,
                                      const std::string &  description) {
    if(matrix.n() > max_size)
      return;
    remove_noise_from_matrix(matrix, threshold_noise);
    fout << description << std::endl;
    matrix.print_formatted(fout);
    fout << std::endl;
  };

  const auto extract_interior_dofs = [&](const unsigned int block_index) {
    /// accumulate "global" dof indices over all (physical) subdomains
    const auto & patch_dof_worker = patch_transfer->get_patch_dof_worker(block_index);
    std::vector<types::global_dof_index> dof_indices_accumulated;
    for(auto j = 0U; j < patch_dof_worker.n_subdomains(); ++j)
    {
      patch_transfer->reinit(j);
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(j); ++lane)
      {
        const auto & dof_indices_on_patch =
          patch_transfer->get_global_dof_indices(block_index, lane);
        std::copy(dof_indices_on_patch.cbegin(),
                  dof_indices_on_patch.cend(),
                  std::back_inserter(dof_indices_accumulated));
      }
    }

    /// neglect all duplicates which occur due to overlap
    std::sort(dof_indices_accumulated.begin(), dof_indices_accumulated.end());
    const auto end_of_unique =
      std::unique(dof_indices_accumulated.begin(), dof_indices_accumulated.end());

    return std::vector<types::global_dof_index>(dof_indices_accumulated.begin(), end_of_unique);
  };

  const auto & mirror_p_v_block = [&](auto & mat, const unsigned int n_dofs_v) {
    for(auto i = n_dofs_v; i < mat.m(); ++i)
      for(auto j = 0U; j < n_dofs_v; ++j)
        mat(j, i) = mat(i, j);
  };

  std::vector<types::global_dof_index> block_sizes_col;
  for(auto b = 0U; b < level_matrix.n_block_cols(); ++b)
    block_sizes_col.push_back(level_matrix.block(0, b).n());

  Util::BlockMatrixWrapper<TrilinosWrappers::BlockSparseMatrix> matrix_wrap(level_matrix,
                                                                            block_sizes_col);
  auto level_fullmatrix = matrix_wrap.as_fullmatrix();
  remove_noise_from_matrix(level_fullmatrix, threshold_noise);
  print_fullmatrix(level_fullmatrix, "level matrix A:");

  using level_precond_type = std::decay<decltype(*level_precond)>::type;
  Util::BlockMatrixWrapper<level_precond_type> precond_wrap(*level_precond, block_sizes_col);
  auto                                         level_fullprecond = precond_wrap.as_fullmatrix();
  if(do_mirror_p_v_block)
    mirror_p_v_block(level_fullprecond, n_level_dofs_v);
  remove_noise_from_matrix(level_fullprecond, threshold_noise);
  print_fullmatrix(level_fullprecond, "preconditioner B^{-1}:");

  FullMatrix<double> BinvA(level_fullmatrix.m());
  level_fullprecond.mmult(BinvA, level_fullmatrix);
  // print_fullmatrix(BinvA, "B^{-1}A:");

  FullMatrix<double> E(IdentityMatrix(level_fullmatrix.m()));
  E.add(-1., BinvA);
  remove_noise_from_matrix(E, threshold_noise);
  print_fullmatrix(E, "error propagation matrix E=I-B^{-1}A:");

  ///// compare local matrices
  for(auto j = 0U; j < patch_dof_worker_v.n_subdomains(); ++j)
  {
    const auto & local_solver = local_matrices[j];
    patch_transfer->reinit(j);

    for(auto lane = 0U; lane < patch_dof_worker_v.n_lanes_filled(j); ++lane)
    {
      std::vector<std::vector<types::global_dof_index>> dof_indices_on_patch;
      dof_indices_on_patch.emplace_back(patch_transfer->get_global_dof_indices(0U, lane));
      dof_indices_on_patch.emplace_back(patch_transfer->get_global_dof_indices(1U, lane));
      const auto n_dofs_v = dof_indices_on_patch.front().size();

      auto tildeAj_inv = table_to_fullmatrix(Tensors::inverse_matrix_to_table(local_solver), lane);
      tildeAj_inv /= omega;
      if(do_mirror_p_v_block)
        mirror_p_v_block(tildeAj_inv, n_dofs_v);
      print_fullmatrix(tildeAj_inv, "tildeAj_inv");

      /// DEBUG only if RaviartThomasStream
      // auto Cj = table_to_fullmatrix(Tensors::matrix_to_table(local_solver.solver_sf), lane);
      // print_fullmatrix(Cj, "Cj (stream function problem):");
      // const auto eigenvalues_Cj = compute_eigenvalues(Cj);
      // fout << "complex eigenvalues of Cj (stream function problem): " << std::endl;
      // fout << vector_to_string(eigenvalues_Cj) << std::endl;

      FullMatrix<double> Aj(patch_transfer->n_dofs_per_patch());
      for(auto bj = 0U; bj < 2U; ++bj)
        for(auto bi = 0U; bi < 2U; ++bi)
        {
          const auto &       dof_indices_row = dof_indices_on_patch.at(bi);
          const auto &       dof_indices_col = dof_indices_on_patch.at(bj);
          FullMatrix<double> tmp(dof_indices_row.size(), dof_indices_col.size());
          tmp.extract_submatrix_from(level_matrix.block(bi, bj), dof_indices_row, dof_indices_col);
          std::vector<unsigned int> local_indices_row(dof_indices_row.size());
          std::iota(local_indices_row.begin(), local_indices_row.end(), 0U + bi * n_dofs_v);
          std::vector<unsigned int> local_indices_col(dof_indices_col.size());
          std::iota(local_indices_col.begin(), local_indices_col.end(), 0U + bj * n_dofs_v);
          tmp.scatter_matrix_to(local_indices_row, local_indices_col, Aj);
        }
      remove_noise_from_matrix(Aj, threshold_noise);
      print_fullmatrix(Aj, "Rj A Rj^T:");

      FullMatrix<double> approx_identity(Aj.m());
      tildeAj_inv.mmult(approx_identity, Aj);
      remove_noise_from_matrix(approx_identity, threshold_noise);
      print_fullmatrix(approx_identity, "tildeAj^{-1} Rj A Rj^T:");

      const auto & evs = compute_eigenvalues(approx_identity);
      fout << "eigenvalues of tildeAj^{-1} Rj A Rj^T " << std::endl;
      fout << vector_to_string(evs) << std::endl;

      // FullMatrix<double> Q(Aj.m());
      // const auto &       geigenvalues = compute_generalized_eigenvalues_symm(Aj, tildeAj_inv, Q);
      // fout << "generalized eigenvalues of Rj A Rj^T x = lambda Aj x:" << std::endl;
      // fout << vector_to_string(geigenvalues) << std::endl;
    }
  }

  /// neglect constrained dofs at the boundary if needed
  std::vector<types::global_dof_index> interior_dofs;
  {
    const auto & interior_dofs_v = extract_interior_dofs(0U);
    std::copy(interior_dofs_v.begin(), interior_dofs_v.end(), std::back_inserter(interior_dofs));
    const auto & interior_dofs_p = extract_interior_dofs(1U);
    std::transform(interior_dofs_p.begin(),
                   interior_dofs_p.end(),
                   std::back_inserter(interior_dofs),
                   [&](const auto index) { return index + n_level_dofs_v; });
  }

  {
    const auto copyof_A = level_fullmatrix;
    level_fullmatrix.reinit(interior_dofs.size(), interior_dofs.size());
    level_fullmatrix.extract_submatrix_from(copyof_A, interior_dofs, interior_dofs);
    remove_noise_from_matrix(level_fullmatrix, threshold_noise);

    const auto copyof_E = E;
    E.reinit(interior_dofs.size(), interior_dofs.size());
    E.extract_submatrix_from(copyof_E, interior_dofs, interior_dofs);
    remove_noise_from_matrix(E, threshold_noise);
  }

  print_fullmatrix(level_fullmatrix, "level matrix A (no boundary!):");

  print_fullmatrix(E, "error propagation matrix E=I-B^{-1}A (no boundary!):");

  {
    const auto complex_eigenvalues = compute_eigenvalues(E);
    fout << "complex eigenvalues of E:" << std::endl;
    fout << vector_to_string(complex_eigenvalues) << std::endl;
  }

  return 0;
}
