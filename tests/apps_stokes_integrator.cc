
/*
 * Test Stokes integrators.
 *
 *  Created on: May 21, 2020
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "linelasticity_problem.h"
#include "stokes.h"
#include "stokes_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace Stokes;



template<typename T>
class TestStokesIntegrator : public testing::Test
{
protected:
  static constexpr int          dim         = T::template value<0>();
  static constexpr int          fe_degree_p = T::template value<1>();
  static constexpr int          fe_degree_v = fe_degree_p + 1;
  static constexpr unsigned int macro_size  = VectorizedArray<double>::size();


  void
  SetUp() override
  {
    ASSERT_FALSE(fe_degree_p == 0)
      << "This test is expected to fail, because there exist no piecewise constant functions with inter-element continuity (if needed use FE_DGQ<dim>(0) instead of FE_Q<dim>(0))...";
    if(fe_degree_p == 0)
      return;

    ofs.open("apps_stokes_integrator.log", std::ios_base::app);
    const bool is_first_proc   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const bool is_higher_order = fe_degree_v > 2;
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, !is_higher_order && is_first_proc);

    // {
    //   auto & pre_smoother                   = rt_parameters.multigrid.pre_smoother;
    //   pre_smoother.variant                  = SmootherParameter::SmootherVariant::Schwarz;
    //   pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::vertex;
    //   pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    //   rt_parameters.multigrid.post_smoother = pre_smoother;
    // }

    {
      const auto patch_variant    = TPSS::PatchVariant::vertex;
      const auto smoother_variant = TPSS::SmootherVariant::additive;
      const auto damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
      options.setup(/*FGRMES_GMGVelocity + Schwarz*/ 3, damping, patch_variant, smoother_variant);
    }
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  check_system_matrix_velocity()
  {
    *pcout_owned << "//////////   STOKES PROBLEM" << std::endl;
    EquationData equation_data;
    using StokesProblem = ModelProblem<dim, fe_degree_p, Method::TaylorHoodDGQ>;
    std::shared_ptr<const StokesProblem> stokes_problem;
    auto new_problem   = std::make_shared<StokesProblem>(options.prms, equation_data);
    new_problem->pcout = pcout_owned;
    new_problem->make_grid();
    new_problem->setup_system();
    const auto & mgc = *(new_problem->make_multigrid_velocity());
    stokes_problem   = new_problem;
    stokes_problem->print_informations();

    *pcout_owned << "//////////   LINEAR ELASTICITY PROBLEM" << std::endl;
    LinElasticity::EquationData equation_data_elasticity;
    equation_data_elasticity.integrator_variant =
      LinElasticity::EquationData::IntegratorVariant::strain;
    equation_data_elasticity.ip_factor = 1.;
    using ElasticityProblem            = LinElasticity::ModelProblem<dim, fe_degree_p + 1>;
    const auto elasticity_problem =
      std::make_shared<ElasticityProblem>(*pcout_owned, options.prms, equation_data_elasticity);
    elasticity_problem->create_triangulation();
    elasticity_problem->assemble_matrix();
    elasticity_problem->print_informations();
    const auto max_level = stokes_problem->max_level();

    FullMatrix<double> A(mgc.mg_matrices[max_level].m());
    FullMatrix<double> Aref(elasticity_problem->sparse_matrix.m());
    ASSERT_EQ(A.m(), Aref.m());
    A.copy_from(mgc.mg_matrices[max_level]);
    Aref.copy_from(elasticity_problem->sparse_matrix);
    compare_matrix(A, Aref);
  }


  void
  check_system_matrix(const bool check_diagonal = false, const bool check_rhs = false)
  {
    {
      const auto patch_variant    = TPSS::PatchVariant::vertex;
      const auto smoother_variant = TPSS::SmootherVariant::additive;
      const auto damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
      options.setup(/*GMRES_GMG + Schwarz*/ 4, damping, patch_variant, smoother_variant);
    }

    EquationData equation_data;
    using StokesProblem = ModelProblem<dim, fe_degree_p, Method::TaylorHood>;

    *pcout_owned << "//////////   STOKES PROBLEM (step-56)" << std::endl;
    auto stokes_problem_step56   = std::make_shared<StokesProblem>(options.prms, equation_data);
    stokes_problem_step56->pcout = pcout_owned;
    stokes_problem_step56->make_grid();
    stokes_problem_step56->setup_system();
    stokes_problem_step56->assemble_system_step56();
    stokes_problem_step56->print_informations();

    *pcout_owned << "//////////   STOKES PROBLEM" << std::endl;
    auto stokes_problem   = std::make_shared<StokesProblem>(options.prms, equation_data);
    stokes_problem->pcout = pcout_owned;
    stokes_problem->make_grid();
    stokes_problem->setup_system();
    // stokes_problem->assemble_system_velocity_pressure();
    stokes_problem->assemble_system();
    stokes_problem->print_informations();

    /// compare (only!) the right hand side
    if(check_rhs)
    {
      const auto & system_rhs     = stokes_problem->system_rhs;
      const auto & system_rhs_ref = stokes_problem_step56->system_rhs;
      *pcout_owned << "... comparing system RHS (velocity)\n";
      compare_vector(system_rhs.block(0), system_rhs_ref.block(0));
      *pcout_owned << "... comparing system RHS (pressure)\n";
      compare_vector(system_rhs.block(1), system_rhs_ref.block(1));
      return;
    }

    /// compare the system matrix (or only its diagonal!)
    FullMatrix<double> A(stokes_problem->system_matrix.m());
    FullMatrix<double> Aref(stokes_problem_step56->system_matrix.m());
    ASSERT_EQ(A.m(), Aref.m());
    if(check_diagonal)
    {
      const auto zero_out = [](auto & block_matrix) {
        for(auto bi = 0U; bi < block_matrix.n_block_rows(); ++bi)
          for(auto bj = 0U; bj < block_matrix.n_block_cols(); ++bj)
            if(bi != bj)
              block_matrix.block(bi, bj) *= 0.;
      };
      zero_out(stokes_problem_step56->system_matrix);
      zero_out(stokes_problem->system_matrix);
    }
    A.copy_from(stokes_problem->system_matrix);
    Aref.copy_from(stokes_problem_step56->system_matrix);
    compare_matrix(A, Aref);
  }


  template<Stokes::Method method = Method::Qkplus2_DGPk>
  void
  check_local_solvers_velocity(const bool only_diagonal = false)
  {
    const auto patch_variant    = TPSS::PatchVariant::vertex;
    const auto smoother_variant = TPSS::SmootherVariant::additive;
    const auto damping          = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    options.setup(/*FGMRES_GMGvelocity*/ 3, damping, patch_variant, smoother_variant);

    EquationData equation_data;

    using StokesProblem = ModelProblem<dim, fe_degree_p, method>;
    std::shared_ptr<const StokesProblem> stokes_problem;
    auto new_problem   = std::make_shared<StokesProblem>(options.prms, equation_data);
    new_problem->pcout = pcout_owned;
    new_problem->make_grid();
    new_problem->setup_system();
    const auto mgc = new_problem->make_multigrid_velocity();
    stokes_problem = new_problem;
    stokes_problem->print_informations();
    const auto max_level = stokes_problem->max_level();

    using MatrixIntegrator =
      Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree_v, double, StokesProblem::dof_layout_v>;
    using LocalMatrix   = typename MatrixIntegrator::matrix_type;
    using PatchTransfer = TPSS::PatchTransfer<dim, double>;

    const auto mg_smoother = mgc->mg_schwarz_smoother_pre;
    ASSERT_TRUE(mg_smoother) << "mg_smoother is not initialized.";
    const auto    subdomain_handler = mg_smoother->get_subdomain_handler();
    PatchTransfer patch_transfer(*subdomain_handler);
    const auto &  patch_worker = patch_transfer.get_patch_dof_worker();
    const auto    n_subdomains = patch_worker.get_partition_data().n_subdomains();

    std::vector<LocalMatrix> local_matrices(n_subdomains);
    MatrixIntegrator         integrator;
    integrator.initialize(equation_data);
    integrator.template assemble_subspace_inverses<bool>(
      *subdomain_handler,
      local_matrices,
      /*dummy*/ false,
      patch_worker.get_partition_data().get_patch_range());

    const TrilinosWrappers::SparseMatrix * system_matrix = &(mgc->mg_matrices[max_level]);
    // AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("TODO..."));

    /// compare local matrices
    for(auto patch = 0U; patch < n_subdomains; ++patch)
    {
      patch_transfer.reinit(patch);
      for(auto lane = 0U; lane < patch_worker.n_lanes_filled(patch); ++lane)
      {
        std::vector<types::global_dof_index> dof_indices_on_patch;
        {
          const auto view = patch_transfer.get_dof_indices(lane);
          std::copy(view.cbegin(), view.cend(), std::back_inserter(dof_indices_on_patch));
        }
        FullMatrix<double> local_matrix(dof_indices_on_patch.size());
        local_matrix.extract_submatrix_from(*system_matrix,
                                            dof_indices_on_patch,
                                            dof_indices_on_patch);

        auto local_matrix_tp = table_to_fullmatrix(local_matrices[patch].as_table(), lane);

        if(only_diagonal)
        {
          const auto zero_out_block =
            [&](auto & matrix, const unsigned int row, const unsigned int col) {
              const unsigned int n_dofs = dof_indices_on_patch.size();
              AssertDimension(n_dofs % dim, 0);
              const unsigned int n_dofs_per_block = n_dofs / dim;
              const unsigned     row_start        = row * n_dofs_per_block;
              const unsigned     col_start        = col * n_dofs_per_block;
              for(auto i = row_start; i < row_start + n_dofs_per_block; ++i)
                for(auto j = col_start; j < col_start + n_dofs_per_block; ++j)
                  matrix(i, j) = 0.;
            };
          for(auto row = 0U; row < dim; ++row)
            for(auto col = 0U; col < dim; ++col)
              if(row != col)
              {
                zero_out_block(local_matrix, row, col);
                zero_out_block(local_matrix_tp, row, col);
              }
        }

        compare_matrix(local_matrix_tp, local_matrix);
      }
    }
  }


  void
  check_local_solvers_block(const bool check_diagonal     = false,
                            const bool check_level_matrix = false)
  {
    EXPECT_FALSE(fe_degree_p == 0)
      << "This test is expected to fail, because MatrixFree does not support finite elements of degree zero. Therefore, we return here...";
    if(fe_degree_p == 0)
      return;

    {
      const auto patch_variant    = TPSS::PatchVariant::vertex;
      const auto smoother_variant = TPSS::SmootherVariant::additive;
      const auto damping = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
      options.setup(/*GMRES_GMG + Schwarz*/ 4, damping, patch_variant, smoother_variant);
    }

    EquationData equation_data;

    /// TODO tensor product smoothers...
    using StokesProblem = ModelProblem<dim, fe_degree_p, Method::DGQkplus2_DGPk>;

    *pcout_owned << "TEST. check_local_solvers_block(check_diagonal = " << std::boolalpha
                 << check_diagonal << ", check_level_matrix = " << check_level_matrix << ")"
                 << std::endl
                 << std::endl;
    const auto stokes_problem = std::make_shared<StokesProblem>(options.prms, equation_data);
    stokes_problem->pcout     = pcout_owned;
    stokes_problem->make_grid();
    stokes_problem->setup_system();
    stokes_problem->assemble_system();
    const auto & mgc = *(stokes_problem->make_multigrid_velocity_pressure());
    stokes_problem->print_informations();
    const auto max_level = stokes_problem->max_level();

    if(check_level_matrix)
    {
      const auto &       level_matrix = mgc.mg_matrices[max_level];
      FullMatrix<double> A(level_matrix.m());
      A.copy_from(level_matrix);
      const auto &       system_matrix = stokes_problem->system_matrix;
      FullMatrix<double> Aref(system_matrix.m());
      Aref.copy_from(system_matrix);
      compare_matrix(A, Aref);
      return;
    }

    using MatrixIntegrator =
      VelocityPressure::FD::MatrixIntegrator<dim, fe_degree_p, double, StokesProblem::dof_layout_v>;
    using LocalMatrix = typename MatrixIntegrator::matrix_type;

    MatrixIntegrator integrator;
    integrator.initialize(equation_data);
    const auto mg_smoother = mgc.mg_schwarz_smoother_pre;
    ASSERT_TRUE(mg_smoother) << "mg_smoother is not initialized.";
    const auto   subdomain_handler     = mg_smoother->get_subdomain_handler();
    const auto   patch_transfer        = integrator.get_patch_transfer(*subdomain_handler);
    const auto & patch_worker_velocity = patch_transfer->get_patch_dof_worker(0);
    const auto   n_subdomains          = patch_worker_velocity.get_partition_data().n_subdomains();

    std::vector<LocalMatrix> local_matrices(n_subdomains);
    integrator.template assemble_subspace_inverses<bool>(
      *subdomain_handler,
      local_matrices,
      /*dummy*/ false,
      patch_worker_velocity.get_partition_data().get_patch_range());

    const auto * level_matrix = &(mgc.mg_matrices[max_level]);

    /// compare local matrices
    for(auto patch = 0U; patch < n_subdomains; ++patch)
    {
      patch_transfer->reinit(patch);
      FullMatrix<double> local_matrix(patch_transfer->n_dofs_per_patch());
      for(auto lane = 0U; lane < patch_worker_velocity.n_lanes_filled(patch); ++lane)
      {
        local_matrix *= 0.;

        {
          /// Patch-wise local and global dof indices of velocity block.
          const auto & patch_transfer_velocity = patch_transfer->get_patch_transfer(0);
          std::vector<types::global_dof_index> velocity_dof_indices_on_patch;
          {
            const auto view = patch_transfer_velocity.get_dof_indices(lane);
            std::copy(view.cbegin(),
                      view.cend(),
                      std::back_inserter(velocity_dof_indices_on_patch));
          }
          std::vector<unsigned int> velocity_local_dof_indices(
            velocity_dof_indices_on_patch.size());
          std::iota(velocity_local_dof_indices.begin(), velocity_local_dof_indices.end(), 0U);

          /// Patch-wise local and global dof indices of pressure block.
          const auto & patch_transfer_pressure = patch_transfer->get_patch_transfer(1);
          std::vector<types::global_dof_index> pressure_dof_indices_on_patch;
          {
            const auto view = patch_transfer_pressure.get_dof_indices(lane);
            std::copy(view.cbegin(),
                      view.cend(),
                      std::back_inserter(pressure_dof_indices_on_patch));
          }
          std::vector<unsigned int> pressure_local_dof_indices(
            pressure_dof_indices_on_patch.size());
          std::iota(pressure_local_dof_indices.begin(),
                    pressure_local_dof_indices.end(),
                    velocity_dof_indices_on_patch.size());

          /// Extract and insert local velocity block.
          FullMatrix<double> local_block_velocity(velocity_dof_indices_on_patch.size());
          local_block_velocity.extract_submatrix_from(level_matrix->block(0, 0),
                                                      velocity_dof_indices_on_patch,
                                                      velocity_dof_indices_on_patch);
          local_block_velocity.scatter_matrix_to(velocity_local_dof_indices,
                                                 velocity_local_dof_indices,
                                                 local_matrix);

          /// Extract and insert local pressure block.
          FullMatrix<double> local_block_pressure(pressure_dof_indices_on_patch.size());
          local_block_pressure.extract_submatrix_from(level_matrix->block(1, 1),
                                                      pressure_dof_indices_on_patch,
                                                      pressure_dof_indices_on_patch);
          local_block_pressure.scatter_matrix_to(pressure_local_dof_indices,
                                                 pressure_local_dof_indices,
                                                 local_matrix);

          /// velocity-pressure
          FullMatrix<double> local_block_velocity_pressure(velocity_dof_indices_on_patch.size(),
                                                           pressure_dof_indices_on_patch.size());
          local_block_velocity_pressure.extract_submatrix_from(level_matrix->block(0, 1),
                                                               velocity_dof_indices_on_patch,
                                                               pressure_dof_indices_on_patch);
          local_block_velocity_pressure.scatter_matrix_to(velocity_local_dof_indices,
                                                          pressure_local_dof_indices,
                                                          local_matrix);


          /// pressure-velocity
          FullMatrix<double> local_block_pressure_velocity(pressure_dof_indices_on_patch.size(),
                                                           velocity_dof_indices_on_patch.size());
          local_block_pressure_velocity.extract_submatrix_from(level_matrix->block(1, 0),
                                                               pressure_dof_indices_on_patch,
                                                               velocity_dof_indices_on_patch);
          local_block_pressure_velocity.scatter_matrix_to(pressure_local_dof_indices,
                                                          velocity_local_dof_indices,
                                                          local_matrix);
        }

        auto local_matrix_tp = table_to_fullmatrix(local_matrices[patch].as_table(), lane);

        if(check_diagonal)
        {
          const auto zero_out_block =
            [&](auto & matrix, const unsigned int row, const unsigned int col) {
              /// row : block row index of the (row,col)-block
              /// col : block column index of the (row,col)-block
              const unsigned int n_dofs_per_block_row = patch_transfer->n_dofs_per_patch(row);
              const unsigned int n_dofs_per_block_col = patch_transfer->n_dofs_per_patch(col);
              auto               row_start            = 0U;
              for(auto b = 0U; b < row; ++b)
                row_start += patch_transfer->n_dofs_per_patch(b);
              auto col_start = 0U;
              for(auto b = 0U; b < col; ++b)
                col_start += patch_transfer->n_dofs_per_patch(b);
              for(auto i = row_start; i < row_start + n_dofs_per_block_row; ++i)
                for(auto j = col_start; j < col_start + n_dofs_per_block_col; ++j)
                  matrix(i, j) = 0.;
            };
          for(auto row = 0U; row < 2; ++row)
            for(auto col = 0U; col < 2; ++col)
              if(row != col)
              {
                zero_out_block(local_matrix, row, col);
                zero_out_block(local_matrix_tp, row, col);
              }
        }

        compare_matrix(local_matrix_tp, local_matrix);
      }
    }
  }


  template<Stokes::Method method>
  void
  check_matrixintegratorlmw(const std::array<unsigned int, 2> block_index)
  {
    const auto [block_row, block_column] = block_index;
    // const bool do_velocity_velocity      = block_row == 0U && block_column == 0U;
    // const bool do_velocity_pressure      = block_row == 0U && block_column == 1U;
    // const bool do_pressure_velocity      = block_row == 1U && block_column == 0U;
    const bool do_pressure_pressure = block_row == 1U && block_column == 1U;

    ASSERT_FALSE(do_pressure_pressure) << "block is zero thus not implemented...";

    /// TODO define parameters outside of this function ?
    const auto patch_variant    = TPSS::PatchVariant::vertex;
    const auto smoother_variant = TPSS::SmootherVariant::additive;
    const auto damping          = TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    options.setup(/*CG_GMG*/ 5, damping, patch_variant, smoother_variant);
    if(method == Stokes::Method::RaviartThomas)
      options.prms.mesh.do_colorization = true;

    /// TODO define equation data outside of this function ?
    EquationData equation_data;
    if(options.prms.mesh.do_colorization)
      for(types::boundary_id id = 0; id < GeometryInfo<dim>::faces_per_cell; ++id)
        equation_data.dirichlet_boundary_ids_velocity.insert(id);

    using StokesProblem = ModelProblem<dim, fe_degree_p, method>;

    const auto stokes_problem = std::make_shared<StokesProblem>(options.prms, equation_data);
    stokes_problem->pcout     = pcout_owned;
    stokes_problem->make_grid();
    stokes_problem->setup_system();
    const auto mgc = stokes_problem->make_multigrid_velocity_pressure();
    stokes_problem->print_informations();

    using MatrixIntegrator  = VelocityPressure::FD::MatrixIntegratorLMW<dim,
                                                                       fe_degree_p,
                                                                       double,
                                                                       StokesProblem::dof_layout_v,
                                                                       StokesProblem::fe_degree_v>;
    using local_matrix_type = typename MatrixIntegrator::matrix_type;

    ASSERT_TRUE(mgc->mg_schwarz_smoother_pre) << "mg_smoother is not initialized.";

    const auto   level             = stokes_problem->max_level();
    const auto   mgss              = mgc->mg_schwarz_smoother_pre;
    const auto   subdomain_handler = mgss->get_subdomain_handler(level);
    const auto & partition_data    = subdomain_handler->get_partition_data();
    const TrilinosWrappers::BlockSparseMatrix & level_matrix = mgc->mg_matrices[level];
    const auto                                  n_subdomains = partition_data.n_subdomains();
    std::vector<local_matrix_type>              local_matrices(n_subdomains);

    MatrixIntegrator integrator;
    integrator.initialize(equation_data);

    integrator.assemble_subspace_inverses(*subdomain_handler,
                                          local_matrices,
                                          level_matrix,
                                          partition_data.get_patch_range());

    using MatrixIntegratorCut =
      VelocityPressure::FD::MatrixIntegratorCut<dim,
                                                fe_degree_p,
                                                double,
                                                StokesProblem::dof_layout_v,
                                                StokesProblem::fe_degree_v>;

    std::vector<local_matrix_type> local_matrices_cut(n_subdomains);

    MatrixIntegratorCut integrator_cut;
    integrator_cut.initialize(equation_data);

    integrator_cut.assemble_subspace_inverses(*subdomain_handler,
                                              local_matrices_cut,
                                              level_matrix,
                                              partition_data.get_patch_range());

    const auto   patch_transfer     = integrator.get_patch_transfer(*subdomain_handler);
    const auto & patch_dof_worker_v = patch_transfer->get_patch_dof_worker(0);

    ASSERT_EQ(local_matrices.size(), local_matrices_cut.size());
    for(auto patch_index = 0U; patch_index < local_matrices.size(); ++patch_index)
    {
      for(auto lane = 0U; lane < patch_dof_worker_v.n_lanes_filled(patch_index); ++lane)
      {
        auto & patch_matrix     = local_matrices[patch_index];
        auto & patch_matrix_cut = local_matrices_cut[patch_index];

        const auto & fullmatrix =
          table_to_fullmatrix(patch_matrix.get_block(block_row, block_column).as_table(), lane);
        const auto & fullmatrix_cut =
          table_to_fullmatrix(patch_matrix_cut.get_block(block_row, block_column).as_table(), lane);
        compare_matrix(fullmatrix, fullmatrix_cut);
      }
    }
  }


  void
  compare_matrix(const FullMatrix<double> & patch_matrix_full,
                 const FullMatrix<double> & other) const
  {
    Util::compare_matrix(patch_matrix_full, other, *pcout_owned);
  }


  void
  compare_vector(const Vector<double> & vector, const Vector<double> & other) const
  {
    Util::compare_vector(vector, other, *pcout_owned);
  }


  void
  compare_vector(const LinearAlgebra::distributed::Vector<double> & vector,
                 const LinearAlgebra::distributed::Vector<double> & other) const
  {
    Util::compare_vector(vector, other, *pcout_owned);
  }


  void
  compare_inverse_matrix(const FullMatrix<double> & inverse_patch_matrix,
                         const FullMatrix<double> & other) const
  {
    Util::compare_inverse_matrix(inverse_patch_matrix, other, *pcout_owned);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  StokesFlow<dim, fe_degree_p> options;
  // RT::Parameter                                  options.prms;
  std::shared_ptr<const MatrixFree<dim, double>> mf_storage;
  std::shared_ptr<SubdomainHandler<dim, double>> subdomain_handler;
};



TYPED_TEST_SUITE_P(TestStokesIntegrator);



TYPED_TEST_P(TestStokesIntegrator, CheckSystemMatrixVelocity)
{
  using Fixture = TestStokesIntegrator<TypeParam>;
  ASSERT_TRUE(false) << "TODO...";
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::check_system_matrix_velocity();
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::check_system_matrix_velocity();
}



TYPED_TEST_P(TestStokesIntegrator, CheckLocalSolversVelocityMPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::check_local_solvers_velocity(true);
  Fixture::check_local_solvers_velocity(false);
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::check_local_solvers_velocity(true);
  Fixture::check_local_solvers_velocity(false);
}



TYPED_TEST_P(TestStokesIntegrator, CheckLocalSolversDGVelocity)
{
  using Fixture = TestStokesIntegrator<TypeParam>;
  ASSERT_TRUE(false) << "TODO...";
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(true);
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(false);
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(true);
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(false);
}



TYPED_TEST_P(TestStokesIntegrator, CheckSystemMatrix)
{
  using Fixture = TestStokesIntegrator<TypeParam>;
  ASSERT_TRUE(false) << "TODO...";
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::check_system_matrix(true);
  Fixture::check_system_matrix(false);
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::check_system_matrix(true);
  Fixture::check_system_matrix(false);
}



TYPED_TEST_P(TestStokesIntegrator, CheckSystemRHS)
{
  using Fixture = TestStokesIntegrator<TypeParam>;
  ASSERT_TRUE(false) << "TODO...";
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::check_system_matrix(false, true);
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::check_system_matrix(false, true);
}



TYPED_TEST_P(TestStokesIntegrator, CheckLevelMatrixVelocityPressure)
{
  using Fixture = TestStokesIntegrator<TypeParam>;
  ASSERT_TRUE(false) << "TODO...";
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::check_local_solvers_block(false, true);
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::check_local_solvers_block(false, true);
}



TYPED_TEST_P(TestStokesIntegrator, CheckLocalSolversVelocityPressure)
{
  using Fixture = TestStokesIntegrator<TypeParam>;
  ASSERT_TRUE(false) << "TODO...";
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 0;
  Fixture::check_local_solvers_block(true);
  Fixture::check_local_solvers_block(false);
  Fixture::options.prms.mesh.n_refinements = 1;
  Fixture::check_local_solvers_block(true);
  Fixture::check_local_solvers_block(false);
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwQ_velocityvelocity_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::Qkplus2_DGPk>({0U, 0U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::Qkplus2_DGPk>({0U, 0U});
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwQ_velocitypressure_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::Qkplus2_DGPk>({0U, 1U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::Qkplus2_DGPk>({0U, 1U});
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwQ_pressurevelocity_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::Qkplus2_DGPk>({1U, 0U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::Qkplus2_DGPk>({1U, 0U});
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwDGQ_velocityvelocity_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::DGQkplus2_DGPk>({0U, 0U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::DGQkplus2_DGPk>({0U, 0U});
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwDGQ_velocitypressure_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::DGQkplus2_DGPk>({0U, 1U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::DGQkplus2_DGPk>({0U, 1U});
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwDGQ_pressurevelocity_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::DGQkplus2_DGPk>({1U, 0U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::DGQkplus2_DGPk>({1U, 0U});
}



TYPED_TEST_P(TestStokesIntegrator, matrixintegratorlmwRT_velocityvelocity_MPI)
{
  using Fixture                               = TestStokesIntegrator<TypeParam>;
  Fixture::options.prms.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::options.prms.mesh.n_repetitions    = 2;
  Fixture::options.prms.mesh.n_refinements    = 1;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::RaviartThomas>({0U, 0U});
  Fixture::options.prms.mesh.n_refinements = 2;
  Fixture::template check_matrixintegratorlmw<Stokes::Method::RaviartThomas>({0U, 0U});
}



REGISTER_TYPED_TEST_SUITE_P(TestStokesIntegrator,
                            CheckSystemMatrixVelocity,
                            CheckLocalSolversVelocityMPI,
                            CheckLocalSolversDGVelocity,
                            CheckSystemMatrix,
                            CheckSystemRHS,
                            CheckLevelMatrixVelocityPressure,
                            CheckLocalSolversVelocityPressure,
                            matrixintegratorlmwQ_velocityvelocity_MPI,
                            matrixintegratorlmwQ_velocitypressure_MPI,
                            matrixintegratorlmwQ_pressurevelocity_MPI,
                            matrixintegratorlmwDGQ_velocityvelocity_MPI,
                            matrixintegratorlmwDGQ_velocitypressure_MPI,
                            matrixintegratorlmwDGQ_pressurevelocity_MPI,
                            matrixintegratorlmwRT_velocityvelocity_MPI);



using TestParamsConstant  = testing::Types<Util::NonTypeParams<2, 0>>;
using TestParamsLinear    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
// using TestParamsHighOrder = testing::Types<Util::NonTypeParams<2, 5>>;



INSTANTIATE_TYPED_TEST_SUITE_P(Constant2D, TestStokesIntegrator, TestParamsConstant);
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestStokesIntegrator, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestStokesIntegrator, TestParamsQuadratic);
// INSTANTIATE_TYPED_TEST_SUITE_P(HighOrder2D, TestStokesIntegrator, TestParamsHighOrder);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_stokes_integrator.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
