
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
    ofs.open("apps_stokes_integrator.log", std::ios_base::app);
    const bool is_first_proc   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const bool is_higher_order = fe_degree_v > 2;
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, !is_higher_order && is_first_proc);

    {
      auto & pre_smoother                   = rt_parameters.multigrid.pre_smoother;
      pre_smoother.variant                  = SmootherParameter::SmootherVariant::Schwarz;
      pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::vertex;
      pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
      rt_parameters.multigrid.post_smoother = pre_smoother;
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
    rt_parameters.solver.variant = "FGMRES_GMGvelocity";
    using StokesProblem          = ModelProblem<dim, fe_degree_p, Method::TaylorHoodDGQ>;
    std::shared_ptr<const StokesProblem> stokes_problem;
    auto new_problem   = std::make_shared<StokesProblem>(rt_parameters, equation_data);
    new_problem->pcout = pcout_owned;
    new_problem->make_grid();
    new_problem->setup_system();
    new_problem->prepare_multigrid_velocity();
    stokes_problem = new_problem;
    stokes_problem->print_informations();
    const auto & mgc = stokes_problem->mgc_velocity;

    *pcout_owned << "//////////   LINEAR ELASTICITY PROBLEM" << std::endl;
    LinElasticity::EquationData equation_data_elasticity;
    equation_data_elasticity.integrator_variant =
      LinElasticity::EquationData::IntegratorVariant::strain;
    equation_data_elasticity.ip_factor = 1.;
    using ElasticityProblem            = LinElasticity::ModelProblem<dim, fe_degree_p + 1>;
    const auto elasticity_problem =
      std::make_shared<ElasticityProblem>(*pcout_owned, rt_parameters, equation_data_elasticity);
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
    EquationData equation_data;
    rt_parameters.solver.variant = "GMRES_GMG";
    using StokesProblem          = ModelProblem<dim, fe_degree_p, Method::TaylorHood>;

    *pcout_owned << "//////////   STOKES PROBLEM (step-56)" << std::endl;
    auto stokes_problem_step56   = std::make_shared<StokesProblem>(rt_parameters, equation_data);
    stokes_problem_step56->pcout = pcout_owned;
    stokes_problem_step56->make_grid();
    stokes_problem_step56->setup_system();
    stokes_problem_step56->assemble_system_step56();
    stokes_problem_step56->print_informations();

    *pcout_owned << "//////////   STOKES PROBLEM" << std::endl;
    auto stokes_problem   = std::make_shared<StokesProblem>(rt_parameters, equation_data);
    stokes_problem->pcout = pcout_owned;
    stokes_problem->make_grid();
    stokes_problem->setup_system();
    stokes_problem->assemble_system_velocity_pressure();
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


  template<Stokes::Method method = Method::TaylorHood>
  void
  check_local_solvers_velocity(const bool only_diagonal = false)
  {
    EquationData equation_data;
    rt_parameters.solver.variant = "FGMRES_GMGvelocity";
    using StokesProblem          = ModelProblem<dim, fe_degree_p, method>;
    std::shared_ptr<const StokesProblem> stokes_problem;
    auto new_problem   = std::make_shared<StokesProblem>(rt_parameters, equation_data);
    new_problem->pcout = pcout_owned;
    new_problem->make_grid();
    new_problem->setup_system();
    new_problem->prepare_multigrid_velocity();
    stokes_problem = new_problem;
    stokes_problem->print_informations();
    const auto max_level = stokes_problem->max_level();

    using MatrixIntegrator =
      Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree_v, double, StokesProblem::dof_layout_v>;
    using LocalMatrix   = typename MatrixIntegrator::matrix_type;
    using PatchTransfer = TPSS::PatchTransfer<dim, double>;

    const auto & mgc         = stokes_problem->mgc_velocity;
    const auto   mg_smoother = mgc.mg_schwarz_smoother_pre;
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

    const SparseMatrix<double> * system_matrix;
    system_matrix = &(mgc.mg_matrices[max_level]);

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
  check_local_solvers_block()
  {
    constexpr auto method = Method::TaylorHood;

    EquationData equation_data;
    rt_parameters.solver.variant = "GMRES_GMG";
    using StokesProblem          = ModelProblem<dim, fe_degree_p, method>;
    std::shared_ptr<const StokesProblem> stokes_problem;
    auto new_problem   = std::make_shared<StokesProblem>(rt_parameters, equation_data);
    new_problem->pcout = pcout_owned;
    new_problem->make_grid();
    new_problem->setup_system();
    // new_problem->prepare_multigrid();
    stokes_problem = new_problem;
    stokes_problem->print_informations();
    // const auto max_level = stokes_problem->max_level();

    // using MatrixIntegrator =
    //   Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree_v, double,
    //   StokesProblem::dof_layout_v>;
    // using LocalMatrix   = typename MatrixIntegrator::matrix_type;
    // using PatchTransfer = TPSS::PatchTransfer<dim, double>;

    // const auto mg_smoother = stokes_problem->mg_schwarz_smoother_pre;
    // ASSERT_TRUE(mg_smoother) << "mg_smoother is not initialized.";
    // const auto    subdomain_handler = mg_smoother->get_subdomain_handler();
    // PatchTransfer patch_transfer(*subdomain_handler);
    // const auto &  patch_worker = patch_transfer.get_patch_dof_worker();
    // const auto    n_subdomains = patch_worker.get_partition_data().n_subdomains();

    // std::vector<LocalMatrix> local_matrices(n_subdomains);
    // MatrixIntegrator         integrator;
    // integrator.initialize(equation_data);
    // integrator.template assemble_subspace_inverses<bool>(
    //   *subdomain_handler,
    //   local_matrices,
    //   /*dummy*/ false,
    //   patch_worker.get_partition_data().get_patch_range());

    // const SparseMatrix<double> * system_matrix;
    // system_matrix = &(stokes_problem->mg_matrices[max_level]);

    // /// compare local matrices
    // for(auto patch = 0U; patch < n_subdomains; ++patch)
    // {
    //   patch_transfer.reinit(patch);
    //   for(auto lane = 0U; lane < patch_worker.n_lanes_filled(patch); ++lane)
    //   {
    //     std::vector<types::global_dof_index> dof_indices_on_patch;
    //     {
    //       const auto view = patch_transfer.get_dof_indices(lane);
    //       std::copy(view.cbegin(), view.cend(), std::back_inserter(dof_indices_on_patch));
    //     }
    //     FullMatrix<double> local_matrix(dof_indices_on_patch.size());
    //     local_matrix.extract_submatrix_from(*system_matrix,
    //                                         dof_indices_on_patch,
    //                                         dof_indices_on_patch);

    //     auto local_matrix_tp = table_to_fullmatrix(local_matrices[patch].as_table(), lane);

    //     if(only_diagonal)
    //     {
    //       const auto zero_out_block =
    //         [&](auto & matrix, const unsigned int row, const unsigned int col) {
    //           const unsigned int n_dofs = dof_indices_on_patch.size();
    //           AssertDimension(n_dofs % dim, 0);
    //           const unsigned int n_dofs_per_block = n_dofs / dim;
    //           const unsigned     row_start        = row * n_dofs_per_block;
    //           const unsigned     col_start        = col * n_dofs_per_block;
    //           for(auto i = row_start; i < row_start + n_dofs_per_block; ++i)
    //             for(auto j = col_start; j < col_start + n_dofs_per_block; ++j)
    //               matrix(i, j) = 0.;
    //         };
    //       for(auto row = 0U; row < dim; ++row)
    //         for(auto col = 0U; col < dim; ++col)
    //           if(row != col)
    //           {
    //             zero_out_block(local_matrix, row, col);
    //             zero_out_block(local_matrix_tp, row, col);
    //           }
    //     }

    //     compare_matrix(local_matrix_tp, local_matrix);
    //   }
    // }
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
  compare_inverse_matrix(const FullMatrix<double> & inverse_patch_matrix,
                         const FullMatrix<double> & other) const
  {
    Util::compare_inverse_matrix(inverse_patch_matrix, other, *pcout_owned);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  RT::Parameter                                  rt_parameters;
  std::shared_ptr<const MatrixFree<dim, double>> mf_storage;
  std::shared_ptr<SubdomainHandler<dim, double>> subdomain_handler;
};



TYPED_TEST_SUITE_P(TestStokesIntegrator);

TYPED_TEST_P(TestStokesIntegrator, CheckSystemMatrixVelocity)
{
  using Fixture                                = TestStokesIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_system_matrix_velocity();
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_system_matrix_velocity();
}

TYPED_TEST_P(TestStokesIntegrator, CheckLocalSolversVelocity)
{
  using Fixture                                = TestStokesIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_local_solvers_velocity(true);
  Fixture::check_local_solvers_velocity(false);
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_local_solvers_velocity(true);
  Fixture::check_local_solvers_velocity(false);
}

TYPED_TEST_P(TestStokesIntegrator, CheckLocalSolversDGVelocity)
{
  using Fixture                                = TestStokesIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(true);
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(false);
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(true);
  Fixture::template check_local_solvers_velocity<Method::TaylorHoodDGQ>(false);
}

TYPED_TEST_P(TestStokesIntegrator, CheckSystemMatrix)
{
  using Fixture                                = TestStokesIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_system_matrix(true);
  Fixture::check_system_matrix(false);
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_system_matrix(true);
  Fixture::check_system_matrix(false);
}

TYPED_TEST_P(TestStokesIntegrator, CheckSystemRHS)
{
  using Fixture                                = TestStokesIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_system_matrix(false, true);
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_system_matrix(false, true);
}

// TYPED_TEST_P(TestStokesIntegrator, CheckLocalSolversBlock)
// {
//   using Fixture                                = TestStokesIntegrator<TypeParam>;
//   Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
//   Fixture::rt_parameters.mesh.n_repetitions    = 2;
//   Fixture::rt_parameters.mesh.n_refinements    = 0;
//   Fixture::check_local_solvers_block();
// }

REGISTER_TYPED_TEST_SUITE_P(TestStokesIntegrator,
                            CheckSystemMatrixVelocity,
                            CheckLocalSolversVelocity,
                            CheckLocalSolversDGVelocity,
			    CheckSystemMatrix,
			    CheckSystemRHS/*,
					       CheckLocalSolversBlock*/);

using TestParamsConstant  = testing::Types<Util::NonTypeParams<2, 0>>;
using TestParamsLinear    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
using TestParamsHighOrder = testing::Types<Util::NonTypeParams<2, 5>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Constant2D, TestStokesIntegrator, TestParamsConstant);
INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestStokesIntegrator, TestParamsLinear);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestStokesIntegrator, TestParamsQuadratic);
INSTANTIATE_TYPED_TEST_SUITE_P(HighOrder2D, TestStokesIntegrator, TestParamsHighOrder);



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
