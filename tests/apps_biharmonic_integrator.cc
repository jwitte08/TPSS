
/*
 * Test Laplace integrators.
 *
 *  Created on: Apr 07, 2020
 *      Author: witte
 */

/// googletest
#include <gtest/gtest.h>

/// apps/
#include "biharmonic_problem.h"

/// tests/
#include "test_utilities.h"

using namespace dealii;
using namespace Biharmonic;



template<typename T>
class TestBiharmonicIntegrator : public testing::Test
{
protected:
  static constexpr int dim                 = T::template value<0>();
  static constexpr int fe_degree           = T::template value<1>();
  using BiharmonicProblem                  = ModelProblem<dim>;
  static constexpr unsigned int macro_size = VectorizedArray<double>::size();


  void
  SetUp() override
  {
    ofs.open("apps_biharmonic_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout_owned              = std::make_shared<ConditionalOStream>(ofs, is_first_proc);
  }


  void
  TearDown() override
  {
    ofs.close();
  }


  void
  initialize()
  {
    const auto initialize_problem = [&](auto & new_problem) {
      new_problem->pcout = pcout_owned;
      new_problem->make_grid();
      new_problem->triangulation.refine_global(rt_parameters.mesh.n_refinements);
      new_problem->setup_system();
      new_problem->assemble_system();
    };

    /// initialize biharmonic model problem
    const auto new_problem = std::make_shared<BiharmonicProblem>(fe_degree);
    initialize_problem(new_problem);
    biharmonic_problem = new_problem;

    /// matrix-free dummy
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.mg_level = biharmonic_problem->max_level();
      QGauss<1>  quadrature(fe_degree + 1);
      const auto mf_storage_ = std::make_shared<MatrixFree<dim, double>>();
      mf_storage_->reinit(biharmonic_problem->mapping,
                          biharmonic_problem->dof_handler,
                          biharmonic_problem->constraints,
                          quadrature,
                          additional_data);
      this->mf_storage = mf_storage_;
    }

    /// distribute subdomains
    {
      typename SubdomainHandler<dim, double>::AdditionalData additional_data;
      additional_data.level            = biharmonic_problem->max_level();
      additional_data.smoother_variant = TPSS::SmootherVariant::additive;
      additional_data.patch_variant    = TPSS::PatchVariant::vertex;
      const auto subdomain_handler_    = std::make_shared<SubdomainHandler<dim, double>>();
      subdomain_handler_->reinit(mf_storage, additional_data);
      this->subdomain_handler = subdomain_handler_;
    }
  }


  void
  check_local_matrices()
  {
    initialize();

    TPSS::PatchTransfer<dim, double, fe_degree> patch_transfer(*subdomain_handler);
    const auto & patch_worker = patch_transfer.get_patch_dof_worker();
    const auto   n_subdomains = patch_worker.get_partition_data().n_subdomains();

    /// assemble local matrices (as Kronecker tensor)
    using FDIntegrator = C0IP::FD::MatrixIntegrator<dim, fe_degree, double>;
    using LocalMatrix  = typename FDIntegrator::matrix_type;
    std::vector<LocalMatrix> local_matrices(n_subdomains);
    FDIntegrator             integrator;
    integrator.template assemble_subspace_inverses<bool>(
      *subdomain_handler,
      local_matrices,
      /*dummy*/ false,
      patch_worker.get_partition_data().get_patch_range());

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
        local_matrix.extract_submatrix_from(biharmonic_problem->system_matrix,
                                            dof_indices_on_patch,
                                            dof_indices_on_patch);
        const auto local_matrix_tp = table_to_fullmatrix(local_matrices[patch].as_table(), lane);
        compare_matrix(local_matrix_tp, local_matrix);

        const auto local_inverse_tp =
          table_to_fullmatrix(local_matrices[patch].as_inverse_table(), lane);
        FullMatrix<double> C(local_matrix.m());
        local_inverse_tp.mmult(C, local_matrix);
        C.print_formatted(std::cout);
        const auto eigenvalues = compute_eigenvalues(C);
        std::cout << vector_to_string(eigenvalues) << std::endl;
      }
    }
  }


  void
  check_1d_eigendecompositions()
  {
    initialize();

    using FDIntegrator = C0IP::FD::MatrixIntegrator<dim, fe_degree, double>;
    using FDEvaluator  = typename FDIntegrator::evaluator_type;
    FDIntegrator integrator;
    FDEvaluator  eval(*subdomain_handler);

    const auto patch = 0U;
    eval.reinit(patch);
    const auto BB   = integrator.template assemble_bilaplace_tensor<false>(eval);
    const auto BBip = integrator.assemble_bilaplace_tensor(eval);
    const auto LL   = integrator.assemble_laplace_tensor(eval);
    const auto LLip = integrator.template assemble_laplace_tensor<true>(eval);
    const auto MM   = integrator.assemble_mass_tensor(eval);

    const auto         lane = 0U;
    auto               B    = table_to_fullmatrix(BB[0], lane);
    FullMatrix<double> Qb(B.m(), B.n());
    const auto         eigenvalues_B = compute_eigenvalues_symm(B, Qb);
    auto               Bip           = table_to_fullmatrix(BBip[0], lane);
    FullMatrix<double> Qbip(Bip.m(), Bip.n());
    const auto         eigenvalues_Bip = compute_eigenvalues_symm(Bip, Qbip);
    auto               L               = table_to_fullmatrix(LL[0], lane);
    FullMatrix<double> Ql(L.m(), L.n());
    const auto         eigenvalues_L = compute_eigenvalues_symm(L, Ql);
    auto               Lip = table_to_fullmatrix(LLip[0], lane); // same as L due to zero-jumps!
    FullMatrix<double> Qlip(Lip.m(), Lip.n());
    const auto         eigenvalues_Lip = compute_eigenvalues_symm(Lip, Qlip);
    auto               M               = table_to_fullmatrix(MM[0], lane);
    FullMatrix<double> Qm(M.m(), M.n());
    const auto         eigenvalues_M = compute_eigenvalues_symm(M, Qm);
    FullMatrix<double> sqL(L.m(), L.n());
    L.mmult(sqL, L);
    FullMatrix<double> Qsql(sqL.m(), sqL.n());
    const auto         eigenvalues_sqL = compute_eigenvalues_symm(sqL, Qsql);

    std::cout << "Eigenvalues(B):" << std::endl;
    eigenvalues_B.print(std::cout);
    std::cout << "Eigenvalues(Bip):" << std::endl;
    eigenvalues_Bip.print(std::cout);
    std::cout << "Eigenvalues(L):" << std::endl;
    eigenvalues_L.print(std::cout);
    std::cout << "Eigenvalues(Lip):" << std::endl;
    eigenvalues_Lip.print(std::cout);
    std::cout << "Eigenvalues(M):" << std::endl;
    eigenvalues_M.print(std::cout);
    std::cout << "Eigenvalues(L^2):" << std::endl;
    eigenvalues_sqL.print(std::cout);

    // // std::cout << "compare Qb and Ql:" << std::endl;
    // // compare_matrix(Qb, Ql);
    // // std::cout << "compare Qbip and Ql:" << std::endl;
    // // compare_matrix(Qbip, Ql);
    // // std::cout << "compare Qbip and Qsql:" << std::endl;
    // // compare_matrix(Qbip, Qsql);
    // // std::cout << "compare Qbip and Qm:" << std::endl;
    // // compare_matrix(Qbip, Qm);

    // // // DEBUG: Qb is orthogonal!
    // // FullMatrix<double> QbT_Qb(Qb.n());
    // // Qb.Tmmult(QbT_Qb, Qb);
    // // QbT_Qb.print_formatted(std::cout);

    // // { // check eigendecomposition of Bip
    // //   FullMatrix<double> Lambda(Bip.m());
    // //   FullMatrix<double> tmp(Bip.m());
    // //   Qbip.Tmmult(tmp, Bip);
    // //   tmp.mmult(Lambda, Qbip);
    // //   std::cout << "check eigendecomposition:   Qbip^T Bip Qbip = Lambda(Bip)" << std::endl;
    // //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    // // }

    // // { // check if eigenvectors of Bip diagonalize L
    // //   FullMatrix<double> Lambda(L.m());
    // //   FullMatrix<double> tmp(L.m());
    // //   Qbip.Tmmult(tmp, L);
    // //   tmp.mmult(Lambda, Qbip);
    // //   std::cout << "check:   Qbip^T L Qbip = Lambda(L)" << std::endl;
    // //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    // // }

    // // { // check if eigenvectors of Bip diagonalize L^2
    // //   FullMatrix<double> Lambda(sqL.m());
    // //   FullMatrix<double> tmp(sqL.m());
    // //   Qbip.Tmmult(tmp, sqL);
    // //   tmp.mmult(Lambda, Qbip);
    // //   std::cout << "check:   Qbip^T L^2 Qbip = Lambda(L^2)" << std::endl;
    // //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    // // }

    // { // generalized eigenproblem of Bip subject to L
    //   FullMatrix<double> Q(Bip.m(), Bip.n());
    //   compute_generalized_eigenvalues_symm(Bip, L, Q);

    //   FullMatrix<double> Lambda(Bip.m());
    //   FullMatrix<double> tmp(Bip.m());
    //   Q.Tmmult(tmp, Bip);
    //   tmp.mmult(Lambda, Q);
    //   std::cout << "check:   Q^T Bip Q = Lambda(Bip)   (generalized eigenvalues subject to L)"
    //             << std::endl;
    //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    // }

    // { // generalized eigenproblem of Bip subject to L^2
    //   FullMatrix<double> Q(Bip.m(), Bip.n());
    //   compute_generalized_eigenvalues_symm(Bip, sqL, Q);

    //   FullMatrix<double> Lambda(Bip.m());
    //   FullMatrix<double> tmp(Bip.m());
    //   Q.Tmmult(tmp, Bip);
    //   tmp.mmult(Lambda, Q);
    //   std::cout << "check:   Q^T Bip Q = Lambda(Bip)   (generalized eigenvalues subject to L^2)"
    //             << std::endl;
    //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    // }

    // // { // generalized eigenproblem of B subject to L
    // //   FullMatrix<double> Q(B.m(), B.n());
    // //   compute_generalized_eigenvalues_symm(B, L, Q);

    // //   FullMatrix<double> Lambda(B.m());
    // //   FullMatrix<double> tmp(B.m());
    // //   Q.Tmmult(tmp, B);
    // //   tmp.mmult(Lambda, Q);
    // //   std::cout << "check:   Q^T B Q = Lambda(B)   (generalized eigenvalues subject to L)"
    // //             << std::endl;
    // //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    // // }

    // { // check if gen. eigenvectors of Bip subject to M diagonalize L or L^2
    //   FullMatrix<double> Q(Bip.m(), Bip.n());
    //   compute_generalized_eigenvalues_symm(Bip, M, Q);

    //   FullMatrix<double> Lambda(Bip.m());
    //   FullMatrix<double> tmp(Bip.m());
    //   Q.Tmmult(tmp, Bip);
    //   tmp.mmult(Lambda, Q);
    //   std::cout << "check:   Q^T Bip Q = Lambda(Bip)   (generalized eigenvalues subject to M)"
    //             << std::endl;
    //   Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);

    //   {
    //     FullMatrix<double> Lambda(L.m());
    //     FullMatrix<double> tmp(L.m());
    //     Q.Tmmult(tmp, L);
    //     tmp.mmult(Lambda, Q);
    //     std::cout << "check:   Q^T L Q = Lambda(L)" << std::endl;
    //     Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    //   }
    //   {
    //     FullMatrix<double> Lambda(sqL.m());
    //     FullMatrix<double> tmp(sqL.m());
    //     Q.Tmmult(tmp, sqL);
    //     tmp.mmult(Lambda, Q);
    //     std::cout << "check:   Q^T L^2 Q = Lambda(L^2)" << std::endl;
    //     Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
    //   }
    // }

    // {
    //   FullMatrix<double> Q(Bip.m(), Bip.n());
    //   compute_generalized_eigenvalues_symm(Bip, M, Q);

    //   FullMatrix<double> Lambda(Bip.m());
    //   FullMatrix<double> tmp(Bip.m());
    //   Q.Tmmult(tmp, Bip);
    //   tmp.mmult(Lambda, Q);
    //   FullMatrix<double> invsqrtLambda(Bip.m());
    //   for(auto i = 0U; i < Lambda.m(); ++i)
    //     invsqrtLambda(i, i) = 1. / std::sqrt(Lambda(i, i));
    //   FullMatrix<double> invsqrtBip(Bip.m());
    // }

    {
      std::cout << "\nConstructing 1D reference problem" << std::endl;
      Triangulation<1> tria;
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(1);
      FE_Q<1>       fe(fe_degree);
      DoFHandler<1> dofh;
      dofh.initialize(tria, fe);
      std::cout << "Distributed " << dofh.n_dofs() << " dofs subject to " << fe.get_name()
                << std::endl;

      /// assemble mass matrix
      QGauss<1>                            quad(fe_degree + 1);
      FEValues<1>                          fev(fe, quad, update_values | update_JxW_values);
      const auto                           n_dofs_per_cell = fe.dofs_per_cell;
      const auto                           n_q_points      = quad.size();
      FullMatrix<double>                   Mass(dofh.n_dofs(), dofh.n_dofs());
      FullMatrix<double>                   cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
      std::vector<unsigned int>            inner_dof_indices;
      for(const auto & cell : dofh.active_cell_iterators())
      {
        fev.reinit(cell);
        cell_matrix = 0.;
        for(auto q = 0U; q < n_q_points; ++q)
          for(auto i = 0U; i < n_dofs_per_cell; ++i)
            for(auto j = 0U; j < n_dofs_per_cell; ++j)
              cell_matrix(i, j) += fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);

        cell->get_dof_indices(dof_indices);
        for(auto i = 0U; i < n_dofs_per_cell; ++i)
          for(auto j = 0U; j < n_dofs_per_cell; ++j)
            Mass.add(dof_indices[i], dof_indices[j], cell_matrix(i, j));

        const bool is_left_cell  = cell->index() == 0;
        const bool is_right_cell = cell->index() == 1;
        Assert(is_left_cell || is_right_cell, ExcMessage("only two cells allowed"));
        Assert(dof_indices.size() > 2, ExcMessage("at least quadratic polynomials"));
        if(is_left_cell)
        {
          /// push back interior dofs
          std::copy(dof_indices.cbegin() + 2,
                    dof_indices.cend(),
                    std::back_inserter(inner_dof_indices));
          /// push back dof at interface
          inner_dof_indices.push_back(dof_indices[1]);
        }
        if(is_right_cell)
        {
          /// push back interior dofs
          std::copy(dof_indices.cbegin() + 2,
                    dof_indices.cend(),
                    std::back_inserter(inner_dof_indices));
        }
      }
      AssertDimension(inner_dof_indices.size(), dofh.n_dofs() - 2);
      std::cout << "Global DoFs on patch in lexicographic order: "
                << vector_to_string(inner_dof_indices) << std::endl;
      FullMatrix<double> mass(dofh.n_dofs() - 2, dofh.n_dofs() - 2);
      mass.extract_submatrix_from(Mass, inner_dof_indices, inner_dof_indices);

      /// compare mass matrices
      FullMatrix<double> Qmass(mass.m(), mass.n());
      const auto         eigenvalues_mass = compute_eigenvalues_symm(mass, Qmass);
      std::cout << "Eigenvalues(mass):" << std::endl;
      eigenvalues_mass.print(std::cout);
      compare_matrix(M, mass);

      /// store evaluations of shape functions (phi_i) in GNUPLOT data files
      for(auto i = 0U; i < inner_dof_indices.size(); ++i)
      {
        DataOut<1> data_out;
        data_out.attach_dof_handler(dofh);
        std::string    name = "phi_" + Utilities::int_to_string(i);
        const auto     dof  = inner_dof_indices[i];
        Vector<double> e_dof(dofh.n_dofs());
        e_dof(dof) = 1.;
        data_out.add_data_vector(e_dof, name, DataOut<1>::type_dof_data);
        data_out.build_patches(/*n_subdivisions*/ 50);
        std::ofstream file(name + ".dat");
        data_out.write_gnuplot(file);
      }

      const auto write_eigenvalue_data = [&](const std::vector<double> & lambdas,
                                             const std::string           prefix = "eigenvalues") {
        AssertDimension(lambdas.size(), inner_dof_indices.size());
        std::ofstream file(prefix + ".dat");
        for(auto i = 0U; i < inner_dof_indices.size(); ++i)
          file << i << " " << lambdas[i] << "\n";
      };

      const auto write_eigenfunction_data = [&](const FullMatrix<double> & Q,
                                                const std::string prefix = "eigenfunction") {
        AssertDimension(Q.n(), inner_dof_indices.size());
        for(auto i = 0U; i < inner_dof_indices.size(); ++i)
        {
          DataOut<1> data_out;
          data_out.attach_dof_handler(dofh);
          std::string    name = prefix + "_" + Utilities::int_to_string(i);
          Vector<double> v(dofh.n_dofs()); // eigenvector
          AssertDimension(Q.m(), inner_dof_indices.size());
          for(auto j = 0U; j < Q.m(); ++j)
          {
            const auto dof = inner_dof_indices[j];
            v(dof)         = Q(i, j);
          }
          data_out.add_data_vector(v, name, DataOut<1>::type_dof_data);
          data_out.build_patches(/*n_subdivisions*/ 100);
          std::ofstream file(name + ".dat");
          data_out.write_gnuplot(file);
        }
      };

      {
        FullMatrix<double> Q(Bip.m(), Bip.n());
        const auto         eigenvalues = compute_generalized_eigenvalues_symm(Bip, sqL, Q);
        std::cout << vector_to_string(eigenvalues) << std::endl;

        FullMatrix<double> Lambda(Bip.m());
        FullMatrix<double> tmp(Bip.m());
        Q.Tmmult(tmp, Bip);
        tmp.mmult(Lambda, Q);
        FullMatrix<double> Lambda_ref(Q.m());
        for(auto i = 0U; i < eigenvalues.size(); ++i)
          Lambda_ref(i, i) = eigenvalues[i];
        compare_matrix(Lambda, Lambda_ref);
        // Lambda.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-8);
        write_eigenfunction_data(Q, "EF_Bip_wrt_sqL");
        write_eigenvalue_data(eigenvalues, "EV_Bip_wrt_sqL");
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
  compare_inverse_matrix(const FullMatrix<double> & inverse_patch_matrix,
                         const FullMatrix<double> & other) const
  {
    Util::compare_inverse_matrix(inverse_patch_matrix, other, *pcout_owned);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  RT::Parameter                                  rt_parameters;
  std::shared_ptr<const BiharmonicProblem>       biharmonic_problem;
  std::shared_ptr<const MatrixFree<dim, double>> mf_storage;
  std::shared_ptr<SubdomainHandler<dim, double>> subdomain_handler;
};



TYPED_TEST_SUITE_P(TestBiharmonicIntegrator);

TYPED_TEST_P(TestBiharmonicIntegrator, CheckVertexPatchMatrix)
{
  using Fixture                             = TestBiharmonicIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.n_refinements = 0;
  Fixture::check_local_matrices();
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_local_matrices();
}

TYPED_TEST_P(TestBiharmonicIntegrator, CheckVertexPatchEigendecomp1D)
{
  using Fixture                             = TestBiharmonicIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.n_refinements = 0;
  Fixture::check_1d_eigendecompositions();
}

REGISTER_TYPED_TEST_SUITE_P(TestBiharmonicIntegrator,
                            CheckVertexPatchMatrix,
                            CheckVertexPatchEigendecomp1D);

using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
using TestParamsHighOrder =
  testing::Types<Util::NonTypeParams<2, 5>, Util::NonTypeParams<2, 7>, Util::NonTypeParams<2, 11>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestBiharmonicIntegrator, TestParamsQuadratic);
INSTANTIATE_TYPED_TEST_SUITE_P(HighOrder2D, TestBiharmonicIntegrator, TestParamsHighOrder);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("apps_biharmonic_integrator.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
