
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
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();
  using BiharmonicProblem        = ModelProblem<dim, fe_degree>;
  using PatchTransfer = typename C0IP::FD::MatrixIntegrator<dim, fe_degree>::transfer_type;
  static constexpr unsigned int macro_size = VectorizedArray<double>::size();


  void
  SetUp() override
  {
    ofs.open("apps_biharmonic_integrator.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const bool is_quadratic  = fe_degree == 2;
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, is_quadratic && is_first_proc);

    rt_parameters.multigrid.pre_smoother.schwarz.patch_variant    = TPSS::PatchVariant::vertex;
    rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant = TPSS::SmootherVariant::additive;
    rt_parameters.multigrid.post_smoother.schwarz = rt_parameters.multigrid.pre_smoother.schwarz;
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
      // new_problem->triangulation.refine_global(rt_parameters.mesh.n_refinements);
      new_problem->setup_system();
      new_problem->assemble_system();
    };

    /// initialize biharmonic model problem
    const auto new_problem = std::make_shared<BiharmonicProblem>(rt_parameters);
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
  check_level_matrix()
  {
    rt_parameters.multigrid.pre_smoother.variant  = SmootherParameter::SmootherVariant::None;
    rt_parameters.multigrid.post_smoother.variant = rt_parameters.multigrid.pre_smoother.variant;

    biharmonic_problem        = std::make_shared<BiharmonicProblem>(rt_parameters);
    biharmonic_problem->pcout = pcout_owned;
    biharmonic_problem->make_grid();
    biharmonic_problem->setup_system();
    biharmonic_problem->assemble_system();
    biharmonic_problem->prepare_multigrid();

    const auto & system_matrix = biharmonic_problem->system_matrix;
    const auto   level         = biharmonic_problem->max_level();
    const auto & level_matrix  = biharmonic_problem->mg_matrices[level];

    if(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
    {
      FullMatrix<double> system_fullmatrix(system_matrix.m(), system_matrix.n());
      system_fullmatrix.copy_from(
        static_cast<const TrilinosWrappers::SparseMatrix &>(system_matrix));
      FullMatrix<double> level_fullmatrix(level_matrix.m(), level_matrix.n());
      level_fullmatrix.copy_from(static_cast<const TrilinosWrappers::SparseMatrix &>(level_matrix));
      compare_matrix(system_fullmatrix, level_fullmatrix);
    }

    else
    {
      const auto partitioner       = system_matrix.partitioner;
      const auto partitioner_level = level_matrix.partitioner;
      (void)partitioner, (void)partitioner_level;
      Assert(partitioner->is_compatible(*partitioner_level),
             ExcMessage("mismatching vector partitioners"));

      LinearAlgebra::distributed::Vector<double> src, dst, dst_level;
      system_matrix.initialize_dof_vector(src);
      fill_with_random_values(src);
      system_matrix.initialize_dof_vector(dst);
      dst_level = dst;

      system_matrix.vmult(dst, src);
      level_matrix.vmult(dst_level, src);

      compare_vector(dst_level, dst);
    }
  }


  void
  check_local_matrices()
  {
    initialize();

    PatchTransfer patch_transfer(*subdomain_handler);
    const auto &  patch_worker = patch_transfer.get_patch_dof_worker();
    const auto    n_subdomains = patch_worker.get_partition_data().n_subdomains();

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
        C.print_formatted(pcout_owned->get_stream());
        const auto eigenvalues = compute_eigenvalues(C);
        *pcout_owned << vector_to_string(eigenvalues) << std::endl;
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

    const auto check_eigenvectors =
      [this](const auto & A, const auto & Q, const auto & eigenvalues) {
        FullMatrix<double> Lambda(A.m());
        FullMatrix<double> tmp(A.m());
        Q.Tmmult(tmp, A);
        tmp.mmult(Lambda, Q);
        FullMatrix<double> Lambda_ref(Q.m());
        for(auto i = 0U; i < eigenvalues.size(); ++i)
          Lambda_ref(i, i) = eigenvalues[i];
        compare_matrix(Lambda, Lambda_ref);
      };

    const auto         lane = 0U;
    auto               B    = table_to_fullmatrix(BB[0], lane);
    FullMatrix<double> Qb(B.m(), B.n());
    const auto         eigenvalues_B = compute_eigenvalues_symm(B, Qb);
    check_eigenvectors(B, Qb, eigenvalues_B);
    auto               Bip = table_to_fullmatrix(BBip[0], lane);
    FullMatrix<double> Qbip(Bip.m(), Bip.n());
    const auto         eigenvalues_Bip = compute_eigenvalues_symm(Bip, Qbip);
    check_eigenvectors(Bip, Qbip, eigenvalues_Bip);
    auto               L = table_to_fullmatrix(LL[0], lane);
    FullMatrix<double> Ql(L.m(), L.n());
    const auto         eigenvalues_L = compute_eigenvalues_symm(L, Ql);
    check_eigenvectors(L, Ql, eigenvalues_L);
    auto               Lip = table_to_fullmatrix(LLip[0], lane); // same as L due to zero-jumps!
    FullMatrix<double> Qlip(Lip.m(), Lip.n());
    const auto         eigenvalues_Lip = compute_eigenvalues_symm(Lip, Qlip);
    check_eigenvectors(Lip, Qlip, eigenvalues_Lip);
    auto               M = table_to_fullmatrix(MM[0], lane);
    FullMatrix<double> Qm(M.m(), M.n());
    const auto         eigenvalues_M = compute_eigenvalues_symm(M, Qm);
    check_eigenvectors(M, Qm, eigenvalues_M);
    FullMatrix<double> sqL(L.m(), L.n());
    L.mmult(sqL, L);
    FullMatrix<double> Qsql(sqL.m(), sqL.n());
    const auto         eigenvalues_sqL = compute_eigenvalues_symm(sqL, Qsql);
    check_eigenvectors(sqL, Qsql, eigenvalues_sqL);

    // // std::cout << "Eigenvalues(B):" << std::endl;
    // // eigenvalues_B.print(std::cout);
    // // std::cout << "Eigenvalues(Bip):" << std::endl;
    // // eigenvalues_Bip.print(std::cout);
    // // std::cout << "Eigenvalues(L):" << std::endl;
    // // eigenvalues_L.print(std::cout);
    // // std::cout << "Eigenvalues(Lip):" << std::endl;
    // // eigenvalues_Lip.print(std::cout);
    // // std::cout << "Eigenvalues(M):" << std::endl;
    // // eigenvalues_M.print(std::cout);
    // // std::cout << "Eigenvalues(L^2):" << std::endl;
    // // eigenvalues_sqL.print(std::cout);

    // {
    //   *pcout_owned << "\nIs L^2 diagonalizable in terms of eigenvectors Ql of L?" << std::endl;
    //   FullMatrix<double> Lambda(sqL.m());
    //   FullMatrix<double> tmp(sqL.m());
    //   Qsql.Tmmult(tmp, sqL);
    //   tmp.mmult(Lambda, Qsql);
    //   FullMatrix<double> Lambda_ref(Qsql.m());
    //   for(auto i = 0U; i < eigenvalues_sqL.size(); ++i)
    //     Lambda_ref(i, i) = eigenvalues_sqL[i];
    //   compare_matrix(Lambda, Lambda_ref);

    //   // is diagonalizable?
    //   Ql.Tmmult(tmp, sqL);
    //   tmp.mmult(Lambda, Ql);
    //   compare_matrix(Lambda, Lambda_ref);
    // }
    // {
    //   *pcout_owned << "\nIs Bip diagonalizable in terms of eigenvectors Ql of L?" << std::endl;
    //   FullMatrix<double> Lambda(Bip.m());
    //   FullMatrix<double> tmp(Bip.m());
    //   Qbip.Tmmult(tmp, Bip);
    //   tmp.mmult(Lambda, Qbip);
    //   FullMatrix<double> Lambda_ref(Qbip.m());
    //   for(auto i = 0U; i < eigenvalues_Bip.size(); ++i)
    //     Lambda_ref(i, i) = eigenvalues_Bip[i];
    //   compare_matrix(Lambda, Lambda_ref);

    //   // is diagonalizable?
    //   Ql.Tmmult(tmp, Bip);
    //   tmp.mmult(Lambda, Ql);
    //   compare_matrix(Lambda, Lambda_ref);
    // }
    // {
    //   *pcout_owned << "\nIs B diagonalizable in terms of eigenvectors Ql of L?" << std::endl;
    //   FullMatrix<double> Lambda(B.m());
    //   FullMatrix<double> tmp(B.m());
    //   Qb.Tmmult(tmp, B);
    //   tmp.mmult(Lambda, Qb);
    //   FullMatrix<double> Lambda_ref(Qb.m());
    //   for(auto i = 0U; i < eigenvalues_B.size(); ++i)
    //     Lambda_ref(i, i) = eigenvalues_B[i];
    //   compare_matrix(Lambda, Lambda_ref);

    //   // is diagonalizable?
    //   Ql.Tmmult(tmp, B);
    //   tmp.mmult(Lambda, Ql);
    //   compare_matrix(Lambda, Lambda_ref);
    //   compare_matrix(Qb, Ql);
    // }

    {
      *pcout_owned << "\nConstructing 1D reference problem" << std::endl;
      Triangulation<1> tria;
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(1);
      FE_Q<1>       fe(fe_degree);
      DoFHandler<1> dofh;
      dofh.initialize(tria, fe);
      *pcout_owned << "Distributed " << dofh.n_dofs() << " dofs subject to " << fe.get_name()
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
      *pcout_owned << "Global DoFs on patch in lexicographic order: "
                   << vector_to_string(inner_dof_indices) << std::endl;
      FullMatrix<double> mass(dofh.n_dofs() - 2, dofh.n_dofs() - 2);
      mass.extract_submatrix_from(Mass, inner_dof_indices, inner_dof_indices);

      /// compare mass matrices
      FullMatrix<double> Qmass(mass.m(), mass.n());
      const auto         eigenvalues_mass = compute_eigenvalues_symm(mass, Qmass);
      *pcout_owned << "Eigenvalues(mass):" << std::endl;
      eigenvalues_mass.print(ofs);
      compare_matrix(M, mass);

      /// store evaluations of shape functions (phi_i) in GNUPLOT data files
      constexpr unsigned int n_subdivisions = 400;
      for(auto i = 0U; i < inner_dof_indices.size(); ++i)
      {
        DataOut<1> data_out;
        data_out.attach_dof_handler(dofh);
        std::string    name = "phi_" + Utilities::int_to_string(i);
        const auto     dof  = inner_dof_indices[i];
        Vector<double> e_dof(dofh.n_dofs());
        e_dof(dof) = 1.;
        data_out.add_data_vector(e_dof, name, DataOut<1>::type_dof_data);
        data_out.build_patches(n_subdivisions);
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
          v = 0.;
          AssertDimension(Q.m(), inner_dof_indices.size());
          for(auto j = 0U; j < Q.m(); ++j)
          {
            const auto dof = inner_dof_indices[j];
            v(dof)         = Q(j, i);
          }
          data_out.add_data_vector(v, name, DataOut<1>::type_dof_data);
          data_out.build_patches(n_subdivisions);
          std::ofstream file(name + ".dat");
          data_out.write_gnuplot(file);
        }
      };

      { // write eigendecomposition data:   Bip q = \lambda L^2 q
        FullMatrix<double> Q(Bip.m(), Bip.n());
        const auto         eigenvalues = compute_generalized_eigenvalues_symm(Bip, sqL, Q);
        check_eigenvectors(Bip, Q, eigenvalues);
        write_eigenfunction_data(Q, "EF_Bip_wrt_sqL");
        write_eigenvalue_data(eigenvalues, "EV_Bip_wrt_sqL");
      }
      { // write eigendecomposition data:   L q = \lambda M q
        FullMatrix<double> Q(L.m(), L.n());
        const auto         eigenvalues = compute_generalized_eigenvalues_symm(L, M, Q);
        check_eigenvectors(L, Q, eigenvalues);
        write_eigenfunction_data(Q, "EF_L_wrt_M");
        write_eigenvalue_data(eigenvalues, "EV_L_wrt_M");
      }
      { // write eigendecomposition data:   Bip q = \lambda M q
        FullMatrix<double> Q(Bip.m(), Bip.n());
        const auto         eigenvalues = compute_generalized_eigenvalues_symm(Bip, M, Q);
        check_eigenvectors(Bip, Q, eigenvalues);
        write_eigenfunction_data(Q, "EF_Bip_wrt_M");
        write_eigenvalue_data(eigenvalues, "EV_Bip_wrt_M");
      }

      { // write eigendecomposition data:   Bip q = \lambda q
        std::vector<double> evs(eigenvalues_Bip.begin(), eigenvalues_Bip.end());
        write_eigenfunction_data(Qbip, "EF_Bip");
        write_eigenvalue_data(evs, "EV_Bip");
      }
      { // write eigendecomposition data:   B q = \lambda q
        std::vector<double> evs(eigenvalues_B.begin(), eigenvalues_B.end());
        write_eigenfunction_data(Qb, "EF_B");
        write_eigenvalue_data(evs, "EV_B");
      }
      { // write eigendecomposition data:   L q = \lambda q
        std::vector<double> evs(eigenvalues_L.begin(), eigenvalues_L.end());
        write_eigenfunction_data(Ql, "EF_L");
        write_eigenvalue_data(evs, "EV_L");
      }
      { // write eigendecomposition data:   L^2 q = \lambda q
        std::vector<double> evs(eigenvalues_sqL.begin(), eigenvalues_sqL.end());
        write_eigenfunction_data(Qsql, "EF_sqL");
        write_eigenvalue_data(evs, "EV_sqL");
      }
      { // write eigendecomposition data:   M q = \lambda q
        std::vector<double> evs(eigenvalues_M.begin(), eigenvalues_M.end());
        write_eigenfunction_data(Qm, "EF_M");
        write_eigenvalue_data(evs, "EV_M");
      }
    }
  }


  void
  compare_matrix(const FullMatrix<double> & matrix, const FullMatrix<double> & other) const
  {
    Util::compare_matrix(matrix, other, *pcout_owned);
  }


  void
  compare_inverse_matrix(const FullMatrix<double> & inverse_matrix,
                         const FullMatrix<double> & other) const
  {
    Util::compare_inverse_matrix(inverse_matrix, other, *pcout_owned);
  }


  template<typename VectorType>
  void
  compare_vector(const VectorType & vec, const VectorType & other) const
  {
    Util::compare_vector(vec, other, *pcout_owned);
  }


  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;

  RT::Parameter                                  rt_parameters;
  std::shared_ptr<BiharmonicProblem>             biharmonic_problem;
  std::shared_ptr<const MatrixFree<dim, double>> mf_storage;
  std::shared_ptr<SubdomainHandler<dim, double>> subdomain_handler;
};



TYPED_TEST_SUITE_P(TestBiharmonicIntegrator);

TYPED_TEST_P(TestBiharmonicIntegrator, CheckVertexPatchMatrix)
{
  using Fixture                                = TestBiharmonicIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_local_matrices();
  Fixture::rt_parameters.mesh.n_refinements = 1;
  Fixture::check_local_matrices();
}

TYPED_TEST_P(TestBiharmonicIntegrator, CheckLevelMatrixMPI)
{
  using Fixture                                = TestBiharmonicIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 1;
  Fixture::check_level_matrix();
  Fixture::rt_parameters.mesh.n_refinements = 2;
  Fixture::check_level_matrix();
}

TYPED_TEST_P(TestBiharmonicIntegrator, CheckVertexPatchEigendecomp1D)
{
  using Fixture                                = TestBiharmonicIntegrator<TypeParam>;
  Fixture::rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
  Fixture::rt_parameters.mesh.n_repetitions    = 2;
  Fixture::rt_parameters.mesh.n_refinements    = 0;
  Fixture::check_1d_eigendecompositions();
}

REGISTER_TYPED_TEST_SUITE_P(TestBiharmonicIntegrator,
                            CheckLevelMatrixMPI,
                            CheckVertexPatchMatrix,
                            CheckVertexPatchEigendecomp1D);

using TestParamsQuadratic = testing::Types<Util::NonTypeParams<2, 2>>;
// using TestParamsHighOrder = testing::Types<Util::NonTypeParams<2, 5>, Util::NonTypeParams<2, 7>>;
using TestParamsHighOrder = testing::Types<Util::NonTypeParams<2, 3>, Util::NonTypeParams<2, 5>>;

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
