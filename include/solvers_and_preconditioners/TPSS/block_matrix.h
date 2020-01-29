/*
 * block_matrix.h
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#ifndef BLOCK_MATRIX_H_
#define BLOCK_MATRIX_H_

#include "kroneckersvd.h"
#include "tensor_product_matrix.h"

using namespace dealii;

namespace Tensors
{
template<int order, typename Number, int n_rows_1d = -1>
class SchurComplementFast : public TensorProductMatrix<order, Number, n_rows_1d>
{
public:
  using matrix_type = TensorProductMatrix<order, Number, n_rows_1d>;
  using value_type  = typename matrix_type::value_type;

  void
  reinit(const matrix_type & A_in,
         const matrix_type & B_in,
         const matrix_type & C_in,
         const matrix_type & D_in,
         const int           lambda_rank    = order,
         const int           kronecker_rank = order)
  {
    static_assert(order == 2);
    using State = typename matrix_type::State;

    AssertThrow(lambda_rank == -1 || lambda_rank > 0, ExcMessage("Invalid lambda_rank."));
    AssertThrow(kronecker_rank == -1 || kronecker_rank > 0, ExcMessage("Invalid kronecker_rank."));
    Assert(A_in.get_state() == State::skd, ExcMessage("Not a separable Kronecker decomposition."));

    /// compute inverse eigenvalues
    const auto            eigenvalues = A_in.get_eigenvalues();
    AlignedVector<Number> inverse_eigenvalues(eigenvalues.size());
    std::transform(eigenvalues.begin(),
                   eigenvalues.end(),
                   inverse_eigenvalues.begin(),
                   [](const auto & lambda) { return static_cast<Number>(1. / lambda); });

    /// compute KSVD of inverse eigenvalue matrix
    const int                                        lambda_rank_max = A_in.m(0);
    std::vector<std::array<Table<2, Number>, order>> ksvd_eigenvalues(
      lambda_rank == -1 ? lambda_rank_max : lambda_rank);
    for(auto & tensor : ksvd_eigenvalues)
      for(auto d = 0U; d < order; ++d)
        tensor[d].reinit(A_in.m(d), A_in.m(d));
    compute_ksvd<Number>(inverse_eigenvalues, ksvd_eigenvalues);

    /// compute elementary tensors representing approximation of -A^{-1}
    std::vector<std::array<Table<2, Number>, order>> eigenvectors(1), eigenvectorsT(1);
    eigenvectors.front() = A_in.get_eigenvectors();
    std::transform(eigenvectors.front().cbegin(),
                   eigenvectors.front().cend(),
                   eigenvectorsT.front().begin(),
                   [](const auto & tab) { return Tensors::transpose(tab); });
    const auto Lambda_QT         = Tensors::product<order, Number>(ksvd_eigenvalues, eigenvectorsT);
    eigenvectors.front().front() = Tensors::scale(-1., eigenvectors.front().front());
    const auto   minus_Q_Lambda_QT = Tensors::product<order, Number>(eigenvectors, Lambda_QT);
    const auto & minus_Ainv        = minus_Q_Lambda_QT;

    /// compute elementary tensors representing approximated S = D - C * A^{-1} * B
    const auto & B_tensors           = B_in.get_elementary_tensors();
    const auto   minus_Ainv_B        = Tensors::product<order, Number>(minus_Ainv, B_tensors);
    const auto & C_tensors           = C_in.get_elementary_tensors();
    const auto   minus_C_Ainv_B      = Tensors::product<order, Number>(C_tensors, minus_Ainv_B);
    auto         schur_tensors       = D_in.get_elementary_tensors();
    auto         schur_tensors_exact = D_in.get_elementary_tensors();
    std::copy(minus_C_Ainv_B.cbegin(),
              minus_C_Ainv_B.cend(),
              std::back_inserter(schur_tensors_exact));
    // // TODO without copy !?
    // std::vector<std::array<Table<2, Number>, order>> tmp;
    // std::remove_copy_if(schur_tensors.cbegin(),
    //                     schur_tensors.cend(),
    //                     std::back_inserter(tmp),
    //                     Tensors::is_nearly_zero<order, Number>);
    // std::swap(schur_tensors, tmp);

    /// exact mode
    if(kronecker_rank == -1)
    {
      matrix_type::reinit(schur_tensors_exact);
      return;
    }

    /// compute the KSVD of the Schur matrix
    const int kronecker_rank_max = D_in.get_elementary_tensors().size() +
                                   C_in.get_elementary_tensors().size() * ksvd_eigenvalues.size() *
                                     B_in.get_elementary_tensors().size();
    std::vector<std::array<Table<2, Number>, order>> ksvd_schur(
      kronecker_rank == -1 ? kronecker_rank_max : kronecker_rank);
    for(auto & tensor : ksvd_schur)
      for(auto d = 0U; d < order; ++d)
        tensor[d].reinit(A_in.m(d), A_in.m(d));
    compute_ksvd<Number>(schur_tensors_exact, ksvd_schur);

    /// ALTERNATIVE no fast diagonalization
    matrix_type::reinit(ksvd_schur);

    /// DEBUG

    /// eigenvalues
    const auto print_eigenvalues = [](const auto & tensors, const std::string name) {
      TensorProductMatrix<order, Number, n_rows_1d> tpmat(tensors);
      const auto &                                  mat = table_to_fullmatrix(tpmat.as_table(), 0);
      const auto &                                  eigenvalues = compute_eigenvalues(mat);
      std::cout << "Eigenvalues of " << name << std::endl;
      std::cout << vector_to_string(eigenvalues) << std::endl;
    };
    const auto print_inverse_eigenvalues = [](const auto & tensors, const std::string name) {
      TensorProductMatrix<order, Number, n_rows_1d> tpmat(tensors);
      const auto &                                  mat = table_to_fullmatrix(tpmat.as_table(), 0);
      const auto &                                  eigenvalues = compute_eigenvalues(mat);
      std::vector<std::complex<double>>             inverse_eigenvalues;
      std::transform(eigenvalues.begin(),
                     eigenvalues.end(),
                     std::back_inserter(inverse_eigenvalues),
                     [](const std::complex<double> & lambda) { return 1. / lambda; });
      std::sort(inverse_eigenvalues.begin(),
                inverse_eigenvalues.end(),
                [](const auto & a, const auto & b) { return a.real() < b.real(); });
      std::reverse(inverse_eigenvalues.begin(), inverse_eigenvalues.end());
      std::cout << "Inverse eigenvalues of " << name << std::endl;
      std::cout << vector_to_string(inverse_eigenvalues) << std::endl;
    };

    // print_eigenvalues(minus_Ainv, "-A^{-1}");
    // print_eigenvalues(D_in.get_elementary_tensors(), "D");
    // print_eigenvalues(B_in.get_elementary_tensors(), "B");
    // print_eigenvalues(minus_C_Ainv_B, "-CA^{-1}B");
    // {
    //   TensorProductMatrix<order, Number, n_rows_1d> S(schur_tensors_exact);
    //   const auto &                                  S_full = table_to_fullmatrix(S.as_table(),
    //   0); TensorProductMatrix<order, Number, n_rows_1d> Stilde(ksvd_schur); const auto &
    //   Stilde_inv = table_to_fullmatrix(Stilde.as_inverse_table(), 0); FullMatrix<double>
    //   Itilde(Stilde_inv.m(), S_full.n()); Stilde_inv.mmult(Itilde, S_full); const auto &
    //   eigenvalues = compute_eigenvalues(Itilde); std::cout << "Eigenvalues of Stilde^{-1}S" <<
    //   std::endl; std::cout << vector_to_string(eigenvalues) << std::endl;
    // }
    // print_eigenvalues(schur_tensors_exact, "S");
    // print_eigenvalues(ksvd_schur, "Stilde");
    // for (auto r = 0U; r < ksvd_schur.size(); ++r)
    //   {
    // 	std::vector<std::array<Table<2, Number>, order>> rth_tensor;
    // 	rth_tensor.emplace_back(ksvd_schur[r]);
    // 	std::ostringstream oss;
    // 	oss << "Stilde[" << r << "]";
    // 	print_eigenvalues(rth_tensor, oss.str());
    //   }
    // {
    //   std::vector<std::array<Table<2, Number>, order>> schur_minus_rank1_tensor;
    //   std::copy(schur_tensors_exact.cbegin(),
    //             schur_tensors_exact.cend(),
    //             std::back_inserter(schur_minus_rank1_tensor));
    //   auto rank1_tensor    = ksvd_schur[0];
    //   rank1_tensor.front() = Tensors::scale(-1., rank1_tensor.front());
    //   schur_minus_rank1_tensor.emplace_back(rank1_tensor);
    //   print_eigenvalues(schur_minus_rank1_tensor, "S - Stilde[0]");
    // }

    /// DEBUG S = D(I-D^{-1}CA^{-1}B)

    // std::vector<std::array<Table<2, Number>, order>> Dinv_ksvd;
    // { // KSVD of D^{-1}
    //   const auto            eigenvalues = D_in.get_eigenvalues();
    //   AlignedVector<Number> inverse_eigenvalues(eigenvalues.size());
    //   std::transform(eigenvalues.begin(),
    //                  eigenvalues.end(),
    //                  inverse_eigenvalues.begin(),
    //                  [](const auto & lambda) { return 1. / lambda; });
    //   std::vector<std::array<Table<2, Number>, order>> ksvd_eigenvalues(
    //     lambda_rank == -1 ? lambda_rank_max : lambda_rank);
    //   for(auto & tensor : ksvd_eigenvalues)
    //     for(auto d = 0U; d < order; ++d)
    //       tensor[d].reinit(D_in.m(d), D_in.m(d));
    //   compute_ksvd<Number>(inverse_eigenvalues, ksvd_eigenvalues);
    //   std::vector<std::array<Table<2, Number>, order>> eigenvectors(1), eigenvectorsT(1);
    //   eigenvectors.front() = D_in.get_eigenvectors();
    //   std::transform(eigenvectors.front().cbegin(),
    //                  eigenvectors.front().cend(),
    //                  eigenvectorsT.front().begin(),
    //                  [](const auto & tab) { return Tensors::transpose(tab); });
    //   const auto Lambda_QT   = Tensors::product<order, Number>(ksvd_eigenvalues, eigenvectorsT);
    //   const auto Q_Lambda_QT = Tensors::product<order, Number>(eigenvectors, Lambda_QT);
    //   std::copy(Q_Lambda_QT.begin(), Q_Lambda_QT.end(), std::back_inserter(Dinv_ksvd));
    // }

    // print_eigenvalues(Dinv_ksvd, "KSVD of D^{-1}");
    // print_inverse_eigenvalues(D_in.get_elementary_tensors(), "D");
    // const auto Dinv_minus_C_Ainv_B = Tensors::product<order, Number>(Dinv_ksvd, minus_C_Ainv_B);
    // std::vector<std::array<Table<2, Number>, order>> I_tensors(1);
    // for(auto & tensor : I_tensors)
    //   for(auto d = 0U; d < order; ++d)
    //   {
    //     auto & matrix = tensor[d];
    //     matrix.reinit(D_in.m(d), D_in.m(d));
    //     for(auto i = 0U; i < D_in.m(d); ++i)
    //       matrix(i, i) = 1.;
    //   }
    // std::vector<std::array<Table<2, Number>, order>> freestyle;
    // std::copy(Dinv_minus_C_Ainv_B.begin(),
    //           Dinv_minus_C_Ainv_B.end(),
    //           std::back_inserter(freestyle));
    // freestyle.emplace_back(I_tensors.front());
    // // print_eigenvalues(Dinv_minus_C_Ainv_B, "-D^{-1}CA^{-1}B");
    // // print_eigenvalues(freestyle, "I-D^{-1}CA^{-1}B");
    // const auto long_schur_tensors =
    //   Tensors::product<order, Number>(D_in.get_elementary_tensors(), freestyle);
    // std::vector<std::array<Table<2, Number>, order>> freestyle_schur(
    //   kronecker_rank == -1 ? kronecker_rank_max : kronecker_rank);
    // for(auto & tensor : freestyle_schur)
    //   for(auto d = 0U; d < order; ++d)
    //     tensor[d].reinit(D_in.m(d), D_in.m(d));
    // compute_ksvd<Number>(freestyle, freestyle_schur);
    // // print_eigenvalues(freestyle_schur, "KSVD of I-D^{-1}CA^{-1}B");
    // const auto long_schur_ksvd =
    //   Tensors::product<order, Number>(D_in.get_elementary_tensors(), freestyle_schur);
    // // print_eigenvalues(schur_tensors_exact, "S");
    // // print_eigenvalues(long_schur_tensors, "long S");
    // print_eigenvalues(long_schur_ksvd, "long Stilde");
    // {
    //   TensorProductMatrix<order, Number, n_rows_1d> S(schur_tensors_exact);
    //   const auto &                                  S_full = table_to_fullmatrix(S.as_table(),
    //   0); TensorProductMatrix<order, Number, n_rows_1d> Stilde(long_schur_ksvd); const auto &
    //   Stilde_inv = table_to_fullmatrix(Stilde.as_inverse_table(), 0);

    //   // /// dismiss negative eigenvalues
    //   // const auto singular_values = compute_singular_values(Stilde_inv);
    //   // std::cout << "Singular values of Stilde^{-1}" << std::endl;
    //   // std::cout << vector_to_string(singular_values) << std::endl;

    //   // const auto & matrix = Stilde_inv;
    //   // LAPACKFullMatrix<Number> lapack_matrix(matrix.m());
    //   // lapack_matrix = matrix;
    //   // lapack_matrix.compute_eigenvalues();
    //   // LAPACKFullMatrix<Number> reduced_matrix(lapack_matrix.m());
    //   // reduced_matrix *= 0.;
    //   // for(auto i = 0U; i < reduced_matrix.size(); ++i)
    //   // 	{
    //   // 	  reduced_matrix.rank1_update(
    //   // 	}

    //   FullMatrix<double> Itilde(Stilde_inv.m(), S_full.n());
    //   Stilde_inv.mmult(Itilde, S_full);
    //   const auto & eigenvalues = compute_eigenvalues(Itilde);
    //   std::cout << "Eigenvalues of Stilde^{-1}S" << std::endl;
    //   std::cout << vector_to_string(eigenvalues) << std::endl;
    // }


    const auto print_eigenvalues_symm = [&](const Vector<double> & eigenvalues,
                                            const std::string      name) {
      std::vector<double> evs(eigenvalues.begin(), eigenvalues.end());
      std::sort(evs.begin(), evs.end());
      std::reverse(evs.begin(), evs.end());
      std::cout << "Eigenvalues of " << name << " (symm)" << std::endl;
      std::cout << vector_to_string(evs) << std::endl;
    };

    // // /// DEBUG
    // /// D = QLQ^T with QQ^T = I     D = QL^{1/2}L^{1/2}Q^T
    // /// S = QL^{1/2}(I - L^{-1/2}Q^TCA^{-1}BQL^{-1/2})L^{1/2}Q^T
    // {
    //   TensorProductMatrix<order, Number, n_rows_1d> DD(D_in.get_elementary_tensors());
    //   const auto &                                  D = table_to_fullmatrix(DD.as_table(), 0);
    //   FullMatrix<double>                            Q;
    //   const auto                                    eigenvalues = compute_eigenvalues_symm(D, Q);
    //   {
    //     print_eigenvalues_symm(eigenvalues, "D");
    //     print_eigenvalues(D_in.get_elementary_tensors(), "D");
    //   }
    //   FullMatrix<double> LmbSqrtInv(eigenvalues.size(), eigenvalues.size());
    //   for(auto i = 0U; i < eigenvalues.size(); ++i)
    //     LmbSqrtInv(i, i) = 1. / std::sqrt(eigenvalues(i));
    //   FullMatrix<double> LmbSqrtInvQT(LmbSqrtInv.m(), Q.m());
    //   LmbSqrtInv.mTmult(LmbSqrtInvQT, Q);
    //   {
    //     FullMatrix<double> QLmbInvQT(D_in.m());
    //     LmbSqrtInvQT.Tmmult(QLmbInvQT, LmbSqrtInvQT);
    //     FullMatrix<double> dummy;
    //     const auto         eigenvalues = compute_eigenvalues_symm(QLmbInvQT, dummy);
    //     print_eigenvalues_symm(eigenvalues, "D");
    //     print_inverse_eigenvalues(D_in.get_elementary_tensors(), "D");
    //   }
    //   TensorProductMatrix<order, Number, n_rows_1d> mmCAinvB(minus_C_Ainv_B);
    //   const auto &       mCAinvB = table_to_fullmatrix(mmCAinvB.as_table(), 0);
    //   FullMatrix<double> tmp(D_in.m());
    //   LmbSqrtInvQT.mmult(tmp, mCAinvB);
    //   FullMatrix<double> tmp2(D_in.m());
    //   tmp.mTmult(tmp2, LmbSqrtInvQT);
    //   for(auto i = 0U; i < tmp2.m(); ++i)
    //     tmp2(i, i) += 1.;
    //   FullMatrix<double> LmbSqrt(eigenvalues.size(), eigenvalues.size());
    //   for(auto i = 0U; i < eigenvalues.size(); ++i)
    //     LmbSqrt(i, i) = std::sqrt(eigenvalues(i));
    //   FullMatrix<double> LmbSqrtQT(D_in.m());
    //   LmbSqrt.mTmult(LmbSqrtQT, Q);
    //   FullMatrix<double> tmp3(D_in.m());
    //   LmbSqrtQT.Tmmult(tmp3, tmp2);
    //   FullMatrix<double> S(D_in.m());
    //   tmp3.mmult(S, LmbSqrtQT);
    //   {
    //     FullMatrix<double> dummy;
    //     const auto         eigenvalues = compute_eigenvalues_symm(S, dummy);
    //     print_eigenvalues_symm(eigenvalues, "S");
    //     print_eigenvalues(schur_tensors_exact, "S");
    //   }
    // }

    {
      auto                  eigenvalues = D_in.get_eigenvalues();
      AlignedVector<Number> sqrt_eigenvalues(eigenvalues.size());
      std::transform(eigenvalues.begin(),
                     eigenvalues.end(),
                     sqrt_eigenvalues.begin(),
                     [](const auto & lambda) { return sqrt(lambda); });
      std::vector<std::array<Table<2, Number>, order>> Lsqrt(lambda_rank == -1 ? lambda_rank_max :
                                                                                 lambda_rank);
      for(auto & tensor : Lsqrt)
        for(auto d = 0U; d < order; ++d)
          tensor[d].reinit(D_in.m(d), D_in.m(d));
      compute_ksvd<Number>(sqrt_eigenvalues, Lsqrt);
      // print_eigenvalues(Lsqrt, "sqrt of Lambda(D)");

      AlignedVector<Number> inverse_sqrt_eigenvalues(eigenvalues.size());
      std::transform(eigenvalues.begin(),
                     eigenvalues.end(),
                     inverse_sqrt_eigenvalues.begin(),
                     [](const auto & lambda) { return 1. / sqrt(lambda); });
      std::vector<std::array<Table<2, Number>, order>> inverse_Lsqrt(
        lambda_rank == -1 ? lambda_rank_max : lambda_rank);
      for(auto & tensor : inverse_Lsqrt)
        for(auto d = 0U; d < order; ++d)
          tensor[d].reinit(D_in.m(d), D_in.m(d));
      compute_ksvd<Number>(inverse_sqrt_eigenvalues, inverse_Lsqrt);

      std::vector<std::array<Table<2, Number>, order>> eigenvectors(1), eigenvectorsT(1);
      eigenvectors.front() = D_in.get_eigenvectors();
      std::transform(eigenvectors.front().cbegin(),
                     eigenvectors.front().cend(),
                     eigenvectorsT.front().begin(),
                     [](const auto & tab) { return Tensors::transpose(tab); });
      std::vector<std::array<Table<2, Number>, order>> mass;
      mass.emplace_back(D_in.get_mass());
      // {
      // 	const auto tmp1 = Tensors::product<order, Number>(mass, eigenvectors);
      // 	const auto Lambda = Tensors::product<order, Number>(Lsqrt, Lsqrt);
      // 	const auto tmp2 = Tensors::product<order, Number>(tmp1, Lambda);
      // 	const auto tmp3 = Tensors::product<order, Number>(tmp2, eigenvectorsT);
      // 	const auto tmp4 = Tensors::product<order, Number>(tmp3, mass);
      // 	print_eigenvalues(tmp4, "D decomp");
      // 	print_eigenvalues(D_in.get_elementary_tensors(), "D");
      // }
      // {
      // 	const auto tmp1 = Tensors::product<order, Number>(mass, eigenvectors);
      // 	const auto tmp2 = Tensors::product<order, Number>(tmp1, eigenvectorsT);
      // 	print_eigenvalues(tmp2, "I?");
      // }
      std::vector<std::array<Table<2, Number>, order>> identity(1);
      for(auto & tensor : identity)
        for(auto d = 0U; d < order; ++d)
        {
          auto & matrix = tensor[d];
          matrix.reinit(D_in.m(d), D_in.m(d));
          for(auto i = 0U; i < D_in.m(d); ++i)
            matrix(i, i) = 1.;
        }
      const auto tmp1 = Tensors::product<order, Number>(minus_C_Ainv_B, eigenvectors);
      const auto tmp2 = Tensors::product<order, Number>(tmp1, inverse_Lsqrt);
      const auto tmp3 = Tensors::product<order, Number>(eigenvectorsT, tmp2);
      auto       Z    = Tensors::product<order, Number>(inverse_Lsqrt, tmp3);
      Z.emplace_back(identity.front());
      std::vector<std::array<Table<2, Number>, order>> Z_ksvd(
        kronecker_rank == -1 ? kronecker_rank_max : kronecker_rank);
      for(auto & tensor : Z_ksvd)
        for(auto d = 0U; d < order; ++d)
          tensor[d].reinit(D_in.m(d), D_in.m(d));
      compute_ksvd<Number>(Z, Z_ksvd);
      {
        TensorProductMatrix<order, Number, n_rows_1d> ZZtilde(Z_ksvd);
        const auto         Ztilde = table_to_fullmatrix(ZZtilde.as_table(), 0);
        FullMatrix<double> Q(D_in.m());
        auto               eigenvalues = compute_eigenvalues_symm(Ztilde, Q);
        // print_eigenvalues_symm(eigenvalues, "KSVD of Z");
        // print_eigenvalues(Z, "Z");
        const auto tmp4    = Tensors::product<order, Number>(mass, eigenvectors);
        const auto MQLsqrt = Tensors::product<order, Number>(tmp4, Lsqrt);
        TensorProductMatrix<order, Number, n_rows_1d> YY(MQLsqrt);
        const auto                                    Y = table_to_fullmatrix(YY.as_table(), 0);
        FullMatrix<double>                            YZtilde(D_in.m());
        Y.mmult(YZtilde, Ztilde);
        FullMatrix<double> Stilde(D_in.m());
        YZtilde.mTmult(Stilde, Y);
        eigenvalues = compute_eigenvalues_symm(Stilde, Q);
        {
          print_eigenvalues_symm(eigenvalues, "Stilde");
          auto tmp_eigenvalues = compute_eigenvalues(Stilde);
          std::sort(tmp_eigenvalues.begin(),
                    tmp_eigenvalues.end(),
                    [](const auto & a, const auto & b) { return a.real() < b.real(); });
          std::reverse(tmp_eigenvalues.begin(), tmp_eigenvalues.end());
          std::cout << vector_to_string(tmp_eigenvalues) << std::endl;
          print_eigenvalues(schur_tensors_exact, "S");
        }
        FullMatrix<double> inverse_Lmb_Stilde(D_in.m());
        for(auto i = 0U; i < eigenvalues.size(); ++i)
          inverse_Lmb_Stilde(i, i) = 1. / eigenvalues(i);
        {
          FullMatrix<double> tmp1(D_in.m());
          Q.mmult(tmp1, inverse_Lmb_Stilde);
          FullMatrix<double> Stilde_inv(D_in.m());
          tmp1.mTmult(Stilde_inv, Q);

          TensorProductMatrix<order, Number, n_rows_1d> SS(schur_tensors_exact);
          const auto                                    S = table_to_fullmatrix(SS.as_table(), 0);
          FullMatrix<double>                            Itilde(S.m());
          Stilde_inv.mmult(Itilde, S);
          {
            auto eigenvalues = compute_eigenvalues(Itilde);
            std::cout << "Eigenvalues of Stilde^{-1}S" << std::endl;
            std::cout << vector_to_string(eigenvalues) << std::endl;
          }
        }
        auto sorted_eigenvalues = eigenvalues;
        std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end());
        // std::reverse(sorted_eigenvalues.begin(), sorted_eigenvalues.end());
        const unsigned int n_dismiss = 8;
        const double       threshold = sorted_eigenvalues[n_dismiss - 1] + 0.001;
        std::cout << "number of dismissed eigenvalues: " << n_dismiss << std::endl;
        std::cout << "threshold: " << threshold << std::endl;
        std::vector<std::size_t> indices_dismiss;
        for(auto i = 0U; i < eigenvalues.size(); ++i)
        {
          // std::cout << eigenvalues(i) << std::endl;
          if(eigenvalues(i) < threshold)
            indices_dismiss.push_back(i);
        }
        std::cout << vector_to_string(indices_dismiss) << std::endl;
        for(const auto j : indices_dismiss)
        {
          inverse_Lmb_Stilde(j, j) = 0.001;
          // for (auto i = 0U; i < Q.n(); ++i)
          //   Q(i,j) = 0.;
        }
        {
          FullMatrix<double> tmp1(D_in.m());
          Q.mmult(tmp1, inverse_Lmb_Stilde);
          FullMatrix<double> Stilde_inv(D_in.m());
          tmp1.mTmult(Stilde_inv, Q);

          TensorProductMatrix<order, Number, n_rows_1d> SS(schur_tensors_exact);
          const auto                                    S = table_to_fullmatrix(SS.as_table(), 0);
          FullMatrix<double>                            Itilde(S.m());
          Stilde_inv.mmult(Itilde, S);
          {
            auto eigenvalues = compute_eigenvalues(Itilde);
            std::cout << "Eigenvalues of Stilde^{-1}S" << std::endl;
            std::cout << vector_to_string(eigenvalues) << std::endl;
          }
        }
      }
    }

    // matrix_type::reinit(long_schur_ksvd);



    // std::vector<std::array<Table<2, Number>, order>> Lambda_minus_QT_CAB_Q;
    // { // KSVD of D^{-1}
    //   auto                                             eigenvalues = D_in.get_eigenvalues();
    //   std::vector<std::array<Table<2, Number>, order>> D_eigenvalues(2);
    //   for(auto & tensor : D_eigenvalues)
    //     for(auto d = 0U; d < order; ++d)
    //       tensor[d].reinit(D_in.m(d), D_in.m(d));
    //   compute_ksvd<Number>(eigenvalues, D_eigenvalues);
    //   print_eigenvalues(D_eigenvalues, "EVs of D");
    //   const auto          my_eigenvalues = D_in.get_eigenvalues();
    //   std::vector<double> tmp_eigenvalues;
    //   std::transform(my_eigenvalues.begin(),
    //                  my_eigenvalues.end(),
    //                  std::back_inserter(tmp_eigenvalues),
    //                  [](const auto lambda) { return lambda[0]; });
    //   std::sort(tmp_eigenvalues.begin(), tmp_eigenvalues.end());
    //   std::reverse(tmp_eigenvalues.begin(), tmp_eigenvalues.end());
    //   std::cout << vector_to_string(tmp_eigenvalues) << std::endl;

    //   std::vector<std::array<Table<2, Number>, order>> my_eigenvectors(1), my_eigenvectorsT(1);
    //   my_eigenvectors.front() = D_in.get_eigenvectors();
    //   std::transform(my_eigenvectors.front().cbegin(),
    //                  my_eigenvectors.front().cend(),
    //                  my_eigenvectorsT.front().begin(),
    //                  [](const auto & tab) { return Tensors::transpose(tab); });
    //   const auto Dtmp1 = Tensors::product<order, Number>(D_eigenvalues, my_eigenvectors);
    //   const auto Dtmp2 = Tensors::product<order, Number>(my_eigenvectorsT, Dtmp1);
    //   print_eigenvalues(Dtmp2, "Dtmp");
    //   print_eigenvalues(D_in.get_elementary_tensors(), "D");

    //   // const auto tmp1 = Tensors::product<order, Number>(minus_C_Ainv_B, eigenvectors);
    //   // const auto tmp  = Tensors::product<order, Number>(eigenvectorsT, tmp1);
    //   // std::copy(tmp.begin(), tmp.end(), std::back_inserter(Lambda_minus_QT_CAB_Q));
    //   // std::copy(D_eigenvalues.begin(),
    //   //           D_eigenvalues.end(),
    //   //           std::back_inserter(Lambda_minus_QT_CAB_Q));
    //   // print_eigenvalues(Lambda_minus_QT_CAB_Q, "blub");
    //   // std::vector<std::array<Table<2, Number>, order>> blub_schur(
    //   //   kronecker_rank == -1 ? kronecker_rank_max : kronecker_rank);
    //   // for(auto & tensor : blub_schur)
    //   //   for(auto d = 0U; d < order; ++d)
    //   //     tensor[d].reinit(D_in.m(d), D_in.m(d));
    //   // compute_ksvd<Number>(Lambda_minus_QT_CAB_Q, blub_schur);
    //   // print_eigenvalues(blub_schur, "KSVD of blub");
    //   // const auto tmp2 = Tensors::product<order, Number>(Lambda_minus_QT_CAB_Q, eigenvectorsT);
    //   // const auto S    = Tensors::product<order, Number>(eigenvectors, tmp2);
    //   // print_eigenvalues(S, "blub S");
    //   // print_eigenvalues(schur_tensors_exact, "S");
    // }



    // /// freestyle
    // std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    // std::cout << "                                                  freestyle" << std::endl;
    // std::vector<std::array<Table<2, Number>, order>> freestyle;
    // freestyle = D_in.get_elementary_tensors();
    // std::vector<std::array<Table<2, Number>, order>> I_tensors(1);
    // for(auto & tensor : I_tensors)
    //   for(auto d = 0U; d < order; ++d)
    // 	{
    // 	  auto & matrix = tensor[d];
    // 	  matrix.reinit(A_in.m(d), A_in.m(d));
    // 	  for (auto i = 0U; i < A_in.m(d); ++i)
    // 	    matrix(i,i) = 1.;
    // 	}
    // I_tensors.front().front() = Tensors::scale(1., I_tensors.front().front());
    // const auto minus_Ainv_I        = Tensors::product<order, Number>(minus_Ainv, I_tensors);
    // const auto minus_I_Ainv_I        = Tensors::product<order, Number>(I_tensors, minus_Ainv_I);
    // std::copy(minus_I_Ainv_I.cbegin(),
    //           minus_I_Ainv_I.cend(),
    //           std::back_inserter(freestyle));
    // std::vector<std::array<Table<2, Number>, order>> freestyle_schur(
    // 								     kronecker_rank == -1 ? kronecker_rank_max : kronecker_rank);
    // for(auto & tensor : freestyle_schur)
    //   for(auto d = 0U; d < order; ++d)
    //     tensor[d].reinit(A_in.m(d), A_in.m(d));
    // compute_ksvd<Number>(freestyle, freestyle_schur);
    // {
    //   TensorProductMatrix<order, Number, n_rows_1d> S(schur_tensors_exact);
    //   const auto & S_full = table_to_fullmatrix(S.as_table(), 0);
    //   TensorProductMatrix<order, Number, n_rows_1d> Stilde(freestyle_schur);
    //   const auto &       Stilde_inv = table_to_fullmatrix(Stilde.as_inverse_table(), 0);
    //   FullMatrix<double> Itilde(Stilde_inv.m(), S_full.n());
    //   Stilde_inv.mmult(Itilde, S_full);
    //   const auto & eigenvalues = compute_eigenvalues(Itilde);
    //   std::cout << "Eigenvalues of Stilde^{-1}S" << std::endl;
    //   std::cout << vector_to_string(eigenvalues) << std::endl;
    // }
    // print_eigenvalues(schur_tensors_exact, "S");
    // print_eigenvalues(freestyle_schur, "Stilde");

    // matrix_type::reinit(freestyle_schur);
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    matrix_type::vmult(dst_view, src_view);
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    matrix_type::apply_inverse(dst_view, src_view);
  }
};


/**
 * Schur complement S = D - C * A^{-1} * B
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
class SchurComplement
{
public:
  using matrix_type = MatrixType;
  using value_type  = typename matrix_type::value_type;

  void
  reinit(const matrix_type & A_in,
         const matrix_type & B_in,
         const matrix_type & C_in,
         const matrix_type & D_in,
         const int           dummy1 = 0,
         const int           dummy2 = 0)
  {
    (void)dummy1;
    (void)dummy2;
    A = &A_in;
    B = &B_in;
    C = &C_in;
    D = &D_in;
  }

  unsigned int
  m() const
  {
    Assert(A && B && C && D, ExcMessage("Not initialized."));
    AssertDimension(D->m(), C->m());
    return D->m();
  }

  unsigned int
  n() const
  {
    Assert(A && B && C && D, ExcMessage("Not initialized."));
    AssertDimension(D->n(), B->n());
    return D->n();
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    Assert(A && B && C && D, ExcMessage("Not initialized."));
    AssertDimension(D->n(), src_view.size());
    AssertDimension(D->m(), dst_view.size());
    AssertDimension(B->n(), src_view.size());
    AssertDimension(A->n(), A->m());
    AssertDimension(A->m(), B->m());
    AssertDimension(C->n(), A->m());
    AssertDimension(C->m(), dst_view.size());

    std::lock_guard<std::mutex> lock(this->mutex);
    tmp_array.clear();
    tmp_array.resize(B->m()); // TODO resize to max
    const auto dst_view_of_B = ArrayView(tmp_array.begin(), B->m());
    B->vmult(dst_view_of_B, src_view);
    Assert(A->m() <= dst_view.size(), ExcMessage("TODO dst_view not usable as temporary array."));
    const auto dst_view_of_Ainv = ArrayView(dst_view.begin(), A->m());
    A->apply_inverse(dst_view_of_Ainv, dst_view_of_B);
    tmp_array.clear();
    tmp_array.resize(C->n()); // TODO
    const auto dst_view_of_C = ArrayView(tmp_array.begin(), C->m());
    C->vmult(dst_view_of_C, dst_view_of_Ainv);

    D->vmult(dst_view, src_view);
    std::transform(dst_view.cbegin(),
                   dst_view.cend(),
                   dst_view_of_C.cbegin(),
                   dst_view.begin(),
                   std::minus<Number>{});
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    Assert(A && B && C && D, ExcMessage("Not initialized."));
    if(!Sinv)
      Sinv = std::make_shared<VectorizedInverse<Number>>(as_table());
    Sinv->vmult(dst_view, src_view);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
  }

private:
  const matrix_type *                                      A = nullptr;
  const matrix_type *                                      B = nullptr;
  const matrix_type *                                      C = nullptr;
  const matrix_type *                                      D = nullptr;
  mutable std::shared_ptr<const VectorizedInverse<Number>> Sinv;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};

/**
 * Computes the block Gaussian elimination of the 2x2 block matrix
 *
 *    | A B |
 *    | C D |
 *
 * using the Schur complement S for D, namely S = D - C * A^{-1} * B. The inverse
 * of the 2x2 block matrix is given by
 *
 *    | I   -A^{-1} * B | | A^{-1}   0      | | I             0 |
 *    | 0   I           | | 0        S^{-1} | | -C * A^{-1}   I |
 *
 * where the inverse of S is the dominating complexity.
 */
template<typename MatrixType,
         typename SchurType = SchurComplement<MatrixType>,
         typename Number    = typename MatrixType::value_type>
class BlockGaussianInverse
{
public:
  using matrix_type = MatrixType;
  using value_type  = typename matrix_type::value_type;

  BlockGaussianInverse(const matrix_type & A_in,
                       const matrix_type & B_in,
                       const matrix_type & C_in,
                       const matrix_type & D_in,
                       const int           lambda_rank    = 0,
                       const int           kronecker_rank = 0)
    : A(A_in), B(B_in), C(C_in), D(D_in)
  {
    // AssertThrow(!(kronecker_rank != 0 && lambda_rank == 0), ExcMessage("lambda_rank is
    // invalid."));
    if(kronecker_rank == 0 && lambda_rank == 0)
      S.reinit(A_in, B_in, C_in, D_in);
    else if(kronecker_rank == 0 && lambda_rank != 0)
      S.reinit(A_in, B_in, C_in, D_in, lambda_rank);
    else
      S.reinit(A_in, B_in, C_in, D_in, lambda_rank, kronecker_rank);
  }

  unsigned int
  m() const
  {
    return A.n() + D.n();
  }

  unsigned int
  n() const
  {
    return A.m() + D.m();
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    VectorizedInverse<Number> inverse;
    AssertDimension(D.m(), D.n()); // -> S^{-1} possible
    AssertDimension(A.m(), A.n()); // -> A^{-1} possible
    AssertDimension(A.m(), B.m());
    AssertDimension(C.m(), D.m());
    AssertDimension(A.n(), C.n());
    AssertDimension(B.n(), D.n());
    const unsigned int m0 = A.m();
    const unsigned int m1 = D.m();
    AssertDimension(src_view.size(), m0 + m1);
    AssertDimension(dst_view.size(), m0 + m1);

    /// MEMORY INEFFICENT CODE

    /// src1[0,m0) <- [I 0] src_view = src_view[0,m0);
    const auto src1_view_m0 = ArrayView(src_view.begin(), m0);
    /// src1[m0,m0+m1) <- [-CA^{-1}  I] src_view = - CA^{-1} src_view[0,m0) + src_view[m0,m0+m1)
    AlignedVector<Number> tmp1_array(m0);
    const auto            dst_view_Ainv = ArrayView(tmp1_array.begin(), m0);
    const auto            src_view_Ainv = ArrayView(src_view.begin(), m0);
    A.apply_inverse(dst_view_Ainv, src_view_Ainv);
    AlignedVector<Number> tmp2_array(m1);
    const auto            dst_view_CAinv = ArrayView(tmp2_array.begin(), m1);
    C.vmult(dst_view_CAinv, dst_view_Ainv);
    AlignedVector<Number> tmp3_array(m1);
    const auto            src1_view_m1 = ArrayView(tmp3_array.begin(), m1);
    const auto            src_view_m1  = ArrayView(src_view.begin() + m0, m1);
    std::transform(src_view_m1.cbegin(),
                   src_view_m1.end(),
                   dst_view_CAinv.cbegin(),
                   src1_view_m1.begin(),
                   std::minus<Number>());

    /// src2[0,m0) <- [A^{-1}  0] src1 = A^{-1} src1_view_m0
    AlignedVector<Number> tmp4_array(m0);
    const auto            src2_view_m0 = ArrayView(tmp4_array.begin(), m0);
    A.apply_inverse(src2_view_m0, src1_view_m0);
    /// src2[m0,m0+m1) <- [0  S^{-1}] src1 = S^{-1} src_1_view_m1
    AlignedVector<Number> tmp5_array(m1);
    const auto            src2_view_m1 = ArrayView(tmp5_array.begin(), m1);
    S.apply_inverse(src2_view_m1, src1_view_m1);

    /// dst[0,m0) <- [I  -A^{-1}B] src2 = src2_view_m0 - A^{-1}B src2_view_m1
    AlignedVector<Number> tmp6_array(m0);
    const auto            dst_view_B = ArrayView(tmp6_array.begin(), m0);
    B.vmult(dst_view_B, src2_view_m1);
    AlignedVector<Number> tmp7_array(m0);
    const auto            dst_view_AinvB = ArrayView(tmp7_array.begin(), m0);
    A.apply_inverse(dst_view_AinvB, dst_view_B);
    std::transform(src2_view_m0.cbegin(),
                   src2_view_m0.cend(),
                   dst_view_AinvB.cbegin(),
                   dst_view.begin(),
                   std::minus<Number>());
    /// dst[m0,m0+m1) <- [0  I] src2 = src2_view_m1
    std::copy(src2_view_m1.cbegin(), src2_view_m1.cend(), dst_view.begin() + m0);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

private:
  const matrix_type & A;
  const matrix_type & B;
  const matrix_type & C;
  const matrix_type & D;
  SchurType           S;
};


/*
 * mode = -1 : Compute exact Gaussian block elimination
 * mode =  0 : Compute fast diagonalized Schur complement
 */
template<int order, typename Number, int mode = -1, int n_rows_1d = -1>
class BlockMatrix
{
public:
  using matrix_type = TensorProductMatrix<order, Number, n_rows_1d>;
  using value_type  = typename matrix_type::value_type;

  BlockMatrix() = default;

  BlockMatrix &
  operator=(const BlockMatrix & other)
  {
    resize(other.n_block_rows, other.n_block_cols);
    blocks = other.blocks;
    return *this;
  }

  /**
   * Deletes current block structure.
   */
  void
  clear()
  {
    std::fill(n_.begin(), n_.end(), 0U);
    blocks.clear();
    inverse_2x2.reset();
    fast_inverse_2x2.reset();
    basic_inverse.reset();
  }

  /**
   * Deletes old and resizes to square block structure.
   */
  void
  resize(const std::size_t n_rows)
  {
    resize(n_rows, n_rows);
  }

  /**
   * Deletes old and resizes to new block structure.
   */
  void
  resize(const std::size_t n_rows, const std::size_t n_cols)
  {
    clear();
    blocks.resize(n_rows * n_cols);
    n_[0] = n_rows;
    n_[1] = n_cols;
  }

  matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index)
  {
    AssertIndexRange(row_index, n_block_rows());
    AssertIndexRange(col_index, n_block_cols());
    /// possible change of block requires recomputation of inverse
    inverse_2x2.reset();
    return blocks[block_index(row_index, col_index)];
  }

  const matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index) const
  {
    AssertIndexRange(row_index, n_block_rows());
    AssertIndexRange(col_index, n_block_cols());
    return blocks[block_index(row_index, col_index)];
  }

  std::size_t
  m(const std::size_t row_index) const
  {
    AssertIndexRange(row_index, n_block_rows());
    return get_block(row_index, 0).m();
  }

  std::size_t
  n(const std::size_t col_index) const
  {
    AssertIndexRange(col_index, n_block_cols());
    return get_block(0, col_index).n();
  }

  std::size_t
  m() const
  {
    Assert(check_row_sizes(), ExcMessage("Inconsistent number of rows of block matrices."));
    std::size_t n_rows = 0;
    for(std::size_t row_index = 0; row_index < n_block_rows(); ++row_index)
      n_rows += m(row_index);
    return n_rows;
  }

  std::size_t
  n() const
  {
    Assert(check_col_sizes(), ExcMessage("Inconsistent number of rows of block matrices."));
    std::size_t n_cols = 0;
    for(std::size_t col_index = 0; col_index < n_block_cols(); ++col_index)
      n_cols += n(col_index);
    return n_cols;
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
  }

  void
  vmult(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             block,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { block.vmult_add(dst, src); },
                     dst,
                     src);
  }

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    const bool is_2x2_block_matrix = n_block_rows() == 2 && n_block_cols() == 2;
    AssertThrow(is_2x2_block_matrix, ExcMessage("TODO"));

    if(mode == 0) // fast mode
    {
      if(!fast_inverse_2x2)
      {
        fast_inverse_2x2 =
          std::make_shared<BlockGaussianInverse<matrix_type, FastDiagSchurType>>(get_block(0, 0),
                                                                                 get_block(0, 1),
                                                                                 get_block(1, 0),
                                                                                 get_block(1, 1),
                                                                                 lambda_rank,
                                                                                 kronecker_rank);
      }
      fast_inverse_2x2->vmult(dst, src);
    }

    else if(mode == -1) // exact mode
    {
      if(!inverse_2x2)
        inverse_2x2 = std::make_shared<BlockGaussianInverse<matrix_type>>(get_block(0, 0),
                                                                          get_block(0, 1),
                                                                          get_block(1, 0),
                                                                          get_block(1, 1));
      inverse_2x2->vmult(dst, src);

      /// ALTERNATIVE: standard inverse based on LAPACK
      // if(!basic_inverse)
      // 	basic_inverse = std::make_shared<const VectorizedInverse<Number>>(as_table());
      // basic_inverse->vmult(dst, src);
    }
  }

  std::array<std::size_t, 2>
  size() const
  {
    return n_;
  }

  std::size_t
  n_block_rows() const
  {
    return n_[0];
  }

  std::size_t
  n_block_cols() const
  {
    return n_[1];
  }

  void
  set_lambda_rank(int rank)
  {
    lambda_rank = rank;
  }

  void
  set_kronecker_rank(int rank)
  {
    kronecker_rank = rank;
  }

private:
  using FastDiagSchurType = SchurComplementFast<order, Number, n_rows_1d>;

  std::size_t
  block_index(const std::size_t row_index, const std::size_t col_index) const
  {
    const std::array<std::size_t, 2> row_and_col = {row_index, col_index};
    return Tensors::multi_to_uniindex<2>(row_and_col, n_);
  }

  /**
   * Applies the function @p action to each matrix of the block
   * diagonal and the associated vector slices of @p dst and @p src.
   * The signature of the action function should be equivalent to
   * action(matrix_type& m, ArrayView<...> dst, ArrayView<const ...> src)
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const
  {
    AssertDimension(src.size(), n());
    AssertDimension(dst.size(), m());
    Assert(src.size() == dst.size(),
           ExcMessage("TODO BlockMatrix is not quadratic.")); // quadratic !
    Assert(n_rows_1d == -1 || src.size() % n_rows_1d == 0,
           ExcMessage("Input vector sizes are not a multiple of the static size."));
    Assert(!blocks.empty(), ExcMessage("Blocks are not initialized."));
    Assert(is_valid(), ExcMessage("Matrix sizes of blocks mismatch."));
    std::fill(dst.begin(), dst.end(), static_cast<Number>(0.));

    // *** apply sliced vectors to the associated block matrix
    std::size_t row_start = 0;
    for(std::size_t row = 0; row < n_block_rows(); ++row)
    {
      std::size_t col_start = 0;
      for(std::size_t col = 0; col < n_block_cols(); ++col)
      {
        const ArrayView<Number>       dst_block(dst.begin() + row_start, m(row));
        const ArrayView<const Number> src_block(src.begin() + col_start, n(col));
        action(get_block(row, col), dst_block, src_block);
        col_start += n(col);
      }
      Assert(col_start == src.size(), ExcMessage("Inconsistent slicing."));
      row_start += m(row);
    }
    Assert(row_start == dst.size(), ExcMessage("Inconsistent slicing."));
  }

  bool
  check_row_sizes() const
  {
    for(std::size_t row = 0; row < n_block_rows(); ++row)
      for(std::size_t col = 0; col < n_block_cols(); ++col)
        if(get_block(row, col).m() != get_block(row, 0).m())
          return false;
    return true;
  }

  bool
  check_col_sizes() const
  {
    for(std::size_t col = 0; col < n_block_cols(); ++col)
      for(std::size_t row = 0; row < n_block_rows(); ++row)
        if(get_block(row, col).n() != get_block(0, col).n())
          return false;
    return true;
  }

  bool
  is_valid() const
  {
    return check_row_sizes() && check_col_sizes();
  }

  /**
   * The number of blocks per row and column
   */
  std::array<std::size_t, 2> n_ = {0, 0};

  /**
   * The vector containing the matrix blocks.
   */
  AlignedVector<matrix_type> blocks;

  /**
   * The inverse of a 2 x 2 block matrix based on block Gaussian elimination.
   */
  mutable std::shared_ptr<const BlockGaussianInverse<matrix_type>> inverse_2x2;
  /// ALTERNATIVE: standard inverse based on LAPACK
  mutable std::shared_ptr<const VectorizedInverse<Number>> basic_inverse;

  /**
   * The inverse of a 2 x 2 block matrix based on approximate block Gaussian
   * elimination.
   */
  mutable std::shared_ptr<const BlockGaussianInverse<matrix_type, FastDiagSchurType>>
      fast_inverse_2x2;
  int lambda_rank    = order;
  int kronecker_rank = order;
};



template<int dim, typename Number, int n_rows_1d = -1>
class BlockMatrixDiagonal
{
public:
  using matrix_type = TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>;
  using value_type  = typename matrix_type::value_type;

  /**
   * Deletes old and resizes to new block structure.
   */
  void
  resize(const std::size_t n_blocks)
  {
    blocks.clear();
    blocks.resize(n_blocks);
  }

  matrix_type &
  get_block(const std::size_t block_index)
  {
    AssertIndexRange(block_index, blocks.size());
    return blocks[block_index];
  }

  const matrix_type &
  get_block(const std::size_t block_index) const
  {
    AssertIndexRange(block_index, blocks.size());
    return blocks[block_index];
  }

  unsigned int
  m() const
  {
    const unsigned int n_rows =
      std::accumulate(blocks.begin(), blocks.end(), 0, [](const auto sum, const auto & matrix) {
        return sum + matrix.m();
      });
    return n_rows;
  }

  unsigned int
  n() const
  {
    const unsigned int n_columns =
      std::accumulate(blocks.begin(), blocks.end(), 0, [](const auto sum, const auto & matrix) {
        return sum + matrix.m();
      });
    return n_columns;
  }

  void
  vmult(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             matrix,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { matrix.vmult(dst, src); },
                     dst,
                     src);
  }

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const
  {
    blockwise_action([](const matrix_type &             matrix,
                        const ArrayView<Number> &       dst,
                        const ArrayView<const Number> & src) { matrix.apply_inverse(dst, src); },
                     dst,
                     src);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
  }

private:
  /**
   * Applies the function @p action to each matrix of the block
   * diagonal and the associated vector slices of @p dst and @p src.
   * The signature of the action function should be equivalent to
   * action(matrix_type& m, ArrayView<...> dst, ArrayView<const ...> src)
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const
  {
    AssertDimension(src.size(), n());
    AssertDimension(dst.size(), m());
    AssertThrow(src.size() == dst.size(), ExcNotImplemented()); // quadratic !
    AssertThrow(n_rows_1d == -1 || src.size() % n_rows_1d == 0,
                ExcMessage("Input vector sizes are not a multiple of the static size."));
    AssertThrow(!blocks.empty(), ExcMessage("Blocks are not initialized."));

    // *** apply sliced vectors to the associated block matrix
    std::size_t row_end = 0;
    std::size_t col_end = 0;
    for(const auto & matrix : blocks)
    {
      const auto row_start = row_end;
      const auto col_start = col_end;
      row_end += matrix.m();
      col_end += matrix.n();
      const ArrayView<Number>       dst_block(dst.begin() + row_start, matrix.m());
      const ArrayView<const Number> src_block(src.begin() + col_start, matrix.n());
      action(matrix, dst_block, src_block);
    }
    AssertThrow(row_end == dst.size() && col_end == src.size(),
                ExcMessage("Inconsistent slicing."));
  }

  /**
   * The vector containing the matrix blocks.
   */
  AlignedVector<matrix_type> blocks;
};

} // namespace Tensors

#endif // BLOCK_MATRIX_H_
