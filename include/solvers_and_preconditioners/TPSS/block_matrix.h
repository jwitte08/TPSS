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
  using matrix_type                        = TensorProductMatrix<order, Number, n_rows_1d>;
  using value_type                         = typename matrix_type::value_type;
  using State                              = typename matrix_type::State;
  static constexpr unsigned int macro_size = get_macro_size<Number>();

  void
  remove_zeros(std::vector<std::array<Table<2, Number>, order>> & tensors)
  {
    std::vector<std::array<Table<2, Number>, order>> tmp;
    std::remove_copy_if(tensors.cbegin(),
                        tensors.cend(),
                        std::back_inserter(tmp),
                        Tensors::is_nearly_zero<order, Number>);
    std::swap(tensors, tmp);
  }

  // TODO efficient comp. of min eigenvalue
  Number
  compute_min_lambda(const matrix_type tpmatrix)
  {
    Number min_lambda;
    for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
    {
      const auto         matrix = table_to_fullmatrix(tpmatrix.as_table(), lane);
      FullMatrix<double> dummy;
      const auto         eigenvalues = compute_eigenvalues_symm(matrix, dummy);
      scalar_value(min_lambda, lane) = eigenvalues[0];
    }
    return min_lambda;
  }


  int
  get_lambda_rank_max(const matrix_type & matrix)
  {
    return matrix.m(0);
  }

  std::vector<std::array<Table<2, Number>, order>>
  get_minusCAinvB(const matrix_type & A_in,
                  const matrix_type & B_in,
                  const matrix_type & C_in,
                  const int           lambda_rank)
  {
    std::vector<std::array<Table<2, Number>, order>> minus_C_Ainv_B;
    {
      /// compute KSVD of A's inverse eigenvalues
      const auto            eigenvalues = A_in.get_eigenvalues();
      AlignedVector<Number> inverse_eigenvalues(eigenvalues.size());
      std::transform(eigenvalues.begin(),
                     eigenvalues.end(),
                     inverse_eigenvalues.begin(),
                     [](const auto & lambda) { return 1. / lambda; });
      std::vector<std::array<Table<2, Number>, order>> ksvd_eigenvalues(lambda_rank);
      for(auto & tensor : ksvd_eigenvalues)
        for(auto d = 0U; d < order; ++d)
          tensor[d].reinit(A_in.m(d), A_in.m(d));
      compute_ksvd<Number>(inverse_eigenvalues, ksvd_eigenvalues);
      remove_zeros(ksvd_eigenvalues);

      /// compute elementary tensors representing approximation of -A^{-1}
      std::vector<std::array<Table<2, Number>, order>> eigenvectors;
      eigenvectors.emplace_back(A_in.get_eigenvector_tensor());
      scratch = Tensors::productT<order, Number>(ksvd_eigenvalues, eigenvectors);
      const auto & minus_eigenvectors = Tensors::scale<order, Number>(-1., eigenvectors);
      scratch        = Tensors::product<order, Number>(minus_eigenvectors, scratch);
      scratch        = Tensors::product<order, Number>(scratch, B_in.get_elementary_tensors());
      minus_C_Ainv_B = Tensors::product<order, Number>(C_in.get_elementary_tensors(), scratch);
    }
    return minus_C_Ainv_B;
  }

  std::array<Table<2, Number>, order>
  get_identity_tensor(const matrix_type & matrix_in)
  {
    std::array<Table<2, Number>, order> tensor;
    for(auto d = 0U; d < order; ++d)
    {
      auto & matrix = tensor[d];
      matrix.reinit(matrix_in.m(d), matrix_in.m(d));
      for(auto i = 0U; i < matrix_in.m(d); ++i)
        matrix(i, i) = 1.;
    }
    return tensor;
  };

  std::vector<std::array<Table<2, Number>, order>>
  init_Z(const std::vector<std::array<Table<2, Number>, order>> & minus_C_Ainv_B,
         const matrix_type &                                      D_in,
         const int                                                lambda_rank)
  {
    /// compute Kronecker approximation of generalized Lambda(D)^{1/2}
    const auto & eigenvalues      = D_in.get_eigenvalues();
    const auto   sqrt_of_Lambda_  = std::make_shared<AlignedVector<Number>>(eigenvalues.size());
    auto &       sqrt_eigenvalues = *sqrt_of_Lambda_;
    std::transform(eigenvalues.begin(),
                   eigenvalues.end(),
                   sqrt_eigenvalues.begin(),
                   [](const auto & lambda) { return sqrt(lambda); });
    sqrt_of_Lambda = sqrt_of_Lambda_;

    /// compute Kronecker approximation of generalized Lambda(D)^{-1/2}
    const auto invsqrt_of_Lambda_ = std::make_shared<AlignedVector<Number>>(eigenvalues.size());
    auto &     inverse_sqrt_eigenvalues = *invsqrt_of_Lambda_;
    std::transform(eigenvalues.begin(),
                   eigenvalues.end(),
                   inverse_sqrt_eigenvalues.begin(),
                   [](const auto & lambda) { return 1. / sqrt(lambda); });
    invsqrt_of_Lambda = invsqrt_of_Lambda_;
    /// TODO replace Table with efficient format for diagonals (e.g. vectors)
    std::vector<std::array<Table<2, Number>, order>> inverse_Lsqrt(lambda_rank);
    for(auto & tensor : inverse_Lsqrt)
      for(auto d = 0U; d < order; ++d)
        tensor[d].reinit(D_in.m(d), D_in.m(d));
    compute_ksvd<Number>(inverse_sqrt_eigenvalues, inverse_Lsqrt);
    remove_zeros(inverse_Lsqrt);

    /// Kronecker decomposition of generalized eigenvectors(D) and mass(D)
    std::vector<std::array<Table<2, Number>, order>> eigenvectors;
    eigenvectors.emplace_back(D_in.get_eigenvector_tensor());
    Q = std::make_shared<TensorProductMatrix<order, Number, n_rows_1d>>(eigenvectors);
    std::vector<std::array<Table<2, Number>, order>> mass;
    mass.emplace_back(D_in.get_mass_tensor());
    M = std::make_shared<TensorProductMatrix<order, Number, n_rows_1d>>(mass);

    /// Z = (I - L^{-1/2}Q^TCA^{-1}BQL^{-1/2})
    scratch = Tensors::product<order, Number>(minus_C_Ainv_B, eigenvectors);
    scratch = Tensors::product<order, Number>(scratch, inverse_Lsqrt);
    scratch = Tensors::Tproduct<order, Number>(eigenvectors, scratch);
    scratch = Tensors::product<order, Number>(inverse_Lsqrt, scratch);
    scratch.emplace_back(get_identity_tensor(D_in));
    std::reverse(scratch.begin(), scratch.end());
    return scratch;
  }

  void
  reinit(const matrix_type & A_in,
         const matrix_type & B_in,
         const matrix_type & C_in,
         const matrix_type & D_in,
         const double        factor         = 1.,
         const int           lambda_rank    = order,
         const int           kronecker_rank = order)
  {
    // const auto print_eigenvalues = [](const auto & tensors, const std::string name) {
    //   TensorProductMatrix<order, Number, n_rows_1d> tpmat(tensors);
    //   const auto &                                  mat = table_to_fullmatrix(tpmat.as_table(),
    //   0); const auto &                                  eigenvalues = compute_eigenvalues(mat);
    //   std::cout << "Eigenvalues of " << name << std::endl;
    //   std::cout << vector_to_string(eigenvalues) << std::endl;
    //   return eigenvalues;
    // };
    // const auto print_inverse_eigenvalues = [](const auto & tensors, const std::string name) {
    //   TensorProductMatrix<order, Number, n_rows_1d> tpmat(tensors);
    //   const auto &                                  mat = table_to_fullmatrix(tpmat.as_table(),
    //   0); const auto &                                  eigenvalues = compute_eigenvalues(mat);
    //   std::vector<std::complex<double>>             inverse_eigenvalues;
    //   std::transform(eigenvalues.begin(),
    //                  eigenvalues.end(),
    //                  std::back_inserter(inverse_eigenvalues),
    //                  [](const std::complex<double> & lambda) { return 1. / lambda; });
    //   std::sort(inverse_eigenvalues.begin(),
    //             inverse_eigenvalues.end(),
    //             [](const auto & a, const auto & b) { return a.real() < b.real(); });
    //   std::reverse(inverse_eigenvalues.begin(), inverse_eigenvalues.end());
    //   std::cout << "Inverse eigenvalues of " << name << std::endl;
    //   std::cout << vector_to_string(inverse_eigenvalues) << std::endl;
    //   return inverse_eigenvalues;
    // };
    // const auto print_eigenvalues_symm = [&](const Vector<double> & eigenvalues,
    //                                         const std::string      name) {
    //   std::vector<double> evs(eigenvalues.begin(), eigenvalues.end());
    //   std::sort(evs.begin(), evs.end());
    //   std::reverse(evs.begin(), evs.end());
    //   std::cout << "Eigenvalues of " << name << " (symm)" << std::endl;
    //   std::cout << vector_to_string(evs) << std::endl;
    // };
    // const auto remove_zeros = [](std::vector<std::array<Table<2, Number>, order>> & tensors) {
    //   std::vector<std::array<Table<2, Number>, order>> tmp;
    //   std::remove_copy_if(tensors.cbegin(),
    //                       tensors.cend(),
    //                       std::back_inserter(tmp),
    //                       Tensors::is_nearly_zero<order, Number>);
    //   std::swap(tensors, tmp);
    // };

    AssertThrow(order == 2, ExcMessage("Require two dimensions."));
    AssertThrow(lambda_rank == -1 || lambda_rank > 0, ExcMessage("Invalid lambda_rank."));
    AssertThrow(kronecker_rank == -1 || kronecker_rank > 0, ExcMessage("Invalid kronecker_rank."));
    AssertThrow(A_in.get_state() == State::separable,
                ExcMessage("Not a separable Kronecker decomposition."));

    /// compute tensors approximating -BAinvC
    const auto lambda_rank_A  = lambda_rank == -1 ? get_lambda_rank_max(A_in) : lambda_rank;
    auto       minus_C_Ainv_B = get_minusCAinvB(A_in, B_in, C_in, lambda_rank_A);


    /*
     * Mode: exact
     *
     * Compute Schur complement S (exact up to Lambda) and basic inverse of S
     */

    if(kronecker_rank == -1)
    {
      mode                   = Mode::exact;
      const auto & D_tensors = D_in.get_elementary_tensors();
      std::copy(D_tensors.cbegin(), D_tensors.cend(), std::back_inserter(minus_C_Ainv_B));
      matrix_type::reinit(minus_C_Ainv_B);
      return;
    }


    /*
     * Mode: ksvd_inv
     *
     * Compute KSVD of the inverse Schur complement S (exact up to Lambda).
     */

    if(factor < 0.)
    {
      /// compute exact inverse
      mode                   = Mode::exact;
      const auto & D_tensors = D_in.get_elementary_tensors();
      std::copy(D_tensors.cbegin(), D_tensors.cend(), std::back_inserter(minus_C_Ainv_B));
      matrix_type::reinit(minus_C_Ainv_B);
      const auto & inverse_exact = Tensors::inverse_matrix_to_table(*this);

      /// KSVD of exact inverse
      const int  kronecker_rank_max  = D_in.get_elementary_tensors().size() + minus_C_Ainv_B.size();
      const auto kronecker_rank_Sinv = kronecker_rank == -1 ? kronecker_rank_max : kronecker_rank;
      std::vector<std::array<Table<2, Number>, order>> inverse_ksvd(kronecker_rank_Sinv);
      for(auto & tensor : inverse_ksvd)
        for(auto d = 0U; d < order; ++d)
          tensor[d].reinit(D_in.m(d), D_in.m(d));
      compute_ksvd<Number>(inverse_exact, inverse_ksvd);

      /// set KSVD of exact inverse
      matrix_type::reinit(inverse_ksvd);
      mode = Mode::ksvd_inv;
      return;
    }


    /* Mode: ksvd
     *
     * Compute the KSVD of Z:
     *
     * generalized eigenproblem regarding D:
     *    Q^T D Q = L   and  Q^T M Q = I
     *
     * ->   (Q M)^{-1} = Q^T   or   M Q^T Q = I
     * ->   D = M Q Lsqrt Lsqrt Q^T M
     *
     * compute Schur complement as follows
     *   S = M Q Lsqrt Z Lsqrt Q^T M
     *   with Z = (I - Linvsqrt Q^T C Ainv B Q Linvsqrt)
     *
     * Compute rank2 KSVD of Z in two steps: First we compute the rank1 KSVD Z1
     * of Z. Afterwards we might scale the identity tensor of the remainder
     * Zscaled = (Z - Z1) enforcing positive definiteness. Finally, we compute
     * the rank1 KSVD of Zscaled obtaining a (positive definite) rank2
     * approximation of Z.
     */

    AssertThrow(D_in.get_state() == State::separable,
                ExcMessage("Not a separable Kronecker decomposition."));

    /// Compute rank1 KSVD of Z
    {
      std::vector<std::array<Table<2, Number>, order>> scaled_ksvd_of_Z;
      const auto lambda_rank_D = lambda_rank == -1 ? get_lambda_rank_max(D_in) : lambda_rank;
      auto       Z_tensors     = init_Z(minus_C_Ainv_B, D_in, lambda_rank_D);
      std::vector<std::array<Table<2, Number>, order>> rank1_ksvd(1);
      for(auto & tensor : rank1_ksvd)
        for(auto d = 0U; d < order; ++d)
          tensor[d].reinit(D_in.m(d), D_in.m(d));
      compute_ksvd<Number>(Z_tensors, rank1_ksvd);
      scaled_ksvd_of_Z.emplace_back(rank1_ksvd.front());
      // {
      //   print_eigenvalues(rank1_ksvd, "rank1 KSVD of Z");
      // }

      /// compute remainder Z - Ztilde1
      auto tensor_ksvd1    = rank1_ksvd.front();
      tensor_ksvd1.front() = LinAlg::scaling(-1., tensor_ksvd1.front());
      Z_tensors.emplace_back(tensor_ksvd1);
      // {
      //   print_eigenvalues(Z_tensors, "Z - rank1_ksvd_of_Z");
      // }

      /// compute rank-1 KSVD of remainder (unscaled)
      if(factor == 1.)
      {
        compute_ksvd<Number>(Z_tensors, rank1_ksvd);
        scaled_ksvd_of_Z.emplace_back(rank1_ksvd.front());
      }

      /// compute rank-1 KSVD of remainder until rank-2 approximation of Z is
      /// positive definite
      else
      {
        Number min_lambda_first    = compute_min_lambda(rank1_ksvd);
        Number min_lambda_estimate = -1.;
        /// scaling factor enforcing positive definiteness
        Number pd_factor = 1.;
        /// increment of pd_factor
        Number delta = 0.;
        /// previous_deviation
        Number previous_deviation = 0.;
        while(less_than_all_lanes(min_lambda_estimate, static_cast<Number>(0.)))
        {
          // !!!
          pd_factor += delta;
          Z_tensors.front().front() = LinAlg::scaling(get_identity_tensor(D_in).front(), pd_factor);
          compute_ksvd<Number>(Z_tensors, rank1_ksvd);
          // {
          //   print_eigenvalues(Z_tensors, "scaled Z - rank1_ksvd_of_Z");
          //   print_eigenvalues(rank1_ksvd, "rank1 KSVD of (Z - rank1_ksvd_of_Z)");
          // }

          Number min_lambda_second = compute_min_lambda(rank1_ksvd);
          min_lambda_estimate      = min_lambda_first + min_lambda_second;
          const auto flag          = less_than(min_lambda_estimate, static_cast<Number>(0.));
          for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
            if(flag[lane])
            {
              const auto deviation                   = abs(scalar_value(min_lambda_estimate, lane));
              const auto prev_deviation              = scalar_value(previous_deviation, lane);
              scalar_value(delta, lane)              = factor * deviation + 0.1 * prev_deviation;
              scalar_value(previous_deviation, lane) = deviation;
            }
            else
              scalar_value(previous_deviation, lane) = 0.;
        }
        scaled_ksvd_of_Z.emplace_back(rank1_ksvd.front());
      }

      /// the sum of both rank1 KSVDs determines the rank2 KSVD
      mode = Mode::ksvd;
      matrix_type::reinit(scaled_ksvd_of_Z); // TODO fast diagonalization

      // {
      //   auto schur_tensors_exact = D_in.get_elementary_tensors();
      //   std::copy(minus_C_Ainv_B.cbegin(),
      //             minus_C_Ainv_B.cend(),
      //             std::back_inserter(schur_tensors_exact));
      //   print_eigenvalues(scaled_ksvd_of_Z, "scaled rank2 KSVD of Z");
      //   print_eigenvalues(schur_tensors_exact, "S (exact up to Lambda)");
      //   const auto matrix      = table_to_fullmatrix(Tensors::matrix_to_table(*this), 0);
      //   const auto eigenvalues = compute_eigenvalues(matrix);
      //   std::cout << "Eigenvalues of (scaled = " << factor << ") rank2 KSVD of S" << std::endl;
      //   std::cout << vector_to_string(eigenvalues) << std::endl;
      // }
    }
  }


  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    if(mode == Mode::exact)
      matrix_type::vmult(dst_view, src_view);
    else if(mode == Mode::ksvd)
    {
      Assert(M, ExcMessage("M isn't initialized."));
      Assert(!(M->empty()), ExcMessage("M is empty"));
      Assert(Q, ExcMessage("Q isn't initialized."));
      Assert(!(Q->empty()), ExcMessage("Q is empty"));
      Assert(sqrt_of_Lambda, ExcMessage("sqrt_of_Lambda isn't initialized."));
      Assert(!(sqrt_of_Lambda->empty()), ExcMessage("sqrt_of_Lambda is empty"));

      AlignedVector<Number> tmp(dst_view.size());
      const auto            tmp_view = make_array_view<Number>(tmp.begin(), tmp.end());
      M->vmult(dst_view, src_view);
      Q->Tvmult(tmp_view, dst_view);
      std::transform(sqrt_of_Lambda->begin(),
                     sqrt_of_Lambda->end(),
                     tmp_view.begin(),
                     dst_view.begin(),
                     std::multiplies<Number>());
      matrix_type::vmult(tmp_view, dst_view); // TODO
      std::transform(sqrt_of_Lambda->begin(),
                     sqrt_of_Lambda->end(),
                     tmp_view.begin(),
                     dst_view.begin(),
                     std::multiplies<Number>());
      Q->vmult(tmp_view, dst_view);
      M->vmult(dst_view, tmp_view);
    }
    else if(mode == Mode::ksvd_inv)
      matrix_type::apply_inverse(dst_view, src_view);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    if(mode == Mode::exact)
      matrix_type::apply_inverse(dst_view, src_view);
    else if(mode == Mode::ksvd)
    {
      AlignedVector<Number> tmp(dst_view.size());
      const auto            tmp_view = make_array_view<Number>(tmp.begin(), tmp.end());
      Q->Tvmult(dst_view, src_view);
      std::transform(invsqrt_of_Lambda->begin(),
                     invsqrt_of_Lambda->end(),
                     dst_view.begin(),
                     tmp_view.begin(),
                     std::multiplies<Number>());
      matrix_type::apply_inverse(dst_view, tmp_view); // TODO
      std::transform(invsqrt_of_Lambda->begin(),
                     invsqrt_of_Lambda->end(),
                     dst_view.begin(),
                     tmp_view.begin(),
                     std::multiplies<Number>());
      Q->vmult(dst_view, tmp_view);
    }
    else if(mode == Mode::ksvd_inv)
      matrix_type::vmult(dst_view, src_view);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
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

  enum class Mode
  {
    exact,
    ksvd,
    ksvd_inv
  };
  Mode mode = Mode::exact;

  std::shared_ptr<const TensorProductMatrix<order, Number, n_rows_1d>> M;
  std::shared_ptr<const TensorProductMatrix<order, Number, n_rows_1d>> Q;
  std::shared_ptr<const AlignedVector<Number>>                         sqrt_of_Lambda;
  std::shared_ptr<const AlignedVector<Number>>                         invsqrt_of_Lambda;
  std::shared_ptr<const std::array<FullMatrix<double>, macro_size>>    inverse;
  std::vector<std::array<Table<2, Number>, order>>                     scratch;
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
         const double        dummy3 = 1.,
         const int           dummy1 = 0,
         const int           dummy2 = 0)
  {
    (void)dummy1;
    (void)dummy2;
    (void)dummy3;
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
    const auto n_rows_max = std::max(B->m(), C->m());

    std::lock_guard<std::mutex> lock(this->mutex);

    /// compute vmult for C A^{-1} B
    tmp_array.resize(n_rows_max + A->n());
    const auto dst_view_of_B = ArrayView(tmp_array.begin(), B->m());
    B->vmult(dst_view_of_B, src_view);
    const auto dst_view_of_AinvB = ArrayView(tmp_array.begin() + n_rows_max, A->n());
    A->apply_inverse(dst_view_of_AinvB, dst_view_of_B);
    const auto dst_view_of_CAinvB = ArrayView(tmp_array.begin(), C->m());
    C->vmult(dst_view_of_CAinvB, dst_view_of_AinvB);

    /// compute vmult for D - C A^{-1} B
    D->vmult(dst_view, src_view);
    std::transform(dst_view.cbegin(),
                   dst_view.cend(),
                   dst_view_of_CAinvB.cbegin(),
                   dst_view.begin(),
                   std::minus<Number>{});
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    Assert(A && B && C && D, ExcMessage("Not initialized."));
    if(!Sinv)
      Sinv = std::make_shared<InverseTable<Number>>(as_table());
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
  const matrix_type *                                 A = nullptr;
  const matrix_type *                                 B = nullptr;
  const matrix_type *                                 C = nullptr;
  const matrix_type *                                 D = nullptr;
  mutable std::shared_ptr<const InverseTable<Number>> Sinv;

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
 * where the inverse of S dominates the computational complexity of this
 * inversion algorithm, if we assume A^{-1} as given.
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
                       const double        factor         = 1.,
                       const int           lambda_rank    = 0,
                       const int           kronecker_rank = 0)
    : A(A_in), B(B_in), C(C_in), D(D_in)
  {
    // AssertThrow(!(kronecker_rank != 0 && lambda_rank == 0), ExcMessage("lambda_rank is
    // invalid."));
    if(kronecker_rank == 0 && lambda_rank == 0)
      S.reinit(A_in, B_in, C_in, D_in, factor);
    else if(kronecker_rank == 0 && lambda_rank != 0)
      S.reinit(A_in, B_in, C_in, D_in, factor, lambda_rank);
    else
      S.reinit(A_in, B_in, C_in, D_in, factor, lambda_rank, kronecker_rank);
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
    InverseTable<Number> inverse;
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

    /// MEMORY INEFFICENT CODE (TODO)

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

  Table<2, Number>
  as_transpose_table() const
  {
    return Tensors::transpose_matrix_to_table(*this);
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
                                                                                 schur_factor,
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

      // /// ALTERNATIVE: standard inverse based on LAPACK
      // if(!basic_inverse)
      //   basic_inverse = std::make_shared<const InverseTable<Number>>(as_table());
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

  void
  set_schur_factor(double factor)
  {
    schur_factor = factor;
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
   *
   * TODO This method has not been tested for non-quadratic blocks.
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const
  {
    AssertDimension(src.size(), n());
    AssertDimension(dst.size(), m());
    // Assert(src.size() == dst.size(),
    //        ExcMessage("TODO BlockMatrix is not quadratic.")); // quadratic !
    Assert(n_rows_1d == -1 || src.size() % n_rows_1d == 0,
           ExcMessage("Input vector sizes are not a multiple of the static size."));
    Assert(!blocks.empty(), ExcMessage("Blocks are not initialized."));
    Assert(is_valid(), ExcMessage("Matrix sizes of blocks mismatch."));
    std::fill(dst.begin(), dst.end(), static_cast<Number>(0.));

    // *** apply sliced vectors to the associated block matrix
    std::size_t row_start = 0;
    for(std::size_t row = 0; row < n_block_rows(); ++row)
    {
      const ArrayView<Number> dst_block(dst.begin() + row_start, m(row));
      std::size_t             col_start = 0;
      for(std::size_t col = 0; col < n_block_cols(); ++col)
      {
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
  mutable std::shared_ptr<const InverseTable<Number>> basic_inverse;

  /**
   * The inverse of a 2 x 2 block matrix based on approximate block Gaussian
   * elimination.
   */
  mutable std::shared_ptr<const BlockGaussianInverse<matrix_type, FastDiagSchurType>>
         fast_inverse_2x2;
  int    lambda_rank    = order;
  int    kronecker_rank = order;
  double schur_factor   = 1.;
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
                        const ArrayView<const Number> & src) { matrix.vmult_add(dst, src); },
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



/**
 * Schur complement S = D - C * A^{-1} * B
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
class SchurComplementBase
{
public:
  struct AdditionalData
  {
    typename InverseTable<Number>::AdditionalData basic_inverse;

    unsigned int kronecker_rank = numbers::invalid_unsigned_int;
    /**
     * The number of rows/columns (pair.first/pair.second) of matrix A_d, which is
     * the Kronecker factor of dimension d (array position) for the Kronecker
     * product A_1 (x) A_0
     */
    std::array<std::pair<unsigned int, unsigned int>, 2> size_of_kronecker_factors;
    std::size_t lanczos_iterations = static_cast<std::size_t>(-1);
  };

  using matrix_type = MatrixType;
  using value_type  = typename matrix_type::value_type;

  unsigned int
  m() const;

  unsigned int
  n() const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  Table<2, Number>
  as_table() const;

protected:
  SchurComplementBase(const matrix_type & A_in,
                      const matrix_type & B_in,
                      const matrix_type & C_in,
                      const matrix_type & D_in);

  const matrix_type * A = nullptr;
  const matrix_type * B = nullptr;
  const matrix_type * C = nullptr;
  const matrix_type * D = nullptr;

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
 * Schur complement S = D - C * A^{-1} * B
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
class SchurComplementBasic : public SchurComplementBase<MatrixType, Number>
{
public:
  using Base                = SchurComplementBase<MatrixType, Number>;
  using inverse_matrix_type = InverseTable<Number>;
  using typename Base::AdditionalData;

  SchurComplementBasic(const MatrixType & A_in,
                       const MatrixType & B_in,
                       const MatrixType & C_in,
                       const MatrixType & D_in);

  void
  invert(const AdditionalData & additional_data = AdditionalData{});

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  Table<2, Number>
  as_inverse_table() const;

private:
  std::shared_ptr<const InverseTable<Number>> Sinv;
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
 * where the inverse of S dominates the computational complexity of this
 * inversion algorithm, if we assume A^{-1} as given.
 */
template<typename MatrixType, typename SchurType, typename Number = typename MatrixType::value_type>
class BlockGaussianInverseBase
{
public:
  using matrix_type = MatrixType;
  using schur_type  = SchurType;
  using value_type  = Number;

  unsigned int
  m() const;

  unsigned int
  n() const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  Table<2, Number>
  as_table() const;

protected:
  BlockGaussianInverseBase(const matrix_type & A_in,
                           const matrix_type & B_in,
                           const matrix_type & C_in,
                           const schur_type &  S_in);

  const matrix_type & A;
  const matrix_type & B;
  const matrix_type & C;
  const schur_type    S;
};



template<typename MatrixType, typename Number = typename MatrixType::value_type>
class BlockGaussianInverseBasic
  : public BlockGaussianInverseBase<MatrixType, SchurComplementBasic<MatrixType, Number>, Number>
{
public:
  using Base =
    BlockGaussianInverseBase<MatrixType, SchurComplementBasic<MatrixType, Number>, Number>;
  using AdditionalData = typename Base::schur_type::AdditionalData;

  BlockGaussianInverseBasic(const MatrixType &   A_in,
                            const MatrixType &   B_in,
                            const MatrixType &   C_in,
                            const MatrixType &   D_in,
                            const AdditionalData additional_data = AdditionalData{});
};



template<typename MatrixType, typename Number = typename MatrixType::value_type>
class BlockMatrixBase
{
public:
  using matrix_type = MatrixType;
  using value_type  = Number;

  /**
   * Calls clear() first and then resizes this object to a (n_rows, n_rows)
   * block matrix.
   */
  void
  resize(const std::size_t n_rows);

  /**
   * Calls clear() first and then resizes this object to a (n_rows, n_cols)
   * block matrix.
   */
  void
  resize(const std::size_t n_rows, const std::size_t n_cols);

  /**
   * Read access to (row_index, col_index) - block.
   */
  const matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index) const;

  /**
   * Returns the number of rows of the blocks with row index @p row_index.
   */
  std::size_t
  m(const std::size_t row_index) const;

  /**
   * Returns the number of columns of the blocks with column index @p column_index.
   */
  std::size_t
  n(const std::size_t column_index) const;

  /**
   * Returns the total number of rows, this means the accumulated number of
   * rows over all blocks with fixed but arbitrary column index.
   */
  std::size_t
  m() const;

  /**
   * Returns the total number of columns, this means the accumulated number of
   * columns over all blocks with fixed but arbitrary row index.
   */
  std::size_t
  n() const;

  Table<2, Number>
  as_table() const;

  // Table<2, Number>
  // as_transpose_table() const;

  void
  vmult(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const;

  std::array<std::size_t, 2>
  size() const;

  std::size_t
  n_block_rows() const;

  std::size_t
  n_block_cols() const;

protected:
  BlockMatrixBase() = default;

  BlockMatrixBase &
  operator=(const BlockMatrixBase & other);

  /**
   * Deletes all blocks and resizes this object to a (0, 0) block matrix.
   */
  virtual void
  clear();

  std::size_t
  block_index(const std::size_t row_index, const std::size_t col_index) const;

  /**
   * Read-write access to (row_index, col_index) - block. This method resets
   * the underlying inverse due to possible modifications to this block
   * matrix.
   *
   * The user has to take care that all blocks have consistent sizes. For
   * example, all blocks with row index @p row_index must have the same number
   * of rows.
   */
  matrix_type &
  get_block(const std::size_t row_index, const std::size_t col_index);

  /**
   * Applies the function @p action to each matrix of the block
   * diagonal and the associated vector slices of @p dst and @p src.
   * The signature of the action function should be equivalent to
   * action(matrix_type& m, ArrayView<...> dst, ArrayView<const ...> src)
   *
   * TODO This method has not been tested for non-quadratic blocks.
   */
  template<typename ActionType>
  void
  blockwise_action(const ActionType &              action,
                   const ArrayView<Number> &       dst,
                   const ArrayView<const Number> & src) const;

  bool
  has_consistent_block_rows() const;

  bool
  has_consistent_block_columns() const;

  bool
  is_valid() const;

private:
  /**
   * The number of blocks per row and column
   */
  std::array<std::size_t, 2> n_blocks = {0, 0};

  /**
   * The vector containing the matrix blocks.
   */
  AlignedVector<matrix_type> blocks;
};



template<typename MatrixType, typename Number = typename MatrixType::value_type>
class BlockMatrixBasic2x2 : public BlockMatrixBase<MatrixType, Number>
{
public:
  using Base                = BlockMatrixBase<MatrixType, Number>;
  using inverse_matrix_type = BlockGaussianInverseBasic<MatrixType, Number>;
  using AdditionalData      = typename inverse_matrix_type::AdditionalData;

  BlockMatrixBasic2x2();

  BlockMatrixBasic2x2(const MatrixType & block_00,
                      const MatrixType & block_01,
                      const MatrixType & block_10,
                      const MatrixType & block_11);

  BlockMatrixBasic2x2 &
  operator=(const BlockMatrixBasic2x2 & other);

  virtual void
  clear();

  void
  resize(const std::size_t n_rows) = delete;

  void
  resize(const std::size_t n_rows, const std::size_t n_cols) = delete;

  MatrixType &
  get_block(const std::size_t row_index, const std::size_t col_index);

  /**
   * This method inverts the 2x2 block matrix by means of Gaussian block
   * elimination (internally, an instance of BlockGaussianInverseBasic-type is
   * used). Settings used for inversion are are passed as @p additional_data (to
   * this instance).
   *
   * If @p omit_inversion_of_blocks is false we invert the 0-0 block with
   * default options. If you want to invert the 0-0 block with certain
   * parameters just query this block by get_block(0, 0), then, call invert()
   * and finally set @p omit_inversion_of_blocks to true, which suppresses a
   * fresh inversion of the 0-0 block with default parameters when calling this
   * method.
   */
  void
  invert(const AdditionalData & additional_data          = AdditionalData{},
         const bool             omit_inversion_of_blocks = false);

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const;

  Table<2, Number>
  as_inverse_table() const;

private:
  std::shared_ptr<BlockGaussianInverseBasic<MatrixType, Number>> basic_inverse;
};



template<typename MatrixType, typename Number = typename MatrixType::value_type>
class BlockMatrixBasic : public BlockMatrixBase<MatrixType, Number>
{
public:
  using Base                = BlockMatrixBase<MatrixType, Number>;
  using inverse_matrix_type = InverseTable<Number>;
  using AdditionalData      = typename inverse_matrix_type::AdditionalData;

  BlockMatrixBasic() = default;

  BlockMatrixBasic &
  operator=(const BlockMatrixBasic & other);

  BlockMatrixBasic &
  operator=(const BlockMatrixBasic2x2<MatrixType, Number> & other)
  {
    Base::resize(2U, 2U);
    Base::get_block(0, 0) = other.get_block(0, 0);
    Base::get_block(0, 1) = other.get_block(0, 1);
    Base::get_block(1, 0) = other.get_block(1, 0);
    Base::get_block(1, 1) = other.get_block(1, 1);
  }

  virtual void
  clear();

  MatrixType &
  get_block(const std::size_t row_index, const std::size_t col_index);

  void
  invert(const typename InverseTable<Number>::AdditionalData & additional_data =
           typename InverseTable<Number>::AdditionalData{});

  void
  apply_inverse(const ArrayView<Number> & dst, const ArrayView<const Number> & src) const;

  Table<2, Number>
  as_inverse_table() const;

private:
  std::shared_ptr<InverseTable<Number>> basic_inverse;
};



} // namespace Tensors

#include "block_matrix.templates.h"

#endif // BLOCK_MATRIX_H_
