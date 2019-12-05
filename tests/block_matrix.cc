
/*
 * Tests for block_matrix.h:
 *
 * - vmult()
 * - as_table()
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

#include "test_utilities.h"

using namespace dealii;



template<int dim, typename Number, int n_rows_1d = -1>
struct TestBlockMatrix
{
  using BlockMatrix       = typename Tensors::BlockMatrix<dim, Number, n_rows_1d>;
  using State             = typename BlockMatrix::matrix_type::State;
  using scalar_value_type = typename ExtractScalarType<Number>::type;

  template<typename MatrixType>
  static void
  reinit_matrix_with_random_values(MatrixType &       matrix,
                                   const unsigned int n_rows,
                                   const unsigned int n_cols)
  {
    matrix.reinit(n_rows, n_cols);
    for(unsigned int i = 0; i < n_rows; ++i)
      for(unsigned int j = 0; j < n_cols; ++j)
        matrix(i, j) = random_value();
  }

  void
  fill_block(const std::size_t                     row,
             const std::size_t                     col,
             const std::vector<Table<2, Number>> & left,
             const std::vector<Table<2, Number>> & right)
  {
    AssertThrow(dim == 2, ExcMessage("Only two dimensions are valid."));
    std::vector<std::array<Table<2, Number>, dim>> tensors;
    std::transform(left.cbegin(),
                   left.cend(),
                   right.cbegin(),
                   std::back_inserter(tensors),
                   [](const auto & l, const auto & r) {
                     std::array<Table<2, Number>, dim> tensor = {l, r};
                     return tensor;
                   });
    block_matrix.get_block(row, col).reinit(tensors, State::basic);
  }

  Tensors::BlockMatrix<dim, Number, n_rows_1d> block_matrix;
};

template<typename T>
class FixBlockMatrixVmult : public testing::Test
{
protected:
  static constexpr int dim = T::template type<0>::template value<0>();
  using Number             = typename T::template type<1>;
  // using Tester                             = TestBlockMatrix<dim, Number>;
  using TesterV                            = TestBlockMatrix<dim, VectorizedArray<Number>>;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  void
  do_vectorized_test()
  {
    const auto & compare_per_lane = [this](const auto & left00,
                                           const auto & right00,
                                           const auto & left01,
                                           const auto & right01,
                                           const auto & left10,
                                           const auto & right10,
                                           const auto & left11,
                                           const auto & right11) {
      for(unsigned lane = 0; lane < macro_size; ++lane)
      {
        /// initialize block matrix
        TesterV test;
        test.block_matrix.resize(2U, 2U);
        test.fill_block(0U, 0U, left00, right00);
        test.fill_block(1U, 0U, left10, right10);
        test.fill_block(0U, 1U, left01, right01);
        test.fill_block(1U, 1U, left11, right11);

        /// initialize reference matrix
        const auto m00 = test.block_matrix.get_block(0, 0).m();
        // const auto m01 = test.block_matrix.get_block(0,1).m();
        const auto m10 = test.block_matrix.get_block(1, 0).m();
        // const auto m11 = test.block_matrix.get_block(1,1).m();
        const auto n00 = test.block_matrix.get_block(0, 0).n();
        const auto n01 = test.block_matrix.get_block(0, 1).n();
        // const auto n10 = test.block_matrix.get_block(1,0).n();
        // const auto n11 = test.block_matrix.get_block(1,1).n();
        const auto         m = m00 + m10;
        const auto         n = n00 + n01;
        FullMatrix<Number> ref_matrix(m, n);
        const auto fm00 = table_to_fullmatrix(test.block_matrix.get_block(0, 0).as_table(), lane);
        const auto fm01 = table_to_fullmatrix(test.block_matrix.get_block(0, 1).as_table(), lane);
        const auto fm10 = table_to_fullmatrix(test.block_matrix.get_block(1, 0).as_table(), lane);
        const auto fm11 = table_to_fullmatrix(test.block_matrix.get_block(1, 1).as_table(), lane);
        Tensors::insert_block(ref_matrix, fm00, 0U, 0U);
        Tensors::insert_block(ref_matrix, fm01, 0U, n00);
        Tensors::insert_block(ref_matrix, fm10, m00, 0U);
        Tensors::insert_block(ref_matrix, fm11, m00, n00);

        /// compare matrices
        std::ostringstream oss;
        const auto         full_matrix = table_to_fullmatrix(test.block_matrix.as_table(), lane);
        oss << "BlockMatrix as FullMatrix: @ lane " << lane << "\n";
        full_matrix.print_formatted(oss);
        oss << "Reference matrix: @ lane " << lane << "\n";
        ref_matrix.print_formatted(oss);
        auto diff(full_matrix);
        diff.add(-1., ref_matrix);
        EXPECT_PRED_FORMAT2(testing::FloatLE,
                            diff.frobenius_norm(),
                            std::numeric_limits<Number>::epsilon() *
                              std::min(100., ref_matrix.frobenius_norm()))
          << oss.str();
        // std::cout << oss.str();

        const bool is_2x2_block_matrix =
          test.block_matrix.n_block_rows() == 2 && test.block_matrix.n_block_cols() == 2;
        if(is_2x2_block_matrix)
        {
          /// compare Schur complement
          Tensors::SchurComplement S(test.block_matrix.get_block(0, 0),
                                     test.block_matrix.get_block(0, 1),
                                     test.block_matrix.get_block(1, 0),
                                     test.block_matrix.get_block(1, 1));
          const auto               S_full = table_to_fullmatrix(S.as_table(), lane);
          oss << "Schur complement:\n";
          S_full.print_formatted(oss);

          const auto Ainv =
            table_to_fullmatrix(test.block_matrix.get_block(0, 0).as_inverse_table(), lane);
          const auto B = table_to_fullmatrix(test.block_matrix.get_block(0, 1).as_table(), lane);
          const auto C = table_to_fullmatrix(test.block_matrix.get_block(1, 0).as_table(), lane);
          const auto D = table_to_fullmatrix(test.block_matrix.get_block(1, 1).as_table(), lane);
          FullMatrix<Number> AinvB(Ainv.m(), B.n()), Sref(C.m(), Ainv.n());
          Ainv.mmult(AinvB, B);
          C.mmult(Sref, AinvB);
          Sref *= -1.;
          Sref.add(1., D);
          oss << "Reference Schur complement:\n";
          Sref.print_formatted(oss);

          /// compare the inverse of the Schur complement
          const auto Sinv_full = table_to_fullmatrix(S.as_inverse_table(), lane);
          oss << "Inverse of Schur complement:\n";
          Sinv_full.print_formatted(oss);
          Sref.invert(Sref);
          oss << "Inverse of reference Schur complement:\n";
          Sref.print_formatted(oss);

          /// compare the inverse of the block matrix
          const auto block_inverse =
            table_to_fullmatrix(test.block_matrix.as_inverse_table(), lane);
          oss << "Inverse of block matrix @ lane " << lane << ":\n";
          block_inverse.print_formatted(oss);

          FullMatrix<Number> id(block_inverse.m(), block_inverse.n());
          block_inverse.mmult(id, full_matrix);
          oss << "A^{-1} A:\n";
          id.print_formatted(oss);

          const unsigned n_entries = id.m() * id.n();
          for(auto i = 0U; i < id.m(); ++i)
          {
            EXPECT_NEAR(id(i, i), 1., std::numeric_limits<Number>::epsilon() * 10. * n_entries);
            for(auto j = 0U; j < id.m(); ++j)
              if(i != j)
                EXPECT_NEAR(id(i, j), 0., std::numeric_limits<Number>::epsilon() * 10. * n_entries);
          }
          // std::cout << oss.str();
        }
      }
    };

    /// identity
    {
      const unsigned int                m0 = 2, n0 = 2;
      Table<2, VectorizedArray<Number>> zero(m0, n0), id(m0, n0);
      zero.fill(static_cast<VectorizedArray<Number>>(0.));
      id = zero;
      for(unsigned int i = 0; i < std::min(m0, n0); ++i)
        id(i, i) = 1.;
      std::vector<Table<2, VectorizedArray<Number>>> id_   = {id};
      std::vector<Table<2, VectorizedArray<Number>>> zero_ = {zero};
      compare_per_lane(id_, id_, zero_, zero_, zero_, zero_, id_, id_);
    }
    // /// identity x identity (not invertible)
    // {
    //   const unsigned int                m0 = 2, n0 = 2;
    //   Table<2, VectorizedArray<Number>> id(m0, n0);
    //   id.fill(static_cast<VectorizedArray<Number>>(0.));
    //   for(unsigned int i = 0; i < std::min(m0, n0); ++i)
    //     id(i, i) = 1.;
    //   std::vector<Table<2, VectorizedArray<Number>>> left  = {id};
    //   std::vector<Table<2, VectorizedArray<Number>>> right = {id};
    //   compare_per_lane(left, right, left, right, left, right, left, right);
    // }
    // /// identity x power-2 diagonal (not invertible)
    // {
    //   const unsigned int                m0 = 2, n0 = 2;
    //   Table<2, VectorizedArray<Number>> id(m0, n0), pow2(m0, n0);
    //   id.fill(static_cast<VectorizedArray<Number>>(0.));
    //   for(unsigned int i = 0; i < std::min(m0, n0); ++i)
    //     id(i, i) = 1.;
    //   pow2.fill(static_cast<VectorizedArray<Number>>(0.));
    //   for(unsigned int i = 0; i < std::min(m0, n0); ++i)
    //     pow2(i, i) = static_cast<VectorizedArray<Number>>(1 << i);
    //   std::vector<Table<2, VectorizedArray<Number>>> left  = {id};
    //   std::vector<Table<2, VectorizedArray<Number>>> right = {pow2};
    //   compare_per_lane(left, right, left, left, left, left, left, right);
    // }
    /// random
    {
      // TODO non-square matrices
      const unsigned int m0 = 2;
      const unsigned int m1 = 2;
      const unsigned int n0 = 2;
      const unsigned int n1 = 2;

      const auto & fill_left_and_right =
        [&](auto & left, auto & right, const unsigned m, const unsigned n) {
          for(auto & mat : left)
            TesterV::reinit_matrix_with_random_values(mat, m, n);
          for(auto & mat : right)
            TesterV::reinit_matrix_with_random_values(mat, m, n);
        };
      std::vector<Table<2, VectorizedArray<Number>>> left00(dim);
      std::vector<Table<2, VectorizedArray<Number>>> right00(dim);
      fill_left_and_right(left00, right00, m0, n0);
      std::vector<Table<2, VectorizedArray<Number>>> left01(dim);
      std::vector<Table<2, VectorizedArray<Number>>> right01(dim);
      fill_left_and_right(left01, right01, m0, n1);
      std::vector<Table<2, VectorizedArray<Number>>> left10(dim);
      std::vector<Table<2, VectorizedArray<Number>>> right10(dim);
      fill_left_and_right(left10, right10, m1, n0);
      std::vector<Table<2, VectorizedArray<Number>>> left11(dim);
      std::vector<Table<2, VectorizedArray<Number>>> right11(dim);
      fill_left_and_right(left11, right11, m1, n1);
      compare_per_lane(left00, right00, left01, right01, left10, right10, left11, right11);
    }
  }
};

TYPED_TEST_SUITE_P(FixBlockMatrixVmult);
TYPED_TEST_P(FixBlockMatrixVmult, CompareVmultAndMatrix)
{
  using Fixture = FixBlockMatrixVmult<TypeParam>;
  Fixture::do_vectorized_test();
}

REGISTER_TYPED_TEST_SUITE_P(FixBlockMatrixVmult, CompareVmultAndMatrix);

using ParamsTwoDimensionsDouble = testing::Types<Util::TypeList<Util::NonTypeParams<2>, double>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensionsDouble, FixBlockMatrixVmult, ParamsTwoDimensionsDouble);
