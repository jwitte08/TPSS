
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

#include <fstream>

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
                     std::array<Table<2, Number>, dim> tensor = {r, l};
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
  enum class TestVariant
  {
    matrix,
    inverse,
    schur,
    inverse_schur
  };

  static constexpr int dim                 = T::template type<0>::template value<0>();
  using Number                             = typename T::template type<1>;
  using TesterV                            = TestBlockMatrix<dim, VectorizedArray<Number>>;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  void
  SetUp() override
  {
    ofs.open("block_matrix.log", std::ios_base::app);
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, true);
  }

  void
  TearDown() override
  {
    ofs.close();
  }

  void
  compare_matrix(const FullMatrix<Number> & matrix, const FullMatrix<Number> & other)
  {
    Util::compare_matrix(matrix, other, *pcout_owned);
  }

  void
  compare_inverse_matrix(const FullMatrix<Number> & inverse_matrix,
                         const FullMatrix<Number> & other)
  {
    Util::compare_inverse_matrix(inverse_matrix, other, *pcout_owned);
  }

  void
  do_vectorized_test(const TestVariant test_variant = TestVariant::matrix)
  {
    const auto & compare_per_lane = [&](const auto & left00,
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
        const auto         m00 = test.block_matrix.get_block(0, 0).m();
        const auto         m10 = test.block_matrix.get_block(1, 0).m();
        const auto         n00 = test.block_matrix.get_block(0, 0).n();
        const auto         n01 = test.block_matrix.get_block(0, 1).n();
        const auto         m   = m00 + m10;
        const auto         n   = n00 + n01;
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
        if(TestVariant::matrix == test_variant)
        {
          const auto block_matrix_full = table_to_fullmatrix(test.block_matrix.as_table(), lane);
          *pcout_owned << "Compare block matrix @ lane " << lane << ":\n";
          compare_matrix(block_matrix_full, ref_matrix);
        }

        const bool is_2x2_block_matrix =
          test.block_matrix.n_block_rows() == 2 && test.block_matrix.n_block_cols() == 2;
        if(is_2x2_block_matrix)
        {
          std::ostringstream oss;

          /// compare Schur complement
          Tensors::SchurComplement S(test.block_matrix.get_block(0, 0),
                                     test.block_matrix.get_block(0, 1),
                                     test.block_matrix.get_block(1, 0),
                                     test.block_matrix.get_block(1, 1));
          const auto               S_full = table_to_fullmatrix(S.as_table(), lane);
          const auto               Ainv =
            table_to_fullmatrix(test.block_matrix.get_block(0, 0).as_inverse_table(), lane);
          const auto B = table_to_fullmatrix(test.block_matrix.get_block(0, 1).as_table(), lane);
          const auto C = table_to_fullmatrix(test.block_matrix.get_block(1, 0).as_table(), lane);
          const auto D = table_to_fullmatrix(test.block_matrix.get_block(1, 1).as_table(), lane);
          FullMatrix<Number> AinvB(Ainv.m(), B.n()), Sref(C.m(), Ainv.n());
          Ainv.mmult(AinvB, B);
          C.mmult(Sref, AinvB);
          Sref *= -1.;
          Sref.add(1., D);
          if(TestVariant::schur == test_variant)
          {
            *pcout_owned << "Compare Schur complement @ lane " << lane << ":\n";
            compare_matrix(S_full, Sref);
          }

          /// compare the inverse of the Schur complement
          if(TestVariant::inverse_schur == test_variant)
          {
            const auto Sinv_full = table_to_fullmatrix(S.as_inverse_table(), lane);
            *pcout_owned << "Compare inverse Schur complement @ lane " << lane << ":\n";
            compare_inverse_matrix(Sinv_full, Sref);
          }

          /// compare the inverse of the block matrix
          if(TestVariant::inverse == test_variant)
          {
            const auto block_matrix_inverse =
              table_to_fullmatrix(test.block_matrix.as_inverse_table(), lane);
            *pcout_owned << "Compare inverse block matrix @ lane " << lane << ":\n";
            compare_inverse_matrix(block_matrix_inverse, ref_matrix);
          }
        }
      }
    };

    /// random-scaled identity
    {
      const unsigned int                m0 = 2, n0 = 2;
      Table<2, VectorizedArray<Number>> zero(m0, n0), id(m0, n0);
      zero.fill(static_cast<VectorizedArray<Number>>(0.));
      id               = zero;
      const auto value = make_random_value<VectorizedArray<Number>>();
      for(unsigned int i = 0; i < std::min(m0, n0); ++i)
        id(i, i) = value;
      std::vector<Table<2, VectorizedArray<Number>>> id_   = {id};
      std::vector<Table<2, VectorizedArray<Number>>> zero_ = {zero};
      compare_per_lane(id_, id_, zero_, zero_, zero_, zero_, id_, id_);
    }
    /// random full
    {
      // TODO non-square matrices
      const unsigned int m0 = 2;
      const unsigned int m1 = 2;
      const unsigned int n0 = 2;
      const unsigned int n1 = 2;

      const auto & fill_left_and_right =
        [&](auto & left, auto & right, const unsigned m, const unsigned n) {
          for(auto & mat : left)
            fill_matrix_with_random_values(mat, m, n);
          for(auto & mat : right)
            fill_matrix_with_random_values(mat, m, n);
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

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;
};

TYPED_TEST_SUITE_P(FixBlockMatrixVmult);
TYPED_TEST_P(FixBlockMatrixVmult, CompareMatrix)
{
  using Fixture = FixBlockMatrixVmult<TypeParam>;
  Fixture::do_vectorized_test();
}

TYPED_TEST_P(FixBlockMatrixVmult, CompareSchur)
{
  using Fixture = FixBlockMatrixVmult<TypeParam>;
  Fixture::do_vectorized_test(Fixture::TestVariant::schur);
}

TYPED_TEST_P(FixBlockMatrixVmult, CompareInverseSchur)
{
  using Fixture = FixBlockMatrixVmult<TypeParam>;
  Fixture::do_vectorized_test(Fixture::TestVariant::inverse_schur);
}

TYPED_TEST_P(FixBlockMatrixVmult, CompareInverse)
{
  using Fixture = FixBlockMatrixVmult<TypeParam>;
  Fixture::do_vectorized_test(Fixture::TestVariant::inverse);
}

REGISTER_TYPED_TEST_SUITE_P(FixBlockMatrixVmult,
                            CompareMatrix,
                            CompareSchur,
                            CompareInverseSchur,
                            CompareInverse);

using ParamsTwoDimensionsDouble = testing::Types<Util::TypeList<Util::NonTypeParams<2>, double>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensionsDouble, FixBlockMatrixVmult, ParamsTwoDimensionsDouble);
