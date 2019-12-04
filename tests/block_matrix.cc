
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
      }
    };

    /// identity
    {
      const unsigned int                m0 = 2, n0 = 2;
      Table<2, VectorizedArray<Number>> id(m0, n0);
      id.fill(static_cast<VectorizedArray<Number>>(0.));
      for(unsigned int i = 0; i < m0; ++i)
        id(i, i) = 1.;
      std::vector<Table<2, VectorizedArray<Number>>> left  = {id};
      std::vector<Table<2, VectorizedArray<Number>>> right = {id};
      compare_per_lane(left, right, left, right, left, right, left, right);
    }
    /// random
    {
      // TODO non-quadratic matrices
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
