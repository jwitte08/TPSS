
/*
 * Tests for tensor_product_matrix.h:
 *
 * - vmult()
 * - as_table()
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/tensor_product_matrix.h"

#include "test_utilities.h"

using namespace dealii;

template<int dim, typename Number, int n_rows_1d = -1>
struct TestTensorProductMatrix
{
  using TPM               = typename Tensors::TensorProductMatrix<dim, Number, n_rows_1d>;
  using scalar_value_type = typename ExtractScalarType<Number>::type;

  void
  compare_vmult(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
  {
    TPM tp_matrix;
    tp_matrix.reinit(elementary_tensors, state);
    AlignedVector<Number> dst(tp_matrix.n());
    fill_with_random_values(dst);
    const auto src(dst);

    const auto src_view = make_array_view(src.begin(), src.end());
    const auto dst_view = make_array_view(dst.begin(), dst.end());
    tp_matrix.vmult(dst_view, src_view);

    const auto src_other = array_view_to_vector(src_view, lane);
    auto       dst_other(src_other);
    other.vmult(dst_other, src_other);

    const auto dst_cview = make_array_view<const Number>(dst.begin(), dst.end());
    auto       diff(array_view_to_vector(dst_cview, lane));
    diff -= dst_other;
    // TODO write PredicateFormatter (see googletest's advanced.md on github) to
    // compare against zero
    EXPECT_PRED_FORMAT2(testing::FloatLE,
                        diff.l2_norm(),
                        std::numeric_limits<scalar_value_type>::epsilon() *
                          std::max(10., dst_other.l2_norm()));
  }

  void
  compare_matrix(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
  {
    TPM tp_matrix;
    tp_matrix.reinit(elementary_tensors, state);
    auto full_matrix = table_to_fullmatrix(tp_matrix.as_table(), lane);
    auto diff(full_matrix);
    diff.add(-1., other);

    std::ostringstream oss;
    oss << "TP matrix:\n";
    full_matrix.print_formatted(oss);
    oss << "Reference matrix:\n";
    other.print_formatted(oss);
    const double n_entries = other.m() * other.n();
    EXPECT_PRED_FORMAT2(testing::FloatLE,
                        diff.frobenius_norm() / n_entries,
                        std::numeric_limits<scalar_value_type>::epsilon() *
                          std::max(10., other.frobenius_norm() / n_entries))
      << oss.str();
  }

  void
  compare_inverse_matrix(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
  {
    TPM tp_matrix;
    tp_matrix.reinit(elementary_tensors, state);
    const auto inverse_matrix = table_to_fullmatrix(tp_matrix.as_inverse_table(), lane);
    LAPACKFullMatrix<scalar_value_type> lapack_other;
    lapack_other.reinit(other.m());
    lapack_other = other;
    lapack_other.compute_inverse_svd();
    const auto & lapack_other_as_table = Tensors::lapack_matrix_to_table(lapack_other);
    const auto   inverse_other         = table_to_fullmatrix(lapack_other_as_table);

    std::ostringstream oss;
    oss << "Inverse tensor product matrix:\n";
    inverse_matrix.print_formatted(oss);
    oss << "Reference inverse matrix:\n";
    inverse_other.print_formatted(oss);
    auto diff(inverse_matrix);
    diff.add(-1., inverse_other);
    const double n_entries = inverse_other.m() * inverse_other.n();
    EXPECT_PRED_FORMAT2(testing::FloatLE,
                        diff.frobenius_norm() / n_entries,
                        std::numeric_limits<scalar_value_type>::epsilon() *
                          std::max(10., inverse_other.frobenius_norm() / n_entries))
      << "@ lane " << lane << ":\n"
      << oss.str();
  }

  void
  compare_copy()
  {
    TPM tp_matrix;
    tp_matrix.reinit(elementary_tensors, state);
    TPM tp_matrix_copy;
    tp_matrix_copy        = tp_matrix;
    const auto table      = tp_matrix.as_table();
    const auto table_copy = tp_matrix_copy.as_table();
    for(auto i = 0U; i < table.n_rows(); ++i)
      for(auto j = 0U; j < table.n_cols(); ++j)
        EXPECT_NEAR(table(i, j),
                    table_copy(i, j),
                    std::numeric_limits<scalar_value_type>::epsilon());
  }

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

  std::vector<std::array<Table<2, Number>, dim>> elementary_tensors;
  typename TPM::State                            state = TPM::State::invalid;
};

template<typename T>
class FixTensorProductMatrix : public testing::Test
{
protected:
  enum class TestVariant
  {
    vmult,
    apply_inverse,
    copy
  };

  static constexpr int dim                 = T::template type<0>::template value<0>();
  using Number                             = typename T::template type<1>;
  using Tester                             = TestTensorProductMatrix<dim, Number>;
  using TesterV                            = TestTensorProductMatrix<dim, VectorizedArray<Number>>;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  FullMatrix<Number>
  assemble_reference(const std::vector<FullMatrix<Number>> & left,
                     const std::vector<FullMatrix<Number>> & right)
  {
    std::vector<FullMatrix<Number>> products;
    std::transform(left.cbegin(),
                   left.cend(),
                   right.cbegin(),
                   std::back_inserter(products),
                   [](const auto & l, const auto & r) { return Tensors::kronecker_product(l, r); });
    FullMatrix<Number> sum_of_products(products.front().m());
    sum_of_products = 0.;
    for(const auto & mat : products)
      sum_of_products = Tensors::sum(sum_of_products, mat);
    return sum_of_products;
  }

  void
  test_vmult_or_apply_inverse(const TestVariant test_variant = TestVariant::vmult)
  {
    using State          = typename Tester::TPM::State;
    const unsigned int m = 3;

    const auto & test_impl = [test_variant, this](const auto & left, const auto & right) {
      Tester test;
      std::transform(left.cbegin(),
                     left.cend(),
                     right.cbegin(),
                     std::back_inserter(test.elementary_tensors),
                     [](const auto & l, const auto & r) {
                       std::array<Table<2, Number>, dim> tensor = {r, l};
                       return tensor;
                     });

      auto sum_of_products = assemble_reference(left, right);
      test.state           = State::basic;
      if(test_variant == TestVariant::vmult)
      {
        test.compare_vmult(sum_of_products);
        test.compare_matrix(sum_of_products);
      }
      else if(test_variant == TestVariant::apply_inverse)
        test.compare_inverse_matrix(sum_of_products);
      else if(test_variant == TestVariant::copy)
        test.compare_copy();
    };

    /// identity
    {
      std::vector<FullMatrix<Number>> left(1);
      std::vector<FullMatrix<Number>> right(1);
      left.front()       = IdentityMatrix(m);
      right.front()      = left.front();
      left.front()       = 0.;
      left.front()(0, 0) = 1.;
      test_impl(left, right);
    }
    /// random
    {
      std::vector<FullMatrix<Number>> left(dim);
      std::vector<FullMatrix<Number>> right(dim);
      for(auto & mat : left)
        Tester::reinit_matrix_with_random_values(mat, m, m);
      for(auto & mat : right)
        Tester::reinit_matrix_with_random_values(mat, m, m);
      test_impl(left, right);
    }
  }

  void
  test_vmult_or_apply_inverseV(const TestVariant test_variant = TestVariant::vmult)
  {
    using State          = typename TesterV::TPM::State;
    const unsigned int m = 3;

    const auto & compare_per_lane = [test_variant, this](const auto & left, const auto & right) {
      for(unsigned lane = 0; lane < macro_size; ++lane)
      {
        TesterV test;
        std::transform(left.cbegin(),
                       left.cend(),
                       right.cbegin(),
                       std::back_inserter(test.elementary_tensors),
                       [](const auto & l, const auto & r) {
                         std::array<Table<2, VectorizedArray<Number>>, dim> tensor = {r, l};
                         return tensor;
                       });
        std::vector<FullMatrix<Number>> left_full;
        std::vector<FullMatrix<Number>> right_full;
        std::transform(left.cbegin(),
                       left.cend(),
                       std::back_inserter(left_full),
                       [lane](const auto & table) { return table_to_fullmatrix(table, lane); });
        std::transform(right.cbegin(),
                       right.cend(),
                       std::back_inserter(right_full),
                       [lane](const auto & table) { return table_to_fullmatrix(table, lane); });
        const auto sum_of_products = assemble_reference(left_full, right_full);
        test.state                 = State::basic;
        if(test_variant == TestVariant::vmult)
        {
          test.compare_vmult(sum_of_products, lane);
          test.compare_matrix(sum_of_products, lane);
        }
        else if(test_variant == TestVariant::apply_inverse)
          test.compare_inverse_matrix(sum_of_products, lane);
      }
    };

    /// identity
    {
      std::vector<Table<2, VectorizedArray<Number>>> left(1);
      std::vector<Table<2, VectorizedArray<Number>>> right(1);
      auto &                                         l = left.front();
      l.reinit(m, m);
      l.fill(static_cast<VectorizedArray<Number>>(0.));
      for(unsigned int mm = 0; mm < m; ++mm)
        l(mm, mm) = 1.;
      auto & r = right.front();
      r.reinit(m, m);
      r(0, 0) = 1.;
      compare_per_lane(left, right);
    }
    /// random
    {
      std::vector<Table<2, VectorizedArray<Number>>> left(dim);
      std::vector<Table<2, VectorizedArray<Number>>> right(dim);
      for(auto & mat : left)
        TesterV::reinit_matrix_with_random_values(mat, m, m);
      for(auto & mat : right)
        TesterV::reinit_matrix_with_random_values(mat, m, m);
      compare_per_lane(left, right);
    }
  }
};

TYPED_TEST_SUITE_P(FixTensorProductMatrix);
TYPED_TEST_P(FixTensorProductMatrix, VmultAndMatrix)
{
  using Fixture = FixTensorProductMatrix<TypeParam>;
  Fixture::test_vmult_or_apply_inverse();
  Fixture::test_vmult_or_apply_inverseV();
}

TYPED_TEST_P(FixTensorProductMatrix, ApplyInverse)
{
  using Fixture     = FixTensorProductMatrix<TypeParam>;
  using TestVariant = typename Fixture::TestVariant;
  Fixture::test_vmult_or_apply_inverse(TestVariant::apply_inverse);
  Fixture::test_vmult_or_apply_inverseV(TestVariant::apply_inverse);
}

TYPED_TEST_P(FixTensorProductMatrix, Copy)
{
  using Fixture     = FixTensorProductMatrix<TypeParam>;
  using TestVariant = typename Fixture::TestVariant;
  Fixture::test_vmult_or_apply_inverse(TestVariant::copy);
}

REGISTER_TYPED_TEST_SUITE_P(FixTensorProductMatrix, VmultAndMatrix, ApplyInverse, Copy);

using ParamsTwoDimensionsDouble = testing::Types<Util::TypeList<Util::NonTypeParams<2>, double>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensionsDouble,
                               FixTensorProductMatrix,
                               ParamsTwoDimensionsDouble);
