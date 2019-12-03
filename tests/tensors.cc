
/*
 * Tests for tensors.h:
 *
 * - transforming multi- into uni-index and vice versa
 *
 * NOTE: In addition, the tests provide a sandbox for more advanced gtest
 * functionalities.
 *
 *  Created on: Dec 07, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"

#include "test_utilities.h"

using namespace dealii;

template<int order>
void
test_index_map_isotropic_impl(const unsigned int size)
{
  const unsigned int n_elems = Utilities::pow(size, order);
  for(unsigned int i = 0; i < n_elems; ++i)
  {
    /// via isotropic method
    const auto multii = Tensors::uni_to_multiindex<order>(i, size);
    /// via anisotropic method
    std::array<unsigned, order> sizes;
    sizes.fill(size);
    const auto multii_ani = Tensors::uni_to_multiindex<order>(i, sizes);
    EXPECT_EQ(multii, multii_ani);

    /// compare inverse isotropic method
    const auto ii = Tensors::multi_to_uniindex<order>(multii, size);
    EXPECT_EQ(i, ii)
      << "The composition of the multi-index map and its inverse isn't mismatches the identity map.";
  }
}

template<int order>
void
test_index_map_anisotropic_impl(const std::array<unsigned int, order> & sizes)
{
  const unsigned int n_elems =
    std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<unsigned>());
  for(unsigned int i = 0; i < n_elems; ++i)
  {
    const auto multii = Tensors::uni_to_multiindex<order>(i, sizes);
    const auto ii     = Tensors::multi_to_uniindex<order>(multii, sizes);
    EXPECT_EQ(i, ii)
      << "The composition of the multi-index map and its inverse isn't mismatches the identity map.";
  }
}



/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Regular tests described by the closed function macro
/// TEST(TestSuiteName,TestName)

TEST(Tensors, MultiIndexIsotropic)
{
  constexpr int      order = 4;
  const unsigned int size  = 5;
  test_index_map_isotropic_impl<order>(size);
}

TEST(Tensors, MultiIndexAnisotropic)
{
  constexpr int                   order = 5;
  std::array<unsigned int, order> sizes = {6, 1, 7, 5, 3};
  test_index_map_anisotropic_impl<order>(sizes);
}



/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Value-parametrized tests via TEST_P(TestSuiteName, TestName)

class TestMultiIndexBase : public testing::Test
{
protected:
  TestMultiIndexBase()
  {
    /// You can do set-up work for each test here.
  }

  virtual ~TestMultiIndexBase() override
  {
    /// You can do clean-up work that doesn't throw exceptions here.
  }

  /// If the constructor and destructor are not enough for setting up
  /// and cleaning up each test, you can define the following methods:

  virtual void
  SetUp() override
  {
    /// Code here will be called immediately after the constructor (right
    /// before each test).
  }

  virtual void
  TearDown() override
  {
    /// Code here will be called immediately after each test (right
    /// before the destructor).
  }

  /// Here common parameters, values and methods might be defined to be used by
  /// derived tests.
  static constexpr int order_ = 5;
};

/// The fixture class (in gtest speak) which is called by TEST_F if no
/// value-parameters are queried. With value-parameterization the setup macro is
/// TEST_P(FixtureClass, TestName).
class TestMultiIndexIsotropic : public TestMultiIndexBase,
                                public testing::WithParamInterface<unsigned int>
{
protected:
  void
  test_index_map(const unsigned int size)
  {
    test_index_map_isotropic_impl<order_>(size);
  }
};

/// NOTE: It is necessary that the name of the test suite equals the name of the
/// fixture class, here 'TestMultiIndexIsotropic'.
TEST_P(TestMultiIndexIsotropic, VarySize)
{
  test_index_map(GetParam());
}

/// Finally, instantiate the value-parameters by means of the convenience
/// generators, e.g. a range of values (Range), set of values (Values) and
/// C-like or STL container (ValuesIn).
INSTANTIATE_TEST_SUITE_P(StridedRange, TestMultiIndexIsotropic, testing::Range(1U, 8U, 3U));
INSTANTIATE_TEST_SUITE_P(Values, TestMultiIndexIsotropic, testing::Values(2U, 5U, 10U));
INSTANTIATE_TEST_SUITE_P(ValuesIn, TestMultiIndexIsotropic, testing::ValuesIn({3U, 6U, 9U}));



/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Value-parametrized tests with constant expressions

/// Gtest provides a convenience class TestWithParam<T> that is derived from
/// Test and WithParamInterface<T>, the two base classes we require to drive the
/// TEST_P() macro.

/// NOTE: It is not possible to vary constant expressions and/or template
/// parameters as value-parameters, e.g. the tensor order in our model
/// tests. The GetParam() method is not constexpr such that a run time parameter
/// is returned.
struct ArrayWrap
{
  ArrayWrap() = default;

  ArrayWrap(std::array<unsigned, 3> && array_in)
  {
    sizes.swap(array_in);
  }

  std::array<unsigned int, 3> sizes = {3U, 7U, 12U};
};

class TestMultiIndexAnisotropic : public testing::TestWithParam<std::array<unsigned int, 3>>
{
protected:
  void
  test_index_map_impl()
  {
    const auto & prm = GetParam();
    test_index_map_anisotropic_impl<3>(prm);
  }
};

TEST_P(TestMultiIndexAnisotropic, VarySizes)
{
  test_index_map_impl();
}

/// NOTE: One has to pass values via the gtest parameter generators Range,
/// Values, ValuesIn, Bool and Combine (read the advanced.md documenation).
INSTANTIATE_TEST_SUITE_P(InitList,
                         TestMultiIndexAnisotropic,
                         testing::Values(ArrayWrap({9U, 8U, 7U}).sizes,
                                         ArrayWrap{}.sizes,
                                         std::array<unsigned int, 3>{12U, 8U, 9U}));



/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Type-parametrized tests are setup by the macro TYPED_TEST_P(TestSuiteName,
/// TestName). Again, the templated fixture class has to be named in terms of
/// the TestSuiteName. Note the registration of all tests by means of
/// REGISTER_TYPED_TEST_SUITE_P(TestSuiteName, TestNameA, TestNameB, ...)
/// followed by the instantiation macro
/// INSTANTIATE_TYPED_TEST_SUITE_P(InstantiationName, TestSuiteName,
/// TypeSet). TypeSet is a typedef of testing::Types<TypeA,TypeB,...> with the
/// type-parameters TypeA, TypeB, ...

/// Empty fixture class. Everything is defined within the test macro.
template<typename T>
class TestMultiIndexIsotropicStatic : public testing::Test
{
};


/// In contrast to previous test macros it is necessary to define the test suite
/// explicitly.
TYPED_TEST_SUITE_P(TestMultiIndexIsotropicStatic);

/// The template parameter of the fixture class is accessible as 'TypeParam'.
TYPED_TEST_P(TestMultiIndexIsotropicStatic, FixSizeVaryOrder)
{
  constexpr auto order_ = TypeParam::template value<0>();
  test_index_map_isotropic_impl<order_>(5);
}

/// The template parameter of the fixture class is accessible as 'TypeParam'.
TYPED_TEST_P(TestMultiIndexIsotropicStatic, VarySizeAndOrder)
{
  constexpr auto order_ = TypeParam::template value<0>();
  /// As long as 'unsigned int' is unique within template parameter pack of
  /// NonTypeParams we are able to access it via type. For more details see the
  /// documenation of std::get for std::tuple.
  constexpr auto size_ = TypeParam::template value<unsigned int>();
  test_index_map_isotropic_impl<order_>(size_);
}

/// First, register all the tests defined by their names.
REGISTER_TYPED_TEST_SUITE_P(TestMultiIndexIsotropicStatic, FixSizeVaryOrder, VarySizeAndOrder);

/// The instantiation of the fixture class template is the final step.
using TestParams = testing::Types<Util::NonTypeParams<1, 1U>, Util::NonTypeParams<6, 4U>>;
INSTANTIATE_TYPED_TEST_SUITE_P(NonTypeParams, TestMultiIndexIsotropicStatic, TestParams);



/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Tests for Tensors::TensorProductMatrix.

template<int dim, typename Number, int n_rows_1d = -1>
struct TestTensorProductMatrix
{
  using TPM               = typename Tensors::TensorProductMatrix<dim, Number, n_rows_1d>;
  using scalar_value_type = typename ExtractScalarType<Number>::type;

  void
  compare_vmult(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
  {
    TPM tp_matrix;
    tp_matrix.reinit(left, right, state);
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
                          std::min(100., dst_other.l2_norm()));
  }

  void
  compare_matrix(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
  {
    TPM tp_matrix;
    tp_matrix.reinit(left, right, state);
    auto full_matrix = table_to_fullmatrix(tp_matrix.as_table(), lane);
    auto diff(full_matrix);
    diff.add(-1., other);

    std::ostringstream oss;
    oss << "TP matrix:\n";
    full_matrix.print_formatted(oss);
    oss << "Reference matrix:\n";
    other.print_formatted(oss);
    EXPECT_PRED_FORMAT2(testing::FloatLE,
                        diff.frobenius_norm(),
                        std::numeric_limits<scalar_value_type>::epsilon() *
                          std::min(100., other.frobenius_norm()))
      << oss.str();
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

  std::vector<Table<2, Number>> left;
  std::vector<Table<2, Number>> right;
  typename TPM::State           state = TPM::State::invalid;
};

template<typename T>
class FixTensorProductMatrixVmult : public testing::Test
{
protected:
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
  do_test()
  {
    using State = typename Tester::TPM::State;
    Tester test;

    const unsigned int m = 3;

    /// identity
    {
      std::vector<FullMatrix<Number>> left(1);
      std::vector<FullMatrix<Number>> right(1);
      left.front()       = IdentityMatrix(m);
      right.front()      = left.front();
      left.front()       = 0.;
      left.front()(0, 0) = 1.;
      test.left.clear(), test.right.clear();
      std::copy(left.cbegin(), left.cend(), std::back_inserter(test.left));
      std::copy(right.cbegin(), right.cend(), std::back_inserter(test.right));

      auto sum_of_products = assemble_reference(left, right);
      test.state           = State::basic;
      test.compare_vmult(sum_of_products);
      test.compare_matrix(sum_of_products);
    }
    /// random
    {
      std::vector<FullMatrix<Number>> left(dim);
      std::vector<FullMatrix<Number>> right(dim);
      for(auto & mat : left)
        Tester::reinit_matrix_with_random_values(mat, m, m);
      for(auto & mat : right)
        Tester::reinit_matrix_with_random_values(mat, m, m);
      test.left.clear(), test.right.clear();
      std::copy(left.cbegin(), left.cend(), std::back_inserter(test.left));
      std::copy(right.cbegin(), right.cend(), std::back_inserter(test.right));

      auto sum_of_products = assemble_reference(left, right);
      test.state           = State::basic;
      test.compare_vmult(sum_of_products);
      test.compare_matrix(sum_of_products);
    }
  }

  void
  do_vectorized_test()
  {
    using State          = typename TesterV::TPM::State;
    const unsigned int m = 3;

    const auto & compare_per_lane = [this](const auto & left, const auto & right) {
      for(unsigned lane = 0; lane < macro_size; ++lane)
      {
        TesterV test;
        std::copy(left.cbegin(), left.cend(), std::back_inserter(test.left));
        std::copy(right.cbegin(), right.cend(), std::back_inserter(test.right));
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
        test.compare_vmult(sum_of_products, lane);
        test.compare_matrix(sum_of_products, lane);
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

TYPED_TEST_SUITE_P(FixTensorProductMatrixVmult);
TYPED_TEST_P(FixTensorProductMatrixVmult, CompareVmultAndMatrix)
{
  using Fixture = FixTensorProductMatrixVmult<TypeParam>;
  Fixture::do_test();
  Fixture::do_vectorized_test();
}

REGISTER_TYPED_TEST_SUITE_P(FixTensorProductMatrixVmult, CompareVmultAndMatrix);

using ParamsTwoDimensionsDouble = testing::Types<Util::TypeList<Util::NonTypeParams<2>, double>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensionsDouble,
                               FixTensorProductMatrixVmult,
                               ParamsTwoDimensionsDouble);

/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// Tests for Tensors::TensorProductMatrix.

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
    block_matrix.get_block(row, col).reinit(left, right, State::basic);
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
