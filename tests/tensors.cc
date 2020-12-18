
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
    // std::cout << "i = " << i << " -> ii = " << Tensors::multiindex_to_string<order>(multii) <<
    // std::endl;
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
