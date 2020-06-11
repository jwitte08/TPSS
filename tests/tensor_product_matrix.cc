
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

#include <fstream>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/tensor_product_matrix.h"

#include "test_utilities.h"

using namespace dealii;

template<int dim, typename Number, int n_rows_1d = -1>
struct TestTensorProductMatrix
{
  using TPM               = typename Tensors::TensorProductMatrix<dim, Number, n_rows_1d>;
  using scalar_value_type = typename ExtractScalarType<Number>::type;

  TestTensorProductMatrix(ConditionalOStream & pcout_in) : pcout(pcout_in){};

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
    auto tp_matrix_full = table_to_fullmatrix(tp_matrix.as_table(), lane);
    Util::compare_matrix(tp_matrix_full, other, pcout);
  }

  void
  compare_inverse_matrix(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
  {
    TPM tp_matrix;
    tp_matrix.reinit(elementary_tensors, state);
    const auto inverse_matrix = table_to_fullmatrix(tp_matrix.as_inverse_table(), lane);
    Util::compare_inverse_matrix(inverse_matrix, other, pcout);
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

  ConditionalOStream &                           pcout;
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
  static constexpr unsigned int macro_size = VectorizedArray<Number>::size();

  void
  SetUp() override
  {
    ofs.open("tensor_product_matrix.log", std::ios_base::app);
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, true);
  }

  void
  TearDown() override
  {
    ofs.close();
  }

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
      Tester test(*pcout_owned);
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
        *pcout_owned << "compare tensor product matrix:\n";
        test.compare_vmult(sum_of_products);
        test.compare_matrix(sum_of_products);
      }
      else if(test_variant == TestVariant::apply_inverse)
      {
        *pcout_owned << "compare inverse tensor product matrix:\n";
        test.compare_inverse_matrix(sum_of_products);
      }
      else if(test_variant == TestVariant::copy)
      {
        *pcout_owned << "test copy-assign:\n";
        test.compare_copy();
      }
    };

    /// identity x identity
    {
      std::vector<FullMatrix<Number>> left(1);
      std::vector<FullMatrix<Number>> right(1);
      left.front()  = IdentityMatrix(m);
      right.front() = left.front();
      test_impl(left, right);
    }
    /// random
    {
      std::vector<FullMatrix<Number>> left(dim);
      std::vector<FullMatrix<Number>> right(dim);
      for(auto & mat : left)
        fill_matrix_with_random_values(mat, m, m);
      for(auto & mat : right)
        fill_matrix_with_random_values(mat, m, m);
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
        TesterV test(*pcout_owned);
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
          *pcout_owned << "compare tensor product matrix @ lane" << lane << ":\n";
          test.compare_vmult(sum_of_products, lane);
          test.compare_matrix(sum_of_products, lane);
        }
        else if(test_variant == TestVariant::apply_inverse)
        {
          *pcout_owned << "compare inverse tensor product matrix @ lane" << lane << ":\n";
          test.compare_inverse_matrix(sum_of_products, lane);
        }
      }
    };

    /// random-scaled identity
    {
      std::vector<Table<2, VectorizedArray<Number>>> left(1);
      std::vector<Table<2, VectorizedArray<Number>>> right(1);
      auto &                                         l = left.front();
      l.reinit(m, m);
      l.fill(static_cast<VectorizedArray<Number>>(0.));
      const auto value = make_random_value<VectorizedArray<Number>>();
      for(unsigned int mm = 0; mm < m; ++mm)
        l(mm, mm) = value;
      right.front() = l;
      compare_per_lane(left, right);
    }
    /// random
    {
      std::vector<Table<2, VectorizedArray<Number>>> left(dim);
      std::vector<Table<2, VectorizedArray<Number>>> right(dim);
      for(auto & mat : left)
        fill_matrix_with_random_values(mat, m, m);
      for(auto & mat : right)
        fill_matrix_with_random_values(mat, m, m);
      compare_per_lane(left, right);
    }
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;
};

TYPED_TEST_SUITE_P(FixTensorProductMatrix);
TYPED_TEST_P(FixTensorProductMatrix, VmultAndMatrix)
{
  using Fixture = FixTensorProductMatrix<TypeParam>;
  Fixture::test_vmult_or_apply_inverse();
}

TYPED_TEST_P(FixTensorProductMatrix, ApplyInverse)
{
  using Fixture     = FixTensorProductMatrix<TypeParam>;
  using TestVariant = typename Fixture::TestVariant;
  Fixture::test_vmult_or_apply_inverse(TestVariant::apply_inverse);
}

TYPED_TEST_P(FixTensorProductMatrix, Copy)
{
  using Fixture     = FixTensorProductMatrix<TypeParam>;
  using TestVariant = typename Fixture::TestVariant;
  Fixture::test_vmult_or_apply_inverse(TestVariant::copy);
}

TYPED_TEST_P(FixTensorProductMatrix, VmultAndMatrixVectorized)
{
  using Fixture = FixTensorProductMatrix<TypeParam>;
  Fixture::test_vmult_or_apply_inverseV();
}

TYPED_TEST_P(FixTensorProductMatrix, ApplyInverseVectorized)
{
  using Fixture     = FixTensorProductMatrix<TypeParam>;
  using TestVariant = typename Fixture::TestVariant;
  Fixture::test_vmult_or_apply_inverseV(TestVariant::apply_inverse);
}

TYPED_TEST_P(FixTensorProductMatrix, TvmultVectorized)
{
  using Fixture      = FixTensorProductMatrix<TypeParam>;
  constexpr auto dim = Fixture::dim;
  using Number       = typename Fixture::Number;
  using value_type   = VectorizedArray<typename Fixture::Number>;
  std::ofstream ofs;
  ofs.open("tensor_product_matrix.log", std::ios_base::app);
  ConditionalOStream pcout(ofs, true);

  const unsigned int m = 4;

  /// random
  std::vector<std::array<Table<2, value_type>, dim>> tensors(2);
  for(auto & tensor : tensors)
    for(auto & mat : tensor)
      fill_matrix_with_random_values(mat, m, m);

  for(unsigned lane = 0; lane < get_macro_size<value_type>(); ++lane)
  {
    Tensors::TensorProductMatrix<dim, value_type> tp_matrix(tensors);
    const auto         reference = table_to_fullmatrix(tp_matrix.as_table(), lane);
    FullMatrix<Number> reference_transposed(reference.m(), reference.n());
    reference_transposed.copy_transposed(reference);
    const auto matrix_transposed = table_to_fullmatrix(tp_matrix.as_transpose_table(), lane);
    Util::compare_matrix(matrix_transposed, reference_transposed, pcout);
  }
}

REGISTER_TYPED_TEST_SUITE_P(FixTensorProductMatrix,
                            Copy,
                            VmultAndMatrix,
                            VmultAndMatrixVectorized,
                            ApplyInverse,
                            ApplyInverseVectorized,
                            TvmultVectorized);

using ParamsTwoDimensionsDouble = testing::Types<Util::TypeList<Util::NonTypeParams<2>, double>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensionsDouble,
                               FixTensorProductMatrix,
                               ParamsTwoDimensionsDouble);



struct Tester
{
  Tester()
  {
    ofs.open("tensor_product_matrix.log", std::ios_base::app);
    pcout = std::make_shared<ConditionalOStream>(ofs, true);
  }

  ~Tester()
  {
    ofs.close();
  }

  template<typename Number>
  void
  compare_matrix(const FullMatrix<Number> & matrix, const FullMatrix<Number> & other) const
  {
    Util::compare_matrix(matrix, other, *pcout);
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
};

TEST(Order2, vmult_id_times_id)
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;
  std::array<Table<2, double>, order>         rank1_tensor;
  const auto                                  n_rows_1d = 3U;

  for(auto & matrix : rank1_tensor)
  {
    matrix.reinit(n_rows_1d, n_rows_1d);
    for(auto i = 0U; i < n_rows_1d; ++i)
      matrix(i, i) = 1.;
  }

  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor);
  tp_matrix.reinit(tensors);
  /// as_table() uses vmult() for each unit vector to construct the kronecker
  /// product matrix
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  FullMatrix<double> reference_matrix(tp_matrix.m());
  for(auto i = 0U; i < reference_matrix.m(); ++i)
    reference_matrix(i, i) = 1.;

  tester.compare_matrix(matrix, reference_matrix);
}

TEST(Order2, vmult_id_times_2x2)
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;

  Table<2, double> id;
  id.reinit(2U, 2U);
  for(auto i = 0U; i < 2U; ++i)
    id(i, i) = 1.;

  Table<2, double> M;
  M.reinit(2U, 2U);
  /// first column
  M(0, 0) = 1.;
  M(1, 0) = 2.;
  /// second column
  M(0, 1) = 3.;
  M(1, 1) = 4.;

  std::array<Table<2, double>, order>              rank1_tensor{M, id};
  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor);
  tp_matrix.reinit(tensors);
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  FullMatrix<double> reference_matrix(tp_matrix.m());
  const auto         fill_submatrix = [&](const auto & block, const auto offset) {
    for(auto i = 0U; i < block.m(); ++i)
      for(auto j = 0U; j < block.n(); ++j)
        reference_matrix(offset + i, offset + j) = block(i, j);
  };
  fill_submatrix(table_to_fullmatrix(M), 0U);
  fill_submatrix(table_to_fullmatrix(M), 2U);

  tester.compare_matrix(matrix, reference_matrix);
}

TEST(Order2, vmult_id_times_2x3)
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;

  Table<2, double> id;
  id.reinit(2U, 3U);
  for(auto i = 0U; i < 2U; ++i)
    id(i, i) = 1.;

  Table<2, double> M;
  M.reinit(2U, 3U);
  /// first column
  M(0, 0) = 1.;
  M(1, 0) = 2.;
  /// second column
  M(0, 1) = 3.;
  M(1, 1) = 4.;
  /// third column
  M(0, 2) = 5.;
  M(1, 2) = 6.;

  std::array<Table<2, double>, order>              rank1_tensor{M, id};
  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor);
  tp_matrix.reinit(tensors);
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  FullMatrix<double> reference_matrix(tp_matrix.m(), tp_matrix.n());
  const auto         fill_submatrix =
    [&](const auto & block, const auto offset_row, const auto offset_column) {
      for(auto i = 0U; i < block.m(); ++i)
        for(auto j = 0U; j < block.n(); ++j)
          reference_matrix(offset_row + i, offset_column + j) = block(i, j);
    };
  fill_submatrix(table_to_fullmatrix(M), 0U, 0U);
  fill_submatrix(table_to_fullmatrix(M), 2U, 3U);

  tester.compare_matrix(matrix, reference_matrix);
}

TEST(Order2, vmult_2x3_times_2x3)
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;

  Table<2, double> M;
  M.reinit(2U, 3U);
  /// first column
  M(0, 0) = 1.;
  M(1, 0) = 2.;
  /// second column
  M(0, 1) = 3.;
  M(1, 1) = 4.;
  /// third column
  M(0, 2) = 5.;
  M(1, 2) = 6.;

  std::array<Table<2, double>, order>              rank1_tensor{M, M};
  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor);
  tp_matrix.reinit(tensors);
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  const auto reference_matrix = table_to_fullmatrix(Tensors::kronecker_product(M, M));

  tester.compare_matrix(matrix, reference_matrix);
}

TEST(Order2, vmult_random_sum_of_2x3_times_2x3)
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;

  Table<2, double> M;
  M.reinit(2U, 3U);
  fill_with_random_values(M);
  Table<2, double> N;
  N.reinit(2U, 3U);
  fill_with_random_values(N);

  std::array<Table<2, double>, order>              rank1_tensor_0{M, N};
  std::array<Table<2, double>, order>              rank1_tensor_1{N, M};
  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor_0);
  tensors.emplace_back(rank1_tensor_1);
  tp_matrix.reinit(tensors);
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  FullMatrix<double> reference_matrix(matrix.m(), matrix.n());
  reference_matrix.add(1.,
                       table_to_fullmatrix(Tensors::kronecker_product(M, N)),
                       1.,
                       table_to_fullmatrix(Tensors::kronecker_product(N, M)));

  tester.compare_matrix(matrix, reference_matrix);
}

TEST(Order2, vmult_id3x1_times_2x3)
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;

  Table<2, double> id;
  id.reinit(3U, 1U);
  id(0U, 0U) = 1.;

  Table<2, double> M;
  M.reinit(2U, 3U);
  /// first column
  M(0, 0) = 1.;
  M(1, 0) = 2.;
  /// second column
  M(0, 1) = 3.;
  M(1, 1) = 4.;
  /// third column
  M(0, 2) = 5.;
  M(1, 2) = 6.;

  std::array<Table<2, double>, order>              rank1_tensor{M, id};
  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor);
  tp_matrix.reinit(tensors);
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  FullMatrix<double> reference_matrix(tp_matrix.m(), tp_matrix.n());
  const auto         fill_submatrix =
    [&](const auto & block, const auto offset_row, const auto offset_column) {
      for(auto i = 0U; i < block.m(); ++i)
        for(auto j = 0U; j < block.n(); ++j)
          reference_matrix(offset_row + i, offset_column + j) = block(i, j);
    };
  fill_submatrix(table_to_fullmatrix(M), 0U, 0U);

  tester.compare_matrix(matrix, reference_matrix);
}

template<bool random = false, bool rank1 = false, bool transpose = false>
void
vmult_random_anisotropic()
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, double> tp_matrix;

  Table<2, double> M_0, M_1;
  for(auto M : {&M_0, &M_1})
  {
    M->reinit(2U, 3U);
    if(random)
      fill_with_random_values(*M);
    else
    {
      /// first column
      (*M)(0, 0) = 1.;
      (*M)(1, 0) = 2.;
      /// second column
      (*M)(0, 1) = 3.;
      (*M)(1, 1) = 4.;
      /// third column
      (*M)(0, 2) = 5.;
      (*M)(1, 2) = 6.;
    }
  }

  Table<2, double> N_0, N_1;
  for(auto N : {&N_0, &N_1})
  {
    N->reinit(4U, 2U);
    if(random)
      fill_with_random_values(*N);
    else
      std::fill(&((*N)(0, 0)), &((*N)(0, 0)) + N->n_elements(), 1.);
  }

  std::array<Table<2, double>, order>              rank1_tensor_0{M_0, N_0};
  std::array<Table<2, double>, order>              rank1_tensor_1{M_1, N_1};
  std::vector<std::array<Table<2, double>, order>> tensors;
  tensors.emplace_back(rank1_tensor_0);
  if(!rank1)
    tensors.emplace_back(rank1_tensor_1);
  tp_matrix.reinit(tensors);
  const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

  /// Note that Tensors::kronecker_product() does not follow (deal.II)
  /// convention in the sense of the zeroth matrix is the rightmost factor in
  /// the Kronecker product and so on. Here, kronecker_product(N, M) actually
  /// computes N (x) M.
  FullMatrix<double> reference_matrix(matrix.m(), matrix.n());
  reference_matrix.add(1.,
                       table_to_fullmatrix(Tensors::kronecker_product(N_0, M_0)),
                       rank1 ? 0. : 1.,
                       table_to_fullmatrix(Tensors::kronecker_product(N_1, M_1)));

  if(transpose)
  {
    const auto         matrix = table_to_fullmatrix(tp_matrix.as_transpose_table());
    FullMatrix<double> transpose_reference(matrix.m(), matrix.n());
    transpose_reference.copy_transposed(reference_matrix);
    tester.compare_matrix(matrix, transpose_reference);
  }
  else
    tester.compare_matrix(matrix, reference_matrix);
}

TEST(Order2, vmult_one4x2_times_2x3)
{
  vmult_random_anisotropic<false, true, false>();
}

TEST(Order2, vmult_sum_of_one4x2_times_2x3)
{
  vmult_random_anisotropic<false, false, false>();
}

TEST(Order2, vmult_of_4x2_times_2x3)
{
  vmult_random_anisotropic<true, true, false>();
}

TEST(Order2, vmult_sum_of_4x2_times_2x3)
{
  vmult_random_anisotropic<true, false, false>();
}

TEST(Order2, Tvmult_one4x2_times_2x3)
{
  vmult_random_anisotropic<false, true, true>();
}

TEST(Order2, Tvmult_sum_of_one4x2_times_2x3)
{
  vmult_random_anisotropic<false, false, true>();
}

TEST(Order2, Tvmult_of_4x2_times_2x3)
{
  vmult_random_anisotropic<true, true, true>();
}

TEST(Order2, Tvmult_sum_of_4x2_times_2x3)
{
  vmult_random_anisotropic<true, false, true>();
}

int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("tensor_product_matrix.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
