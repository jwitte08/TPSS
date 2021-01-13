
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
#include "solvers_and_preconditioners/TPSS/vectorization.h"

#include "test_utilities.h"



using namespace dealii;

// template<int dim, typename Number, int n_rows_1d = -1>
// struct TestTensorProductMatrix
// {
//   using TPM               = typename Tensors::TensorProductMatrix<dim, Number, n_rows_1d>;
//   using scalar_value_type = typename ExtractScalarType<Number>::type;

//   TestTensorProductMatrix(ConditionalOStream & pcout_in) : pcout(pcout_in){};

//   void
//   compare_vmult(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
//   {
//     TPM tp_matrix;
//     tp_matrix.reinit(elementary_tensors, state);
//     AlignedVector<Number> dst(tp_matrix.n());
//     fill_with_random_values(dst);
//     const auto src(dst);

//     const auto src_view = make_array_view(src.begin(), src.end());
//     const auto dst_view = make_array_view(dst.begin(), dst.end());
//     tp_matrix.vmult(dst_view, src_view);

//     const auto src_other = array_view_to_vector(src_view, lane);
//     auto       dst_other(src_other);
//     other.vmult(dst_other, src_other);

//     const auto dst_cview = make_array_view<const Number>(dst.begin(), dst.end());
//     auto       diff(array_view_to_vector(dst_cview, lane));
//     diff -= dst_other;
//     // TODO write PredicateFormatter (see googletest's advanced.md on github) to
//     // compare against zero
//     EXPECT_PRED_FORMAT2(testing::FloatLE,
//                         diff.l2_norm(),
//                         std::numeric_limits<scalar_value_type>::epsilon() *
//                           std::max(10., dst_other.l2_norm()));
//   }

//   void
//   compare_matrix(const FullMatrix<scalar_value_type> & other, const unsigned int lane = 0)
//   {
//     TPM tp_matrix;
//     tp_matrix.reinit(elementary_tensors, state);
//     auto tp_matrix_full = table_to_fullmatrix(tp_matrix.as_table(), lane);
//     Util::compare_matrix(tp_matrix_full, other, pcout);
//   }

//   void
//   compare_inverse_matrix(const FullMatrix<scalar_value_type> & other, const unsigned int lane =
//   0)
//   {
//     TPM tp_matrix;
//     tp_matrix.reinit(elementary_tensors, state);
//     const auto inverse_matrix = table_to_fullmatrix(tp_matrix.as_inverse_table(), lane);
//     Util::compare_inverse_matrix(inverse_matrix, other, pcout);
//   }

//   void
//   compare_copy()
//   {
//     TPM tp_matrix;
//     tp_matrix.reinit(elementary_tensors, state);
//     TPM tp_matrix_copy;
//     tp_matrix_copy        = tp_matrix;
//     const auto table      = tp_matrix.as_table();
//     const auto table_copy = tp_matrix_copy.as_table();
//     for(auto i = 0U; i < table.n_rows(); ++i)
//       for(auto j = 0U; j < table.n_cols(); ++j)
//         EXPECT_NEAR(table(i, j),
//                     table_copy(i, j),
//                     std::numeric_limits<scalar_value_type>::epsilon());
//   }

//   ConditionalOStream &                           pcout;
//   std::vector<std::array<Table<2, Number>, dim>> elementary_tensors;
//   typename TPM::State                            state = TPM::State::invalid;
// };

// template<typename T>
// class FixTensorProductMatrix : public testing::Test
// {
// protected:
//   enum class TestVariant
//   {
//     vmult,
//     apply_inverse,
//     copy
//   };

//   static constexpr int dim                 = T::template type<0>::template value<0>();
//   using Number                             = typename T::template type<1>;
//   using Tester                             = TestTensorProductMatrix<dim, Number>;
//   using TesterV                            = TestTensorProductMatrix<dim,
//   VectorizedArray<Number>>; static constexpr unsigned int macro_size =
//   VectorizedArray<Number>::size();

//   void
//   SetUp() override
//   {
//     ofs.open("tensor_product_matrix.log", std::ios_base::app);
//     pcout_owned = std::make_shared<ConditionalOStream>(ofs, true);
//   }

//   void
//   TearDown() override
//   {
//     ofs.close();
//   }

//   FullMatrix<Number>
//   assemble_reference(const std::vector<FullMatrix<Number>> & left,
//                      const std::vector<FullMatrix<Number>> & right)
//   {
//     std::vector<FullMatrix<Number>> products;
//     std::transform(left.cbegin(),
//                    left.cend(),
//                    right.cbegin(),
//                    std::back_inserter(products),
//                    [](const auto & l, const auto & r) { return Tensors::kronecker_product(l, r);
//                    });
//     FullMatrix<Number> sum_of_products(products.front().m());
//     sum_of_products = 0.;
//     for(const auto & mat : products)
//       sum_of_products = LinAlg::sum(sum_of_products, mat);
//     return sum_of_products;
//   }

//   void
//   test_vmult_or_apply_inverse(const TestVariant test_variant = TestVariant::vmult)
//   {
//     using State          = typename Tester::TPM::State;
//     const unsigned int m = 3;

//     const auto & test_impl = [test_variant, this](const auto & left, const auto & right) {
//       Tester test(*pcout_owned);
//       std::transform(left.cbegin(),
//                      left.cend(),
//                      right.cbegin(),
//                      std::back_inserter(test.elementary_tensors),
//                      [](const auto & l, const auto & r) {
//                        std::array<Table<2, Number>, dim> tensor = {r, l};
//                        return tensor;
//                      });

//       auto sum_of_products = assemble_reference(left, right);
//       test.state           = State::basic;
//       if(test_variant == TestVariant::vmult)
//       {
//         *pcout_owned << "compare tensor product matrix:\n";
//         test.compare_vmult(sum_of_products);
//         test.compare_matrix(sum_of_products);
//       }
//       else if(test_variant == TestVariant::apply_inverse)
//       {
//         *pcout_owned << "compare inverse tensor product matrix:\n";
//         test.compare_inverse_matrix(sum_of_products);
//       }
//       else if(test_variant == TestVariant::copy)
//       {
//         *pcout_owned << "test copy-assign:\n";
//         test.compare_copy();
//       }
//     };

//     /// identity x identity
//     {
//       std::vector<FullMatrix<Number>> left(1);
//       std::vector<FullMatrix<Number>> right(1);
//       left.front()  = IdentityMatrix(m);
//       right.front() = left.front();
//       test_impl(left, right);
//     }
//     /// random
//     {
//       std::vector<FullMatrix<Number>> left(dim);
//       std::vector<FullMatrix<Number>> right(dim);
//       for(auto & mat : left)
//         fill_matrix_with_random_values(mat, m, m);
//       for(auto & mat : right)
//         fill_matrix_with_random_values(mat, m, m);
//       test_impl(left, right);
//     }
//   }

//   void
//   test_vmult_or_apply_inverseV(const TestVariant test_variant = TestVariant::vmult)
//   {
//     using State          = typename TesterV::TPM::State;
//     const unsigned int m = 3;

//     const auto & compare_per_lane = [test_variant, this](const auto & left, const auto & right) {
//       for(unsigned lane = 0; lane < macro_size; ++lane)
//       {
//         TesterV test(*pcout_owned);
//         std::transform(left.cbegin(),
//                        left.cend(),
//                        right.cbegin(),
//                        std::back_inserter(test.elementary_tensors),
//                        [](const auto & l, const auto & r) {
//                          std::array<Table<2, VectorizedArray<Number>>, dim> tensor = {r, l};
//                          return tensor;
//                        });
//         std::vector<FullMatrix<Number>> left_full;
//         std::vector<FullMatrix<Number>> right_full;
//         std::transform(left.cbegin(),
//                        left.cend(),
//                        std::back_inserter(left_full),
//                        [lane](const auto & table) { return table_to_fullmatrix(table, lane); });
//         std::transform(right.cbegin(),
//                        right.cend(),
//                        std::back_inserter(right_full),
//                        [lane](const auto & table) { return table_to_fullmatrix(table, lane); });
//         const auto sum_of_products = assemble_reference(left_full, right_full);
//         test.state                 = State::basic;
//         if(test_variant == TestVariant::vmult)
//         {
//           *pcout_owned << "compare tensor product matrix @ lane" << lane << ":\n";
//           test.compare_vmult(sum_of_products, lane);
//           test.compare_matrix(sum_of_products, lane);
//         }
//         else if(test_variant == TestVariant::apply_inverse)
//         {
//           *pcout_owned << "compare inverse tensor product matrix @ lane" << lane << ":\n";
//           test.compare_inverse_matrix(sum_of_products, lane);
//         }
//       }
//     };

//     /// random-scaled identity
//     {
//       std::vector<Table<2, VectorizedArray<Number>>> left(1);
//       std::vector<Table<2, VectorizedArray<Number>>> right(1);
//       auto &                                         l = left.front();
//       l.reinit(m, m);
//       l.fill(static_cast<VectorizedArray<Number>>(0.));
//       const auto value = make_random_value<VectorizedArray<Number>>();
//       for(unsigned int mm = 0; mm < m; ++mm)
//         l(mm, mm) = value;
//       right.front() = l;
//       compare_per_lane(left, right);
//     }
//     /// random
//     {
//       std::vector<Table<2, VectorizedArray<Number>>> left(dim);
//       std::vector<Table<2, VectorizedArray<Number>>> right(dim);
//       for(auto & mat : left)
//         fill_matrix_with_random_values(mat, m, m);
//       for(auto & mat : right)
//         fill_matrix_with_random_values(mat, m, m);
//       compare_per_lane(left, right);
//     }
//   }

//   std::ofstream                       ofs;
//   std::shared_ptr<ConditionalOStream> pcout_owned;
// };

// TYPED_TEST_SUITE_P(FixTensorProductMatrix);
// TYPED_TEST_P(FixTensorProductMatrix, VmultAndMatrix)
// {
//   using Fixture = FixTensorProductMatrix<TypeParam>;
//   Fixture::test_vmult_or_apply_inverse();
// }

// TYPED_TEST_P(FixTensorProductMatrix, ApplyInverse)
// {
//   using Fixture     = FixTensorProductMatrix<TypeParam>;
//   using TestVariant = typename Fixture::TestVariant;
//   Fixture::test_vmult_or_apply_inverse(TestVariant::apply_inverse);
// }

// TYPED_TEST_P(FixTensorProductMatrix, Copy)
// {
//   using Fixture     = FixTensorProductMatrix<TypeParam>;
//   using TestVariant = typename Fixture::TestVariant;
//   Fixture::test_vmult_or_apply_inverse(TestVariant::copy);
// }

// TYPED_TEST_P(FixTensorProductMatrix, VmultAndMatrixVectorized)
// {
//   using Fixture = FixTensorProductMatrix<TypeParam>;
//   Fixture::test_vmult_or_apply_inverseV();
// }

// TYPED_TEST_P(FixTensorProductMatrix, ApplyInverseVectorized)
// {
//   using Fixture     = FixTensorProductMatrix<TypeParam>;
//   using TestVariant = typename Fixture::TestVariant;
//   Fixture::test_vmult_or_apply_inverseV(TestVariant::apply_inverse);
// }

// TYPED_TEST_P(FixTensorProductMatrix, TvmultVectorized)
// {
//   using Fixture      = FixTensorProductMatrix<TypeParam>;
//   constexpr auto dim = Fixture::dim;
//   using Number       = typename Fixture::Number;
//   using value_type   = VectorizedArray<typename Fixture::Number>;
//   std::ofstream ofs;
//   ofs.open("tensor_product_matrix.log", std::ios_base::app);
//   ConditionalOStream pcout(ofs, true);

//   const unsigned int m = 4;

//   /// random
//   std::vector<std::array<Table<2, value_type>, dim>> tensors(2);
//   for(auto & tensor : tensors)
//     for(auto & mat : tensor)
//       fill_matrix_with_random_values(mat, m, m);

//   for(unsigned lane = 0; lane < get_macro_size<value_type>(); ++lane)
//   {
//     Tensors::TensorProductMatrix<dim, value_type> tp_matrix(tensors);
//     const auto         reference = table_to_fullmatrix(tp_matrix.as_table(), lane);
//     FullMatrix<Number> reference_transposed(reference.m(), reference.n());
//     reference_transposed.copy_transposed(reference);
//     const auto matrix_transposed = table_to_fullmatrix(tp_matrix.as_transpose_table(), lane);
//     Util::compare_matrix(matrix_transposed, reference_transposed, pcout);
//   }
// }

// REGISTER_TYPED_TEST_SUITE_P(FixTensorProductMatrix,
//                             Copy,
//                             VmultAndMatrix,
//                             VmultAndMatrixVectorized,
//                             ApplyInverse,
//                             ApplyInverseVectorized,
//                             TvmultVectorized);

// using ParamsTwoDimensionsDouble = testing::Types<Util::TypeList<Util::NonTypeParams<2>, double>>;
// INSTANTIATE_TYPED_TEST_SUITE_P(TwoDimensionsDouble,
//                                FixTensorProductMatrix,
//                                ParamsTwoDimensionsDouble);



struct Tester
{
  enum class Method
  {
    reinit,
    get_eigenvalues,
    apply_inverse
  };

  struct Features
  {
    Method method_variant = Method::reinit;
  };

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

  template<typename Number>
  void
  compare_inverse_matrix(const FullMatrix<Number> & inverse_matrix,
                         const FullMatrix<Number> & matrix) const
  {
    Util::compare_inverse_matrix(inverse_matrix, matrix, *pcout);
  }

  template<typename Number>
  void
  print_matrix(const FullMatrix<Number> & matrix, const std::string & description) const
  {
    Util::print_matrix(matrix, description, *pcout);
  }

  operator ConditionalOStream &()
  {
    return *pcout;
  };

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
};



TEST(IsotropicSquare_2Double, basic_vmult)
{
  /// basic_vmult_id_times_id
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

  /// basic_vmult_id_times_2x2
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
}



template<bool random = false, bool rank1 = false, bool transpose = false, typename Number = double>
void
vmult_random_anisotropic()
{
  Tester                                      tester;
  constexpr int                               order = 2;
  Tensors::TensorProductMatrix<order, Number> tp_matrix;
  using scalar_value_type = typename Tensors::TensorProductMatrix<order, Number>::scalar_value_type;

  Table<2, Number> M_0, M_1;
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

  Table<2, Number> N_0, N_1;
  for(auto N : {&N_0, &N_1})
  {
    N->reinit(4U, 2U);
    if(random)
      fill_with_random_values(*N);
    else
      N->fill(static_cast<Number>(1.));
  }

  std::array<Table<2, Number>, order>              rank1_tensor_0{M_0, N_0};
  std::array<Table<2, Number>, order>              rank1_tensor_1{M_1, N_1};
  std::vector<std::array<Table<2, Number>, order>> tensors;
  tensors.emplace_back(rank1_tensor_0);
  if(!rank1)
    tensors.emplace_back(rank1_tensor_1);
  tp_matrix.reinit(tensors);

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto matrix = table_to_fullmatrix(tp_matrix.as_table(), lane);

    /// Note that Tensors::kronecker_product() does not follow (deal.II)
    /// convention in the sense of the zeroth matrix is the rightmost factor in
    /// the Kronecker product and so on. Here, kronecker_product(N, M) actually
    /// computes N (x) M.
    FullMatrix<scalar_value_type> reference_matrix(matrix.m(), matrix.n());
    reference_matrix.add(1.,
                         table_to_fullmatrix(Tensors::kronecker_product(N_0, M_0), lane),
                         rank1 ? 0. : 1.,
                         table_to_fullmatrix(Tensors::kronecker_product(N_1, M_1), lane));

    if(transpose)
    {
      const auto         matrix = table_to_fullmatrix(tp_matrix.as_transpose_table(), lane);
      FullMatrix<double> transpose_reference(matrix.m(), matrix.n());
      transpose_reference.copy_transposed(reference_matrix);
      tester.compare_matrix(matrix, transpose_reference);
    }
    else
      tester.compare_matrix(matrix, reference_matrix);
  }
}



TEST(Anisotropic_2Double, basic_vmult)
{
  /// basic_vmult_id_times_2x3)
  {
    Tester                                      tester;
    constexpr int                               order = 2;
    Tensors::TensorProductMatrix<order, double> tp_matrix;

    Table<2, double> id;
    id.reinit(2U, 2U);
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

  /// basic_vmult_2x3_times_2x3)
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

  /// basic_vmult_random_sum_of_2x3_times_2x3)
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

  /// basic_vmult_id3x1_times_2x3)
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

  /// basic_vmult_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, true, false>();
  }

  /// basic_vmult_sum_of_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, false, false>();
  }

  /// basic_vmult_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, true, false>();
  }

  /// basic_vmult_sum_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, false, false>();
  }
}



TEST(Anisotropic_2VectorizedArrayDouble, basic_vmult)
{
  /// basic_vmult_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, true, false, VectorizedArray<double>>();
  }

  /// basic_vmult_sum_of_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, false, false, VectorizedArray<double>>();
  }

  /// basic_vmult_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, true, false, VectorizedArray<double>>();
  }

  /// basic_vmult_sum_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, false, false, VectorizedArray<double>>();
  }
}



TEST(Anisotropic_2Double, basic_Tvmult)
{
  /// basic_Tvmult_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, true, true>();
  }

  /// basic_Tvmult_sum_of_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, false, true>();
  }

  /// basic_Tvmult_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, true, true>();
  }

  /// basic_Tvmult_sum_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, false, true>();
  }
}



TEST(Anisotropic_2VectorizedArrayDouble, basic_Tvmult)
{
  /// basic_Tvmult_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, true, true, VectorizedArray<double>>();
  }

  /// basic_Tvmult_sum_of_one4x2_times_2x3)
  {
    vmult_random_anisotropic<false, false, true, VectorizedArray<double>>();
  }

  /// basic_Tvmult_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, true, true, VectorizedArray<double>>();
  }

  /// basic_Tvmult_sum_of_4x2_times_2x3)
  {
    vmult_random_anisotropic<true, false, true, VectorizedArray<double>>();
  }
}



template<typename Number>
Table<2, Number>
make_random_matrix(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> rndm_matrix;
  fill_matrix_with_random_values(rndm_matrix, n_rows, n_cols);
  return rndm_matrix;
}



template<typename Number>
Table<2, Number>
make_random_matrix_symm(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> rndm_matrix;
  fill_matrix_with_random_values(rndm_matrix, n_rows, n_cols);

  /// symmetrize
  Table<2, Number> matrix;
  matrix.reinit(n_rows, n_cols);
  for(auto i = 0U; i < n_rows; ++i)
    for(auto j = 0U; j < n_cols; ++j)
      matrix(i, j) = (rndm_matrix(i, j) + rndm_matrix(j, i)) / 2.;
  return matrix;
}



template<typename Number>
Table<2, Number>
make_random_matrix_spd(const unsigned int n_rows, const unsigned int n_cols)
{
  /// symmetric
  Table<2, Number> matrix = make_random_matrix_symm<Number>(n_rows, n_cols);
  /// positive definite?
  for(auto i = 0U; i < std::min(n_rows, n_cols); ++i)
    matrix(i, i) += static_cast<Number>(std::max(n_rows, n_cols));
  return matrix;
}



template<typename Number>
Table<2, Number>
make_one_matrix(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> matrix;
  matrix.reinit(n_rows, n_cols);
  matrix.fill(static_cast<Number>(1.));
  return matrix;
}

template<typename Number>
Table<2, Number>
make_identity_matrix(const unsigned int n_rows, const unsigned int n_cols)
{
  Table<2, Number> matrix;
  matrix.reinit(n_rows, n_cols);
  for(auto i = 0U; i < std::min(n_rows, n_cols); ++i)
    matrix(i, i) = static_cast<Number>(1.);
  return matrix;
}



template<typename Number>
void
check_basic_vmult(
  const std::vector<typename Tensors::TensorProductMatrix<3, Number>::tensor_type> & tensors)
{
  Tester tester;

  /// compute reference matrix (Tensors::kronecker_product(A, B) computes A(x)B)
  const auto & [M0, M1, M2] = tensors.front();
  const auto & M2_x_M1_x_M0 = Tensors::kronecker_product(M2, Tensors::kronecker_product(M1, M0));
  auto         reference_matrix = M2_x_M1_x_M0;
  for(auto r = 1U; r < tensors.size(); ++r)
  {
    const auto & [M0, M1, M2] = tensors.at(r);
    const auto & M2_x_M1_x_M0 = Tensors::kronecker_product(M2, Tensors::kronecker_product(M1, M0));
    reference_matrix          = LinAlg::sum(reference_matrix, M2_x_M1_x_M0);
  }

  Tensors::TensorProductMatrix<3, Number> tpm_basic(tensors);

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto & fullmatrix           = table_to_fullmatrix(tpm_basic.as_table(), lane);
    const auto & reference_fullmatrix = table_to_fullmatrix(reference_matrix, lane);
    tester.compare_matrix(fullmatrix, reference_fullmatrix);
  }
}

template<typename Number>
void
test_basic_vmult_3D()
{
  constexpr int order = 3;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  const unsigned int m = 3;
  const unsigned int n = m;

  const auto & Id  = make_identity_matrix<Number>(m, n);
  const auto & One = make_one_matrix<Number>(m, n);
  const auto & A   = make_random_matrix_symm<Number>(m, n);
  const auto & AA  = make_random_matrix_symm<Number>(m, n);
  const auto & B   = make_random_matrix_spd<Number>(m, n);
  const auto & BB  = make_random_matrix_spd<Number>(m, n);

  /// rank-1 test cases
  {
    std::vector<tensor_type> rank1_tensors = {{Id, Id, Id}};
    check_basic_vmult<Number>(rank1_tensors);
  }
  {
    std::vector<tensor_type> rank1_tensors = {{Id, One, Id}};
    check_basic_vmult<Number>(rank1_tensors);
  }
  {
    std::vector<tensor_type> rank1_tensors = {{Id, One, A}};
    check_basic_vmult<Number>(rank1_tensors);
  }
  {
    std::vector<tensor_type> rank1_tensors = {{A, AA, A}};
    check_basic_vmult<Number>(rank1_tensors);
  }

  /// rank-2 test cases
  {
    std::vector<tensor_type> rank1_tensors = {{Id, Id, Id}, {Id, Id, Id}};
    check_basic_vmult<Number>(rank1_tensors);
  }
  {
    std::vector<tensor_type> rank1_tensors = {{A, AA, A}, {B, BB, One}};
    check_basic_vmult<Number>(rank1_tensors);
  }

  /// rank-3 test cases
  {
    std::vector<tensor_type> rank1_tensors = {{A, AA, A}, {Id, Id, Id}, {B, BB, One}};
    check_basic_vmult<Number>(rank1_tensors);
  }
}

TEST(IsotropicSquare_3Double, basic_vmult)
{
  /// basic_vmult_id_times_id_times_id)
  {
    Tester        tester;
    constexpr int order     = 3;
    const auto    n_rows_1d = 3U;

    std::array<Table<2, double>, order> rank1_tensor;
    for(auto & matrix : rank1_tensor)
      matrix = make_identity_matrix<double>(n_rows_1d, n_rows_1d);

    Tensors::TensorProductMatrix<order, double> tp_matrix(rank1_tensor);

    /// as_table() uses vmult() for each unit vector to construct the kronecker
    /// product matrix
    const auto matrix = table_to_fullmatrix(tp_matrix.as_table());

    FullMatrix<double> reference_matrix(tp_matrix.m());
    for(auto i = 0U; i < reference_matrix.m(); ++i)
      reference_matrix(i, i) = 1.;

    tester.compare_matrix(matrix, reference_matrix);
  }

  /// various tests
  test_basic_vmult_3D<double>();
}

TEST(IsotropicSquare_3VectorizedArrayDouble, basic_vmult)
{
  test_basic_vmult_3D<VectorizedArray<double>>();
}



/// This test assumes previous tests for vmult() in basic state do not fail.
template<int order, typename Number>
void
check_basic_apply_inverse(
  const std::vector<typename Tensors::TensorProductMatrix<order, Number>::tensor_type> & tensors)
{
  Tester tester;

  Tensors::TensorProductMatrix<order, Number> tpm_basic(tensors);

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto & fullinverse          = table_to_fullmatrix(tpm_basic.as_inverse_table(), lane);
    const auto & reference_fullmatrix = table_to_fullmatrix(tpm_basic.as_table(), lane);
    tester.compare_inverse_matrix(fullinverse, reference_fullmatrix);
  }
}

template<typename Number>
void
test_basic_apply_inverse()
{
  constexpr int order = 2;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  const unsigned int m = 3;
  const unsigned int n = m;

  const auto & Id = make_identity_matrix<Number>(m, n);
  const auto & A  = make_random_matrix<Number>(m, n);
  const auto & AA = make_random_matrix<Number>(m, n);
  const auto & B  = make_random_matrix<Number>(m, n);
  const auto & BB = make_random_matrix<Number>(m, n);

  /// rank-1 test cases
  {
    std::vector<tensor_type> rank1_tensors = {{Id, Id}};
    check_basic_apply_inverse<order, Number>(rank1_tensors);
  }
  {
    std::vector<tensor_type> rank1_tensors = {{A, AA}};
    check_basic_apply_inverse<order, Number>(rank1_tensors);
  }

  /// rank-2 test cases
  {
    std::vector<tensor_type> rank1_tensors = {{Id, Id}, {Id, Id}};
    check_basic_apply_inverse<order, Number>(rank1_tensors);
  }
  {
    std::vector<tensor_type> rank1_tensors = {{A, AA}, {B, BB}};
    check_basic_apply_inverse<order, Number>(rank1_tensors);
  }

  /// rank-3 test cases
  {
    std::vector<tensor_type> rank1_tensors = {{A, AA}, {AA, B}, {B, BB}};
    check_basic_apply_inverse<order, Number>(rank1_tensors);
  }
}

TEST(IsotropicSquare_2Double, basic_apply_inverse)
{
  test_basic_apply_inverse<double>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, basic_apply_inverse)
{
  test_basic_apply_inverse<VectorizedArray<double>>();
}

template<typename Number>
void
test_basic_apply_inverse_3D()
{
  constexpr int order = 3;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  for(const unsigned int m : {2U, 3U})
  {
    const auto n = m;

    const auto & Id  = make_identity_matrix<Number>(m, n);
    const auto & A   = make_random_matrix<Number>(m, n);
    const auto & AA  = make_random_matrix<Number>(m, n);
    const auto & AAA = make_random_matrix<Number>(m, n);
    const auto & B   = make_random_matrix<Number>(m, n);
    const auto & BB  = make_random_matrix<Number>(m, n);
    const auto & BBB = make_random_matrix<Number>(m, n);

    /// rank-1 test cases
    {
      std::vector<tensor_type> rank1_tensors = {{Id, Id, Id}};
      check_basic_apply_inverse<order, Number>(rank1_tensors);
    }
    {
      std::vector<tensor_type> rank1_tensors = {{A, AA, AAA}};
      check_basic_apply_inverse<order, Number>(rank1_tensors);
    }

    /// rank-2 test cases
    {
      std::vector<tensor_type> rank1_tensors = {{Id, Id, Id}, {Id, Id, Id}};
      check_basic_apply_inverse<order, Number>(rank1_tensors);
    }
    {
      std::vector<tensor_type> rank1_tensors = {{A, AA, AAA}, {B, BB, BBB}};
      check_basic_apply_inverse<order, Number>(rank1_tensors);
    }

    /// rank-3 test cases
    {
      std::vector<tensor_type> rank1_tensors = {{A, AA, AAA}, {AA, B, A}, {B, BB, BBB}};
      check_basic_apply_inverse<order, Number>(rank1_tensors);
    }
  }
}

TEST(IsotropicSquare_3Double, basic_apply_inverse)
{
  test_basic_apply_inverse_3D<double>();
}

TEST(IsotropicSquare_3VectorizedArrayDouble, basic_apply_inverse)
{
  test_basic_apply_inverse_3D<VectorizedArray<double>>();
}



template<int order, typename Number>
void
check_special_state(
  std::vector<typename Tensors::TensorProductMatrix<order, Number>::tensor_type> tensors,
  typename Tensors::TensorProductMatrix<order, Number>::State                    state,
  std::bitset<order>                                                             spd_mask,
  const Tester::Features &                                                       features)
{
  using tensor_type       = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;
  using State             = typename Tensors::TensorProductMatrix<order, Number>::State;
  using scalar_value_type = typename Tensors::TensorProductMatrix<order, Number>::scalar_value_type;

  Tester tester;

  ASSERT_TRUE(state == State::ranktwo || state == State::separable || state == State::rankone)
    << "Invalid state!";

  typename Tensors::TensorProductMatrix<order, Number>::AdditionalData additional_data;
  additional_data.state    = state;
  additional_data.spd_mask = spd_mask;
  Tensors::TensorProductMatrix<order, Number> tpm_with_state(tensors, additional_data);
  Tensors::TensorProductMatrix<order, Number> tpm_basic;
  if(state == State::separable)
  {
    ASSERT_EQ(tensors.size(), 2U) << "only mass first and derivative second";
    std::vector<tensor_type> expansion(order);
    const auto &             MM = tensors.front();
    const auto &             DD = tensors.back();
    for(auto r = 0U; r < order; ++r)
      for(auto d = 0U; d < order; ++d)
      {
        if(r == d)
          expansion[r][d] = DD[d];
        else
          expansion[r][d] = MM[d];
      }
    tpm_basic.reinit(expansion);
  }
  else
    tpm_basic.reinit(tensors);

  const auto & eigenvalues  = tpm_with_state.get_eigenvalues();
  const auto & eigenvectors = tpm_with_state.get_eigenvectors();

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto & reference_fullmatrix = table_to_fullmatrix(tpm_basic.as_table(), lane);
    const auto & fullQ                = table_to_fullmatrix(eigenvectors, lane);
    const auto & diagonalization = merge_reverse_decomposition(fullQ, reference_fullmatrix, fullQ);
    const auto & lambdas         = alignedvector_to_vector(eigenvalues, lane);

    /// checks if the eigenvectors computed within reinit() actually diagonalize
    /// the underlying matrix
    if(features.method_variant == Tester::Method::reinit ||
       features.method_variant == Tester::Method::get_eigenvalues)
    {
      tester.print_matrix(reference_fullmatrix, "M:");
      tester.print_matrix(fullQ, "eigenvectors (column-wise):");
      *tester.pcout << vector_to_string(compute_eigenvalues(reference_fullmatrix)) << std::endl;
      *tester.pcout << vector_to_string(lambdas) << std::endl;
      tester.print_matrix(diagonalization, "Q^T M Q:");

      /// check off-diagonal entries of diagonalization
      auto D = diagonalization;
      for(auto i = 0U; i < D.m(); ++i)
        D(i, i) = 0.;
      FullMatrix<scalar_value_type> fullzero(D.m());
      tester.compare_matrix(D, fullzero);

      /// checks if the eigenvalues returned by get_eigenvalues() are the same as
      /// those obtained from the eigendecomposition @p diagonalization
      if(features.method_variant == Tester::Method::get_eigenvalues)
      {
        FullMatrix<scalar_value_type> fullLambda(lambdas.size());
        for(auto i = 0U; i < lambdas.size(); ++i)
          fullLambda(i, i) = lambdas[i];
        tester.compare_matrix(fullLambda, diagonalization);
      }
    }

    else if(features.method_variant == Tester::Method::apply_inverse)
    {
      const auto fullinverse = table_to_fullmatrix(tpm_with_state.as_inverse_table(), lane);
      tester.compare_inverse_matrix(fullinverse, reference_fullmatrix);
    }

    else
      EXPECT_TRUE(false) << "Test variant isn't supported/implemented.";
  }
}

template<int order, typename Number>
void
check_ranktwo(
  std::vector<typename Tensors::TensorProductMatrix<order, Number>::tensor_type> tensors,
  std::bitset<order>                                                             spd_mask,
  const Tester::Features &                                                       features)
{
  using State = typename Tensors::TensorProductMatrix<order, Number>::State;
  check_special_state<order, Number>(tensors, State::ranktwo, spd_mask, features);
}

template<int order, typename Number>
void
check_separable(
  std::vector<typename Tensors::TensorProductMatrix<order, Number>::tensor_type> tensors,
  const Tester::Features &                                                       features)
{
  using State = typename Tensors::TensorProductMatrix<order, Number>::State;
  check_special_state<order, Number>(tensors, State::separable, std::bitset<order>{}, features);
}

template<int order, typename Number>
void
check_rankone(
  std::vector<typename Tensors::TensorProductMatrix<order, Number>::tensor_type> tensors,
  const Tester::Features &                                                       features)
{
  using State = typename Tensors::TensorProductMatrix<order, Number>::State;
  check_special_state<order, Number>(tensors, State::rankone, std::bitset<order>{}, features);
}



template<typename Number>
void
test_ranktwo_reinit()
{
  constexpr int order = 2;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;

  for(const auto m : {2U, 3U})
  {
    const unsigned int n = m;

    const auto & Id  = make_identity_matrix<Number>(m, n);
    const auto & One = make_one_matrix<Number>(m, n);
    const auto & A   = make_random_matrix_symm<Number>(m, n);
    const auto & AA  = make_random_matrix_symm<Number>(m, n);
    const auto & B   = make_random_matrix_spd<Number>(m, n);
    const auto & BB  = make_random_matrix_spd<Number>(m, n);

    {
      std::vector<tensor_type> tensors = {{Id, Id}, {Id, Id}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B, Id}, {Id, One}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B, BB}, {A, AA}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, AA}, {B, BB}};
      std::bitset<order>       spd_mask("11");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, BB}, {B, AA}};
      std::bitset<order>       spd_mask("01");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }
  }
}

TEST(IsotropicSquare_2Double, ranktwo_reinit)
{
  using Number = double;
  test_ranktwo_reinit<Number>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, ranktwo_reinit)
{
  using Number = VectorizedArray<double>;
  test_ranktwo_reinit<Number>();
}

template<typename Number>
void
test_ranktwo_3D(const Tester::Features & test_features)
{
  constexpr int order = 3;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  for(const auto m : {2U, 3U})
  {
    const unsigned int n = m;

    const auto & Id  = make_identity_matrix<Number>(m, n);
    const auto & A   = make_random_matrix_symm<Number>(m, n);
    const auto & AA  = make_random_matrix_symm<Number>(m, n);
    const auto & AAA = make_random_matrix_symm<Number>(m, n);
    const auto & B   = make_random_matrix_spd<Number>(m, n);
    const auto & BB  = make_random_matrix_spd<Number>(m, n);
    const auto & BBB = make_random_matrix_spd<Number>(m, n);

    {
      std::vector<tensor_type> tensors = {{Id, Id, Id}, {Id, Id, Id}};
      std::bitset<order>       spd_mask("000");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B, BB, BBB}, {A, AA, AAA}};
      std::bitset<order>       spd_mask("000");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, AA, AAA}, {B, BB, BBB}};
      std::bitset<order>       spd_mask("111");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, BB, BBB}, {B, AA, AAA}};
      std::bitset<order>       spd_mask("001");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, BB, AAA}, {B, AA, BBB}};
      std::bitset<order>       spd_mask("101");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }
  }
}

TEST(IsotropicSquare_3Double, ranktwo_reinit)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_ranktwo_3D<Number>(test_features);
}

TEST(IsotropicSquare_3VectorizedArrayDouble, ranktwo_reinit)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_ranktwo_3D<Number>(test_features);
}



template<int order, typename Number>
void
test_ranktwo_get_eigenvalues()
{
  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;

  const unsigned int m = 3;
  const unsigned int n = m;

  const auto & Id  = make_identity_matrix<Number>(m, n);
  const auto & One = make_one_matrix<Number>(m, n);
  const auto & A   = make_random_matrix_symm<Number>(m, n);
  const auto & AA  = make_random_matrix_symm<Number>(m, n);
  const auto & B   = make_random_matrix_spd<Number>(m, n);
  const auto & BB  = make_random_matrix_spd<Number>(m, n);

  {
    std::vector<tensor_type> tensors = {{Id, Id}, {Id, Id}};
    std::bitset<order>       spd_mask("00");
    check_ranktwo<order, Number>(tensors, spd_mask, test_features);
  }

  {
    std::vector<tensor_type> tensors = {{B, BB}, {A, AA}};
    std::bitset<order>       spd_mask("00");
    check_ranktwo<order, Number>(tensors, spd_mask, test_features);
  }

  {
    std::vector<tensor_type> tensors = {{A, AA}, {B, BB}};
    std::bitset<order>       spd_mask("11");
    check_ranktwo<order, Number>(tensors, spd_mask, test_features);
  }

  {
    std::vector<tensor_type> tensors = {{A, BB}, {B, AA}};
    std::bitset<order>       spd_mask("01");
    check_ranktwo<order, Number>(tensors, spd_mask, test_features);
  }

  {
    std::vector<tensor_type> tensors = {{B, AA}, {A, BB}};
    std::bitset<order>       spd_mask("10");
    check_ranktwo<order, Number>(tensors, spd_mask, test_features);
  }
}

TEST(IsotropicSquare_2Double, ranktwo_get_eigenvalues)
{
  constexpr int order = 2;
  using Number        = double;
  test_ranktwo_get_eigenvalues<order, Number>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, ranktwo_get_eigenvalues)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_ranktwo_get_eigenvalues<order, Number>();
}

TEST(IsotropicSquare_3Double, ranktwo_get_eigenvalues)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_ranktwo_3D<Number>(test_features);
}

TEST(IsotropicSquare_3VectorizedArrayDouble, ranktwo_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_ranktwo_3D<Number>(test_features);
}



template<int order, typename Number>
void
test_ranktwo_apply_inverse()
{
  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;

  for(const auto m : {2U, 3U})
  {
    const unsigned int n = m;

    const auto & Id  = make_identity_matrix<Number>(m, n);
    const auto & One = make_one_matrix<Number>(m, n);
    const auto & A   = make_random_matrix_symm<Number>(m, n);
    const auto & AA  = make_random_matrix_symm<Number>(m, n);
    const auto & B   = make_random_matrix_spd<Number>(m, n);
    const auto & BB  = make_random_matrix_spd<Number>(m, n);

    {
      std::vector<tensor_type> tensors = {{Id, Id}, {Id, Id}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id, Id}, {A, AA}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B, BB}, {A, AA}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, AA}, {B, BB}};
      std::bitset<order>       spd_mask("11");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A, BB}, {B, AA}};
      std::bitset<order>       spd_mask("01");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B, AA}, {A, BB}};
      std::bitset<order>       spd_mask("10");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }
  }
}

TEST(IsotropicSquare_2Double, ranktwo_apply_inverse)
{
  constexpr int order = 2;
  using Number        = double;
  test_ranktwo_apply_inverse<order, Number>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, ranktwo_apply_inverse)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_ranktwo_apply_inverse<order, Number>();
}

TEST(IsotropicSquare_3Double, ranktwo_apply_inverse)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_ranktwo_3D<Number>(test_features);
}

TEST(IsotropicSquare_3VectorizedArrayDouble, ranktwo_apply_inverse)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_ranktwo_3D<Number>(test_features);
}



template<typename Number>
void
test_separable(const Tester::Features & test_features)
{
  constexpr int order = 2;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  for(const auto m : {2U, 3U})
  {
    const unsigned int n = m;

    const auto & Id = make_identity_matrix<Number>(m, n);
    const auto & D  = make_random_matrix_symm<Number>(m, n);
    const auto & DD = make_random_matrix_symm<Number>(m, n);
    const auto & M  = make_random_matrix_spd<Number>(m, n);
    const auto & MM = make_random_matrix_spd<Number>(m, n);

    {
      std::vector<tensor_type> tensors = {{Id, Id}, {Id, Id}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id, Id}, {D, DD}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{M, MM}, {D, DD}};
      check_separable<order, Number>(tensors, test_features);
    }
  }
}



template<typename Number>
void
test_separable_3D(const Tester::Features & test_features)
{
  constexpr int order = 3;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  for(const auto m : {2U, 3U})
  {
    const unsigned int n = m;

    const auto & Id  = make_identity_matrix<Number>(m, n);
    const auto & D   = make_random_matrix_symm<Number>(m, n);
    const auto & DD  = make_random_matrix_symm<Number>(m, n);
    const auto & DDD = make_random_matrix_symm<Number>(m, n);
    const auto & M   = make_random_matrix_spd<Number>(m, n);
    const auto & MM  = make_random_matrix_spd<Number>(m, n);
    const auto & MMM = make_random_matrix_spd<Number>(m, n);

    {
      std::vector<tensor_type> tensors = {{Id, Id, Id}, {Id, Id, Id}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id, Id, Id}, {D, DD, DDD}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{M, MM, MMM}, {D, DD, DDD}};
      check_separable<order, Number>(tensors, test_features);
    }
  }
}



template<typename Number>
void
test_separable_reinit()
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_separable<Number>(test_features);
}

TEST(IsotropicSquare_2Double, separable_reinit)
{
  using Number = double;
  test_separable_reinit<Number>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, separable_reinit)
{
  using Number = VectorizedArray<double>;
  test_separable_reinit<Number>();
}

template<typename Number>
void
test_separable_3D_reinit()
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_separable_3D<Number>(test_features);
}

TEST(IsotropicSquare_3Double, separable_reinit)
{
  using Number = double;
  test_separable_3D_reinit<Number>();
}

TEST(IsotropicSquare_3VectorizedArrayDouble, separable_reinit)
{
  using Number = VectorizedArray<double>;
  test_separable_3D_reinit<Number>();
}



template<typename Number>
void
test_separable_get_eigenvalues()
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_separable<Number>(test_features);
}

TEST(IsotropicSquare_2Double, separable_get_eigenvalues)
{
  using Number = double;
  test_separable_get_eigenvalues<Number>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, separable_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  test_separable_get_eigenvalues<Number>();
}

template<typename Number>
void
test_separable_3D_get_eigenvalues()
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_separable_3D<Number>(test_features);
}

TEST(IsotropicSquare_3Double, separable_get_eigenvalues)
{
  using Number = double;
  test_separable_3D_get_eigenvalues<Number>();
}

TEST(IsotropicSquare_3VectorizedArrayDouble, separable_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  test_separable_3D_get_eigenvalues<Number>();
}



template<typename Number>
void
test_separable_apply_inverse()
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_separable<Number>(test_features);
}

TEST(IsotropicSquare_2Double, separable_apply_inverse)
{
  using Number = double;
  test_separable_apply_inverse<Number>();
}

TEST(IsotropicSquare_2VectorizedArrayDouble, separable_apply_inverse)
{
  using Number = VectorizedArray<double>;
  test_separable_apply_inverse<Number>();
}

template<typename Number>
void
test_separable_3D_apply_inverse()
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_separable_3D<Number>(test_features);
}

TEST(IsotropicSquare_3Double, separable_apply_inverse)
{
  using Number = double;
  test_separable_3D_apply_inverse<Number>();
}

TEST(IsotropicSquare_3VectorizedArrayDouble, separable_apply_inverse)
{
  using Number = VectorizedArray<double>;
  test_separable_3D_apply_inverse<Number>();
}



template<typename Number>
void
test_ranktwo_anisotropic(const Tester::Features & test_features)
{
  constexpr int order = 2;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  const auto test_impl = [&](const std::array<unsigned int, order> m) {
    const auto & Id0 = make_identity_matrix<Number>(m[0], m[0]);
    const auto & A0  = make_random_matrix_symm<Number>(m[0], m[0]);
    const auto & B0  = make_random_matrix_spd<Number>(m[0], m[0]);
    const auto & Id1 = make_identity_matrix<Number>(m[1], m[1]);
    const auto & A1  = make_random_matrix_symm<Number>(m[1], m[1]);
    const auto & B1  = make_random_matrix_spd<Number>(m[1], m[1]);

    {
      std::vector<tensor_type> tensors = {{Id0, Id1}, {Id0, Id1}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id0, Id1}, {A0, A1}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B0, B1}, {A0, A1}};
      std::bitset<order>       spd_mask("00");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A0, A1}, {B0, B1}};
      std::bitset<order>       spd_mask("11");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A0, B1}, {B0, A1}};
      std::bitset<order>       spd_mask("01");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B0, A1}, {A0, B1}};
      std::bitset<order>       spd_mask("10");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }
  };

  test_impl({2U, 3U});

  test_impl({3U, 2U});

  test_impl({4U, 2U});

  test_impl({1U, 3U});
}

TEST(AnisotropicSquare_2Double, ranktwo_reinit)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_ranktwo_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, ranktwo_reinit)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_ranktwo_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2Double, ranktwo_get_eigenvalues)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_ranktwo_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, ranktwo_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_ranktwo_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2Double, ranktwo_apply_inverse)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_ranktwo_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, ranktwo_apply_inverse)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_ranktwo_anisotropic<Number>(test_features);
}



template<typename Number>
void
test_ranktwo_3D_anisotropic(const Tester::Features & test_features)
{
  constexpr int order = 3;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  const auto test_impl = [&](const std::array<unsigned int, order> m) {
    const auto & Id0 = make_identity_matrix<Number>(m[0], m[0]);
    const auto & A0  = make_random_matrix_symm<Number>(m[0], m[0]);
    const auto & B0  = make_random_matrix_spd<Number>(m[0], m[0]);
    const auto & Id1 = make_identity_matrix<Number>(m[1], m[1]);
    const auto & A1  = make_random_matrix_symm<Number>(m[1], m[1]);
    const auto & B1  = make_random_matrix_spd<Number>(m[1], m[1]);
    const auto & Id2 = make_identity_matrix<Number>(m[2], m[2]);
    const auto & A2  = make_random_matrix_symm<Number>(m[2], m[2]);
    const auto & B2  = make_random_matrix_spd<Number>(m[2], m[2]);

    {
      std::vector<tensor_type> tensors = {{Id0, Id1, Id2}, {Id0, Id1, Id2}};
      std::bitset<order>       spd_mask("000");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id0, Id1, Id2}, {A0, A1, A2}};
      std::bitset<order>       spd_mask("000");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B0, B1, B2}, {A0, A1, A2}};
      std::bitset<order>       spd_mask("000");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A0, A1, A2}, {B0, B1, B2}};
      std::bitset<order>       spd_mask("111");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A0, B1, B2}, {B0, A1, A2}};
      std::bitset<order>       spd_mask("001");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B0, A1, B2}, {A0, B1, A2}};
      std::bitset<order>       spd_mask("010");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{B0, B1, A2}, {A0, A1, B2}};
      std::bitset<order>       spd_mask("100");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{A0, A1, B2}, {B0, B1, A2}};
      std::bitset<order>       spd_mask("011");
      check_ranktwo<order, Number>(tensors, spd_mask, test_features);
    }
  };

  test_impl({2U, 3U, 4U});

  test_impl({2U, 4U, 3U});

  test_impl({4U, 2U, 3U});

  test_impl({4U, 3U, 2U});
}

TEST(AnisotropicSquare_3Double, ranktwo_reinit)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_ranktwo_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, ranktwo_reinit)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_ranktwo_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3Double, ranktwo_get_eigenvalues)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_ranktwo_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, ranktwo_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_ranktwo_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3Double, ranktwo_apply_inverse)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_ranktwo_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, ranktwo_apply_inverse)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_ranktwo_3D_anisotropic<Number>(test_features);
}



template<typename Number>
void
test_separable_anisotropic(const Tester::Features & test_features)
{
  constexpr int order = 2;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  const auto test_impl = [&](const std::array<unsigned int, order> m) {
    const auto & Id0 = make_identity_matrix<Number>(m[0], m[0]);
    const auto & D0  = make_random_matrix_symm<Number>(m[0], m[0]);
    const auto & M0  = make_random_matrix_spd<Number>(m[0], m[0]);
    const auto & Id1 = make_identity_matrix<Number>(m[1], m[1]);
    const auto & D1  = make_random_matrix_symm<Number>(m[1], m[1]);
    const auto & M1  = make_random_matrix_spd<Number>(m[1], m[1]);

    {
      std::vector<tensor_type> tensors = {{Id0, Id1}, {Id0, Id1}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id0, Id1}, {D0, D1}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{M0, M1}, {D0, D1}};
      check_separable<order, Number>(tensors, test_features);
    }
  };

  test_impl({2U, 3U});

  test_impl({3U, 2U});

  test_impl({4U, 2U});

  test_impl({1U, 3U});
}

TEST(AnisotropicSquare_2Double, separable_reinit)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_separable_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, separable_reinit)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_separable_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2Double, separable_get_eigenvalues)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_separable_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, separable_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_separable_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2Double, separable_apply_inverse)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_separable_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, separable_apply_inverse)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_separable_anisotropic<Number>(test_features);
}



template<typename Number>
void
test_separable_3D_anisotropic(const Tester::Features & test_features)
{
  constexpr int order = 3;

  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  const auto test_impl = [&](const std::array<unsigned int, order> m) {
    const auto & Id0 = make_identity_matrix<Number>(m[0], m[0]);
    const auto & D0  = make_random_matrix_symm<Number>(m[0], m[0]);
    const auto & M0  = make_random_matrix_spd<Number>(m[0], m[0]);
    const auto & Id1 = make_identity_matrix<Number>(m[1], m[1]);
    const auto & D1  = make_random_matrix_symm<Number>(m[1], m[1]);
    const auto & M1  = make_random_matrix_spd<Number>(m[1], m[1]);
    const auto & Id2 = make_identity_matrix<Number>(m[2], m[2]);
    const auto & D2  = make_random_matrix_symm<Number>(m[2], m[2]);
    const auto & M2  = make_random_matrix_spd<Number>(m[2], m[2]);

    {
      std::vector<tensor_type> tensors = {{Id0, Id1, Id2}, {Id0, Id1, Id2}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{Id0, Id1, Id2}, {D0, D1, D2}};
      check_separable<order, Number>(tensors, test_features);
    }

    {
      std::vector<tensor_type> tensors = {{M0, M1, M2}, {D0, D1, D2}};
      check_separable<order, Number>(tensors, test_features);
    }
  };

  test_impl({2U, 3U, 4U});

  test_impl({2U, 4U, 3U});

  test_impl({4U, 2U, 3U});

  test_impl({4U, 3U, 2U});
}

TEST(AnisotropicSquare_3Double, separable_reinit)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_separable_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, separable_reinit)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_separable_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3Double, separable_get_eigenvalues)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_separable_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, separable_get_eigenvalues)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_separable_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3Double, separable_apply_inverse)
{
  using Number = double;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_separable_3D_anisotropic<Number>(test_features);
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, separable_apply_inverse)
{
  using Number = VectorizedArray<double>;
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_separable_3D_anisotropic<Number>(test_features);
}



template<int order, typename Number>
void
test_rankone(const std::array<unsigned int, order> & size, const Tester::Features & test_features)
{
  using tensor_type = typename Tensors::TensorProductMatrix<order, Number>::tensor_type;

  /// identity tensor
  {
    std::vector<tensor_type> tensors(1U);
    for(auto d = 0U; d < order; ++d)
    {
      const auto m       = size[d];
      tensors.front()[d] = make_identity_matrix<Number>(m, m);
    }
    check_rankone<order, Number>(tensors, test_features);
  }

  /// tensor of s.p.d. matrices
  {
    std::vector<tensor_type> tensors(1U);
    for(auto d = 0U; d < order; ++d)
    {
      const auto m       = size[d];
      tensors.front()[d] = make_random_matrix_spd<Number>(m, m);
    }
    check_rankone<order, Number>(tensors, test_features);
  }

  /// tensor of symmetric matrices
  {
    std::vector<tensor_type> tensors(1U);
    for(auto d = 0U; d < order; ++d)
    {
      const auto m       = size[d];
      tensors.front()[d] = make_random_matrix_symm<Number>(m, m);
    }
    check_rankone<order, Number>(tensors, test_features);
  }
}



template<int order, typename Number>
void
test_rankone_reinit(const std::array<unsigned int, order> & size)
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::reinit;
  test_rankone<order, Number>(size, test_features);
}

TEST(IsotropicSquare_2Double, rankone_reinit)
{
  constexpr int order = 2;
  using Number        = double;
  test_rankone_reinit<order, Number>({2U, 2U});
  test_rankone_reinit<order, Number>({3U, 3U});
  test_rankone_reinit<order, Number>({7U, 7U});
}

TEST(IsotropicSquare_2VectorizedArrayDouble, rankone_reinit)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_rankone_reinit<order, Number>({2U, 2U});
  test_rankone_reinit<order, Number>({3U, 3U});
  test_rankone_reinit<order, Number>({7U, 7U});
}

TEST(AnisotropicSquare_2Double, rankone_reinit)
{
  constexpr int order = 2;
  using Number        = double;
  test_rankone_reinit<order, Number>({2U, 3U});
  test_rankone_reinit<order, Number>({5U, 3U});
  test_rankone_reinit<order, Number>({7U, 9U});
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, rankone_reinit)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_rankone_reinit<order, Number>({2U, 3U});
  test_rankone_reinit<order, Number>({5U, 3U});
  test_rankone_reinit<order, Number>({7U, 9U});
}

TEST(IsotropicSquare_3Double, rankone_reinit)
{
  constexpr int order = 3;
  using Number        = double;
  test_rankone_reinit<order, Number>({2U, 2U, 2U});
  test_rankone_reinit<order, Number>({3U, 3U, 3U});
  test_rankone_reinit<order, Number>({7U, 7U, 7U});
}

TEST(IsotropicSquare_3VectorizedArrayDouble, rankone_reinit)
{
  constexpr int order = 3;
  using Number        = VectorizedArray<double>;
  test_rankone_reinit<order, Number>({2U, 2U, 2U});
  test_rankone_reinit<order, Number>({3U, 3U, 3U});
  test_rankone_reinit<order, Number>({7U, 7U, 7U});
}

TEST(AnisotropicSquare_3Double, rankone_reinit)
{
  constexpr int order = 3;
  using Number        = double;
  test_rankone_reinit<order, Number>({2U, 3U, 4U});
  test_rankone_reinit<order, Number>({5U, 3U, 3U});
  test_rankone_reinit<order, Number>({7U, 9U, 2U});
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, rankone_reinit)
{
  constexpr int order = 3;
  using Number        = VectorizedArray<double>;
  test_rankone_reinit<order, Number>({2U, 3U, 4U});
  test_rankone_reinit<order, Number>({5U, 3U, 3U});
  test_rankone_reinit<order, Number>({7U, 9U, 2U});
}



template<int order, typename Number>
void
test_rankone_get_eigenvalues(const std::array<unsigned int, order> & size)
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::get_eigenvalues;
  test_rankone<order, Number>(size, test_features);
}

TEST(IsotropicSquare_2Double, rankone_get_eigenvalues)
{
  constexpr int order = 2;
  using Number        = double;
  test_rankone_get_eigenvalues<order, Number>({2U, 2U});
  test_rankone_get_eigenvalues<order, Number>({3U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 7U});
}

TEST(IsotropicSquare_2VectorizedArrayDouble, rankone_get_eigenvalues)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_rankone_get_eigenvalues<order, Number>({2U, 2U});
  test_rankone_get_eigenvalues<order, Number>({3U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 7U});
}

TEST(AnisotropicSquare_2Double, rankone_get_eigenvalues)
{
  constexpr int order = 2;
  using Number        = double;
  test_rankone_get_eigenvalues<order, Number>({2U, 3U});
  test_rankone_get_eigenvalues<order, Number>({5U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 9U});
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, rankone_get_eigenvalues)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_rankone_get_eigenvalues<order, Number>({2U, 3U});
  test_rankone_get_eigenvalues<order, Number>({5U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 9U});
}

TEST(IsotropicSquare_3Double, rankone_get_eigenvalues)
{
  constexpr int order = 3;
  using Number        = double;
  test_rankone_get_eigenvalues<order, Number>({2U, 2U, 2U});
  test_rankone_get_eigenvalues<order, Number>({3U, 3U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 7U, 7U});
}

TEST(IsotropicSquare_3VectorizedArrayDouble, rankone_get_eigenvalues)
{
  constexpr int order = 3;
  using Number        = VectorizedArray<double>;
  test_rankone_get_eigenvalues<order, Number>({2U, 2U, 2U});
  test_rankone_get_eigenvalues<order, Number>({3U, 3U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 7U, 7U});
}

TEST(AnisotropicSquare_3Double, rankone_get_eigenvalues)
{
  constexpr int order = 3;
  using Number        = double;
  test_rankone_get_eigenvalues<order, Number>({2U, 3U, 4U});
  test_rankone_get_eigenvalues<order, Number>({5U, 3U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 9U, 2U});
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, rankone_get_eigenvalues)
{
  constexpr int order = 3;
  using Number        = VectorizedArray<double>;
  test_rankone_get_eigenvalues<order, Number>({2U, 3U, 4U});
  test_rankone_get_eigenvalues<order, Number>({5U, 3U, 3U});
  test_rankone_get_eigenvalues<order, Number>({7U, 9U, 2U});
}



template<int order, typename Number>
void
test_rankone_apply_inverse(const std::array<unsigned int, order> & size)
{
  Tester::Features test_features;
  test_features.method_variant = Tester::Method::apply_inverse;
  test_rankone<order, Number>(size, test_features);
}

TEST(IsotropicSquare_2Double, rankone_apply_inverse)
{
  constexpr int order = 2;
  using Number        = double;
  test_rankone_apply_inverse<order, Number>({2U, 2U});
  test_rankone_apply_inverse<order, Number>({3U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 7U});
}

TEST(IsotropicSquare_2VectorizedArrayDouble, rankone_apply_inverse)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_rankone_apply_inverse<order, Number>({2U, 2U});
  test_rankone_apply_inverse<order, Number>({3U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 7U});
}

TEST(AnisotropicSquare_2Double, rankone_apply_inverse)
{
  constexpr int order = 2;
  using Number        = double;
  test_rankone_apply_inverse<order, Number>({2U, 3U});
  test_rankone_apply_inverse<order, Number>({5U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 9U});
}

TEST(AnisotropicSquare_2VectorizedArrayDouble, rankone_apply_inverse)
{
  constexpr int order = 2;
  using Number        = VectorizedArray<double>;
  test_rankone_apply_inverse<order, Number>({2U, 3U});
  test_rankone_apply_inverse<order, Number>({5U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 9U});
}

TEST(IsotropicSquare_3Double, rankone_apply_inverse)
{
  constexpr int order = 3;
  using Number        = double;
  test_rankone_apply_inverse<order, Number>({2U, 2U, 2U});
  test_rankone_apply_inverse<order, Number>({3U, 3U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 7U, 7U});
}

TEST(IsotropicSquare_3VectorizedArrayDouble, rankone_apply_inverse)
{
  constexpr int order = 3;
  using Number        = VectorizedArray<double>;
  test_rankone_apply_inverse<order, Number>({2U, 2U, 2U});
  test_rankone_apply_inverse<order, Number>({3U, 3U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 7U, 7U});
}

TEST(AnisotropicSquare_3Double, rankone_apply_inverse)
{
  constexpr int order = 3;
  using Number        = double;
  test_rankone_apply_inverse<order, Number>({2U, 3U, 4U});
  test_rankone_apply_inverse<order, Number>({5U, 3U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 9U, 2U});
}

TEST(AnisotropicSquare_3VectorizedArrayDouble, rankone_apply_inverse)
{
  constexpr int order = 3;
  using Number        = VectorizedArray<double>;
  test_rankone_apply_inverse<order, Number>({2U, 3U, 4U});
  test_rankone_apply_inverse<order, Number>({5U, 3U, 3U});
  test_rankone_apply_inverse<order, Number>({7U, 9U, 2U});
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
