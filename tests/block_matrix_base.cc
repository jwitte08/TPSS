
/*
 * Tests for block_matrix.h:
 *
 * BlockMatrixBasic:
 *
 *  Created on: Aug 13, 2020
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include <gtest/gtest.h>

#include <fstream>

#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

#include "test_utilities.h"

using namespace dealii;



// enum class BMVariant
// {
//   basic,
//   basic2x2
// };

// template<BMVariant variant, typename Number>
// struct BlockMatrixSelector
// {
// };

// template<typename Number>
// struct BlockMatrixSelector<BMVariant::basic, Number>
// {
//   using type = Tensors::BlockMatrixBasic<MatrixAsTable<Number>>;
// };

// template<typename Number>
// struct BlockMatrixSelector<BMVariant::basic2x2, Number>
// {
//   using type = Tensors::BlockMatrixBasic2x2<MatrixAsTable<Number>>;
// };



template<typename T>
class FixBlockMatrixBasicMatrixAsTable : public testing::Test
{
  // static constexpr BMVariant block_matrix_variant = T::template type<0>::template value<0>();
  using Number                     = typename T::template type<0>;
  using matrix_type                = MatrixAsTable<Number>;
  using block_matrix_type          = Tensors::BlockMatrixBasic<MatrixAsTable<Number>>;
  using block_matrix_type_2x2      = Tensors::BlockMatrixBasic2x2<MatrixAsTable<Number>>;
  using scalar_value_type          = typename ExtractScalarType<Number>::type;
  static constexpr auto macro_size = MatrixAsTable<Number>::macro_size;

protected:
  void
  check_vmult(const std::vector<std::size_t> & row_sizes,
              const std::vector<std::size_t> & column_sizes) const
  {
    block_matrix_type block_matrix;
    matrix_type       ref_matrix;
    fill_random_matrix(block_matrix, ref_matrix, row_sizes, column_sizes);

    compare_matrix(block_matrix, ref_matrix);
  }

  void
  check_apply_inverse(const std::vector<std::size_t> & row_sizes,
                      const std::vector<std::size_t> & column_sizes) const
  {
    block_matrix_type block_matrix;
    matrix_type       ref_matrix;
    fill_random_matrix(block_matrix, ref_matrix, row_sizes, column_sizes);

    block_matrix.invert();
    ref_matrix.invert();

    compare_inverse_matrix(block_matrix, ref_matrix);
  }

  void
  check_vmult_2x2(const std::vector<std::size_t> & row_sizes,
                  const std::vector<std::size_t> & column_sizes) const
  {
    block_matrix_type tmp;
    matrix_type       ref_matrix;
    fill_random_matrix(tmp, ref_matrix, row_sizes, column_sizes);
    block_matrix_type_2x2 block_matrix(tmp.get_block(0, 0),
                                       tmp.get_block(0, 1),
                                       tmp.get_block(1, 0),
                                       tmp.get_block(1, 1));

    compare_matrix(block_matrix, ref_matrix);
  }

  void
  check_apply_inverse_2x2(const std::vector<std::size_t> & row_sizes,
                          const std::vector<std::size_t> & column_sizes) const
  {
    block_matrix_type tmp;
    matrix_type       ref_matrix;
    fill_random_matrix(tmp, ref_matrix, row_sizes, column_sizes);
    block_matrix_type_2x2 block_matrix(tmp.get_block(0, 0),
                                       tmp.get_block(0, 1),
                                       tmp.get_block(1, 0),
                                       tmp.get_block(1, 1));

    block_matrix.invert();
    ref_matrix.invert();

    compare_inverse_matrix(block_matrix, ref_matrix);
  }

  void
  fill_random_matrix(block_matrix_type &              block_matrix,
                     matrix_type &                    ref_matrix,
                     const std::vector<std::size_t> & row_sizes,
                     const std::vector<std::size_t> & column_sizes) const
  {
    /// DEBUG
    // const auto fill_block_with_zeros = [](auto & block, const auto m, const auto n) {
    //   auto & mat = block.as_table();
    //   mat.reinit(m, n);
    //   mat.fill(0.);
    // };
    const auto fill_block_with_randoms = [](auto & block, const auto m, const auto n) {
      auto & mat = block.as_table();
      fill_matrix_with_random_values(mat, m, n);
    };
    const auto compute_starts = [](const auto & sizes) {
      std::vector<std::size_t> starts;
      starts.push_back(0U);
      std::partial_sum(sizes.cbegin(), sizes.cend(), std::back_inserter(starts));
      return starts;
    };

    const auto n_block_rows    = row_sizes.size();
    const auto n_block_columns = column_sizes.size();
    const auto row_starts      = compute_starts(row_sizes);
    const auto column_starts   = compute_starts(column_sizes);

    block_matrix.resize(n_block_rows, n_block_columns);

    ref_matrix.as_table().reinit(row_starts.back(), column_starts.back());

    for(auto row = 0U; row < n_block_rows; ++row)
      for(auto column = 0U; column < n_block_columns; ++column)
      {
        const auto m      = row_sizes[row];
        const auto n      = column_sizes[column];
        const auto mstart = row_starts[row];
        const auto nstart = column_starts[column];

        auto & block = block_matrix.get_block(row, column);
        fill_block_with_randoms(block, m, n);

        ref_matrix.fill_submatrix(block.as_table(), mstart, nstart);
      }

    ASSERT_EQ(ref_matrix.m(), block_matrix.m());
    ASSERT_EQ(ref_matrix.n(), block_matrix.n());
  }

  void
  SetUp() override
  {
    ofs.open("block_matrix_base.log", std::ios_base::app);
    // const bool is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout_owned = std::make_shared<ConditionalOStream>(ofs, /*is_first_proc*/ true);
  }

  void
  TearDown() override
  {
    ofs.close();
  }

  template<typename BlockMatrixType>
  void
  compare_matrix(const BlockMatrixType & block_matrix, const matrix_type & other) const
  {
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      const auto & mat = table_to_fullmatrix(block_matrix.as_table(), lane);
      const auto & ref = table_to_fullmatrix(other.as_table(), lane);
      Util::compare_matrix(mat, ref, *pcout_owned);
    }
  }

  template<typename BlockMatrixType>
  void
  compare_inverse_matrix(const BlockMatrixType & block_matrix, const matrix_type & other) const
  {
    for(auto lane = 0U; lane < macro_size; ++lane)
    {
      const auto & mat = table_to_fullmatrix(block_matrix.as_inverse_table(), lane);
      const auto & ref = table_to_fullmatrix(other.as_inverse_table(), lane);
      Util::compare_matrix(mat, ref, *pcout_owned);
    }
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout_owned;
};

TYPED_TEST_SUITE_P(FixBlockMatrixBasicMatrixAsTable);

TYPED_TEST_P(FixBlockMatrixBasicMatrixAsTable, Vmult2x2)
{
  using Fixture = FixBlockMatrixBasicMatrixAsTable<TypeParam>;

  Fixture::check_vmult({2, 2}, {2, 2});

  Fixture::check_vmult({3, 5}, {3, 5});

  Fixture::check_vmult({3, 5}, {4, 2});
}

TYPED_TEST_P(FixBlockMatrixBasicMatrixAsTable, Vmult2x3)
{
  using Fixture = FixBlockMatrixBasicMatrixAsTable<TypeParam>;

  Fixture::check_vmult({2, 2}, {2, 2, 2});

  Fixture::check_vmult({3, 5}, {3, 5, 7});

  Fixture::check_vmult({3, 5}, {4, 2, 3});
}

TYPED_TEST_P(FixBlockMatrixBasicMatrixAsTable, ApplyInverse2x2)
{
  using Fixture = FixBlockMatrixBasicMatrixAsTable<TypeParam>;

  Fixture::check_apply_inverse({2, 2}, {2, 2});

  Fixture::check_apply_inverse({3, 5}, {3, 5});
}

TYPED_TEST_P(FixBlockMatrixBasicMatrixAsTable, VmultBlockMatrixBasic2x2)
{
  using Fixture = FixBlockMatrixBasicMatrixAsTable<TypeParam>;

  Fixture::check_vmult_2x2({2, 2}, {2, 2});

  Fixture::check_vmult_2x2({3, 5}, {3, 5});

  Fixture::check_vmult_2x2({4, 2}, {4, 2});
}

TYPED_TEST_P(FixBlockMatrixBasicMatrixAsTable, ApplyInverseBlockMatrixBasic2x2)
{
  using Fixture = FixBlockMatrixBasicMatrixAsTable<TypeParam>;

  Fixture::check_apply_inverse_2x2({2, 2}, {2, 2});

  Fixture::check_apply_inverse_2x2({3, 5}, {3, 5});

  Fixture::check_apply_inverse_2x2({4, 2}, {4, 2});
}

REGISTER_TYPED_TEST_SUITE_P(FixBlockMatrixBasicMatrixAsTable,
                            Vmult2x2,
                            Vmult2x3,
                            ApplyInverse2x2,
                            VmultBlockMatrixBasic2x2,
                            ApplyInverseBlockMatrixBasic2x2);

// using ParamsDouble     = testing::Types<Util::TypeList<Util::NonTypeParams<BMVariant::basic>,
// double>>; using ParamsVectDouble     =
// testing::Types<Util::TypeList<Util::NonTypeParams<BMVariant::basic>, VectorizedArray<double>>>;
using ParamsDouble     = testing::Types<Util::TypeList<double>>;
using ParamsVectDouble = testing::Types<Util::TypeList<VectorizedArray<double>>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Double, FixBlockMatrixBasicMatrixAsTable, ParamsDouble);
INSTANTIATE_TYPED_TEST_SUITE_P(VectorizedDouble,
                               FixBlockMatrixBasicMatrixAsTable,
                               ParamsVectDouble);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("block_matrix_base.log", std::ios_base::out);
  ofs.close();

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
