#include <deal.II/base/table.h>
#include <gtest/gtest.h>

#include "solvers_and_preconditioners/TPSS/kroneckersvd.h"
#include "solvers_and_preconditioners/TPSS/tensor_product_matrix.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"

#include "test_utilities.h"



struct Tester
{
  enum class Method
  {
    full,
    tensor,
    diagonal
  };

  Tester()
  {
    ofs.open("kroneckersvd.log", std::ios_base::app);
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
  print_matrix(const FullMatrix<Number> & matrix, const std::string & description) const
  {
    Util::print_matrix(matrix, description, *pcout);
  }

  operator std::ostream &()
  {
    AssertThrow(pcout->is_active(), ExcMessage("TODO inactive case"));
    return pcout->get_stream();
  };

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
};



template<typename Number>
void
check_kronecker_svd(const Tensors::TensorProductMatrix<2, Number> & matrix,
                    std::vector<std::array<Table<2, Number>, 2>> &  ksvd,
                    const std::size_t                               lanczos_iterations = -1)
{
  Tester tester;

  compute_ksvd<Number>(matrix.get_elementary_tensors(), ksvd, lanczos_iterations);

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto fullmatrix = table_to_fullmatrix(matrix.as_table(), lane);
    const auto fullksvd =
      table_to_fullmatrix(Tensors::TensorProductMatrix<2, Number>(ksvd).as_table(), lane);

    // tester << "lane: " << lane << " | lanczos_iterations: " << lanczos_iterations << std::endl;
    tester.compare_matrix(fullksvd, fullmatrix);
  }
}



template<typename Number>
void
check_kronecker_svd(const Table<2, Number> &                       matrix,
                    std::vector<std::array<Table<2, Number>, 2>> & ksvd,
                    const std::size_t                              lanczos_iterations = -1)
{
  Tester tester;

  compute_ksvd<Number>(matrix, ksvd, lanczos_iterations);

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto fullmatrix = table_to_fullmatrix(matrix, lane);
    const auto fullksvd =
      table_to_fullmatrix(Tensors::TensorProductMatrix<2, Number>(ksvd).as_table(), lane);
    tester.compare_matrix(fullksvd, fullmatrix);
  }
}



template<typename Number>
void
check_kronecker_svd(const AlignedVector<Number> &                  diagonal,
                    std::vector<std::array<Table<2, Number>, 2>> & ksvd,
                    const std::size_t                              lanczos_iterations = -1)
{
  Tester tester;

  compute_ksvd<Number>(diagonal, ksvd, lanczos_iterations);

  DiagonalMatrixWrap<Number> matrix;
  matrix.reinit(diagonal);

  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto fullmatrix = table_to_fullmatrix(matrix.as_table(), lane);
    const auto fullksvd =
      table_to_fullmatrix(Tensors::TensorProductMatrix<2, Number>(ksvd).as_table(), lane);
    tester.compare_matrix(fullksvd, fullmatrix);
  }
}



template<typename Number,
         bool do_zero_out_first_lane          = false,
         bool test_minimal_lanczos_iterations = false,
         bool test_large_sizes                = false>
void
test_kronecker_svd()
{
  const auto test_impl = [&](std::array<unsigned int, 2> rows,
                             std::array<unsigned int, 2> columns) {
    const auto zero_out_first_lane = [&](auto & rank1_tensors) {
      for(auto & tensor : rank1_tensors)
        for(auto & mat : tensor)
          for(auto i = 0U; i < mat.size(0); ++i)
            for(auto j = 0U; j < mat.size(1); ++j)
              scalar_value(mat(i, j), /*lane*/ 0U) = 0.;
    };

    const auto check_kronecker_svd_impl = [&](auto &            rank1_tensors,
                                              const std::size_t ksvd_rank_in = -1) {
      if(do_zero_out_first_lane)
        zero_out_first_lane(rank1_tensors);

      const std::size_t ksvd_rank =
        ksvd_rank_in == static_cast<std::size_t>(-1) ? rank1_tensors.size() : ksvd_rank_in;

      std::vector<std::array<Table<2, Number>, 2>> ksvd =
        Tensors::make_zero_rank1_tensors<2, Number>(ksvd_rank, rows, columns);

      if(test_minimal_lanczos_iterations)
        check_kronecker_svd(Tensors::TensorProductMatrix<2, Number>(rank1_tensors),
                            ksvd,
                            rank1_tensors.size() + 1);
      else
        check_kronecker_svd(Tensors::TensorProductMatrix<2, Number>(rank1_tensors), ksvd);
    };

    const auto & [m1, m0] = rows;
    const auto & [n1, n0] = columns;

    const auto Id0 = Util::make_identity_matrix<Number>(m0, n0);
    const auto Id1 = Util::make_identity_matrix<Number>(m1, n1);
    const auto A0  = Util::make_random_matrix<Number>(m0, n0);
    const auto A1  = Util::make_random_matrix<Number>(m1, n1);
    const auto B0  = Util::make_random_matrix<Number>(m0, n0);
    const auto B1  = Util::make_random_matrix<Number>(m1, n1);
    const auto C0  = Util::make_random_matrix<Number>(m0, n0);
    const auto C1  = Util::make_random_matrix<Number>(m1, n1);

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0}, {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0},
                                                                 {Id1, Id0},
                                                                 {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{A1, A0}, {A1, A0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{B1, B0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{C1, C0},
                                                                 {B1, B0},
                                                                 {A1, A0},
                                                                 {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors);
    }
  };

  if(test_large_sizes)
  {
    // test_impl({14U, 11U}, {13U, 7U});

    test_impl({24U, 21U}, {23U, 27U});
  }

  else
  {
    test_impl({3U, 4U}, {3U, 4U});

    test_impl({7U, 7U}, {11U, 11U});

    test_impl({2U, 4U}, {5U, 3U});

    test_impl({4U, 7U}, {9U, 8U});
  }
}



template<typename Number,
         bool do_zero_out_first_lane          = false,
         bool test_minimal_lanczos_iterations = false,
         bool test_large_sizes                = false>
void
test_kronecker_svd_full()
{
  const auto test_impl = [&](std::array<unsigned int, 2> rows,
                             std::array<unsigned int, 2> columns) {
    const auto zero_out_first_lane = [&](auto & rank1_tensors) {
      for(auto & tensor : rank1_tensors)
        for(auto & mat : tensor)
          for(auto i = 0U; i < mat.size(0); ++i)
            for(auto j = 0U; j < mat.size(1); ++j)
              scalar_value(mat(i, j), /*lane*/ 0U) = 0.;
    };

    const auto check_kronecker_svd_impl = [&](auto &            rank1_tensors,
                                              const std::size_t ksvd_rank_in = -1) {
      if(do_zero_out_first_lane)
        zero_out_first_lane(rank1_tensors);

      const std::size_t ksvd_rank =
        ksvd_rank_in == static_cast<std::size_t>(-1) ? rank1_tensors.size() : ksvd_rank_in;

      std::vector<std::array<Table<2, Number>, 2>> ksvd =
        Tensors::make_zero_rank1_tensors<2, Number>(ksvd_rank, rows, columns);


      if(test_minimal_lanczos_iterations)
        check_kronecker_svd(Tensors::TensorProductMatrix<2, Number>(rank1_tensors).as_table(),
                            ksvd,
                            rank1_tensors.size() + 1);
      else
        check_kronecker_svd(Tensors::TensorProductMatrix<2, Number>(rank1_tensors).as_table(),
                            ksvd);
    };

    const auto & [m1, m0] = rows;
    const auto & [n1, n0] = columns;

    const auto Id0 = Util::make_identity_matrix<Number>(m0, n0);
    const auto Id1 = Util::make_identity_matrix<Number>(m1, n1);
    const auto A0  = Util::make_random_matrix<Number>(m0, n0);
    const auto A1  = Util::make_random_matrix<Number>(m1, n1);
    const auto B0  = Util::make_random_matrix<Number>(m0, n0);
    const auto B1  = Util::make_random_matrix<Number>(m1, n1);
    const auto C0  = Util::make_random_matrix<Number>(m0, n0);
    const auto C1  = Util::make_random_matrix<Number>(m1, n1);

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0}, {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0},
                                                                 {Id1, Id0},
                                                                 {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{A1, A0}, {A1, A0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{B1, B0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{C1, C0},
                                                                 {B1, B0},
                                                                 {A1, A0},
                                                                 {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors);
    }
  };

  if(test_large_sizes)
  {
    // test_impl({14U, 11U}, {13U, 7U});

    test_impl({24U, 21U}, {23U, 27U});
  }

  else
  {
    test_impl({3U, 4U}, {3U, 4U});

    test_impl({7U, 7U}, {11U, 11U});

    test_impl({2U, 4U}, {5U, 3U});

    test_impl({4U, 7U}, {9U, 8U});
  }
}



template<typename Number,
         bool do_zero_out_first_lane          = false,
         bool test_minimal_lanczos_iterations = false,
         bool test_large_sizes                = false>
void
test_kronecker_svd_diagonal()
{
  const auto test_impl = [&](std::array<unsigned int, 2> rows,
                             std::array<unsigned int, 2> columns) {
    const auto zero_out_first_lane = [&](auto & rank1_tensors) {
      for(auto & tensor : rank1_tensors)
        for(auto & mat : tensor)
          for(auto i = 0U; i < mat.size(0); ++i)
            for(auto j = 0U; j < mat.size(1); ++j)
              scalar_value(mat(i, j), /*lane*/ 0U) = 0.;
    };

    const auto check_kronecker_svd_impl = [&](auto &            rank1_tensors,
                                              const std::size_t ksvd_rank_in = -1) {
      if(do_zero_out_first_lane)
        zero_out_first_lane(rank1_tensors);

      const std::size_t ksvd_rank =
        ksvd_rank_in == static_cast<std::size_t>(-1) ? rank1_tensors.size() : ksvd_rank_in;

      std::vector<std::array<Table<2, Number>, 2>> ksvd =
        Tensors::make_zero_rank1_tensors<2, Number>(ksvd_rank, rows, columns);

      const auto get_diagonal =
        [](const std::vector<std::array<Table<2, Number>, 2>> & rank1_tensors) {
          Tensors::TensorProductMatrix<2, Number> tpm(rank1_tensors);
          const auto &                            mat = tpm.as_table();
          AssertDimension(mat.size(0), mat.size(1));
          AlignedVector<Number> diagonal(mat.size(0));
          for(auto i = 0U; i < mat.size(0); ++i)
            diagonal[i] = mat(i, i);
          return diagonal;
        };

      const std::size_t n_lanczos_iterations =
        test_minimal_lanczos_iterations ? rank1_tensors.size() + 1 : -1;
      check_kronecker_svd(get_diagonal(rank1_tensors), ksvd, n_lanczos_iterations);
    };

    const auto & [m1, m0] = rows;
    const auto & [n1, n0] = columns;

    const auto Id0 = Util::make_identity_matrix<Number>(m0, n0);
    const auto Id1 = Util::make_identity_matrix<Number>(m1, n1);

    const auto make_random_diagonal_matrix = [](const unsigned int m, const unsigned int n) {
      Table<2, Number> mat = Util::make_random_matrix<Number>(m, n);
      for(auto i = 0U; i < mat.size(0); ++i)
        for(auto j = 0U; j < mat.size(1); ++j)
          if(i != j)
            mat(i, j) = static_cast<Number>(0.);
      return mat;
    };

    const auto A0 = make_random_diagonal_matrix(m0, n0);
    const auto A1 = make_random_diagonal_matrix(m1, n1);
    const auto B0 = make_random_diagonal_matrix(m0, n0);
    const auto B1 = make_random_diagonal_matrix(m1, n1);
    const auto C0 = make_random_diagonal_matrix(m0, n0);
    const auto C1 = make_random_diagonal_matrix(m1, n1);

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0}};

      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0}, {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{Id1, Id0},
                                                                 {Id1, Id0},
                                                                 {Id1, Id0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{A1, A0}, {A1, A0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors, 1U);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{B1, B0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors);
    }

    {
      std::vector<std::array<Table<2, Number>, 2>> rank1_tensors{{C1, C0}, {B1, B0}, {A1, A0}};
      check_kronecker_svd_impl(rank1_tensors);
    }
  };

  if(test_large_sizes)
  {
    test_impl({24U, 21U}, {24U, 21U});
  }

  else
  {
    test_impl({3U, 3U}, {3U, 3U});

    test_impl({3U, 4U}, {3U, 4U});

    test_impl({7U, 3U}, {7U, 3U});
  }
}



void
test_rank_two_kronecker_svd_full()
{
  /* Matrices:
     t1:
     |1    0    1  |
     |1    1   0.1 |
     |1   0.1   -7 |
     |0    -1   0  |
     t2:
     |1    2|
     |2    4|
     t3:
     |0.5  3    2  |
     |3.4  1  -0.5 |
     | 2 -0.4   0  |
     | 0   0    0  |
     t4:
     |1    0|
     |1    1|   */

  Table<2, double> t1 = Table<2, double>(4, 3);
  t1(0, 0)            = 1;
  t1(1, 1)            = 1;
  t1(0, 1)            = 0;
  t1(1, 0)            = 0;
  t1(0, 2)            = 1;
  t1(1, 2)            = 0.1;
  t1(2, 0)            = 1;
  t1(2, 1)            = 0.1;
  t1(2, 2)            = -7;
  t1(3, 1)            = -1;
  Table<2, double> t2 = Table<2, double>(2, 2);
  t2(0, 0)            = 1;
  t2(1, 1)            = 4;
  t2(0, 1)            = 2;
  t2(1, 0)            = 2;
  Table<2, double> t3 = Table<2, double>(4, 3);
  t3(0, 0)            = 0.5;
  t3(1, 1)            = 1;
  t3(0, 1)            = 3;
  t3(1, 0)            = 3.4;
  t3(0, 2)            = 2;
  t3(1, 2)            = -0.4;
  t3(2, 0)            = 2;
  t3(2, 1)            = -0.5;
  t3(2, 2)            = 0;
  Table<2, double> t4 = Table<2, double>(2, 2);
  t4(0, 0)            = 1;
  t4(1, 1)            = 0;
  t4(0, 1)            = 1;
  t4(1, 0)            = 1;

  std::array<Table<2, double>, 2> kp1 = {t2, t1};
  std::array<Table<2, double>, 2> kp2 = {t4, t3};
  Table<2, double>                mat1 =
    LinAlg::sum(Tensors::kronecker_product(t1, t2), Tensors::kronecker_product(t3, t4));

  std::array<std::size_t, 2>                   rows{t2.size(0), t1.size(0)};
  std::array<std::size_t, 2>                   columns{t2.size(1), t1.size(1)};
  std::vector<std::array<Table<2, double>, 2>> ksvd_of_mat1 =
    Tensors::make_zero_rank1_tensors<2, double>(2U, rows, columns);

  check_kronecker_svd<double>(mat1, ksvd_of_mat1);
}



void
test_rank_two_kronecker_svd()
{
  /* Matrices:
     t1:
     |1    0    1  |
     |1    1   0.1 |
     |1   0.1   -7 |
     |0    -1   0  |
     t2:
     |1    2|
     |2    4|
     t3:
     |0.5  3    2  |
     |3.4  1  -0.5 |
     | 2 -0.4   0  |
     | 0   0    0  |
     t4:
     |1    0|
     |1    1|   */

  Table<2, double> t1 = Table<2, double>(4, 3);
  Table<2, double> t2 = Table<2, double>(2, 2);
  t1(0, 0)            = 1;
  t1(1, 1)            = 1;
  t1(0, 1)            = 0;
  t1(1, 0)            = 0;
  t1(0, 2)            = 1;
  t1(1, 2)            = 0.1;
  t1(2, 0)            = 1;
  t1(2, 1)            = 0.1;
  t1(2, 2)            = -7;
  t1(3, 1)            = -1;
  t2(0, 0)            = 1;
  t2(1, 1)            = 4;
  t2(0, 1)            = 2;
  t2(1, 0)            = 2;
  Table<2, double> t3 = Table<2, double>(4, 3);
  Table<2, double> t4 = Table<2, double>(2, 2);
  t3(0, 0)            = 0.5;
  t3(1, 1)            = 1;
  t3(0, 1)            = 3;
  t3(1, 0)            = 3.4;
  t3(0, 2)            = 2;
  t3(1, 2)            = -0.4;
  t3(2, 0)            = 2;
  t3(2, 1)            = -0.5;
  t3(2, 2)            = 0;
  t4(0, 0)            = 1;
  t4(1, 1)            = 0;
  t4(0, 1)            = 1;
  t4(1, 0)            = 1;

  std::array<Table<2, double>, 2>              kp1  = {t2, t1};
  std::array<Table<2, double>, 2>              kp2  = {t4, t3};
  std::vector<std::array<Table<2, double>, 2>> mat1 = {kp1, kp2};

  std::array<std::size_t, 2>                   rows{t2.size(0), t1.size(0)};
  std::array<std::size_t, 2>                   columns{t2.size(1), t1.size(1)};
  std::vector<std::array<Table<2, double>, 2>> ksvd_of_mat1 =
    Tensors::make_zero_rank1_tensors<2, double>(2U, rows, columns);

  check_kronecker_svd<double>(mat1, ksvd_of_mat1);
}



void
test_rank_two_vectorized_kronecker_svd()
{
  Table<2, VectorizedArray<double>> t5 = Table<2, VectorizedArray<double>>(2, 3);
  Table<2, VectorizedArray<double>> t6 = Table<2, VectorizedArray<double>>(2, 2);
  t5(0, 0)                             = make_vectorized_array<double>(1);
  t5(1, 1)                             = make_vectorized_array<double>(1);
  t5(0, 1)                             = make_vectorized_array<double>(0);
  t5(1, 0)                             = make_vectorized_array<double>(0);
  t6(0, 0)                             = make_vectorized_array<double>(1);
  t6(1, 1)                             = make_vectorized_array<double>(4);
  if(VectorizedArray<double>::size() > 2)
    t6(1, 1)[2] = 3.1415;
  t6(0, 1)                             = make_vectorized_array<double>(2);
  t6(1, 0)                             = make_vectorized_array<double>(2);
  Table<2, VectorizedArray<double>> t7 = Table<2, VectorizedArray<double>>(2, 3);
  Table<2, VectorizedArray<double>> t8 = Table<2, VectorizedArray<double>>(2, 2);
  t7(0, 0)                             = make_vectorized_array<double>(0.5);
  t7(1, 1)                             = make_vectorized_array<double>(1);
  t7(0, 1)                             = make_vectorized_array<double>(3);
  t7(0, 1)[1]                          = -2;
  t7(1, 0)                             = make_vectorized_array<double>(3.4);
  t8(0, 0)                             = make_vectorized_array<double>(1);
  t8(1, 1)                             = make_vectorized_array<double>(0);
  t8(0, 1)                             = make_vectorized_array<double>(1);
  t8(0, 0)[0]                          = -2;
  if(VectorizedArray<double>::size() > 2)
    t8(0, 0)[3] = -200;
  t8(1, 0) = make_vectorized_array<double>(1);

  std::array<Table<2, VectorizedArray<double>>, 2>              kp3  = {t5, t6};
  std::array<Table<2, VectorizedArray<double>>, 2>              kp4  = {t7, t8};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 2>> mat2 = {kp3, kp4};

  std::array<std::size_t, 2>                                    rows{t5.size(0), t6.size(0)};
  std::array<std::size_t, 2>                                    columns{t5.size(1), t6.size(1)};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 2>> ksvd_of_mat2 =
    Tensors::make_zero_rank1_tensors<2, VectorizedArray<double>>(2U, rows, columns);

  check_kronecker_svd<VectorizedArray<double>>(mat2, ksvd_of_mat2);
}



void
test_rank_one_kronecker_svd()
{
  /* Matrices:
     t1:
     |1   1.70  1  |
     |1    1   0.1 |
     |1   0.1   -7 |
     |0    -1   0  |
     t2:
     |1    2|
     |200  4|
  */

  Table<2, double> t1 = Table<2, double>(4, 3);
  Table<2, double> t2 = Table<2, double>(2, 2);
  t1(0, 0)            = 1;
  t1(1, 1)            = 1;
  t1(0, 1)            = 1.7;
  t1(1, 0)            = 0;
  t1(0, 2)            = 1;
  t1(1, 2)            = 0.1;
  t1(2, 0)            = 1;
  t1(2, 1)            = 0.1;
  t1(2, 2)            = -7;
  t1(3, 1)            = -1;
  t2(0, 0)            = 1;
  t2(1, 1)            = 4;
  t2(0, 1)            = 2;
  t2(1, 0)            = 200;

  std::array<Table<2, double>, 2>              kp1  = {t2, t1};
  std::vector<std::array<Table<2, double>, 2>> mat1 = {kp1};

  std::array<std::size_t, 2>                   rows{t2.size(0), t1.size(0)};
  std::array<std::size_t, 2>                   columns{t2.size(1), t1.size(1)};
  std::vector<std::array<Table<2, double>, 2>> ksvd_of_mat1 =
    Tensors::make_zero_rank1_tensors<2, double>(1U, rows, columns);

  check_kronecker_svd<double>(mat1, ksvd_of_mat1);
}



void
test_diagonal_kronecker_svd()
{
  AlignedVector<double> diagonal(12);
  diagonal[0]  = 1;
  diagonal[1]  = 2;
  diagonal[2]  = 3;
  diagonal[3]  = 4;
  diagonal[4]  = 5;
  diagonal[5]  = 6;
  diagonal[6]  = 7;
  diagonal[7]  = 8;
  diagonal[8]  = 9;
  diagonal[9]  = 10;
  diagonal[10] = 11;
  diagonal[11] = 12;

  Table<2, double>                             t1(3, 3);
  Table<2, double>                             t2(4, 4);
  std::array<Table<2, double>, 2>              app_kp = {t1, t2};
  std::vector<std::array<Table<2, double>, 2>> approx = {app_kp, app_kp};

  compute_ksvd<double>(diagonal, approx);

  // .710868 was generated by working python program, error would be smaller for more digits
  EXPECT_TRUE(std::abs(approx[0][1](0, 0) - 0.710868) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](1, 1) - 1.735563) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](2, 2) - 2.760258) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](3, 3) - 3.784953) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](0, 0) - 2.545889) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](1, 1) - 2.899023) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](2, 2) - 3.252157) < 1e-6);
}



void
test_diagonal_vectorized_kronecker_svd()
{
  AlignedVector<VectorizedArray<double>> diagonal(12);
  diagonal[0]    = make_vectorized_array<double>(1);
  diagonal[1]    = make_vectorized_array<double>(2);
  diagonal[2]    = make_vectorized_array<double>(3);
  diagonal[3]    = make_vectorized_array<double>(4);
  diagonal[3][1] = -12;
  diagonal[4]    = make_vectorized_array<double>(5);
  diagonal[5]    = make_vectorized_array<double>(6);
  diagonal[6]    = make_vectorized_array<double>(7);
  diagonal[7]    = make_vectorized_array<double>(8);
  diagonal[8]    = make_vectorized_array<double>(9);
  diagonal[8][1] = -33;
  diagonal[9]    = make_vectorized_array<double>(10);
  diagonal[10]   = make_vectorized_array<double>(11);
  diagonal[11]   = make_vectorized_array<double>(12);

  Table<2, VectorizedArray<double>>                             t1(3, 3);
  Table<2, VectorizedArray<double>>                             t2(4, 4);
  std::array<Table<2, VectorizedArray<double>>, 2>              app_kp = {t1, t2};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 2>> approx = {app_kp, app_kp};

  compute_ksvd<VectorizedArray<double>>(diagonal, approx);

  // .710868 was generated by working python program, error would be smaller for
  // more digits
  EXPECT_TRUE(std::abs(approx[0][1](0, 0)[1] + 0.422376) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](1, 1)[1] + 1.250524) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](2, 2)[1] - 5.699786) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](3, 3)[1] + 1.457304) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](0, 0)[1] - 1.097814) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](1, 1)[1] - 0.618112) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](2, 2)[1] + 5.896291) < 1e-6);
}



TEST(Anisotropic_Double, compute_ksvd_tensor)
{
  test_kronecker_svd<double>();
}

TEST(Anisotropic_VectorizedArrayDouble, compute_ksvd_tensor)
{
  test_kronecker_svd<VectorizedArray<double>>();
}

TEST(AnisotropicZeroLane_VectorizedArrayDouble, compute_ksvd_tensor)
{
  test_kronecker_svd<VectorizedArray<double>, /*zero_out_first_lane*/ true>();
}

TEST(AnisotropicMinimalLanczosIterations_Double, compute_ksvd_tensor)
{
  test_kronecker_svd<double, false, /*test_minimal_lanczos_iteratios*/ true>();
}

TEST(AnisotropicMinimalLanczosIterationsVectorizedArrayDouble, compute_ksvd_tensor)
{
  test_kronecker_svd<VectorizedArray<double>, false, /*test_minimal_lanczos_iteratios*/ true>();
}

TEST(LargeAnisotropicMinimalLanczosIterations_Double, compute_ksvd_tensor)
{
  test_kronecker_svd<double, false, true, true>();
}

TEST(LargeAnisotropicMinimalLanczosIterationsVectorizedArrayDouble, compute_ksvd_tensor)
{
  test_kronecker_svd<VectorizedArray<double>, false, true, true>();
}

TEST(RankOne_Double, compute_ksvd_tensor)
{
  test_rank_one_kronecker_svd();
}

TEST(RankTwo_Double, compute_ksvd_tensor)
{
  test_rank_two_kronecker_svd();
}

TEST(RankTwo_VectorizedArrayDouble, compute_ksvd_tensor)
{
  test_rank_two_vectorized_kronecker_svd();
}



TEST(RankTwo_Double, compute_ksvd_full)
{
  test_rank_two_kronecker_svd_full();
}

TEST(Anisotropic_Double, compute_ksvd_full)
{
  test_kronecker_svd_full<double>();
}

TEST(Anisotropic_VectorizedArrayDouble, compute_ksvd_full)
{
  test_kronecker_svd_full<VectorizedArray<double>>();
}

TEST(AnisotropicZeroLane_VectorizedArrayDouble, compute_ksvd_full)
{
  test_kronecker_svd_full<VectorizedArray<double>, /*zero_out_first_lane*/ true>();
}

TEST(AnisotropicMinimalLanczosIterations_Double, compute_ksvd_full)
{
  test_kronecker_svd_full<double, false, /*test_minimal_lanczos_iteratios*/ true>();
}



TEST(HardcodedSimon_Double, compute_ksvd_diagonal)
{
  test_diagonal_kronecker_svd();
}

TEST(HardcodedSimon_VectorizedArrayDouble, compute_ksvd_diagonal)
{
  test_diagonal_vectorized_kronecker_svd();
}

TEST(AnisotropicSquared_Double, compute_ksvd_diagonal)
{
  test_kronecker_svd_diagonal<double>();
}

TEST(AnisotropicSquared_VectorizedArrayDouble, compute_ksvd_diagonal)
{
  test_kronecker_svd_diagonal<VectorizedArray<double>>();
}

TEST(AnisotropicSquaredMinimalLanczosIterations_Double, compute_ksvd_diagonal)
{
  test_kronecker_svd_diagonal<double, false, true>();
}

TEST(LargeAnisotropicSquared_Double, compute_ksvd_diagonal)
{
  test_kronecker_svd_diagonal<double, false, false, true>();
}

TEST(IsotropicSquared_VectorizedArrayDouble, compute_ksvd_diagonal)
{
  using Number         = VectorizedArray<double>;
  const unsigned int m = 3;

  Tester tester;

  AlignedVector<Number> diagonal;
  diagonal.resize(m * m);
  fill_with_random_values(diagonal);
  Table<2, Number> diagonal_matrix;
  diagonal_matrix.reinit(m * m, m * m);
  for(auto i = 0U; i < m * m; ++i)
    diagonal_matrix(i, i) = diagonal[i];

  std::vector<std::array<Table<2, Number>, 2>> kd_tensors;
  kd_tensors.resize(m);
  for(auto & tensors : kd_tensors)
    for(auto & elem : tensors)
      elem.reinit(m, m);
  compute_ksvd(diagonal, kd_tensors);

  Tensors::TensorProductMatrix<2, Number> kd;
  kd.reinit(kd_tensors);
  for(auto lane = 0U; lane < get_macro_size<Number>(); ++lane)
  {
    const auto & kd_full   = table_to_fullmatrix(kd.as_table(), lane);
    const auto & reference = table_to_fullmatrix(diagonal_matrix, lane);
    tester.compare_matrix(kd_full, reference);
  }
}



void
test_three_dim_kronecker_cp()
{
  /* Matrices:
     t1:
     |1    0    1  |
     |1    1   0.1 |
     |1   0.1   -7 |
     |0    -1   0  |
     t2:
     |1    2|
     |2    4|
     t3:
     |0.5  3    2  |
     |3.4  1  -0.5 |
     | 2 -0.4   0  |
     | 0   0    0  |
     t4:
     |1    0|
     |1    1|
     t5:
     0.8*t3     */

  Table<2, double> t1                               = Table<2, double>(4, 3);
  Table<2, double> t2                               = Table<2, double>(2, 2);
  t1(0, 0)                                          = 1;
  t1(1, 1)                                          = 1;
  t1(0, 1)                                          = 0;
  t1(1, 0)                                          = 0;
  t1(0, 2)                                          = 1;
  t1(1, 2)                                          = 0.1;
  t1(2, 0)                                          = 1;
  t1(2, 1)                                          = 0.1;
  t1(2, 2)                                          = -7;
  t1(3, 1)                                          = -1;
  t2(0, 0)                                          = 1;
  t2(1, 1)                                          = 4;
  t2(0, 1)                                          = 2;
  t2(1, 0)                                          = 2;
  Table<2, double> t3                               = Table<2, double>(4, 3);
  Table<2, double> t4                               = Table<2, double>(2, 2);
  t3(0, 0)                                          = 0.5;
  t3(1, 1)                                          = 1;
  t3(0, 1)                                          = 3;
  t3(1, 0)                                          = 3.4;
  t3(0, 2)                                          = 2;
  t3(1, 2)                                          = -0.4;
  t3(2, 0)                                          = 2;
  t3(2, 1)                                          = -0.5;
  t3(2, 2)                                          = 0;
  t4(0, 0)                                          = 1;
  t4(1, 1)                                          = 0;
  t4(0, 1)                                          = 1;
  t4(1, 0)                                          = 1;
  Table<2, double>                             t5   = LinAlg::scaling(t3, 0.8);
  std::array<Table<2, double>, 3>              kp1  = {t1, t2, t2};
  std::array<Table<2, double>, 3>              kp2  = {t3, t4, t4};
  std::array<Table<2, double>, 3>              kp3  = {t5, t4, t4};
  std::vector<std::array<Table<2, double>, 3>> mat1 = {
    kp1, kp2, kp3}; // Since the initial values are chosen differently for the CP-ALS algorithm
  // we
  // start with a Rank-3 representation of a Kronecker-rank 2 matrix
  std::vector<std::array<Table<2, double>, 3>> approx = {kp1, kp1};
  compute_kcp<double>(mat1, approx);

  Table<2, double> approximate_matrix =
    LinAlg::sum(Tensors::kronecker_product(approx[0][0],
                                           Tensors::kronecker_product(approx[0][1], approx[0][2])),
                Tensors::kronecker_product(approx[1][0],
                                           Tensors::kronecker_product(approx[1][1], approx[1][2])));

  Table<2, double> original_matrix =
    LinAlg::sum(Tensors::kronecker_product(t1, Tensors::kronecker_product(t2, t2)),
                LinAlg::sum(Tensors::kronecker_product(t3, Tensors::kronecker_product(t4, t4)),
                            Tensors::kronecker_product(t5, Tensors::kronecker_product(t4, t4))));

  Tester       tester;
  const auto & fullmatrix = table_to_fullmatrix(approximate_matrix);
  const auto & reference  = table_to_fullmatrix(original_matrix);
  tester.compare_matrix(fullmatrix, reference);
}



void
test_three_dim_kronecker_cp_vectorized()
{
  Table<2, VectorizedArray<double>> t1 = Table<2, VectorizedArray<double>>(4, 3);
  Table<2, VectorizedArray<double>> t2 = Table<2, VectorizedArray<double>>(2, 2);
  t1(0, 0)                             = make_vectorized_array<double>(1);
  t1(1, 1)                             = make_vectorized_array<double>(1);
  t1(0, 1)                             = make_vectorized_array<double>(0);
  t1(1, 0)                             = make_vectorized_array<double>(0);
  t1(0, 2)                             = make_vectorized_array<double>(1);
  t1(1, 2)                             = make_vectorized_array<double>(0.1);
  t1(2, 0)                             = make_vectorized_array<double>(1);
  t1(2, 1)                             = make_vectorized_array<double>(0.1);
  t1(2, 2)                             = make_vectorized_array<double>(-7);
  t1(3, 1)                             = make_vectorized_array<double>(-1);
  t2(0, 0)                             = make_vectorized_array<double>(1);
  t2(1, 1)                             = make_vectorized_array<double>(4);
  t2(0, 1)                             = make_vectorized_array<double>(2);
  t2(1, 0)                             = make_vectorized_array<double>(2);
  Table<2, VectorizedArray<double>> t3 = Table<2, VectorizedArray<double>>(4, 3);
  Table<2, VectorizedArray<double>> t4 = Table<2, VectorizedArray<double>>(2, 2);
  t3(0, 0)                             = make_vectorized_array<double>(0.5);
  t3(1, 1)                             = make_vectorized_array<double>(1);
  t3(0, 1)                             = make_vectorized_array<double>(3);
  t3(1, 0)                             = make_vectorized_array<double>(3.4);
  t3(0, 2)                             = make_vectorized_array<double>(2);
  t3(1, 2)                             = make_vectorized_array<double>(-0.4);
  t3(2, 0)                             = make_vectorized_array<double>(2);
  t3(2, 1)                             = make_vectorized_array<double>(-0.5);
  t3(2, 2)                             = make_vectorized_array<double>(0);
  t3(2, 0)[1]                          = 7.3;
  t4(0, 0)                             = make_vectorized_array<double>(1);
  t4(1, 1)                             = make_vectorized_array<double>(0);
  t4(0, 1)                             = make_vectorized_array<double>(1);
  t4(1, 0)                             = make_vectorized_array<double>(1);
  t4(1, 1)[0]                          = 999;
  Table<2, VectorizedArray<double>> t5 = LinAlg::scaling(t3, make_vectorized_array(0.8));
  std::array<Table<2, VectorizedArray<double>>, 3>              kp1  = {t1, t2, t2};
  std::array<Table<2, VectorizedArray<double>>, 3>              kp2  = {t3, t4, t4};
  std::array<Table<2, VectorizedArray<double>>, 3>              kp3  = {t5, t4, t4};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 3>> mat1 = {
    kp1, kp2, kp3}; // Since the initial values are chosen differently for the CP-ALS algorithm
  // we
  // start with a Rank-3 representation of a Kronecker-rank 2 matrix
  std::vector<std::array<Table<2, VectorizedArray<double>>, 3>> approx = {kp1, kp1};
  compute_kcp<VectorizedArray<double>>(mat1, approx);

  Table<2, VectorizedArray<double>> approximate_matrix =
    LinAlg::sum(Tensors::kronecker_product(approx[0][0],
                                           Tensors::kronecker_product(approx[0][1], approx[0][2])),
                Tensors::kronecker_product(approx[1][0],
                                           Tensors::kronecker_product(approx[1][1], approx[1][2])));

  Table<2, VectorizedArray<double>> original_matrix =
    LinAlg::sum(Tensors::kronecker_product(t1, Tensors::kronecker_product(t2, t2)),
                LinAlg::sum(Tensors::kronecker_product(t3, Tensors::kronecker_product(t4, t4)),
                            Tensors::kronecker_product(t5, Tensors::kronecker_product(t4, t4))));

  Tester tester;
  for(auto lane = 0U; lane < VectorizedArray<double>::size(); ++lane)
  {
    const auto & fullmatrix = table_to_fullmatrix(approximate_matrix, lane);
    const auto & reference  = table_to_fullmatrix(original_matrix, lane);
    tester.compare_matrix(fullmatrix, reference);
  }
}

TEST(KroneckerSVD, ThreeDim)
{
  test_three_dim_kronecker_cp();
}

TEST(KroneckerSVD, ThreeDimVectorized)
{
  test_three_dim_kronecker_cp_vectorized();
}



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("kroneckersvd.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
