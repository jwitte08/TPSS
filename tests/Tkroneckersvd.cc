#include <deal.II/base/table.h>
#include <gtest/gtest.h>

#include "solvers_and_preconditioners/TPSS/kroneckersvd.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"


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

  Table<2, double> t1                                 = Table<2, double>(4, 3);
  Table<2, double> t2                                 = Table<2, double>(2, 2);
  t1(0, 0)                                            = 1;
  t1(1, 1)                                            = 1;
  t1(0, 1)                                            = 0;
  t1(1, 0)                                            = 0;
  t1(0, 2)                                            = 1;
  t1(1, 2)                                            = 0.1;
  t1(2, 0)                                            = 1;
  t1(2, 1)                                            = 0.1;
  t1(2, 2)                                            = -7;
  t1(3, 1)                                            = -1;
  t2(0, 0)                                            = 1;
  t2(1, 1)                                            = 4;
  t2(0, 1)                                            = 2;
  t2(1, 0)                                            = 2;
  Table<2, double> t3                                 = Table<2, double>(4, 3);
  Table<2, double> t4                                 = Table<2, double>(2, 2);
  t3(0, 0)                                            = 0.5;
  t3(1, 1)                                            = 1;
  t3(0, 1)                                            = 3;
  t3(1, 0)                                            = 3.4;
  t3(0, 2)                                            = 2;
  t3(1, 2)                                            = -0.4;
  t3(2, 0)                                            = 2;
  t3(2, 1)                                            = -0.5;
  t3(2, 2)                                            = 0;
  t4(0, 0)                                            = 1;
  t4(1, 1)                                            = 0;
  t4(0, 1)                                            = 1;
  t4(1, 0)                                            = 1;
  std::array<Table<2, double>, 2>              kp1    = {t1, t2};
  std::array<Table<2, double>, 2>              kp2    = {t3, t4};
  std::vector<std::array<Table<2, double>, 2>> mat1   = {kp1, kp2};
  std::vector<std::array<Table<2, double>, 2>> approx = {kp1, kp1};
  compute_ksvd<double>(mat1, approx);
  EXPECT_TRUE(Tensors::sum(Tensors::kronecker_product(approx[0][0], approx[0][1]),
                           Tensors::kronecker_product(approx[1][0], approx[1][1])) ==
              Tensors::sum(Tensors::kronecker_product(t1, t2), Tensors::kronecker_product(t3, t4)))
    << "The rank two Kronecker SVD of the matrix does not match the matrix";
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
  if(VectorizedArray<double>::n_array_elements > 2)
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
  if(VectorizedArray<double>::n_array_elements > 2)
    t8(0, 0)[3] = -200;
  t8(1, 0) = make_vectorized_array<double>(1);
  std::array<Table<2, VectorizedArray<double>>, 2>              kp3  = {t5, t6};
  std::array<Table<2, VectorizedArray<double>>, 2>              kp4  = {t7, t8};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 2>> mat2 = {kp3, kp4};

  Table<2, VectorizedArray<double>>                app_t5 = Table<2, VectorizedArray<double>>(2, 3);
  Table<2, VectorizedArray<double>>                app_t6 = Table<2, VectorizedArray<double>>(2, 2);
  std::array<Table<2, VectorizedArray<double>>, 2> app_kp3 = {app_t5, app_t6};
  Table<2, VectorizedArray<double>>                app_t7 = Table<2, VectorizedArray<double>>(2, 3);
  Table<2, VectorizedArray<double>>                app_t8 = Table<2, VectorizedArray<double>>(2, 2);
  std::array<Table<2, VectorizedArray<double>>, 2> app_kp4              = {app_t7, app_t8};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 2>> approx2 = {app_kp3, app_kp4};
  compute_ksvd<VectorizedArray<double>>(mat2, approx2);

  EXPECT_TRUE(Tensors::sum(Tensors::kronecker_product(approx2[0][0], approx2[0][1]),
                           Tensors::kronecker_product(approx2[1][0], approx2[1][1])) ==
              Tensors::sum(Tensors::kronecker_product(t5, t6), Tensors::kronecker_product(t7, t8)))
    << "The rank two Kronecker SVD of the vectorized matrix does not match the matrix";
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

  Table<2, double> t1                                 = Table<2, double>(4, 3);
  Table<2, double> t2                                 = Table<2, double>(2, 2);
  t1(0, 0)                                            = 1;
  t1(1, 1)                                            = 1;
  t1(0, 1)                                            = 1.7;
  t1(1, 0)                                            = 0;
  t1(0, 2)                                            = 1;
  t1(1, 2)                                            = 0.1;
  t1(2, 0)                                            = 1;
  t1(2, 1)                                            = 0.1;
  t1(2, 2)                                            = -7;
  t1(3, 1)                                            = -1;
  t2(0, 0)                                            = 1;
  t2(1, 1)                                            = 4;
  t2(0, 1)                                            = 2;
  t2(1, 0)                                            = 200;
  std::array<Table<2, double>, 2>              kp1    = {t1, t2};
  std::vector<std::array<Table<2, double>, 2>> mat1   = {kp1};
  std::vector<std::array<Table<2, double>, 2>> approx = {kp1};
  compute_ksvd<double>(mat1, approx);

  EXPECT_TRUE(Tensors::kronecker_product(approx[0][1], approx[0][0]) ==
              Tensors::kronecker_product(t2, t1))
    << "The rank one Kronecker SVD of the matrix does not match the matrix";
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
  EXPECT_TRUE(std::abs(approx[0][1](0, 0) - 0.710868) <
              1e-6); // .710868 was generated by working python program, error would be smaller for
  // more digits
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
  EXPECT_TRUE(std::abs(approx[0][1](0, 0)[1] + 0.422376) <
              1e-6); // .710868 was generated by working python program, error would be smaller for
  // more digits
  EXPECT_TRUE(std::abs(approx[0][1](1, 1)[1] + 1.250524) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](2, 2)[1] - 5.699786) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][1](3, 3)[1] + 1.457304) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](0, 0)[1] - 1.097814) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](1, 1)[1] - 0.618112) < 1e-6);
  EXPECT_TRUE(std::abs(approx[0][0](2, 2)[1] + 5.896291) < 1e-6);
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
  Table<2, double>                             t5   = matrix_scaling(t3, 0.8);
  std::array<Table<2, double>, 3>              kp1  = {t1, t2, t2};
  std::array<Table<2, double>, 3>              kp2  = {t3, t4, t4};
  std::array<Table<2, double>, 3>              kp3  = {t5, t4, t4};
  std::vector<std::array<Table<2, double>, 3>> mat1 = {
    kp1, kp2, kp3}; // Since the initial values are chosen differently for the CP-ALS algorithm we
                    // start with a Rank-3 representation of a Kronecker-rank 2 matrix
  std::vector<std::array<Table<2, double>, 3>> approx = {kp1, kp1};
  compute_kcp<double>(mat1, approx);
  Table<2, double> approximate_matrix =
    Tensors::sum(Tensors::kronecker_product(approx[0][0],
                                            Tensors::kronecker_product(approx[0][1], approx[0][2])),
                 Tensors::kronecker_product(
                   approx[1][0], Tensors::kronecker_product(approx[1][1], approx[1][2])));
  Table<2, double> original_matrix =
    Tensors::sum(Tensors::kronecker_product(t1, Tensors::kronecker_product(t2, t2)),
                 Tensors::sum(Tensors::kronecker_product(t3, Tensors::kronecker_product(t4, t4)),
                              Tensors::kronecker_product(t5, Tensors::kronecker_product(t4, t4))));
  EXPECT_TRUE(approximate_matrix == original_matrix);
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
  Table<2, VectorizedArray<double>> t5 = matrix_scaling(t3, make_vectorized_array(0.8));
  std::array<Table<2, VectorizedArray<double>>, 3>              kp1  = {t1, t2, t2};
  std::array<Table<2, VectorizedArray<double>>, 3>              kp2  = {t3, t4, t4};
  std::array<Table<2, VectorizedArray<double>>, 3>              kp3  = {t5, t4, t4};
  std::vector<std::array<Table<2, VectorizedArray<double>>, 3>> mat1 = {
    kp1, kp2, kp3}; // Since the initial values are chosen differently for the CP-ALS algorithm we
                    // start with a Rank-3 representation of a Kronecker-rank 2 matrix
  std::vector<std::array<Table<2, VectorizedArray<double>>, 3>> approx = {kp1, kp1};
  compute_kcp<VectorizedArray<double>>(mat1, approx);
  Table<2, VectorizedArray<double>> approximate_matrix =
    Tensors::sum(Tensors::kronecker_product(approx[0][0],
                                            Tensors::kronecker_product(approx[0][1], approx[0][2])),
                 Tensors::kronecker_product(
                   approx[1][0], Tensors::kronecker_product(approx[1][1], approx[1][2])));
  Table<2, VectorizedArray<double>> original_matrix =
    Tensors::sum(Tensors::kronecker_product(t1, Tensors::kronecker_product(t2, t2)),
                 Tensors::sum(Tensors::kronecker_product(t3, Tensors::kronecker_product(t4, t4)),
                              Tensors::kronecker_product(t5, Tensors::kronecker_product(t4, t4))));
  EXPECT_TRUE(approximate_matrix == original_matrix);
}

TEST(KroneckerSVD, Diag)
{
  test_diagonal_kronecker_svd();
}
TEST(KroneckerSVD, DiagVectorized)
{
  test_diagonal_vectorized_kronecker_svd();
}
TEST(KroneckerSVD, RankOne)
{
  test_rank_one_kronecker_svd();
}
TEST(KroneckerSVD, RankTwo)
{
  test_rank_two_kronecker_svd();
}
TEST(KroneckerSVD, RankTwoVectorized)
{
  test_rank_two_vectorized_kronecker_svd();
}
TEST(KroneckerSVD, ThreeDim)
{
  test_three_dim_kronecker_cp();
}
TEST(KroneckerSVD, ThreeDimVectorized)
{
  test_three_dim_kronecker_cp_vectorized();
}