/*
 * generic_functionalities.h
 *
 *  Created on: Aug 11, 2018
 *      Author: witte
 */

#ifndef TESTS_GENERICFUNCTIONALITIES_H_
#define TESTS_GENERICFUNCTIONALITIES_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <bitset>
#include <iomanip>
#include <string>
#include <type_traits>

#include "vectorization.h"


using namespace dealii;


/// TODO move to alignedlinalg.h ...
template<typename Number>
std::vector<std::complex<Number>>
compute_eigenvalues(LAPACKFullMatrix<Number> & matrix)
{
  AssertDimension(matrix.m(), matrix.n());
  matrix.compute_eigenvalues();
  std::vector<std::complex<Number>> eigenvalues(matrix.m());
  for(auto i = 0U; i < eigenvalues.size(); ++i)
    eigenvalues[i] = matrix.eigenvalue(i);
  std::sort(eigenvalues.begin(), eigenvalues.end(), [](const auto & a, const auto & b) {
    return a.real() < b.real();
  });
  std::reverse(eigenvalues.begin(), eigenvalues.end());
  return eigenvalues;
}



/// TODO move to alignedlinalg.h ...
template<typename Number>
std::vector<std::complex<Number>>
compute_eigenvalues(const FullMatrix<Number> & matrix)
{
  AssertDimension(matrix.m(), matrix.n());
  LAPACKFullMatrix<Number> lapack_matrix(matrix.m());
  lapack_matrix = matrix;
  return compute_eigenvalues(lapack_matrix);
}



/// TODO move to alignedlinalg.h ...
template<typename Number>
Vector<Number>
compute_eigenvalues_symm(LAPACKFullMatrix<Number> & matrix, FullMatrix<Number> & Q)
{
  AssertDimension(matrix.m(), matrix.n());
  Vector<Number> eigenvalues;
  matrix.compute_eigenvalues_symmetric(std::numeric_limits<Number>::lowest(),
                                       std::numeric_limits<Number>::max(),
                                       2. * std::numeric_limits<Number>::epsilon(),
                                       eigenvalues,
                                       Q);
  return eigenvalues;
}



/// TODO move to alignedlinalg.h ...
template<typename Number>
Vector<Number>
compute_eigenvalues_symm(const FullMatrix<Number> & matrix, FullMatrix<Number> & Q)
{
  LAPACKFullMatrix<Number> lapack_matrix(matrix.m());
  lapack_matrix = matrix;
  return compute_eigenvalues_symm(lapack_matrix, Q);
}



/// TODO move to alignedlinalg.h ...
/*
 * Compute generalized eigenvalues and eigenvectors given the generalized
 * eigenvalue problem A v = \lambda B v. Each column of Q represents an
 * eigenvector, in the order of the returned eigenvalues.
 */
template<typename Number>
std::vector<Number>
compute_generalized_eigenvalues_symm(LAPACKFullMatrix<Number> & A,
                                     LAPACKFullMatrix<Number> & B,
                                     FullMatrix<Number> &       Q)
{
  AssertDimension(A.m(), A.n());
  AssertDimension(B.m(), B.n());
  AssertDimension(A.m(), B.n());
  std::vector<Vector<Number>> eigenvectors(A.m(), Vector<Number>(A.m()));
  A.compute_generalized_eigenvalues_symmetric(B, eigenvectors);
  Q.reinit(eigenvectors.front().size(), eigenvectors.size());
  for(auto i = 0U; i < Q.m(); ++i)
    for(auto j = 0U; j < Q.n(); ++j)
      Q(i, j) = eigenvectors[j][i];
  std::vector<Number> eigenvalues(eigenvectors.size());
  for(auto i = 0U; i < eigenvalues.size(); ++i)
    eigenvalues[i] = A.eigenvalue(i).real();
  return eigenvalues;
}



/// TODO move to alignedlinalg.h ...
template<typename Number>
std::vector<Number>
compute_generalized_eigenvalues_symm(const FullMatrix<Number> & A,
                                     const FullMatrix<Number> & B,
                                     FullMatrix<Number> &       Q)
{
  LAPACKFullMatrix<Number> A_lapack(A.m(), A.n());
  A_lapack = A;
  LAPACKFullMatrix<Number> B_lapack(B.m(), B.n());
  B_lapack = B;
  return compute_generalized_eigenvalues_symm(A_lapack, B_lapack, Q);
}



/// TODO move to alignedlinalg.h ...
template<typename Number>
std::vector<Number>
compute_singular_values(LAPACKFullMatrix<Number> & matrix)
{
  AssertDimension(matrix.m(), matrix.n());
  matrix.compute_svd();
  std::vector<Number> singular_values(matrix.m());
  for(auto i = 0U; i < singular_values.size(); ++i)
    singular_values[i] = matrix.singular_value(i);
  std::sort(singular_values.begin(), singular_values.end());
  std::reverse(singular_values.begin(), singular_values.end());
  return singular_values;
}



/// TODO move to alignedlinalg.h ...
template<typename Number>
std::vector<Number>
compute_singular_values(const FullMatrix<Number> & matrix)
{
  AssertDimension(matrix.m(), matrix.n());
  LAPACKFullMatrix<Number> lapack_matrix(matrix.m());
  lapack_matrix = matrix;
  return compute_singular_values(lapack_matrix);
}



/// TODO move to alignedlinalg.h ...
/**
 * Computes and returns the singular value decomposition A = U Sigma VT as a
 * triplet of matrices [U, Sigma, VT]. Note, that the input matrix @p matrix is
 * modified by compute_svd().
 */
template<typename Number>
std::array<LAPACKFullMatrix<Number>, 3>
compute_svd(LAPACKFullMatrix<Number> & matrix)
{
  matrix.compute_svd();

  std::array<LAPACKFullMatrix<Number>, 3> svd;
  auto &                                  U     = svd[0];
  auto &                                  Sigma = svd[1];
  Sigma.reinit(matrix.m(), matrix.n());
  auto & VT = svd[2];

  U                                    = matrix.get_svd_u();
  VT                                   = matrix.get_svd_vt();
  const unsigned int n_singular_values = std::min(matrix.m(), matrix.n());
  for(auto i = 0U; i < n_singular_values; ++i)
    Sigma(i, i) = matrix.singular_value(i);

  return svd;
}



/// TODO move to alignedlinalg.h ...
/**
 * Computes the singular value decomposition A = U Sigma VT and returns the
 * inverse triplet of matrices [UT, Sigma^{-1}, V]. If singular values are below
 * the given threshold @p threshold their reciprocal is set to zero, which
 * avoids a division by (nearly) zero. Note, that the input matrix @p matrix is
 * modified by compute_svd().
 */
template<typename Number>
std::array<LAPACKFullMatrix<Number>, 3>
compute_inverse_svd(LAPACKFullMatrix<Number> & matrix,
                    const Number               threshold = zero_threshold<Number>)
{
  matrix.compute_svd();

  std::array<LAPACKFullMatrix<Number>, 3> svd;
  auto &                                  V = svd[0];
  V.reinit(matrix.n());
  auto & invSigma = svd[1];
  invSigma.reinit(matrix.n(), matrix.m());
  auto & UT = svd[2];
  UT.reinit(matrix.m());

  matrix.get_svd_u().transpose(UT);
  matrix.get_svd_vt().transpose(V);
  const unsigned int n_singular_values = std::min(matrix.m(), matrix.n());
  for(auto i = 0U; i < n_singular_values; ++i)
    invSigma(i, i) = inverse_scalar_if_impl(matrix.singular_value(i), threshold);

  return svd;
}



/// TODO move to alignedlinalg.h ...
/**
 * Computes and returns the triple matrix-matrix-product X*Y*Z.
 */
template<typename Number>
LAPACKFullMatrix<Number>
merge_lapack_decomposition(const LAPACKFullMatrix<Number> & X,
                           const LAPACKFullMatrix<Number> & Y,
                           const LAPACKFullMatrix<Number> & Z)
{
  LAPACKFullMatrix<double> YZ(Y.m(), Z.n());
  LAPACKFullMatrix<double> XYZ(X.m(), Z.n());
  Y.mmult(YZ, Z);
  X.mmult(XYZ, YZ);
  return XYZ;
}



/// TODO move to alignedlinalg.h ...
/**
 * Merges a matrix factorization of the form
 *
 *    U M V^T
 *
 * for default template parameters. If is_reverse is true,
 *
 *    U^T M V
 *
 * is computed instead.
 */
template<typename Number, bool is_reverse = false>
FullMatrix<Number>
merge_decomposition_impl(const FullMatrix<Number> & U,
                         const FullMatrix<Number> & M,
                         const FullMatrix<Number> & V)
{
  FullMatrix<Number> MVT_or_MV(M.m(), is_reverse ? V.n() : V.m());
  if(is_reverse) // M V
    M.mmult(MVT_or_MV, V);
  else // M V^T
    M.mTmult(MVT_or_MV, V);

  FullMatrix<Number> UMVT_or_UTMV(is_reverse ? U.n() : U.m(), MVT_or_MV.n());
  if(is_reverse) // U^T M V
    U.Tmmult(UMVT_or_UTMV, MVT_or_MV);
  else // U M V^T
    U.mmult(UMVT_or_UTMV, MVT_or_MV);

  return UMVT_or_UTMV;
}


/// TODO move to alignedlinalg.h ...
/// computes U M V^T
template<typename Number>
FullMatrix<Number>
merge_decomposition(const FullMatrix<Number> & U,
                    const FullMatrix<Number> & M,
                    const FullMatrix<Number> & V)
{
  return merge_decomposition_impl(U, M, V);
}



/// TODO move to alignedlinalg.h ...
/// computes U^T M V
template<typename Number>
FullMatrix<Number>
merge_reverse_decomposition(const FullMatrix<Number> & U,
                            const FullMatrix<Number> & M,
                            const FullMatrix<Number> & V)
{
  return merge_decomposition_impl<Number, true>(U, M, V);
}



template<typename T1, typename T2>
std::ostream &
operator<<(std::ostream & os, const std::pair<T1, T2> & pair)
{
  return os << "(" << pair.first << ", " << pair.second << ")";
}



std::string
bool_to_str(const bool b)
{
  return std::string(b ? "true" : "false");
}



Utilities::MPI::MinMaxAvg
operator/(const Utilities::MPI::MinMaxAvg & mma_in, const double t)
{
  Utilities::MPI::MinMaxAvg mma;
  mma.min = mma_in.min / t;
  mma.max = mma_in.max / t;
  mma.avg = mma_in.avg / t;
  mma.sum = mma_in.sum / t;
  return mma;
}



template<typename Number = double>
Number
make_random_value()
{
  return static_cast<Number>(rand()) / RAND_MAX;
}



template<>
VectorizedArray<double>
make_random_value()
{
  VectorizedArray<double> value;
  for(auto lane = 0U; lane < VectorizedArray<double>::size(); ++lane)
    value[lane] = make_random_value<double>();
  return value;
}



template<typename VectorType>
void
fill_with_random_values(VectorType & vec)
{
  for(auto it = vec.begin(); it != vec.end(); ++it)
    *it = make_random_value();
}



template<typename ElementType>
void
fill_with_random_values(ArrayView<ElementType> view)
{
  for(auto it = view.begin(); it != view.end(); ++it)
    *it = make_random_value<ElementType>();
}



template<typename Number>
void
fill_with_random_values(Table<2, Number> & table)
{
  for(auto i = 0U; i < table.size(0); ++i)
    for(auto j = 0U; j < table.size(1); ++j)
      table(i, j) = make_random_value<Number>();
}



template<typename MatrixType, typename Number = typename MatrixType::value_type>
void
fill_matrix_with_random_values(MatrixType & matrix)
{
  fill_matrix_with_random_values<MatrixType, Number>(matrix, matrix.m(), matrix.n());
}



template<typename MatrixType, typename Number = typename MatrixType::value_type>
void
fill_matrix_with_random_values(MatrixType &       matrix,
                               const unsigned int n_rows,
                               const unsigned int n_cols)
{
  matrix.reinit(n_rows, n_cols);
  for(unsigned int i = 0; i < n_rows; ++i)
    for(unsigned int j = 0; j < n_cols; ++j)
      matrix(i, j) = make_random_value<Number>();
}



template<typename Number>
Vector<Number>
array_view_to_vector(const ArrayView<const Number> & view, const unsigned int dummy = 0)
{
  (void)dummy;
  return Vector<Number>(view.cbegin(), view.cend());
}



template<typename Number>
Vector<Number>
array_view_to_vector(const ArrayView<const VectorizedArray<Number>> & view,
                     const unsigned int                               lane = 0)
{
  AssertIndexRange(lane, VectorizedArray<Number>::size());
  Vector<Number> vec(view.size());
  std::transform(view.cbegin(), view.cend(), vec.begin(), [lane](const auto & elem) {
    return elem[lane];
  });
  return vec;
}



template<typename Number>
Vector<Number>
array_view_to_vector(const ArrayView<VectorizedArray<Number>> & view, const unsigned int lane = 0)
{
  ArrayView<const VectorizedArray<Number>> cview(view.begin(), view.size());
  return array_view_to_vector(cview, lane);
}



template<typename T>
std::string
vector_to_string(const std::vector<T> & vector)
{
  if(vector.empty())
    return "[]";

  std::ostringstream oss;
  oss << std::scientific << std::setprecision(4);
  oss << "[";
  for(unsigned i = 0; i < vector.size(); ++i)
    oss << vector[i] << ((i + 1) < vector.size() ? ", " : "]");
  return oss.str();
}



template<typename E>
std::vector<const E *>
to_vector_of_ptrs(const std::vector<E> & vec)
{
  std::vector<const E *> vec_of_ptrs;
  std::transform(vec.cbegin(), vec.cend(), std::back_inserter(vec_of_ptrs), [](const auto & elem) {
    return &elem;
  });
  return vec_of_ptrs;
}



template<typename T>
std::string
set_to_string(const std::set<T> & set)
{
  std::vector<T> set_as_vector(set.cbegin(), set.cend());
  return vector_to_string(set_as_vector);
}



class NullStreambuf : public std::streambuf
{
  char dummyBuffer[64];

protected:
  virtual int
  overflow(int c)
  {
    setp(dummyBuffer, dummyBuffer + sizeof(dummyBuffer));
    return (c == traits_type::eof()) ? '\0' : c;
  }
};



class NullOStream : private NullStreambuf, public std::ostream
{
public:
  NullOStream() : std::ostream(this)
  {
  }

  NullStreambuf *
  rdbuf()
  {
    return this;
  }
};



template<typename FloatType>
std::string
to_string_conditional(const FloatType value)
{
  static_assert(std::is_floating_point<FloatType>::value, "FloatType is not floating point type.");
  std::ostringstream ostream;
  if(1.e-3 < value && value < 1.e+3)
    ostream << std::defaultfloat << value;
  else
    ostream << std::scientific << std::setprecision(3) << value;

  return ostream.str();
}



std::string
extend_string(std::string label, const std::size_t size)
{
  AssertThrow(label.length() < size + 1, ExcMessage("String length exceeds column size!"));
  const auto remaining_size = size - label.length();
  for(unsigned b = 0; b < remaining_size; ++b)
    label += " ";
  return label;
}



template<typename Number>
std::string
fstring_column_default(const Number value, const unsigned size)
{
  std::ostringstream ostream;
  ostream << value;
  return (extend_string(ostream.str(), size));
}



std::string
fstring_column(const double value, const unsigned size)
{
  auto str{std::move(to_string_conditional(value))};
  return (extend_string(str, size));
}



std::string
fstring_column(const float value, const unsigned size)
{
  auto str{std::move(to_string_conditional(value))};
  return (extend_string(str, size));
}



std::string
fstring_column(const int value, const unsigned size)
{
  return fstring_column_default<int>(value, size);
}



std::string
fstring_column(const unsigned int value, const unsigned size)
{
  return fstring_column_default<unsigned int>(value, size);
}



std::string
fstring_column(const long unsigned int value, const unsigned size)
{
  return fstring_column_default<long unsigned int>(value, size);
}



std::string
fstring_column(const long long unsigned int value, const unsigned size)
{
  return fstring_column_default<long long unsigned int>(value, size);
}



std::string
fstring_column(const std::string value, const unsigned size)
{
  return fstring_column_default<std::string>(value, size);
}



std::string
fstring_column(const char value[], const unsigned size)
{
  return fstring_column_default<std::string>(value, size);
}



std::string
fstring_column(const bool value, const unsigned size)
{
  std::ostringstream ostream;
  ostream << std::boolalpha << value;
  return (extend_string(ostream.str(), size));
}



template<typename T>
std::string
fstring_column(const std::vector<T> & vec_value, const unsigned size)
{
  std::ostringstream ostrstream;
  for(const auto value : vec_value)
    ostrstream << fstring_column(value, size);
  return ostrstream.str();
}



template<typename OSTREAM>
struct Printer
{
  template<typename Arg>
  void
  print_row_impl(OSTREAM & out, const unsigned, Arg && arg)
  {
    std::ostringstream ostream;
    ostream << std::forward<Arg>(arg);
    out << fstring_column(std::forward<Arg>(arg), ostream.str().length() + 4 + 3 + 1) << std::endl;
  }

  template<typename Arg, typename... Args>
  void
  print_row_impl(OSTREAM & out, const unsigned size, Arg && arg, Args &&... args)
  {
    out << fstring_column(std::forward<Arg>(arg), size);
    print_row_impl(out, size, args...);
  }

  template<typename Arg>
  void
  print_row_variable_impl(OSTREAM & out, Arg && arg)
  {
    std::ostringstream ostream;
    ostream << std::forward<Arg>(arg);
    out << fstring_column(std::forward<Arg>(arg), ostream.str().length() + 4 + 3 + 1) << std::endl;
  }

  template<typename Arg>
  void
  print_row_variable_impl(OSTREAM & out, const unsigned size, Arg && arg)
  {
    out << fstring_column(std::forward<Arg>(arg), size);
    out << std::endl;
  }

  template<typename Arg, typename... Args>
  void
  print_row_variable_impl(OSTREAM & out, const unsigned size, Arg && arg, Args &&... args)
  {
    out << fstring_column(std::forward<Arg>(arg), size);
    print_row_variable_impl(out, args...);
  }
};



template<typename Arg>
void
print_row(ConditionalOStream & out, const unsigned size, Arg && arg)
{
  Printer<ConditionalOStream>{}.print_row_impl(out, size, std::forward<Arg>(arg));
}



template<typename... Args>
void
print_row(ConditionalOStream & out, const unsigned size, Args &&... args)
{
  Printer<ConditionalOStream>{}.print_row_impl(out, size, std::forward<Args>(args)...);
}



template<typename Arg>
void
print_row(std::ostream & out, const unsigned size, Arg && arg)
{
  Printer<std::ostream>{}.print_row_impl(out, size, std::forward<Arg>(arg));
}



template<typename... Args>
void
print_row(std::ostream & out, const unsigned size, Args &&... args)
{
  Printer<std::ostream>{}.print_row_impl(out, size, std::forward<Args>(args)...);
}



template<typename Arg>
void
print_row_variable(ConditionalOStream & out, Arg && arg)
{
  Printer<ConditionalOStream>{}.print_row_variable_impl(out, std::forward<Arg>(arg));
}



template<typename Arg>
void
print_row_variable(ConditionalOStream & out, const unsigned size, Arg && arg)
{
  Printer<ConditionalOStream>{}.print_row_variable_impl(out, size, std::forward<Arg>(arg));
}



template<typename... Args>
void
print_row_variable(ConditionalOStream & out, Args &&... args)
{
  Printer<ConditionalOStream>{}.print_row_variable_impl(out, std::forward<Args>(args)...);
}



template<typename Arg>
void
print_row_variable(std::ostream & out, Arg && arg)
{
  Printer<std::ostream>{}.print_row_variable_impl(out, std::forward<Arg>(arg));
}



template<typename Arg>
void
print_row_variable(std::ostream & out, const unsigned size, Arg && arg)
{
  Printer<std::ostream>{}.print_row_variable_impl(out, size, std::forward<Arg>(arg));
}



template<typename... Args>
void
print_row_variable(std::ostream & out, Args &&... args)
{
  Printer<std::ostream>{}.print_row_variable_impl(out, std::forward<Args>(args)...);
}

#endif /* TESTS_GENERICFUNCTIONALITIES_H_ */
