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

#include <iomanip>
#include <string>
#include <type_traits>



using namespace dealii;

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
  for(auto lane = 0U; lane < VectorizedArray<double>::n_array_elements; ++lane)
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
    *it = (double)rand() / RAND_MAX;
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
std::string
varray_to_string(const VectorizedArray<Number> & array)
{
  std::ostringstream osstream;
  osstream << "[";
  constexpr auto n_elems = VectorizedArray<Number>::n_array_elements;
  for(unsigned int k = 0; k < n_elems - 1; ++k)
    osstream << array[k] << "|";
  osstream << array[n_elems - 1] << "]";
  return osstream.str();
}


/**
 * Extracts and converts the matrix associated to the lane @p lane
 * of the vectorized matrix @p table into the FullMatrix format.
 */
template<typename Number>
FullMatrix<Number>
vectorized_table_to_fullmatrix(const Table<2, VectorizedArray<Number>> & table,
                               const unsigned int                        lane = 0)
{
  AssertIndexRange(lane, VectorizedArray<Number>::n_array_elements);
  FullMatrix<Number> matrix{table.n_rows(), table.n_cols()};
  for(unsigned int i = 0; i < table.n_rows(); ++i)
    for(unsigned int j = 0; j < table.n_cols(); ++j)
      matrix(i, j) = (table(i, j))[lane];
  return matrix;
}


template<typename Number>
FullMatrix<Number>
table_to_fullmatrix(const Table<2, VectorizedArray<Number>> & table, const unsigned int lane = 0)
{
  return vectorized_table_to_fullmatrix(table, lane);
}


template<typename Number>
FullMatrix<Number>
table_to_fullmatrix(const Table<2, Number> & table, const unsigned int dummy = 0)
{
  (void)dummy;
  FullMatrix<Number> matrix{table.n_rows(), table.n_cols()};
  for(unsigned int i = 0; i < table.n_rows(); ++i)
    for(unsigned int j = 0; j < table.n_cols(); ++j)
      matrix(i, j) = table(i, j);
  return matrix;
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
  AssertIndexRange(lane, VectorizedArray<Number>::n_array_elements);
  Vector<Number> vec(view.size());
  std::transform(view.cbegin(), view.cend(), vec.begin(), [lane](const auto & elem) {
    return elem[lane];
  });
  return vec;
}

template<typename T>
std::string
vector_to_string(const std::vector<T> & vector)
{
  if(vector.empty())
    return "[]";

  std::ostringstream oss;
  oss << "[";
  for(unsigned i = 0; i < vector.size(); ++i)
    oss << vector[i] << ((i + 1) < vector.size() ? ", " : "]");
  return oss.str();
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
