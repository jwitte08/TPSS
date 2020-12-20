/**
 * utilites.h
 *
 * collection of helper functions
 *
 *  Created on: Sep 26, 2019
 *      Author: witte
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>


#include "git_version.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"


#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>



using namespace dealii;

namespace Util
{
template<typename T>
std::string
parameter_to_fstring(const std::string & description, const T parameter)
{
  AssertIndexRange(description.size(), 43);
  std::ostringstream oss;
  print_row_variable(oss, 2, "", 43, description, parameter);
  return oss.str();
}



std::string
git_version_to_fstring()
{
  std::ostringstream oss;
  oss << parameter_to_fstring("Git - deal.II version: ", DEAL_II_GIT_SHORTREV);
  oss << parameter_to_fstring("Git - deal.II branch: ", DEAL_II_GIT_BRANCH);
  oss << parameter_to_fstring("Git - TPSS version: ", GIT_COMMIT_HASH);
  oss << parameter_to_fstring("Git - TPSS branch: ", GIT_BRANCH);
  return oss.str();
}



std::string
generic_info_to_fstring()
{
  std::ostringstream oss;
  oss << Util::git_version_to_fstring();
  oss << Util::parameter_to_fstring("Date:", Utilities::System::get_date());
  oss << Util::parameter_to_fstring("Number of MPI processes:",
                                    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  oss << Util::parameter_to_fstring("Number of threads per MPI proc:",
                                    MultithreadInfo::n_threads());
  oss << Util::parameter_to_fstring("Vectorization level:",
                                    Utilities::System::get_current_vectorization_level());
  const auto size_of_global_dof_index = sizeof(types::global_dof_index{0});
  oss << Util::parameter_to_fstring("Size of global_dof_index (bits):",
                                    8 * size_of_global_dof_index);
  return oss.str();
}



static constexpr char const * skipper = "o";

std::vector<char const *>
args_to_strings(const int argc_in, char * argv_in[])
{
  std::vector<char const *> tmp;
  std::copy_n(argv_in, argc_in, std::back_inserter(tmp));
  return tmp;
}

struct ConditionalAtoi
{
  ConditionalAtoi(const int argc_in, char * argv_in[]) : argv(args_to_strings(argc_in, argv_in))
  {
  }

  template<typename T>
  void
  operator()(T & prm, const std::size_t index)
  {
    if(argv.size() <= index)
      return;
    if(std::strcmp(argv[index], skipper) == 0)
      return;
    prm = std::atoi(argv[index]);
  }

  std::vector<char const *> argv;
};

struct ConditionalAtof
{
  ConditionalAtof(const int argc_in, char * argv_in[]) : argv(args_to_strings(argc_in, argv_in))
  {
  }

  template<typename T>
  void
  operator()(T & prm, const std::size_t index)
  {
    if(argv.size() <= index)
      return;
    if(std::strcmp(argv[index], skipper) == 0)
      return;
    prm = std::atof(argv[index]);
  }

  std::vector<char const *> argv;
};



constexpr unsigned long long
pow(const unsigned int base, const int iexp)
{
  // The "exponentiation by squaring" algorithm used below has to be
  // compressed to one statement due to C++11's restrictions on constexpr
  // functions. A more descriptive version would be:
  //
  // <code>
  // if (iexp <= 0)
  //   return 1;
  //
  // // if the current exponent is not divisible by two,
  // // we need to account for that.
  // const unsigned int prefactor = (iexp % 2 == 1) ? base : 1;
  //
  // // a^b = (a*a)^(b/2)      for b even
  // // a^b = a*(a*a)^((b-1)/2 for b odd
  // return prefactor * ::Utilities::pow(base*base, iexp/2);
  // </code>

  return iexp <= 0 ? 1 : (((iexp % 2 == 1) ? base : 1) * ::Util::pow(base * base, iexp / 2));
}



std::string
si_metric_prefix(unsigned long long measurement)
{
  std::array<std::string, 8> prefixes = {"", "k", "M", "G", "T", "P", "E", "Z"};
  std::ostringstream         oss;

  constexpr std::size_t base = 1000;
  std::size_t           iexp = 0;
  unsigned long long    div  = measurement;
  while(!(div < 1000))
    div = measurement / Util::pow(base, ++iexp);

  oss << div << prefixes[iexp];
  return oss.str();
}



std::string
damping_to_fstring(double factor)
{
  std::ostringstream oss;
  oss << factor;
  return oss.str();
}



std::string
short_name(const std::string & str_in)
{
  std::string sname = str_in.substr(0, 4);
  std::transform(sname.begin(), sname.end(), sname.begin(), [](auto c) { return std::tolower(c); });
  return sname;
}



template<typename MatrixType,
         typename VectorType = LinearAlgebra::distributed::Vector<typename MatrixType::value_type>>
struct MatrixWrapper
{
  using value_type  = typename MatrixType::value_type;
  using vector_type = VectorType;

  MatrixWrapper(const MatrixType & matrix_in) : matrix(matrix_in)
  {
  }

  types::global_dof_index
  m() const
  {
    return matrix.m();
  }

  types::global_dof_index
  n() const
  {
    return matrix.n();
  }

  void
  vmult(const ArrayView<value_type> dst_view, const ArrayView<const value_type> src_view) const
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1U,
                ExcMessage("No MPI support"));
    vector_type dst(dst_view.size());
    vector_type src(src_view.size());

    std::copy(src_view.cbegin(), src_view.cend(), src.begin());
    matrix.vmult(dst, src);
    std::copy(dst.begin(), dst.end(), dst_view.begin());
  }

  FullMatrix<value_type>
  as_fullmatrix()
  {
    return table_to_fullmatrix(Tensors::matrix_to_table(*this));
  }

  const MatrixType & matrix;
};

} // end namespace Util

#endif /* UTILITIES_H_ */
