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

#include "git_version.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

#include <iostream>

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
  oss << Util::parameter_to_fstring("Vectorization level:",
                                    Utilities::System::get_current_vectorization_level());
  return oss.str();
}

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

} // end namespace Util

#endif /* UTILITIES_H_ */
