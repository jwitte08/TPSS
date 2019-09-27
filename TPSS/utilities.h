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
  oss << std::endl;
  return oss.str();
}
} // end namespace Util

#endif /* UTILITIES_H_ */
