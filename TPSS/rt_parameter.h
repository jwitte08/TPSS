
/*
 * gather all runtime parameters
 *
 * - MeshParameter
 * - ...
 *
 *  Created on: Oct 08, 2019
 *      Author: witte
 */

#ifndef RT_PARAMETER_H
#define RT_PARAMETER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include "mesh.h"
#include "multigrid.h"
#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"
#include "utilities.h"

using namespace dealii;

struct SolverParameter
{
  enum class PreconditionVariant
  {
    None,
    GMG
  };
  static std::string
  str_precondition_variant(const PreconditionVariant variant);

  std::string         variant              = "none"; // see SolverSelector
  double              abs_tolerance        = 1.e-12;
  double              rel_tolerance        = -1.;
  int                 n_iterations_max     = 100;
  PreconditionVariant precondition_variant = PreconditionVariant::None;

  std::string
  to_string() const;
};

namespace RT
{
struct Parameter
{
  MeshParameter   mesh;
  SolverParameter solver;
  MGParameter     multigrid;
  int             n_cycles = 0;

  std::string
  to_string() const;
};
} // namespace RT

// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
SolverParameter::str_precondition_variant(const PreconditionVariant variant)
{
  const std::string str_variant[] = {"None", "GMG"};
  return str_variant[(int)variant];
}

std::string
SolverParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Solver:", variant);
  oss << Util::parameter_to_fstring("Absolute tolerance:", abs_tolerance);
  oss << Util::parameter_to_fstring("Relative tolerance:", rel_tolerance);
  oss << Util::parameter_to_fstring("Number of maximal iterations:", n_iterations_max);
  oss << Util::parameter_to_fstring("Preconditioner:",
                                    str_precondition_variant(precondition_variant));
  return oss.str();
}

namespace RT
{
std::string
Parameter::to_string() const
{
  std::ostringstream oss;
  oss << mesh.to_string() << std::endl;
  oss << solver.to_string() << std::endl;
  if(solver.precondition_variant == SolverParameter::PreconditionVariant::GMG)
    oss << multigrid.to_string() << std::endl;
  return oss.str();
}
} // namespace RT

#endif /* RT_PARAMETER_H */
