
/*
 * helpers on multigrid infrastructure in deal.II
 *
 *  Created on: Oct 08, 2019
 *      Author: witte
 */

#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include "mesh.h"
#include "multigrid.h"
#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"
#include "utilities.h"

using namespace dealii;

struct SmootherParameter
{
  enum class SmootherVariant // TODO
  {
    Schwarz
  };
  static std::string
  str_smoother_variant(const SmootherVariant variant);

  SmootherParameter::SmootherVariant variant = SmootherParameter::SmootherVariant::Schwarz;
  int                                n_smoothing_steps = 1;
  SchwarzSmootherData                schwarz;

  std::string
  to_string() const;
};

struct CoarseGridParameter
{
  enum class SolverVariant
  {
    None,
    IterativeAcc
  };
  static std::string
  str_solver_variant(const SolverVariant variant);

  double      accuracy         = 1.e-4;
  std::string iterative_solver = "none"; // see SolverSelector
};

struct MGParameter
{
  std::string       cycle_variant = "V-cycle"; // TODO use enum
  SmootherParameter smoother;
  int               coarse_level              = 0;
  bool              mg_smoother_post_reversed = false;

  std::string
  to_string() const;
};

// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
CoarseGridParameter::str_solver_variant(const CoarseGridParameter::SolverVariant variant)
{
  const std::string str_variant[] = {"None", "Accurate Iterative Solver"};
  return str_variant[(int)variant];
}

std::string
SmootherParameter::str_smoother_variant(const SmootherParameter::SmootherVariant variant)
{
  const std::string str_variant[] = {"Schwarz Smoother"};
  return str_variant[(int)variant];
}

std::string
SmootherParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Number of smoothing steps:", n_smoothing_steps);
  oss << Util::parameter_to_fstring("Smoother:", str_smoother_variant(variant));
  if(variant == SmootherVariant::Schwarz)
  {
    ConditionalOStream pcout(oss, true);
    schwarz.print(pcout);
  }
  return oss.str();
}

std::string
MGParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Multigrid:", cycle_variant);
  oss << smoother.to_string() << std::endl;
  return oss.str();
}

#endif /* MULTIGRID_H */
