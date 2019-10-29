
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

  SolverVariant solver_variant   = SolverVariant::None;
  double        accuracy         = 1.e-4;
  std::string   iterative_solver = "none"; // see SolverSelector

  std::string
  to_string() const;
};

struct MGParameter
{
  std::string         cycle_variant = "V-cycle"; // TODO use enum
  SmootherParameter   pre_smoother;
  SmootherParameter   post_smoother;
  CoarseGridParameter coarse_grid;
  int                 coarse_level = 0;

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
CoarseGridParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Coarse grid solver:", str_solver_variant(solver_variant));
  if(solver_variant == SolverVariant::IterativeAcc)
  {
    oss << Util::parameter_to_fstring("Iterative solver:", iterative_solver);
    oss << Util::parameter_to_fstring("Accuracy:", accuracy);
  }
  return oss.str();
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
  oss << Util::parameter_to_fstring("/// Pre-smoother", "");
  oss << pre_smoother.to_string();
  oss << Util::parameter_to_fstring("/// Post-smoother", "");
  oss << post_smoother.to_string();
  oss << Util::parameter_to_fstring("/// Coarse grid solver", "");
  oss << coarse_grid.to_string();
  return oss.str();
}

#endif /* MULTIGRID_H */
