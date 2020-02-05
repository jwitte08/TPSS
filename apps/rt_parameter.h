
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
  static std::string
  lookup_solver_variant(const SmootherParameter pre_smoother,
                        const SmootherParameter post_smoother);

  std::string         variant              = "none"; // see SolverSelector
  double              abs_tolerance        = 1.e-14;
  double              rel_tolerance        = -1.;
  int                 n_iterations_max     = 100;
  PreconditionVariant precondition_variant = PreconditionVariant::None;

  void
  set_solver_variant(const SmootherParameter pre_smoother, const SmootherParameter post_smoother);

  std::string
  to_string() const;
};



namespace RT
{
struct Parameter
{
  bool                                                        compressed = false;
  std::pair<types::global_dof_index, types::global_dof_index> dof_limits =
    {numbers::invalid_dof_index, numbers::invalid_dof_index};
  MeshParameter   mesh;
  MGParameter     multigrid;
  unsigned int    n_cycles = 0;
  SolverParameter solver;

  bool
  exceeds_dof_limits(const types::global_dof_index n_dofs) const;

  void
  reset_solver_variant()
  {
    solver.variant =
      SolverParameter::lookup_solver_variant(multigrid.pre_smoother, multigrid.post_smoother);
  };

  void
  reset_damping_factor(int dimensions)
  {
    const auto reset = [&](auto & schwarz_data) {
      schwarz_data.damping_factor = TPSS::lookup_damping_factor(schwarz_data.patch_variant,
                                                                schwarz_data.smoother_variant,
                                                                dimensions);
    };
    reset(multigrid.pre_smoother.schwarz);
    reset(multigrid.post_smoother.schwarz);
  }

  void
  set_compressed(const bool flag = true);

  std::string
  to_string() const;
}; // namespace RT
} // namespace RT

// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
SolverParameter::str_precondition_variant(const PreconditionVariant variant)
{
  const std::string str_variant[] = {"None", "GMG"};
  return str_variant[(int)variant];
}


std::string
SolverParameter::lookup_solver_variant(const SmootherParameter pre_smoother,
                                       const SmootherParameter post_smoother)
{
  std::string variant    = "";
  const bool  is_schwarz = pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz &&
                          post_smoother.variant == SmootherParameter::SmootherVariant::Schwarz;
  AssertThrow(is_schwarz, ExcMessage("TODO"));
  if(is_schwarz)
  {
    const bool pre_smoother_is_additive =
      pre_smoother.schwarz.smoother_variant == TPSS::SmootherVariant::additive;
    const bool pre_smoother_is_symmetric =
      pre_smoother_is_additive || pre_smoother.schwarz.symmetrize_smoothing;
    const bool post_smoother_is_additive =
      post_smoother.schwarz.smoother_variant == TPSS::SmootherVariant::additive;
    const bool post_smoother_is_symmetric =
      post_smoother_is_additive || post_smoother.schwarz.symmetrize_smoothing;
    const bool smoothers_are_symmetric = pre_smoother_is_symmetric && post_smoother_is_symmetric;
    const bool vcycle_is_symmetric =
      (pre_smoother.schwarz.reverse_smoothing && !post_smoother.schwarz.reverse_smoothing) ||
      (!pre_smoother.schwarz.reverse_smoothing && post_smoother.schwarz.reverse_smoothing);
    if(smoothers_are_symmetric || vcycle_is_symmetric)
      variant = "cg";
    else
      variant = "gmres";
  }
  return variant;
}


void
SolverParameter::set_solver_variant(const SmootherParameter pre_smoother,
                                    const SmootherParameter post_smoother)
{
  variant = lookup_solver_variant(pre_smoother, post_smoother);
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
bool
Parameter::exceeds_dof_limits(const types::global_dof_index n_dofs) const
{
  if(dof_limits == std::make_pair(numbers::invalid_dof_index, numbers::invalid_dof_index))
    return false;
  Assert(dof_limits.first < dof_limits.second, ExcMessage("Invalid closed range."));
  bool exceeds = n_dofs < dof_limits.first || dof_limits.second < n_dofs;
  return exceeds;
}

void
Parameter::set_compressed(const bool flag)
{
  compressed                         = flag;
  multigrid.pre_smoother.compressed  = flag;
  multigrid.post_smoother.compressed = flag;
}

std::string
Parameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Compressed mode:", compressed);
  oss << Util::parameter_to_fstring("Number of run cycles:", n_cycles);
  oss << std::endl;
  oss << mesh.to_string() << std::endl;
  oss << solver.to_string() << std::endl;
  if(solver.precondition_variant == SolverParameter::PreconditionVariant::GMG)
    oss << multigrid.to_string();
  return oss.str();
}
} // namespace RT

#endif /* RT_PARAMETER_H */
