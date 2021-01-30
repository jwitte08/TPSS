
/*
 * collection of (iterative) solvers
 *
 *  Created on: Dec 17, 2020
 *      Author: witte
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>

using namespace dealii;



struct SolverParameter
{
  enum class PreconditionVariant
  {
    None,
    GMG
  };

  enum class ControlVariant
  {
    absolute,
    relative
  };

  std::string         variant                   = "none"; // see SolverSelector
  ControlVariant      control_variant           = ControlVariant::relative;
  double              abs_tolerance             = 1.e-14;
  double              rel_tolerance             = 1.e-8;
  int                 n_iterations_max          = 100;
  PreconditionVariant precondition_variant      = PreconditionVariant::None;
  bool                use_right_preconditioning = false;

  static std::string
  str_control_variant(const ControlVariant variant);

  static std::string
  str_precondition_variant(const PreconditionVariant variant);

  std::string
  str_control_variant() const;

  std::string
  str_precondition_variant() const;

  std::string
  to_string() const;
};



std::shared_ptr<SolverControl>
get_solver_control_impl(const SolverParameter & solver_prms)
{
  auto solver_control = [&]() -> std::shared_ptr<SolverControl> {
    if(solver_prms.control_variant == SolverParameter::ControlVariant::relative)
    {
      auto control = std::make_shared<ReductionControl>();
      control->set_reduction(solver_prms.rel_tolerance);
      return control;
    }
    else if(solver_prms.control_variant == SolverParameter::ControlVariant::absolute)
      return std::make_shared<SolverControl>();
    else
      AssertThrow(false, ExcMessage("ControlVariant isn't supported."));
    return nullptr;
  }();
  AssertThrow(solver_control, ExcMessage("ControlVariant isn't supported."));
  solver_control->set_max_steps(solver_prms.n_iterations_max);
  solver_control->set_tolerance(solver_prms.abs_tolerance);
  solver_control->log_history(true);
  solver_control->log_result(true);
  solver_control->enable_history_data();

  /// DEBUG
  // IterationNumberControl solver_control;
  // solver_control.set_max_steps(solver_prms.n_iterations_max);
  // solver_control.set_tolerance(solver_prms.abs_tolerance);
  // solver_control.log_history(true);
  // solver_control.log_result(true);
  // solver_control.enable_history_data();

  return solver_control;
}



// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
SolverParameter::str_control_variant(const ControlVariant variant)
{
  const std::string str_variant[] = {"absolute tolerance", "relative tolerance"};
  return str_variant[(int)variant];
}


std::string
SolverParameter::str_precondition_variant(const PreconditionVariant variant)
{
  const std::string str_variant[] = {"None", "GMG"};
  return str_variant[(int)variant];
}


std::string
SolverParameter::str_control_variant() const
{
  return str_control_variant(control_variant);
}


std::string
SolverParameter::str_precondition_variant() const
{
  return str_precondition_variant(precondition_variant);
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
  const bool use_preconditioner_for_gmres =
    variant == "gmres" && precondition_variant != PreconditionVariant::None;
  if(use_preconditioner_for_gmres)
    oss << Util::parameter_to_fstring("Use as right preconditioner?", use_right_preconditioning);
  oss << Util::parameter_to_fstring("Solver control:", str_control_variant(control_variant));
  return oss.str();
}

#endif /* SOLVER_H */
