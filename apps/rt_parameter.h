
/*
 * collection of runtime parameters
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
#include "solver.h"
#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"
#include "utilities.h"



using namespace dealii;



namespace RT
{
static const std::pair<types::global_dof_index, types::global_dof_index> invalid_dof_limits =
  {numbers::invalid_dof_index, numbers::invalid_dof_index};

/**
 * Collection of runtime parameters commonly used by applications.
 */
struct Parameter
{
  std::pair<types::global_dof_index, types::global_dof_index> dof_limits = invalid_dof_limits;
  unsigned int                                                n_cycles   = 0;

  MeshParameter   mesh;
  MGParameter     multigrid;
  SolverParameter solver;

  bool do_visualize = false;
  bool use_tbb      = false;

  bool
  exceeds_dof_limits(const long long unsigned int n_dofs) const;

  void
  reset_solver_variant();

  void
  reset_damping_factor(int dimensions);

  std::string
  to_string() const;

  template<int dim, typename Number>
  void
  fill_schwarz_smoother_data(
    typename SubdomainHandler<dim, Number>::AdditionalData & additional_data,
    const bool                                               is_pre_smoother) const;
};



// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
lookup_solver_variant(const SmootherParameter pre_smoother, const SmootherParameter post_smoother)
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


bool
Parameter::exceeds_dof_limits(const long long unsigned int n_dofs) const
{
  if(dof_limits == std::make_pair(numbers::invalid_dof_index, numbers::invalid_dof_index))
    return false;
  Assert(dof_limits.first < dof_limits.second, ExcMessage("Invalid closed range."));
  const auto [lower_bound, upper_bound] = dof_limits;
  bool exceeds = n_dofs < static_cast<long long unsigned int>(lower_bound) ||
                 static_cast<long long unsigned int>(upper_bound) < n_dofs;
  return exceeds;
}


void
Parameter::reset_solver_variant()
{
  solver.variant = lookup_solver_variant(multigrid.pre_smoother, multigrid.post_smoother);
}


void
Parameter::reset_damping_factor(int dimensions)
{
  const auto reset = [&](auto & schwarz_data) {
    schwarz_data.damping_factor = TPSS::lookup_damping_factor(schwarz_data.patch_variant,
                                                              schwarz_data.smoother_variant,
                                                              dimensions);
  };
  reset(multigrid.pre_smoother.schwarz);
  reset(multigrid.post_smoother.schwarz);
}


template<int dim, typename Number>
void
Parameter::fill_schwarz_smoother_data(
  typename SubdomainHandler<dim, Number>::AdditionalData & additional_data,
  const bool                                               pre_smoother) const
{
  if(pre_smoother)
    ::fill_schwarz_smoother_data<dim, Number>(additional_data, multigrid.pre_smoother.schwarz);
  else // post smoother
    ::fill_schwarz_smoother_data<dim, Number>(additional_data, multigrid.post_smoother.schwarz);
}


std::string
Parameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Number of run cycles:", n_cycles);
  const auto [n_dofs_min, n_dofs_max] = dof_limits;
  if(n_dofs_min != numbers::invalid_dof_index && n_dofs_max != numbers::invalid_dof_index)
  {
    oss << Util::parameter_to_fstring("Lower dof limit:", n_dofs_min);
    oss << Util::parameter_to_fstring("Upper dof limit:", n_dofs_max);
  }
  oss << Util::parameter_to_fstring("Using TBB:", use_tbb);
  oss << std::endl;

  oss << mesh.to_string() << std::endl;
  oss << solver.to_string() << std::endl;
  if(solver.precondition_variant == SolverParameter::PreconditionVariant::GMG)
    oss << multigrid.to_string();

  return oss.str();
}

} // namespace RT

#endif /* RT_PARAMETER_H */
