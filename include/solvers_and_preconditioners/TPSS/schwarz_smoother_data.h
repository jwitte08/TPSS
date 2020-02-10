/*
 * schwarz_smoother_data.h
 *
 * TODO move SchwarzSmootherData to a convenient file
 *
 *  Created on: Sep 16, 2016
 *      Author: jwitte
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZDATA_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZDATA_H_

#include <string>
#include <vector>

#include "TPSS.h"
#include "generic_functionalities.h"
#include "subdomain_handler.h"

struct SchwarzSmootherData
{
  SchwarzSmootherData()
    : patch_variant(TPSS::PatchVariant::invalid),
      smoother_variant(TPSS::SmootherVariant::invalid),
      number_of_smoothing_steps(1),
      damping_factor(1.0),
      local_damping_factor(1.0),
      manual_coloring(false),
      n_q_points_surrogate(5),
      normalize_surrogate_patch(false),
      use_arc_length(true),
      reverse_smoothing(false),
      symmetrize_smoothing(false),
      caching_strategy(TPSS::CachingStrategy::Cached),
      print_details(false)
  {
  }

  template<typename T>
  void
  print_parameter(ConditionalOStream & cout,
                  const std::string &  description,
                  const T              parameter) const
  {
    print_row_variable(cout, 2, "", 43, description, parameter);
  }

  void
  print(ConditionalOStream & pcout) const
  {
    std::string str_patch_variant[]    = {"Undefined", "Cell-Based", "Vertex Patch"};
    std::string str_smoother_variant[] = {"Undefined", "Additive", "Multiplicative"};
    std::string str_schwarz_operator =
      str_smoother_variant[(int)smoother_variant] + " " + str_patch_variant[(int)patch_variant];

    print_parameter(pcout, "Schwarz operator", str_schwarz_operator);
    print_parameter(pcout, "Number of smoothing steps", number_of_smoothing_steps);
    print_parameter(pcout, "Damping factor", damping_factor);
    print_parameter(pcout, "Patch-local damping factor", local_damping_factor);

    print_parameter(pcout, "Manual coloring", manual_coloring);
    print_parameter(pcout, "Symmetrized Schwarz operator", symmetrize_smoothing);
    print_parameter(pcout, "Reversed Schwarz operator", reverse_smoothing);

    print_parameter(pcout, "Number of quad points (surrogate)", n_q_points_surrogate);
    print_parameter(pcout, "Normalize surrogate patches", normalize_surrogate_patch);
    print_parameter(pcout, "Compute arc length (surrogate)", use_arc_length);

    print_parameter(pcout, "Caching strategy", TPSS::str_caching_strategy(caching_strategy));
  }

  bool
  operator==(const SchwarzSmootherData & other) const
  {
    bool is_equal = true;
    is_equal &= patch_variant == other.patch_variant;
    is_equal &= smoother_variant == other.smoother_variant;
    is_equal &= damping_factor == other.damping_factor;
    is_equal &= local_damping_factor == other.local_damping_factor;
    is_equal &= manual_coloring == other.manual_coloring;
    is_equal &= n_q_points_surrogate == other.n_q_points_surrogate;
    is_equal &= use_arc_length == other.use_arc_length;
    is_equal &= reverse_smoothing == other.reverse_smoothing;
    is_equal &= symmetrize_smoothing == other.symmetrize_smoothing;
    is_equal &= print_details == other.print_details;
    is_equal &= caching_strategy == other.caching_strategy;
    return is_equal;
  }

  // ...
  TPSS::PatchVariant patch_variant;

  // ...
  TPSS::SmootherVariant smoother_variant;

  // number of iterations per smoothing step
  unsigned int number_of_smoothing_steps;

  // damping factor
  double damping_factor;

  // patch-wise damping factor to satisfy the local stability
  double local_damping_factor;

  // TODO ...
  bool manual_coloring;

  // characterizing quadrature to compute surrogate patches
  unsigned int n_q_points_surrogate;

  // normalize the computed average scales of the surrogate patch?
  bool normalize_surrogate_patch;

  // compute the average scaling (to build surrogate patches) in terms
  // of the arc length between opposite faces (otherwise direct
  // distance)
  bool use_arc_length;

  // apply the colors in reverse order
  bool reverse_smoothing;

  // symmetrizes the multiplicative Schwarz preconditioner
  bool symmetrize_smoothing;

  // which patch related data is cached or computed on-the-fly
  TPSS::CachingStrategy caching_strategy;

  // print detailed information regarding the patch distribution, etc.
  bool print_details;
};



template<int dim, typename Number>
void
fill_schwarz_smoother_data(typename SubdomainHandler<dim, Number>::AdditionalData & additional_data,
                           const SchwarzSmootherData & schwarz_smoother_data)
{
  /// Schwarz operator
  additional_data.patch_variant    = schwarz_smoother_data.patch_variant;
  additional_data.smoother_variant = schwarz_smoother_data.smoother_variant;
  /// Surrogates
  additional_data.n_q_points_surrogate      = schwarz_smoother_data.n_q_points_surrogate;
  additional_data.normalize_surrogate_patch = schwarz_smoother_data.normalize_surrogate_patch;
  additional_data.use_arc_length            = schwarz_smoother_data.use_arc_length;
  /// Misc
  additional_data.caching_strategy = schwarz_smoother_data.caching_strategy;
  additional_data.print_details    = schwarz_smoother_data.print_details;
}



#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZDATA_H_ */
