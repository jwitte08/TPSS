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
      symmetrize_smoothing(false),
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
    std::string str_patch_variant[] = {"Undefined", "Cell-Patch", "Vertex-Patch"};

    std::string str_smoother_variant[] = {"Undefined", "Additive", "Multiplicative"};

    std::string str_schwarz_operator = str_smoother_variant[(int)smoother_variant] + ", " +
                                       str_patch_variant[(int)patch_variant] + " Operator";
    print_parameter(pcout, "Schwarz operator", str_schwarz_operator);
    print_parameter(pcout, "Number of smoothing steps", number_of_smoothing_steps);
    print_parameter(pcout, "Damping factor", damping_factor);
    print_parameter(pcout, "Patch-wise damping factor", local_damping_factor);
    print_parameter(pcout, "Manual coloring", manual_coloring ? "true" : "false");
    print_parameter(pcout, "Number of quad points (surrogate)", n_q_points_surrogate);
    print_parameter(pcout,
                    "Normalize surrogate patches",
                    normalize_surrogate_patch ? "true" : "false");
    print_parameter(pcout, "Arc length for surrogate", use_arc_length ? "true" : "false");
    if(smoother_variant != TPSS::SmootherVariant::additive)
      print_parameter(pcout,
                      "Symmetrize Schwarz operator",
                      symmetrize_smoothing ? "true" : "false");
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

  // symmetrizes the multiplicative Schwarz preconditioner
  bool symmetrize_smoothing;

  // print detailed information regarding the patch distribution, etc.
  bool print_details;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZDATA_H_ */
