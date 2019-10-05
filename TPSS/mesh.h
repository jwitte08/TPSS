
/*
 * create different mesh variants specified by the set of mesh parameters
 *
 * distorted cube: randomly distorts the interior vertices
 *
 *  Created on: Oct 05, 2019
 *      Author: witte
 */

#ifndef TPSS_MESH_H
#define TPSS_MESH_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

#include "utilities.h"
// #include "solvers_and_preconditioners/TPSS/patch_info.h"

using namespace dealii;

struct MeshParameter
{
  enum class GeometryVariant
  {
    None,
    CubeDistorted
  };
  static std::string
  str_geometry_variant(const GeometryVariant variant)
  {
    const std::string str_variant[] = {"None", "CubeDistorted"};
    return str_variant[(int)variant];
  }

  GeometryVariant variant       = GeometryVariant::None;
  int             n_refinements = -1;
  int             n_repetitions = -1;
  double          distortion    = -1.0;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Geometry:", str_geometry_variant(variant));
    oss << Util::parameter_to_fstring("N of refinements:", n_refinements);
    oss << Util::parameter_to_fstring("N of repetitions:", n_repetitions);
    oss << Util::parameter_to_fstring("Distortion factor:", distortion);
    return oss.str();
  }
};



// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++



template<int dim>
std::string
create_distorted_cube(Triangulation<dim> & tria, const MeshParameter & prm)
{
  AssertThrow(prm.distortion < 0.5, ExcMessage("Check distortion factor."));
  Assert(0. <= prm.distortion, ExcMessage("Distortion factor is invalid."));
  Assert(0 <= prm.n_refinements, ExcMessage("N of refinements is negative."));
  Assert(0 < prm.n_repetitions, ExcMessage("At least one repitition required"));
  tria.clear();

  // domain
  std::ostringstream oss;
  const double       left = 0.0, right = 1.0;
  const auto         str_domain = [&]() {
    std::ostringstream oss;
    for(unsigned int d = 0; d < dim; ++d)
      oss << "(" << left << ", " << right << (d != (dim - 1) ? ") x " : ")");
    return oss.str();
  };
  oss << Util::parameter_to_fstring("Domain:", str_domain());

  // create root mesh
  GridGenerator::subdivided_hyper_cube(tria, prm.n_repetitions, left, right);
  GridTools::distort_random(prm.distortion,
                            tria,
                            /*keep_boundary*/ true);
  const auto str_root_mesh = [&]() {
    std::ostringstream oss;
    const unsigned     n_root_cells_per_dim = prm.n_repetitions;
    for(unsigned int d = 0; d < dim; ++d)
      oss << n_root_cells_per_dim << (d != (dim - 1) ? " x " : "");
    return oss.str();
  };
  oss << Util::parameter_to_fstring("Distorted root mesh:", str_root_mesh());
  oss << Util::parameter_to_fstring("N of root cells:", tria.n_global_active_cells());
  oss << Util::parameter_to_fstring("Distortion in %:", prm.distortion * 100);

  // refine
  tria.refine_global(prm.n_refinements);
  oss << Util::parameter_to_fstring("N of active cells:", tria.n_global_active_cells());

  return oss.str();
}

#endif /* TPSS_MESH_H */
