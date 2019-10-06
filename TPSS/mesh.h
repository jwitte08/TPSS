
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

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

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
  str_geometry_variant(const GeometryVariant variant);

  std::string
  to_string() const;

  GeometryVariant variant       = GeometryVariant::None;
  int             n_refinements = -1;
  int             n_repetitions = -1;
  double          distortion    = -1.0;
};

// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
MeshParameter::str_geometry_variant(const GeometryVariant variant)
{
  const std::string str_variant[] = {"None", "CubeDistorted"};
  return str_variant[(int)variant];
}

std::string
MeshParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Geometry:", str_geometry_variant(variant));
  oss << Util::parameter_to_fstring("N of refinements:", n_refinements);
  oss << Util::parameter_to_fstring("N of repetitions:", n_repetitions);
  oss << Util::parameter_to_fstring("Distortion factor:", distortion);
  return oss.str();
}

template<int dim>
std::string
create_distorted_cube(Triangulation<dim> & tria, const MeshParameter & prm)
{
  AssertThrow(prm.distortion < 0.5, ExcMessage("Check distortion factor."));
  Assert(0. <= prm.distortion, ExcMessage("Distortion factor is invalid."));
  Assert(0 <= prm.n_refinements, ExcMessage("N of refinements is negative."));
  Assert(0 < prm.n_repetitions, ExcMessage("At least one repitition required"));
  Assert(prm.variant == MeshParameter::GeometryVariant::CubeDistorted,
         ExcMessage("Check geometry variant"));
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

template<int dim>
void
visualize_triangulation(const Triangulation<dim> & tria, const std::string prefix = "")
{
  DataOut<dim>    data_out;
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_DGQ<dim>(0)); // one (dof) value per cell !
  data_out.attach_dof_handler(dof_handler);

  // print the subdomain_ids
  Vector<double> subdomain(tria.n_active_cells()); // owned active cells !
  for(unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain", DataOut<dim>::type_cell_data);

  const auto get_filename = [&](const int proc_id) {
    std::ostringstream oss;
    oss << prefix << "active_cells_" << dim << "D";
    if(proc_id == -1)
      oss << ".pvtu";
    else
      oss << "_proc" << Utilities::int_to_string(proc_id, 4) << ".vtu";
    return oss.str();
  };
  const int   my_proc_id  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::string my_filename = get_filename(my_proc_id);
  data_out.build_patches();
  {
    std::ofstream file(my_filename);
    data_out.write_vtu(file);
  }
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    const unsigned           n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    for(unsigned int id = 0; id < n_mpi_procs; ++id)
      filenames.push_back(get_filename(id));
    std::ofstream master_file(get_filename(-1));
    data_out.write_pvtu_record(master_file, filenames);
  }
}

#endif /* TPSS_MESH_H */
