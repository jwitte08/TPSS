
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

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "solvers_and_preconditioners/TPSS/tensors.h"
#include "utilities.h"

using namespace dealii;



struct MeshParameter
{
  enum class GeometryVariant
  {
    None,
    Cube,
    CubeDistorted,
    CuboidSubdivided
  };

  unsigned int
  n_root_cells_1d(const unsigned int dimension) const;

  static std::string
  str_geometry_variant(const GeometryVariant variant);

  std::string
  to_string() const;

  GeometryVariant           geometry_variant = GeometryVariant::None;
  int                       n_refinements    = -1;
  int                       n_repetitions    = -1;
  std::vector<unsigned int> n_subdivisions;
  double                    distortion = -1.0;
};



unsigned int
MeshParameter::n_root_cells_1d(const unsigned int dimension) const
{
  if(geometry_variant == MeshParameter::GeometryVariant::Cube)
    return n_repetitions;
  else if(geometry_variant == MeshParameter::GeometryVariant::CubeDistorted)
    return n_repetitions;
  else if(geometry_variant == MeshParameter::GeometryVariant::CuboidSubdivided)
  {
    AssertIndexRange(dimension, n_subdivisions);
    return n_subdivisions[dimension];
  }
  AssertThrow(false, ExcMessage("Geometry variant is not supported."));
  return numbers::invalid_unsigned_int;
}


std::string
MeshParameter::str_geometry_variant(const GeometryVariant variant)
{
  const std::string str_variant[] = {"None", "Cube", "CubeDistorted", "CuboidSubdivided"};
  return str_variant[(int)variant];
}


std::string
MeshParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Geometry:", str_geometry_variant(geometry_variant));
  oss << Util::parameter_to_fstring("Number of refinements:", n_refinements);
  oss << Util::parameter_to_fstring("Number of repetitions:", n_repetitions);
  oss << Util::parameter_to_fstring("Subdivisions per direction:",
                                    vector_to_string(n_subdivisions));
  oss << Util::parameter_to_fstring("Distortion factor:", distortion);
  return oss.str();
}


/*
 * Creates a unit (hyper-)cube with @ prm.n_repetitions root cells per
 * direction. The root mesh is uniformly refined based on @p prm.n_refinements.
 */
template<int dim>
std::string
create_unit_cube(Triangulation<dim> & tria, const MeshParameter & prm)
{
  Assert(0 <= prm.n_refinements, ExcMessage("Number of refinements is negative."));
  Assert(0 < prm.n_repetitions, ExcMessage("At least one repitition required"));
  Assert(prm.geometry_variant == MeshParameter::GeometryVariant::Cube,
         ExcMessage("Check the geometry variant."));
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
  const auto str_root_mesh = [&]() {
    std::ostringstream oss;
    const unsigned     n_root_cells_per_dim = prm.n_repetitions;
    for(unsigned int d = 0; d < dim; ++d)
      oss << n_root_cells_per_dim << (d != (dim - 1) ? " x " : "");
    return oss.str();
  };
  oss << Util::parameter_to_fstring("Root mesh:", str_root_mesh());
  oss << Util::parameter_to_fstring("Number of root cells:", tria.n_global_active_cells());

  // refine
  tria.refine_global(prm.n_refinements);
  oss << Util::parameter_to_fstring("Number of active cells:", tria.n_global_active_cells());

  return oss.str();
}


/*
 * Creates a unit (hyper-)cube with @ prm.n_repetitions root cells per
 * direction. Interior vertices of the root mesh are distorted into random
 * direction by @p prm.distortion (in %) of the longest edge attached to the
 * vertex. Afterwards root mesh cells are uniformly refined based on @p
 * prm.n_refinements.
 */
template<int dim>
std::string
create_distorted_cube(Triangulation<dim> & tria, const MeshParameter & prm)
{
  AssertThrow(prm.distortion < 0.5, ExcMessage("Check distortion factor."));
  Assert(0. <= prm.distortion, ExcMessage("Distortion factor is invalid."));
  Assert(0 <= prm.n_refinements, ExcMessage("Number of refinements is negative."));
  Assert(0 < prm.n_repetitions, ExcMessage("At least one repitition required"));
  Assert(prm.geometry_variant == MeshParameter::GeometryVariant::CubeDistorted,
         ExcMessage("Check the geometry variant."));
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
  oss << Util::parameter_to_fstring("Number of root cells:", tria.n_global_active_cells());
  oss << Util::parameter_to_fstring("Distortion in %:", prm.distortion * 100);

  // refine
  tria.refine_global(prm.n_refinements);
  oss << Util::parameter_to_fstring("Number of active cells:", tria.n_global_active_cells());

  return oss.str();
}


/*
 * Creates a (hyper-)cuboid with @ prm.n_subdivisions[d] root cells per
 * direction @p d. Root cells are congruent unit (hyper-)cubes. The root mesh is
 * uniformly refined based on @p prm.n_refinements.
 */
template<int dim>
std::string
create_subdivided_cuboid(Triangulation<dim> & tria, const MeshParameter & prm)
{
  Assert(0 <= prm.n_refinements, ExcMessage("Number of refinements is negative."));
  Assert(!prm.n_subdivisions.empty(), ExcMessage("No subdivisions have been set."));
  Assert(prm.n_subdivisions.size() == dim, ExcMessage("Invalid number of subdivisions."));
  Assert(prm.geometry_variant == MeshParameter::GeometryVariant::CuboidSubdivided,
         ExcMessage("Check the geometry variant."));
  tria.clear();

  // domain
  std::ostringstream oss;
  constexpr double   h = 1.;
  Point<dim>         left, right; // origin
  for(unsigned int d = 0; d < dim; ++d)
    right[d] = h * prm.n_subdivisions[d];
  const auto str_domain = [&]() {
    std::ostringstream oss;
    for(unsigned int d = 0; d < dim; ++d)
      oss << "(" << left[d] << ", " << right[d] << (d != (dim - 1) ? ") x " : ")");
    return oss.str();
  };
  oss << Util::parameter_to_fstring("Domain:", str_domain());

  // create root mesh
  GridGenerator::subdivided_hyper_rectangle(tria, prm.n_subdivisions, left, right);
  const auto str_root_mesh = [&]() {
    std::ostringstream oss;
    for(unsigned int d = 0; d < dim; ++d)
      oss << prm.n_subdivisions[d] << (d != (dim - 1) ? " x " : "");
    return oss.str();
  };
  oss << Util::parameter_to_fstring("Root mesh:", str_root_mesh());
  oss << Util::parameter_to_fstring("Number of root cells:", tria.n_global_active_cells());

  // refine
  tria.refine_global(prm.n_refinements);
  oss << Util::parameter_to_fstring("Number of active cells:", tria.n_global_active_cells());

  return oss.str();
}


template<int dim>
std::string
create_mesh(Triangulation<dim> & triangulation, const MeshParameter & prms)
{
  std::ostringstream oss;
  if(prms.geometry_variant == MeshParameter::GeometryVariant::Cube)
    oss << create_unit_cube(triangulation, prms);
  else if(prms.geometry_variant == MeshParameter::GeometryVariant::CubeDistorted)
    oss << create_distorted_cube(triangulation, prms);
  else if(prms.geometry_variant == MeshParameter::GeometryVariant::CuboidSubdivided)
    oss << create_subdivided_cuboid(triangulation, prms);
  else
    AssertThrow(false, ExcMessage("Unsupported geometry variant. Check your run time parameters."));
  return oss.str();
}


template<int dim>
types::global_dof_index
estimate_n_dofs(const FiniteElement<dim> & fe, const MeshParameter & prms)
{
  const auto dof_layout = TPSS::get_dof_layout(fe);

  if(TPSS::DoFLayout::DGQ == dof_layout)
  {
    /// construct root mesh obtaining the number of root cells
    parallel::distributed::Triangulation<dim> tria(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    MeshParameter root_prms = prms;
    root_prms.n_refinements = 0;
    create_mesh(tria, root_prms);
    const auto n_root_cells = tria.n_global_active_cells();

    /// assume uniform refinement and estimate the number of dofs
    const unsigned n_child_cells = (1 << dim);
    const unsigned n_cells       = n_root_cells * Utilities::pow(n_child_cells, prms.n_refinements);
    const unsigned n_dofs_per_cell =
      fe.n_dofs_per_cell(); // Utilities::pow(fe.n_dofs_per_cell(), dim);
    const types::global_dof_index n_dofs_est = n_cells * n_dofs_per_cell;
    return n_dofs_est;
  }

  else if(TPSS::DoFLayout::Q == dof_layout)
  {
    const unsigned                           n_dofs_per_cell_1d = fe.tensor_degree() + 1;
    std::array<types::global_dof_index, dim> n_dofs_1d;
    for(auto d = 0U; d < dim; ++d)
    {
      const auto n_cells_1d = prms.n_root_cells_1d(d) * (1 << prms.n_refinements);
      n_dofs_1d[d]          = n_cells_1d * (n_dofs_per_cell_1d - 1) + 1;
    }
    Tensors::TensorHelper<dim, types::global_dof_index> dof_tensor(n_dofs_1d);
    return dof_tensor.n_flat();
  }

  AssertThrow(false, ExcMessage("Dof layout is not supported."));
  return numbers::invalid_dof_index;
}



template<int dim>
struct IntegerCoordinateRoot
{
  IntegerCoordinateRoot() = delete;

  IntegerCoordinateRoot(const MeshParameter & mesh_prms)
    : root_to_coordinate(map_root_to_coordinate_impl(mesh_prms))
  {
  }

  Point<dim, unsigned int>
  operator()(const unsigned int index) const
  {
    return root_to_coordinate.at(index);
  }

  std::map<unsigned int, Point<dim, unsigned>>
  map_root_to_coordinate_impl(const MeshParameter & mesh_prms) const
  {
    const bool is_hypercube =
      MeshParameter::GeometryVariant::Cube == mesh_prms.geometry_variant ||
      MeshParameter::GeometryVariant::CubeDistorted == mesh_prms.geometry_variant;
    const bool is_subdivided_cuboid =
      MeshParameter::GeometryVariant::CuboidSubdivided == mesh_prms.geometry_variant;
    if(is_hypercube)
      return root_to_coordinate_cube(mesh_prms.n_repetitions);
    else if(is_subdivided_cuboid)
      return root_to_coordinate_subdivided(mesh_prms.n_subdivisions);
    else
      AssertThrow(false, ExcMessage("Not implemented"));
    return std::map<unsigned int, Point<dim, unsigned>>{};
  }

  std::map<unsigned int, Point<dim, unsigned>>
  root_to_coordinate_cube(const unsigned int n_repetitions) const
  {
    // TODO hacked! pass triangulation by reference?
    parallel::distributed::Triangulation<dim> tria(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    MeshParameter prms;
    prms.n_refinements    = 0;
    prms.n_repetitions    = n_repetitions;
    prms.geometry_variant = MeshParameter::GeometryVariant::Cube;
    create_unit_cube(tria, prms);
    // GridGenerator::subdivided_hyper_cube(tria, n_repetitions, 0.0, 1.0);
    const double h                  = 1.0 / n_repetitions;
    const auto   get_interval_index = [&](const double x) -> unsigned int {
      for(unsigned index = 0; index < n_repetitions; ++index)
        if(index * h < x && x < ((index + 1) * h))
          return index;
      AssertThrow(false, ExcMessage("No valid interval found."));
      return -1;
    };

    // For each root cell determine the lexicographic cell numbering
    std::map<unsigned int, Point<dim, unsigned>> root_to_coord;
    for(const auto & cell : tria.active_cell_iterators())
    {
      const CellId         cell_id    = cell->id();
      const unsigned       root_index = cell_id.to_binary<dim>()[0];
      const auto           center     = cell->center();
      Point<dim, unsigned> coord;
      for(int d = 0; d < dim; ++d)
        coord(d) = get_interval_index(center[d]);
      const auto check = root_to_coord.insert({root_index, coord});
      (void)check;
      Assert(check.second, ExcMessage("Insertion failed."));
      // std::cout << "root: " << root_index << " coord: " << coord << " center: " << cell->center()
      //           << std::endl;
    }
    return root_to_coord;
  }

  std::map<unsigned int, Point<dim, unsigned>>
  root_to_coordinate_subdivided(const std::vector<unsigned int> n_subdivisions) const
  {
    Assert(n_subdivisions.size() == dim, ExcMessage("Invalid number of subdivisions."));
    // TODO hacked! pass triangulation by reference?
    parallel::distributed::Triangulation<dim> tria(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    MeshParameter prms;
    prms.n_refinements    = 0;
    prms.n_subdivisions   = n_subdivisions;
    prms.geometry_variant = MeshParameter::GeometryVariant::CuboidSubdivided;
    create_subdivided_cuboid(tria, prms);
    const auto get_interval_index = [&](const double x, const unsigned d) -> unsigned int {
      const double h = 1.0;
      for(unsigned index = 0; index < n_subdivisions[d]; ++index)
        if(index * h < x && x < ((index + 1) * h))
          return index;
      AssertThrow(false, ExcMessage("No valid interval found."));
      return -1;
    };

    // For each root cell determine the lexicographic cell numbering
    std::map<unsigned int, Point<dim, unsigned>> root_to_coord;
    for(const auto & cell : tria.active_cell_iterators())
    {
      const CellId         cell_id    = cell->id();
      const unsigned       root_index = cell_id.to_binary<dim>()[0];
      const auto           center     = cell->center();
      Point<dim, unsigned> coord;
      for(int d = 0; d < dim; ++d)
        coord(d) = get_interval_index(center[d], d);
      const auto check = root_to_coord.insert({root_index, coord});
      (void)check;
      Assert(check.second, ExcMessage("Insertion failed."));
      // std::cout << "root: " << root_index << " coord: " << coord << " center: " << cell->center()
      //           << std::endl;
    }
    return root_to_coord;
  }

  /*
   * Maps indices of root cells to its integer coordinate. The root
   * mesh is defined in terms of the mesh parameters.
   */
  std::map<unsigned int, Point<dim, unsigned>> root_to_coordinate;
};



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
