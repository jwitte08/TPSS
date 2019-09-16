/*
 * laplace_problem.h
 *
 * DG Laplace problem used to test Schwarz smoothers.
 *
 *  Created on: Aug 11, 2018
 *      Author: witte
 */

#ifndef TESTS_LAPLACEPROBLEM_H_
#define TESTS_LAPLACEPROBLEM_H_

#include <deal.II/base/convergence_table.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_selector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/meshworker/assembler.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>

#include "operators/linear_operator_base.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/time_info.h"
#include "solvers_and_preconditioners/preconditioner/schwarz_preconditioner.h"
#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"

#include "laplace_integrator.h"

namespace Laplace
{
struct Parameter
{
  enum class SolverVariant
  {
    CG,
    GMRES
  };
  static const std::string str_solver_variant[];
  enum class PreconditionVariant
  {
    ID,
    MG
  };
  const char *
  str_precondition_variant(const PreconditionVariant variant) const
  {
    const char * str_variant[] = {"identity", "multigrid"};
    return str_variant[(int)variant];
  }
  enum class GeometryVariant
  {
    Cube,
    CubeDistorted,
    Ball,
    PaperHermite,
    PaperHermiteAffine,
    SubdividedCubeoid
  };
  static const std::string str_geometry_variant[];
  enum class MGSmootherVariant
  {
    Schwarz,
    ChebyshevSchwarz
  };
  static const std::string str_mg_smoother_variant[];
  enum class CoarseGridVariant
  {
    IterativeFixed,
    ChebyshevAccurate
  };
  static const std::string str_coarse_grid_variant[];

  Parameter()
    : dof_limit_min(1),
      dof_limit_max(1e7),
      n_refines(1),
      n_cell_repetitions(1),
      allow_one_level_only(false),
      coarse_level(0),
      n_cycles(1),
      geometry_variant(GeometryVariant::Cube),
      n_refines_distort(0),
      distortion_factor(0.01),
      solver_variant(SolverVariant::CG),
      solver_reduction(1.e-8),
      solver_max_iterations(111),
      precondition_variant(PreconditionVariant::MG),
      mf_tasks_scheme_id(0),
      n_threads(0),
      mg_smoother_variant(MGSmootherVariant::Schwarz),
      compute_damping_factor(true),
      mg_smoother_post_reversed(false),
      non_overlapping(false),
      chebyshev_degree(1),
      smoothing_range(15.),
      n_cg_steps_chebyshev(10),
      mg_coarse_grid_variant(CoarseGridVariant::ChebyshevAccurate),
      mg_coarse_iterations(40),
      mg_coarse_tolerance(1.e-12),
      mg_coarse_chebyshev_degree(numbers::invalid_unsigned_int),
      mg_coarse_chebyshev_reduction(1.e-8),
      sync_timings(true),
      filename("post_process_data.txt"),
      write_fe_output(false),
      write_timer_output(false),
      write_post_process_output(false),
      compute_errors(false)
  {
    schwarz_smoother_data.patch_variant    = TPSS::PatchVariant::cell;
    schwarz_smoother_data.smoother_variant = TPSS::SmootherVariant::additive;
  }

  void
  check() const
  {
    AssertThrow(coarse_level < n_refines || allow_one_level_only,
                ExcMessage("Coarse level equals or exceeds finest level!"));
    // AssertThrow (n_cell_repetitions%2 == 1,
    // 		   ExcMessage("Cell repetition is not odd!"));
    AssertThrow(n_refines_distort <= n_refines,
                ExcMessage(
                  "The refinement step at which we distort exceeds the global refinement step!"));
    AssertThrow(distortion_factor < 0.5,
                ExcMessage("Distortion factor must not exceed 0.5 to prevent negative volumes."));
    if(non_overlapping)
      AssertThrow(schwarz_smoother_data.patch_variant == TPSS::PatchVariant::vertex &&
                    schwarz_smoother_data.smoother_variant == TPSS::SmootherVariant::additive &&
                    (geometry_variant == GeometryVariant::Cube ||
                     geometry_variant == GeometryVariant::CubeDistorted),
                  ExcMessage("Geometry, patch or smoother variant is invalid."));
    AssertThrow(mg_coarse_chebyshev_reduction < 1,
                ExcMessage("A relative reduction tolerance of less than one is required."));
  }

  void
  print(ConditionalOStream & cout) const
  {
    std::ostringstream ostream;

    // *** system
    ostream << n_refines;
    if(n_cycles > 1)
      ostream << ",...," << n_refines + n_cycles - 1;
    print_row_variable(cout, 2, "", 43, "Looping over system levels:", ostream.str());
    cout << std::endl;

    // *** solver
    print_row_variable(cout, 2, "", 43, "lower DoF limit:", dof_limit_min);
    print_row_variable(cout, 2, "", 43, "upper DoF limit:", dof_limit_max);
    const std::string str_solver[] = {"CG", "GMRES"};
    print_row_variable(cout, 2, "", 43, "Iterative solver:", str_solver[(int)solver_variant]);
    print_row_variable(cout, 2, "", 2, ">", 41, "Reduction:", solver_reduction);
    print_row_variable(
      cout, 2, "", 2, ">", 41, "Maximum number of iterations:", solver_max_iterations);
    print_row_variable(
      cout, 2, "", 43, "Preconditioner:", str_precondition_variant(precondition_variant));
    cout << std::endl;

    // *** print information of MatrixFree strategy
    const std::string str_tasks_scheme[] = {"none",
                                            "partition_partition",
                                            "partition_color",
                                            "color"};
    print_row_variable(cout,
                       2,
                       "",
                       43,
                       "Matrix-free parallel scheme of tasks:",
                       str_tasks_scheme[mf_tasks_scheme_id]);
    cout << std::endl;

    // *** print information of Schwarz smoother
    const auto print_schwarz_smoother_data = [&]() {
      if(non_overlapping)
        print_row_variable(cout, 2, "", 43, " --- ", "Non-Overlapping");
      schwarz_smoother_data.print(cout);
      cout << std::endl;
    };

    switch(mg_smoother_variant)
    {
      case MGSmootherVariant::Schwarz:
      {
        print_row_variable(cout, 2, "", 43, "Multigrid Smoother (Pre&Post):", "Schwarz");
        print_schwarz_smoother_data();
        break;
      }
      case MGSmootherVariant::ChebyshevSchwarz:
      {
        print_row_variable(cout, 2, "", 43, "Multigrid Smoother (Pre&Post):", "Chebyshev");
        print_row_variable(cout, 2, "", 2, ">", 41, "Polynomial degree:", chebyshev_degree);
        print_row_variable(cout,
                           2,
                           "",
                           2,
                           ">",
                           41,
                           "Smoothing range:",
                           Utilities::int_to_string(smoothing_range, 2));
        print_row_variable(
          cout, 2, "", 2, ">", 41, "Max. number of PCG steps:", n_cg_steps_chebyshev);
        print_row_variable(cout, 2, "", 43, "", "preconditioned by");
        print_schwarz_smoother_data();
        break;
      }
    }

    // *** print coarse grid information
    switch(mg_coarse_grid_variant)
    {
      case CoarseGridVariant::IterativeFixed:
      {
        print_row_variable(cout, 2, "", 43, "Coarse-Grid Solver:", "Fixed iterative solver");
        print_row_variable(
          cout, 2, "", 2, ">", 41, "Iterative method:", str_solver[(int)solver_variant]);
        print_row_variable(
          cout, 2, "", 2, ">", 41, "Max. number of iterations:", mg_coarse_iterations);
        print_row_variable(cout, 2, "", 2, ">", 41, "Max. tolerance:", mg_coarse_tolerance);
        break;
      }
      case CoarseGridVariant::ChebyshevAccurate:
      {
        print_row_variable(cout, 2, "", 43, "Coarse-Grid Solver:", "Accurate Chebyshev smoother");
        std::string str_degree = mg_coarse_chebyshev_degree == numbers::invalid_unsigned_int ?
                                   "t.b.c." :
                                   std::to_string(mg_coarse_chebyshev_degree);
        print_row_variable(cout, 2, "", 2, ">", 41, "Polynomial degree:", str_degree);
        print_row_variable(
          cout, 2, "", 2, ">", 41, "Guaranteed reduction/Accuracy:", mg_coarse_chebyshev_reduction);
        print_row_variable(
          cout, 2, "", 2, ">", 41, "Schwarz operator:", "Additive, Cell-Patch operator");
        break;
      }
      default:
        AssertThrow(false, ExcMessage("Coarse-Grid variant does not exist!"));
    }
  }

  unsigned int dof_limit_min;
  unsigned int dof_limit_max;
  unsigned int n_refines;
  unsigned int n_cell_repetitions;
  bool         allow_one_level_only;
  unsigned int coarse_level;
  unsigned int n_cycles;

  GeometryVariant           geometry_variant;
  unsigned int              n_refines_distort;
  double                    distortion_factor;
  std::vector<unsigned int> n_subdivisions;

  SolverVariant       solver_variant;
  double              solver_reduction;
  unsigned            solver_max_iterations;
  PreconditionVariant precondition_variant;

  unsigned int mf_tasks_scheme_id;
  unsigned int n_threads;

  MGSmootherVariant   mg_smoother_variant;
  bool                compute_damping_factor;
  bool                mg_smoother_post_reversed;
  bool                non_overlapping;
  unsigned int        chebyshev_degree;
  double              smoothing_range;
  unsigned int        n_cg_steps_chebyshev;
  SchwarzSmootherData schwarz_smoother_data;

  CoarseGridVariant mg_coarse_grid_variant;
  unsigned          mg_coarse_iterations;
  double            mg_coarse_tolerance;
  unsigned int      mg_coarse_chebyshev_degree;
  double            mg_coarse_chebyshev_reduction;

  bool        sync_timings;
  std::string filename;
  bool        write_fe_output;
  bool        write_timer_output;
  bool        write_post_process_output;
  bool        compute_errors;
};
const std::string Parameter::str_solver_variant[] = {"cg", "gmres"};
// enum class GeometryVariant
// {
//   Cube,
//   CubeDistorted,
//   Ball
// };
const std::string Parameter::str_geometry_variant[] = {"cube",
                                                       "cubedist",
                                                       "ball",
                                                       "subdivided_cubeoid"};
// enum class MGSmootherVariant
// {
//   Schwarz,
//   ChebyshevSchwarz
// };
const std::string Parameter::str_mg_smoother_variant[] = {"schwarz", "chebyschwarz"};
// enum class CoarseGridVariant
// {
//   IterativeFixed,
//   ChebyshevAccurate
// };
const std::string Parameter::str_coarse_grid_variant[] = {"fixed", "chebacc"};

struct PostProcessData
{
  std::vector<unsigned int> n_mg_levels;
  std::vector<unsigned int> n_cells_global;
  std::vector<unsigned int> n_dofs_global;
  std::vector<unsigned int> n_colors_system;
  std::vector<double>       solve_time;
  std::vector<double>       setup_time;
  std::vector<double>       n_iterations_system;
  std::vector<double>       average_reduction_system;
  std::vector<double>       L2_error;
  std::vector<double>       H1semi_error;
};

struct MultigridSignal
{
  MultigridSignal(const unsigned min_level, const unsigned max_level)
    : time_infos(min_level, max_level)
  {
  }

  void
  operator()(const bool is_active, const unsigned level)
  {
    if(is_active)
      time.restart();
    else
      time_infos[level].add_time(time.wall_time());
  }

  TimeInfo
  get_level_time(const unsigned int level, const std::string & description = "TBA") const
  {
    AssertThrow(time_infos.min_level() <= level && level <= time_infos.max_level(),
                ExcMessage("Level exceeds the range of TimeInfos stored."));
    TimeInfo time_info(time_infos[level]);
    time_info.description = description;
    return time_info;
  }

  TimeInfo
  get_total_time(const std::string & description = "TBA") const
  {
    TimeInfo time_info_total;
    for(unsigned l = time_infos.min_level(); l <= time_infos.max_level(); ++l)
    {
      time_info_total.time += time_infos[l].time;
      time_info_total.n_calls += time_infos[l].n_calls;
    }
    time_info_total.description = description;
    return time_info_total;
  }

  Timer                   time;
  MGLevelObject<TimeInfo> time_infos;
};

template<int dim>
Point<dim, unsigned int>
get_integer_coords(const CellId cell_id)
{
  // Get child indices
  std::vector<unsigned int> child_indices;
  std::string               cell_id_str = cell_id.to_string();
  while(cell_id_str.size() > 4)
  {
    child_indices.insert(child_indices.begin(), Utilities::string_to_int(&(cell_id_str.back())));
    cell_id_str.pop_back();
  }

  // Initialize global coordinate with coarse cell coordinate
  Point<dim, unsigned int> global_coord;
  const unsigned int       coarse_id = cell_id.to_binary<dim>()[0];
  {
    const std::bitset<dim> bit_indices(coarse_id);
    for(unsigned int d = 0; d < dim; ++d)
      global_coord(d) = bit_indices[d];
  }

  // Compute local coordinate and add to global
  for(auto c : child_indices)
  {
    Point<dim, unsigned int> local_coord;
    {
      const std::bitset<dim> bit_indices(c);
      for(unsigned int d = 0; d < dim; ++d)
        local_coord(d) = bit_indices[d];
    }

    global_coord *= 2;
    global_coord += local_coord;
  }

  return global_coord;
}

template<int dim>
std::vector<std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>>
get_coloring(const DoFHandler<dim> & dof, const unsigned int level)
{
  std::vector<std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>>> coloring(2);

  for(auto & cell : dof.mg_cell_iterators_on_level(level))
    if(cell->is_locally_owned_on_level())
    {
      // This is for cell patches, so each patch is only 1 cell
      std::vector<typename DoFHandler<dim>::level_cell_iterator> patch;
      patch.push_back(cell);

      // Get integer coordinates
      Point<dim, unsigned int> cell_int_coords = get_integer_coords<dim>(cell->id());

      // If integer coordinates sum to an even
      // number give color 0, else give color 1
      unsigned int color = 0;
      unsigned int sum   = 0;
      for(unsigned int d = 0; d < dim; ++d)
        sum += cell_int_coords(d);
      if(sum % 2 == 1)
        color = 1;

      // Add patch to coloring
      coloring[color].push_back(patch);
    }

  return coloring;
}

template<int dim>
struct ManualColoringCP
{
  using CellIterator = typename TPSS::PatchInfo<dim>::CellIterator;
  // using AdditionalData = typename TPSS::PatchInfo<dim>::AdditionalData;

  std::vector<std::vector<std::vector<CellIterator>>>
  operator()(const DoFHandler<dim> *                             dof_handler,
             const typename TPSS::PatchInfo<dim>::AdditionalData additional_data)
  {
    std::vector<std::vector<std::vector<CellIterator>>> colored_patches;
    colored_patches = get_coloring(*dof_handler, additional_data.level);
    return colored_patches;
  }
};

template<int dim>
struct CubeWorker
{
  using CellIterator  = typename TPSS::PatchInfo<dim>::CellIterator;
  using PatchIterator = typename TPSS::PatchInfo<dim>::PatchIterator;

  /*
   * Executes an operation on a CellIterator and its right neighbor in
   * x-direction.
   */
  template<typename Operation>
  void
  operate_in_x_direction(const Operation & cell_operation, const CellIterator & left_cell) const
  {
    cell_operation(left_cell);
    const auto & right_cell = left_cell->neighbor(1);
    Assert(right_cell.state() != IteratorState::IteratorStates::invalid, ExcInternalError());
    cell_operation(right_cell);
  }

  /*
   * Executes an operation in x-direction on a CellIterator and its
   * right neighbor with respect to the y-direction, respectively.
   */
  template<typename Operation>
  void
  operate_in_xy_directions(const Operation & cell_operation, const CellIterator & left_cell_y) const
  {
    operate_in_x_direction(cell_operation, left_cell_y);
    const auto & right_cell_y = left_cell_y->neighbor(3);
    operate_in_x_direction(cell_operation, right_cell_y);
  }

  /*
   * Executes an operation in xy-directions on a CellIterator and
   * its right neighbor with respect to the z-direction,
   * respectively
   */
  template<typename Operation>
  void
  operate_in_xyz_directions(const Operation &    cell_operation,
                            const CellIterator & left_cell_z) const
  {
    operate_in_xy_directions(cell_operation, left_cell_z);
    const auto & right_cell_z = left_cell_z->neighbor(5);
    operate_in_xy_directions(cell_operation, right_cell_z);
  }


  /*
   * distribute CellIterators forming a vertex patch with respect to the
   * first cell
   */
  std::vector<CellIterator>
  distribute_patch(const CellIterator & first_cell) const
  {
    std::vector<CellIterator> collection;
    auto &&                   add_cell_iterator = [&](const CellIterator & cell_it) {
      collection.emplace_back(cell_it);
    };

    if(dim == 1)
      operate_in_x_direction(add_cell_iterator, first_cell);
    else if(dim == 2)
      operate_in_xy_directions(add_cell_iterator, first_cell);
    else if(dim == 3)
      operate_in_xyz_directions(add_cell_iterator, first_cell);
    return collection;
  }

  /*
   * checks if a vertex is at the physical boundary
   */
  bool
  is_boundary_vertex(const CellIterator & cell, const unsigned int vertex_id) const
  {
    return std::any_of(std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       [&cell](const auto & face_no) { return cell->at_boundary(face_no); });
  }
};

template<int dim>
class ManualColoring : protected CubeWorker<dim>
{
public:
  using CellIterator  = typename CubeWorker<dim>::CellIterator;
  using PatchIterator = typename CubeWorker<dim>::PatchIterator;

  std::vector<std::vector<PatchIterator>>
  operator()(const DoFHandler<dim> *                             dof_handler,
             const typename TPSS::PatchInfo<dim>::AdditionalData additional_data,
             std::vector<std::vector<CellIterator>> &            cell_collections)
  {
    const auto level = additional_data.level;

    // LAMBDA checks if we can stride 'n_steps' cells in direction
    // 'direction', return a boolean and modify the incoming
    // CellIterator 'cell' by the requested strides if possible
    const auto && do_cell_stride =
      [level](CellIterator & cell, const int direction, const int n_steps) {
        AssertIndexRange(direction, 2 * dim);
        Assert(n_steps > 0, ExcInvalidState());
        for(int step = 0; step < n_steps; ++step)
        {
          if(cell->neighbor_level(direction) != static_cast<int>(level))
            return false;
          cell = cell->neighbor(direction);
        }
        return true;
      };

    // *** we pick the first cell (ASSERT: left lower corner!) and
    // *** build a vertex patch. Each cell within the patch gets its
    // *** own color. In addition, each color has a complementary
    // *** cell color, see handwritten notes.
    const unsigned int n_colors_max = 2 * (1 << dim);
    Assert(std::distance(dof_handler->begin(level), dof_handler->end(level)) >= 1 << dim,
           ExcNotImplemented());
    std::vector<CellIterator> initial_first_cells =
      std::move(this->distribute_patch(dof_handler->begin(level)));
    AssertDimension(initial_first_cells.size(), n_colors_max / 2);

    // *** colorize given a color and its complementary color
    std::vector<std::vector<CellIterator>> colored_first_cells;
    for(unsigned int color = 0; color < n_colors_max / 2; ++color)
    {
      // pair of colored & complementary colored cell iterator, see hand-written notes
      std::array<std::vector<CellIterator>, 2> collection_pair;
      std::pair<CellIterator, bool>            cell(initial_first_cells[color],
                                         /*is it complementary cell?*/ false);
      auto                                     cstart_stride_x  = cell;
      auto                                     cstart_stride_xy = cell;
      while(true) // collect cells in xyz
      {
        while(true) // collect cells in xy
        {
          while(true) // collect cells in x
          {
            if(this->is_boundary_vertex(cell.first, GeometryInfo<dim>::vertices_per_cell - 1))
              break; // break if 'cell' is not a valid first cell
            // forming a vertex patch
            const bool has_complementary_color = cell.second;
            collection_pair[has_complementary_color ? 1 : 0].push_back(cell.first);
            if(dim < 1 || !do_cell_stride(cell.first, 1, 2)) // stride twice in x
              break;
            else // "flip color"
              cell.second = !cell.second;
          }
          if(dim < 2 || !do_cell_stride(cstart_stride_x.first, 3, 2)) // stride twice in y
            break;
          else // "flip color"
            cstart_stride_x.second = !cstart_stride_x.second;
          cell = cstart_stride_x;
        }
        if(dim < 3 || !do_cell_stride(cstart_stride_xy.first, 5, 2)) // stride twice in z
          break;
        else // "flip color"
          cstart_stride_xy.second = !cstart_stride_xy.second;
        cell            = cstart_stride_xy;
        cstart_stride_x = cstart_stride_xy;
      }
      for(auto && collection : collection_pair)
        if(!collection.empty())
          colored_first_cells.emplace_back(std::move(collection));
    }

    // *** starting with the first cell of a regular vertex patch we
    // *** submit all CellIterators, constructing the patch, into
    // *** cell_collections and afterwards distribute the colored
    // *** PatchIterators
    std::vector<std::vector<PatchIterator>> colored_patch_iterators;
    const unsigned int                      n_colors = colored_first_cells.size();
    const unsigned int                      n_patches =
      std::accumulate(colored_first_cells.cbegin(),
                      colored_first_cells.cend(),
                      0,
                      [](const auto val, const auto vec) { return val + vec.size(); });
    AssertDimension(n_patches, cell_collections.size());
    cell_collections.clear();
    cell_collections.reserve(n_patches);
    colored_patch_iterators.resize(n_colors);

    for(unsigned int color = 0; color < n_colors; ++color)
    {
      // *** first, store the patches in cell_collections ...
      const unsigned int start_pos = cell_collections.size();
      for(const auto & first_cell : colored_first_cells[color])
        cell_collections.emplace_back(std::move(this->distribute_patch(first_cell)));

      // *** ... then, submit the iterators to stored patches into colored_patch_iterators
      PatchIterator patch_it = cell_collections.cbegin() + start_pos;
      colored_patch_iterators[color].reserve(colored_first_cells[color].size());
      for(; patch_it != cell_collections.cend(); ++patch_it)
        colored_patch_iterators[color].emplace_back(patch_it);
      AssertDimension(colored_patch_iterators[color].size(), colored_first_cells[color].size());
    }

    return colored_patch_iterators;
  }
};

template<int dim>
class NonOverlappingVertexPatch : protected CubeWorker<dim>
{
public:
  using CellIterator  = typename CubeWorker<dim>::CellIterator;
  using PatchIterator = typename CubeWorker<dim>::PatchIterator;

  void
  operator()(const DoFHandler<dim> *                             dof_handler,
             const typename TPSS::PatchInfo<dim>::AdditionalData additional_data,
             std::vector<std::vector<CellIterator>> &            cell_collections)
  {
    const auto level = additional_data.level;

    // LAMBDA checks if we can stride 'n_steps' cells in direction
    // 'direction', return a boolean and modify the incoming
    // CellIterator 'cell' by the requested strides if possible
    const auto && do_cell_stride =
      [level](CellIterator & cell, const int direction, const int n_steps) {
        AssertIndexRange(direction, 2 * dim);
        Assert(n_steps > 0, ExcInvalidState());
        for(int step = 0; step < n_steps; ++step)
        {
          if(cell->neighbor_level(direction) != static_cast<int>(level))
            return false;
          cell = cell->neighbor(direction);
        }
        return true;
      };

    // pair of colored & complementary colored cell iterator, see hand-written notes
    std::array<std::vector<CellIterator>, 2> collection_pair;
    std::pair<CellIterator, bool>            cell(dof_handler->begin(level),
                                       /*is it complementary cell?*/ false);

    cell_collections.clear();
    auto cstart_stride_x  = cell;
    auto cstart_stride_xy = cell;
    while(true) // collect cells in xyz
    {
      while(true) // collect cells in xy
      {
        while(true) // collect cells in x
        {
          if(this->is_boundary_vertex(cell.first, GeometryInfo<dim>::vertices_per_cell - 1))
            break; // break if 'cell' is not a valid first cell
          // *** forming a vertex patch
          // const bool has_complementary_color = cell.second;
          // collection_pair[has_complementary_color ? 1 : 0].push_back (cell.first);
          cell_collections.emplace_back(std::move(this->distribute_patch(cell.first)));
          if(dim < 1 || !do_cell_stride(cell.first, 1, 2)) // stride twice in x
            break;
          else // "flip color"
            cell.second = !cell.second;
        }
        if(dim < 2 || !do_cell_stride(cstart_stride_x.first, 3, 2)) // stride twice in y
          break;
        else // "flip color"
          cstart_stride_x.second = !cstart_stride_x.second;
        cell = cstart_stride_x;
      }
      if(dim < 3 || !do_cell_stride(cstart_stride_xy.first, 5, 2)) // stride twice in z
        break;
      else // "flip color"
        cstart_stride_xy.second = !cstart_stride_xy.second;
      cell            = cstart_stride_xy;
      cstart_stride_x = cstart_stride_xy;
    }
  }
};

template<int dim>
class SolutionBase
{
protected:
  static const std::size_t n_source_centers = 3;
  static const Point<dim>  source_centers[n_source_centers];
  static const double      width;
};

template<>
const Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] = {Point<1>(0.0),
                                                                                     Point<1>(0.25),
                                                                                     Point<1>(0.6)};

template<>
const Point<2> SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  {Point<2>(0.0, +0.0), Point<2>(0.25, 0.85), Point<2>(+0.6, 0.4)};

template<>
const Point<3> SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] =
  {Point<3>(0.0, 0.0, 0.0), Point<3>(0.25, 0.85, 0.85), Point<3>(0.6, 0.4, 0.4)};

template<int dim>
const double SolutionBase<dim>::width = 1. / 3.;

template<int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
  Solution() : Function<dim>(), SolutionBase<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    double       val   = 0;
    const double pi    = dealii::numbers::PI;
    const double width = SolutionBase<dim>::width;
    for(unsigned int i = 0; i < SolutionBase<dim>::n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - SolutionBase<dim>::source_centers[i];
      val += std::exp(-x_minus_xi.norm_square() / (width * width));
    }
    val /= dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * width);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int = 0) const override final
  {
    dealii::Tensor<1, dim> grad;
    const double           pi    = dealii::numbers::PI;
    const double           width = SolutionBase<dim>::width;
    for(unsigned int i = 0; i < SolutionBase<dim>::n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - SolutionBase<dim>::source_centers[i];
      grad +=
        -2. / (width * width) * std::exp(-x_minus_xi.norm_square() / (width * width)) * x_minus_xi;
    }
    return (grad / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * width));
  }

  virtual double
  laplacian(const dealii::Point<dim> & p, const unsigned int = 0) const override final
  {
    double       lapl  = 0;
    const double pi    = dealii::numbers::PI;
    const double width = SolutionBase<dim>::width;
    for(unsigned int i = 0; i < SolutionBase<dim>::n_source_centers; ++i)
    {
      const auto x_minus_xi = p - SolutionBase<dim>::source_centers[i];
      lapl += (2. / (width * width) * (x_minus_xi * x_minus_xi) - static_cast<double>(dim)) *
              std::exp(-x_minus_xi.norm_square() / (width * width));
    }
    lapl *= 2. / (width * width) / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * width);
    return lapl;
  }
};

constexpr double wave_number = 3.;

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() : Function<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    double val = 0;
    val -= solution_function.laplacian(p);
    return val;
  }

private:
  Solution<dim> solution_function;
};

template<int dim>
class SolutionPaperHermite : public Function<dim>
{
public:
  SolutionPaperHermite() : Function<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const;
};



template<int dim>
double
SolutionPaperHermite<dim>::value(const Point<dim> & p, const unsigned int) const
{
  double val = 1;
  for(unsigned int d = 0; d < dim; ++d)
    val *= std::sin(numbers::PI * p[d] * wave_number);
  return val;
}



template<int dim>
Tensor<1, dim>
SolutionPaperHermite<dim>::gradient(const Point<dim> & p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  for(unsigned int d = 0; d < dim; ++d)
  {
    return_value[d] = numbers::PI * wave_number * std::cos(numbers::PI * p[d] * wave_number);
    for(unsigned int e = 0; e < dim; ++e)
      if(d != e)
        return_value[d] *= std::sin(numbers::PI * p[d] * wave_number);
  }
  return return_value;
}



template<int dim>
class RightHandSidePaperHermite : public Function<dim>
{
public:
  RightHandSidePaperHermite() : Function<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const;
};


template<int dim>
double
RightHandSidePaperHermite<dim>::value(const Point<dim> & p, const unsigned int) const
{
  SolutionPaperHermite<dim> sol;
  return dim * numbers::PI * wave_number * numbers::PI * wave_number * sol.value(p);
}

template<int dim, int fe_degree, typename Number = double, int n_patch_dofs = -1>
struct MatrixOperator : public Subscriptor
{
  Parameter parameters;

  ConditionalOStream &       pcout;
  ConditionalOStream         time_details;
  mutable std::fstream       fstream;
  mutable ConditionalOStream pp_fstream;
  mutable PostProcessData    pp_data;

  const unsigned n_q_points;
  unsigned       global_level;
  std::string    str_geometry = "TBA";

  std::vector<std::shared_ptr<Manifold<dim>>> manifolds;
  parallel::distributed::Triangulation<dim>   triangulation;
  DoFHandler<dim>                             dof_handler;
  std::shared_ptr<FiniteElement<dim>>         fe;
  const MappingQGeneric<dim>                  mapping;

  SparsityPattern      sparsity;
  SparseMatrix<double> sparse_matrix;

  using SYSTEM_MATRIX = Laplace::MF::Operator<dim, fe_degree, Number>;
  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  SYSTEM_MATRIX                                  system_matrix;

  using value_type_mg = Number;
  using LEVEL_MATRIX  = Laplace::CombinedOperator<dim, fe_degree, value_type_mg>;
  using PATCH_MATRIX  = TensorProductMatrixSymmetricSum<dim, VectorizedArray<Number>, n_patch_dofs>;
  using SCHWARZ_PRECONDITIONER =
    SchwarzPreconditioner<dim,
                          LEVEL_MATRIX,
                          LinearAlgebra::distributed::Vector<value_type_mg>,
                          PATCH_MATRIX>;
  using SMOOTHER = SchwarzSmoother<dim,
                                   LEVEL_MATRIX,
                                   SCHWARZ_PRECONDITIONER,
                                   LinearAlgebra::distributed::Vector<value_type_mg>>;
  using CHEBYSHEV_SMOOTHER =
    PreconditionChebyshev<LEVEL_MATRIX,
                          LinearAlgebra::distributed::Vector<value_type_mg>,
                          SCHWARZ_PRECONDITIONER>;
  static constexpr unsigned int n_patch_dofs_coarse =
    fe_degree + 1; // IMPORTANT: cell patch for coarse level
  using PATCH_MATRIX_COARSE =
    TensorProductMatrixSymmetricSum<dim, VectorizedArray<Number>, n_patch_dofs_coarse>;
  using CHEBPREC_COARSE = SchwarzPreconditioner<dim,
                                                LEVEL_MATRIX,
                                                LinearAlgebra::distributed::Vector<value_type_mg>,
                                                PATCH_MATRIX_COARSE>;
  using CHEB_COARSE     = PreconditionChebyshev<LEVEL_MATRIX,
                                            LinearAlgebra::distributed::Vector<value_type_mg>,
                                            CHEBPREC_COARSE>;

  PreconditionIdentity                                         precondition_identity;
  MGLevelObject<LEVEL_MATRIX>                                  mg_matrices;
  MGTransferMatrixFree<dim, value_type_mg>                     mg_transfer;
  ManualColoring<dim>                                          make_manual_coloring;
  ManualColoringCP<dim>                                        make_manual_coloring_cp;
  NonOverlappingVertexPatch<dim>                               make_non_overlapping_vertex_patch;
  MGLevelObject<std::shared_ptr<const SCHWARZ_PRECONDITIONER>> mg_schwarz_precondition;
  MGSmootherPrecondition<LEVEL_MATRIX,
                         SCHWARZ_PRECONDITIONER,
                         LinearAlgebra::distributed::Vector<value_type_mg>>
    mg_smoother_precondition;
  MGSmootherRelaxation<LEVEL_MATRIX, SMOOTHER, LinearAlgebra::distributed::Vector<value_type_mg>>
    mg_smoother;
  std::shared_ptr<
    MGSmootherRelaxation<LEVEL_MATRIX, SMOOTHER, LinearAlgebra::distributed::Vector<value_type_mg>>>
    mg_smoother_post;
  MGSmootherRelaxation<LEVEL_MATRIX,
                       CHEBYSHEV_SMOOTHER,
                       LinearAlgebra::distributed::Vector<value_type_mg>>
                                                                            mg_smoother_cheb;
  const MGSmootherBase<LinearAlgebra::distributed::Vector<value_type_mg>> * mg_smoother_ptr;
  MGLevelObject<LEVEL_MATRIX>                                               mg_coarse_matrix;
  std::shared_ptr<const MGSmootherBase<LinearAlgebra::distributed::Vector<value_type_mg>>>
                                                                    mg_coarse_smooth;
  IterationNumberControl                                            coarse_control_fixed;
  SolverSelector<LinearAlgebra::distributed::Vector<value_type_mg>> coarse_solver;
  std::shared_ptr<const MGCoarseGridBase<LinearAlgebra::distributed::Vector<value_type_mg>>>
    mg_coarse_grid;

  LinearAlgebra::distributed::Vector<Number> solution;
  LinearAlgebra::distributed::Vector<Number> system_rhs;

  mutable ConvergenceTable convergence_table;
  unsigned int             n_timings_setup;
  unsigned int             n_timings_solve;

  MatrixOperator(ConditionalOStream & pcout_in,
                 const Parameter &    parameters_in,
                 const bool           do_tria = true)
    : parameters(parameters_in),
      pcout(pcout_in),
      time_details(pcout_in.get_stream(),
                   parameters.write_timer_output &&
                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      pp_fstream(pcout_in.get_stream() /*fstream*/,
                 parameters.write_post_process_output &&
                   Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      n_q_points(
        fe_degree +
        1) // parameters.geometry_variant==Parameter::GeometryVariant::Ball?2*fe_degree+1:fe_degree+1)
           // // in case of a non-linear mapping??
      ,
      global_level(static_cast<unsigned>(-1)),
      triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe(std::make_shared<FE_DGQ<dim>>(fe_degree)),
      mapping(fe_degree),
      n_timings_setup(0),
      n_timings_solve(0)
  {
    parameters.check();

    if(do_tria)
      create_triangulation(parameters.n_refines, parameters.n_cycles == 1);

    AssertThrow(
      n_q_points == fe_degree + 1,
      ExcMessage(
        "Current SYSTEM_MATRIX and LEVEL_MATRIX type only implemented for n_q_points==fe_degree+1."));

    parameters.schwarz_smoother_data.n_q_points_surrogate = std::min(8, fe_degree + 1);
    const auto patch_variant    = parameters.schwarz_smoother_data.patch_variant;
    const auto smoother_variant = parameters.schwarz_smoother_data.smoother_variant;
    if(parameters.compute_damping_factor)
    {
      if(parameters.non_overlapping)
        parameters.schwarz_smoother_data.damping_factor =
          TPSS::lookup_damping_factor(TPSS::PatchVariant::cell, smoother_variant, dim);
      else
        parameters.schwarz_smoother_data.damping_factor =
          TPSS::lookup_damping_factor(patch_variant, smoother_variant, dim);
    }
  }

  std::vector<unsigned int>
  get_timings_partition()
  {
    std::vector<unsigned int> partition;
    partition.emplace_back(0);
    partition.emplace_back(partition.back() + n_timings_setup);
    partition.emplace_back(partition.back() + n_timings_solve);
    return partition;
  }

  void
  fprint_time_details(const std::string & description, const double time_value)
  {
    std::ostringstream ostream;
    ostream << std::scientific << std::setprecision(3) << time_value << " s";
    auto && ftime = ostream.str();
    print_row_variable(time_details, 2, "T", 43, description, ftime);
  }

  bool
  create_triangulation(const unsigned n_refines, const bool print_details = false)
  {
    // *** initialize triangulation
    triangulation.clear();
    manifolds.clear();
    this->global_level            = -1;
    auto           mesh_info      = std::make_pair<bool, std::string>(false, "");
    const unsigned DOF_LIMIT_MIN_ = parameters.dof_limit_min;
    const unsigned DOF_LIMIT_MAX_ = parameters.dof_limit_max;

    switch(parameters.geometry_variant)
    {
      case Parameter::GeometryVariant::Cube:
      {
        // function
        const auto create_hyper_cube = [DOF_LIMIT_MIN_,
                                        DOF_LIMIT_MAX_](Triangulation<dim> & tria,
                                                        const unsigned int   n_refinements,
                                                        const unsigned int   n_repetitions) {
          constexpr unsigned n_dofs_per_cell_est = dim * Utilities::pow(fe_degree + 1, dim);
          const unsigned     n_cells_per_dim     = n_repetitions * (1 << n_refinements);
          const unsigned     n_cells_est         = Utilities::pow(n_cells_per_dim, dim);
          const unsigned     n_dofs_est          = n_cells_est * n_dofs_per_cell_est;
          if(n_dofs_est < DOF_LIMIT_MIN_ || DOF_LIMIT_MAX_ < n_dofs_est)
            return std::make_pair<bool, std::string>(false, "mesh exceeds limits!");

          const double left = 0.0, right = 1.0;
          GridGenerator::subdivided_hyper_cube(tria, n_repetitions, left, right);
          tria.refine_global(n_refinements);

          std::ostringstream oss;
          oss << "domain: ";
          for(unsigned int d = 0; d < dim; ++d)
            oss << "(" << left << ", " << right << (d != (dim - 1) ? ") x " : ")\n");
          oss << "mesh: ";
          for(unsigned int d = 0; d < dim; ++d)
            oss << n_cells_per_dim << (d != (dim - 1) ? " x " : "\n");

          const auto info = std::make_pair<bool, std::string>(true, oss.str());
          return info;
        };

        mesh_info = create_hyper_cube(triangulation, n_refines, parameters.n_cell_repetitions);
        const bool triangulation_was_created = mesh_info.first;
        if(!triangulation_was_created) // invalid mesh
          return false;

        global_level = triangulation.n_levels() - 1;
        if(print_details)
          pcout << mesh_info.second << std::endl;
        // TODO bad style
        return true; // valid mesh
      }
      case Parameter::GeometryVariant::CubeDistorted:
      {
        // function
        const auto create_distorted_cube =
          [DOF_LIMIT_MIN_, DOF_LIMIT_MAX_](Triangulation<dim> & tria,
                                           const double         distortion,
                                           const unsigned int   n_refinements_distort,
                                           const unsigned int   n_refinements,
                                           const unsigned int   n_repetitions) {
            AssertThrow(n_refinements_distort <= n_refinements, ExcMessage("Check refinements."));
            AssertThrow(distortion < 0.5, ExcMessage("Check distortion factor."));
            constexpr unsigned n_dofs_per_cell_est = dim * Utilities::pow(fe_degree + 1, dim);
            const unsigned     n_cells_per_dim     = n_repetitions * (1 << n_refinements);
            const unsigned     n_cells_est         = Utilities::pow(n_cells_per_dim, dim);
            const unsigned     n_dofs_est          = n_cells_est * n_dofs_per_cell_est;
            if(n_dofs_est < DOF_LIMIT_MIN_ || DOF_LIMIT_MAX_ < n_dofs_est)
              return std::make_pair<bool, std::string>(false, "mesh exceeds limits!");

            const double left = 0.0, right = 1.0;
            GridGenerator::subdivided_hyper_cube(tria, n_repetitions, left, right);
            tria.refine_global(n_refinements_distort);
            GridTools::distort_random(distortion,
                                      tria,
                                      /*keep_boundary*/ true);
            const unsigned int n_remaining = n_refinements - n_refinements_distort;
            tria.refine_global(n_remaining);

            std::ostringstream oss;
            oss << "domain: ";
            for(unsigned int d = 0; d < dim; ++d)
              oss << "(" << left << ", " << right << (d != (dim - 1) ? ") x " : ")\n");
            oss << "mesh: ";
            for(unsigned int d = 0; d < dim; ++d)
              oss << n_cells_per_dim << (d != (dim - 1) ? " x " : "\n");
            const unsigned n_cells_per_dim_distort = n_repetitions * (1 << n_refinements_distort);
            oss << "based on " << distortion * 100 << "%-distorted mesh: ";
            for(unsigned int d = 0; d < dim; ++d)
              oss << n_cells_per_dim_distort << (d != (dim - 1) ? " x " : "\n");

            const auto info = std::make_pair<bool, std::string>(true, oss.str());
            return info;
          };

        mesh_info = create_distorted_cube(triangulation,
                                          parameters.distortion_factor,
                                          parameters.n_refines_distort,
                                          n_refines,
                                          parameters.n_cell_repetitions);

        const bool triangulation_was_created = mesh_info.first;
        if(!triangulation_was_created) // invalid mesh
          return false;

        global_level = triangulation.n_levels() - 1;
        if(print_details)
          pcout << mesh_info.second << std::endl;
        // TODO bad style
        return true; // valid mesh
      }
      case Parameter::GeometryVariant::Ball:
      {
        // function
        const auto create_hyper_ball =
          [DOF_LIMIT_MIN_, DOF_LIMIT_MAX_](Triangulation<dim> &                          tria,
                                           std::vector<std::shared_ptr<Manifold<dim>>> & manifolds,
                                           const unsigned int n_refinements) {
            // hyperball: d-dimensional hyperball is uniquely determined by the center point and
            // radius
            Point<dim> center;
            for(unsigned int d = 0; d < dim; ++d)
              center(d) = 0.5;
            const double radius{1.25 * center.distance(Point<dim>{})};

            GridGenerator::hyper_ball(tria, center, radius);
            constexpr unsigned n_dofs_per_cell_est = dim * Utilities::pow(fe_degree + 1, dim);
            const unsigned     n_cells_initial     = tria.n_active_cells();
            const unsigned     n_cells_per_dim     = (1 << n_refinements);
            const unsigned     n_cells_est = n_cells_initial * Utilities::pow(n_cells_per_dim, dim);
            const unsigned     n_dofs_est  = n_cells_est * n_dofs_per_cell_est;
            if(n_dofs_est < DOF_LIMIT_MIN_ || DOF_LIMIT_MAX_ < n_dofs_est)
            {
              // tria.clear();
              return std::make_pair<bool, std::string>(false, "mesh exceeds limits!");
            }

            const auto boundary_description = std::make_shared<SphericalManifold<dim>>(center);
            const auto inner_manifold = std::make_shared<TransfiniteInterpolationManifold<dim>>();
            manifolds.push_back(boundary_description); // take ownership
            manifolds.push_back(inner_manifold);       // take ownership
            tria.set_all_manifold_ids(1);
            tria.set_all_manifold_ids_on_boundary(0);
            tria.set_manifold(0, *boundary_description);
            inner_manifold->initialize(tria);
            tria.set_manifold(1, *inner_manifold);
            tria.refine_global(n_refinements);

            std::ostringstream oss;
            oss << "domain: ";
            for(unsigned int d = 0; d < dim; ++d)
              oss << "B[(" << center << "); " << std::to_string(radius).substr(0, 4) << "]\n";
            oss << "mesh: ";
            for(unsigned int d = 0; d < dim; ++d)
              oss << "uniformly refining " << n_cells_initial << " to " << tria.n_active_cells()
                  << "\n";

            const auto info = std::make_pair<bool, std::string>(true, oss.str());
            return info;
          };

        mesh_info = create_hyper_ball(triangulation, manifolds, n_refines);
        const bool triangulation_was_created = mesh_info.first;
        if(!triangulation_was_created) // invalid mesh
          return false;

        global_level = triangulation.n_levels() - 1;
        if(print_details)
          pcout << mesh_info.second << std::endl;
        // TODO bad style
        return true; // valid mesh
        break;
      }
      case Parameter::GeometryVariant::PaperHermite:
      {
        // triangulation.clear();
        const unsigned cycle = n_refines;
        pcout << "Cycle " << cycle << std::endl;

        Timer                        time;
        std::size_t                  projected_size = numbers::invalid_size_type;
        unsigned int                 n_refine       = 0;
        Tensor<1, dim, unsigned int> mesh_sizes;
        n_refine                     = cycle / dim;
        const unsigned int remainder = cycle % dim;
        Point<dim>         p1;
        for(unsigned int d = 0; d < dim; ++d)
          p1[d] = -1;
        Point<dim> p2;
        for(unsigned int d = 0; d < remainder; ++d)
          p2[d] = 3;
        for(unsigned int d = remainder; d < dim; ++d)
          p2[d] = 1;
        std::vector<unsigned int> subdivisions(dim, 1);
        for(unsigned int d = 0; d < remainder; ++d)
          subdivisions[d] = 2;
        const unsigned int base_refine = (1 << n_refine);
        projected_size                 = 1;
        for(unsigned int d = 0; d < dim; ++d)
        {
          mesh_sizes[d] = base_refine * subdivisions[d];
          projected_size *= base_refine * subdivisions[d] * (fe_degree + 1);
        }
        GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

        if(projected_size < DOF_LIMIT_MIN_)
          return false;

        if(projected_size > DOF_LIMIT_MAX_)
        {
          pcout << "Projected size " << projected_size << " higher than max size, terminating."
                << std::endl;
          pcout << std::endl;
          return false;
        }

        triangulation.refine_global(n_refine);
        global_level = (triangulation.n_levels() - 1);
        print_row_variable(pcout, 2, "T", 43, "Time create grid:", time.wall_time());

        std::ostringstream oss;
        // std::locale s = oss.get_stream().getloc();
        oss.imbue(std::locale("en_US.UTF-8"));
        // oss << "Number of degrees of freedom  "
        //       << dof_handler.n_dofs() << " = (";
        oss << "PaperHermiteCartesian = (";
        for(unsigned int d = 0; d < dim; ++d)
          oss << mesh_sizes[d] << (d < dim - 1 ? " x " : ")");
        // pcout.get_stream().imbue(s);
        str_geometry = oss.str();
        pcout << str_geometry << std::endl;
        return true;
      }
      case Parameter::GeometryVariant::PaperHermiteAffine:
      {
        AssertThrow(dim == 3, ExcMessage("dimension not supported."));
        pcout << "Cycle " << n_refines << std::endl;
        unsigned int projected_size = 1;
        for(unsigned int d = 0; d < dim; ++d)
          projected_size *= (1 << n_refines) * (fe_degree + 1);
        if(projected_size < DOF_LIMIT_MIN_)
          return false;
        if(projected_size > DOF_LIMIT_MAX_)
        {
          pcout << "Projected size " << projected_size << " higher than max size, terminating."
                << std::endl;
          pcout << std::endl;
          return false;
        }
        std::ostringstream oss;
        oss << "PaperHermiteAffine = (";
        for(unsigned int d = 0; d < dim; ++d)
          oss << (1 << n_refines) << (d < dim - 1 ? " x " : ")");
        str_geometry = oss.str();
        pcout << str_geometry << std::endl;

        std::array<std::pair<double, double>, 3> intervals;
        intervals[0] = std::make_pair(-0.95, 0.95);
        intervals[1] = std::make_pair(-0.90, 0.89);
        intervals[2] = std::make_pair(-0.85, 0.83);
        Point<dim> p1, p2;
        for(unsigned int d = 0; d < dim; ++d)
        {
          p1[d] = intervals[d].first;
          p2[d] = intervals[d].second;
        }
        Triangulation<dim> brick;
        GridGenerator::hyper_rectangle(brick, p1, p2);
        const auto              cell = brick.begin();
        std::vector<Point<dim>> vertices_affine;
        vertices_affine.resize(GeometryInfo<dim>::vertices_per_cell);
        Tensor<2, dim, double> jacobian_affine;
        jacobian_affine[0][0] = 1.12;
        jacobian_affine[1][0] = 0.24;
        jacobian_affine[2][0] = 0.36;
        jacobian_affine[0][1] = 0.24;
        jacobian_affine[1][1] = 1.36;
        jacobian_affine[2][1] = 0.48;
        jacobian_affine[0][2] = 0.36;
        jacobian_affine[1][2] = 0.48;
        jacobian_affine[2][2] = 1.60;
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const auto vertex  = cell->vertex(v);
          vertices_affine[v] = Point<dim>(jacobian_affine * vertex);
        }
        GridGenerator::general_cell(triangulation, vertices_affine);
        triangulation.refine_global(n_refines);
        break;
      }
    }

    global_level = (triangulation.n_levels() - 1);

    if(print_details)
      print_mesh_info();
    return true;
  }

  template<typename Number2>
  std::shared_ptr<const MatrixFree<dim, Number2>>
  build_mf_storage(const unsigned level = static_cast<unsigned>(-1))
  {
    typename MatrixFree<dim, Number2>::AdditionalData mf_additional_data;
    const auto                                        p_scheme =
      static_cast<typename MatrixFree<dim, Number2>::AdditionalData::TasksParallelScheme>(
        parameters.mf_tasks_scheme_id);

    mf_additional_data.tasks_parallel_scheme = p_scheme;
    const auto mapping_update_flags =
      dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points;
    mf_additional_data.mapping_update_flags                = mapping_update_flags;
    mf_additional_data.mapping_update_flags_inner_faces    = mapping_update_flags;
    mf_additional_data.mapping_update_flags_boundary_faces = mapping_update_flags;
    if(level != static_cast<unsigned>(-1))
      mf_additional_data.level_mg_handler = level;
    AffineConstraints<double> constraints_dummy;
    constraints_dummy.close();
    const auto mf_storage = std::make_shared<MatrixFree<dim, Number2>>();
    mf_storage->reinit(
      mapping, dof_handler, constraints_dummy, QGauss<1>(n_q_points), mf_additional_data);
    return mf_storage;
  }

  template<typename Number2>
  std::shared_ptr<const SubdomainHandler<dim, Number2>>
  build_patch_storage(const unsigned                                        level,
                      const std::shared_ptr<const MatrixFree<dim, Number2>> mf_storage)
  {
    typename SubdomainHandler<dim, Number2>::AdditionalData fdss_additional_data;
    fdss_additional_data.level            = level;
    fdss_additional_data.n_threads        = parameters.n_threads;
    fdss_additional_data.patch_variant    = parameters.schwarz_smoother_data.patch_variant;
    fdss_additional_data.smoother_variant = parameters.schwarz_smoother_data.smoother_variant;
    fdss_additional_data.print_details    = parameters.schwarz_smoother_data.print_details;
    if(parameters.schwarz_smoother_data.manual_coloring)
    {
      fdss_additional_data.manual_coloring_func = std::ref(make_manual_coloring);
      using CellIterator                        = typename TPSS::PatchInfo<dim>::CellIterator;
      fdss_additional_data.manual_coloring_func_cp =
        [](const DoFHandler<dim> *                             dof_handler,
           const typename TPSS::PatchInfo<dim>::AdditionalData additional_data)
        -> std::vector<std::vector<std::vector<CellIterator>>> {
        return ManualColoringCP<dim>{}(dof_handler, additional_data);
      };
    }
    if(parameters.non_overlapping)
      fdss_additional_data.manual_gathering_func = std::ref(make_non_overlapping_vertex_patch);
    fdss_additional_data.n_q_points_surrogate =
      parameters.schwarz_smoother_data.n_q_points_surrogate;
    fdss_additional_data.normalize_surrogate_patch =
      parameters.schwarz_smoother_data.normalize_surrogate_patch;
    fdss_additional_data.use_arc_length = parameters.schwarz_smoother_data.use_arc_length;
    const auto patch_storage            = std::make_shared<SubdomainHandler<dim, Number2>>();
    patch_storage->reinit(mf_storage, fdss_additional_data);
    return patch_storage;
  }

  void
  distribute_dofs()
  {
    dof_handler.initialize(triangulation, *fe);
    dof_handler.distribute_mg_dofs();
  }

  std::vector<TimeInfo>
  setup(const bool build_rhs         = false,
        const bool print_details     = false,
        const bool setup_mg_smooprec = false)
  {
    pp_data.setup_time.push_back(0);
    std::vector<TimeInfo> time_infos;
    time_details << "Setup time:\n";
    Timer time;
    time.restart();

    // *** initialize (multigrid) degrees of freedom
    distribute_dofs();

    time.stop();
    time_infos.emplace_back(time.wall_time(), "Distribute (multigrid) DoFs:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    time.restart();

    // *** initialize matrix-free system
    this->mf_storage = build_mf_storage<Number>();
    system_matrix.clear();
    system_matrix.initialize(mf_storage);

    // *** initialize global vectors
    mf_storage->initialize_dof_vector(solution);
    mf_storage->initialize_dof_vector(system_rhs);

    time.stop();
    time_infos.emplace_back(time.wall_time(), "Setup matrix-free storage:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    time.restart();

    // *** compute the right hand side
    if(build_rhs)
      compute_rhs();

    time.stop();
    time_infos.emplace_back(time.wall_time(), "Compute right hand side:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    time.restart();

    // *** clear obsolete multigrid data
    mg_schwarz_precondition.resize(0, 0);
    mg_smoother_cheb.clear();
    mg_smoother.clear();
    mg_coarse_grid.reset();
    mg_coarse_smooth.reset();

    // *** setup multigrid data
    const unsigned mg_level_min = parameters.coarse_level;
    const unsigned mg_level_max = global_level;
    pp_data.n_mg_levels.push_back(mg_level_max - mg_level_min + 1);

    // *** initialize level matrices for the multigrid algorithm
    mg_matrices.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
    {
      const auto mf_storage_level = build_mf_storage<value_type_mg>(level);
      mg_matrices[level].initialize(mf_storage_level);
    }

    time.stop();
    time_infos.emplace_back(time.wall_time(), "[MG] Setup matrix-free storage:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    time.restart();

    // *** initialize multigrid transfer
    mg_transfer.build(dof_handler);

    time.stop();
    time_infos.emplace_back(time.wall_time(), "[MG] Build transfer:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    time.restart();

    // *** initialize smoother
    switch(parameters.mg_smoother_variant)
    {
      case Parameter::MGSmootherVariant::Schwarz:
      {
        typename SMOOTHER::AdditionalData dummy_data;
        mg_smoother.initialize(mg_matrices, dummy_data); // set mg_matrices in MGSmootherRelaxation
        if(setup_mg_smooprec)
        {
          typename SCHWARZ_PRECONDITIONER::AdditionalData dummy_data2;
          mg_smoother_precondition.initialize(
            mg_matrices, dummy_data2); // set mg_matrices in MGSmootherPrecondition
        }
        mg_schwarz_precondition.resize(mg_level_min, mg_level_max);
        for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
        {
          const auto mf_storage_on_level = mg_matrices[level].get_matrix_free();
          const auto patch_storage = build_patch_storage<value_type_mg>(level, mf_storage_on_level);

          // *** setup Schwarz preconditioner
          typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
          precondition_data.relaxation = parameters.schwarz_smoother_data.damping_factor;
          precondition_data.local_relaxation =
            parameters.schwarz_smoother_data.local_damping_factor;
          precondition_data.symmetrized     = parameters.schwarz_smoother_data.symmetrize_smoothing;
          const auto schwarz_preconditioner = std::make_shared<SCHWARZ_PRECONDITIONER>();
          schwarz_preconditioner->initialize(patch_storage, mg_matrices[level], precondition_data);
          mg_schwarz_precondition[level] = schwarz_preconditioner; // book-keeping

          // *** setup smoother
          typename SMOOTHER::AdditionalData mg_smoother_data;
          mg_smoother_data.number_of_smoothing_steps =
            parameters.schwarz_smoother_data.number_of_smoothing_steps;
          mg_smoother.smoothers[level].initialize(
            mg_matrices[level],
            schwarz_preconditioner,
            mg_smoother_data); // actual initialization of Schwarz smoother within
                               // MGSmootherRelaxation

          // *** setup smoother within MGSmootherPrecondition
          if(setup_mg_smooprec)
            mg_smoother_precondition.smoothers[level].initialize(
              patch_storage,
              mg_matrices[level],
              precondition_data); // actual initialization of preconditioner within
                                  // MGSmootherPrecondition
        }

        mg_smoother_ptr = &mg_smoother;
        if(parameters.mg_smoother_post_reversed)
        {
          mg_smoother_post = std::make_shared<
            MGSmootherRelaxation<LEVEL_MATRIX,
                                 SMOOTHER,
                                 LinearAlgebra::distributed::Vector<value_type_mg>>>();
          mg_smoother_post->initialize(mg_matrices, dummy_data);
          for(unsigned int level = mg_matrices.min_level(); level < mg_matrices.max_level() + 1;
              ++level)
          {
            const auto & smoother               = mg_smoother.smoothers[level];
            const auto & schwarz_preconditioner = smoother.get_preconditioner();
            typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data =
              schwarz_preconditioner.get_additional_data();
            precondition_data.reverse              = true;
            const auto schwarz_preconditioner_copy = std::make_shared<SCHWARZ_PRECONDITIONER>();
            schwarz_preconditioner_copy->initialize(schwarz_preconditioner, precondition_data);
            const auto & smoother_data = smoother.get_additional_data();
            mg_smoother_post->smoothers[level].initialize(mg_matrices[level],
                                                          schwarz_preconditioner_copy,
                                                          smoother_data);
          }
        }
        break;
      }

      case Parameter::MGSmootherVariant::ChebyshevSchwarz:
      {
        MGLevelObject<typename CHEBYSHEV_SMOOTHER::AdditionalData> mg_data_cheb;
        mg_data_cheb.resize(mg_level_min, mg_level_max);
        mg_schwarz_precondition.resize(mg_level_min, mg_level_max);
        for(unsigned int level = mg_level_min; level < mg_level_max + 1; ++level)
        {
          const auto mf_storage_on_level = mg_matrices[level].get_matrix_free();
          const auto patch_storage = build_patch_storage<value_type_mg>(level, mf_storage_on_level);

          // *** setup Schwarz preconditioner
          typename SCHWARZ_PRECONDITIONER::AdditionalData precondition_data;
          // const auto patch_variant = parameters.schwarz_smoother_data.patch_variant;
          // const auto smoother_variant = parameters.schwarz_smoother_data.smoother_variant;
          // precondition_data.relaxation = TPSS::lookup_damping_factor (patch_variant,
          // smoother_variant, dim);
          precondition_data.relaxation = parameters.schwarz_smoother_data.damping_factor;
          precondition_data.local_relaxation =
            parameters.schwarz_smoother_data.local_damping_factor;
          precondition_data.symmetrized     = parameters.schwarz_smoother_data.symmetrize_smoothing;
          const auto schwarz_preconditioner = std::make_shared<SCHWARZ_PRECONDITIONER>();
          schwarz_preconditioner->initialize(patch_storage, mg_matrices[level], precondition_data);
          mg_schwarz_precondition[level] = schwarz_preconditioner; // book-keeping

          // *** setup smoother data
          mg_data_cheb[level].preconditioner      = schwarz_preconditioner;
          mg_data_cheb[level].degree              = parameters.chebyshev_degree;
          mg_data_cheb[level].smoothing_range     = parameters.smoothing_range;
          mg_data_cheb[level].eig_cg_n_iterations = parameters.n_cg_steps_chebyshev;
          // mg_data_cheb[level].eig_cg_residual = 1.e-6;
          // mg_data_cheb[level].print_timings = true;
          // mg_data_cheb[level].max_eigenvalue = 1.;
        }
        mg_smoother_cheb.initialize(mg_matrices, mg_data_cheb);
        mg_smoother_cheb.set_steps(parameters.schwarz_smoother_data.number_of_smoothing_steps);

        mg_smoother_ptr = &mg_smoother_cheb;
        break;
      }

      default:
        AssertThrow(false, ExcMessage("Multigrid smoother variant not implemented!"));
    }
    const auto precondition_max = mg_schwarz_precondition[mg_level_max];
    const auto n_colors =
      precondition_max->get_subdomain_handler()->get_partition_data().n_colors();
    pp_data.n_colors_system.push_back(n_colors);

    time.stop();
    time_infos.emplace_back(time.wall_time(), "[MG] Build multigrid smoother:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    std::vector<TimeInfo> time_infos_smoother = std::move(get_time_info_schwarz());
    std::remove_copy_if(time_infos_smoother.begin(),
                        time_infos_smoother.end(),
                        std::back_inserter(time_infos),
                        [](const auto & info) { return !info.is_active(); });
    time.restart();

    // *** initialize coarse-grid solver
    const unsigned coarse_level = mg_matrices.min_level();
    switch(parameters.mg_coarse_grid_variant)
    {
      case Parameter::CoarseGridVariant::IterativeFixed:
      {
        coarse_control_fixed.set_max_steps(parameters.mg_coarse_iterations);
        coarse_control_fixed.set_tolerance(parameters.mg_coarse_tolerance);
        // coarse_solver.select ("cg");
        coarse_solver.select(Parameter::str_solver_variant[(int)parameters.solver_variant]);
        coarse_solver.set_control(coarse_control_fixed);
        auto mg_coarse = std::make_shared<
          MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<value_type_mg>,
                                      decltype(coarse_solver),
                                      LEVEL_MATRIX,
                                      SCHWARZ_PRECONDITIONER>
          // PreconditionIdentity>
          >();
        mg_coarse->initialize(coarse_solver,
                              mg_matrices[coarse_level],
                              *(mg_schwarz_precondition[coarse_level]));
        // precondition_identity);
        mg_coarse_grid = mg_coarse;
        break;
      }

      case Parameter::CoarseGridVariant::ChebyshevAccurate:
      {
        // *** setup coarse matrix (matrix-free)
        mg_coarse_matrix.resize(coarse_level, coarse_level); // Wrapper
        const auto mf_storage = mg_matrices[coarse_level].get_matrix_free();
        mg_coarse_matrix[coarse_level].initialize(mf_storage);
        auto & coarse_matrix = mg_coarse_matrix[coarse_level];

        // *** setup patch storage (CELL ADDITIVE)
        typename SubdomainHandler<dim, value_type_mg>::AdditionalData fdss_additional_data;
        fdss_additional_data.level            = coarse_level;
        fdss_additional_data.patch_variant    = TPSS::PatchVariant::cell;
        fdss_additional_data.smoother_variant = TPSS::SmootherVariant::additive;
        const auto patch_storage = std::make_shared<SubdomainHandler<dim, value_type_mg>>();
        patch_storage->reinit(mf_storage, fdss_additional_data);

        // *** setup Schwarz preconditioner
        typename CHEBPREC_COARSE::AdditionalData precondition_data;
        precondition_data.relaxation =
          TPSS::lookup_damping_factor(fdss_additional_data.patch_variant,
                                      fdss_additional_data.smoother_variant,
                                      dim);
        const auto schwarz_preconditioner = std::make_shared<CHEBPREC_COARSE>();
        schwarz_preconditioner->initialize(patch_storage, coarse_matrix, precondition_data);

        // *** setup smoother data
        using CHEB_COARSE = PreconditionChebyshev<LEVEL_MATRIX,
                                                  LinearAlgebra::distributed::Vector<value_type_mg>,
                                                  CHEBPREC_COARSE>;
        typename CHEB_COARSE::AdditionalData data_cheb;
        data_cheb.preconditioner      = schwarz_preconditioner;
        data_cheb.degree              = parameters.mg_coarse_chebyshev_degree;
        data_cheb.smoothing_range     = parameters.mg_coarse_chebyshev_reduction;
        data_cheb.eig_cg_n_iterations = coarse_matrix.m();
        // data_cheb.print_timings = true;
        auto mg_smoother_only_coarse = std::make_shared<
          MGSmootherRelaxation<LEVEL_MATRIX,
                               CHEB_COARSE,
                               LinearAlgebra::distributed::Vector<value_type_mg>>>();
        mg_smoother_only_coarse->initialize(mg_coarse_matrix, data_cheb);
        mg_coarse_smooth = mg_smoother_only_coarse;

        // *** setup coarse solver
        auto mg_coarse = std::make_shared<
          MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<value_type_mg>>>();
        mg_coarse->initialize(*mg_coarse_smooth);
        mg_coarse_grid = mg_coarse;
        break;
      }

      default:
        AssertThrow(false, ExcMessage("Coarse grid solver is not implemented!"));
    }

    time.stop();
    time_infos.emplace_back(time.wall_time(), "[MG] Build coarse-grid solver:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    pp_data.setup_time.back() += time.wall_time();
    time_infos.emplace_back(pp_data.setup_time.back(), "Total setup time:");
    fprint_time_details(time_infos.back().description, time_infos.back().time);
    time_details << std::endl;

    if(print_details)
    {
      // *** print dofs
      print_dof_info();

      // *** print detailed information of the numerical methods set up
      parameters.print(pcout);
      pcout << std::endl;
    }

    pcout << " ... set up linear system & multigrid!\n\n";

    return time_infos;
  }

  void
  compute_rhs()
  {
    system_rhs = 0.;

    FEEvaluation<dim, fe_degree> phi(*mf_storage);
    // RightHandSidePaperHermite<dim> rhs_function;
    // SolutionPaperHermite<dim>      exact_solution;
    RightHandSide<dim> rhs_function;
    Solution<dim>      exact_solution;
    for(unsigned int cell = 0; cell < mf_storage->n_cell_batches(); ++cell)
    {
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        VectorizedArray<double>             rhs_val     = VectorizedArray<double>();
        Point<dim, VectorizedArray<double>> point_batch = phi.quadrature_point(q);
        for(unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
        {
          Point<dim> single_point;
          for(unsigned int d = 0; d < dim; ++d)
            single_point[d] = point_batch[d][v];
          rhs_val[v] = rhs_function.value(single_point);
        }
        phi.submit_value(rhs_val, q);
      }
      phi.integrate_scatter(true, false, system_rhs);
    }

    FEFaceEvaluation<dim, fe_degree> phi_face(*mf_storage, true);
    for(unsigned int face = mf_storage->n_inner_face_batches();
        face < mf_storage->n_inner_face_batches() + mf_storage->n_boundary_face_batches();
        ++face)
    {
      phi_face.reinit(face);

      const VectorizedArray<double> inverse_length_normal_to_face =
        std::abs((phi_face.get_normal_vector(0) * phi_face.inverse_jacobian(0))[dim - 1]);
      const VectorizedArray<double> sigma =
        inverse_length_normal_to_face * system_matrix.get_penalty_factor();

      for(unsigned int q = 0; q < phi_face.n_q_points; ++q)
      {
        VectorizedArray<double> test_value              = VectorizedArray<double>(),
                                test_normal_gradient    = VectorizedArray<double>();
        Point<dim, VectorizedArray<double>> point_batch = phi_face.quadrature_point(q);

        for(unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
        {
          Point<dim> single_point;
          for(unsigned int d = 0; d < dim; ++d)
            single_point[d] = point_batch[d][v];

          test_value[v] = 2.0 * exact_solution.value(single_point);
        }
        phi_face.submit_value(test_value * sigma - test_normal_gradient, q);
        phi_face.submit_normal_derivative(-0.5 * test_value, q);
      }
      phi_face.integrate_scatter(true, true, system_rhs);
    }

    system_rhs.compress(VectorOperation::add);
  }

  double
  compute_n_iterations_fractional(const ReductionControl & solver_control)
  {
    const auto   res_data     = solver_control.get_history_data();
    const double first_res    = res_data.front();
    const double forelast_res = *(++res_data.rbegin());
    const double last_res     = res_data.back();

    const double reduction   = solver_control.reduction();
    const double tol         = first_res * reduction;
    const double rate_actual = std::log10(forelast_res / last_res);
    const double rate_needed = std::log10(forelast_res / tol);
    Assert(rate_actual > 0, ExcInternalError());
    Assert(rate_needed > 0, ExcInternalError());
    const double last_step_fraction = rate_needed / rate_actual;
    Assert(last_step_fraction < 1., ExcInternalError());
    const double n_fractional_iterations =
      static_cast<double>(solver_control.last_step() - 1 + last_step_fraction);
    return n_fractional_iterations;
  }

  std::vector<TimeInfo>
  solve(const bool print_details = false)
  {
    Timer                 time;
    std::vector<TimeInfo> time_infos;

    // *** setup multigrid object
    AssertThrow(mg_smoother_ptr != nullptr, ExcMessage("Pointer to MGSmoother is not set."));
    AssertThrow(mg_coarse_grid, ExcMessage("Pointer to MGSmoother is not set."));
    const MGSmootherBase<LinearAlgebra::distributed::Vector<value_type_mg>> * pre_smooth =
      mg_smoother_ptr;
    const MGSmootherBase<LinearAlgebra::distributed::Vector<value_type_mg>> * post_smooth =
      mg_smoother_ptr;
    if(mg_smoother_post)
      post_smooth = mg_smoother_post.get();
    mg::Matrix<LinearAlgebra::distributed::Vector<value_type_mg>> mg_matrix(mg_matrices);
    Multigrid<LinearAlgebra::distributed::Vector<value_type_mg>>  mg(
      mg_matrix,
      *mg_coarse_grid,
      mg_transfer,
      // *mg_smoother_ptr, // mg_smoother,
      // *mg_smoother_ptr, //*/mg_smoother,
      *pre_smooth,
      *post_smooth,
      mg_matrices.min_level(),
      mg_matrices.max_level());

    // *** setup multigrid signals
    const unsigned  mg_level_min = mg_matrices.min_level();
    const unsigned  mg_level_max = mg_matrices.max_level();
    MultigridSignal signal_coarse_solve{mg_level_min, mg_level_max};
    mg.connect_coarse_solve(std::ref(signal_coarse_solve));
    MultigridSignal signal_restriction{mg_level_min, mg_level_max};
    mg.connect_restriction(std::ref(signal_restriction));
    MultigridSignal signal_prolongation{mg_level_min, mg_level_max};
    mg.connect_prolongation(std::ref(signal_prolongation));
    MultigridSignal signal_pre_smooth{mg_level_min, mg_level_max};
    // TimeInfo mg_chebyshev_eigenestimate_time;
    // mg_chebyshev_eigenestimate_time.description = "[MG] Chebyshev eigenvalue estimation:";
    // const bool is_chebyshev_smoother
    //   = Parameter::MGSmootherVariant::ChebyshevSchwarz == parameters.mg_smoother_variant;
    // auto extended_signal_pre_smooth
    //   = [&,is_first_visit=bool{true}](const bool is_active, const unsigned level)mutable
    //   {
    // 	signal_pre_smooth (is_active, level);
    // 	if (is_chebyshev_smoother && is_first_visit && !is_active)
    // 	  {
    // 	    mg_chebyshev_eigenestimate_time.add_time (signal_pre_smooth.get_level_time(level).time);
    // 	    is_first_visit = (level != (mg_level_min+1));
    // 	  }
    //   };
    // mg.connect_pre_smoother_step (std::ref(extended_signal_pre_smooth));
    mg.connect_pre_smoother_step(std::ref(signal_pre_smooth));
    MultigridSignal signal_post_smooth{mg_level_min, mg_level_max};
    mg.connect_post_smoother_step(std::ref(signal_post_smooth));

    // *** setup multigrid preconditioner
    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<value_type_mg>,
                   MGTransferMatrixFree<dim, value_type_mg>>
      preconditioner(dof_handler, mg, mg_transfer);

    // *** setup time info mask of mg smoother to dismiss setup times later
    std::vector<char> active_time_mask_before;
    {
      const auto & schwarz_preconditioner = *(mg_schwarz_precondition[mg_level_max]);
      const auto   time_data              = schwarz_preconditioner.get_time_data();
      for(const auto & info : time_data)
        active_time_mask_before.emplace_back(info.is_active());
    }

    // *** setup solver data
    const double     tolerance{1.e-16};
    const double     reduction{parameters.solver_reduction};
    const unsigned   max_iterations{parameters.solver_max_iterations};
    ReductionControl solver_control(max_iterations, tolerance, reduction, true, true);
    solver_control.enable_history_data();

    // *** solve the linear system
    pcout << " Solve linear system ... \n\n";
    const auto & internal_solve = [&](auto && iterative_solver, const auto & preconditioner) {
      time.restart();
      iterative_solver.solve(system_matrix, solution, system_rhs, preconditioner);

      time.stop();
      pp_data.solve_time.push_back(time.wall_time());
    };

    switch(parameters.solver_variant)
    {
      case Parameter::SolverVariant::CG:
      {
        if(parameters.precondition_variant == Laplace::Parameter::PreconditionVariant::ID)
          internal_solve(SolverCG<LinearAlgebra::distributed::Vector<double>>(solver_control),
                         PreconditionIdentity{});
        else
          internal_solve(SolverCG<LinearAlgebra::distributed::Vector<double>>(solver_control),
                         preconditioner);
        break;
      }
      case Parameter::SolverVariant::GMRES:
      {
        if(parameters.precondition_variant == Laplace::Parameter::PreconditionVariant::ID)
          internal_solve(SolverGMRES<LinearAlgebra::distributed::Vector<double>>(solver_control),
                         PreconditionIdentity{});
        else
          internal_solve(SolverGMRES<LinearAlgebra::distributed::Vector<double>>(solver_control),
                         preconditioner);
        break;
      }
      default:
        AssertThrow(false, ExcMessage("Solver variant not implemented!"));
        break;
    }
    pcout << " ... done! \n\n";

    // *** gather and print timings
    auto time_infos_system_matrix{system_matrix.get_time_data()};
    for(auto & info : time_infos_system_matrix) // system matrix
    {
      info.description = "[Sys] " + info.description;
      time_infos.emplace_back(info);
    }
    // if(parameters.mg_coarse_grid_variant == Parameter::CoarseGridVariant::ChebyshevAccurate)
    //   if(auto mgsr_coarse_cheb = dynamic_cast<
    //        const MGSmootherRelaxation<LEVEL_MATRIX,
    //                                   CHEB_COARSE,
    //                                   LinearAlgebra::distributed::Vector<value_type_mg>> *>(
    //        mg_coarse_smooth.get()))
    //     // time_infos.emplace_back(
    //     //   mgsr_coarse_cheb->smoothers[mg_level_min].get_eigenestimate_timing(),
    //     //   "[MG] Chebyshev eigen-estimation (coarse):"); // Chebyshev as coarse-grid
    // if(parameters.mg_smoother_variant == Parameter::MGSmootherVariant::ChebyshevSchwarz)
    // {
    //   TimeInfo mg_chebyshev_eigenestimate_time;
    //   mg_chebyshev_eigenestimate_time.description =
    //     "[MG] Chebyshev eigen-estimation:"; // Chebyshev smoother
    //   for(unsigned int level = mg_level_min + 1; level < mg_level_max + 1; ++level)
    //     mg_chebyshev_eigenestimate_time.add_time(
    //       mg_smoother_cheb.smoothers[level].get_eigenestimate_timing());
    //   time_infos.emplace_back(mg_chebyshev_eigenestimate_time);
    // }
    time_infos.emplace_back(signal_coarse_solve.get_total_time("[MG] Coarse solve:"));
    time_infos.emplace_back(signal_restriction.get_total_time("[MG] Restriction:"));
    time_infos.emplace_back(signal_prolongation.get_total_time("[MG] Prolongation:"));
    time_infos.emplace_back(signal_pre_smooth.get_total_time("[MG] Pre-smooth:"));
    time_infos.emplace_back(signal_post_smooth.get_total_time("[MG] Post-smooth:"));
    const auto time_infos_smoother = std::move(get_time_info_schwarz()); // Schwarz smoother
    auto       was_active_before   = active_time_mask_before.cbegin();
    std::remove_copy_if(time_infos_smoother.begin(),
                        time_infos_smoother.end(),
                        std::back_inserter(time_infos),
                        [&](const auto & info) {
                          return info.is_active() == static_cast<bool>(*(was_active_before++));
                        });
    time_infos.emplace_back(time.wall_time(), "Total solve time:");

    time_details << "Solve time:\n";
    for(const auto & info : time_infos)
      fprint_time_details(info.description, info.time);
    time_details << std::endl;

    // *** post process residual data
    // const auto   n_fractional_iterations = compute_n_iterations_fractional(solver_control);
    const double reduction_rate =
      std::pow(solver_control.last_value() / solver_control.initial_value(),
               1. / solver_control.last_step());
    const double frac_steps = std::log(solver_control.reduction()) / std::log(reduction_rate);

    // pp_data.average_reduction_system.push_back(solver_control.average_reduction());
    // pp_data.n_iterations_system.push_back(n_fractional_iterations);
    pp_data.average_reduction_system.push_back(reduction_rate);
    pp_data.n_iterations_system.push_back(frac_steps);

    print_row_variable(
      pcout, 45, "  Average reduction:", reduction_rate); // solver_control.average_reduction());
    print_row_variable(pcout, 45, "  Fractional steps:", frac_steps);
    print_row_variable(pcout, 45, "  Iteration steps:", solver_control.last_step());
    pcout << std::endl;
    if(print_details)
    {
      pcout << "  Calls: \n";
      for(auto info : time_infos_system_matrix)
        print_row_variable(pcout, 2, "", 53, info.description, info.n_calls);
      auto time_infos_max_level{mg_matrices[mg_level_max].get_time_data()};
      time_infos_max_level.emplace_back(
        signal_pre_smooth.get_level_time(mg_level_max, "Pre-smooth:"));
      time_infos_max_level.emplace_back(
        signal_post_smooth.get_level_time(mg_level_max, "Post-smooth:"));
      time_infos_max_level.emplace_back(
        (mg_schwarz_precondition[mg_level_max]->get_time_data())[0]);
      time_infos_max_level.emplace_back(
        (mg_schwarz_precondition[mg_level_max]->get_time_data())[1]);
      for(auto info : time_infos_max_level)
        print_row_variable(pcout, 2, "", 53, "[MGmax] " + info.description, info.n_calls);
      pcout << std::endl;
    }

    return time_infos;
  }

  void
  print_mesh_info() const
  {
    const unsigned n_mg_levels = global_level - parameters.coarse_level + 1;
    pcout << " ... constructed mesh hierarchy (" << dim << "D) with " << n_mg_levels
          << " multigrid levels!\n\n";

    print_row_variable(pcout, 2, "", 43, "Geometry:", str_geometry);
    pcout << std::endl;

    print_row_variable(pcout, 5, "", 15, "level:", 15, "n_cells:");
    for(unsigned l = parameters.coarse_level; l < triangulation.n_levels(); ++l)
      print_row_variable(pcout, 5, "", 15, l, 15, triangulation.n_cells(l));
    pcout << std::endl;
  }

  void
  print_dof_info() const
  {
    print_row_variable(pcout, 2, "", 43, "Finite element:", fe->get_name());
    pcout << std::endl;

    print_row_variable(pcout, 5, "", 15, "level:", 15, "n_dofs:");
    for(unsigned l = parameters.coarse_level; l < triangulation.n_levels(); ++l)
      print_row_variable(pcout, 5, "", 15, l, 15, dof_handler.n_dofs(l));
    pcout << std::endl;
  }

  double
  compute_l2_error() const
  {
    const auto & mf_storage = *(system_matrix.get_matrix_free());
    // SolutionPaperHermite<dim> analytic_solution;
    Solution<dim> analytic_solution;

    double                                                 global_error  = 0;
    double                                                 global_volume = 0;
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(mf_storage);
    for(unsigned int cell = 0; cell < mf_storage.n_macro_cells(); ++cell)
    {
      phi.reinit(cell);
      phi.gather_evaluate(solution, true, false);
      VectorizedArray<Number> local_error  = VectorizedArray<Number>();
      VectorizedArray<Number> local_volume = VectorizedArray<Number>();
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        VectorizedArray<Number> exact_values;
        auto                    p_vec = phi.quadrature_point(q);
        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          Point<dim> p;
          for(unsigned int d = 0; d < dim; ++d)
            p[d] = p_vec[d][v];
          exact_values[v] = analytic_solution.value(p);
        }
        local_error +=
          (phi.get_value(q) - exact_values) * (phi.get_value(q) - exact_values) * phi.JxW(q);
        local_volume += phi.JxW(q);
      }
      for(unsigned int v = 0; v < mf_storage.n_active_entries_per_cell_batch(cell); ++v)
      {
        global_error += local_error[v];
        global_volume += local_volume[v];
      }
    }
    global_error  = Utilities::MPI::sum(global_error, MPI_COMM_WORLD);
    global_volume = Utilities::MPI::sum(global_volume, MPI_COMM_WORLD);
    return std::sqrt(global_error / global_volume);
  }

  void
  analyze_results() const
  {
    // // OLD OLD OLD
    Vector<float> error_per_cell(triangulation.n_active_cells());
    // VectorTools::integrate_difference(mapping,
    //                                   dof_handler,
    //                                   solution,
    //                                   SolutionPaperHermite<dim>{},
    //                                   error_per_cell,
    //                                   QGauss<dim>(fe->degree + 2),
    //                                   VectorTools::L2_norm);
    // const double L2_error =
    //   std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));
    // pp_data.L2_error.push_back(L2_error);
    // // NEW NEW NEW
    pp_data.L2_error.push_back(compute_l2_error());

    Vector<float> error_per_cell2(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>{},
                                      // SolutionPaperHermite<dim>{},
                                      error_per_cell,
                                      QGauss<dim>(fe->degree + 2),
                                      VectorTools::H1_seminorm);
    const double H1semi_error =
      std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));
    pp_data.H1semi_error.push_back(H1semi_error);
  }

  void
  post_process() const
  {
    if(!pp_fstream.is_active())
      return;

    fstream.open(parameters.filename, std::ios_base::app);
    pp_fstream << "Poisson Problem: ... TODO ... \n";
    parameters.print(pp_fstream);
    pp_fstream << std::endl;

    AssertThrow(pp_data.setup_time.size() == pp_data.solve_time.size(),
                ExcMessage("Mismatch of timing container in PostProcessData"));
    AssertThrow(pp_data.setup_time.size() == pp_data.n_dofs_global.size(),
                ExcMessage("Mismatch of timing container in PostProcessData"));
    print_row(pp_fstream, 20, "#DoFs (global)", "Setup: [s]", "Solve: [s]", "\n");
    for(std::size_t t = 0; t < pp_data.setup_time.size(); ++t)
      print_row(
        pp_fstream, 20, pp_data.n_dofs_global[t], pp_data.setup_time[t], pp_data.solve_time[t]);
    pp_fstream << std::endl;

    AssertThrow(pp_data.n_cells_global.size() == pp_data.n_dofs_global.size(),
                ExcMessage("Mismatch of system data container in PostProcessData"));
    AssertThrow(pp_data.n_cells_global.size() == pp_data.average_reduction_system.size(),
                ExcMessage("Mismatch of system data container in PostProcessData"));
    AssertThrow(pp_data.n_cells_global.size() == pp_data.n_iterations_system.size(),
                ExcMessage("Mismatch of system data container in PostProcessData"));
    const unsigned global_level_initial = parameters.n_refines;
    const unsigned mg_levels_initial    = global_level_initial - parameters.coarse_level + 1;
    print_row(pp_fstream,
              20,
              "#mg_levels",
              "#cells (global):",
              "#DoFs (global)",
              "#iterations",
              "avg. reduction",
              "\n");
    for(std::size_t t = 0; t < pp_data.setup_time.size(); ++t)
      print_row(pp_fstream,
                20,
                mg_levels_initial + t,
                pp_data.n_cells_global[t],
                pp_data.n_dofs_global[t],
                pp_data.n_iterations_system[t],
                pp_data.average_reduction_system[t]);
    pp_fstream << std::endl;

    if(parameters.compute_errors)
    {
      AssertThrow(pp_data.L2_error.size() == pp_data.n_dofs_global.size(),
                  ExcMessage("Mismatch of error container in PostProcessData"));
      AssertThrow(pp_data.H1semi_error.size() == pp_data.n_dofs_global.size(),
                  ExcMessage("Mismatch of error container in PostProcessData"));
      for(std::size_t t = 0; t < pp_data.setup_time.size(); ++t)
      {
        convergence_table.add_value("#DoFs", pp_data.n_dofs_global[t]);
        convergence_table.add_value("L2 error", pp_data.L2_error[t]);
        // convergence_table.add_value("H1semi error", pp_data.H1semi_error[t]);
      }

      convergence_table.set_precision("L2 error", 3);
      convergence_table.set_scientific("L2 error", true);
      // convergence_table.set_precision("H1semi error", 3);
      // convergence_table.set_scientific("H1semi error", true);

      convergence_table.evaluate_convergence_rates("L2 error", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates("L2 error",
                                                   "#DoFs",
                                                   ConvergenceTable::reduction_rate_log2,
                                                   dim);
      // convergence_table.evaluate_convergence_rates("H1semi error",
      //                                              ConvergenceTable::reduction_rate);
      // convergence_table.evaluate_convergence_rates("H1semi error",
      //                                              "#DoFs",
      //                                              ConvergenceTable::reduction_rate_log2,
      //                                              dim);
      if(pp_fstream.is_active())
        convergence_table.write_text(pp_fstream.get_stream());
    }

    pp_fstream << "\n\n\n";
    fstream.close();
  }

  void
  write_output(const bool print_output = true)
  {
    if(!print_output)
      return;

    Assert(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
           ExcMessage("Demanding one MPI process only!"));

    std::string  filename = "solution-" + Utilities::int_to_string(global_level, 2);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping,
                           /*n_subdivisions*/ fe_degree,
                           DataOut<dim>::CurvedCellRegion::curved_inner_cells);

    fstream.open(filename + ".vtu", std::ios_base::out);
    data_out.write_vtu(fstream);
    fstream.close();
  }

  std::vector<std::vector<TimeInfo>>
  run(bool print_details = true)
  {
    std::vector<std::vector<TimeInfo>> time_infos; // infos per cycle
    for(unsigned cycle = 0; cycle < parameters.n_cycles; ++cycle)
    {
      const bool is_last_cycle = cycle == parameters.n_cycles - 1;
      print_details            = print_details && is_last_cycle;

      // if(cycle > 0)
      const bool tria_was_created =
        create_triangulation(parameters.n_refines + cycle, print_details);
      if(!tria_was_created)
        continue;

      pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());

      auto time_infos_setup = setup(/*build_rhs?*/ true, print_details);
      pp_data.n_dofs_global.push_back(dof_handler.n_dofs());
      n_timings_setup = time_infos_setup.size();

      auto time_infos_solve = solve(print_details);
      n_timings_solve       = time_infos_solve.size();

      if(parameters.compute_errors)
        analyze_results();
      write_output(/*print_output?*/ parameters.write_fe_output);

      for(const auto & info : time_infos_solve)
        time_infos_setup.emplace_back(info);
      time_infos.emplace_back(time_infos_setup);
      pcout << " .. finished cycle " << cycle << std::endl;
    }
    // post_process();

    return time_infos;
  }

  void
  assemble_matrix()
  {
    // *** initialize sparse matrix
    Assert(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
           ExcMessage(
             "In order to use the serial sparse matrix format only 1 mpi process is allowed!"));
    const unsigned int     n_dofs = dof_handler.n_dofs();
    DynamicSparsityPattern dsp(n_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity.copy_from(dsp);
    sparse_matrix.reinit(sparsity);

    MeshWorker::IntegrationInfoBox<dim> info_box;
    UpdateFlags                         update_flags = update_values | update_gradients;
    info_box.add_update_flags_all(update_flags);
    info_box.initialize_gauss_quadrature(n_q_points, n_q_points, n_q_points);
    info_box.initialize(*fe, mapping);

    MeshWorker::DoFInfo<dim>                                  dof_info(dof_handler);
    MeshWorker::Assembler::MatrixSimple<SparseMatrix<Number>> assembler;
    assembler.initialize(sparse_matrix);
    Laplace::MW::MatrixIntegrator<dim> integrator;
    MeshWorker::integration_loop<dim, dim>(
      dof_handler.begin_active(), dof_handler.end(), dof_info, info_box, integrator, assembler);
  }

  std::vector<TimeInfo>
  get_time_info_schwarz()
  {
    const unsigned        mg_level_min = mg_schwarz_precondition.min_level();
    const unsigned        mg_level_max = mg_schwarz_precondition.max_level();
    std::vector<TimeInfo> time_infos_smoother;
    for(unsigned int level = mg_level_min + 1; level < mg_level_max + 1; ++level)
    {
      const auto & schwarz_preconditioner = *(mg_schwarz_precondition[level]);
      const auto   time_data              = schwarz_preconditioner.get_time_data();
      auto         time_info              = time_infos_smoother.begin();
      if(time_infos_smoother.empty())
        time_infos_smoother = time_data;
      else
        for(auto input = time_data.cbegin(); input != time_data.cend(); ++input, ++time_info)
          *time_info += *input;
    }
    return time_infos_smoother;
  }
};

} // end namespace Laplace

#endif /* TESTS_LAPLACEPROBLEM_H_ */
