#ifndef APPS_STOKESPROBLEMNOBLOCKS_H_
#define APPS_STOKESPROBLEMNOBLOCKS_H_

/// based on step-56
/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 * Author: Ryan Grove, Clemson University
 *         Timo Heister, Clemson University
 */

/**
 * Stokes problem...
 *
 * Created on: May 18, 2020
 *     Author: witte
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/sparse_ilu.h>


#include <fstream>
#include <iostream>


#include "coloring.h"
#include "equation_data.h"
#include "multigrid.h"
#include "postprocess.h"
#include "rt_parameter.h"
#include "solvers_and_preconditioners/TPSS/move_to_deal_ii.h"
#include "stokes_integrator.h"

namespace Stokes
{
namespace NoBlocks
{
using namespace dealii;


/**
 * TODO...
 */
template<int dim,
         int fe_degree_p,
         typename Number                = double,
         TPSS::DoFLayout dof_layout_v   = TPSS::DoFLayout::Q,
         int             fe_degree_v    = fe_degree_p + 1,
         LocalAssembly   local_assembly = LocalAssembly::Tensor>
class SparseMatrixAugmented
  : public TrilinosWrappers::SparseMatrix,
    public VelocityPressure::FD::
      MatrixIntegrator<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>
{
  static_assert(std::is_same<Number, double>::value, "Trilinos supports double-precision.");

public:
  using value_type            = Number;
  using matrix_type           = TrilinosWrappers::SparseMatrix;
  using local_integrator_type = VelocityPressure::FD::
    MatrixIntegrator<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>;

  void
  initialize(const TrilinosWrappers::SparsityPattern & dsp,
             const IndexSet &                          locally_owned_dof_indices,
             const IndexSet &                          ghosted_dof_indices,
             const MPI_Comm &                          mpi_communicator);

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in);

  void
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> & vec) const;

  void
  clear();

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const;

  template<typename VectorType>
  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    matrix_type::vmult(dst, src);
  }

  void
  vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const;

  operator const FullMatrix<Number> &() const;

  std::shared_ptr<const MatrixFree<dim, Number>>     mf_storage;
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
  mutable std::shared_ptr<FullMatrix<Number>>        fullmatrix;
};



template<typename Number>
class BlockSparseMatrixFiltered : public BlockSparseMatrix<Number>
{
public:
  using matrix_type = BlockSparseMatrix<Number>;

  void
  vmult(BlockVector<Number> & dst, const BlockVector<Number> & src) const
  {
    matrix_type::vmult(dst, src);

    if(constant_pressure_mode)
    {
      auto & dst_pressure = dst.block(1);
      AssertDimension(constant_pressure_mode->size(), dst_pressure.size());
      const Number inner_product_value = (*constant_pressure_mode) * dst_pressure;
      dst_pressure.add(-inner_product_value, *constant_pressure_mode);
    }
  }

  const Vector<double> * constant_pressure_mode = nullptr;
};



/**
 * The base class defines compile time parameters depending on the choice of the
 * finite element method.
 */
template<Method method, int dim, int fe_degree_p>
struct ModelProblemBase
{
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::TaylorHood, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::Q;
  using fe_type_v                               = FE_Q<dim>;
  using fe_type_p                               = FE_Q<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Tensor;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::TaylorHoodDGQ, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::DGQ;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::Q;
  using fe_type_v                               = FE_DGQ<dim>;
  using fe_type_p                               = FE_Q<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Tensor;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::Qkplus2_DGPk, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGP;
  using fe_type_v                               = FE_Q<dim>;
  using fe_type_p                               = FE_DGP<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 2;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Cut;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::DGQkplus2_DGPk, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::DGQ;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGP;
  using fe_type_v                               = FE_DGQ<dim>;
  using fe_type_p                               = FE_DGP<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Cut;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::RaviartThomas, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::RT;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGQ;
  // using fe_type_v                               = FE_RaviartThomasNodal_new<dim>;
  using fe_type_v                               = FE_RaviartThomas<dim>;
  using fe_type_p                               = FE_DGQLegendre<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p;
  static constexpr LocalAssembly local_assembly = LocalAssembly::Cut;
};



/**
 * This is the main class which combines the typical building blocks of a
 * (deal.II) finite element method.
 *
 * TODO Description...
 */
template<int dim, int fe_degree_p, Method method = Method::TaylorHood>
class ModelProblem : public ModelProblemBase<method, dim, fe_degree_p>
{
  static_assert(dim == 2, "only 2D");

  using Base = ModelProblemBase<method, dim, fe_degree_p>;

public:
  using Base::dof_layout_p;
  using Base::dof_layout_v;
  using Base::fe_type_p;
  using Base::fe_type_v;
  using Base::local_assembly;
  using vector_type = LinearAlgebra::distributed::Vector<double>;

  static constexpr int fe_degree_v = Base::fe_degree_v;
  static constexpr int n_q_points_1d =
    fe_degree_v + 1 + (dof_layout_v == TPSS::DoFLayout::RT ? 1 : 0);

  ModelProblem(const RT::Parameter & rt_parameters_in, const EquationData & equation_data_in);

  void
  run();

  bool
  make_grid();

  bool
  make_grid(const unsigned int n_refinements);

  void
  setup_system();

  std::shared_ptr<LinearAlgebra::distributed::Vector<double>>
  compute_mass_foreach_pressure_dof() const;

  std::shared_ptr<LinearAlgebra::distributed::Vector<double>>
  compute_mass_foreach_pressure_dof(const unsigned int level) const;

  void
  setup_system_velocity(const bool do_cuthill_mckee);

  void
  setup_system_pressure(const bool do_cuthill_mckee);

  void
  assemble_system();

  void
  assemble_system_velocity_pressure();

  void
  solve();

  void
  correct_mean_value_pressure();

  std::shared_ptr<Vector<double>>
  compute_L2_error_velocity() const;

  std::shared_ptr<Vector<double>>
  compute_L2_error_pressure() const;

  std::shared_ptr<Vector<double>>
  compute_H1semi_error_velocity() const;

  void
  compute_errors();

  void
  output_results(const unsigned int refinement_cycle) const;

  unsigned int
  max_level() const;

  unsigned int
  n_colors_system();

  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const;

  void
  print_informations() const;

  std::vector<types::global_dof_index>
  shift_pressure_dof_indices(const std::vector<types::global_dof_index> & dof_indices) const
  {
    std::vector<types::global_dof_index> shifted_dof_indices;
    types::global_dof_index              n_velocity_dofs = dof_handler_velocity.n_dofs();
    std::transform(dof_indices.cbegin(),
                   dof_indices.cend(),
                   std::back_inserter(shifted_dof_indices),
                   [&](const auto i) { return i + n_velocity_dofs; });
    return shifted_dof_indices;
  }

  std::shared_ptr<SolverControl>
  get_solver_control() const;

  const FiniteElement<dim> &
  get_fe_velocity() const;

  const FiniteElement<dim> &
  get_fe_pressure() const;

  const unsigned int                  mpi_rank;
  const bool                          is_first_proc;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  std::shared_ptr<Function<dim>>      analytical_solution;
  std::shared_ptr<Function<dim>>      load_function;
  mutable PostProcessData             pp_data;
  mutable PostProcessData             pp_data_pressure;

  parallel::distributed::Triangulation<dim> triangulation;
  MappingQ<dim>                             mapping;
  std::shared_ptr<FiniteElement<dim>>       fe;
  DoFHandler<dim>                           dof_handler;
  DoFHandler<dim>                           dof_handler_velocity;
  DoFHandler<dim>                           dof_handler_pressure;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> zero_constraints_velocity;
  AffineConstraints<double> mean_value_constraints;
  AffineConstraints<double> constraints_velocity;
  AffineConstraints<double> constraints_pressure;

  Table<2, DoFTools::Coupling> cell_integrals_mask;
  Table<2, DoFTools::Coupling> face_integrals_mask;

  SparseMatrixAugmented<dim, fe_degree_p, double, dof_layout_v, fe_degree_v, local_assembly>
    system_matrix;

  vector_type                                system_solution;
  vector_type                                system_delta_x;
  vector_type                                system_rhs;
  LinearAlgebra::distributed::Vector<double> constant_pressure_mode;

private:
  std::shared_ptr<FiniteElement<dim>>
  generate_fe() const;

  bool
  check_finite_elements() const;

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner, const std::string solver_variant);

  void
  make_grid_impl(const MeshParameter & mesh_prms);

  template<bool is_multigrid = false>
  std::shared_ptr<LinearAlgebra::distributed::Vector<double>>
  compute_mass_foreach_pressure_dof_impl(
    const unsigned int level = numbers::invalid_unsigned_int) const;
};



////////// Definitions



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in)
{
  mf_storage = mf_storage_in;
  local_integrator_type::initialize(equation_data_in);
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::
  initialize(const TrilinosWrappers::SparsityPattern & dsp,
             const IndexSet &                          locally_owned_dof_indices,
             const IndexSet &                          ghosted_dof_indices,
             const MPI_Comm &                          mpi_communicator)
{
  matrix_type::reinit(dsp);
  partitioner = std::make_shared<const Utilities::MPI::Partitioner>(locally_owned_dof_indices,
                                                                    ghosted_dof_indices,
                                                                    mpi_communicator);
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> & vec) const
{
  Assert(partitioner, ExcMessage("Did you initialize partitioner?"));
  vec.reinit(partitioner);
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::clear()
{
  fullmatrix.reset();
  mf_storage.reset();
  partitioner.reset();
  matrix_type::clear();
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
std::shared_ptr<const MatrixFree<dim, Number>>
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::
  get_matrix_free() const
{
  AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  return mf_storage;
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
void
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::vmult(
  const ArrayView<Number>       dst,
  const ArrayView<const Number> src) const
{
  AssertThrow(false, ExcMessage("TODO"));
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("No MPI support"));
  // AssertDimension(dst.size(), matrix_type::m());
  // AssertDimension(src.size(), matrix_type::n());
  // std::vector<typename matrix_type::size_type> n_rows_per_block;
  // AssertDimension(matrix_type::n_block_rows(), matrix_type::n_block_cols());
  // for(auto b = 0U; b < matrix_type::n_block_rows(); ++b)
  //   n_rows_per_block.emplace_back(matrix_type::block(b, b).m());
  // BlockVector<Number> v(n_rows_per_block); // src
  // std::copy(src.cbegin(), src.cend(), v.begin());
  // BlockVector<Number> w(n_rows_per_block); // dst
  // matrix_type::vmult(w, v);                // w = A v
  // std::copy(w.begin(), w.end(), dst.begin());
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly>
SparseMatrixAugmented<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v, local_assembly>::
operator const FullMatrix<Number> &() const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("No MPI support"));
  if(!fullmatrix)
  {
    const auto & tmp = Tensors::matrix_to_table(*this);
    fullmatrix       = std::make_shared<FullMatrix<Number>>(table_to_fullmatrix(tmp));
  }
  return *fullmatrix;
}



template<int dim, int fe_degree_p, Method method>
ModelProblem<dim, fe_degree_p, method>::ModelProblem(const RT::Parameter & rt_parameters_in,
                                                     const EquationData &  equation_data_in)
  : mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
    is_first_proc(mpi_rank == 0U),
    pcout(std::make_shared<ConditionalOStream>(std::cout, is_first_proc)),
    rt_parameters(rt_parameters_in),
    equation_data(equation_data_in),
    analytical_solution([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::DivFree)
        return std::make_shared<DivergenceFree::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipNormal)
        return std::make_shared<DivergenceFree::NoSlipNormal::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeBell)
        return std::make_shared<DivergenceFree::GaussianBell::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleNoSlip)
        return std::make_shared<DivergenceFree::Poiseuille::NoSlip::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlip)
        return std::make_shared<DivergenceFree::NoSlip::Solution<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleInhom)
        return std::make_shared<DivergenceFree::Poiseuille::Inhom::Solution<dim>>();
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    load_function([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::DivFree)
        return std::make_shared<DivergenceFree::Load<dim>>();
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipNormal)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeBell)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleNoSlip)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlip)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution); // !!!
      // return std::make_shared<DivergenceFree::NoSlip::Load<dim>>(); // !!!
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleInhom)
        return std::make_shared<ManufacturedLoad<dim>>(analytical_solution);
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    // triangulation(Triangulation<dim>::maximum_smoothing),
    mapping(1),
    // Finite element for the whole system:
    fe(generate_fe()),
    dof_handler(triangulation)
// /// TODO !!! coloring depends on the discrete pressure space as well
// user_coloring([&]() -> std::shared_ptr<ColoringBase<dim>> {
//   if constexpr(dof_layout_v == TPSS::DoFLayout::Q)
//     return std::make_shared<TiledColoring<dim>>(rt_parameters_in.mesh);
//   else if(dof_layout_v == TPSS::DoFLayout::DGQ)
//     return std::make_shared<RedBlackColoring<dim>>(rt_parameters_in.mesh);
//   return std::shared_ptr<ColoringBase<dim>>();
// }()),
// mgc_velocity(rt_parameters_in.multigrid, equation_data_in),
// mgc_velocity_pressure(rt_parameters_in.multigrid, equation_data_in)
{
  Assert(check_finite_elements(), ExcMessage("Check default finite elements and dof_layout."));
}



template<int dim, int fe_degree_p, Method method>
unsigned int
ModelProblem<dim, fe_degree_p, method>::max_level() const
{
  return triangulation.n_global_levels() - 1;
}



template<int dim, int fe_degree_p, Method method>
unsigned int
ModelProblem<dim, fe_degree_p, method>::n_colors_system()
{
  // auto & mgc = mgc_velocity;
  // if(mgc.mg_schwarz_smoother_pre)
  //   return mgc.mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
  return numbers::invalid_unsigned_int;
}



template<int dim, int fe_degree_p, Method method>
template<typename T>
void
ModelProblem<dim, fe_degree_p, method>::print_parameter(const std::string & description,
                                                        const T &           value) const
{
  *pcout << Util::parameter_to_fstring(description, value);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::print_informations() const
{
  *pcout << equation_data.to_string();
  *pcout << std::endl;
  print_parameter("Finite element:", fe->get_name());
  *pcout << rt_parameters.to_string();
  *pcout << std::endl;
}



template<int dim, int fe_degree_p, Method method>
const FiniteElement<dim> &
ModelProblem<dim, fe_degree_p, method>::get_fe_velocity() const
{
  std::vector<bool> velocity_mask(dim, true);
  velocity_mask.push_back(false);
  const auto & fe_velocity = fe->get_sub_fe(velocity_mask);
  return fe_velocity;
}


template<int dim, int fe_degree_p, Method method>
const FiniteElement<dim> &
ModelProblem<dim, fe_degree_p, method>::get_fe_pressure() const
{
  std::vector<bool> pressure_mask(dim, false);
  pressure_mask.push_back(true);
  const auto & fe_pressure = fe->get_sub_fe(pressure_mask);
  return fe_pressure;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<FiniteElement<dim>>
ModelProblem<dim, fe_degree_p, method>::generate_fe() const
{
  if constexpr(dof_layout_v == TPSS::DoFLayout::RT)
    return std::make_shared<FESystem<dim>>(typename Base::fe_type_v(fe_degree_v),
                                           1,
                                           typename Base::fe_type_p(fe_degree_p),
                                           1);
  else
    return std::make_shared<FESystem<dim>>(FESystem<dim>(typename Base::fe_type_v(fe_degree_v),
                                                         dim),
                                           1,
                                           typename Base::fe_type_p(fe_degree_p),
                                           1);
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::check_finite_elements() const
{
  //: check fe
  AssertDimension(fe->n_base_elements(), 2); // velocity + pressure
  const auto & fe_v = fe->base_element(0);
  AssertDimension(fe_v.n_components(), dim); // velocity
  AssertDimension(fe_v.element_multiplicity(0), dof_layout_v == TPSS::DoFLayout::RT ? 1 : dim);
  AssertThrow(TPSS::get_dof_layout(fe_v.base_element(0)) == dof_layout_v,
              ExcMessage("velocity part of fe and dof_layout are incompatible."));
  const auto & fe_p = fe->base_element(1);
  AssertDimension(fe_p.n_components(), 1); // pressure

  return true;
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::make_grid()
{
  make_grid_impl(rt_parameters.mesh);
  return true;
}



template<int dim, int fe_degree_p, Method method>
bool
ModelProblem<dim, fe_degree_p, method>::make_grid(const unsigned int n_refinements)
{
  MeshParameter mesh_prms = rt_parameters.mesh;
  mesh_prms.n_refinements = n_refinements;

  //: estimate number of dofs (velocity + pressure)
  AssertDimension(fe->n_base_elements(), 2); // velocity + pressure
  const auto & fe_v = fe->base_element(0);
  AssertDimension(fe_v.n_components(), dim); // velocity
  const auto n_dofs_est_v = [&]() {
    if(fe_v.element_multiplicity(0) == dim)
      return dim * estimate_n_dofs(fe_v.base_element(0), mesh_prms);
    return estimate_n_dofs(fe_v, mesh_prms);
  }();
  const auto & fe_p = fe->base_element(1);
  AssertDimension(fe_p.n_components(), 1); // pressure
  const auto n_dofs_est_p = estimate_n_dofs(fe_p, mesh_prms);
  const auto n_dofs_est   = n_dofs_est_v + n_dofs_est_p;
  if(rt_parameters.exceeds_dof_limits(n_dofs_est))
    return false;

  make_grid_impl(mesh_prms);
  return true;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::make_grid_impl(const MeshParameter & mesh_prms)
{
  triangulation.clear();
  *pcout << create_mesh(triangulation, mesh_prms) << std::endl;
  pp_data.n_cells_global.push_back(triangulation.n_global_active_cells());
  pp_data.n_dimensions = dim;
}



template<int dim, int fe_degree_p, Method method>
template<bool is_multigrid>
std::shared_ptr<LinearAlgebra::distributed::Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_mass_foreach_pressure_dof_impl(
  const unsigned int level) const
{
  const auto n_dofs_pressure = [this, level]() {
    if(is_multigrid)
      return dof_handler_pressure.n_dofs(level);
    else
      return dof_handler_pressure.n_dofs();
  }();

  AssertThrow(!is_multigrid, ExcMessage("TODO MPI..."));

  const auto & locally_owned_dof_indices = dof_handler_pressure.locally_owned_dofs();
  IndexSet     locally_relevant_dof_indices;
  DoFTools::extract_locally_relevant_dofs(dof_handler_pressure, locally_relevant_dof_indices);

  std::shared_ptr<LinearAlgebra::distributed::Vector<double>> mass_foreach_dof_ptr; // book-keeping
  mass_foreach_dof_ptr =
    std::make_shared<LinearAlgebra::distributed::Vector<double>>(locally_owned_dof_indices,
                                                                 locally_relevant_dof_indices,
                                                                 MPI_COMM_WORLD);

  AffineConstraints<double> constraints_pressure;
  constraints_pressure.clear();
  constraints_pressure.reinit(locally_relevant_dof_indices);
  constraints_pressure.close();

  TrilinosWrappers::SparsityPattern dsp(locally_owned_dof_indices,
                                        locally_owned_dof_indices,
                                        locally_relevant_dof_indices,
                                        MPI_COMM_WORLD);
  if(is_multigrid)
    MGTools::make_sparsity_pattern(dof_handler_pressure, dsp, level);
  else
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp, constraints_pressure);
  dsp.compress();

  TrilinosWrappers::SparseMatrix pressure_mass_matrix;
  pressure_mass_matrix.reinit(dsp);

  using Pressure::MW::CopyData;
  using Pressure::MW::ScratchData;
  using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, is_multigrid>;

  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    matrix_integrator.cell_mass_worker(cell, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    constraints_pressure.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
      copy_data.cell_matrix, copy_data.local_dof_indices, pressure_mass_matrix);
  };

  const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
  const UpdateFlags interface_update_flags = update_default;

  ScratchData<dim> scratch_data(
    mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);
  CopyData copy_data(dof_handler_pressure.get_fe().dofs_per_cell);
  if(is_multigrid)
    MeshWorker::mesh_loop(dof_handler_pressure.begin_mg(level),
                          dof_handler_pressure.end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  else
    MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
                          dof_handler_pressure.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);

  pressure_mass_matrix.compress(VectorOperation::add);

  LinearAlgebra::distributed::Vector<double> constant_one_vector(
    mass_foreach_dof_ptr->get_partitioner());
  std::fill(constant_one_vector.begin(), constant_one_vector.end(), 1.);
  pressure_mass_matrix.vmult(*mass_foreach_dof_ptr, constant_one_vector);

  if(is_first_proc)
    AssertThrow((*mass_foreach_dof_ptr)(0) > 0., ExcMessage("First DoF has no positive mass."));
  return mass_foreach_dof_ptr;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<LinearAlgebra::distributed::Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_mass_foreach_pressure_dof() const
{
  return compute_mass_foreach_pressure_dof_impl<false>();
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<LinearAlgebra::distributed::Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_mass_foreach_pressure_dof(
  const unsigned int level) const
{
  return compute_mass_foreach_pressure_dof_impl<true>(level);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system_velocity(const bool do_cuthill_mckee)
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  /// distribute dofs and initialize MGCollection
  dof_handler_velocity.initialize(triangulation, get_fe_velocity());
  if(do_cuthill_mckee)
  {
    AssertThrow(false, ExcMessage("TODO MPI"));
    DoFRenumbering::Cuthill_McKee(dof_handler_velocity);
  }

  /// mpi-relevant dof indices
  IndexSet locally_relevant_dof_indices;
  DoFTools::extract_locally_relevant_dofs(dof_handler_velocity, locally_relevant_dof_indices);

  /// homogeneous boundary conditions for the solution update
  zero_constraints_velocity.clear();
  zero_constraints.reinit(locally_relevant_dof_indices);
  if(dof_layout_v == TPSS::DoFLayout::Q)
  {
    print_parameter("Interpolating zero boundary (velo)", "...");
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      DoFTools::make_zero_boundary_constraints(dof_handler_velocity,
                                               boundary_id,
                                               zero_constraints_velocity);
  }
  else if(dof_layout_v == TPSS::DoFLayout::RT)
  {
    print_parameter("Projecting div-conf. zero boundary (velo)", "...");
    Functions::ZeroFunction<dim> zero_velocity(dim);
    /// We use dof_handler by purpose bypassing the assertion in
    /// project_boundary_values_div_conforming() since finite elements of type
    /// FE_System are not checked for compatibility. Otherwise
    /// project_boundary_values_div_conforming() throws an exception if the
    /// finite element isn't of type FE_RaviartThomas.
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      VectorToolsFix::project_boundary_values_div_conforming(
        dof_handler_velocity, 0U, zero_velocity, boundary_id, zero_constraints_velocity, mapping);
    /// !!! use VectorTools above?
  }
  zero_constraints_velocity.close();

  /// inhomogeneous boundary conditions for the particular solution
  constraints_velocity.clear();
  constraints_velocity.reinit(locally_relevant_dof_indices);
  if(dof_layout_v == TPSS::DoFLayout::Q)
  {
    print_parameter("Interpolating zero boundary (velo)", "...");
    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
    std::map<types::boundary_id, const Function<dim, double> *> boundary_id_to_function;
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      boundary_id_to_function.emplace(boundary_id, &analytical_solution_velocity);
    VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                             boundary_id_to_function,
                                             constraints_velocity);
  }
  else if(dof_layout_v == TPSS::DoFLayout::RT)
  {
    print_parameter("Projecting div-conf. boundary (velo)", "...");
    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
    /// We use dof_handler by purpose here bypassing the assertion in
    /// project_boundary_values_div_conforming(), since the underlying finite
    /// element is of type FE_System and therefore not checked (otherwise the
    /// function presumes a finite element of type FE_RaviartThomas!).
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      VectorToolsFix::project_boundary_values_div_conforming(dof_handler_velocity,
                                                             0U,
                                                             analytical_solution_velocity,
                                                             boundary_id,
                                                             constraints_velocity,
                                                             mapping);
    /// !!! use VectorTools above?
  }
  constraints_velocity.close();
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system_pressure(const bool do_cuthill_mckee)
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  /// distributing (and reordering) dofs
  dof_handler_pressure.initialize(triangulation, get_fe_pressure());
  if(do_cuthill_mckee)
  {
    AssertThrow(false, ExcMessage("TODO MPI"));
    const bool cuthill_mckee_is_compatible = dof_layout_v == dof_layout_p;
    AssertThrow(
      cuthill_mckee_is_compatible,
      ExcMessage(
        "In general, reordering velocity as well as pressure dofs by a Cuthill-McKee algorithm does not provide the same order as a Cuthill-McKee reordering on the combined velocity-pressure dofs. If the same dof layout is used for the velocity and pressure the reorderings might coincide."));
    DoFRenumbering::Cuthill_McKee(dof_handler_pressure);
  }

  /// mpi-relevant dof indices
  const auto & locally_owned_dof_indices = dof_handler_pressure.locally_owned_dofs();
  IndexSet     locally_relevant_dof_indices;
  DoFTools::extract_locally_relevant_dofs(dof_handler_pressure, locally_relevant_dof_indices);

  /// mean-value constraints: Use the space of mean-value free L^2 functions as
  /// pressure ansatz space. Therefore, compute the (unconstrained) pressure
  /// mass matrix and apply the coefficient vector which interpolates the
  /// constant-one-function. This leads to a weight for each degree of
  /// freedom. If the sum of these weights is zero the discrete pressure
  /// function is mean-value free.
  constraints_pressure.clear();
  constraints_pressure.reinit(locally_relevant_dof_indices);
  if(equation_data.force_mean_value_constraint)
  {
    print_parameter("Computing mean-value constraints (press)", "...");
    const types::global_dof_index first_pressure_dof = 0U;
    if(locally_owned_dof_indices.is_element(first_pressure_dof))
      constraints_pressure.add_line(first_pressure_dof);

    // const auto   mass_foreach_dof_ptr = compute_mass_foreach_pressure_dof();
    // const auto & mass_foreach_dof     = *mass_foreach_dof_ptr;

    // const bool is_dgq_legendre =
    //   dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
    // const bool is_legendre_type = is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP;
    // const auto n_dofs_pressure  = dof_handler_pressure.n_dofs();
    // {
    //   /// !!! TODO communication missing for more than one proc...
    //   const double mass_of_first_dof = mass_foreach_dof(0);
    //   constraints_pressure.add_line(0U);
    //   const auto         n_dofs_per_cell = get_fe_pressure().dofs_per_cell;
    //   const unsigned int stride          = is_legendre_type ? n_dofs_per_cell : 1U;
    //   const unsigned int start           = is_legendre_type ? stride : 1U;
    //   for(auto i = start; i < n_dofs_pressure; i += stride)
    //     constraints_pressure.add_entry(0U, i, -mass_foreach_dof(i) / mass_of_first_dof);
    // }
  }
  constraints_pressure.close();
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::setup_system()
{
  Assert(check_finite_elements(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  dof_handler.initialize(triangulation, *fe);

  /// ILU behaves better if we apply a reordering to reduce fillin.
  if(equation_data.use_cuthill_mckee)
  {
    AssertThrow(false, ExcMessage("TODO MPI"));
    const bool cuthill_mckee_is_compatible = dof_layout_v == dof_layout_p;
    const bool cuthill_mckee_pays_off =
      rt_parameters.solver.variant == "FGMRES_ILU" && cuthill_mckee_is_compatible;
    AssertThrow(
      cuthill_mckee_pays_off,
      ExcMessage(
        "For the current solver setting a Cuthill-McKee renumbering seems not be worthwile."));
    DoFRenumbering::Cuthill_McKee(dof_handler);
  }

  // This ensures that all velocity dofs are enumerated before the pressure
  // unknowns. This allows us to use blocks for vectors and matrices and allows
  // us to get the same DoF numbering for dof_handler and its unmerged
  // counterparts dof_handler_velocity and dof_handler_pressure.
  std::vector<unsigned int> component_mask(dim + 1, 0U);
  component_mask[dim] = 1U; // pressure
  DoFRenumbering::component_wise(dof_handler, component_mask);

  const auto & locally_owned_dof_indices = dof_handler.locally_owned_dofs();
  IndexSet     locally_relevant_dof_indices;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dof_indices);

  std::vector<unsigned int>                  block_component{0U, 1U};
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_dofs_velocity = dofs_per_block[0];
  const unsigned int n_dofs_pressure = dofs_per_block[1];

  setup_system_velocity(equation_data.use_cuthill_mckee);

  setup_system_pressure(equation_data.use_cuthill_mckee);

  /// No-slip boundary conditions (velocity)
  zero_constraints.clear();
  zero_constraints.reinit(locally_relevant_dof_indices);
  if(dof_layout_v == TPSS::DoFLayout::Q)
  {
    print_parameter("Interpolating zero boundary", "...");
    const FEValuesExtractors::Vector velocities(0);
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               boundary_id,
                                               zero_constraints,
                                               fe->component_mask(velocities));
  }
  else if(dof_layout_v == TPSS::DoFLayout::RT)
  {
    print_parameter("Projecting div-conf. zero boundary", "...");
    Functions::ZeroFunction<dim> zero_velocity(dim);
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      VectorToolsFix::project_boundary_values_div_conforming(
        dof_handler, 0U, zero_velocity, boundary_id, zero_constraints, mapping);
  }
  zero_constraints.close();

  {
    mean_value_constraints.clear();
    mean_value_constraints.reinit(locally_relevant_dof_indices);

    /// Mean value condition (pressure) ...
    if(equation_data.force_mean_value_constraint)
    {
      print_parameter("Computing mean-value constraints", "...");
      const types::global_dof_index offset_pressure_dofs = n_dofs_velocity;
      if(constraints_pressure.n_constraints() > 0U)
      {
        for(const auto line : constraints_pressure.get_lines())
        {
          /// NOTE AffineConstraint::shift() did not work due to issues with
          /// locally relevant index sets.
          const auto row = line.index + offset_pressure_dofs;
          std::vector<std::pair<types::global_dof_index, double>> entries;
          std::transform(line.entries.cbegin(),
                         line.entries.cend(),
                         std::back_inserter(entries),
                         [&](auto entry) {
                           entry.first += offset_pressure_dofs;
                           return entry;
                         });
          mean_value_constraints.add_line(row);
          mean_value_constraints.add_entries(row, entries);
          mean_value_constraints.set_inhomogeneity(row, line.inhomogeneity);
        }
      }
      // else
      // 	{
      // 	  mean_value_constraints.reinit(constraints_pressure.get_local_lines());
      // 	  mean_value_constraints.shift(offset_pressure_dofs);
      // 	}

      // for(const auto line : constraints_pressure.get_lines())
      // 	{
      // 	  std::cout << "[" << mpi_rank << "]line " << line.index << std::endl;
      // 	  for (const auto & [column, value] : line.entries)
      // 	    std::cout << "[" << mpi_rank << "]entries " << column << " " << value << std::endl;
      // 	  std::cout << "[" << mpi_rank << "]inhom " << line.inhomogeneity << std::endl;
      // 	}
      // AssertThrow(false, ExcMessage(""));

      // /// shift dofs of the pressure block by n_dofs_velocity
      // mean_value_constraints.add_line(shift_pressure_dofs);
      // for(const auto & [column, value] : mean_value_free_entries)
      //   mean_value_constraints.add_entry(shift_pressure_dofs, shift_pressure_dofs + column,
      //   value);
    }
    /// ... or mean value filter
    else
    {
      AssertThrow(is_first_proc, ExcMessage("TODO MPI"));
      constant_pressure_mode.reinit(n_dofs_pressure);
      const bool is_dgq_legendre =
        dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
      if(is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP)
      {
        const auto n_dofs_per_cell = dof_handler_pressure.get_fe().dofs_per_cell;
        AssertDimension(n_dofs_pressure % n_dofs_per_cell, 0);
        for(auto i = 0U; i < n_dofs_pressure; i += n_dofs_per_cell)
          constant_pressure_mode[i] = 1.;
      }
      else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
      {
        constant_pressure_mode = 1.;
      }
      else
        AssertThrow(false, ExcMessage("This pressure dof layout is not supported."));
    }

    Assert(mean_value_constraints.is_consistent_in_parallel(
             Utilities::MPI::all_gather(MPI_COMM_WORLD, locally_owned_dof_indices),
             [&]() {
               IndexSet locally_active_dof_indices;
               DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dof_indices);
               return locally_active_dof_indices;
             }(),
             MPI_COMM_WORLD),
           ExcMessage("mean_value_constraints are not consistent in parallel."));
    mean_value_constraints.close();
  }

  zero_constraints.merge(mean_value_constraints);

  system_matrix.clear();

  cell_integrals_mask.reinit(dim + 1, dim + 1);
  face_integrals_mask.reinit(dim + 1, dim + 1);
  for(auto i = 0U; i < dim + 1; ++i)
    for(auto j = 0U; j < dim + 1; ++j)
    {
      cell_integrals_mask(i, j) = DoFTools::Coupling::always;
      if(dof_layout_v == TPSS::DoFLayout::DGQ || dof_layout_v == TPSS::DoFLayout::RT)
        face_integrals_mask(i, j) = DoFTools::Coupling::always;
      else if(dof_layout_v == TPSS::DoFLayout::Q)
        face_integrals_mask(i, j) = DoFTools::Coupling::none;
      else
        AssertThrow(false, ExcMessage("This dof layout is not supported."));
    }

  TrilinosWrappers::SparsityPattern dsp(locally_owned_dof_indices,
                                        locally_owned_dof_indices,
                                        locally_relevant_dof_indices,
                                        MPI_COMM_WORLD);
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       zero_constraints,
                                       false /*???*/,
                                       cell_integrals_mask,
                                       face_integrals_mask,
                                       mpi_rank);
  dsp.compress();

  system_matrix.initialize(dsp,
                           locally_owned_dof_indices,
                           locally_relevant_dof_indices,
                           MPI_COMM_WORLD);

  system_matrix.initialize_dof_vector(system_solution);
  zero_constraints.set_zero(system_solution);       // zero out
  constraints_velocity.distribute(system_solution); // part. velocity solution!

  system_matrix.initialize_dof_vector(system_delta_x);
  zero_constraints.set_zero(system_delta_x);
  zero_constraints_velocity.distribute(system_delta_x); // hom. velocity solution

  system_matrix.initialize_dof_vector(system_rhs);

  print_parameter("Number of degrees of freedom (velocity):", n_dofs_velocity);
  print_parameter("Number of degrees of freedom (pressure):", n_dofs_pressure);
  print_parameter("Number of degrees of freedom (total):", n_dofs_velocity + n_dofs_pressure);
  *pcout << std::endl;

  pp_data.n_dofs_global.push_back(n_dofs_velocity);
  pp_data_pressure.n_dofs_global.push_back(n_dofs_pressure);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system_velocity_pressure()
{
  constexpr bool use_sipg_method     = dof_layout_v == TPSS::DoFLayout::DGQ;
  constexpr bool use_hdivsipg_method = dof_layout_v == TPSS::DoFLayout::RT;
  constexpr bool use_conf_method     = dof_layout_v == TPSS::DoFLayout::Q;

  system_rhs.zero_out_ghosts();

  /// Assemble the velocity block, here block(0,0).
  {
    using Velocity::SIPG::MW::CopyData;
    using Velocity::SIPG::MW::ScratchData;
    using MatrixIntegrator = Velocity::SIPG::MW::MatrixIntegrator<dim, false>;

    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> load_function_velocity(load_function.get(), component_range);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);

    system_solution.update_ghost_values();
    const auto * particular_solution_velocity =
      (use_conf_method || use_hdivsipg_method) ? &(system_solution) : nullptr;

    MatrixIntegrator matrix_integrator(&load_function_velocity,
                                       &analytical_solution_velocity,
                                       particular_solution_velocity,
                                       equation_data);

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & f,
                           const unsigned int & sf,
                           const auto &         ncell,
                           const unsigned int & nf,
                           const unsigned int & nsf,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      if(use_sipg_method)
        matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
      else if(use_hdivsipg_method)
        // matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
        matrix_integrator.face_worker_tangential(
          cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This velocity dof layout is not supported."));
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      if(use_sipg_method)
        matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
      else if(use_hdivsipg_method)
        // matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
        matrix_integrator.boundary_worker_tangential(cell, face_no, scratch_data, copy_data);
      // matrix_integrator.boundary_worker_tangential_old(cell, face_no, scratch_data,  copy_data);
      else
        AssertThrow(false, ExcMessage("This velocity dof layout is not supported."));
    };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints_velocity.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        copy_data.cell_matrix,
        copy_data.cell_rhs_test,
        copy_data.local_dof_indices_test,
        system_matrix,
        system_rhs);

      for(auto & cdf : copy_data.face_data)
      {
        if(!use_hdivsipg_method)
          AssertDimension(cdf.cell_rhs_test.size(), 0);

        if(cdf.cell_rhs_test.size() == 0)
          zero_constraints_velocity
            .template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
              cdf.cell_matrix, cdf.joint_dof_indices_test, system_matrix);
        else
        {
          zero_constraints_velocity
            .template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
              cdf.cell_matrix,
              cdf.cell_rhs_test,
              cdf.joint_dof_indices_test,
              system_matrix,
              system_rhs);
        }
      }
    };

    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                               update_quadrature_points | update_JxW_values |
                                               update_normal_vectors;

    ScratchData<dim> scratch_data(mapping,
                                  dof_handler_velocity.get_fe(),
                                  dof_handler_velocity.get_fe(),
                                  n_q_points_1d,
                                  update_flags,
                                  update_flags,
                                  interface_update_flags,
                                  interface_update_flags);

    CopyData copy_data(dof_handler_velocity.get_fe().dofs_per_cell);

    if(use_conf_method)
      MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                            dof_handler_velocity.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method || use_hdivsipg_method)
      MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                            dof_handler_velocity.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker,
                            face_worker);
    else
      AssertThrow(false, ExcMessage("This FEM is not implemented."));
  }

  /// Assemble the pressure block, here block(1,1).
  {
    using Pressure::MW::CopyData;
    using Pressure::MW::ScratchData;
    using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, false>;

    const auto             component_range = std::make_pair<unsigned int>(dim, dim + 1);
    FunctionExtractor<dim> load_function_pressure(load_function.get(), component_range);

    MatrixIntegrator matrix_integrator(&load_function_pressure, nullptr, nullptr, equation_data);

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        copy_data.cell_matrix,
        copy_data.cell_rhs,
        shift_pressure_dof_indices(copy_data.local_dof_indices),
        system_matrix,
        system_rhs);
    };

    const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_default;

    ScratchData<dim> scratch_data(
      mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);

    CopyData copy_data(dof_handler_pressure.get_fe().dofs_per_cell);

    MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
                          dof_handler_pressure.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

  /// Assemble the mixed velocity-pressure block, that is test functions are
  /// from velocity space and ansatz functions from pressure space. The
  /// "flipped" pressure-velocity block is assembled as well.
  {
    using VelocityPressure::MW::Mixed::CopyData;
    using VelocityPressure::MW::Mixed::ScratchData;
    using MatrixIntegrator = VelocityPressure::MW::Mixed::MatrixIntegrator<dim, false>;
    using CellIterator     = typename MatrixIntegrator::IteratorType;

    const auto * particular_solution_velocity =
      (use_conf_method || use_hdivsipg_method) ? &(system_solution) : nullptr;

    const auto             component_range_velocity = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(),
                                                        component_range_velocity);

    MatrixIntegrator matrix_integrator(particular_solution_velocity,
                                       /*particular_solution_pressure*/ nullptr,
                                       &analytical_solution_velocity,
                                       /*&analytical_solution_pressure*/ nullptr,
                                       equation_data);

    auto cell_worker =
      [&](const CellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                                 cell->level(),
                                 cell->index(),
                                 &dof_handler_pressure);
        matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & f,
                           const unsigned int & sf,
                           const auto &         ncell,
                           const unsigned int & nf,
                           const unsigned int & nsf,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                               cell->level(),
                               cell->index(),
                               &dof_handler_pressure);
      CellIterator ncell_ansatz(&dof_handler_pressure.get_triangulation(),
                                ncell->level(),
                                ncell->index(),
                                &dof_handler_pressure);
      if(use_sipg_method)
        matrix_integrator.face_worker(
          cell, cell_ansatz, f, sf, ncell, ncell_ansatz, nf, nsf, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This FEM is not supported."));
    };

    auto boundary_worker = [&](const CellIterator & cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      CellIterator cell_ansatz(&dof_handler_pressure.get_triangulation(),
                               cell->level(),
                               cell->index(),
                               &dof_handler_pressure);
      if(use_sipg_method)
        matrix_integrator.boundary_worker(cell, cell_ansatz, face_no, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This FEM is not supported."));
    };

    const auto copier = [&](const CopyData & copy_data) {
      zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        copy_data.cell_matrix,
        copy_data.local_dof_indices_test,
        zero_constraints,
        shift_pressure_dof_indices(copy_data.local_dof_indices_ansatz),
        system_matrix);
      zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        copy_data.cell_matrix_flipped,
        shift_pressure_dof_indices(copy_data.local_dof_indices_ansatz),
        zero_constraints,
        copy_data.local_dof_indices_test,
        system_matrix);

      zero_constraints.distribute_local_to_global(copy_data.cell_rhs_test,
                                                  copy_data.local_dof_indices_test,
                                                  system_rhs);
      zero_constraints.distribute_local_to_global(copy_data.cell_rhs_ansatz,
                                                  shift_pressure_dof_indices(
                                                    copy_data.local_dof_indices_ansatz),
                                                  system_rhs);

      for(auto & cdf : copy_data.face_data)
      {
        zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cdf.cell_matrix,
          cdf.joint_dof_indices_test,
          zero_constraints,
          shift_pressure_dof_indices(cdf.joint_dof_indices_ansatz),
          system_matrix);
        zero_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cdf.cell_matrix_flipped,
          shift_pressure_dof_indices(cdf.joint_dof_indices_ansatz),
          zero_constraints,
          cdf.joint_dof_indices_test,
          system_matrix);

        /// For Hdiv-IP there might be liftings from the velocity written to the
        /// pressure RHS.
        AssertDimension(cdf.cell_rhs_test.size(), 0);
        if(dof_layout_v != TPSS::DoFLayout::RT)
          AssertDimension(cdf.cell_rhs_ansatz.size(), 0);

        if(cdf.cell_rhs_test.size() != 0)
          zero_constraints.distribute_local_to_global(cdf.cell_rhs_test,
                                                      cdf.joint_dof_indices_test,
                                                      system_rhs);
        if(cdf.cell_rhs_ansatz.size() != 0)
          zero_constraints.distribute_local_to_global(cdf.cell_rhs_ansatz,
                                                      shift_pressure_dof_indices(
                                                        cdf.joint_dof_indices_ansatz),
                                                      system_rhs);
      }
    };

    const UpdateFlags update_flags_velocity =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags update_flags_pressure =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags_velocity =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
    const UpdateFlags interface_update_flags_pressure =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

    ScratchData<dim> scratch_data(mapping,
                                  dof_handler_velocity.get_fe(),
                                  dof_handler_pressure.get_fe(),
                                  n_q_points_1d,
                                  update_flags_velocity,
                                  update_flags_pressure,
                                  interface_update_flags_velocity,
                                  interface_update_flags_pressure);

    CopyData copy_data(dof_handler_velocity.get_fe().dofs_per_cell,
                       dof_handler_pressure.get_fe().dofs_per_cell);

    if(use_conf_method || use_hdivsipg_method)
      MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                            dof_handler_velocity.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method)
      MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                            dof_handler_velocity.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker,
                            face_worker);
    else
      AssertThrow(false, ExcMessage("This FEM is not supported."));
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::assemble_system()
{
  assemble_system_velocity_pressure();

  if(rt_parameters.solver.variant == "FGMRES_ILU" ||
     rt_parameters.solver.variant == "FGMRES_GMGvelocity")
  {
    AssertThrow(false, ExcMessage("TODO MPI..."));
    // pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));

    // using Pressure::MW::CopyData;
    // using Pressure::MW::ScratchData;
    // using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, false>;

    // MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

    // auto cell_worker =
    //   [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    //     matrix_integrator.cell_mass_worker(cell, scratch_data, copy_data);
    //   };

    // const auto copier = [&](const CopyData & copy_data) {
    //   constraints_pressure.template distribute_local_to_global<SparseMatrix<double>>(
    //     copy_data.cell_matrix, copy_data.local_dof_indices, pressure_mass_matrix);
    // };

    // const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
    // const UpdateFlags interface_update_flags = update_default;

    // ScratchData<dim> scratch_data(
    //   mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags,
    //   interface_update_flags);
    // CopyData copy_data(dof_handler_pressure.get_fe().dofs_per_cell);
    // MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
    //                       dof_handler_pressure.end(),
    //                       cell_worker,
    //                       copier,
    //                       scratch_data,
    //                       copy_data,
    //                       MeshWorker::assemble_own_cells);
  }
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<SolverControl>
ModelProblem<dim, fe_degree_p, method>::get_solver_control() const
{
  return get_solver_control_impl(rt_parameters.solver);
}


template<int dim, int fe_degree_p, Method method>
template<typename PreconditionerType>
void
ModelProblem<dim, fe_degree_p, method>::iterative_solve_impl(
  const PreconditionerType & preconditioner,
  const std::string          solver_variant)
{
  auto solver_control = get_solver_control();

  SolverSelector<vector_type> iterative_solver;
  iterative_solver.set_control(*solver_control);
  iterative_solver.select(solver_variant);
  if(solver_variant == "gmres")
  {
    AssertThrow(false, ExcMessage("TODO MPI..."));
    // SolverGMRES<BlockVector<double>>::AdditionalData additional_data;
    // additional_data.right_preconditioning = rt_parameters.solver.use_right_preconditioning;
    // // additional_data.use_default_residual = false;
    // additional_data.max_n_tmp_vectors = 100; // rt_parameters.solver.n_iterations_max;
    // iterative_solver.set_data(additional_data);
  }
  iterative_solver.solve(system_matrix, system_delta_x, system_rhs, preconditioner);
  /// distribute() is needed to apply the mean value constraint (Dirichlet
  /// conditions of velocity have already been applied to system_solution)
  if(equation_data.force_mean_value_constraint)
    mean_value_constraints.distribute(system_delta_x);
  system_solution += system_delta_x;

  auto reduction_control = dynamic_cast<ReductionControl *>(solver_control.get());
  if(reduction_control)
  {
    const auto [n_frac, reduction_rate] = compute_fractional_steps(*reduction_control);
    pp_data.average_reduction_system.push_back(reduction_rate);
    pp_data.n_iterations_system.push_back(n_frac);
    print_parameter("Average reduction (solver):", reduction_rate);
    print_parameter("Number of iterations (solver):", n_frac);
  }
  else
  {
    pp_data.average_reduction_system.push_back(solver_control->average_reduction());
    pp_data.n_iterations_system.push_back(solver_control->last_step());
    print_parameter("Average reduction (solver):", solver_control->average_reduction());
    print_parameter("Number of iterations (solver):", solver_control->last_step());
  }


  // ReductionControl solver_control;
  // solver_control.set_max_steps(rt_parameters.solver.n_iterations_max);
  // solver_control.set_reduction(rt_parameters.solver.rel_tolerance);
  // solver_control.set_tolerance(rt_parameters.solver.abs_tolerance);
  // solver_control.log_history(true);
  // solver_control.log_result(true);
  // solver_control.enable_history_data();

  // SolverSelector<BlockVector<double>> iterative_solver;
  // iterative_solver.set_control(solver_control);
  // iterative_solver.select(solver_variant);
  // if(solver_variant == "gmres")
  // {
  //   SolverGMRES<BlockVector<double>>::AdditionalData additional_data;
  //   additional_data.right_preconditioning = rt_parameters.solver.use_right_preconditioning;
  //   // additional_data.use_default_residual = false;
  //   additional_data.max_n_tmp_vectors = 100; // rt_parameters.solver.n_iterations_max;
  //   iterative_solver.set_data(additional_data);
  // }
  // iterative_solver.solve(system_matrix, system_delta_x, system_rhs, preconditioner);
  // /// distribute() is needed to apply the mean value constraint (Dirichlet
  // /// conditions of velocity have already been applied to system_solution)
  // if(equation_data.force_mean_value_constraint)
  //   mean_value_constraints.distribute(system_delta_x);
  // // zero_constraints.distribute(system_delta_x); // !!!
  // system_solution += system_delta_x;

  // const auto [n_frac, reduction_rate] = compute_fractional_steps(solver_control);
  // pp_data.average_reduction_system.push_back(reduction_rate);
  // pp_data.n_iterations_system.push_back(n_frac);
  // print_parameter("Average reduction (solver):", reduction_rate);
  // print_parameter("Number of iterations (solver):", n_frac);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::solve()
{
  // // This is used to pass whether or not we want to solve for A inside
  // // the preconditioner.  One could change this to false to see if
  // // there is still convergence and if so does the program then run
  // // faster or slower
  // const bool use_expensive = true;

  if(rt_parameters.solver.variant == "direct")
  {
    auto solver_control = get_solver_control();

    TrilinosWrappers::SolverDirect::AdditionalData features;
    features.output_solver_details = true;

    TrilinosWrappers::SolverDirect solver(*solver_control, features);
    solver.solve(system_matrix, system_delta_x, system_rhs);

    zero_constraints.distribute(system_delta_x);
    system_solution += system_delta_x;

    pp_data.average_reduction_system.push_back(0.);
    pp_data.n_iterations_system.push_back(0.);

    print_parameter("Average reduction (solver):", "direct (trilinos)");
    print_parameter("Number of iterations (solver):", "direct (trilinos)");



    // SparseDirectUMFPACK A_direct;
    // A_direct.template initialize<BlockSparseMatrix<double>>(system_matrix);
    // A_direct.vmult(system_delta_x, system_rhs);
    // /// distribute() is needed to apply the mean value constraint (Dirichlet
    // /// conditions of velocity have already been applied to system_solution)
    // Assert(equation_data.force_mean_value_constraint, ExcMessage("Use mean value constraint."));
    // zero_constraints.distribute(system_delta_x); // !!! no-normal flux + mean value
    // mean_value_constraints.distribute(system_delta_x);
    // system_solution += system_delta_x;

    // pp_data.average_reduction_system.push_back(0.);
    // pp_data.n_iterations_system.push_back(0.);
    // print_parameter("Average reduction (solver):", "direct solver");
    // print_parameter("Number of iterations (solver):", "---");
    // return;
  }

  //   else if(rt_parameters.solver.variant == "FGMRES_ILU")
  //   {
  //     SparseILU<double> A_preconditioner;
  //     A_preconditioner.initialize(system_matrix.block(0, 0));

  //     SparseILU<double> S_preconditioner;
  //     S_preconditioner.initialize(pressure_mass_matrix);

  //     const BlockSchurPreconditioner<typename std::decay<decltype(A_preconditioner)>::type,
  //                                    typename std::decay<decltype(S_preconditioner)>::type>
  //       preconditioner(
  //         system_matrix, pressure_mass_matrix, A_preconditioner, S_preconditioner,
  //         use_expensive);

  //     iterative_solve_impl(preconditioner, "fgmres");
  //     *pcout << preconditioner.get_summary() << std::endl;
  //   }

  //   else if(rt_parameters.solver.variant == "FGMRES_GMGvelocity")
  //   {
  //     prepare_multigrid_velocity();
  //     auto & A_preconditioner = mgc_velocity.get_preconditioner();

  //     SparseILU<double> S_preconditioner;
  //     S_preconditioner.initialize(pressure_mass_matrix, SparseILU<double>::AdditionalData());

  //     const BlockSchurPreconditioner<typename std::decay<decltype(A_preconditioner)>::type,
  //                                    SparseILU<double>>
  //       preconditioner(
  //         system_matrix, pressure_mass_matrix, A_preconditioner, S_preconditioner,
  //         use_expensive);

  //     iterative_solve_impl(preconditioner, "fgmres");
  //     *pcout << preconditioner.get_summary();
  //   }

  //   else if(rt_parameters.solver.variant == "GMRES_GMG")
  //   {
  //     prepare_multigrid_velocity_pressure();
  //     auto & preconditioner = mgc_velocity_pressure.get_preconditioner();

  //     iterative_solve_impl(preconditioner, "gmres");
  //   }

  //   else if(rt_parameters.solver.variant == "CG_GMG")
  //   {
  //     prepare_multigrid_velocity_pressure();
  //     auto & preconditioner = mgc_velocity_pressure.get_preconditioner();

  //     iterative_solve_impl(preconditioner, "cg");
  //   }

  else if(rt_parameters.solver.variant == "CG")
  {
    PreconditionIdentity preconditioner;
    rt_parameters.solver.n_iterations_max *= 100.;

    iterative_solve_impl(preconditioner, "cg");
  }

  else
    AssertThrow(false, ExcMessage("Please, choose a valid solver variant."));

  /// Post processing of discrete solution
  const double mean_pressure =
    VectorTools::compute_mean_value(dof_handler, QGauss<dim>(n_q_points_1d), system_solution, dim);
  const bool is_dgq_legendre =
    dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
  const bool is_legendre_type = is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP;
  const auto n_dofs_pressure  = dof_handler_pressure.n_dofs();
  const auto n_dofs_velocity  = dof_handler_velocity.n_dofs();
  if(is_legendre_type)
  {
    AssertThrow(is_first_proc, ExcMessage("TODO MPI..."));
    const auto n_dofs_per_cell = get_fe_pressure().dofs_per_cell;
    AssertDimension(n_dofs_pressure % n_dofs_per_cell, 0);
    // auto & dof_values_pressure = system_solution.block(1);
    for(auto i = 0U; i < n_dofs_pressure; i += n_dofs_per_cell)
      system_solution[i + n_dofs_velocity] -= mean_pressure;
    // dof_values_pressure[i] -= mean_pressure;
  }
  else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
    for(auto i = 0U; i < n_dofs_pressure; ++i)
      system_solution[i + n_dofs_velocity] -= mean_pressure;
  // system_solution.block(1).add(-mean_pressure);
  else
    AssertThrow(false, ExcMessage("This dof layout is not supported."));

  print_parameter("Mean of pressure corrected by:", -mean_pressure);
  *pcout << std::endl;
}



// template<int dim, int fe_degree_p, Method method>
// void
// ModelProblem<dim, fe_degree_p, method>::correct_mean_value_pressure()
// {
//   const double mean_pressure =
//     VectorTools::compute_mean_value(dof_handler, QGauss<dim>(n_q_points_1d), system_solution,
//     dim);
//   const bool is_dgq_legendre =
//     dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;
//   const bool is_legendre_type = is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP;
//   if(is_legendre_type)
//   {
//     const auto n_dofs_per_cell = get_fe_pressure().dofs_per_cell;
//     const auto n_dofs_pressure = system_solution.block(1).size();
//     AssertDimension(n_dofs_pressure % n_dofs_per_cell, 0);
//     Vector<double> & dof_values_pressure = system_solution.block(1);
//     for(auto i = 0U; i < n_dofs_pressure; i += n_dofs_per_cell)
//       dof_values_pressure[i] -= mean_pressure;
//   }
//   else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
//     system_solution.block(1).add(-mean_pressure);
//   else
//     AssertThrow(false, ExcMessage("This dof layout is not supported."));

//   print_parameter("Mean of pressure corrected by:", -mean_pressure);
//   *pcout << std::endl;
// }



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_L2_error_velocity() const
{
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
  const auto difference_per_cell = std::make_shared<Vector<double>>(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    *difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  return difference_per_cell;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_L2_error_pressure() const
{
  const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
  const auto difference_per_cell = std::make_shared<Vector<double>>(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    *difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::L2_norm,
                                    &pressure_mask);
  return difference_per_cell;
}



template<int dim, int fe_degree_p, Method method>
std::shared_ptr<Vector<double>>
ModelProblem<dim, fe_degree_p, method>::compute_H1semi_error_velocity() const
{
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
  const auto difference_per_cell = std::make_shared<Vector<double>>(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    system_solution,
                                    *analytical_solution,
                                    *difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::H1_norm,
                                    &velocity_mask);
  return difference_per_cell;
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::compute_errors()
{
  {
    const auto   difference_per_cell = compute_L2_error_velocity();
    const double Velocity_L2_error =
      VectorTools::compute_global_error(triangulation, *difference_per_cell, VectorTools::L2_norm);
    print_parameter("Velocity error in the L2 norm:", Velocity_L2_error);
    pp_data.L2_error.push_back(Velocity_L2_error);
  }

  {
    const auto   difference_per_cell = compute_L2_error_pressure();
    const double Pressure_L2_error =
      VectorTools::compute_global_error(triangulation, *difference_per_cell, VectorTools::L2_norm);
    print_parameter("Pressure error in the L2 norm:", Pressure_L2_error);
    pp_data_pressure.L2_error.push_back(Pressure_L2_error);
  }

  {
    const auto   difference_per_cell = compute_H1semi_error_velocity();
    const double Velocity_H1_error =
      VectorTools::compute_global_error(triangulation, *difference_per_cell, VectorTools::H1_norm);
    print_parameter("Velocity error in the H1 seminorm:", Velocity_H1_error);
    pp_data.H1semi_error.push_back(Velocity_H1_error);
  }
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::output_results(const unsigned int refinement_cycle) const
{
  AssertThrow(is_first_proc, ExcMessage("TODO MPI..."));

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  /// discrete solution (velocity, pressure): (u, p)
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");
  data_out.add_data_vector(system_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  /// discrete right hand side (velocity, pressure): (Fu, Fp)
  std::vector<std::string> rhs_names(dim, "rhs_velocity");
  rhs_names.emplace_back("rhs_pressure");
  data_out.add_data_vector(system_rhs,
                           rhs_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  const auto L2_error_v = compute_L2_error_velocity();
  data_out.add_data_vector(*L2_error_v, "velocity_L2_error", DataOut<dim>::type_cell_data);

  const auto L2_error_p = compute_L2_error_pressure();
  data_out.add_data_vector(*L2_error_p, "pressure_L2_error", DataOut<dim>::type_cell_data);

  const auto H1semi_error_v = compute_H1semi_error_velocity();
  data_out.add_data_vector(*H1semi_error_v, "velocity_H1semi_error", DataOut<dim>::type_cell_data);

  data_out.build_patches();

  std::ofstream output("stokes_" + equation_data.sstr_equation_variant() + "_" +
                       Utilities::int_to_string(refinement_cycle, 3) + ".vtk");
  data_out.write_vtk(output);
}



template<int dim, int fe_degree_p, Method method>
void
ModelProblem<dim, fe_degree_p, method>::run()
{
  print_informations();

  const unsigned int n_cycles = rt_parameters.n_cycles;
  for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    {
      std::ostringstream oss;
      oss << "Starting cycle " << cycle + 1 << " of " << n_cycles;
      print_parameter(oss.str(), "...");
    }

    const unsigned int n_refinements = rt_parameters.mesh.n_refinements + cycle;
    if(!make_grid(n_refinements))
    {
      print_parameter("No mesh created", "...");
      *pcout << std::endl << std::endl;
      continue;
    }

    setup_system();

    assemble_system();

    solve();

    compute_errors();

    output_results(cycle);

    Utilities::System::MemoryStats mem;
    Utilities::System::get_memory_stats(mem);
    print_parameter("Memory used (VM Peak):", mem.VmPeak);

    *pcout << std::endl;
  }
}

} // namespace NoBlocks

} // namespace Stokes

#endif // inclusion guard
