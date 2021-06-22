#ifndef APPS_STOKESPROBLEM_H_
#define APPS_STOKESPROBLEM_H_

/// originated from step-56
//
//  Author: Ryan Grove, Clemson University
//          Timo Heister, Clemson University
//

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


#include "solvers_and_preconditioners/TPSS/matrix_utilities.h"
#include "solvers_and_preconditioners/TPSS/move_to_deal_ii.h"


#include "coloring.h"
#include "equation_data.h"
#include "multigrid.h"
#include "postprocess.h"
#include "preconditioner.h"
#include "rt_parameter.h"
#include "sparsity.h"
#include "stokes_integrator.h"



namespace Stokes
{
using namespace dealii;

/**
 * A wrapper class around a (sparse) system/level matrix of type
 * TrilinosWrappers::BlockSparseMatrix and a matrix integrator for local
 * solvers.
 *
 * Convenience functions are added to deal with vectors and MPI-relevant dof
 * partitioning.
 */
template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
class BlockSparseMatrixAugmented : public TrilinosWrappers::BlockSparseMatrix,
                                   public VelocityPressure::MatrixIntegrator<dim,
                                                                             fe_degree_p,
                                                                             Number,
                                                                             dof_layout_v,
                                                                             fe_degree_v,
                                                                             local_assembly,
                                                                             is_simplified>
{
  static_assert(std::is_same<Number, double>::value,
                "TrilinosWrappers support only double-precision.");

public:
  using value_type            = Number;
  using matrix_type           = TrilinosWrappers::BlockSparseMatrix;
  using local_integrator_type = VelocityPressure::MatrixIntegrator<dim,
                                                                   fe_degree_p,
                                                                   Number,
                                                                   dof_layout_v,
                                                                   fe_degree_v,
                                                                   local_assembly,
                                                                   is_simplified>;

  void
  initialize(const TrilinosWrappers::BlockSparsityPattern & dsp,
             const std::vector<IndexSet> &                  locally_owned_dof_indices,
             const std::vector<IndexSet> &                  ghosted_dof_indices,
             const MPI_Comm &                               mpi_communicator);

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const EquationData                             equation_data_in);

  void
  clear();

  void
  initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> & vec) const;

  void
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> & vec,
                        const unsigned int                           block_index) const;

  std::shared_ptr<const Utilities::MPI::Partitioner>
  get_partitioner(const unsigned int block_index) const;

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const;

  template<typename VectorType>
  void
  vmult(VectorType & dst, const VectorType & src) const;

  void
  vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const;

  operator const FullMatrix<Number> &() const;

  std::shared_ptr<const MatrixFree<dim, Number>>                  mf_storage;
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
  mutable std::shared_ptr<FullMatrix<Number>>                     fullmatrix;
};



/**
 * A wrapper class around a sparse matrix of type BlockSparseMatrix which
 * subtracts a given mode after each matrix-vector multiplication with the
 * sparse matrix (@p vmult()).
 */
class BlockSparseMatrixFiltered : public Subscriptor
{
public:
  using matrix_type = TrilinosWrappers::BlockSparseMatrix;
  using value_type  = typename matrix_type::value_type;

  BlockSparseMatrixFiltered(const matrix_type &                                    matrix_in,
                            const LinearAlgebra::distributed::Vector<value_type> & mode_in);
  void
  vmult(LinearAlgebra::distributed::BlockVector<value_type> &       dst,
        const LinearAlgebra::distributed::BlockVector<value_type> & src) const;

  void
  Tvmult(LinearAlgebra::distributed::BlockVector<value_type> &       dst,
         const LinearAlgebra::distributed::BlockVector<value_type> & src) const;

  matrix_type::size_type
  m() const
  {
    return matrix.m();
  }

  matrix_type::size_type
  n() const
  {
    return matrix.n();
  }

  operator const FullMatrix<double> &() const
  {
    AssertThrow(false, ExcMessage("Not implemented..."));
    return fullmatrix_dummy;
  }

  const matrix_type &                                    matrix;
  const LinearAlgebra::distributed::Vector<value_type> * mode;
  FullMatrix<double>                                     fullmatrix_dummy;
};



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
class SparseMatrixAugmented
  : public TrilinosWrappers::SparseMatrix,
    public Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree, double, dof_layout, is_simplified>
{
public:
  using matrix_type = TrilinosWrappers::SparseMatrix;
  using value_type  = double;
  using local_integrator_type =
    Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree, value_type, dof_layout, is_simplified>;
  using vector_type = LinearAlgebra::distributed::Vector<value_type>;

  /**
   * Initializes the sparsity pattern of the underlying sparse matrix and caches
   * the mpi communication pattern for dofs.
   */
  void
  initialize(const TrilinosWrappers::SparsityPattern & dsp,
             const IndexSet &                          locally_owned_dof_indices,
             const IndexSet &                          ghosted_dof_indices,
             const MPI_Comm &                          mpi_communicator);

  /**
   * Initializes the underyling local integrator.
   */
  void
  initialize(std::shared_ptr<const MatrixFree<dim, value_type>> mf_storage_in,
             const EquationData                                 equation_data_in);

  /**
   * Initializes vector @p vec by means of the cached mpi communication pattern.
   */
  void
  initialize_dof_vector(vector_type & vec) const;

  /**
   * Initializes vector @p vec by means of the underlying MatrixFree object.
   */
  void
  initialize_dof_vector_mf(vector_type & vec) const;

  void
  clear();

  using matrix_type::vmult;

  /**
   * In serial code this performs a matrix-vector multplication based on
   * ArrayViews which enables using TPSS-specific interfaces.
   */
  void
  vmult(const ArrayView<value_type> dst, const ArrayView<const value_type> src) const;

  std::shared_ptr<const MatrixFree<dim, value_type>>
  get_matrix_free() const;

  /**
   * In serial code this class is convertible into FullMatrix type.
   */
  operator const FullMatrix<value_type> &() const;

private:
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
  std::shared_ptr<const MatrixFree<dim, value_type>> mf_storage;
  mutable std::shared_ptr<FullMatrix<value_type>>    fullmatrix;
};



/**
 * A block Schur preconditioner (right preconditioner as demanded by FGMRES) for
 * the Stokes block system. For more details see step-56...
 */
template<class PreconditionerAType, class PreconditionerSType>
class BlockSchurPreconditionerStep56 : public Subscriptor
{
public:
  using preconditioner_A_type = PreconditionerAType;
  using preconditioner_S_type = PreconditionerSType;
  using solver_A_type         = IterativeInverse<PreconditionerAType>;
  using solver_S_type         = IterativeInverse<PreconditionerSType>;

  BlockSchurPreconditionerStep56(const TrilinosWrappers::BlockSparseMatrix & system_matrix_in,
                                 const TrilinosWrappers::SparseMatrix &      S_in,
                                 const PreconditionerAType &                 preconditioner_A_in,
                                 const PreconditionerSType &                 preconditioner_S_in,
                                 const bool                                  do_solve_A_in);

  void
  vmult(LinearAlgebra::distributed::BlockVector<double> &       dst,
        const LinearAlgebra::distributed::BlockVector<double> & src) const;

  std::string
  get_summary() const;

private:
  const preconditioner_A_type &                                                 preconditioner_A;
  std::shared_ptr<solver_A_type>                                                solver_A;
  std::shared_ptr<solver_S_type>                                                solver_S;
  std::shared_ptr<PreconditionBlockSchur<preconditioner_A_type, solver_S_type>> block_schur_cheap;
  std::shared_ptr<PreconditionBlockSchur<solver_A_type, solver_S_type>> block_schur_expensive;

  const bool           do_solve_A;
  mutable unsigned int n_iterations_A;
  mutable unsigned int n_iterations_S;
};



/**
 * A helper class which provides the infrastructure for a geometric multigrid
 * method with respect to the velocity component. In this way ModelProblem is
 * not overloaded with objects for each multigrid variant.
 *
 * Any MGCollectionVelocity object is constructed with a set of multigrid and
 * PDE-specific parameters. The actual setup and assembly of the multigrid
 * hierarchy is performed when calling @p initialize(). Then, a multigrid
 * preconditioner can be queried.
 */
template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
class MGCollectionVelocity
{
public:
  static constexpr int n_q_points_1d = n_q_points_1d_impl(fe_degree, dof_layout);

  using vector_type       = LinearAlgebra::distributed::Vector<double>;
  using matrix_type       = SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>;
  using mg_transfer_type  = MGTransferMatrixFree<dim, double>;
  using local_matrix_type = typename matrix_type::local_integrator_type::matrix_type;
  using mg_smoother_schwarz_type =
    MGSmootherSchwarz<dim, matrix_type, local_matrix_type, vector_type>;
  using mg_smoother_gauss_seidel_type = MGSmootherPrecondition<TrilinosWrappers::SparseMatrix,
                                                               TrilinosWrappers::PreconditionSSOR,
                                                               vector_type>;

  /**
   * The constructor passes algorithm- and PDE-related parameters. The actual
   * initialization of the multigrid method is performed when calling @p
   * initialize().
   */
  MGCollectionVelocity(const RT::Parameter & rt_parameters_in,
                       const EquationData &  equation_data_in);

  /**
   * Initializes the multigrid hierarchy, including the setup/assembly of
   * matrices and smoothers for each level.
   */
  void
  initialize(const unsigned int                       mg_level_max,
             const std::shared_ptr<ColoringBase<dim>> user_coloring);

  /**
   * Query the multigrid preconditioner after initialization.
   */
  const PreconditionMG<dim, vector_type, mg_transfer_type> &
  get_preconditioner() const;

  unsigned int
  n_colors() const;

  /// Expose all members which have to be set by the user before initialization...
  const DoFHandler<dim> * dof_handler;
  const Mapping<dim> *    mapping;

  Table<2, DoFTools::Coupling> cell_integrals_mask;
  Table<2, DoFTools::Coupling> face_integrals_mask;

  /// should be private but googletests require public access
  // private:
  void
  clear_data();

  void
  initialize_schwarz_smoothers(const std::shared_ptr<ColoringBase<dim>> user_coloring);

  void
  assemble_multigrid(const unsigned int level, AffineConstraints<double> & level_constraints);

  MGParameter  parameters;
  EquationData equation_data;

  std::shared_ptr<MGConstrainedDoFs>                     mg_constrained_dofs;
  mg_transfer_type                                       mg_transfer;
  MGLevelObject<matrix_type>                             mg_matrices;
  std::shared_ptr<const mg_smoother_schwarz_type>        mg_schwarz_smoother_pre;
  std::shared_ptr<const mg_smoother_schwarz_type>        mg_schwarz_smoother_post;
  std::shared_ptr<const MGSmootherIdentity<vector_type>> mg_smoother_identity;
  std::shared_ptr<const mg_smoother_gauss_seidel_type>   mg_smoother_gauss_seidel;
  const MGSmootherBase<vector_type> *                    mg_smoother_pre;
  const MGSmootherBase<vector_type> *                    mg_smoother_post;
  CoarseGridSolver<matrix_type, vector_type>             coarse_grid_solver;
  const MGCoarseGridBase<vector_type> *                  mg_coarse_grid;
  mg::Matrix<vector_type>                                mg_matrix_wrapper;
  std::shared_ptr<Multigrid<vector_type>>                multigrid;

  mutable std::shared_ptr<const PreconditionMG<dim, vector_type, mg_transfer_type>>
    preconditioner_mg;

  const bool         use_tbb;
  const unsigned int this_mpi_rank;
};



/**
 * A helper class which provides infrastructure for a geometric multigrid
 * method with respect to the velocity-pressure system. In this way ModelProblem is
 * not overloaded with objects for several multigrid variants.
 *
 * Any object of this class is constructed with a set of multigrid and
 * PDE-specific parameters. The actual setup and assembly of the multigrid
 * hierarchy is performed when calling @p initialize(). Then, a multigrid
 * preconditioner can be queried.
 */
template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v   = TPSS::DoFLayout::Q,
         int             fe_degree_v    = fe_degree_p + 1,
         LocalAssembly   local_assembly = LocalAssembly::Tensor,
         bool            is_simplified  = false>
struct MGCollectionVelocityPressure
{
  using vector_type = LinearAlgebra::distributed::BlockVector<double>;
  using matrix_type = BlockSparseMatrixAugmented<dim,
                                                 fe_degree_p,
                                                 double,
                                                 dof_layout_v,
                                                 fe_degree_v,
                                                 local_assembly,
                                                 is_simplified>;
  // using mg_transfer_type  = MGTransferBlockMatrixFree<dim, double>;
  using mg_transfer_type  = MGTransferBlockPrebuilt;
  using local_matrix_type = typename matrix_type::local_integrator_type::matrix_type;
  using mg_smoother_schwarz_type =
    MGSmootherSchwarz<dim, matrix_type, local_matrix_type, vector_type>;

  static constexpr int n_q_points_1d = n_q_points_1d_impl(fe_degree_v, dof_layout_v);

  static constexpr bool use_sipg_method     = dof_layout_v == TPSS::DoFLayout::DGQ;
  static constexpr bool use_hdivsipg_method = dof_layout_v == TPSS::DoFLayout::RT;
  static constexpr bool use_conf_method     = dof_layout_v == TPSS::DoFLayout::Q;

  MGCollectionVelocityPressure(const RT::Parameter & rt_parameters_in,
                               const EquationData &  equation_data_in);

  void
  initialize(const unsigned int                       mg_level_max,
             const std::shared_ptr<ColoringBase<dim>> user_coloring);

  const PreconditionMG<dim, vector_type, mg_transfer_type> &
  get_preconditioner() const;

  unsigned int
  n_colors() const;

  /// Expose all members which have to be set by the user before initialization...
  const DoFHandler<dim> *      dof_handler_velocity;
  const DoFHandler<dim> *      dof_handler_pressure;
  const DoFHandler<dim> *      dof_handler_stream;
  const Mapping<dim> *         mapping;
  Table<2, DoFTools::Coupling> cell_integrals_mask;
  Table<2, DoFTools::Coupling> face_integrals_mask;

  /// should be private but googletests require public access
  // private:
  void
  clear_data();

  void
  initialize_schwarz_smoothers(const std::shared_ptr<ColoringBase<dim>> user_coloring);

  void
  assemble_multigrid(const unsigned int                level,
                     const AffineConstraints<double> & level_constraints_velocity,
                     const AffineConstraints<double> & level_constraints_pressure,
                     const AffineConstraints<double> & level_constraints_stream);

  MGParameter  parameters;
  EquationData equation_data;

  std::shared_ptr<MGConstrainedDoFs>                     mg_constrained_dofs;
  std::shared_ptr<MGConstrainedDoFs>                     mg_constrained_dofs_velocity;
  std::shared_ptr<MGConstrainedDoFs>                     mg_constrained_dofs_pressure;
  std::vector<MGConstrainedDoFs>                         mgconstraineddofs;
  std::shared_ptr<const mg_transfer_type>                mg_transfer;
  MGLevelObject<matrix_type>                             mg_matrices;
  std::shared_ptr<const mg_smoother_schwarz_type>        mg_schwarz_smoother_pre;
  std::shared_ptr<const mg_smoother_schwarz_type>        mg_schwarz_smoother_post;
  std::shared_ptr<const MGSmootherIdentity<vector_type>> mg_smoother_identity;
  const MGSmootherBase<vector_type> *                    mg_smoother_pre;
  const MGSmootherBase<vector_type> *                    mg_smoother_post;

  using coarse_solver_A_type = SolverDirectInverse;
  using coarse_solver_S_type = IterativeInverse<TrilinosWrappers::PreconditionILU>;
  std::shared_ptr<coarse_solver_A_type> coarse_preconditioner_A;
  TrilinosWrappers::SparseMatrix        coarse_mass_matrix_pressure;
  TrilinosWrappers::PreconditionILU     coarse_mass_precondition_ilu_pressure;
  std::shared_ptr<coarse_solver_S_type> coarse_preconditioner_S;
  std::shared_ptr<PreconditionBlockSchur<coarse_solver_A_type, coarse_solver_S_type>>
                                                   coarse_preconditioner;
  LinearAlgebra::distributed::Vector<double>       coarse_constant_pressure_mode;
  std::shared_ptr<const BlockSparseMatrixFiltered> coarse_matrix_with_filter;

  CoarseGridSolver<matrix_type, vector_type>               coarse_grid_solver;
  CoarseGridSolver<BlockSparseMatrixFiltered, vector_type> coarse_grid_solver_with_filter;
  const MGCoarseGridBase<vector_type> *                    mg_coarse_grid;
  mg::Matrix<vector_type>                                  mg_matrix_wrapper;
  std::shared_ptr<Multigrid<vector_type>>                  multigrid;

  mutable std::shared_ptr<const PreconditionMG<dim, vector_type, mg_transfer_type>>
    preconditioner_mg;

  const bool         use_tbb;
  const unsigned int n_mpi_procs;
  const bool         use_coarse_preconditioner;
};



/**
 * The base class defines compile time parameters depending on the choice of the
 * finite element method (stable Stokes decomposition).
 */
template<Method method, int dim, int fe_degree_p>
struct ModelProblemBase
{
};

/// TODO rename... Q_k+2 only if k == 1, otherwise Q_k+1
template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::Qkplus2_DGPk, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGP;
  using fe_type_v                               = FE_Q<dim>;
  using fe_type_p                               = FE_DGP<dim>;
  static constexpr int           fe_degree_v = fe_degree_p == 0 ? fe_degree_p + 2 : fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::LMW;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::DGQkplus2_DGPk, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::DGQ;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGP;
  using fe_type_v                               = FE_DGQ<dim>;
  using fe_type_p                               = FE_DGP<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p + 1;
  static constexpr LocalAssembly local_assembly = LocalAssembly::LMW;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::RaviartThomas, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::RT;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGQ;
  using fe_type_v                               = FE_RaviartThomas_new<dim>;
  using fe_type_p                               = FE_DGQLegendre<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p;
  static constexpr LocalAssembly local_assembly = LocalAssembly::LMW;
};

template<int dim, int fe_degree_p>
struct ModelProblemBase<Method::RaviartThomasStream, dim, fe_degree_p>
{
  static constexpr TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::RT;
  static constexpr TPSS::DoFLayout dof_layout_p = TPSS::DoFLayout::DGQ;
  using fe_type_v                               = FE_RaviartThomas_new<dim>;
  using fe_type_p                               = FE_DGQLegendre<dim>;
  static constexpr int           fe_degree_v    = fe_degree_p;
  static constexpr LocalAssembly local_assembly = LocalAssembly::StreamTensor;
  /// static SWITCH
  // static constexpr LocalAssembly local_assembly = LocalAssembly::StreamLMW;
};



/**
 * The main class which combines the typical building blocks of a (deal.II)
 * finite element method.
 *
 * TODO Description...
 */
template<int dim, int fe_degree_p, Method method, bool is_simplified = false>
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
  using vector_type = LinearAlgebra::distributed::BlockVector<double>;

  static constexpr int static_fe_degree_p = fe_degree_p;
  static constexpr int fe_degree_v        = Base::fe_degree_v;
  static constexpr int n_q_points_1d      = n_q_points_1d_impl(fe_degree_v, dof_layout_v);

  static constexpr bool use_sipg_method     = dof_layout_v == TPSS::DoFLayout::DGQ;
  static constexpr bool use_hdivsipg_method = dof_layout_v == TPSS::DoFLayout::RT;
  static constexpr bool use_conf_method     = dof_layout_v == TPSS::DoFLayout::Q;

  ModelProblem(const RT::Parameter & rt_parameters_in, const EquationData & equation_data_in);

  void
  run();

  bool
  make_grid();

  bool
  make_grid(const unsigned int n_refinements);

  void
  setup_system();

  void
  assemble_system();

  void
  assemble_system_step56();

  std::shared_ptr<const MGCollectionVelocity<dim, fe_degree_v, dof_layout_v, is_simplified>>
  make_multigrid_velocity();

  std::shared_ptr<const MGCollectionVelocityPressure<dim,
                                                     fe_degree_p,
                                                     dof_layout_v,
                                                     fe_degree_v,
                                                     local_assembly,
                                                     is_simplified>>
  make_multigrid_velocity_pressure();

  void
  solve();

  void
  compute_discretization_errors();

  double
  compute_L2_error_velocity() const;

  double
  compute_L2_error_pressure() const;

  double
  compute_H1semi_error_velocity() const;

  void
  output_results(const unsigned int refinement_cycle) const;

  void
  post_process_solution_vector();

  unsigned int
  max_level() const;

  void
  print_informations() const;

  const FiniteElement<dim> &
  get_finite_element_velocity() const;

  const FiniteElement<dim> &
  get_finite_element_pressure() const;

  const unsigned int this_mpi_rank;
  const bool         is_first_proc;

  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
  EquationData                        equation_data;
  std::shared_ptr<Function<dim>>      analytical_solution;
  std::shared_ptr<Function<dim>>      load_function;

  unsigned int            n_colors_system;
  unsigned int            n_mg_levels;
  mutable PostProcessData pp_data;
  mutable PostProcessData pp_data_pressure;

  std::shared_ptr<parallel::distributed::Triangulation<dim>> triangulation;
  MappingQ<dim>                                              mapping;
  std::shared_ptr<FiniteElement<dim>>                        fe;
  DoFHandler<dim>                                            dof_handler_velocity;
  DoFHandler<dim>                                            dof_handler_pressure;

  std::shared_ptr<FiniteElement<dim>> finite_element_stream;
  DoFHandler<dim>                     dof_handler_stream;

  AffineConstraints<double> zero_constraints_velocity;
  AffineConstraints<double> constraints_velocity;
  AffineConstraints<double> constraints_pressure;

  Table<2, DoFTools::Coupling> cell_integrals_mask;
  Table<2, DoFTools::Coupling> face_integrals_mask;
  BlockSparseMatrixAugmented<dim,
                             fe_degree_p,
                             double,
                             dof_layout_v,
                             fe_degree_v,
                             local_assembly,
                             is_simplified>
                                 system_matrix;
  TrilinosWrappers::SparseMatrix pressure_mass_matrix;

  vector_type                                system_solution;
  vector_type                                system_delta_x;
  vector_type                                system_rhs;
  LinearAlgebra::distributed::Vector<double> constant_mode_pressure;

  //: multigrid
  mutable std::shared_ptr<ColoringBase<dim>> user_coloring;

private:
  template<typename T>
  void
  print_parameter(const std::string & description, const T & value) const;

  void
  make_grid_impl(const MeshParameter & mesh_prms);

  LinearAlgebra::distributed::Vector<double>
  compute_mass_foreach_pressure_dof() const;

  LinearAlgebra::distributed::Vector<double>
  make_constant_pressure_mode(const unsigned int level = numbers::invalid_unsigned_int) const;

  void
  assemble_system_velocity_pressure();

  std::shared_ptr<SolverControl>
  make_solver_control() const;

  template<typename PreconditionerType>
  void
  iterative_solve_impl(const PreconditionerType & preconditioner, const std::string solver_variant);

  /**
   * Depending on the template parameter @p method a default finite element
   * (system of velocity and pressure) is generated.
   */
  std::shared_ptr<FiniteElement<dim>>
  make_finite_element() const;

  std::shared_ptr<FiniteElement<dim>>
  make_finite_element_stream() const;

  bool
  check_finite_element() const;

  const bool do_assemble_pressure_mass_matrix;
};



template<bool is_multigrid, int dim>
void
assemble_pressure_mass_matrix_impl(TrilinosWrappers::SparseMatrix &  matrix,
                                   const DoFHandler<dim> &           dof_handler_pressure,
                                   const Mapping<dim> &              mapping,
                                   const AffineConstraints<double> & constraints_pressure,
                                   const EquationData &              equation_data,
                                   const unsigned int level = numbers::invalid_unsigned_int)
{
  using Pressure::MW::CopyData;
  using Pressure::MW::ScratchData;
  using MatrixIntegrator = Pressure::MW::MatrixIntegrator<dim, is_multigrid>;

  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    matrix_integrator.cell_mass_worker(cell, scratch_data, copy_data);
  };

  const auto copier = [&](const CopyData & copy_data) {
    for(const auto & cd : copy_data.cell_data)
      constraints_pressure.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        cd.matrix, cd.dof_indices, matrix);
  };

  const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
  const UpdateFlags interface_update_flags = update_default;

  const unsigned int n_q_points_1d = dof_handler_pressure.get_fe().tensor_degree() + 1;

  ScratchData<dim> scratch_data(
    mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);

  CopyData copy_data;

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

  matrix.compress(VectorOperation::add);
}



////////// Definitions



template<int dim, int fe_degree_p, Method method, bool is_simplified>
ModelProblem<dim, fe_degree_p, method, is_simplified>::ModelProblem(
  const RT::Parameter & rt_parameters_in,
  const EquationData &  equation_data_in)
  : this_mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
    is_first_proc(this_mpi_rank == 0U),
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
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipExp)
        return std::make_shared<DivergenceFree::NoSlip::Exp::Solution<dim>>();
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    load_function([&]() -> std::shared_ptr<Function<dim>> {
      if(equation_data_in.variant == EquationData::Variant::DivFree)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipNormal)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeBell)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleNoSlip)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlip)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreePoiseuilleInhom)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else if(equation_data_in.variant == EquationData::Variant::DivFreeNoSlipExp)
        return std::make_shared<ManufacturedLoad<dim, is_simplified>>(analytical_solution);
      else
        AssertThrow(false, ExcMessage("Not supported..."));
      return nullptr;
    }()),
    n_colors_system(0U),
    n_mg_levels(0U),
    triangulation(std::make_shared<parallel::distributed::Triangulation<dim>>(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)),
    mapping(1),
    fe(make_finite_element()),
    finite_element_stream(make_finite_element_stream()),
    /// TODO coloring should depend on pressure finite elements
    user_coloring([&]() -> std::shared_ptr<ColoringBase<dim>> {
      if constexpr(dof_layout_v == TPSS::DoFLayout::Q)
        return std::make_shared<TiledColoring<dim>>(rt_parameters_in.mesh);
      else if(dof_layout_v == TPSS::DoFLayout::DGQ)
        return std::make_shared<RedBlackColoring<dim>>(rt_parameters_in.mesh);
      else if(dof_layout_v == TPSS::DoFLayout::RT)
        return std::make_shared<RedBlackColoring<dim>>(rt_parameters_in.mesh);
      return std::shared_ptr<ColoringBase<dim>>();
    }()),
    do_assemble_pressure_mass_matrix(rt_parameters.solver.variant == "FGMRES_ILU" ||
                                     rt_parameters.solver.variant == "FGMRES_GMGvelocity")
{
  Assert(check_finite_element(), ExcMessage("Check default finite elements and dof_layout."));
  if(is_first_proc)
  {
    std::ostringstream oss;
    oss << "For this variant most solver variants do not converge."
        << "The mean value constraint seems to be problematic, "
        << "in particular the mean-value filter...";
    AssertThrow(equation_data_in.variant != EquationData::Variant::DivFreeBell,
                ExcMessage(oss.str()));
  }

  if(local_assembly == LocalAssembly::StreamLMW || local_assembly == LocalAssembly::StreamTensor)
    equation_data.setup_stream_functions = true;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
unsigned int
ModelProblem<dim, fe_degree_p, method, is_simplified>::max_level() const
{
  return triangulation->n_global_levels() - 1;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
template<typename T>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::print_parameter(
  const std::string & description,
  const T &           value) const
{
  *pcout << Util::parameter_to_fstring(description, value);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::print_informations() const
{
  print_parameter("////////////////////////////////////////",
                  is_simplified ? "SIMPLIFIED STOKES" : "STOKES");
  *pcout << equation_data.to_string();
  print_parameter("Local Assembly:", EquationData::str_local_assembly(local_assembly));
  *pcout << std::endl;
  print_parameter("Finite element:", fe->get_name());
  *pcout << rt_parameters.to_string();
  *pcout << std::endl;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
const FiniteElement<dim> &
ModelProblem<dim, fe_degree_p, method, is_simplified>::get_finite_element_velocity() const
{
  std::vector<bool> velocity_mask(dim, true);
  velocity_mask.push_back(false);
  const auto & fe_velocity = fe->get_sub_fe(velocity_mask);
  return fe_velocity;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
const FiniteElement<dim> &
ModelProblem<dim, fe_degree_p, method, is_simplified>::get_finite_element_pressure() const
{
  std::vector<bool> pressure_mask(dim, false);
  pressure_mask.push_back(true);
  const auto & fe_pressure = fe->get_sub_fe(pressure_mask);
  return fe_pressure;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
std::shared_ptr<FiniteElement<dim>>
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_finite_element() const
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



template<int dim, int fe_degree_p, Method method, bool is_simplified>
std::shared_ptr<FiniteElement<dim>>
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_finite_element_stream() const
{
  return std::make_shared<FE_Q<dim>>(fe_degree_v + 1);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
bool
ModelProblem<dim, fe_degree_p, method, is_simplified>::check_finite_element() const
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



template<int dim, int fe_degree_p, Method method, bool is_simplified>
bool
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_grid()
{
  make_grid_impl(rt_parameters.mesh);
  return true;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
bool
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_grid(const unsigned int n_refinements)
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



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_grid_impl(
  const MeshParameter & mesh_prms)
{
  triangulation->clear();
  *pcout << create_mesh(*triangulation, mesh_prms) << std::endl;
  pp_data.n_cells_global.push_back(triangulation->n_global_active_cells());
  pp_data.n_dimensions = dim;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
LinearAlgebra::distributed::Vector<double>
ModelProblem<dim, fe_degree_p, method, is_simplified>::compute_mass_foreach_pressure_dof() const
{
  const auto & locally_owned_dof_indices = dof_handler_pressure.locally_owned_dofs();
  IndexSet     locally_relevant_dof_indices;
  DoFTools::extract_locally_relevant_dofs(dof_handler_pressure, locally_relevant_dof_indices);

  LinearAlgebra::distributed::Vector<double> mass_foreach_dof(locally_owned_dof_indices,
                                                              locally_relevant_dof_indices,
                                                              MPI_COMM_WORLD);

  AffineConstraints<double> constraints_dummy;
  constraints_dummy.clear();
  constraints_dummy.reinit(locally_relevant_dof_indices);
  constraints_dummy.close();

  using ::MW::ScratchData;
  using ::MW::DoF::CopyData;

  auto cell_worker = [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
    FEValues<dim> & phi = scratch_data.fe_values;
    phi.reinit(cell);

    const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

    auto & cell_data = copy_data.cell_data.emplace_back(dofs_per_cell);

    cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

    for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_data.rhs(i) += phi.shape_value(i, q) * phi.JxW(q);
  };

  const auto copier = [&](const CopyData & copy_data) {
    for(const auto & cd : copy_data.cell_data)
      constraints_dummy.template distribute_local_to_global(cd.rhs,
                                                            cd.dof_indices,
                                                            mass_foreach_dof);
  };

  const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
  const UpdateFlags interface_update_flags = update_default;

  ScratchData<dim> scratch_data(
    mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);

  CopyData copy_data;

  MeshWorker::mesh_loop(dof_handler_pressure.begin_active(),
                        dof_handler_pressure.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells);

  return mass_foreach_dof;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
LinearAlgebra::distributed::Vector<double>
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_constant_pressure_mode(
  const unsigned int level) const
{
  const bool is_multigrid = level != numbers::invalid_unsigned_int;

  const auto & locally_owned_dof_indices = is_multigrid ?
                                             dof_handler_pressure.locally_owned_mg_dofs(level) :
                                             dof_handler_pressure.locally_owned_dofs();
  IndexSet locally_relevant_dof_indices;
  if(is_multigrid)
    DoFTools::extract_locally_relevant_level_dofs(dof_handler_pressure,
                                                  level,
                                                  locally_relevant_dof_indices);
  else
    DoFTools::extract_locally_relevant_dofs(dof_handler_pressure, locally_relevant_dof_indices);

  LinearAlgebra::distributed::Vector<double> mode(locally_owned_dof_indices,
                                                  locally_relevant_dof_indices,
                                                  MPI_COMM_WORLD);

  const auto compute_impl = [&](const auto & locally_owned_cells_range) {
    const bool is_dgq_legendre =
      dof_handler_pressure.get_fe().get_name().find("FE_DGQLegendre") != std::string::npos;

    std::vector<types::global_dof_index> dof_indices_on_cell(
      dof_handler_pressure.get_fe().dofs_per_cell);
    for(const auto & cell : locally_owned_cells_range)
    {
      cell->get_active_or_mg_dof_indices(dof_indices_on_cell);

      if(is_dgq_legendre || dof_layout_p == TPSS::DoFLayout::DGP)
        mode[dof_indices_on_cell.front()] = 1.;
      else if(dof_layout_p == TPSS::DoFLayout::DGQ || dof_layout_p == TPSS::DoFLayout::Q)
        for(const auto dof_index : dof_indices_on_cell)
          mode[dof_index] = 1.;
      else
        AssertThrow(false, ExcMessage("Dof layout is not supported."));
    }
    mode.compress(VectorOperation::insert);
  };

  if(is_multigrid)
    compute_impl(filter_iterators(dof_handler_pressure.mg_cell_iterators_on_level(level),
                                  IteratorFilters::LocallyOwnedLevelCell()));
  else
    compute_impl(filter_iterators(dof_handler_pressure.active_cell_iterators(),
                                  IteratorFilters::LocallyOwnedCell()));

  return mode;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::setup_system()
{
  Assert(check_finite_element(),
         ExcMessage("Does the choice of finite elements suit the dof_layout?"));

  /// book-keeping locally owned/relevant dof indices
  std::vector<IndexSet> lodof_indices_foreach_block;
  lodof_indices_foreach_block.reserve(2U);
  std::vector<IndexSet> lrdof_indices_foreach_block;
  lrdof_indices_foreach_block.reserve(2U);

  const auto & get_locally_relevant_dof_indices = [](const auto & dofh) {
    IndexSet locally_relevant_dof_indices;
    DoFTools::extract_locally_relevant_dofs(dofh, locally_relevant_dof_indices);
    return locally_relevant_dof_indices;
  };

  //: setup velocity system
  {
    dof_handler_velocity.reinit(*triangulation);
    dof_handler_velocity.distribute_dofs(get_finite_element_velocity());
    AssertThrow(!equation_data.use_cuthill_mckee, ExcMessage("TODO MPI..."));
    // if(do_cuthill_mckee)
    // {
    //   AssertThrow(false, ExcMessage("TODO MPI"));
    //   DoFRenumbering::Cuthill_McKee(dof_handler_velocity);
    // }

    // IndexSet & locally_owned_dof_indices =
    lodof_indices_foreach_block.emplace_back(dof_handler_velocity.locally_owned_dofs());
    IndexSet & locally_relevant_dof_indices = lrdof_indices_foreach_block.emplace_back(
      get_locally_relevant_dof_indices(dof_handler_velocity));

    /// homogeneous boundary conditions for the solution update
    zero_constraints_velocity.clear();
    zero_constraints_velocity.reinit(locally_relevant_dof_indices);
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
      for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
        VectorToolsFix::project_boundary_values_div_conforming(
          dof_handler_velocity, 0U, zero_velocity, boundary_id, zero_constraints_velocity, mapping);
    }
    zero_constraints_velocity.close();

    /// inhomogeneous boundary conditions for the particular solution
    constraints_velocity.clear();
    constraints_velocity.reinit(locally_relevant_dof_indices);
    if(dof_layout_v == TPSS::DoFLayout::Q)
    {
      print_parameter("Interpolating boundary (velo)", "...");
      const auto             component_range = std::make_pair<unsigned int>(0, dim);
      FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(),
                                                          component_range);
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
      FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(),
                                                          component_range);
      for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
        VectorToolsFix::project_boundary_values_div_conforming(dof_handler_velocity,
                                                               0U,
                                                               analytical_solution_velocity,
                                                               boundary_id,
                                                               constraints_velocity,
                                                               mapping);
    }
    constraints_velocity.close();
  }

  //: setup pressure system
  {
    dof_handler_pressure.reinit(*triangulation);
    dof_handler_pressure.distribute_dofs(get_finite_element_pressure());
    AssertThrow(!equation_data.use_cuthill_mckee, ExcMessage("TODO MPI..."));
    // if(do_cuthill_mckee)
    // {
    //   const bool cuthill_mckee_is_compatible = dof_layout_v == dof_layout_p;
    //   AssertThrow(
    //     cuthill_mckee_is_compatible,
    //     ExcMessage(
    //       "In general, reordering velocity as well as pressure dofs by a Cuthill-McKee algorithm
    //       does not provide the same order as a Cuthill-McKee reordering on the combined
    //       velocity-pressure dofs. If the same dof layout is used for the velocity and pressure
    //       the reorderings might coincide."));
    //   DoFRenumbering::Cuthill_McKee(dof_handler_pressure);
    // }

    IndexSet & locally_owned_dof_indices =
      lodof_indices_foreach_block.emplace_back(dof_handler_pressure.locally_owned_dofs());
    IndexSet & locally_relevant_dof_indices = lrdof_indices_foreach_block.emplace_back(
      get_locally_relevant_dof_indices(dof_handler_pressure));

    constraints_pressure.clear();
    constraints_pressure.reinit(locally_relevant_dof_indices);

    /// mean-value constraints: Use the space of mean-value free L^2 functions as
    /// pressure ansatz space. Therefore, compute the (unconstrained) pressure
    /// mass matrix and apply the coefficient vector which interpolates the
    /// constant-one-function. This leads to a weight for each degree of
    /// freedom. If the sum of these weights is zero the discrete pressure
    /// function is mean-value free.
    if(equation_data.do_mean_value_constraint)
    {
      print_parameter("Computing mean-value constraints (press)", "...");

      if(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1U)
      {
        /// If AffineConstraints is initialized only for all relevant indices not all
        /// procs should have communication to the first dof. Thus, constraining all
        /// dofs to fix the first dof does not apply in distributed codes.
        AssertThrow(is_first_proc, ExcMessage("Supported only in serial runs."));

        const auto &                  mass_foreach_dof   = compute_mass_foreach_pressure_dof();
        const types::global_dof_index first_pressure_dof = 0U;
        AssertThrow(mass_foreach_dof(0U) > 0., ExcMessage("First dof has no mass!"));
        constraints_pressure.add_line(first_pressure_dof);
        for(types::global_dof_index i = 1; i < mass_foreach_dof.size(); ++i)
          constraints_pressure.add_entry(0U, i, -mass_foreach_dof(i) / mass_foreach_dof(0));
      }

      else
      {
        const types::global_dof_index first_pressure_dof = 0U;
        if(locally_owned_dof_indices.is_element(first_pressure_dof))
          constraints_pressure.add_line(first_pressure_dof);
      }
    }

    /// Compute the constant pressure mode which is later used to filter
    /// matrix-vector products
    else
    {
      constant_mode_pressure = std::move(make_constant_pressure_mode());
    }

    constraints_pressure.close();
  }

  AssertDimension(lodof_indices_foreach_block.size(), 2U);
  AssertDimension(lrdof_indices_foreach_block.size(), 2U);

  const types::global_dof_index n_dofs_velocity = dof_handler_velocity.n_dofs();
  const types::global_dof_index n_dofs_pressure = dof_handler_pressure.n_dofs();

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
      /// TODO if we do not assemble the pressure mass matrix, the pressure-pressure
      /// block should be ::none
    }

  TrilinosWrappers::BlockSparsityPattern dsp(2U, 2U);

  //: setup sparsity pattern of system matrix
  {
    /// velocity - velocity
    {
      auto & this_dsp = dsp.block(0, 0);

      this_dsp.reinit(lodof_indices_foreach_block[0],
                      lodof_indices_foreach_block[0],
                      lrdof_indices_foreach_block[0],
                      MPI_COMM_WORLD);

      Table<2, DoFTools::Coupling> cell_integrals_mask_velocity;
      Table<2, DoFTools::Coupling> face_integrals_mask_velocity;
      cell_integrals_mask_velocity.reinit(dim, dim);
      face_integrals_mask_velocity.reinit(dim, dim);
      for(auto i = 0U; i < dim; ++i)
        for(auto j = 0U; j < dim; ++j)
        {
          cell_integrals_mask_velocity(i, j) = cell_integrals_mask(i, j);
          face_integrals_mask_velocity(i, j) = face_integrals_mask(i, j);
        }

      DoFTools::make_flux_sparsity_pattern(dof_handler_velocity,
                                           this_dsp,
                                           zero_constraints_velocity,
                                           false,
                                           cell_integrals_mask_velocity,
                                           face_integrals_mask_velocity,
                                           this_mpi_rank);
    }

    /// pressure - pressure
    {
      auto & this_dsp = dsp.block(1, 1);

      this_dsp.reinit(lodof_indices_foreach_block[1],
                      lodof_indices_foreach_block[1],
                      lrdof_indices_foreach_block[1],
                      MPI_COMM_WORLD);

      DoFTools::make_sparsity_pattern(dof_handler_pressure, this_dsp, constraints_pressure, true);
    }

    /// velocity - pressure
    {
      Table<2, DoFTools::Coupling> this_cell_integrals_mask(dim, 1U);
      Table<2, DoFTools::Coupling> this_face_integrals_mask(dim, 1U);
      for(auto i = 0U; i < dim; ++i)
      {
        this_cell_integrals_mask(i, 0) = cell_integrals_mask(i, dim);
        this_face_integrals_mask(i, 0) = face_integrals_mask(i, dim);
      }
      auto & this_dsp = dsp.block(0, 1);
      this_dsp.reinit(lodof_indices_foreach_block[0],
                      lodof_indices_foreach_block[1],
                      lrdof_indices_foreach_block[0],
                      MPI_COMM_WORLD);
      Tools::make_flux_sparsity_pattern(dof_handler_velocity,
                                        dof_handler_pressure,
                                        this_dsp,
                                        this_cell_integrals_mask,
                                        this_face_integrals_mask);
    }

    /// pressure - velocity
    {
      Table<2, DoFTools::Coupling> this_cell_integrals_mask(1U, dim);
      Table<2, DoFTools::Coupling> this_face_integrals_mask(1U, dim);
      for(auto j = 0U; j < dim; ++j)
      {
        this_cell_integrals_mask(0, j) = cell_integrals_mask(dim, j);
        this_face_integrals_mask(0, j) = face_integrals_mask(dim, j);
      }
      auto & this_dsp = dsp.block(1, 0);
      this_dsp.reinit(lodof_indices_foreach_block[1],
                      lodof_indices_foreach_block[0],
                      lrdof_indices_foreach_block[1],
                      MPI_COMM_WORLD);
      Tools::make_flux_sparsity_pattern(dof_handler_pressure,
                                        dof_handler_velocity,
                                        this_dsp,
                                        this_cell_integrals_mask,
                                        this_face_integrals_mask);
    }

    dsp.collect_sizes();
    dsp.compress();
  }

  system_matrix.clear();
  system_matrix.initialize(dsp,
                           lodof_indices_foreach_block,
                           lrdof_indices_foreach_block,
                           MPI_COMM_WORLD);

  if(do_assemble_pressure_mass_matrix)
  {
    pressure_mass_matrix.clear();
    pressure_mass_matrix.reinit(dsp.block(1, 1));
  }

  /// particular solution vector
  system_matrix.initialize_dof_vector(system_solution);
  constraints_velocity.distribute(system_solution.block(0));
  constraints_pressure.set_zero(system_solution.block(1));

  /// homogeneous solution vector
  system_matrix.initialize_dof_vector(system_delta_x);
  zero_constraints_velocity.distribute(system_delta_x.block(0));
  constraints_pressure.set_zero(system_delta_x.block(1));

  /// right-hand side vector
  system_matrix.initialize_dof_vector(system_rhs);

  print_parameter("Number of degrees of freedom (velocity):", n_dofs_velocity);
  print_parameter("Number of degrees of freedom (pressure):", n_dofs_pressure);
  print_parameter("Number of degrees of freedom (total):", n_dofs_velocity + n_dofs_pressure);
  *pcout << std::endl;

  pp_data.n_dofs_global.push_back(n_dofs_velocity);
  pp_data_pressure.n_dofs_global.push_back(n_dofs_pressure);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::assemble_system_velocity_pressure()
{
  system_rhs.zero_out_ghost_values();

  /// Assemble the velocity block, here block(0,0).
  {
    using Velocity::SIPG::MW::CopyData;

    using Velocity::SIPG::MW::ScratchData;

    using MatrixIntegrator = Velocity::SIPG::MW::MatrixIntegrator<dim, false, is_simplified>;

    const auto             component_range = std::make_pair<unsigned int>(0, dim);
    FunctionExtractor<dim> load_function_velocity(load_function.get(), component_range);
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);

    system_solution.block(0).update_ghost_values();
    const auto * particular_solution_velocity =
      (use_conf_method || use_hdivsipg_method) ? &(system_solution.block(0)) : nullptr;

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
        matrix_integrator.boundary_worker_tangential(cell, face_no, scratch_data, copy_data);
      else
        AssertThrow(false, ExcMessage("This velocity dof layout is not supported."));
    };

    const auto distribute_local_to_global_impl = [&](const auto & cd) {
      zero_constraints_velocity.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        cd.matrix, cd.rhs, cd.dof_indices, system_matrix.block(0, 0), system_rhs.block(0));
    };

    const auto copier = [&](const CopyData & copy_data) {
      AssertIndexRange(copy_data.cell_data.size(), 2U);

      for(const auto & cd : copy_data.cell_data)
        distribute_local_to_global_impl(cd);

      for(auto & cdf : copy_data.face_data)
        distribute_local_to_global_impl(cdf);
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

    CopyData copy_data;

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
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
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
      for(const auto & cd : copy_data.cell_data)
        constraints_pressure
          .template distribute_local_to_global /*<TrilinosWrappers::SparseMatrix>*/ (
            /*cd.matrix, */ cd.rhs,
            cd.dof_indices /*, system_matrix.block(1, 1)*/,
            system_rhs.block(1));
    };

    const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_default;

    ScratchData<dim> scratch_data(
      mapping, dof_handler_pressure.get_fe(), n_q_points_1d, update_flags, interface_update_flags);

    CopyData copy_data;

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
    using MatrixIntegrator = VelocityPressure::MW::Mixed::MatrixIntegrator<dim, false>; // TODO ???
    using CellIterator     = typename MatrixIntegrator::IteratorType;

    const auto * particular_solution_velocity =
      (use_conf_method || use_hdivsipg_method) ? &(system_solution.block(0)) : nullptr;

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

    const auto & distribute_local_to_global_impl = [&](const auto & cd, const auto & cd_flipped) {
      zero_constraints_velocity.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        cd.matrix,
        cd.dof_indices,
        constraints_pressure,
        cd.dof_indices_column,
        system_matrix.block(0, 1));
      constraints_pressure.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
        cd_flipped.matrix,
        cd.dof_indices_column,
        zero_constraints_velocity,
        cd.dof_indices,
        system_matrix.block(1, 0));

      zero_constraints_velocity.distribute_local_to_global(cd.rhs,
                                                           cd.dof_indices,
                                                           system_rhs.block(0));
      constraints_pressure.distribute_local_to_global(cd_flipped.rhs,
                                                      cd.dof_indices_column,
                                                      system_rhs.block(1));
    };

    const auto copier = [&](const CopyData & copy_data_pair) {
      const auto & [copy_data, copy_data_flipped] = copy_data_pair;

      AssertDimension(copy_data.cell_data.size(), copy_data_flipped.cell_data.size());
      AssertDimension(copy_data.face_data.size(), copy_data_flipped.face_data.size());

      auto cd_flipped = copy_data_flipped.cell_data.cbegin();
      for(auto cd = copy_data.cell_data.cbegin(); cd != copy_data.cell_data.cend();
          ++cd, ++cd_flipped)
        distribute_local_to_global_impl(*cd, *cd_flipped);

      auto cdf_flipped = copy_data_flipped.face_data.cbegin();
      for(auto cdf = copy_data.face_data.cbegin(); cdf != copy_data.face_data.cend();
          ++cdf, ++cdf_flipped)
        distribute_local_to_global_impl(*cdf, *cdf_flipped);
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

    CopyData copy_data;

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
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
                            boundary_worker,
                            face_worker);
    else
      AssertThrow(false, ExcMessage("This FEM is not supported."));
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  /// DEBUG
  // Util::gather_distributed_blockmatrix(system_matrix);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::assemble_system()
{
  assemble_system_velocity_pressure();

  if(do_assemble_pressure_mass_matrix)
  {
    assemble_pressure_mass_matrix_impl<false>(
      pressure_mass_matrix, dof_handler_pressure, mapping, constraints_pressure, equation_data);
  }
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::assemble_system_step56()
{
  AssertThrow(
    false,
    ExcMessage(
      "This method was available in a previous version when Stokes programs were still running in serial. Visit the git commit tree to find the actual implementation."));
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
std::shared_ptr<
  const MGCollectionVelocity<dim,
                             ModelProblem<dim, fe_degree_p, method, is_simplified>::fe_degree_v,
                             ModelProblem<dim, fe_degree_p, method, is_simplified>::dof_layout_v,
                             is_simplified>>
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_multigrid_velocity()
{
  const auto mgc_velocity =
    std::make_shared<MGCollectionVelocity<dim, fe_degree_v, dof_layout_v, is_simplified>>(
      rt_parameters, equation_data);

  dof_handler_velocity.distribute_mg_dofs();

  mgc_velocity->dof_handler = &dof_handler_velocity;
  mgc_velocity->mapping     = &mapping;
  mgc_velocity->cell_integrals_mask.reinit(dim, dim);
  mgc_velocity->face_integrals_mask.reinit(dim, dim);
  for(auto i = 0U; i < dim; ++i)
    for(auto j = 0U; j < dim; ++j)
    {
      mgc_velocity->cell_integrals_mask(i, j) = cell_integrals_mask(i, j);
      mgc_velocity->face_integrals_mask(i, j) = face_integrals_mask(i, j);
    }

  mgc_velocity->initialize(max_level(), user_coloring);

  const unsigned int mg_level_max = max_level();
  const unsigned int mg_level_min = rt_parameters.multigrid.coarse_level;
  this->n_mg_levels               = mg_level_max - mg_level_min + 1;
  this->n_colors_system           = mgc_velocity->n_colors();

  return mgc_velocity;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
std::shared_ptr<const MGCollectionVelocityPressure<
  dim,
  fe_degree_p,
  ModelProblem<dim, fe_degree_p, method, is_simplified>::dof_layout_v,
  ModelProblem<dim, fe_degree_p, method, is_simplified>::fe_degree_v,
  ModelProblem<dim, fe_degree_p, method, is_simplified>::local_assembly,
  is_simplified>>
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_multigrid_velocity_pressure()
{
  const auto mgc_velocity_pressure =
    std::make_shared<MGCollectionVelocityPressure<dim,
                                                  fe_degree_p,
                                                  dof_layout_v,
                                                  fe_degree_v,
                                                  local_assembly,
                                                  is_simplified>>(rt_parameters, equation_data);

  const auto & prms = rt_parameters.multigrid;

  const unsigned int mg_level_max = max_level();
  const unsigned int mg_level_min = prms.coarse_level;

  dof_handler_velocity.distribute_mg_dofs();
  dof_handler_pressure.distribute_mg_dofs();

  mgc_velocity_pressure->dof_handler_velocity = &dof_handler_velocity;
  mgc_velocity_pressure->dof_handler_pressure = &dof_handler_pressure;
  if(equation_data.setup_stream_functions)
  {
    dof_handler_stream.reinit(*triangulation);
    dof_handler_stream.distribute_dofs(*finite_element_stream);
    dof_handler_stream.distribute_mg_dofs();
    mgc_velocity_pressure->dof_handler_stream = &dof_handler_stream;
  }
  mgc_velocity_pressure->mapping             = &mapping;
  mgc_velocity_pressure->cell_integrals_mask = cell_integrals_mask;
  mgc_velocity_pressure->face_integrals_mask = face_integrals_mask;
  if(prms.coarse_grid.solver_variant == CoarseGridParameter::SolverVariant::Iterative)
    mgc_velocity_pressure->coarse_constant_pressure_mode =
      std::move(make_constant_pressure_mode(mg_level_min));

  mgc_velocity_pressure->initialize(mg_level_max, user_coloring);

  this->n_mg_levels     = mg_level_max - mg_level_min + 1;
  this->n_colors_system = mgc_velocity_pressure->n_colors();

  return mgc_velocity_pressure;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
std::shared_ptr<SolverControl>
ModelProblem<dim, fe_degree_p, method, is_simplified>::make_solver_control() const
{
  return make_solver_control_impl(rt_parameters.solver);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
template<typename PreconditionerType>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::iterative_solve_impl(
  const PreconditionerType & preconditioner,
  const std::string          solver_variant)
{
  auto solver_control = make_solver_control();

  SolverSelector<vector_type> iterative_solver;
  iterative_solver.set_control(*solver_control);
  iterative_solver.select(solver_variant);

  if(solver_variant == "gmres")
  {
    SolverGMRES<vector_type>::AdditionalData additional_data;
    additional_data.right_preconditioning = rt_parameters.solver.use_right_preconditioning;
    if(additional_data.right_preconditioning)
      AssertThrow(false, ExcMessage("not tested after changes..."));
    // additional_data.use_default_residual = false;
    additional_data.max_n_tmp_vectors = 100;
    iterative_solver.set_data(additional_data);
  }

  if(!equation_data.do_mean_value_constraint)
  {
    Assert(constant_mode_pressure.get_partitioner()->is_compatible(
             *(system_delta_x.block(1).get_partitioner())),
           ExcMessage("The vector partitioning is incompatible."));

    /// We "filter" the constant pressure mode after each matrix-vector
    /// multiplication with the system matrix.
    BlockSparseMatrixFiltered system_matrix_with_filter(system_matrix, constant_mode_pressure);

    iterative_solver.solve(system_matrix_with_filter, system_delta_x, system_rhs, preconditioner);
  }
  else
  {
    iterative_solver.solve(system_matrix, system_delta_x, system_rhs, preconditioner);

    /// Apply mean value constraints for the pressure component (Dirichlet
    /// conditions for the velocity component have already been applied to
    /// system_solution).
    constraints_pressure.distribute(system_delta_x.block(1));
  }

  /// Add the homogeneous solution to the particular solution.
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
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::solve()
{
  if(rt_parameters.solver.variant == "direct")
  {
    AssertThrow(
      false,
      ExcMessage(
        "Trilinoswrappers don't support block vectors/matrices. Check previous git commits for more details..."));
  }

  if(rt_parameters.solver.variant == "FGMRES_ILU")
  {
    AssertThrow(
      false,
      ExcMessage(
        "This variant was available in a serial version of this code. Check previous git commits for more details..."));
  }

  else if(rt_parameters.solver.variant == "FGMRES_GMGvelocity")
  {
    // This is used to pass whether or not we want to solve for A inside
    // the preconditioner.  One could change this to false to see if
    // there is still convergence and if so does the program then run
    // faster or slower
    const bool use_expensive = true;

    const auto mgc_velocity = make_multigrid_velocity();

    auto & A_preconditioner = mgc_velocity->get_preconditioner();

    TrilinosWrappers::PreconditionILU S_preconditioner;
    S_preconditioner.initialize(pressure_mass_matrix);

    const BlockSchurPreconditionerStep56<typename std::decay<decltype(A_preconditioner)>::type,
                                         typename std::decay<decltype(S_preconditioner)>::type>
      preconditioner(
        system_matrix, pressure_mass_matrix, A_preconditioner, S_preconditioner, use_expensive);

    /// BlockSchurPreconditionerStep56 is designed as right preconditioner since
    /// SolverFGMRES applies preconditioners always "from the right-hand". If
    /// changing the outer solver keep that in mind. An analogue left
    /// preconditioner is suggested in the extension section of step-22...
    iterative_solve_impl(preconditioner, "fgmres");

    *pcout << preconditioner.get_summary();
  }

  else if(rt_parameters.solver.variant == "GMRES_GMG")
  {
    const auto mgc_velocity_pressure = make_multigrid_velocity_pressure();

    auto & preconditioner = mgc_velocity_pressure->get_preconditioner();

    iterative_solve_impl(preconditioner, "gmres");
  }

  else if(rt_parameters.solver.variant == "CG_GMG")
  {
    const auto mgc_velocity_pressure = make_multigrid_velocity_pressure();

    auto & preconditioner = mgc_velocity_pressure->get_preconditioner();

    iterative_solve_impl(preconditioner, "cg");
  }

  else if(rt_parameters.solver.variant == "CG")
  {
    PreconditionIdentity preconditioner;
    rt_parameters.solver.n_iterations_max *= 10;

    iterative_solve_impl(preconditioner, "cg");
  }

  else if(rt_parameters.solver.variant == "GMRES")
  {
    PreconditionIdentity preconditioner;
    rt_parameters.solver.n_iterations_max *= 10;

    iterative_solve_impl(preconditioner, "gmres");
  }

  else
    AssertThrow(false, ExcMessage("Please, choose a valid solver variant."));

  post_process_solution_vector();

  pp_data.n_mg_levels.push_back(this->n_mg_levels);
  pp_data.n_colors_system.push_back(this->n_colors_system);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::post_process_solution_vector()
{
  const double mean_pressure = VectorTools::compute_mean_value(dof_handler_pressure,
                                                               QGauss<dim>(n_q_points_1d),
                                                               system_solution.block(1),
                                                               0U);

  if(!constant_mode_pressure.get_partitioner()->is_globally_compatible(
       *(system_solution.block(1).get_partitioner())))
    constant_mode_pressure = std::move(make_constant_pressure_mode());

  system_solution.block(1).add(-mean_pressure, constant_mode_pressure);

  print_parameter("Mean of pressure solution adjusted by:", -mean_pressure);
  *pcout << std::endl;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
double
ModelProblem<dim, fe_degree_p, method, is_simplified>::compute_L2_error_velocity() const
{
  const auto             component_range = std::make_pair<unsigned int>(0, dim);
  FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
  Vector<double>         difference_per_cell(triangulation->n_active_cells());
  VectorTools::integrate_difference(dof_handler_velocity,
                                    system_solution.block(0),
                                    analytical_solution_velocity,
                                    difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::L2_norm);
  /// in case of a distributed triangulation a MPI reduction is used internally
  const double error =
    VectorTools::compute_global_error(*triangulation, difference_per_cell, VectorTools::L2_norm);
  return error;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
double
ModelProblem<dim, fe_degree_p, method, is_simplified>::compute_L2_error_pressure() const
{
  const auto             component_range = std::make_pair<unsigned int>(dim, dim + 1);
  FunctionExtractor<dim> analytical_solution_pressure(analytical_solution.get(), component_range);
  Vector<double>         difference_per_cell(triangulation->n_active_cells());
  VectorTools::integrate_difference(dof_handler_pressure,
                                    system_solution.block(1U),
                                    analytical_solution_pressure,
                                    difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::L2_norm);
  /// in case of a distributed triangulation a MPI reduction is used internally
  const double error =
    VectorTools::compute_global_error(*triangulation, difference_per_cell, VectorTools::L2_norm);
  return error;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
double
ModelProblem<dim, fe_degree_p, method, is_simplified>::compute_H1semi_error_velocity() const
{
  const auto             component_range = std::make_pair<unsigned int>(0, dim);
  FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), component_range);
  Vector<double>         difference_per_cell(triangulation->n_active_cells());
  VectorTools::integrate_difference(dof_handler_velocity,
                                    system_solution.block(0U),
                                    analytical_solution_velocity,
                                    difference_per_cell,
                                    QGauss<dim>(n_q_points_1d + 2),
                                    VectorTools::H1_norm);
  /// in case of a distributed triangulation a MPI reduction is used internally
  const double error =
    VectorTools::compute_global_error(*triangulation, difference_per_cell, VectorTools::H1_norm);
  return error;
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::compute_discretization_errors()
{
  {
    const auto l2_error = compute_L2_error_velocity();
    print_parameter("Velocity error in the L2 norm:", l2_error);
    pp_data.L2_error.push_back(l2_error);
  }

  {
    const auto l2_error = compute_L2_error_pressure();
    print_parameter("Pressure error in the L2 norm:", l2_error);
    pp_data_pressure.L2_error.push_back(l2_error);
  }

  {
    const auto h1semi_error = compute_H1semi_error_velocity();
    print_parameter("Velocity error in the H1 seminorm:", h1semi_error);
    pp_data.H1semi_error.push_back(h1semi_error);
  }
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::output_results(
  const unsigned int refinement_cycle) const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation> velocity_interpretation(
    dim, DataComponentInterpretation::component_is_part_of_vector);
  std::vector<DataComponentInterpretation::DataComponentInterpretation> pressure_interpretation(
    1U, DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_velocity);
  // data_out.attach_triangulation(dof_handler_velocity.get_triangulation());

  vector_type exact_solution;
  system_matrix.initialize_dof_vector(exact_solution);

  /// exact solution
  {
    FunctionExtractor<dim> analytical_solution_velocity(analytical_solution.get(), {0, dim});
    VectorTools::interpolate(dof_handler_velocity,
                             analytical_solution_velocity,
                             exact_solution.block(0));
    exact_solution.update_ghost_values();
    /// u
    {
      std::vector<std::string> solution_names(dim, "exact_velocity");
      data_out.add_data_vector(dof_handler_velocity,
                               exact_solution.block(0U),
                               solution_names,
                               velocity_interpretation);
    }
  }

  /// discrete solution (velocity, pressure): (u, p)
  {
    system_solution.update_ghost_values();
    /// u
    {
      std::vector<std::string> solution_names(dim, "velocity");
      data_out.add_data_vector(dof_handler_velocity,
                               system_solution.block(0U),
                               solution_names,
                               velocity_interpretation);
    }
    /// p
    {
      std::vector<std::string> solution_names(1U, "pressure");
      data_out.add_data_vector(dof_handler_pressure,
                               system_solution.block(1U),
                               solution_names,
                               pressure_interpretation);
    }
  }

  /// discrete right hand side (velocity, pressure): (Fu, Fp)
  {
    system_rhs.update_ghost_values();
    /// Fu
    {
      std::vector<std::string> solution_names(dim, "rhs_velocity");
      data_out.add_data_vector(dof_handler_velocity,
                               system_rhs.block(0U),
                               solution_names,
                               velocity_interpretation);
    }
    /// Fp
    {
      std::vector<std::string> solution_names(1U, "rhs_pressure");
      data_out.add_data_vector(dof_handler_pressure,
                               system_rhs.block(1U),
                               solution_names,
                               pressure_interpretation);
    }
  }

  /// mpi partitioning of locally owned cells
  Vector<float> mpi_rank_foreach_cell(triangulation->n_active_cells());
  {
    for(unsigned int i = 0; i < mpi_rank_foreach_cell.size(); ++i)
      mpi_rank_foreach_cell(i) = triangulation->locally_owned_subdomain();
    data_out.add_data_vector(mpi_rank_foreach_cell, "mpi_ranks");
  }

  data_out.build_patches();

  std::ostringstream oss;
  oss << "stokes";
  oss << "_" << equation_data.sstr_equation_variant();
  oss << "_" << dim << "D";
  oss << "_" << fe_degree_p << "pdeg";
  data_out.write_vtu_with_pvtu_record("./",
                                      static_cast<std::string>(oss.str()),
                                      /*counter*/ refinement_cycle,
                                      MPI_COMM_WORLD,
                                      /*digits_for_counter*/ 2);
}



template<int dim, int fe_degree_p, Method method, bool is_simplified>
void
ModelProblem<dim, fe_degree_p, method, is_simplified>::run()
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

    compute_discretization_errors();

    // output_results(cycle);

    Utilities::System::MemoryStats mem;
    Utilities::System::get_memory_stats(mem);
    print_parameter("Memory used (VM Peak):", Util::si_metric_prefix(mem.VmPeak * 1000) + "B");

    *pcout << std::endl;
  }
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::initialize(std::shared_ptr<const MatrixFree<dim, Number>>
                                                                         mf_storage_in,
                                                      const EquationData equation_data_in)
{
  mf_storage = mf_storage_in;
  local_integrator_type::initialize(equation_data_in);
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
BlockSparseMatrixAugmented<
  dim,
  fe_degree_p,
  Number,
  dof_layout_v,
  fe_degree_v,
  local_assembly,
  is_simplified>::initialize(const TrilinosWrappers::BlockSparsityPattern & dsp,
                             const std::vector<IndexSet> & locally_owned_dof_indices_in,
                             const std::vector<IndexSet> & ghosted_dof_indices_in,
                             const MPI_Comm &              mpi_communicator_in)
{
  AssertDimension(locally_owned_dof_indices_in.size(), ghosted_dof_indices_in.size());
  AssertDimension(locally_owned_dof_indices_in.size(), 2U);
  AssertDimension(dsp.n_block_rows(), dsp.n_block_cols());
  AssertDimension(dsp.n_block_rows(), 2U);
  for(auto br = 0U; br < 2U; ++br)
    for(auto bc = 0U; bc < 2U; ++bc)
      Assert(dsp.block(br, bc).is_compressed(), ExcMessage("dsp block is not compressed."));

  matrix_type::reinit(dsp);

  AssertDimension(matrix_type::n_block_rows(), matrix_type::n_block_cols());
  AssertDimension(matrix_type::n_block_rows(), 2U);

  partitioners.clear();
  std::transform(locally_owned_dof_indices_in.cbegin(),
                 locally_owned_dof_indices_in.cend(),
                 ghosted_dof_indices_in.cbegin(),
                 std::back_inserter(partitioners),
                 [&](const auto & lodofs, const auto & gdofs) {
                   return std::make_shared<Utilities::MPI::Partitioner>(lodofs,
                                                                        gdofs,
                                                                        mpi_communicator_in);
                 });

  AssertDimension(matrix_type::n_block_rows(), partitioners.size());
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::clear()
{
  partitioners.clear();
  fullmatrix.reset();
  mf_storage.reset();
  matrix_type::clear();
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
std::shared_ptr<const MatrixFree<dim, Number>>
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::get_matrix_free() const
{
  AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  return mf_storage;
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
BlockSparseMatrixAugmented<
  dim,
  fe_degree_p,
  Number,
  dof_layout_v,
  fe_degree_v,
  local_assembly,
  is_simplified>::initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> & vec) const
{
  AssertDimension(this->n_block_rows(), this->n_block_cols());
  vec.reinit(this->n_block_rows());
  for(auto b = 0U; b < vec.n_blocks(); ++b)
    vec.block(b).reinit(partitioners.at(b));
  vec.collect_sizes();
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
BlockSparseMatrixAugmented<
  dim,
  fe_degree_p,
  Number,
  dof_layout_v,
  fe_degree_v,
  local_assembly,
  is_simplified>::initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> & vec,
                                        const unsigned int block_index) const
{
  AssertIndexRange(block_index, partitioners.size());
  vec.reinit(partitioners.at(block_index));
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
std::shared_ptr<const Utilities::MPI::Partitioner>
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::get_partitioner(const unsigned int block_index) const
{
  AssertIndexRange(block_index, partitioners.size());
  return partitioners.at(block_index);
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
template<typename VectorType>
void
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::vmult(VectorType & dst, const VectorType & src) const
{
  matrix_type::vmult(dst, src);
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::vmult(const ArrayView<Number>       dst,
                                                 const ArrayView<const Number> src) const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("No MPI support"));
  AssertDimension(dst.size(), matrix_type::m());
  AssertDimension(src.size(), matrix_type::n());
  std::vector<typename matrix_type::size_type> n_rows_per_block;
  AssertDimension(matrix_type::n_block_rows(), matrix_type::n_block_cols());
  for(auto b = 0U; b < matrix_type::n_block_rows(); ++b)
    n_rows_per_block.emplace_back(matrix_type::block(b, b).m());
  BlockVector<Number> v(n_rows_per_block); // src
  std::copy(src.cbegin(), src.cend(), v.begin());
  BlockVector<Number> w(n_rows_per_block); // dst
  matrix_type::vmult(w, v);                // w = A v
  std::copy(w.begin(), w.end(), dst.begin());
}



template<int dim,
         int fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
BlockSparseMatrixAugmented<dim,
                           fe_degree_p,
                           Number,
                           dof_layout_v,
                           fe_degree_v,
                           local_assembly,
                           is_simplified>::operator const FullMatrix<Number> &() const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("No MPI support"));

  if(!fullmatrix)
  {
    const auto & tmp = Tensors::matrix_to_table(*this);
    fullmatrix       = std::make_shared<FullMatrix<Number>>(table_to_fullmatrix(tmp));
  }
  return *fullmatrix;
}



BlockSparseMatrixFiltered::BlockSparseMatrixFiltered(
  const matrix_type &                                    matrix_in,
  const LinearAlgebra::distributed::Vector<value_type> & mode_in)
  : matrix(matrix_in), mode(&mode_in)
{
}



void
BlockSparseMatrixFiltered::vmult(
  LinearAlgebra::distributed::BlockVector<value_type> &       dst,
  const LinearAlgebra::distributed::BlockVector<value_type> & src) const
{
  matrix.vmult(dst, src);

  if(mode)
  {
    AssertDimension(2U, dst.n_blocks());
    auto &           dst_pressure        = dst.block(1);
    const value_type inner_product_value = (*mode) * dst_pressure;
    dst_pressure.add(-inner_product_value, *mode);
  }
}



void
BlockSparseMatrixFiltered::Tvmult(
  LinearAlgebra::distributed::BlockVector<value_type> &       dst,
  const LinearAlgebra::distributed::BlockVector<value_type> & src) const
{
  (void)dst, (void)src;
  AssertThrow(false, ExcMessage("TODO..."));
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::initialize(
  const TrilinosWrappers::SparsityPattern & dsp,
  const IndexSet &                          locally_owned_dof_indices,
  const IndexSet &                          ghosted_dof_indices,
  const MPI_Comm &                          mpi_communicator)
{
  Assert(dsp.is_compressed(), ExcMessage("The sparsity pattern isn't compressed."));
  matrix_type::reinit(dsp);
  partitioner = std::make_shared<const Utilities::MPI::Partitioner>(locally_owned_dof_indices,
                                                                    ghosted_dof_indices,
                                                                    mpi_communicator);
}


template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::initialize(
  std::shared_ptr<const MatrixFree<dim, double>> mf_storage_in,
  const EquationData                             equation_data_in)
{
  mf_storage = mf_storage_in;
  local_integrator_type::initialize(equation_data_in);
}


template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::initialize_dof_vector(
  vector_type & vec) const
{
  Assert(partitioner, ExcMessage("Did you initialize partitioner?"));
  vec.reinit(partitioner);
}


template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::initialize_dof_vector_mf(
  vector_type & vec) const
{
  Assert(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  mf_storage->initialize_dof_vector(vec);
}


template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
std::shared_ptr<const MatrixFree<dim, double>>
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::get_matrix_free() const
{
  AssertThrow(mf_storage, ExcMessage("Did you forget to initialize mf_storage?"));
  return mf_storage;
}


template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::vmult(
  const ArrayView<double>       dst,
  const ArrayView<const double> src) const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1, ExcMessage("No MPI support"));
  AssertDimension(dst.size(), matrix_type::m());
  AssertDimension(src.size(), matrix_type::n());
  vector_type v(matrix_type::n()); // src
  std::copy(src.cbegin(), src.cend(), v.begin());
  vector_type w(matrix_type::m()); // dst
  matrix_type::vmult(w, v);        // w = A v
  std::copy(w.begin(), w.end(), dst.begin());
}


template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::clear()
{
  mf_storage.reset();
  partitioner.reset();
  fullmatrix.reset();
  matrix_type::clear();
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
SparseMatrixAugmented<dim, fe_degree, dof_layout, is_simplified>::operator const FullMatrix<
  value_type> &() const
{
  if(!fullmatrix)
  {
    const auto & tmp = Tensors::matrix_to_table(*this);
    fullmatrix       = std::make_shared<FullMatrix<value_type>>(table_to_fullmatrix(tmp));
  }
  return *fullmatrix;
}



template<class PreconditionerAType, class PreconditionerSType>
BlockSchurPreconditionerStep56<PreconditionerAType, PreconditionerSType>::
  BlockSchurPreconditionerStep56(const TrilinosWrappers::BlockSparseMatrix & system_matrix_in,
                                 const TrilinosWrappers::SparseMatrix &      S_in,
                                 const PreconditionerAType &                 preconditioner_A_in,
                                 const PreconditionerSType &                 preconditioner_S_in,
                                 const bool                                  do_solve_A_in)
  : preconditioner_A(preconditioner_A_in),
    do_solve_A(do_solve_A_in),
    n_iterations_A(0),
    n_iterations_S(0)
{
  solver_S                   = std::make_shared<solver_S_type>(S_in, preconditioner_S_in);
  solver_S->solver_variant   = "cg";
  solver_S->n_iterations_max = 999;
  solver_S->rel_accuracy     = 1.e-6;

  if(do_solve_A)
  {
    solver_A = std::make_shared<solver_A_type>(system_matrix_in.block(0, 0), preconditioner_A);
    solver_A->solver_variant   = "cg";
    solver_A->n_iterations_max = 9999;
    solver_A->rel_accuracy     = 1.e-4;
    block_schur_expensive =
      std::make_shared<PreconditionBlockSchur<solver_A_type, solver_S_type>>(system_matrix_in,
                                                                             *solver_A,
                                                                             *solver_S);
  }
  else
  {
    block_schur_cheap =
      std::make_shared<PreconditionBlockSchur<preconditioner_A_type, solver_S_type>>(
        system_matrix_in, preconditioner_A, *solver_S);
  }
}



template<class PreconditionerAType, class PreconditionerSType>
void
BlockSchurPreconditionerStep56<PreconditionerAType, PreconditionerSType>::vmult(
  LinearAlgebra::distributed::BlockVector<double> &       dst,
  const LinearAlgebra::distributed::BlockVector<double> & src) const
{
  if(do_solve_A)
  {
    block_schur_expensive->vmult(dst, src);
    n_iterations_A += block_schur_expensive->preconditioner_A.n_solver_iterations;
    n_iterations_S += block_schur_expensive->preconditioner_S.n_solver_iterations;
  }
  else
  {
    block_schur_cheap->vmult(dst, src);
    n_iterations_A += 1;
    n_iterations_S += block_schur_cheap->preconditioner_S.n_solver_iterations;
  }
}



template<class PreconditionerAType, class PreconditionerSType>
std::string
BlockSchurPreconditionerStep56<PreconditionerAType, PreconditionerSType>::get_summary() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Summary of BlockSchurPreconditionerStep56:", "//////////");
  oss << Util::parameter_to_fstring("Accum. number of iterations (~A^-1):", n_iterations_A);
  oss << Util::parameter_to_fstring("Accum. number of iterations (~S^-1):", n_iterations_S);
  return oss.str();
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::MGCollectionVelocity(
  const RT::Parameter & rt_parameters_in,
  const EquationData &  equation_data_in)
  : dof_handler(nullptr),
    mapping(nullptr),
    parameters(rt_parameters_in.multigrid),
    equation_data(equation_data_in),
    use_tbb(rt_parameters_in.use_tbb),
    this_mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
{
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::clear_data()
{
  preconditioner_mg.reset();
  multigrid.reset();
  mg_matrix_wrapper.reset();
  mg_coarse_grid = nullptr;
  coarse_grid_solver.clear();
  mg_smoother_post = nullptr;
  mg_smoother_pre  = nullptr;
  mg_smoother_gauss_seidel.reset();
  mg_smoother_identity.reset();
  mg_schwarz_smoother_post.reset();
  mg_schwarz_smoother_pre.reset();
  mg_matrices.clear_elements();
  mg_transfer.clear();
  mg_constrained_dofs.reset();
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::initialize(
  const unsigned int                       mg_level_max,
  const std::shared_ptr<ColoringBase<dim>> user_coloring)
{
  //: clear multigrid infrastructure
  clear_data();

  //: setup multigrid data
  const unsigned mg_level_min = parameters.coarse_level;

  //: initialize multigrid constraints
  mg_constrained_dofs = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs->initialize(*dof_handler);
  if(dof_layout == TPSS::DoFLayout::Q)
    mg_constrained_dofs->make_zero_boundary_constraints(
      *dof_handler, equation_data.dirichlet_boundary_ids_velocity);
  else if(dof_layout == TPSS::DoFLayout::RT) // !!!
    for(const auto boundary_id : equation_data.dirichlet_boundary_ids_velocity)
      mg_constrained_dofs->make_no_normal_flux_constraints(*dof_handler, boundary_id, 0U);

  //: initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    const IndexSet & locally_owned_dof_indices = dof_handler->locally_owned_mg_dofs(level);
    IndexSet         locally_relevant_dof_indices;
    DoFTools::extract_locally_relevant_level_dofs(*dof_handler,
                                                  level,
                                                  locally_relevant_dof_indices);

    AffineConstraints<double> level_constraints;
    level_constraints.reinit(locally_relevant_dof_indices);
    if(dof_layout == TPSS::DoFLayout::Q || dof_layout == TPSS::DoFLayout::RT) // !!!
      level_constraints.add_lines(mg_constrained_dofs->get_boundary_indices(level));
    level_constraints.close();

    TrilinosWrappers::SparsityPattern dsp(locally_owned_dof_indices,
                                          locally_owned_dof_indices,
                                          locally_relevant_dof_indices,
                                          MPI_COMM_WORLD);
    /// TODO there is no variant which receives constraints: is it
    /// applicable in the Hdiv case?
    MGTools::make_flux_sparsity_pattern(
      *dof_handler, dsp, level, cell_integrals_mask, face_integrals_mask);
    dsp.compress();

    mg_matrices[level].initialize(dsp,
                                  locally_owned_dof_indices,
                                  locally_relevant_dof_indices,
                                  MPI_COMM_WORLD);

    {
      //: initialize a matrix-free storage dummy
      typename MatrixFree<dim, double>::AdditionalData mf_features;
      mf_features.mg_level = level;
      QGauss<1>  quadrature(n_q_points_1d);
      const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
      mf_storage->reinit(*mapping, *dof_handler, level_constraints, quadrature, mf_features);

      //: initialize FD::MatrixIntegrator (see SparseMatrixAugmented)
      mg_matrices[level].initialize(mf_storage, equation_data);
    }

    /// assemble the velocity system A_l on current level l.
    assemble_multigrid(level, level_constraints);
  }

  //: initialize multigrid transfer R_l
  mg_transfer.initialize_constraints(*mg_constrained_dofs);
  mg_transfer.build(*dof_handler);

  //: initialize Schwarz smoother S_l
  switch(parameters.pre_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
    {
      mg_smoother_identity = std::make_shared<const MGSmootherIdentity<vector_type>>();
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_smoother_identity.get();
      break;
    }
    case SmootherParameter::SmootherVariant::GaussSeidel:
    {
      typename TrilinosWrappers::PreconditionSSOR::AdditionalData ssor_features;
      ssor_features.omega = parameters.pre_smoother.damping_factor;
      auto tmp            = std::make_shared<mg_smoother_gauss_seidel_type>();
      tmp->set_steps(parameters.pre_smoother.n_smoothing_steps);
      tmp->initialize(mg_matrices, ssor_features);
      mg_smoother_gauss_seidel = tmp;
      mg_smoother_pre          = mg_smoother_gauss_seidel.get();
      break;
    }
    case SmootherParameter::SmootherVariant::Schwarz:
    {
      initialize_schwarz_smoothers(user_coloring);
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
      break;
    }
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }
  switch(parameters.post_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_post = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::GaussSeidel:
      AssertThrow(mg_smoother_gauss_seidel, ExcMessage("Not initialized."));
      mg_smoother_post = mg_smoother_gauss_seidel.get();
      break;
    case SmootherParameter::SmootherVariant::Schwarz:
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Not initialized"));
      mg_smoother_post = mg_schwarz_smoother_post.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }

  //: initialize coarse grid solver
  coarse_grid_solver.initialize(mg_matrices[mg_level_min], parameters.coarse_grid);
  mg_coarse_grid = &coarse_grid_solver;

  //: initialize geometric multigrid method
  mg_matrix_wrapper.initialize(mg_matrices);
  multigrid = std::make_shared<Multigrid<vector_type>>(mg_matrix_wrapper,
                                                       *mg_coarse_grid,
                                                       mg_transfer,
                                                       *mg_smoother_pre,
                                                       *mg_smoother_post,
                                                       mg_level_min,
                                                       mg_level_max);
}

template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::initialize_schwarz_smoothers(
  const std::shared_ptr<ColoringBase<dim>> user_coloring)
{
  Assert(parameters.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz,
         ExcMessage("Invalid smoothing variant."));
  for(auto level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
    AssertThrow(mg_matrices[level].get_matrix_free(), ExcMessage("mf_storage is not initialized."));

  //: pre-smoother
  typename mg_smoother_schwarz_type::AdditionalData mgss_data;
  if(parameters.pre_smoother.schwarz.userdefined_coloring)
  {
    Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
    mgss_data.coloring_func = std::ref(*user_coloring);
  }
  mgss_data.parameters = parameters.pre_smoother;
  mgss_data.use_tbb    = use_tbb;
  mgss_data.foreach_dofh.resize(1);                                                        // ???
  mgss_data.foreach_dofh[0].dirichlet_ids = equation_data.dirichlet_boundary_ids_velocity; // ???

  const auto mgss = std::make_shared<mg_smoother_schwarz_type>();
  mgss->initialize(mg_matrices, mgss_data);

  mg_schwarz_smoother_pre = mgss;

  //: post-smoother
  auto mgss_data_post       = mgss_data;
  mgss_data_post.parameters = parameters.post_smoother;

  /// do a shallow copy of mgss (i.e. pre-smoothers) if possible
  const auto mgss_post = std::make_shared<mg_smoother_schwarz_type>();
  mgss_post->initialize(*mg_schwarz_smoother_pre, mgss_data_post);

  mg_schwarz_smoother_post = mgss_post;
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
void
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::assemble_multigrid(
  const unsigned int          level,
  AffineConstraints<double> & level_constraints)
{
  AssertDimension(mg_matrices.min_level(), parameters.coarse_level);

  /// TODO distinguish between Hdiv-IP and SIPG !?
  constexpr bool use_sipg_method =
    dof_layout == TPSS::DoFLayout::DGQ || dof_layout == TPSS::DoFLayout::RT;
  constexpr bool use_conf_method = dof_layout == TPSS::DoFLayout::Q;

  using Velocity::SIPG::MW::CopyData;
  using Velocity::SIPG::MW::ScratchData;
  using MatrixIntegrator  = Velocity::SIPG::MW::MatrixIntegrator<dim, true, is_simplified>;
  using LevelCellIterator = typename MatrixIntegrator::IteratorType;

  MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

  auto cell_worker =
    [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
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
    matrix_integrator.face_worker(cell, f, sf, ncell, nf, nsf, scratch_data, copy_data);
  };

  auto boundary_worker = [&](const auto &         cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) {
    matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
  };

  const auto & distribute_local_to_global_impl = [&](const auto & cd) {
    level_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
      cd.matrix, cd.dof_indices, mg_matrices[level]);
  };

  const auto copier = [&](const CopyData & copy_data) {
    for(auto & cd : copy_data.cell_data)
      distribute_local_to_global_impl(cd);
    for(auto & cdf : copy_data.face_data)
      distribute_local_to_global_impl(cdf);
  };

  const UpdateFlags update_flags =
    update_values | update_gradients | update_quadrature_points | update_JxW_values;
  const UpdateFlags interface_update_flags = update_flags | update_normal_vectors;

  ScratchData<dim> scratch_data(*mapping,
                                dof_handler->get_fe(),
                                dof_handler->get_fe(),
                                n_q_points_1d,
                                update_flags,
                                update_flags,
                                interface_update_flags,
                                interface_update_flags);

  CopyData copy_data;

  if(use_conf_method)
    MeshWorker::mesh_loop(dof_handler->begin_mg(level),
                          dof_handler->end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  else if(use_sipg_method)
    MeshWorker::mesh_loop(dof_handler->begin_mg(level),
                          dof_handler->end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once |
                            MeshWorker::assemble_ghost_faces_once,
                          boundary_worker,
                          face_worker);
  else
    AssertThrow(false, ExcMessage("This FEM is not implemented"));

  mg_matrices[level].compress(VectorOperation::add);
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
const PreconditionMG<
  dim,
  typename MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::vector_type,
  typename MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::mg_transfer_type> &
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::get_preconditioner() const
{
  AssertThrow(multigrid, ExcMessage("Have you called initialize() before?"));
  if(!preconditioner_mg)
    preconditioner_mg =
      std::make_shared<PreconditionMG<dim, vector_type, mg_transfer_type>>(*dof_handler,
                                                                           *multigrid,
                                                                           mg_transfer);
  return *preconditioner_mg;
}



template<int dim, int fe_degree, TPSS::DoFLayout dof_layout, bool is_simplified>
unsigned int
MGCollectionVelocity<dim, fe_degree, dof_layout, is_simplified>::n_colors() const
{
  Assert(mg_smoother_pre || mg_smoother_post, ExcMessage("Not initialized."));
  if(mg_schwarz_smoother_pre)
    return mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
  else if(mg_schwarz_smoother_post)
    return mg_schwarz_smoother_post->get_subdomain_handler()->get_partition_data().n_colors();
  return 0;
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
MGCollectionVelocityPressure<
  dim,
  fe_degree_p,
  dof_layout_v,
  fe_degree_v,
  local_assembly,
  is_simplified>::MGCollectionVelocityPressure(const RT::Parameter & rt_parameters_in,
                                               const EquationData &  equation_data_in)
  : dof_handler_velocity(nullptr),
    dof_handler_pressure(nullptr),
    dof_handler_stream(nullptr),
    mapping(nullptr),
    parameters(rt_parameters_in.multigrid),
    equation_data(equation_data_in),
    use_tbb(rt_parameters_in.use_tbb),
    n_mpi_procs(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
    use_coarse_preconditioner(parameters.coarse_grid.solver_variant ==
                              CoarseGridParameter::SolverVariant::Iterative)
{
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
MGCollectionVelocityPressure<dim,
                             fe_degree_p,
                             dof_layout_v,
                             fe_degree_v,
                             local_assembly,
                             is_simplified>::clear_data()
{
  preconditioner_mg.reset();
  multigrid.reset();
  mg_matrix_wrapper.reset();
  mg_coarse_grid = nullptr;
  coarse_grid_solver.clear();
  mg_smoother_post = nullptr;
  mg_smoother_pre  = nullptr;
  mg_smoother_identity.reset();
  mg_schwarz_smoother_post.reset();
  mg_schwarz_smoother_pre.reset();
  mg_matrices.clear_elements();
  mg_transfer.reset();
  mg_constrained_dofs.reset();
  mgconstraineddofs.clear();
  mg_constrained_dofs_velocity.reset();
  mg_constrained_dofs_pressure.reset();
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
MGCollectionVelocityPressure<
  dim,
  fe_degree_p,
  dof_layout_v,
  fe_degree_v,
  local_assembly,
  is_simplified>::assemble_multigrid(const unsigned int                level,
                                     const AffineConstraints<double> & level_constraints_velocity,
                                     const AffineConstraints<double> & level_constraints_pressure,
                                     const AffineConstraints<double> & level_constraints_stream)
{
  AssertDimension(mg_matrices.min_level(), parameters.coarse_level);
  Assert(dof_handler_velocity, ExcMessage("Did you set dof_handler_velocity?"));
  Assert(dof_handler_pressure, ExcMessage("Did you set dof_handler_pressure?"));
  AssertThrow(TPSS::get_dof_layout(dof_handler_velocity->get_fe()) == dof_layout_v,
              ExcMessage("dof_handler_velocity and dof_layout_v are incompatible."));

  /// Initialize matrix-free storage on current level
  {
    typename MatrixFree<dim, double>::AdditionalData mf_features;
    mf_features.mg_level = level;

    std::vector<const DoFHandler<dim> *> dofhandlers{dof_handler_velocity, dof_handler_pressure};

    std::vector<const AffineConstraints<double> *> constraints_foreach_block{
      &level_constraints_velocity, &level_constraints_pressure};

    if(dof_handler_stream)
    {
      dofhandlers.push_back(dof_handler_stream);
      constraints_foreach_block.push_back(&level_constraints_stream);
    }

    QGauss<1> quadrature(n_q_points_1d);

    const auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
    mf_storage->reinit(*mapping, dofhandlers, constraints_foreach_block, quadrature, mf_features);

    /// initializes the local matrix integrator within BlockSparseMatrixAugmented
    mg_matrices[level].initialize(mf_storage, equation_data);
  }

  /// Assemble velocity-velocity block first.
  {
    using Velocity::SIPG::MW::CopyData;

    using Velocity::SIPG::MW::ScratchData;

    using MatrixIntegrator = Velocity::SIPG::MW::MatrixIntegrator<dim, true, is_simplified>;

    using LevelCellIterator = typename MatrixIntegrator::IteratorType;

    MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

    auto cell_worker =
      [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        matrix_integrator.cell_worker(cell, scratch_data, copy_data);
      };

    auto face_worker = [&](const auto &         cell,
                           const unsigned int & face_no,
                           const unsigned int & sface_no,
                           const auto &         ncell,
                           const unsigned int & nface_no,
                           const unsigned int & nsface_no,
                           ScratchData<dim> &   scratch_data,
                           CopyData &           copy_data) {
      if(use_sipg_method)
        matrix_integrator.face_worker(
          cell, face_no, sface_no, ncell, nface_no, nsface_no, scratch_data, copy_data);
      else if(use_hdivsipg_method)
        matrix_integrator.face_worker_tangential(
          cell, face_no, sface_no, ncell, nface_no, nsface_no, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const auto &         cell,
                               const unsigned int & face_no,
                               ScratchData<dim> &   scratch_data,
                               CopyData &           copy_data) {
      if(use_sipg_method)
        matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
      else if(use_hdivsipg_method)
        matrix_integrator.boundary_worker_tangential(cell, face_no, scratch_data, copy_data);
    };

    const auto distribute_local_to_global_impl = [&](const auto & cd) {
      level_constraints_velocity
        .template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cd.matrix, cd.dof_indices, mg_matrices[level].block(0, 0));
    };

    const auto copier = [&](const CopyData & copy_data) {
      AssertIndexRange(copy_data.cell_data.size(), 2U);
      for(const auto & cd : copy_data.cell_data)
        distribute_local_to_global_impl(cd);
      for(auto & cdf : copy_data.face_data)
        distribute_local_to_global_impl(cdf);
    };

    const UpdateFlags update_flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags = update_flags | update_normal_vectors;

    ScratchData<dim> scratch_data(*mapping,
                                  dof_handler_velocity->get_fe(),
                                  dof_handler_velocity->get_fe(),
                                  n_q_points_1d,
                                  update_flags,
                                  update_flags,
                                  interface_update_flags,
                                  interface_update_flags);

    CopyData copy_data;

    if(use_conf_method)
      MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                            dof_handler_velocity->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method || use_hdivsipg_method)
      MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                            dof_handler_velocity->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
                            boundary_worker,
                            face_worker);
    else
      AssertThrow(false, ExcMessage("This FEM is not supported"));
  }

  /// Assemble pressure-pressure block.
  {
    /// This block is zero!
  }

  /// Assemble velocity-pressure and pressure-velocity blocks
  {
    using VelocityPressure::MW::Mixed::CopyData;
    using VelocityPressure::MW::Mixed::ScratchData;
    using MatrixIntegrator  = VelocityPressure::MW::Mixed::MatrixIntegrator<dim, true>; // TODO ???
    using LevelCellIterator = typename MatrixIntegrator::IteratorType;

    MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, nullptr, equation_data);
    auto             cell_worker =
      [&](const LevelCellIterator & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        LevelCellIterator cell_ansatz(&dof_handler_pressure->get_triangulation(),
                                      cell->level(),
                                      cell->index(),
                                      dof_handler_pressure);
        matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
      };

    auto face_worker = [&](const LevelCellIterator & cell,
                           const unsigned int &      f,
                           const unsigned int &      sf,
                           const LevelCellIterator & ncell,
                           const unsigned int &      nf,
                           const unsigned int &      nsf,
                           ScratchData<dim> &        scratch_data,
                           CopyData &                copy_data) {
      LevelCellIterator cell_ansatz(&dof_handler_pressure->get_triangulation(),
                                    cell->level(),
                                    cell->index(),
                                    dof_handler_pressure);
      LevelCellIterator ncell_ansatz(&dof_handler_pressure->get_triangulation(),
                                     ncell->level(),
                                     ncell->index(),
                                     dof_handler_pressure);
      matrix_integrator.face_worker(
        cell, cell_ansatz, f, sf, ncell, ncell_ansatz, nf, nsf, scratch_data, copy_data);
    };

    auto boundary_worker = [&](const LevelCellIterator & cell,
                               const unsigned int &      face_no,
                               ScratchData<dim> &        scratch_data,
                               CopyData &                copy_data) {
      LevelCellIterator cell_ansatz(&dof_handler_pressure->get_triangulation(),
                                    cell->level(),
                                    cell->index(),
                                    dof_handler_pressure);
      matrix_integrator.boundary_worker(cell, cell_ansatz, face_no, scratch_data, copy_data);
    };

    const auto & distribute_local_to_global_impl = [&](const auto & cd, const auto & cd_flipped) {
      level_constraints_velocity
        .template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cd.matrix,
          cd.dof_indices,
          level_constraints_pressure,
          cd.dof_indices_column,
          mg_matrices[level].block(0, 1));
      level_constraints_pressure
        .template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cd_flipped.matrix,
          cd.dof_indices_column,
          level_constraints_velocity,
          cd.dof_indices,
          mg_matrices[level].block(1, 0));
    };

    const auto copier = [&](const CopyData & copy_data_pair) {
      const auto & [copy_data, copy_data_flipped] = copy_data_pair;

      AssertDimension(copy_data.cell_data.size(), copy_data_flipped.cell_data.size());
      AssertDimension(copy_data.face_data.size(), copy_data_flipped.face_data.size());

      auto cd         = copy_data.cell_data.cbegin();
      auto cd_flipped = copy_data_flipped.cell_data.cbegin();
      for(; cd != copy_data.cell_data.cend(); ++cd, ++cd_flipped)
        distribute_local_to_global_impl(*cd, *cd_flipped);

      auto cdf         = copy_data.face_data.cbegin();
      auto cdf_flipped = copy_data_flipped.face_data.cbegin();
      for(; cdf != copy_data.face_data.cend(); ++cdf, ++cdf_flipped)
        distribute_local_to_global_impl(*cdf, *cdf_flipped);
    };

    const UpdateFlags update_flags_velocity =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags update_flags_pressure =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;
    const UpdateFlags interface_update_flags_velocity =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
    const UpdateFlags interface_update_flags_pressure =
      update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

    ScratchData<dim> scratch_data(*mapping,
                                  dof_handler_velocity->get_fe(),
                                  dof_handler_pressure->get_fe(),
                                  n_q_points_1d,
                                  update_flags_velocity,
                                  update_flags_pressure,
                                  interface_update_flags_velocity,
                                  interface_update_flags_pressure);

    CopyData copy_data;

    if(use_conf_method || use_hdivsipg_method)
      MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                            dof_handler_velocity->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells);
    else if(use_sipg_method)
      MeshWorker::mesh_loop(dof_handler_velocity->begin_mg(level),
                            dof_handler_velocity->end_mg(level),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
                            boundary_worker,
                            face_worker);
    else
      AssertThrow(false, ExcMessage("This FEM is not implemented."));
  }

  mg_matrices[level].compress(VectorOperation::add);
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
MGCollectionVelocityPressure<dim,
                             fe_degree_p,
                             dof_layout_v,
                             fe_degree_v,
                             local_assembly,
                             is_simplified>::
  initialize_schwarz_smoothers(const std::shared_ptr<ColoringBase<dim>> user_coloring)
{
  Assert(parameters.pre_smoother.variant == SmootherParameter::SmootherVariant::Schwarz,
         ExcMessage("Invalid smoothing variant."));
  for(auto level = mg_matrices.min_level(); level <= mg_matrices.max_level(); ++level)
    AssertThrow(mg_matrices[level].mf_storage, ExcMessage("mf_storage is not initialized."));

  typename mg_smoother_schwarz_type::AdditionalData additional_data;

  //: pre-smoother
  {
    const auto mgss = std::make_shared<mg_smoother_schwarz_type>();
    if(parameters.pre_smoother.schwarz.userdefined_coloring)
    {
      Assert(user_coloring, ExcMessage("user_coloring is uninitialized."));
      additional_data.coloring_func = std::ref(*user_coloring);
    }
    additional_data.parameters = parameters.pre_smoother;
    additional_data.use_tbb    = use_tbb;
    additional_data.foreach_dofh.resize(dof_handler_stream ? 3 : 2);
    additional_data.foreach_dofh[0].dirichlet_ids = equation_data.dirichlet_boundary_ids_velocity;
    additional_data.foreach_dofh[1].dirichlet_ids = equation_data.dirichlet_boundary_ids_pressure;
    mgss->initialize(mg_matrices, additional_data);
    mg_schwarz_smoother_pre = mgss;
  }

  //: post-smoother (so far only shallow copy!)
  {
    const auto mgss_post       = std::make_shared<mg_smoother_schwarz_type>();
    additional_data.parameters = parameters.post_smoother;
    mgss_post->initialize(*mg_schwarz_smoother_pre, additional_data);
    mg_schwarz_smoother_post = mgss_post;
  }
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
void
MGCollectionVelocityPressure<dim,
                             fe_degree_p,
                             dof_layout_v,
                             fe_degree_v,
                             local_assembly,
                             is_simplified>::initialize(const unsigned int mg_level_max,
                                                        const std::shared_ptr<ColoringBase<dim>>
                                                          user_coloring)
{
  Assert(dof_handler_velocity, ExcMessage("Did you set dof_handler_velocity?"));
  Assert(dof_handler_pressure, ExcMessage("Did you set dof_handler_pressure?"));
  Assert(mapping, ExcMessage("Did you set mapping?"));

  //: clear multigrid infrastructure
  clear_data();

  //: setup multigrid data
  const unsigned mg_level_min = parameters.coarse_level;

  //: initialize multigrid constraints
  mg_constrained_dofs_velocity = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs_velocity->initialize(*dof_handler_velocity);
  if(dof_layout_v == TPSS::DoFLayout::Q)
    mg_constrained_dofs_velocity->make_zero_boundary_constraints(
      *dof_handler_velocity, equation_data.dirichlet_boundary_ids_velocity);
  /// make_no_normal_flux_constraints() seems not to work...
  else if(dof_layout_v == TPSS::DoFLayout::RT) // !!! TODO
    mg_constrained_dofs_velocity->make_zero_boundary_constraints(
      *dof_handler_velocity, equation_data.dirichlet_boundary_ids_velocity);

  mg_constrained_dofs_pressure = std::make_shared<MGConstrainedDoFs>();
  mg_constrained_dofs_pressure->initialize(*dof_handler_pressure);

  //: initialize level matrices A_l
  mg_matrices.resize(mg_level_min, mg_level_max);
  for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
  {
    /// book-keeping locally owned/relevant dof indices
    std::vector<IndexSet> lodof_indices_foreach_block;
    lodof_indices_foreach_block.emplace_back(dof_handler_velocity->locally_owned_mg_dofs(level));
    lodof_indices_foreach_block.emplace_back(dof_handler_pressure->locally_owned_mg_dofs(level));
    std::vector<IndexSet> lrdof_indices_foreach_block;
    DoFTools::extract_locally_relevant_level_dofs(*dof_handler_velocity,
                                                  level,
                                                  lrdof_indices_foreach_block.emplace_back());
    DoFTools::extract_locally_relevant_level_dofs(*dof_handler_pressure,
                                                  level,
                                                  lrdof_indices_foreach_block.emplace_back());
    IndexSet locally_relevant_dof_indices_stream;
    if(dof_handler_stream)
      DoFTools::extract_locally_relevant_level_dofs(*dof_handler_stream,
                                                    level,
                                                    locally_relevant_dof_indices_stream);

    AffineConstraints<double> level_constraints_velocity;
    level_constraints_velocity.reinit(lrdof_indices_foreach_block[0]);
    if(dof_layout_v == TPSS::DoFLayout::Q || dof_layout_v == TPSS::DoFLayout::RT) // !!!
      level_constraints_velocity.add_lines(
        mg_constrained_dofs_velocity->get_boundary_indices(level));
    level_constraints_velocity.close();

    /// TODO mean value constraints for pressure?
    AffineConstraints<double> level_constraints_pressure;
    level_constraints_pressure.reinit(lrdof_indices_foreach_block[1]);
    level_constraints_pressure.close();

    /// TODO do we need stream function constraints?
    AffineConstraints<double> level_constraints_stream;
    level_constraints_stream.reinit(locally_relevant_dof_indices_stream);
    level_constraints_stream.close();

    TrilinosWrappers::BlockSparsityPattern dsp(2U, 2U);
    {
      /// velocity - velocity
      {
        auto & this_dsp = dsp.block(0, 0);
        this_dsp.reinit(lodof_indices_foreach_block[0],
                        lodof_indices_foreach_block[0],
                        lrdof_indices_foreach_block[0],
                        MPI_COMM_WORLD);
        Table<2, DoFTools::Coupling> cell_integrals_mask_velocity(dim, dim);
        Table<2, DoFTools::Coupling> face_integrals_mask_velocity(dim, dim);
        for(auto i = 0U; i < dim; ++i)
          for(auto j = 0U; j < dim; ++j)
          {
            cell_integrals_mask_velocity(i, j) = cell_integrals_mask(i, j);
            face_integrals_mask_velocity(i, j) = face_integrals_mask(i, j);
          }
        MGTools::make_flux_sparsity_pattern(*dof_handler_velocity,
                                            this_dsp,
                                            level,
                                            cell_integrals_mask_velocity,
                                            face_integrals_mask_velocity);
      }

      /// pressure - pressure (empty but need to setup sizes)
      {
        auto & this_dsp = dsp.block(1, 1);
        this_dsp.reinit(lodof_indices_foreach_block[1],
                        lodof_indices_foreach_block[1],
                        lrdof_indices_foreach_block[1],
                        MPI_COMM_WORLD);
        if(use_coarse_preconditioner && level == mg_level_min)
        {
          AffineConstraints<double> empty_constraints_pressure;
          empty_constraints_pressure.reinit(lrdof_indices_foreach_block[1]);
          empty_constraints_pressure.close();
          MGTools::make_sparsity_pattern(*dof_handler_pressure,
                                         this_dsp,
                                         level,
                                         empty_constraints_pressure);
        }
      }

      /// velocity - pressure
      {
        Table<2, DoFTools::Coupling> this_cell_integrals_mask(dim, 1U);
        Table<2, DoFTools::Coupling> this_face_integrals_mask(dim, 1U);
        for(auto i = 0U; i < dim; ++i)
        {
          this_cell_integrals_mask(i, 0) = cell_integrals_mask(i, dim);
          this_face_integrals_mask(i, 0) = face_integrals_mask(i, dim);
        }
        auto & this_dsp = dsp.block(0, 1);
        this_dsp.reinit(lodof_indices_foreach_block[0],
                        lodof_indices_foreach_block[1],
                        lrdof_indices_foreach_block[0],
                        MPI_COMM_WORLD);
        Tools::make_flux_sparsity_pattern(*dof_handler_velocity,
                                          *dof_handler_pressure,
                                          this_dsp,
                                          this_cell_integrals_mask,
                                          this_face_integrals_mask,
                                          level);
      }

      /// pressure - velocity
      {
        Table<2, DoFTools::Coupling> this_cell_integrals_mask(1U, dim);
        Table<2, DoFTools::Coupling> this_face_integrals_mask(1U, dim);
        for(auto j = 0U; j < dim; ++j)
        {
          this_cell_integrals_mask(0, j) = cell_integrals_mask(dim, j);
          this_face_integrals_mask(0, j) = face_integrals_mask(dim, j);
        }
        auto & this_dsp = dsp.block(1, 0);
        this_dsp.reinit(lodof_indices_foreach_block[1],
                        lodof_indices_foreach_block[0],
                        lrdof_indices_foreach_block[1],
                        MPI_COMM_WORLD);
        Tools::make_flux_sparsity_pattern(*dof_handler_pressure,
                                          *dof_handler_velocity,
                                          this_dsp,
                                          this_cell_integrals_mask,
                                          this_face_integrals_mask,
                                          level);
      }

      dsp.collect_sizes();
      dsp.compress();
    }

    mg_matrices[level].initialize(dsp,
                                  lodof_indices_foreach_block,
                                  lrdof_indices_foreach_block,
                                  MPI_COMM_WORLD);

    //: assemble the velocity-pressure block system A_l on current level l.
    assemble_multigrid(level,
                       level_constraints_velocity,
                       level_constraints_pressure,
                       level_constraints_stream);

    /// DEBUG
    // Util::gather_distributed_blockmatrix(mg_matrices[level], level);

    if(use_coarse_preconditioner && level == mg_level_min)
    {
      coarse_mass_matrix_pressure.reinit(dsp.block(1, 1));
      AffineConstraints<double> empty_constraints_pressure;
      empty_constraints_pressure.reinit(lrdof_indices_foreach_block[1]);
      empty_constraints_pressure.close();
      assemble_pressure_mass_matrix_impl<true>(coarse_mass_matrix_pressure,
                                               *dof_handler_pressure,
                                               *mapping,
                                               empty_constraints_pressure,
                                               equation_data,
                                               level);
    }
  }

  //: initialize multigrid transfer R_l
  {
    const auto                             this_mg_transfer = std::make_shared<mg_transfer_type>();
    std::vector<const MGConstrainedDoFs *> mgconstraineddofs{mg_constrained_dofs_velocity.get(),
                                                             mg_constrained_dofs_pressure.get()};
    this_mg_transfer->initialize_constraints(mgconstraineddofs);
    std::vector<const DoFHandler<dim> *> dofhandlers{dof_handler_velocity, dof_handler_pressure};
    this_mg_transfer->build(dofhandlers);
    mg_transfer = this_mg_transfer;
  }

  //: initialize Schwarz smoother S_l
  switch(parameters.pre_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      mg_smoother_identity = std::make_shared<const MGSmootherIdentity<vector_type>>();
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::Schwarz:
      initialize_schwarz_smoothers(user_coloring);
      AssertThrow(mg_schwarz_smoother_pre, ExcMessage("Not initialized."));
      mg_smoother_pre = mg_schwarz_smoother_pre.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }
  switch(parameters.post_smoother.variant)
  {
    case SmootherParameter::SmootherVariant::None:
      AssertThrow(mg_smoother_identity, ExcMessage("Not initialized."));
      mg_smoother_post = mg_smoother_identity.get();
      break;
    case SmootherParameter::SmootherVariant::Schwarz:
      AssertThrow(mg_schwarz_smoother_post, ExcMessage("Not initialized"));
      mg_smoother_post = mg_schwarz_smoother_post.get();
      break;
    default:
      AssertThrow(false, ExcMessage("Invalid smoothing variant."));
  }

  //: initialize coarse grid solver
  {
    /// NOTE this part is experimental ...
    if(use_coarse_preconditioner)
    {
      /// TODO !!! the parallel interface of CoarseGridSolver is not aware of
      /// threshold_svd which handles the rank-deficiency of the coarse grid
      /// problem. Thus, ignoring the mean value constraint for the pressure
      /// component could lead to problems...
      AssertThrow(false, ExcMessage("Does not lead to a stable solver convergence..."));

      coarse_preconditioner_A =
        std::make_shared<coarse_solver_A_type>(mg_matrices[mg_level_min].block(0, 0));

      /// assemble mass matrix...
      coarse_mass_precondition_ilu_pressure.initialize(coarse_mass_matrix_pressure);
      coarse_preconditioner_S =
        std::make_shared<coarse_solver_S_type>(coarse_mass_matrix_pressure,
                                               coarse_mass_precondition_ilu_pressure);
      coarse_preconditioner_S->solver_variant   = "cg";
      coarse_preconditioner_S->n_iterations_max = 999;
      coarse_preconditioner_S->rel_accuracy     = 1.e-10; // 1.e-6;
      coarse_preconditioner_S->abs_accuracy     = 1.e-12;

      coarse_preconditioner =
        std::make_shared<PreconditionBlockSchur<coarse_solver_A_type, coarse_solver_S_type>>(
          mg_matrices[mg_level_min], *coarse_preconditioner_A, *coarse_preconditioner_S);

      Assert(coarse_constant_pressure_mode.size() != 0, ExcMessage("Not setup..."));
      coarse_matrix_with_filter =
        std::make_shared<BlockSparseMatrixFiltered>(mg_matrices[mg_level_min],
                                                    coarse_constant_pressure_mode);

      coarse_grid_solver_with_filter.initialize(*coarse_matrix_with_filter,
                                                parameters.coarse_grid,
                                                *coarse_preconditioner);

      mg_coarse_grid = &coarse_grid_solver_with_filter;
    }

    else
    {
      coarse_grid_solver.initialize(mg_matrices[mg_level_min], parameters.coarse_grid);
      mg_coarse_grid = &coarse_grid_solver;
    }
  }

  //: initialize geometric multigrid method
  mg_matrix_wrapper.initialize(mg_matrices);
  multigrid = std::make_shared<Multigrid<vector_type>>(mg_matrix_wrapper,
                                                       *mg_coarse_grid,
                                                       *mg_transfer,
                                                       *mg_smoother_pre,
                                                       *mg_smoother_post,
                                                       mg_level_min,
                                                       mg_level_max);
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
const PreconditionMG<dim,
                     typename MGCollectionVelocityPressure<dim,
                                                           fe_degree_p,
                                                           dof_layout_v,
                                                           fe_degree_v,
                                                           local_assembly,
                                                           is_simplified>::vector_type,
                     typename MGCollectionVelocityPressure<dim,
                                                           fe_degree_p,
                                                           dof_layout_v,
                                                           fe_degree_v,
                                                           local_assembly,
                                                           is_simplified>::mg_transfer_type> &
MGCollectionVelocityPressure<dim,
                             fe_degree_p,
                             dof_layout_v,
                             fe_degree_v,
                             local_assembly,
                             is_simplified>::get_preconditioner() const
{
  AssertThrow(multigrid, ExcMessage("multigrid is uninitialized."));
  std::vector<const DoFHandler<dim> *> dofhandlers{dof_handler_velocity, dof_handler_pressure};
  if(!preconditioner_mg)
    preconditioner_mg =
      std::make_shared<PreconditionMG<dim, vector_type, mg_transfer_type>>(dofhandlers,
                                                                           *multigrid,
                                                                           *mg_transfer);
  return *preconditioner_mg;
}



template<int             dim,
         int             fe_degree_p,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v,
         LocalAssembly   local_assembly,
         bool            is_simplified>
unsigned int
MGCollectionVelocityPressure<dim,
                             fe_degree_p,
                             dof_layout_v,
                             fe_degree_v,
                             local_assembly,
                             is_simplified>::n_colors() const
{
  Assert(mg_smoother_pre || mg_smoother_post, ExcMessage("Not initialized."));
  if(mg_schwarz_smoother_pre)
    return mg_schwarz_smoother_pre->get_subdomain_handler()->get_partition_data().n_colors();
  else if(mg_schwarz_smoother_post)
    return mg_schwarz_smoother_post->get_subdomain_handler()->get_partition_data().n_colors();
  return 0;
}

} // namespace Stokes

#endif // inclusion guard
