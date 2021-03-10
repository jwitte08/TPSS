
/*
 * helpers on multigrid infrastructure in deal.II
 *
 *  Created on: Oct 08, 2019
 *      Author: witte
 */

#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_selector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>


#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/matrix_utilities.h"
#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"
#include "solvers_and_preconditioners/smoother/schwarz_smoother.h"


#include "solver.h"
#include "utilities.h"



using namespace dealii;



/**
 * Collection of parameters for smoothers used in multigrid routines.
 */
struct SmootherParameter
{
  enum class SmootherVariant
  {
    None,
    GaussSeidel,
    Schwarz
  };
  static std::string
  str_smoother_variant(const SmootherVariant variant);

  SmootherParameter::SmootherVariant variant = SmootherParameter::SmootherVariant::Schwarz;
  int                                n_smoothing_steps     = 1;
  bool                               use_doubling_of_steps = false;
  SchwarzSmootherData                schwarz;
  double                             damping_factor = 1.;

  bool
  operator==(const SmootherParameter & other) const
  {
    bool is_equal = true;
    is_equal &= variant == other.variant;
    is_equal &= n_smoothing_steps == other.n_smoothing_steps;
    is_equal &= schwarz == other.schwarz;
    is_equal &= damping_factor == other.damping_factor;
    return is_equal;
  }

  std::string
  to_string() const;
};



/**
 * Collection of parameters for the coarsest grid in multigrid routines.
 */
struct CoarseGridParameter
{
  enum class SolverVariant
  {
    None,
    Iterative,
    DirectSVD
  };

  enum class PreconditionVariant
  {
    None,
    User
  };

  static std::string
  str_solver_variant(const SolverVariant variant);

  static std::string
  str_precondition_variant(const PreconditionVariant variant);

  SolverVariant       solver_variant       = SolverVariant::Iterative;
  PreconditionVariant precondition_variant = PreconditionVariant::None;
  std::string         iterative_solver     = "none"; // see SolverSelector
  double              accuracy             = 1.e-12;
  double              threshold_svd        = 0.;
  unsigned int        kernel_size          = 0;

  std::string
  to_string() const;
};



/**
 * Collection of multigrid parameters.
 */
struct MGParameter
{
  CoarseGridParameter coarse_grid;
  int                 coarse_level  = 0;
  std::string         cycle_variant = "V-cycle"; // TODO use enum
  SmootherParameter   pre_smoother;
  SmootherParameter   post_smoother;

  std::string
  to_string() const;
};



std::shared_ptr<SolverControl>
make_coarse_solver_control_impl(const CoarseGridParameter & prms)
{
  const auto solver_control = std::make_shared<SolverControl>();
  solver_control->set_tolerance(prms.accuracy);
  solver_control->log_history(false);
  solver_control->log_result(false);
  return solver_control;
}



/**
 * A wrapper class extending MGCoarseGridSVD to more vector types than 'Vector'
 * (the vector type that is internally used by LAPACKFullMatrix). In particular,
 * we have MPI-capable vector types in mind like
 * 'LinearAlgebra::distributed::Vector'. Of course, parallel vector types are
 * only compatible when the program runs with one MPI process.
 */
template<typename Number, typename VectorType>
class MGCoarseGridSVDSerial : public MGCoarseGridBase<VectorType>
{
public:
  void
  initialize(const FullMatrix<Number> & A, const double threshold = 0.)
  {
    solver_svd.initialize(A, threshold);
  }

  virtual void
  operator()(const unsigned int level, VectorType & dst, const VectorType & src) const override
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1U,
                ExcMessage("Vector type conversion only possible for one MPI process."));
    Vector<Number> tmp_dst(dst.size());
    Vector<Number> tmp_src(src.size());
    std::copy(src.begin(), src.end(), tmp_src.begin());
    solver_svd(level, tmp_dst, tmp_src);
    std::copy(tmp_dst.begin(), tmp_dst.end(), dst.begin());
  }

private:
  MGCoarseGridSVD<Number, Vector<Number>> solver_svd;
};



/**
 * A MGCoarseGridBase derivative enabling the direct solver
 * 'TrilinosWrappers::SolverDirect'.
 */
template<typename VectorType>
struct MGCoarseGridDirect : public MGCoarseGridBase<VectorType>
{
  using Base = MGCoarseGridBase<VectorType>;

  ~MGCoarseGridDirect() override final = default;

  void
  operator()(const unsigned int /*level*/,
             VectorType &       dst,
             const VectorType & src) const override final
  {
    AssertThrow(solver_direct, ExcMessage("solver_direct is uninitialized."));
    solver_direct->solve(dst, src);
  }

  std::shared_ptr<TrilinosWrappers::SolverDirect> solver_direct;
};



/**
 * A MGCoarseGridBase derivative treating block matrices. Template
 * specializations for different vector types can be found below.
 */
template<typename VectorType>
struct MGCoarseGridBlockSVD : public MGCoarseGridBase<VectorType>
{
  // using Base = MGCoarseGridBase<VectorType>;

  // ~MGCoarseGridDirect() override final = default;

  // void
  // operator()(const unsigned int /*level*/,
  //            VectorType &       dst,
  //            const VectorType & src) const override final
  // {
  //   ...
  // }
};



/**
 * This specialization handles parallel BlockVectors. Public members have to be
 * initialized properly: entries of the distributed block matrix have to
 * transferred to the first process following the global block-wise dof
 * numbering.
 */
template<typename Number>
struct MGCoarseGridBlockSVD<LinearAlgebra::distributed::BlockVector<Number>>
  : public MGCoarseGridBase<LinearAlgebra::distributed::BlockVector<Number>>
{
  using Base = MGCoarseGridBase<LinearAlgebra::distributed::BlockVector<Number>>;

  using local_matrix_type = Tensors::BlockMatrixBasic<MatrixAsTable<Number>>;

  ~MGCoarseGridBlockSVD() override final = default;

  void
  operator()(const unsigned int /*level*/,
             LinearAlgebra::distributed::BlockVector<Number> &       dst,
             const LinearAlgebra::distributed::BlockVector<Number> & src) const override final
  {
    Assert(master_mpi_rank != numbers::invalid_unsigned_int, ExcMessage("Master process not set."));
    AssertIndexRange(master_mpi_rank, Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) + 1);
    AssertDimension(Utilities::MPI::max(master_mpi_rank, MPI_COMM_WORLD),
                    Utilities::MPI::min(master_mpi_rank, MPI_COMM_WORLD));
    if(master_mpi_rank == this_mpi_rank)
    {
      Assert(master_matrix, ExcMessage("Master matrix not set"));
      AssertDimension(master_matrix->n_block_rows(), master_matrix->n_block_cols());
      AssertDimension(dst.n_blocks(), master_matrix->n_block_rows());
      AssertDimension(src.n_blocks(), master_matrix->n_block_cols());
    }
    AssertDimension(Utilities::MPI::max(master_mpi_rank == this_mpi_rank ?
                                          master_matrix->n_block_rows() :
                                          0,
                                        MPI_COMM_WORLD),
                    partitioners.size());

    /// switch from the mpi communication pattern of dst and src to the
    /// master-slave pattern cached
    AssertDimension(dst.n_blocks(), partitioners.size());
    AssertDimension(src.n_blocks(), partitioners.size());
    const auto setup_and_fill_vector = [&](auto & tmp_vec, const auto & vec, const bool do_fill) {
      for(auto b = 0U; b < partitioners.size(); ++b)
      {
        const auto & bvec     = vec.block(b);
        auto &       tmp_bvec = tmp_vec.block(b);
        tmp_bvec.reinit(partitioners[b]);
        Assert(tmp_bvec.locally_owned_elements() == bvec.locally_owned_elements(),
               ExcMessage("Mismatching locally owned dof indices."));
        if(do_fill)
          tmp_bvec.copy_locally_owned_data_from(bvec);
      }
      tmp_vec.collect_sizes();
    };
    LinearAlgebra::distributed::BlockVector<Number> tmp_dst(partitioners.size());
    setup_and_fill_vector(tmp_dst, dst, false);
    LinearAlgebra::distributed::BlockVector<Number> tmp_src(partitioners.size());
    setup_and_fill_vector(tmp_src, src, true);
    tmp_src.update_ghost_values();

    /// the gathered global values of src are used to locally invert on the
    /// master process. afterwards scatter local dst values to other processes
    BlockVector<Number> local_dst(partitioners.size());
    BlockVector<Number> local_src(partitioners.size());
    for(auto b = 0U; b < partitioners.size(); ++b)
    {
      const auto & tmp_bsrc   = tmp_src.block(b);
      auto &       local_bsrc = local_src.block(b);
      local_bsrc.reinit(tmp_bsrc.size());
      if(master_mpi_rank == this_mpi_rank)
        for(types::global_dof_index dof = 0; dof < local_bsrc.size(); ++dof)
          local_bsrc(dof) = tmp_bsrc(dof);
      local_dst.block(b).reinit(tmp_dst.block(b).size());
    }
    local_dst.collect_sizes();
    local_src.collect_sizes();

    if(master_mpi_rank == this_mpi_rank)
      master_matrix->apply_inverse(local_dst, local_src);

    tmp_dst.zero_out_ghosts();
    if(master_mpi_rank == this_mpi_rank)
      for(auto b = 0U; b < partitioners.size(); ++b)
        for(types::global_dof_index dof = 0; dof < local_dst.block(b).size(); ++dof)
          tmp_dst.block(b)(dof) = local_dst.block(b)(dof);

    /// finally, fill all dst values to be returned
    tmp_dst.compress(VectorOperation::add);
    for(auto b = 0U; b < partitioners.size(); ++b)
      dst.block(b).copy_locally_owned_data_from(tmp_dst.block(b));
  }

  unsigned int                             master_mpi_rank = numbers::invalid_unsigned_int;
  std::shared_ptr<const local_matrix_type> master_matrix;
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;

  const unsigned int this_mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
};



/**
 * A base class that handles the infrastructure required for coarse grid solvers
 * used in multigrid routines.
 */
/// TODO replace inheritance by operator ()
template<typename VectorType>
class CoarseGridSolverBase : public MGCoarseGridBase<VectorType>
{
public:
  using Base = MGCoarseGridBase<VectorType>;

  virtual ~CoarseGridSolverBase() override = default;

  void
  clear()
  {
    coarse_grid_solver.reset();
    solver_control.reset();
  }

  void virtual
  operator()(const unsigned int level, VectorType & dst, const VectorType & src) const override
  {
    AssertThrow(coarse_grid_solver, ExcMessage("The coarse grid solver is uninitialized."));
    coarse_grid_solver->operator()(level, dst, src);
  }

  std::shared_ptr<SolverControl>
  set_solver_control(const CoarseGridParameter & prms)
  {
    solver_control = make_coarse_solver_control_impl(prms);
    return solver_control;
  }

protected:
  std::shared_ptr<SolverControl>                      solver_control;
  std::shared_ptr<const MGCoarseGridBase<VectorType>> coarse_grid_solver;
};



template<typename MatrixType, typename VectorType>
class CoarseGridSolver : public CoarseGridSolverBase<VectorType>
{
public:
  using Base       = CoarseGridSolverBase<VectorType>;
  using value_type = typename MatrixType::value_type;

  void
  initialize_direct(const TrilinosWrappers::SparseMatrix & coarse_matrix,
                    const CoarseGridParameter &            prms)
  {
    AssertDimension(coarse_matrix.m(), coarse_matrix.n());
    AssertThrow(prms.threshold_svd == 0.,
                ExcMessage("A threshold isn't supported by TrilinosWrappers::SolverDirect."));

    const auto solver_control = Base::set_solver_control(prms);

    const auto solver_direct = std::make_shared<TrilinosWrappers::SolverDirect>(*solver_control);
    solver_direct->initialize(coarse_matrix);

    using coarse_grid_solver_type = MGCoarseGridDirect<VectorType>;
    const auto solver             = std::make_shared<coarse_grid_solver_type>();
    solver->solver_direct         = solver_direct;
    Base::coarse_grid_solver      = solver;
  }

  void
  initialize_direct(const FullMatrix<value_type> & coarse_matrix, const CoarseGridParameter & prms)
  {
    using coarse_grid_solver_type = MGCoarseGridSVDSerial<value_type, VectorType>;
    const auto solver             = std::make_shared<coarse_grid_solver_type>();
    solver->initialize(coarse_matrix, prms.threshold_svd);
    Base::coarse_grid_solver = solver;
  }

  void
  initialize_direct(const TrilinosWrappers::BlockSparseMatrix & distributed_blockmatrix,
                    const CoarseGridParameter &                 prms)
  {
    using coarse_grid_solver_type = MGCoarseGridBlockSVD<VectorType>;
    using local_matrix_type       = typename coarse_grid_solver_type::local_matrix_type;

    AssertDimension(distributed_blockmatrix.m(), distributed_blockmatrix.n());
    AssertDimension(distributed_blockmatrix.n_block_rows(), distributed_blockmatrix.n_block_cols());
    const auto n_blocks = distributed_blockmatrix.n_block_rows();

    const auto solver_control = Base::set_solver_control(prms);

    const auto solver = std::make_shared<coarse_grid_solver_type>();

    const auto this_mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    /// DEBUG
    // std::ostringstream oss;
    // oss << "debug.p" << this_mpi_rank << ".txt";
    // std::ofstream ofs;
    // ofs.open(oss.str(), std::ios_base::out);

    std::vector<IndexSet> global_ranges;
    std::vector<IndexSet> local_ranges;
    unsigned int          n_locally_owned_indices = 0;
    for(auto b = 0U; b < n_blocks; ++b)
    {
      const auto & this_matrix = distributed_blockmatrix.block(b, b);
      AssertDimension(this_matrix.m(), this_matrix.n());
      auto & this_global_range = global_ranges.emplace_back(this_matrix.m());
      this_global_range.add_range(0, this_matrix.m());
      this_global_range.compress();
      auto & this_local_range             = local_ranges.emplace_back(this_matrix.m());
      const auto [local_begin, local_end] = this_matrix.local_range();
      this_local_range.add_range(local_begin, local_end);
      this_local_range.compress();
      n_locally_owned_indices += this_matrix.local_size();
    }

    AssertDimension(Utilities::MPI::sum(n_locally_owned_indices, MPI_COMM_WORLD),
                    distributed_blockmatrix.m());
    AssertDimension(global_ranges.size(), n_blocks);
    AssertDimension(local_ranges.size(), n_blocks);

    {
      const auto n_loi_max = Utilities::MPI::max(n_locally_owned_indices, MPI_COMM_WORLD);
      solver->master_mpi_rank =
        Utilities::MPI::max(n_locally_owned_indices == n_loi_max ? this_mpi_rank : 0U,
                            MPI_COMM_WORLD);
    }

    const bool is_master_proc = this_mpi_rank == solver->master_mpi_rank;

    solver->partitioners.clear();
    for(auto b = 0U; b < n_blocks; ++b)
      solver->partitioners.emplace_back(std::make_shared<const Utilities::MPI::Partitioner>(
        local_ranges[b], is_master_proc ? global_ranges[b] : local_ranges[b], MPI_COMM_WORLD));

    const auto locally_relevant_blockmatrix = std::make_shared<local_matrix_type>();
    locally_relevant_blockmatrix->resize(2U, 2U);

    for(auto brow = 0U; brow < 2U; ++brow)
    {
      const auto                partitioner_row = solver->partitioners[brow];
      std::vector<unsigned int> this_g2l_row;
      if(is_master_proc)
        for(types::global_dof_index dof = 0; dof < partitioner_row->size(); ++dof)
          this_g2l_row.emplace_back(partitioner_row->global_to_local(dof));

      /// DEBUG
      // ofs << "g2l_row: " << vector_to_string(this_g2l_row) << std::endl;

      for(auto bcol = 0U; bcol < 2U; ++bcol)
      {
        const auto partitioner_col = solver->partitioners[bcol];

        std::vector<unsigned int> this_g2l_col;
        if(is_master_proc)
          for(types::global_dof_index dof = 0; dof < partitioner_col->size(); ++dof)
            this_g2l_col.emplace_back(partitioner_col->global_to_local(dof));

        /// DEBUG
        // ofs << "g2l_col: " << vector_to_string(this_g2l_col) << std::endl;

        /// this is a collective operation thus all processes need to call this method
        const auto & locally_relevant_matrix =
          Util::extract_locally_relevant_matrix(distributed_blockmatrix.block(brow, bcol),
                                                partitioner_row,
                                                partitioner_col);

        /// DEBUG
        // ofs << "block: " << brow << "-" << bcol << std::endl;
        // ofs << "size: " << locally_relevant_matrix.m() << "x" << locally_relevant_matrix.n()
        //     << std::endl;
        // locally_relevant_matrix.print_formatted(ofs);
        // ofs << std::endl;

        if(is_master_proc)
        {
          FullMatrix<double> tmp(locally_relevant_matrix.m(), locally_relevant_matrix.n());
          tmp.extract_submatrix_from(locally_relevant_matrix, this_g2l_row, this_g2l_col);
          locally_relevant_blockmatrix->get_block(brow, bcol) = tmp;
        }
      }
    }

    if(is_master_proc)
    {
      typename local_matrix_type::AdditionalData additional_data;
      additional_data.threshold   = prms.threshold_svd;
      additional_data.kernel_size = prms.kernel_size;
      locally_relevant_blockmatrix->invert(additional_data);

      solver->master_matrix = locally_relevant_blockmatrix;
    }

    Base::coarse_grid_solver = solver;
  }

  template<typename PreconditionerType = PreconditionIdentity>
  void
  initialize_iterative(const MatrixType &          coarse_matrix,
                       const CoarseGridParameter & prms,
                       const PreconditionerType &  preconditioner = PreconditionIdentity{})
  {
    constexpr bool is_prec_id = std::is_same<PreconditionerType, PreconditionIdentity>::value;

    const auto solver_control = Base::set_solver_control(prms);
    solver_control->set_max_steps(coarse_matrix.m());

    iterative_solver.set_control(*solver_control);
    iterative_solver.select(prms.iterative_solver);

    switch(prms.precondition_variant)
    {
      case CoarseGridParameter::PreconditionVariant::None:
      {
        AssertThrow(
          is_prec_id,
          ExcMessage(
            "If you want to make use of the preconditioner passed set precondition_variant to PreconditionVariant::User. If you want to apply the iterative solver without preconditioner do not pass a preconditioner to this function."));
        /// this looks akward but we need a local PreconditionIdentity object
        /// that is kept alive...
        using coarse_grid_solver_type = MGCoarseGridIterativeSolver<VectorType,
                                                                    SolverSelector<VectorType>,
                                                                    MatrixType,
                                                                    PreconditionIdentity>;
        Base::coarse_grid_solver      = std::make_shared<coarse_grid_solver_type>(iterative_solver,
                                                                             coarse_matrix,
                                                                             preconditioner_id);
        break;
      }
      case CoarseGridParameter::PreconditionVariant::User:
      {
        using coarse_grid_solver_type = MGCoarseGridIterativeSolver<VectorType,
                                                                    SolverSelector<VectorType>,
                                                                    MatrixType,
                                                                    PreconditionerType>;
        Base::coarse_grid_solver      = std::make_shared<coarse_grid_solver_type>(iterative_solver,
                                                                             coarse_matrix,
                                                                             preconditioner);
        break;
      }
      default:
        AssertThrow(false, ExcMessage("PreconditionVariant isn't supported."));
    }
  }

  template<typename PreconditionerType = PreconditionIdentity>
  void
  initialize(const MatrixType &          coarse_matrix,
             const CoarseGridParameter & prms,
             const PreconditionerType &  preconditioner = PreconditionIdentity{})
  {
    AssertDimension(coarse_matrix.m(), coarse_matrix.n());
    const auto solver_control = Base::set_solver_control(prms);
    solver_control->set_max_steps(coarse_matrix.m());

    switch(prms.solver_variant)
    {
      case CoarseGridParameter::SolverVariant::Iterative:
        initialize_iterative(coarse_matrix, prms, preconditioner);
        return;
      case CoarseGridParameter::SolverVariant::DirectSVD:
        initialize_direct(coarse_matrix, prms);
        return;
      default:
        AssertThrow(false, ExcMessage("Not supported."));
    }
  }

  void
  clear()
  {
    Base::clear();
  }

private:
  SolverSelector<VectorType> iterative_solver;
  /// we need this PreconditionIdentity object...
  const PreconditionIdentity preconditioner_id;
};



// template<typename VectorType>
// class CoarseGridSolver<TrilinosWrappers::SparseMatrix, VectorType>
//   : public CoarseGridSolverBase<VectorType>
// {
// public:
//   using Base = CoarseGridSolverBase<VectorType>;

//   ~CoarseGridSolver() override final = default;

//   void
//   initialize(const TrilinosWrappers::SparseMatrix & coarse_matrix, const CoarseGridParameter &
//   prms)
//   {
//     AssertDimension(coarse_matrix.m(), coarse_matrix.n());
//     const auto solver_control = Base::set_solver_control(prms);
//     solver_control->set_max_steps(coarse_matrix.m());

//     switch(prms.solver_variant)
//     {
//       case CoarseGridParameter::SolverVariant::DirectSVD:
//       {
//         solver_direct = std::make_shared<TrilinosWrappers::SolverDirect>(*solver_control);
//         solver_direct->initialize(coarse_matrix);
//         return;
//       }
//       default:
//         AssertThrow(false, ExcMessage("Not supported."));
//     }
//   }

//   void
//   clear()
//   {
//     solver_direct.reset();
//     Base::clear();
//   }

//   void
//   operator()(const unsigned int level,
//              VectorType &       dst,
//              const VectorType & src) const override final
//   {
//     if(solver_direct)
//     {
//       solver_direct->solve(dst, src);
//       return;
//     }

//     Base::operator()(level, dst, src);
//   }

// private:
//   std::shared_ptr<TrilinosWrappers::SolverDirect> solver_direct;
// };



/**
 * Extend MGTransferPrebuilt to be used with parallel block vectors. So far this
 * class depends on a block structure impose by a DoFHandler for each block.
 */
struct MGTransferBlockPrebuilt
  : public MGTransferBase<LinearAlgebra::distributed::BlockVector<double>>
{
  using value_type  = double;
  using vector_type = LinearAlgebra::distributed::BlockVector<value_type>;

  /**
   * Internal deal.II switch: enabling function overloads of copy_to_mg() and
   * copy_from_mg() based on a set of DoFHandlers instead of a single DoFHandler
   * as first argument.
   */
  static const bool supports_dof_handler_vector = true;

  void
  initialize_constraints(const std::vector<const MGConstrainedDoFs *> & mg_constrained_dofs);

  void
  clear();

  template<int dim>
  void
  build(const std::vector<const DoFHandler<dim> *> dof_handlers);

  virtual void
  prolongate(const unsigned int                                      to_level,
             LinearAlgebra::distributed::BlockVector<double> &       dst,
             const LinearAlgebra::distributed::BlockVector<double> & src) const override final;

  virtual void
  restrict_and_add(
    const unsigned int                                      from_level,
    LinearAlgebra::distributed::BlockVector<double> &       dst,
    const LinearAlgebra::distributed::BlockVector<double> & src) const override final;

  template<int dim>
  void
  copy_to_mg(const std::vector<const DoFHandler<dim> *> &                     dof_handlers,
             MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & dst,
             const LinearAlgebra::distributed::BlockVector<double> &          src) const;

  template<int dim>
  void
  copy_from_mg(const std::vector<const DoFHandler<dim> *> &                           dof_handlers,
               LinearAlgebra::distributed::BlockVector<double> &                      dst,
               const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const;

  template<int dim>
  void
  copy_from_mg_add(
    const std::vector<const DoFHandler<dim> *> &                           dof_handlers,
    LinearAlgebra::distributed::BlockVector<double> &                      dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const;

  template<int dim>
  void
  copy_to_mg(const DoFHandler<dim> &                                          dof_handler,
             MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & dst,
             const LinearAlgebra::distributed::BlockVector<double> &          src) const;

  template<int dim>
  void
  copy_from_mg(const DoFHandler<dim> &                                                dof_handler,
               LinearAlgebra::distributed::BlockVector<double> &                      dst,
               const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const;

  template<int dim>
  void
  copy_from_mg_add(
    const DoFHandler<dim> &                                                dof_handler,
    LinearAlgebra::distributed::BlockVector<double> &                      dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const;

  template<int dim, bool do_add>
  void
  copy_from_mg_impl(
    const std::vector<const DoFHandler<dim> *> &                           dof_handlers,
    LinearAlgebra::distributed::BlockVector<double> &                      dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const;

  std::size_t
  memory_consumption() const;

  void
  set_component_to_block_map(const std::vector<unsigned int> & map_in);

  void
  print_indices(std::ostream & os) const;

  template<bool do_clear_mg_constrained_dofs_dummy>
  void
  initialize_constraints_impl(const std::vector<const MGConstrainedDoFs *> & mg_constrained_dofs);

  template<bool do_clear_mg_constrained_dofs_dummy>
  void
  clear_impl();

  std::vector<MGTransferPrebuilt<LinearAlgebra::distributed::Vector<double>>>
    transfer_foreach_block;

  std::vector<MGConstrainedDoFs> mg_constrained_dofs_dummy;
};



template<int dim, typename MatrixType, typename PatchMatrixType, typename VectorType>
class MGSmootherSchwarz
  : public MGSmootherRelaxation<
      MatrixType,
      SchwarzSmoother<dim,
                      MatrixType,
                      SchwarzPreconditioner<dim, MatrixType, VectorType, PatchMatrixType>,
                      VectorType>,
      VectorType>
{
public:
  using value_type          = typename MatrixType::value_type;
  using preconditioner_type = SchwarzPreconditioner<dim, MatrixType, VectorType, PatchMatrixType>;
  using smoother_type       = SchwarzSmoother<dim, MatrixType, preconditioner_type, VectorType>;
  using Base                = MGSmootherRelaxation<MatrixType, smoother_type, VectorType>;

  struct AdditionalData
  {
    using CellIterator  = typename TPSS::PatchInfo<dim>::CellIterator;
    using PatchIterator = typename TPSS::PatchInfo<dim>::PatchIterator;
    using UserColoring  = typename std::function<std::vector<std::vector<PatchIterator>>(
      const std::vector<std::vector<CellIterator>> &      patches,
      const typename TPSS::PatchInfo<dim>::AdditionalData additional_data)>;
    using DataForeachDoFHandler =
      typename SubdomainHandler<dim, value_type>::AdditionalData::ForeachDoFHandler;

    UserColoring                       coloring_func;
    std::vector<DataForeachDoFHandler> foreach_dofh;
    SmootherParameter                  parameters;
    bool                               use_tbb = false;
  };

  template<typename OtherNumber>
  std::shared_ptr<const SubdomainHandler<dim, OtherNumber>>
  build_patch_storage(const unsigned                                            level,
                      const std::shared_ptr<const MatrixFree<dim, OtherNumber>> mf_storage,
                      const AdditionalData &                                    additional_data)
  {
    const auto & prms = additional_data.parameters;

    /// Fill additional data of SubdomainHandler
    typename SubdomainHandler<dim, OtherNumber>::AdditionalData sd_handler_data;
    fill_schwarz_smoother_data<dim, OtherNumber>(sd_handler_data, prms.schwarz);
    sd_handler_data.level = level;
    if(prms.schwarz.userdefined_coloring)
      sd_handler_data.coloring_func = additional_data.coloring_func;
    sd_handler_data.foreach_dofh = additional_data.foreach_dofh;
    sd_handler_data.use_tbb      = additional_data.use_tbb;

    /// Initialize SubdomainHandler
    const auto patch_storage = std::make_shared<SubdomainHandler<dim, OtherNumber>>();
    patch_storage->reinit(mf_storage, sd_handler_data);
    return patch_storage;
  }

  // TODO const 'mg_matrices' prevented by current initialize of SchwarzPreconditioner
  void
  initialize(MGLevelObject<MatrixType> & mg_matrices, const AdditionalData & additional_data)
  {
    const auto & prms = additional_data.parameters;
    AssertThrow(prms.variant == SmootherParameter::SmootherVariant::Schwarz,
                ExcMessage("Invalid SmootherVariant."));

    const auto &                                 schwarz_data = prms.schwarz;
    typename preconditioner_type::AdditionalData precondition_data;
    precondition_data.relaxation              = schwarz_data.damping_factor;
    precondition_data.symmetrized             = schwarz_data.symmetrize_smoothing;
    precondition_data.reverse                 = schwarz_data.reverse_smoothing;
    precondition_data.use_ras_weights         = schwarz_data.use_ras_weights;
    precondition_data.use_ras_boolean_weights = schwarz_data.use_ras_boolean_weights;

    typename smoother_type::AdditionalData smoother_data;
    smoother_data.number_of_smoothing_steps = prms.n_smoothing_steps;

    this->set_variable(prms.use_doubling_of_steps);

    /// Initialize mg matrices within MGSmootherRelaxation (smoothers have
    /// to be set in an extra step)
    this->mg_matrices = &mg_matrices;
    Base::initialize(mg_matrices, smoother_data);

    /// Initialize the smoothers within MGSmootherRelaxation
    const unsigned int mg_level_min = mg_matrices.min_level();
    const unsigned int mg_level_max = mg_matrices.max_level();
    mg_schwarz_precondition.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
    {
      const auto mf_storage_on_level = mg_matrices[level].get_matrix_free();
      const auto patch_storage =
        build_patch_storage<typename MatrixType::value_type>(level,
                                                             mf_storage_on_level,
                                                             additional_data);

      // *** setup Schwarz preconditioner
      const auto schwarz_preconditioner = std::make_shared<preconditioner_type>();
      schwarz_preconditioner->initialize(patch_storage, mg_matrices[level], precondition_data);
      mg_schwarz_precondition[level] = schwarz_preconditioner; // book-keeping

      // *** setup Schwarz smoother
      Base::smoothers[level].initialize(mg_matrices[level], schwarz_preconditioner, smoother_data);
    }
  }

  void
  initialize(const MGSmootherSchwarz<dim, MatrixType, PatchMatrixType, VectorType> & other,
             const AdditionalData & additional_data)
  {
    const auto & prms = additional_data.parameters;
    AssertThrow(prms.variant == SmootherParameter::SmootherVariant::Schwarz,
                ExcMessage("Invalid SmootherVariant."));

    const auto &                                 schwarz_data = prms.schwarz;
    typename preconditioner_type::AdditionalData precondition_data;
    precondition_data.relaxation  = schwarz_data.damping_factor;
    precondition_data.symmetrized = schwarz_data.symmetrize_smoothing;
    precondition_data.reverse     = schwarz_data.reverse_smoothing;
    typename smoother_type::AdditionalData smoother_data;
    smoother_data.number_of_smoothing_steps = prms.n_smoothing_steps;

    /// initialize the mg matrices within MGSmootherRelaxation (smoothers have
    /// to be set in an extra step)
    this->mg_matrices = other.mg_matrices;
    Base::initialize(*mg_matrices, smoother_data);

    /// check if shallow copyable
    const unsigned int mg_level_min = other.min_level();
    const unsigned int mg_level_max = other.max_level();
    typename SubdomainHandler<dim, typename MatrixType::value_type>::AdditionalData sd_handler_data;
    fill_schwarz_smoother_data<dim, typename MatrixType::value_type>(sd_handler_data, schwarz_data);
    for(unsigned level = mg_level_min; level <= mg_level_max; ++level)
    {
      sd_handler_data.level = level;
      if(prms.schwarz.userdefined_coloring)
        sd_handler_data.coloring_func = additional_data.coloring_func;
      sd_handler_data.use_tbb = additional_data.use_tbb;
      AssertThrow(other.get_preconditioner(level)->is_shallow_copyable(sd_handler_data),
                  ExcMessage("Is not shallow copyable. Check the SchwarzSmootherData settings."));
    }

    /// initialize the smoothers within MGSmootherRelaxation by shallow copies
    /// of underlying Schwarz preconditioners
    mg_schwarz_precondition.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
    {
      const auto schwarz_preconditioner_copy = std::make_shared<preconditioner_type>();
      const auto other_preconditioner        = other.get_preconditioner(level);
      schwarz_preconditioner_copy->initialize(*other_preconditioner, precondition_data);
      mg_schwarz_precondition[level] = schwarz_preconditioner_copy; // book-keeping
      Base::smoothers[level].initialize((*mg_matrices)[level],
                                        schwarz_preconditioner_copy,
                                        smoother_data);
    }
  }

  unsigned int
  min_level() const
  {
    AssertDimension(mg_schwarz_precondition.min_level(), Base::smoothers.min_level());
    return Base::smoothers.min_level();
  }

  unsigned int
  max_level() const
  {
    AssertDimension(mg_schwarz_precondition.max_level(), Base::smoothers.max_level());
    return Base::smoothers.max_level();
  }

  std::shared_ptr<const preconditioner_type>
  get_preconditioner(unsigned int level = numbers::invalid_unsigned_int) const
  {
    if(level == numbers::invalid_unsigned_int)
      level = max_level();
    Assert(min_level() <= level && level <= max_level(),
           ExcIndexRange(level, min_level(), max_level() + 1));
    return mg_schwarz_precondition[level];
  }

  const smoother_type &
  get_smoother(unsigned int level = numbers::invalid_unsigned_int) const
  {
    if(level == numbers::invalid_unsigned_int)
      level = max_level();
    Assert(min_level() <= level && level <= max_level(),
           ExcIndexRange(level, min_level(), max_level() + 1));
    return Base::smoothers[level];
  }

  std::shared_ptr<const SubdomainHandler<dim, typename MatrixType::value_type>>
  get_subdomain_handler(unsigned int level = numbers::invalid_unsigned_int) const
  {
    return get_preconditioner(level)->get_subdomain_handler();
  }

private:
  // TODO non-constness prevented by SchwarzPreconditioner::initialize()
  MGLevelObject<MatrixType> *                               mg_matrices;
  MGLevelObject<std::shared_ptr<const preconditioner_type>> mg_schwarz_precondition;
};

// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++



std::string
CoarseGridParameter::str_solver_variant(const CoarseGridParameter::SolverVariant variant)
{
  const std::string str_variant[] = {"None", "Accurate Iterative Solver", "Direct Solver"};
  return str_variant[(int)variant];
}



std::string
CoarseGridParameter::str_precondition_variant(
  const CoarseGridParameter::PreconditionVariant variant)
{
  const std::string str_variant[] = {"None", "User-defined"};
  return str_variant[(int)variant];
}



std::string
CoarseGridParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Coarse grid solver:", str_solver_variant(solver_variant));
  if(solver_variant == SolverVariant::Iterative)
  {
    oss << Util::parameter_to_fstring("Iterative solver:", iterative_solver);
    oss << Util::parameter_to_fstring("Accuracy:", accuracy);
    oss << Util::parameter_to_fstring("Coarse grid preconditioner:",
                                      str_precondition_variant(precondition_variant));
  }
  else if(solver_variant == SolverVariant::DirectSVD)
  {
    oss << Util::parameter_to_fstring("Threshold (singular values):", threshold_svd);
    oss << Util::parameter_to_fstring("Kernel size:", kernel_size);
  }
  return oss.str();
}



std::string
SmootherParameter::str_smoother_variant(const SmootherParameter::SmootherVariant variant)
{
  const std::string str_variant[] = {"None", "Gauss-Seidel", "Schwarz Smoother"};
  return str_variant[(int)variant];
}



std::string
SmootherParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Number of smoothing steps:", n_smoothing_steps);
  oss << Util::parameter_to_fstring("Doubling of smoothing steps:", use_doubling_of_steps);
  oss << Util::parameter_to_fstring("Smoother:", str_smoother_variant(variant));
  if(variant == SmootherVariant::Schwarz)
  {
    ConditionalOStream pcout(oss, true);
    schwarz.print(pcout);
  }
  return oss.str();
}



std::string
MGParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Multigrid:", cycle_variant);
  oss << Util::parameter_to_fstring("  /// Pre-smoother", "");
  oss << pre_smoother.to_string();
  oss << Util::parameter_to_fstring("  /// Post-smoother", "");
  oss << post_smoother.to_string();
  oss << Util::parameter_to_fstring("  /// Coarse grid solver", "");
  oss << coarse_grid.to_string();
  return oss.str();
}



void
MGTransferBlockPrebuilt::initialize_constraints(
  const std::vector<const MGConstrainedDoFs *> & mg_constrained_dofs)
{
  initialize_constraints_impl<true>(mg_constrained_dofs);
}



void
MGTransferBlockPrebuilt::clear()
{
  clear_impl<true>();
}



template<int dim>
void
MGTransferBlockPrebuilt::build(const std::vector<const DoFHandler<dim> *> dof_handlers)
{
  /// if there exist no transfers for each block we assume the user wants to
  /// construct an unconstrained multigrid transfer: thus, we initialize
  /// transfers by means of empty MGConstrainedDoFs objects cached in @þ
  /// mg_constrained_dofs_dummy

  if(transfer_foreach_block.empty())
  {
    mg_constrained_dofs_dummy.clear();
    mg_constrained_dofs_dummy.resize(dof_handlers.size());
    for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
      mg_constrained_dofs_dummy[b].initialize(*(dof_handlers[b]));

    std::vector<const MGConstrainedDoFs *> mg_constrained_dofs_ptr;
    std::transform(mg_constrained_dofs_dummy.cbegin(),
                   mg_constrained_dofs_dummy.cend(),
                   std::back_inserter(mg_constrained_dofs_ptr),
                   [](const auto & mgcdofs) { return &mgcdofs; });

    initialize_constraints_impl<false>(mg_constrained_dofs_ptr);
  }

  AssertDimension(dof_handlers.size(), transfer_foreach_block.size());
  for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
    transfer_foreach_block[b].build(*(dof_handlers[b]));
}



void
MGTransferBlockPrebuilt::prolongate(
  const unsigned int                                      to_level,
  LinearAlgebra::distributed::BlockVector<double> &       dst,
  const LinearAlgebra::distributed::BlockVector<double> & src) const
{
  AssertDimension(transfer_foreach_block.size(), src.n_blocks());
  AssertDimension(dst.n_blocks(), src.n_blocks()); // ???
  for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
    transfer_foreach_block[b].prolongate(to_level, dst.block(b), src.block(b));
}



void
MGTransferBlockPrebuilt::restrict_and_add(
  const unsigned int                                      from_level,
  LinearAlgebra::distributed::BlockVector<double> &       dst,
  const LinearAlgebra::distributed::BlockVector<double> & src) const
{
  AssertDimension(transfer_foreach_block.size(), src.n_blocks());
  AssertDimension(dst.n_blocks(), src.n_blocks()); // ???
  for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
    transfer_foreach_block[b].restrict_and_add(from_level, dst.block(b), src.block(b));
}



template<int dim>
void
MGTransferBlockPrebuilt::copy_to_mg(
  const DoFHandler<dim> &                                          dof_handler,
  MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & dst,
  const LinearAlgebra::distributed::BlockVector<double> &          src) const
{
  (void)dof_handler, (void)dst, (void)src;
  AssertThrow(false,
              ExcMessage("The block structure of this class is "
                         "imposed by a set of DoFHandlers "
                         "and not by a single DoFHandler "
                         "with block structure."));
}



template<int dim>
void
MGTransferBlockPrebuilt::copy_from_mg(
  const DoFHandler<dim> &                                                dof_handler,
  LinearAlgebra::distributed::BlockVector<double> &                      dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const
{
  (void)dof_handler, (void)dst, (void)src;
  AssertThrow(false,
              ExcMessage("The block structure of this class is "
                         "imposed by a set of DoFHandlers "
                         "and not by a single DoFHandler "
                         "with block structure."));
}



template<int dim>
void
MGTransferBlockPrebuilt::copy_from_mg_add(
  const DoFHandler<dim> &                                                dof_handler,
  LinearAlgebra::distributed::BlockVector<double> &                      dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const
{
  (void)dof_handler, (void)dst, (void)src;
  AssertThrow(false,
              ExcMessage("The block structure of this class is "
                         "imposed by a set of DoFHandlers "
                         "and not by a single DoFHandler "
                         "with block structure."));
}



template<int dim>
void
MGTransferBlockPrebuilt::copy_to_mg(
  const std::vector<const DoFHandler<dim> *> &                     dof_handlers,
  MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & dst,
  const LinearAlgebra::distributed::BlockVector<double> &          src) const
{
  AssertDimension(dof_handlers.size(), transfer_foreach_block.size());
  AssertDimension(dof_handlers.size(), src.n_blocks());

  for(auto l = dst.min_level(); l <= dst.max_level(); ++l)
    if(dst[l].n_blocks() != src.n_blocks())
      dst[l].reinit(src.n_blocks());

  for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
  {
    MGLevelObject<LinearAlgebra::distributed::Vector<double>> block_dst(dst.min_level(),
                                                                        dst.max_level());
    for(auto l = dst.min_level(); l <= dst.max_level(); ++l)
      std::swap(block_dst[l], dst[l].block(b));

    transfer_foreach_block[b].copy_to_mg(*(dof_handlers[b]), block_dst, src.block(b));

    for(auto l = dst.min_level(); l <= dst.max_level(); ++l)
      std::swap(block_dst[l], dst[l].block(b));
  }

  for(auto l = dst.min_level(); l <= dst.max_level(); ++l)
    dst[l].collect_sizes();
}



template<int dim>
void
MGTransferBlockPrebuilt::copy_from_mg(
  const std::vector<const DoFHandler<dim> *> &                           dof_handlers,
  LinearAlgebra::distributed::BlockVector<double> &                      dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const
{
  AssertDimension(dof_handlers.size(), transfer_foreach_block.size());
  AssertDimension(dof_handlers.size(), src[src.max_level()].n_blocks());
  for(auto l = src.min_level(); l <= src.max_level(); ++l)
    if(src[l].n_blocks() != 0U)
      AssertDimension(dof_handlers.size(), src[l].n_blocks());

  /// dst might not be initialized
  if(dst.n_blocks() != src[src.max_level()].n_blocks())
    dst.reinit(src[src.max_level()].n_blocks());

  copy_from_mg_impl<dim, false>(dof_handlers, dst, src);

  dst.collect_sizes();
}



template<int dim>
void
MGTransferBlockPrebuilt::copy_from_mg_add(
  const std::vector<const DoFHandler<dim> *> &                           dof_handlers,
  LinearAlgebra::distributed::BlockVector<double> &                      dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const
{
  AssertDimension(dof_handlers.size(), transfer_foreach_block.size());
  AssertDimension(dof_handlers.size(), src[src.max_level()].n_blocks());
  for(auto l = src.min_level(); l <= src.max_level(); ++l)
    if(src[l].n_blocks() != 0U)
      AssertDimension(dof_handlers.size(), src[l].n_blocks());

  /// we expect dst to be initialized properly if we shall to add values
  AssertDimension(dst.n_blocks(), src[src.max_level()].n_blocks());

  copy_from_mg_impl<dim, true>(dof_handlers, dst, src);
}



template<int dim, bool do_add>
void
MGTransferBlockPrebuilt::copy_from_mg_impl(
  const std::vector<const DoFHandler<dim> *> &                           dof_handlers,
  LinearAlgebra::distributed::BlockVector<double> &                      dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<double>> & src) const
{
  for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
  {
    MGLevelObject<LinearAlgebra::distributed::Vector<double>> block_src(src.min_level(),
                                                                        src.max_level());
    /// TODO check that no communication is pending for src
    for(auto l = src.min_level(); l <= src.max_level(); ++l)
      if(src[l].n_blocks() != 0U)
      {
        /// TODO use reinit() + copy_locally_owned_data()...
        block_src[l].reinit(src[l].block(b));
        block_src[l].copy_locally_owned_data_from(src[l].block(b));
        /// ...or operator=() which additionally sets ghost values?
        // block_src[l] = src[l].block(b);
      }

    if(do_add)
      transfer_foreach_block[b].copy_from_mg_add(*(dof_handlers[b]), dst.block(b), block_src);
    else
      transfer_foreach_block[b].copy_from_mg(*(dof_handlers[b]), dst.block(b), block_src);
  }
}



std::size_t
MGTransferBlockPrebuilt::memory_consumption() const
{
  const std::size_t sum = std::accumulate(transfer_foreach_block.cbegin(),
                                          transfer_foreach_block.cend(),
                                          static_cast<std::size_t>(0),
                                          [](const auto & mem, const auto & transfer) {
                                            return mem + transfer.memory_consumption();
                                          });
  return sum;
}



void
MGTransferBlockPrebuilt::set_component_to_block_map(const std::vector<unsigned int> & map_in)
{
  (void)map_in;
  AssertThrow(false,
              ExcMessage("The block structure of this class is "
                         "imposed by a set of DoFHandlers "
                         "and not by a single DoFHandler "
                         "with block structure."));
}



void
MGTransferBlockPrebuilt::print_indices(std::ostream & os) const
{
  auto b = 0U;
  for(const auto & transfer : transfer_foreach_block)
  {
    os << "block: " << b++ << std::endl;
    transfer.print_indices(os);
    os << std::endl;
  }
}



template<bool do_clear_mg_constrained_dofs_dummy>
void
MGTransferBlockPrebuilt::initialize_constraints_impl(
  const std::vector<const MGConstrainedDoFs *> & mg_constrained_dofs)
{
  clear_impl<do_clear_mg_constrained_dofs_dummy>();
  transfer_foreach_block.resize(mg_constrained_dofs.size());
  for(auto b = 0U; b < transfer_foreach_block.size(); ++b)
    transfer_foreach_block[b].initialize_constraints(*(mg_constrained_dofs[b]));
}



template<bool do_clear_mg_constrained_dofs_dummy>
void
MGTransferBlockPrebuilt::clear_impl()
{
  if(do_clear_mg_constrained_dofs_dummy)
    mg_constrained_dofs_dummy.clear();
  transfer_foreach_block.clear();
}

#endif /* MULTIGRID_H */
