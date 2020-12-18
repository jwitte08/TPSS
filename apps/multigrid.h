
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


#include "solver.h"
#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"
#include "solvers_and_preconditioners/smoother/schwarz_smoother.h"
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
    None
  };

  static std::string
  str_solver_variant(const SolverVariant variant);

  SolverVariant       solver_variant       = SolverVariant::Iterative;
  PreconditionVariant precondition_variant = PreconditionVariant::None;
  std::string         iterative_solver     = "none"; // see SolverSelector
  double              accuracy             = 1.e-12;
  double              threshold_svd        = 0.;

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



/**
 * A wrapper class extending MGCoarseGridSVD to more vector types than 'Vector'
 * (the vector type that is internally used by LAPACKFullMatrix). In particular,
 * we have MPI-capable vector types in mind like
 * 'LinearAlgebra::distributed::Vector'. However, vector types are only compatible
 * when the program runs with one MPI process.
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
    solver_control = std::make_shared<SolverControl>();
    solver_control->set_tolerance(prms.accuracy);
    solver_control->log_history(false);
    solver_control->log_result(false);
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
    const auto solver_control = Base::set_solver_control(prms);
    // solver_control->set_max_steps(coarse_matrix.m());

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
    // const auto & coarse_table       = Tensors::matrix_to_table(coarse_matrix);
    // const auto & coarse_full_matrix = table_to_fullmatrix(coarse_table);
    using coarse_grid_solver_type = MGCoarseGridSVDSerial<value_type, VectorType>;
    const auto solver             = std::make_shared<coarse_grid_solver_type>();
    solver->initialize(coarse_matrix, prms.threshold_svd);
    Base::coarse_grid_solver = solver;
  }

  void
  initialize_iterative(const MatrixType & coarse_matrix, const CoarseGridParameter & prms)
  {
    const auto solver_control = Base::set_solver_control(prms);
    solver_control->set_max_steps(coarse_matrix.m());

    iterative_solver.set_control(*solver_control);
    iterative_solver.select(prms.iterative_solver);

    switch(prms.precondition_variant)
    {
      case CoarseGridParameter::PreconditionVariant::None:
      {
        using coarse_grid_solver_type = MGCoarseGridIterativeSolver<VectorType,
                                                                    SolverSelector<VectorType>,
                                                                    MatrixType,
                                                                    PreconditionIdentity>;
        Base::coarse_grid_solver      = std::make_shared<coarse_grid_solver_type>(iterative_solver,
                                                                             coarse_matrix,
                                                                             preconditioner_id);
        return;
      }
      default:
        AssertThrow(false, ExcMessage("PreconditionVariant isn't supported."));
    }
  }

  void
  initialize(const MatrixType & coarse_matrix, const CoarseGridParameter & prms)
  {
    AssertDimension(coarse_matrix.m(), coarse_matrix.n());
    const auto solver_control = Base::set_solver_control(prms);
    solver_control->set_max_steps(coarse_matrix.m());

    switch(prms.solver_variant)
    {
      case CoarseGridParameter::SolverVariant::Iterative:
        initialize_iterative(coarse_matrix, prms);
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
  const std::string str_variant[] = {"None", "Accurate Iterative Solver", "Full Inverse (SVD)"};
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
  oss << Util::parameter_to_fstring("/// Pre-smoother", "");
  oss << pre_smoother.to_string();
  oss << Util::parameter_to_fstring("/// Post-smoother", "");
  oss << post_smoother.to_string();
  oss << Util::parameter_to_fstring("/// Coarse grid solver", "");
  oss << coarse_grid.to_string();
  return oss.str();
}

#endif /* MULTIGRID_H */
