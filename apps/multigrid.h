
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

#include "solvers_and_preconditioners/TPSS/schwarz_smoother_data.h"
#include "utilities.h"

using namespace dealii;

struct SmootherParameter
{
  enum class SmootherVariant // TODO
  {
    Schwarz
  };
  static std::string
  str_smoother_variant(const SmootherVariant variant);

  bool                               compressed = false;
  SmootherParameter::SmootherVariant variant    = SmootherParameter::SmootherVariant::Schwarz;
  int                                n_smoothing_steps = 1;
  SchwarzSmootherData                schwarz;

  std::string
  to_string() const;
};



struct CoarseGridParameter
{
  enum class SolverVariant
  {
    None,
    IterativeAcc
  };
  static std::string
  str_solver_variant(const SolverVariant variant);

  SolverVariant solver_variant   = SolverVariant::None;
  double        accuracy         = 1.e-4;
  std::string   iterative_solver = "none"; // see SolverSelector

  std::string
  to_string() const;
};



struct MGParameter
{
  std::string         cycle_variant = "V-cycle"; // TODO use enum
  SmootherParameter   pre_smoother;
  SmootherParameter   post_smoother;
  CoarseGridParameter coarse_grid;
  int                 coarse_level = 0;

  std::string
  to_string() const;
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

    UserColoring      coloring_func;
    SmootherParameter parameters;
  };

  template<typename OtherNumber>
  std::shared_ptr<const SubdomainHandler<dim, OtherNumber>>
  build_patch_storage(const unsigned                                            level,
                      const std::shared_ptr<const MatrixFree<dim, OtherNumber>> mf_storage,
                      const AdditionalData &                                    additional_data)
  {
    const auto &                                                prms = additional_data.parameters;
    typename SubdomainHandler<dim, OtherNumber>::AdditionalData sdhandler_data;
    sdhandler_data.level = level;
    // TODO
    sdhandler_data.compressed       = prms.compressed;
    sdhandler_data.patch_variant    = prms.schwarz.patch_variant;
    sdhandler_data.smoother_variant = prms.schwarz.smoother_variant;
    sdhandler_data.print_details    = prms.schwarz.print_details;
    // TODO
    if(prms.schwarz.manual_coloring)
    {
      sdhandler_data.coloring_func = additional_data.coloring_func;
    }
    sdhandler_data.n_q_points_surrogate      = prms.schwarz.n_q_points_surrogate;
    sdhandler_data.normalize_surrogate_patch = prms.schwarz.normalize_surrogate_patch;
    sdhandler_data.use_arc_length            = prms.schwarz.use_arc_length;
    const auto patch_storage = std::make_shared<SubdomainHandler<dim, OtherNumber>>();
    patch_storage->reinit(mf_storage, sdhandler_data);
    return patch_storage;
  }

  // TODO const 'mg_matrices' prevented by current initialize of SchwarzPreconditioner
  void
  initialize(MGLevelObject<MatrixType> & mg_matrices, const AdditionalData & additional_data)
  {
    using MatrixFreeType = typename std::decay<decltype(
      *(mg_matrices[mg_matrices.min_level()].get_matrix_free()))>::type;
    using value_type     = typename MatrixFreeType::value_type;

    const auto & prms = additional_data.parameters;
    AssertThrow(prms.variant == SmootherParameter::SmootherVariant::Schwarz,
                ExcMessage("Invalid SmootherVariant."));

    const auto &                                 schwarz_data = prms.schwarz;
    typename preconditioner_type::AdditionalData precondition_data;
    precondition_data.relaxation       = schwarz_data.damping_factor;
    precondition_data.local_relaxation = schwarz_data.local_damping_factor;
    precondition_data.symmetrized      = schwarz_data.symmetrize_smoothing;
    typename smoother_type::AdditionalData smoother_data;
    smoother_data.number_of_smoothing_steps = prms.n_smoothing_steps;

    /// initialize the mg matrices within MGSmootherRelaxation (smoothers have
    /// to be set in an extra step)
    Base::initialize(mg_matrices, smoother_data);

    /// initialize the smoothers within MGSmootherRelaxation
    const unsigned int mg_level_min = mg_matrices.min_level();
    const unsigned int mg_level_max = mg_matrices.max_level();
    mg_schwarz_precondition.resize(mg_level_min, mg_level_max);
    for(unsigned int level = mg_level_min; level <= mg_level_max; ++level)
    {
      const auto mf_storage_on_level = mg_matrices[level].get_matrix_free();
      const auto patch_storage =
        build_patch_storage<value_type>(level, mf_storage_on_level, additional_data);

      // *** setup Schwarz preconditioner
      const auto schwarz_preconditioner = std::make_shared<preconditioner_type>();
      schwarz_preconditioner->initialize(patch_storage, mg_matrices[level], precondition_data);
      mg_schwarz_precondition[level] = schwarz_preconditioner; // book-keeping

      // *** setup Schwarz smoother
      Base::smoothers[level].initialize(mg_matrices[level], schwarz_preconditioner, smoother_data);
    }
  }

private:
  MGLevelObject<std::shared_ptr<const preconditioner_type>> mg_schwarz_precondition;
};

// +++++++++++++++++++++++++++++++++++ DEFINITIONS +++++++++++++++++++++++++++++++++++

std::string
CoarseGridParameter::str_solver_variant(const CoarseGridParameter::SolverVariant variant)
{
  const std::string str_variant[] = {"None", "Accurate Iterative Solver"};
  return str_variant[(int)variant];
}

std::string
CoarseGridParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Coarse grid solver:", str_solver_variant(solver_variant));
  if(solver_variant == SolverVariant::IterativeAcc)
  {
    oss << Util::parameter_to_fstring("Iterative solver:", iterative_solver);
    oss << Util::parameter_to_fstring("Accuracy:", accuracy);
  }
  return oss.str();
}

std::string
SmootherParameter::str_smoother_variant(const SmootherParameter::SmootherVariant variant)
{
  const std::string str_variant[] = {"Schwarz Smoother"};
  return str_variant[(int)variant];
}

std::string
SmootherParameter::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Number of smoothing steps:", n_smoothing_steps);
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
