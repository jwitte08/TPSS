#ifndef SCHWARZ_PRECONDITIONER_H
#define SCHWARZ_PRECONDITIONER_H

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/tensor_product_matrix.h>

#include <functional>

#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"
#include "solvers_and_preconditioners/TPSS/patch_transfer.h"
#include "solvers_and_preconditioners/TPSS/subdomain_handler.h"
#include "solvers_and_preconditioners/preconditioner/preconditioner_base.h"

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
class SchwarzPreconditioner : public MyPreconditionerBase<VectorType>, public Subscriptor
{
public:
  using value_type = typename ExtractScalarType<typename MatrixType::value_type>::type;

  struct AdditionalData;

  SchwarzPreconditioner() = default;

  SchwarzPreconditioner(const SchwarzPreconditioner &) = delete;

  ~SchwarzPreconditioner() = default;

  SchwarzPreconditioner &
  operator=(const SchwarzPreconditioner &) = delete;

  void
  clear();

  /*
   * Computes the local solvers (local inverse problems) based on the domain decomposition given by
   * @p subdomain_handler and the finite element discretization represented by @p linear
   * operator. In addition, the linear operator must describe the (global) forward problem required
   * to compute residuals.
   */
  void
  initialize(const std::shared_ptr<const SubdomainHandler<dim, value_type>> subdomain_handler,
             /*TODO const?*/ OperatorType &                                 linear_operator,
             const AdditionalData &                                         additional_data);

  /*
   * Performs a shallow copy of the local solvers provided by @p schwarz_preconditioner_in. For
   * example, if another Schwarz preconditioner with a different color sequence, e.g. due to
   * symmetry reasons, is needed but the set of local solvers is the same. This type of
   * modifications are handled by @p additional_data.
   */
  void
  initialize(const SchwarzPreconditioner & schwarz_preconditioner_in,
             const AdditionalData &        additional_data);

  /*
   * Does nothing. Satisfies MGSmootherPrecondition interface.
   */
  void
  initialize(const OperatorType & linear_operator, const AdditionalData & additional_data);

  const AdditionalData &
  get_additional_data() const;

  std::vector<TimeInfo>
  get_time_data() const;

  std::shared_ptr<const SubdomainHandler<dim, value_type>>
  get_subdomain_handler() const;

  void
  vmult(VectorType & dst, const VectorType & src) const override;

  void
  Tvmult(VectorType & dst, const VectorType & src) const;

  void
  vmult_add(VectorType & dst, const VectorType & src) const;

  void
  Tvmult_add(VectorType & dst, const VectorType & src) const;

  /**
   * Satisfies PreconditionerBase interface.
   */
  void
  update(LinearOperatorBase const * matrix_operator) override;

private:
  /**
   * Does one apply_local_solvers() on the first color regarding the right hand side @p rhs
   * and adds the solution into @p solution. The first color is 0 if
   * transpose/AdditionalData::reverse is false or n_colors-1 if transpose/AdditionalData::reverse
   * is true.
   *
   *  Since the previous solution is 0 our residual equals the right hand side @p dst.
   *  Consequently, the compute-residual step can be omitted and we apply all subproblem inverses of
   *  color 0 directly to the right hand side and store the new solution @p solution.  (See Toselli,
   *  Widlund page 52 for details on multiplicative preconditioner!)
   */
  template<bool transpose = false>
  void
  additive_schwarz_operation(VectorType & solution, const VectorType & rhs) const;

  /**
   * Computes the action: u_new = u_prev + sum_{j \in J(color)} R_j^T  A_j^{-1} R_j r
   *
   * u_new    : new solution vector after one colored smoothing step of r, which is returned by
   *            reference @p solution
   *
   * u_prev   : previous solution vector used as initial guess of smoothing step, which is passed by
   *            reference @p solution
   *
   * J(color) : index set containing all subdomains with color @p color
   *
   * R_j      : transfer from global unknowns to local unknowns associated to subspace j
   *            (restriction)
   *
   * A_j^{-1} : local solver of subproblem j which is stored in @p subdomain_to_inverse
   *
   * R_j^T    : transfer from local unknowns associated to subspace j to global unknowns
   *            (prolongation)
   *
   * r        : residual vector b - A u_prev. since u_prev might be zero the residual @p residual is not
   *            computed within this function.
   */
  void
  apply_local_solvers(VectorType &       solution,
                      const VectorType & residual,
                      const unsigned int color) const;

  void
  compute_inverses();

  template<typename OtherVectorType>
  void
  copy_locally_owned_data(OtherVectorType &, const OtherVectorType &) const
  {
    AssertThrow(false, ExcMessage("VectorType not supported."));
  }

  void
  copy_locally_owned_data(LinearAlgebra::distributed::Vector<value_type> &       dst,
                          const LinearAlgebra::distributed::Vector<value_type> & src) const
  {
    dst.copy_locally_owned_data_from(src);
  }

  void
  copy_locally_owned_data(LinearAlgebra::distributed::BlockVector<value_type> &       dst,
                          const LinearAlgebra::distributed::BlockVector<value_type> & src) const
  {
    const unsigned int n_components = subdomain_handler->n_components();
    for(unsigned int b = 0; b < n_components; ++b)
      dst.block(b).copy_locally_owned_data_from(src.block(b));
  }

  /**
   * The sequence of colors for the smoothing step. Colors are defined by the PartitionData of @p
   * subdomain_handler and the ordering is determined in terms of the AdditionalData.
   */
  std::vector<unsigned int>
  get_color_sequence(const bool transpose) const;

  template<typename OtherVectorType>
  void
  initialize_ghost(OtherVectorType &) const
  {
    AssertThrow(false, ExcMessage("VectorType not supported."));
  }

  void
  initialize_ghost(LinearAlgebra::distributed::Vector<value_type> & vec) const
  {
    const auto partitioner = subdomain_handler->get_vector_partitioner();
    vec.reinit(partitioner);
  }

  void
  initialize_ghost(LinearAlgebra::distributed::BlockVector<value_type> & vec) const
  {
    const auto partitioner = subdomain_handler->get_vector_partitioner();
    /// initialize block structure which has to be finalized by collect_sizes()
    const unsigned int n_components = subdomain_handler->n_components();
    vec.reinit(n_components, /*block_size*/ 0, /*omit_zeroing_entries*/ false);
    /// initialize mpi structure for each block
    for(unsigned int b = 0; b < n_components; ++b)
      vec.block(b).reinit(partitioner);
    /// update the information on the block sizes
    vec.collect_sizes();
  }

  template<typename OtherVectorType>
  bool
  is_globally_compatible(
    const OtherVectorType &,
    const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> &) const
  {
    AssertThrow(false, ExcMessage("VectorType not supported."));
    return false;
  }

  bool
  is_globally_compatible(
    const LinearAlgebra::distributed::Vector<value_type> &                  vec1,
    const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> & partitioners) const
  {
    AssertThrow(!partitioners.empty(), ExcMessage("There are no partitioners."));
    const bool is_compatible = vec1.partitioners_are_globally_compatible(*partitioners.front());
    return is_compatible;
  }

  bool
  is_globally_compatible(
    const LinearAlgebra::distributed::BlockVector<value_type> &             vec1,
    const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> & partitioners) const
  {
    bool is_compatible = true;
    AssertThrow(partitioners.size() == subdomain_handler->n_components(),
                ExcMessage("There are no partitioners."));
    for(unsigned int b = 0; b < subdomain_handler->n_components(); ++b)
    {
      is_compatible =
        is_compatible && vec1.block(b).partitioners_are_globally_compatible(*(partitioners[b]));
    }
    return is_compatible;
  }

  /**
   *  Note that the first step in the multiplicative Schwarz smoothing algorithm is same we have to
   *  do for the additive algorithm.
   *
   *  The remaining steps of the multiplicative algorithm consist of computing the residual
   *  regarding the previous solution and applying all subproblem inverses of a given color to the
   *  residual, afterwards. The latter step entails as well the update of the new solution
   *  (apply_local_solvers()).  The right hand side is given by @p rhs and the action of
   *  all successive operations regarding remaining colors is stored in @p solution.  (See Toselli,
   *  Widlund page 52 for details on multiplicative preconditioner!)
   */
  template<bool transpose = false>
  void
  multiplicative_schwarz_operation(VectorType & solution, const VectorType & rhs) const;

  /**
   * Underlying infrastructure storing data with respect to the domain decomposition, partitioning
   * of patches, linking to MatrixFree infrastructure, etc ...
   */
  std::shared_ptr<const SubdomainHandler<dim, value_type>> subdomain_handler;

  /**
   * The transfer between global and patch-local unknowns defining the restriction and prolongation
   * for each subdomain.
   */
  std::shared_ptr<typename OperatorType::transfer_type> transfer;

  /**
   * I forward problem:
   *
   * The linear operator should perform the global matrix-vector product A * u via methods @p vmult()
   * or @p vmult_add() (and maybe in the future also the transpose A^T * u via @p Tvmult() and @p
   * Tvmult_add()
   *
   * II inverse problem:
   *
   * Second, the linear operator is responsible of computing local solvers via the method
   *  template<typename MatrixType, typename OperatorType>
   *  void
   *  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & data,
   *                             std::vector<MatrixType> &             inverses,
   *                             const OperatorType &                  linear_operator,
   *                             const std::pair<unsigned int, unsigned int> subdomain_range) const
   */
  OperatorType * linear_operator = nullptr;

  /**
   * Local solvers stored as (inverse) matrices. The shared pointer allows shallow copies.
   */
  std::shared_ptr<std::vector<MatrixType>> subdomain_to_inverse;

  mutable std::shared_ptr<VectorType> solution_ghosted;

  mutable std::shared_ptr<VectorType> residual_ghosted;

  unsigned int level = -1;

  TPSS::PatchVariant patch_variant = TPSS::PatchVariant::invalid;

  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;

  AdditionalData additional_data;

  mutable std::vector<TimeInfo> time_data;
};



template<int dim, class OperatorType, typename VectorType, typename MatrixType>
struct SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::AdditionalData
{
  double relaxation       = 1.;
  double local_relaxation = 1.;
  bool   reverse          = false;
  bool   symmetrized      = false;
};



/*********************************** inline functions ***********************************/

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
inline const typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::
  AdditionalData &
  SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::get_additional_data() const
{
  return additional_data;
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
inline std::vector<unsigned int>
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::get_color_sequence(
  const bool transpose) const
{
  const auto & partition_data = subdomain_handler->get_partition_data();
  const auto   n_colors       = partition_data.n_colors();

  // *** traverse colors as given by PartitionData
  std::vector<unsigned int> color_sequence;
  for(unsigned int color = 0; color < n_colors; ++color)
    color_sequence.push_back(color); // 0, 1, ..., n_colors-1

  // *** revert the color sequence
  const bool revert_colors = transpose || additional_data.reverse;
  if(revert_colors)
    std::reverse(color_sequence.begin(), color_sequence.end()); // n_colors-1, ..., 1, 0

  // *** additionally traverse colors backwards (avoiding duplication of adjacent colors)
  AssertDimension(color_sequence.size(), n_colors);
  AssertThrow(!color_sequence.empty(), ExcMessage("There are no colors."));
  if(additional_data.symmetrized)
  {
    const auto temp = color_sequence;
    auto       c    = ++temp.rbegin(); // skip last color (avoids duplicate!)
    for(; c != temp.rend(); ++c)
      color_sequence.push_back(*c);
    AssertDimension(color_sequence.size(), 2 * n_colors - 1);
  }

  // // DEBUG
  // std::cout << "color_sequence: ";
  // for (const auto c : color_sequence)
  //   std::cout << c << "  ";
  // std::cout << std::endl;
  return color_sequence;
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
inline std::shared_ptr<const SubdomainHandler<
  dim,
  typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type>>
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::get_subdomain_handler() const
{
  return subdomain_handler;
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
inline std::vector<TimeInfo>
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::get_time_data() const
{
  return time_data;
}

#include "schwarz_preconditioner.templates.h"

#endif
