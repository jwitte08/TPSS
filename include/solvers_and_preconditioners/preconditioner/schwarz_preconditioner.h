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



namespace TPSS
{
namespace internal
{
template<typename OtherVectorType>
void
copy_locally_owned_data(OtherVectorType &, const OtherVectorType &)
{
  AssertThrow(false, ExcMessage("VectorType is not supported."));
}

template<typename Number>
void
copy_locally_owned_data(LinearAlgebra::distributed::Vector<Number> &       dst,
                        const LinearAlgebra::distributed::Vector<Number> & src)
{
  dst.copy_locally_owned_data_from(src);
}

template<typename Number>
void
copy_locally_owned_data(LinearAlgebra::distributed::BlockVector<Number> &       dst,
                        const LinearAlgebra::distributed::BlockVector<Number> & src)
{
  AssertDimension(dst.n_blocks(), src.n_blocks());
  for(unsigned int b = 0; b < src.n_blocks(); ++b)
    dst.block(b).copy_locally_owned_data_from(src.block(b));
}



template<typename Number>
void
zero_out_ghosts_if_needed(Vector<Number> &)
{
  // nothing has to be done...
}

template<typename Number>
void
zero_out_ghosts_if_needed(BlockVector<Number> &)
{
  // nothing has to be done...
}

template<typename Number>
void
zero_out_ghosts_if_needed(LinearAlgebra::distributed::Vector<Number> & vec)
{
  vec.zero_out_ghosts();
}

template<typename Number>
void
zero_out_ghosts_if_needed(LinearAlgebra::distributed::BlockVector<Number> & vec)
{
  vec.zero_out_ghosts();
}

} // namespace internal

} // namespace TPSS



/**
 * TODO[description missing]
 */
template<int dim, class OperatorType, typename VectorType, typename MatrixType>
class SchwarzPreconditioner : public PreconditionerBase<VectorType>
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

  /**
   * Computes local solvers (local inverse problems) based on the domain
   * decomposition given by @p subdomain_handler_in and the finite element
   * discretization represented by @p linear_operator_in. The linear operator
   * must describe the action of the finite element matrix on the current level,
   * which is required to compute residuals (in particular for multiplicative
   * algorithms).
   */
  void
  initialize(const std::shared_ptr<const SubdomainHandler<dim, value_type>> subdomain_handler_in,
             /*TODO const?*/ OperatorType &                                 linear_operator_in,
             const AdditionalData &                                         additional_data_in);

  /**
   * Uses a shallow copy of the local solvers provided by @p other. For example,
   * if this Schwarz preconditioner applies local solvers simply in a different
   * order compared to @p other, but still processes the same set of local
   * solvers, we can save the recomputation of local solvers. Different
   * preconditioner options are passed by @p additional_data.
   */
  void
  initialize(const SchwarzPreconditioner & other, const AdditionalData & additional_data_in);

  /*
   * Does nothing. Satisfies MGSmootherPrecondition interface.
   */
  void
  initialize(const OperatorType & linear_operator, const AdditionalData & additional_data);

  /**
   * Takes ownership of @p subdomain_handler_in and setups all data members
   * which are needed by all initialize() methods.
   */
  void
  internal_initialize(
    const std::shared_ptr<const SubdomainHandler<dim, value_type>> subdomain_handler_in,
    /*TODO const?*/ OperatorType &                                 linear_operator_in,
    const AdditionalData &                                         additional_data_in);

  const AdditionalData &
  get_additional_data() const;

  std::shared_ptr<const std::vector<MatrixType>>
  get_local_solvers() const
  {
    return subdomain_to_inverse;
  }

  std::vector<TimeInfo>
  get_time_data() const;

  std::shared_ptr<const SubdomainHandler<dim, value_type>>
  get_subdomain_handler() const;

  bool
  is_shallow_copyable(
    const typename SubdomainHandler<dim, value_type>::AdditionalData & other_data) const
  {
    const auto & sdhandler   = get_subdomain_handler()->get_additional_data();
    bool         is_copyable = true;

    is_copyable &= sdhandler.level == other_data.level;
    is_copyable &= sdhandler.patch_variant == other_data.patch_variant;
    is_copyable &= sdhandler.smoother_variant == other_data.smoother_variant;
    // TODO coloring ?
    // TODO gathering ?
    is_copyable &= sdhandler.n_q_points_surrogate == other_data.n_q_points_surrogate;
    is_copyable &= sdhandler.normalize_surrogate_patch == other_data.normalize_surrogate_patch;
    is_copyable &= sdhandler.use_arc_length == other_data.use_arc_length;
    is_copyable &= sdhandler.caching_strategy == other_data.caching_strategy;
    is_copyable &= sdhandler.use_tbb == other_data.use_tbb;
    return is_copyable;
  }

  /// TODO
  types::global_dof_index
  m() const
  {
    Assert(subdomain_handler, ExcMessage("subdomain_handler not initialized"));
    AssertDimension(subdomain_handler->n_total_components(), 1U);
    const auto              dof_infos = subdomain_handler->get_dof_infos();
    types::global_dof_index n         = 0;
    for(const auto & info : dof_infos)
      n += info.vector_partitioner->size();
    return n;
  }

  /// TODO
  types::global_dof_index
  n() const
  {
    return m();
  }

  void
  vmult(VectorType & dst, const VectorType & src) const override;

  void
  Tvmult(VectorType & dst, const VectorType & src) const;

  void
  vmult_add(VectorType & dst, const VectorType & src) const;

  void
  Tvmult_add(VectorType & dst, const VectorType & src) const;

private:
  /**
   * Implementing the application of the additive Schwarz preconditioner A^{-1}_{ad}:
   *
   *    x = A^{-1}_{ad} b
   *
   * x : the result is passed to @p solution (note that @p solution is set to zero first)
   *
   * b : the vector @p rhs
   *
   * Internally this method calls apply_local_solvers(solution, rhs) for each
   * color adding up all subspace corrections. Therefore, prior to calling
   * apply_local_solvers() the first time @p solution is set to zero.
   */
  template<bool transpose = false>
  void
  additive_schwarz_operation(VectorType & solution, const VectorType & rhs) const;

  /**
   * Computes the action: u_new = u_prev + omega * sum_{j \in J(color)} R_j^T  A_j^{-1} R_j r
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
   *
   * omega    : damping
   */
  void
  apply_local_solvers(VectorType &       solution,
                      const VectorType & residual,
                      const unsigned int color) const;

  /**
   * We do not control the initialization of solution and residual vectors
   * passed to vmult() and methods the like. In case of MPI parallel vectors the
   * partitioning of vectors passed to @p apply_local_solvers might differ from
   * what we require for (vertex) patches. Therefore, we have to set up ghosted
   * vectors if the global partitioning of @p solution_in and @p residual_in
   * does not coincide with the one stored in @p subdomain_handler.
   */
  template<typename OtherVectorType = VectorType>
  std::pair<OtherVectorType *, const OtherVectorType *>
  preprocess_solution_and_residual_distributed(OtherVectorType &       solution_in,
                                               const OtherVectorType & residual_in) const;

  /**
   * This generic case just throws an exception.
   */
  template<typename OtherVectorType = VectorType>
  std::pair<OtherVectorType *, const OtherVectorType *>
  preprocess_solution_and_residual(OtherVectorType &       solution_in,
                                   const OtherVectorType & residual_in) const;

  /**
   * See the description of preprocess_solution_and_residual_distributed().
   */
  std::pair<LinearAlgebra::distributed::Vector<value_type> *,
            const LinearAlgebra::distributed::Vector<value_type> *>
  preprocess_solution_and_residual(
    LinearAlgebra::distributed::Vector<value_type> &       solution_in,
    const LinearAlgebra::distributed::Vector<value_type> & residual_in) const;

  /**
   * Same as above for parallel block vectors.
   */
  std::pair<LinearAlgebra::distributed::BlockVector<value_type> *,
            const LinearAlgebra::distributed::BlockVector<value_type> *>
  preprocess_solution_and_residual(
    LinearAlgebra::distributed::BlockVector<value_type> &       solution_in,
    const LinearAlgebra::distributed::BlockVector<value_type> & residual_in) const;

  /**
   * Nothing has to be done for serial vectors.
   */
  std::pair<Vector<value_type> *, const Vector<value_type> *>
  preprocess_solution_and_residual(Vector<value_type> &       solution_in,
                                   const Vector<value_type> & residual_in) const;

  /**
   * Same as above for serial block vectors.
   */
  std::pair<BlockVector<value_type> *, const BlockVector<value_type> *>
  preprocess_solution_and_residual(BlockVector<value_type> &       solution_in,
                                   const BlockVector<value_type> & residual_in) const;

  /**
   * If @p solution_src (the solution vector after applying local solvers)
   * coincides with the internal ghosted vector @p solution_ghosted we have to
   * copy all locally owned entries to @p solution_dst (the solution vector
   * returned by this preconditioner).
   */
  template<typename OtherVectorType = VectorType>
  void
  postprocess_solution_distributed(OtherVectorType &       solution_dst,
                                   const OtherVectorType & solution_src) const;

  /**
   * Generic case is just there to throw an exception for unsupported vector
   * types.
   */
  template<typename OtherVectorType = VectorType>
  void
  postprocess_solution(OtherVectorType & solution_dst, const OtherVectorType & solution_src) const;

  /**
   * See the description of postprocess_solution_distributed().
   */
  void
  postprocess_solution(LinearAlgebra::distributed::Vector<value_type> &,
                       const LinearAlgebra::distributed::Vector<value_type> &) const;

  /**
   * Same as above for parallel block vectors.
   */
  void
  postprocess_solution(LinearAlgebra::distributed::BlockVector<value_type> &,
                       const LinearAlgebra::distributed::BlockVector<value_type> &) const;

  /**
   * Nothing has to be done for serial vectors.
   */
  void
  postprocess_solution(Vector<value_type> &, const Vector<value_type> &) const;

  /**
   * Same as above for serial block vectors.
   */
  void
  postprocess_solution(BlockVector<value_type> &, const BlockVector<value_type> &) const;

  void
  compute_inverses();

  void
  compute_ras_weights();

  /**
   * The sequence of colors for the smoothing step. Colors are defined by the PartitionData of @p
   * subdomain_handler and the ordering is determined in terms of the AdditionalData.
   */
  std::vector<unsigned int>
  get_color_sequence(const bool transpose) const;

  /**
   * Does nothing. See overloaded methods...
   */
  template<typename OtherVectorType>
  void
  initialize_ghosted_vector_if_needed(OtherVectorType &) const;

  /**
   * Initializes MPI vectors (here LinearAlgebra::distributed::Vector) with
   * respect to the MPI partitioners stored in SubdomainHandler. This method
   * should only be used for 'private' vectors such that local compatibility
   * checks are sufficient.
   */
  void
  initialize_ghosted_vector_if_needed(LinearAlgebra::distributed::Vector<value_type> & vec) const;

  /**
   * Same as above, but for block vectors (here
   * LinearAlgebra::distributed::BlockVector).
   */
  void
  initialize_ghosted_vector_if_needed(
    LinearAlgebra::distributed::BlockVector<value_type> & vec) const;

  /**
   * In case of serial vectors @p vec is simply resized with respect to the MPI
   * partitioners stored in SubdomainHandler.
   */
  void
  initialize_ghosted_vector_if_needed(Vector<value_type> & vec) const;

  /**
   * Same as above but for block vector type.
   */
  void
  initialize_ghosted_vector_if_needed(BlockVector<value_type> & vec) const;

  /**
   * Does nothing. See specializations below.
   */
  template<typename OtherVectorType>
  bool
  is_globally_compatible(const OtherVectorType &) const;

  /**
   * Initializes MPI vectors, here LinearAlgebra::distributed::Vector with
   * respect to the MPI partitioners stored in SubdomainHandler. This method is
   * applied to external vectors, for instance passed to vmult() and methods the
   * like, and, thus, checks if the global partitioning coincides with the one
   * required by SubdomainHandler.
   */
  bool
  is_globally_compatible(const LinearAlgebra::distributed::Vector<value_type> & vec) const;

  /**
   * Same as above, here for LinearAlgebra::distributed::BlockVector.
   */
  bool
  is_globally_compatible(const LinearAlgebra::distributed::BlockVector<value_type> & vec) const;

  /**
   * Implementing the application of the multiplicative Schwarz preconditioner A^{-1}_{mu}:
   *
   *    x = A^{-1}_{mu} b
   *
   * x : the result is passed to @p solution (note that @p solution is set to zero first)
   *
   * b : the vector @p rhs
   *
   * Internally this method sets @p solution to zero and calls
   * apply_local_solvers(solution, rhs) for the first color. For the
   * intermediate solution x^{(c)} obtained from the previous call
   * apply_local_solvers() we recursively compute the residual
   *
   *   r^{(c)} = b - A * x^{(c)}
   *
   * followed by the call
   *
   *   apply_local_solvers(x^{(c)}, r^{(c)})
   *
   * for each remaining color c.  Details for the recursive algorithm can be
   * found in Toselli, Widlund page 52.
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
   * I forward problem:
   *
   * The linear operator should perform the global matrix-vector product A * u via methods @p vmult()
   * or @p vmult_add() (and maybe in the future also the transpose A^T * u via @p Tvmult() and @p
   * Tvmult_add())
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

  unsigned int level = numbers::invalid_unsigned_int;

  TPSS::PatchVariant patch_variant = TPSS::PatchVariant::invalid;

  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;

  AdditionalData additional_data;

  mutable std::vector<TimeInfo> time_data;

  /*mutable*/ std::shared_ptr<VectorType> ras_weights;
};



template<int dim, class OperatorType, typename VectorType, typename MatrixType>
struct SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::AdditionalData
{
  double       relaxation              = 1.;
  bool         reverse                 = false;
  bool         symmetrized             = false;
  bool         use_ras_weights         = false;
  bool         use_ras_boolean_weights = false;
  unsigned int n_active_blocks         = numbers::invalid_unsigned_int;

  bool
  operator==(const AdditionalData & other_data) const
  {
    bool is_equal = true;
    is_equal &= relaxation == other_data.relaxation;
    is_equal &= reverse == other_data.reverse;
    is_equal &= symmetrized == other_data.symmetrized;
    is_equal &= use_ras_weights == other_data.use_ras_weights;
    is_equal &= use_ras_boolean_weights == other_data.use_ras_boolean_weights;
    is_equal &= n_active_blocks == other_data.n_active_blocks;
    return is_equal;
  }
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
