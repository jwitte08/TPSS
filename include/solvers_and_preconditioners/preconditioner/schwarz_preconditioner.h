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

  ~SchwarzPreconditioner(){};

  SchwarzPreconditioner &
  operator=(const SchwarzPreconditioner &) = delete;

  void
  clear();

  template<typename OtherVectorType>
  void
  initialize_ghost_vector(const OtherVectorType &)
  {
    AssertThrow(false, ExcMessage("VectorType not supported."));
  }

  // TODO support more VectorTypes
  void
  initialize_ghost_vector(const LinearAlgebra::distributed::Vector<value_type> & vec) const
  {
    const auto ghost_partitioner = subdomain_handler->get_vector_partitioner();
    if(vec.get_partitioner().get() == ghost_partitioner.get())
      return;
    LinearAlgebra::distributed::Vector<value_type> copy_vec(vec);
    const_cast<LinearAlgebra::distributed::Vector<value_type> &>(vec).reinit(ghost_partitioner);
    const_cast<LinearAlgebra::distributed::Vector<value_type> &>(vec).copy_locally_owned_data_from(
      copy_vec);
  }

  void
  initialize(const OperatorType &, const AdditionalData &)
  {
    // *** does nothing
    // AssertThrow(false,
    //             ExcMessage("Dummy function fulfilling the MGSmootherPrecondition interface."));
  }

  /*
   * Computes the local solvers based on the domain decomposition in the SubdomainHandler and the
   * typical operators gathered in the OperatorType.
   */
  void
  initialize(const std::shared_ptr<const SubdomainHandler<dim, value_type>> sd_handler,
             // TODO const?
             OperatorType &         op,
             const AdditionalData & addit_data);

  /*
   * Performs a shallow copy of the local solvers provided by schwarz_preconditioner_in with
   * (possibly) new settings given by the AdditionalData.
   */
  void
  initialize(const SchwarzPreconditioner & schwarz_preconditioner_in,
             const AdditionalData &        additional_data);

  std::shared_ptr<const SubdomainHandler<dim, value_type>>
  get_subdomain_handler() const;

  const AdditionalData &
  get_additional_data() const
  {
    return additional_data;
  }

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

  std::vector<TimeInfo>
  get_time_data() const;

private:
  void
  compute_inverses();

  /**
   * Computes the action: u_new = u_prev + sum_{j \in J(color)} R_j^T  A_j^{-1} R_j r
   * where u_prev, the previous solution vector, is provided as input by @p solution,
   *       R_j, the restriction operator to subdomain j, is provided by @p transfer,
   *       A_j^{-1}, the local inverse of subproblem j, is stored in subdomain_to_inverses,
   *       R_j^T, the prolongation operator to subdomain j, is provided by @p transfer,
   *       J(color), the index set of all subdomains with color @color, provided by
   *                 subdomain_partition_data stored in SubdomainHandler,
   *       r, the residual b - A u_prev, is provided by @p residual,
   *       u_new, the new solution after applying the operation described above, is stored,
   *              as output in @solution at the end
   */
  // template<typename VectorType>
  void
  apply_subdomain_inverses_add(VectorType &       solution,
                               const VectorType & residual,
                               const unsigned int color) const;

  /**
   * Does one apply_subdomain_inverses_add() on the first color
   * regarding the right hand side @p rhs and adds the solution into
   * @p solution. The first color is 0 if
   * transpose/AdditionalData::reverse is false or n_colors-1 if
   * transpose/AdditionalData::reverse is true.
   *
   *  Since the previous solution is 0 our residual equals the right hand side @p dst.
   *  Consequently, the compute-residual step can be omitted and we apply all subproblem
   *  inverses of color 0 directly to the right hand side and store the new solution @p solution.
   *  (See Toselli, Widlund page 52 for details on multiplicative preconditioner!)
   */
  template</*typename VectorType,*/ bool transpose = false>
  void
  additive_schwarz_operation(VectorType & solution, const VectorType & rhs) const;

  /**
   *  Note that the first step in the multiplicative Schwarz smoothing algorithm is same
   *  we have to do for the additive algorithm.
   *
   *  The remaining steps of the multiplicative algorithm consist of computing the
   *  residual regarding the previous solution and applying all subproblem inverses
   *  of a given color to the residual, afterwards. The latter step entails as well
   *  the update of the new solution (apply_subdomain_inverses_add()).
   *  The right hand side is given by @p rhs and the action of all successive
   *  operations regarding remaining colors is stored in @p solution.
   *  (See Toselli, Widlund page 52 for details on multiplicative preconditioner!)
   */
  template</*typename VectorType,*/ bool transpose = false>
  void
  multiplicative_schwarz_operation(VectorType & solution, const VectorType & rhs) const;


  /**
   * underlying input defining the differential problem
   * we demand integration interface from the OperatorType
   */
  std::shared_ptr<const SubdomainHandler<dim, value_type>> subdomain_handler;
  std::shared_ptr<typename OperatorType::transfer_type>    transfer;
  // TODO const?
  OperatorType *                           differential_operator = nullptr;
  std::shared_ptr<std::vector<MatrixType>> inverses_owned        = nullptr;
  const std::vector<MatrixType> *          subdomain_to_inverse;

  /**
   * additional information
   */
  unsigned int          level            = -1;
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::invalid;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;
  AdditionalData        additional_data;

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
