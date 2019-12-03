/*
 * SchwarzSmoother.h
 *
 *  Created on: 2018 - August - 09
 *      Author: witte
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZSMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZSMOOTHER_H_


#include "solvers_and_preconditioners/preconditioner/schwarz_preconditioner.h"
#include "solvers_and_preconditioners/smoother/smoother_richardson.h"

template<int dim, typename OperatorType, typename PreconditionerType, typename VectorType>
class SchwarzSmoother : public SmootherRichardson<OperatorType, VectorType>
{
public:
  using Base                = SmootherRichardson<OperatorType, VectorType>;
  using AdditionalData      = typename Base::AdditionalData;
  using value_type          = typename Base::value_type;
  using preconditioner_type = PreconditionerType;

  SchwarzSmoother() = default;

  ~SchwarzSmoother() = default;

  SchwarzSmoother(SchwarzSmoother const &) = delete;

  SchwarzSmoother &
  operator=(SchwarzSmoother const &) = delete;

  void
  initialize(OperatorType &                            operator_in,
             const std::shared_ptr<PreconditionerType> preconditioner_in,
             const AdditionalData &                    additional_data_in)
  {
    // take ownership of preconditioner
    preconditioner_owned = preconditioner_in;

    // initialize SmootherRichardson
    Base::initialize(operator_in, *preconditioner_owned, additional_data_in);
  }

  /*
   * This method satisfies the mg::SmootherRelaxation::initialize interface
   */
  void
  initialize(const OperatorType &, const AdditionalData &)
  {
    // Does nothing ... !
  }

  const PreconditionerType &
  get_preconditioner() const
  {
    Assert(preconditioner_owned, ExcMessage("Schwarz preconditioner has not been initialized."));
    return *preconditioner_owned;
  }

private:
  std::shared_ptr<PreconditionerType> preconditioner_owned;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SCHWARZSMOOTHER_H_ */
