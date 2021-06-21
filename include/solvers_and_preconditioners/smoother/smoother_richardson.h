/*
 * SmootherRichardson.h
 *
 *  Created on: 2018 August 10
 *      Author: witte
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SMOOTHERRICHARDSON_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SMOOTHERRICHARDSON_H_


#include "smoother_base.h"

template<typename OperatorType, typename VectorType>
class SmootherRichardson : public SmootherBase<VectorType>
{
public:
  using value_type = typename OperatorType::value_type;

  SmootherRichardson() : underlying_operator(nullptr), preconditioner(nullptr)
  {
  }

  virtual ~SmootherRichardson() override
  {
    clear();
  }

  SmootherRichardson(SmootherRichardson const &) = delete;

  SmootherRichardson &
  operator=(SmootherRichardson const &) = delete;

  virtual void
  clear()
  {
    underlying_operator = nullptr;
    preconditioner      = nullptr;
  }

  struct AdditionalData
  {
    // number of iterations per smoothing step
    unsigned int number_of_smoothing_steps = 1;

    // damping factor
    double damping_factor = 1.;
  };

  void
  initialize(OperatorType &                   operator_in,
             PreconditionerBase<VectorType> & preconditioner_in,
             AdditionalData const &           additional_data_in)
  {
    underlying_operator = &operator_in;
    preconditioner      = &preconditioner_in;
    additional_data     = additional_data_in;
  }

  const AdditionalData &
  get_additional_data() const
  {
    return additional_data;
  }

  // virtual void
  // update()
  // {
  //   if(preconditioner != nullptr)
  //     preconditioner->update(underlying_operator);
  // }

  virtual void
  vmult(VectorType & dst, VectorType const & src) const override
  {
    vmult_impl<false>(dst, src);
  }

  void
  Tvmult(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(false, dealii::ExcMessage("This method is untested..."));
    vmult_impl<true>(dst, src);
  }

  virtual void
  step(VectorType & dst, VectorType const & src) const override
  {
    step_impl<false>(dst, src);
  }

  // TODO why do we need class template VectorType for SmootherBase? -> Niklas
  void
  Tstep(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(false, dealii::ExcMessage("This method is untested..."));
    step_impl<true>(dst, src);
  }

private:
  /*
   *  Approximately solve linear system of equations
   *
   *    A*x = b   (r=b-A*x)
   *
   *  using the (damped Richardson) iteration
   *
   *    x^{k+1} = x^{k} + omega * P * r^{k}
   *
   *  where
   *
   *    omega: damping factor
   *    A:     underlying_operator
   *    P:     preconditioner
   *    b:     src
   *    x:     dst
   */
  template<bool transpose>
  void
  step_impl(VectorType & dst, VectorType const & src, const unsigned int step_start = 0) const
  {
    Assert(preconditioner != nullptr, ExcNotInitialized());
    Assert(underlying_operator != nullptr, ExcNotInitialized());
    VectorType tmp(src), residual(src);

    // *** compute approximate solutions: x^{k}, k=step_start,...,#smoothing_steps
    for(unsigned int k = step_start; k < additional_data.number_of_smoothing_steps; ++k)
    {
      // *** compute residual r^{k} = b - A * x^{k}
      if(transpose)
        underlying_operator->Tvmult(residual, dst);
      else
        underlying_operator->vmult(residual, dst);
      residual.sadd(-1.0, 1.0, src);

      // *** compute and add correction: x^{k+1} = x^{k} + omega * P * r^{k}
      if(transpose)
      {
        AssertThrow(false, ExcMessage("TODO preconditioner->Tvmult(tmp, residual)"));
      }
      else
        preconditioner->vmult(tmp, residual);
      Assert(additional_data.damping_factor == 1.,
             ExcMessage("Safety stop. For my thesis this parameter should not be used."));
      dst.add(additional_data.damping_factor, tmp);
    }
  }

  template<bool transpose>
  void
  vmult_impl(VectorType & dst, VectorType const & src) const
  {
    Assert(preconditioner != nullptr, ExcNotInitialized());
    Assert(underlying_operator != nullptr, ExcNotInitialized());

    dst = 0.;
    step_impl<transpose>(dst, src);
  }

  OperatorType * underlying_operator;

  PreconditionerBase<VectorType> * preconditioner;

  AdditionalData additional_data;
};



#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SMOOTHERRICHARDSON_H_ */
