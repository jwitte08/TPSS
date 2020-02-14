/*
 * preconditioner_base.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn, witte
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_

using namespace dealii;

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/la_parallel_vector.h>

/*
 * Required to treat BlockVectors.
 * This structure is similar to SmootherBase.
 */
template<typename VectorType>
class PreconditionerBase : public Subscriptor
{
public:
  virtual ~PreconditionerBase()
  {
  }

  virtual void
  vmult(VectorType & dst, VectorType const & src) const = 0;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_ */
