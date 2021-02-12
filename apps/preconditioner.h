
/*
 * collection of preconditioners
 *
 *  Created on: Dec 17, 2020
 *      Author: witte
 */

#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

using namespace dealii;



/**
 * Helper class computing an approximate inverse iteratively. Certainly, not a
 * best practice but all members are public for convenience allowing easy
 * custumization.
 */
struct SolverDirectInverse : public Subscriptor
{
  SolverDirectInverse(const TrilinosWrappers::SparseMatrix & matrix_in);

  void
  vmult(LinearAlgebra::distributed::Vector<double> &       dst,
        const LinearAlgebra::distributed::Vector<double> & src) const;

  const TrilinosWrappers::SparseMatrix & matrix;

  const std::string  solver_variant      = "direct";
  const double       abs_accuracy        = 1.e-12;
  const unsigned int n_solver_iterations = 0;
};



/**
 * Helper class computing an approximate inverse iteratively. Certainly, not a
 * best practice but all members are public for convenience allowing easy
 * custumization.
 */
template<typename PreconditionerType = PreconditionIdentity>
struct IterativeInverse : public Subscriptor
{
  IterativeInverse(const TrilinosWrappers::SparseMatrix & matrix_in,
                   const PreconditionerType & preconditioner_in = PreconditionIdentity{});

  void
  vmult(LinearAlgebra::distributed::Vector<double> &       dst,
        const LinearAlgebra::distributed::Vector<double> & src) const;

  const TrilinosWrappers::SparseMatrix & matrix;
  const PreconditionerType &             preconditioner;

  std::string          solver_variant      = "cg";
  double               abs_accuracy        = 1.e-12;
  double               rel_accuracy        = 1.e-2;
  unsigned int         n_iterations_max    = 1000;
  mutable unsigned int n_solver_iterations = 0;
};



/**
 * Providing an interface to apply a matrix and its approximate inverse. For
 * more details on the inverse visit IterativeInverse.
 */
template<typename PreconditionerType = PreconditionIdentity>
struct LinearOperatorInv : public Subscriptor
{
  LinearOperatorInv(const TrilinosWrappers::SparseMatrix & matrix_in,
                    const PreconditionerType & preconditioner_in = PreconditionIdentity{});

  void
  vmult(LinearAlgebra::distributed::Vector<double> &       dst,
        const LinearAlgebra::distributed::Vector<double> & src) const;

  void
  apply_inverse(LinearAlgebra::distributed::Vector<double> &       dst,
                const LinearAlgebra::distributed::Vector<double> & src) const;

  const TrilinosWrappers::SparseMatrix &                matrix;
  std::shared_ptr<IterativeInverse<PreconditionerType>> inverse;
};



/**
 * A block Schur preconditioner P^-1 for the Stokes system, which reads
 *
 *    | A  B^T |
 *    | B  0   |
 *
 * in matrix form, is defined as product of simple block matrices
 *
 *    | A^{-1}  0 | . | I  B^T | . | I  0      |
 *    | 0       I |   | 0  -I  |   | 0  S^{-1} |
 *
 * where S = B A^{-1} B^T is the negative(!) Schur complement of the Stokes system.
 * We emphasize that P^{-1} is a right preconditioner, to be precise, we obtain
 *
 *    | A  B^T | . P^{-1} = | I       0 |
 *    | B  0   |            | BA^{-1} I |
 *
 * if we apply P^{-1} from the right. (Approximate) inverse matrices A^{-1} and
 * S^{-1} are passed by the constructor as @p preconditioner_A and @p
 * preconditioner_S, respectively. For more details see step-56 ...
 */
template<typename PreconditionerAType, typename PreconditionerSType>
class PreconditionBlockSchur : public Subscriptor
{
public:
  using preconditioner_A_type = PreconditionerAType;
  using preconditioner_S_type = PreconditionerSType;

  /**
   * See the class' description.
   */
  PreconditionBlockSchur(const TrilinosWrappers::BlockSparseMatrix & system_matrix,
                         const PreconditionerAType &                 preconditioner_A,
                         const PreconditionerSType &                 preconditioner_S);

  /**
   * Computes the matrix-vector multiplication with @p src resulting in
   *
   *   |u_dst|  =  | A^{-1}  0 | . | I  B^T | . | I  0      | |u_src|
   *   |p_dst|     | 0       I |   | 0  -I  |   | 0  S^{-1} | |p_src|
   *
   * where @p preconditioner_A and @p preconditioner_S are approximations of
   * A^{-1} and S^{-1}, respectively
   */
  void
  vmult(LinearAlgebra::distributed::BlockVector<double> &       dst,
        const LinearAlgebra::distributed::BlockVector<double> & src) const;

  const TrilinosWrappers::BlockSparseMatrix & system_matrix;
  const PreconditionerAType &                 preconditioner_A;
  const PreconditionerSType &                 preconditioner_S;
};



////////////////////////////// Definitions



SolverDirectInverse::SolverDirectInverse(const TrilinosWrappers::SparseMatrix & matrix_in)
  : matrix(matrix_in)
{
}



void
SolverDirectInverse::vmult(LinearAlgebra::distributed::Vector<double> &       dst,
                           const LinearAlgebra::distributed::Vector<double> & src) const
{
  SolverControl                  solver_control(1U, abs_accuracy, false, false);
  TrilinosWrappers::SolverDirect direct_solver(solver_control);
  direct_solver.solve(matrix, dst, src);
}



template<typename PreconditionerType>
IterativeInverse<PreconditionerType>::IterativeInverse(
  const TrilinosWrappers::SparseMatrix & matrix_in,
  const PreconditionerType &             preconditioner_in)
  : matrix(matrix_in), preconditioner(preconditioner_in)
{
}



template<typename PreconditionerType>
void
IterativeInverse<PreconditionerType>::vmult(
  LinearAlgebra::distributed::Vector<double> &       dst,
  const LinearAlgebra::distributed::Vector<double> & src) const
{
  const auto tolerance = std::max<double>(src.l2_norm() * rel_accuracy, abs_accuracy);
  SolverSelector<LinearAlgebra::distributed::Vector<double>> iterative_solver;
  SolverControl solver_control(n_iterations_max, tolerance);
  iterative_solver.set_control(solver_control);
  iterative_solver.select(solver_variant);
  iterative_solver.solve(matrix, dst, src, preconditioner);
  n_solver_iterations = solver_control.last_step();
}



template<typename PreconditionerType>
LinearOperatorInv<PreconditionerType>::LinearOperatorInv(
  const TrilinosWrappers::SparseMatrix & matrix_in,
  const PreconditionerType &             preconditioner_in)
  : matrix(matrix_in),
    inverse(std::make_shared<IterativeInverse<PreconditionerType>>(matrix_in, preconditioner_in))
{
}



template<typename PreconditionerType>
void
LinearOperatorInv<PreconditionerType>::vmult(
  LinearAlgebra::distributed::Vector<double> &       dst,
  const LinearAlgebra::distributed::Vector<double> & src) const
{
  matrix.vmult(dst, src);
}



template<typename PreconditionerType>
void
LinearOperatorInv<PreconditionerType>::apply_inverse(
  LinearAlgebra::distributed::Vector<double> &       dst,
  const LinearAlgebra::distributed::Vector<double> & src) const
{
  inverse->vmult(dst, src);
}



template<class PreconditionerAType, class PreconditionerSType>
PreconditionBlockSchur<PreconditionerAType, PreconditionerSType>::PreconditionBlockSchur(
  const TrilinosWrappers::BlockSparseMatrix & system_matrix_in,
  const PreconditionerAType &                 preconditioner_A_in,
  const PreconditionerSType &                 preconditioner_S_in)
  : system_matrix(system_matrix_in),
    preconditioner_A(preconditioner_A_in),
    preconditioner_S(preconditioner_S_in)
{
}



template<class PreconditionerAType, class PreconditionerSType>
void
PreconditionBlockSchur<PreconditionerAType, PreconditionerSType>::vmult(
  LinearAlgebra::distributed::BlockVector<double> &       dst,
  const LinearAlgebra::distributed::BlockVector<double> & src) const
{
  const auto & u_src = src.block(0);
  const auto & p_src = src.block(1);
  auto &       u_dst = dst.block(0);
  auto &       p_dst = dst.block(1);

  ///   |u_src|  =  | I  0      | |u_src|
  ///   |p_dst|     | 0  S^{-1} | |p_src|
  {
    // p_dst = 0.; // ???
    preconditioner_S.vmult(p_dst, p_src);
  }

  LinearAlgebra::distributed::Vector<double> u_tmp(src.block(0).get_partitioner());
  ///   |u_tmp|  =  | I  B^T | . | I  0      | |u_src|
  ///   |p_dst|     | 0  -I  |   | 0  S^{-1} | |p_src|
  {
    system_matrix.block(0, 1).vmult(u_tmp, p_dst);
    u_tmp += u_src;
    p_dst *= -1.;
  }

  ///   |u_dst|  =  | A^{-1}  0 | . | I  B^T | . | I  0      | |u_src|
  ///   |p_dst|     | 0       I |   | 0  -I  |   | 0  S^{-1} | |p_src|
  {
    preconditioner_A.vmult(u_dst, u_tmp);
  }

  AssertIsFinite(dst.l2_norm());
}



#endif /* PRECONDITIONER_H */
