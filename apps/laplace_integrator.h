/*
 * laplace_integrator.h
 *
 *  Created on: Aug 11, 2018
 *      Author: witte
 */

#ifndef TESTS_LAPLACEINTEGRATOR_H_
#define TESTS_LAPLACEINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include "solvers_and_preconditioners/TPSS/patch_transfer.h"
#include "solvers_and_preconditioners/preconditioner/schwarz_preconditioner.h"
#include "solvers_and_preconditioners/smoother/schwarz_smoother.h"

using namespace dealii;

/*
 * Uniform pre-factor on the interior penalty.
 */
namespace IP
{
double pre_factor = 1.0;
} // end namespace IP

namespace Laplace
{
// TODO namespace DG
/*
 * Linear operators describing the SIPG disretization with respect to the
 * negative Laplacian:
 *
 * (MW) MeshWorker
 * (MF) MatrixFree
 * (FD) FastDiagonalization
 */


/*
 * A MeshWorker-based MatrixIntegrator to assemble the DG Laplace
 * system matrix. Originates from step-39.
 */
namespace MW
{
template<int dim>
class MatrixIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  void
  cell(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
       typename dealii::MeshWorker::IntegrationInfo<dim> & info) const;
  void
  boundary(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
           typename dealii::MeshWorker::IntegrationInfo<dim> & info) const;
  void
  face(dealii::MeshWorker::DoFInfo<dim> &                  dinfo1,
       dealii::MeshWorker::DoFInfo<dim> &                  dinfo2,
       typename dealii::MeshWorker::IntegrationInfo<dim> & info1,
       typename dealii::MeshWorker::IntegrationInfo<dim> & info2) const;
};

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++   inline definitions: MatrixIntegrator   ++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<int dim>
inline void
MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
  LocalIntegrators::Laplace::cell_matrix(dinfo.matrix(0, false).matrix, info.fe_values());
}


template<int dim>
inline void
MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
                                typename dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  const auto         penalty =
    IP::pre_factor * LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg);
  LocalIntegrators::Laplace::nitsche_matrix(dinfo.matrix(0, false).matrix,
                                            info.fe_values(0),
                                            penalty);
}

template<int dim>
inline void
MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &                  dinfo1,
                            dealii::MeshWorker::DoFInfo<dim> &                  dinfo2,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info1,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  const auto         penalty =
    IP::pre_factor * LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg);
  LocalIntegrators::Laplace::ip_matrix(dinfo1.matrix(0, false).matrix,
                                       dinfo1.matrix(0, true).matrix,
                                       dinfo2.matrix(0, true).matrix,
                                       dinfo2.matrix(0, false).matrix,
                                       info1.fe_values(0),
                                       info2.fe_values(0),
                                       penalty);
}

} // end namespace MW

namespace FD
{
template<typename Evaluator, typename Number = typename Evaluator::value_type>
VectorizedArray<Number>
compute_penalty(const Evaluator & eval,
                const int         direction,
                const int         cell_no,
                const int         cell_no_neighbor)
{
  constexpr auto          fe_degree     = static_cast<Number>(Evaluator::fe_order - 1);
  constexpr auto          degree_factor = fe_degree * (fe_degree + 1.);
  VectorizedArray<Number> penalty       = fe_degree == 0 ? make_vectorized_array<Number>(1.) :
                                                     degree_factor / eval.get_h(direction, cell_no);
  const VectorizedArray<Number> penalty_neighbor =
    fe_degree == 0 ? make_vectorized_array<Number>(1.) :
                     degree_factor / eval.get_h(direction, cell_no_neighbor);

  penalty = 0.5 * (penalty + penalty_neighbor);
  return 2. * penalty;
}

template<typename EvaluatorType>
struct CellMass
{
  using Number                  = typename EvaluatorType::value_type;
  static constexpr int fe_order = EvaluatorType::fe_order;

  void
  operator()(const EvaluatorType &               eval_ansatz,
             const EvaluatorType &               eval,
             Table<2, VectorizedArray<Number>> & cell_matrix,
             const int                           direction,
             const int                           cell_no) const;
};

template<typename EvaluatorType>
struct CellLaplace
{
  using Number                  = typename EvaluatorType::value_type;
  static constexpr int fe_order = EvaluatorType::fe_order;

  void
  operator()(const EvaluatorType &,
             const EvaluatorType &               eval,
             Table<2, VectorizedArray<Number>> & cell_matrix,
             const int                           direction,
             const int                           cell_no) const;
};

template<typename EvaluatorType>
struct FaceLaplace
{
  using Number                  = typename EvaluatorType::value_type;
  static constexpr int fe_order = EvaluatorType::fe_order;

  void
  operator()(const EvaluatorType &,
             const EvaluatorType &               eval,
             Table<2, VectorizedArray<Number>> & cell_matrix,
             const int                           direction,
             const int                           cell_no,
             const int                           face_no) const;

  void
  operator()(const EvaluatorType &,
             const EvaluatorType &               eval,
             Table<2, VectorizedArray<Number>> & cell_matrix01,
             Table<2, VectorizedArray<Number>> & cell_matrix10,
             const int                           cell_no_left,
             const int                           direction) const;
};

template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using This          = MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransfer<dim, Number, fe_degree>;

  // static constexpr int fe_order   = fe_degree + 1;
  // static constexpr int macro_size = VectorizedArray<Number>::n_array_elements;

  template<typename TPMatrix, typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & subdomain_handler,
                             std::vector<TPMatrix> &               local_matrices,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    using Evaluator = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;

    Evaluator              eval(subdomain_handler);
    CellMass<Evaluator>    cell_mass_operation;
    CellLaplace<Evaluator> cell_laplace_operation;
    FaceLaplace<Evaluator> nitsche_operation;

    for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
    {
      eval.reinit(patch);
      const auto mass_matrices = eval.patch_action(cell_mass_operation);
      const auto laplace_matrices =
        eval.patch_action(cell_laplace_operation, nitsche_operation, nitsche_operation);
      local_matrices[patch].reinit(mass_matrices, laplace_matrices);
    }
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & patch_storage) const
  {
    return std::make_shared<transfer_type>(patch_storage);
  }
};

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++   inline definitions: MatrixIntegrator   ++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<typename EvaluatorType>
inline void
CellMass<EvaluatorType>::operator()(const EvaluatorType &               eval_ansatz,
                                    const EvaluatorType &               eval_test,
                                    Table<2, VectorizedArray<Number>> & cell_matrix,
                                    const int                           direction,
                                    const int                           cell_no) const
{
  VectorizedArray<Number> integral;
  for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < EvaluatorType::n_q_points_1d_static; ++q)
      {
        const auto & value_u = eval_ansatz.shape_value(dof_u, q, direction, cell_no);
        const auto & value_v = eval_test.shape_value(dof_v, q, direction, cell_no);
        const auto & dx      = eval_test.get_JxW(q, direction, cell_no);
        integral += value_u * value_v * dx;
      }
      cell_matrix(dof_v, dof_u) += integral;
    }
}

template<typename EvaluatorType>
inline void
CellLaplace<EvaluatorType>::operator()(const EvaluatorType &,
                                       const EvaluatorType &               eval,
                                       Table<2, VectorizedArray<Number>> & cell_matrix,
                                       const int                           direction,
                                       const int                           cell_no) const
{
  auto integral{make_vectorized_array<Number>(0.)};
  for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < EvaluatorType::n_q_points_1d_static; ++q)
      {
        const auto & grad_u = eval.shape_gradient(dof_u, q, direction, cell_no);
        const auto & grad_v = eval.shape_gradient(dof_v, q, direction, cell_no);
        const auto & dx     = eval.get_JxW(q, direction, cell_no);
        integral += grad_u * grad_v * dx;
      }
      cell_matrix(dof_v, dof_u) += integral;
    }
}

template<typename EvaluatorType>
inline void
FaceLaplace<EvaluatorType>::operator()(const EvaluatorType &,
                                       const EvaluatorType &               eval_test,
                                       Table<2, VectorizedArray<Number>> & cell_matrix,
                                       const int                           direction,
                                       const int                           cell_no,
                                       const int                           face_no) const
{
  const auto normal         = eval_test.get_normal(face_no);
  const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
  const auto penalty =
    IP::pre_factor * average_factor * compute_penalty(eval_test, direction, cell_no, cell_no);

  auto value_on_face{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v      = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
    const auto & grad_v = eval_test.shape_gradient_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u      = eval_test.shape_value_face(dof_u, face_no, direction, cell_no);
      const auto & grad_u = eval_test.shape_gradient_face(dof_u, face_no, direction, cell_no);

      /// consistency + symmetry
      value_on_face = -average_factor * (v * normal * grad_u + grad_v * u * normal);

      value_on_face += penalty * v * u;

      cell_matrix(dof_v, dof_u) += value_on_face;
    }
  }
}

template<typename EvaluatorType>
inline void
FaceLaplace<EvaluatorType>::operator()(const EvaluatorType &,
                                       const EvaluatorType &               eval_test,
                                       Table<2, VectorizedArray<Number>> & cell_matrix01,
                                       Table<2, VectorizedArray<Number>> & cell_matrix10,
                                       const int                           cell_no0, // left cell
                                       const int                           direction) const
{
  const auto cell_no1 = cell_no0 + 1;                         // right cell
  const auto face_no0 = 1;                                    // interface seen from left cell
  const auto face_no1 = 0;                                    // interface seen from right cell
  AssertDimension(cell_no0, 0);                               // vertex patch has one interface
  const auto normal0        = eval_test.get_normal(face_no0); // on cell 0
  const auto normal1        = eval_test.get_normal(face_no1); // on cell 1
  const auto average_factor = eval_test.get_average_factor(direction, cell_no0, face_no0);
  const auto penalty        = IP::pre_factor * average_factor *
                       Laplace::FD::compute_penalty(eval_test, direction, cell_no0, face_no0);

  auto value_on_interface01{make_vectorized_array<Number>(0.)};
  auto value_on_interface10{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz & v is test shape function
  {
    const auto & v0      = eval_test.shape_value_face(dof_v, face_no0, direction, cell_no0);
    const auto & grad_v0 = eval_test.shape_gradient_face(dof_v, face_no0, direction, cell_no0);
    const auto & v1      = eval_test.shape_value_face(dof_v, face_no1, direction, cell_no1);
    const auto & grad_v1 = eval_test.shape_gradient_face(dof_v, face_no1, direction, cell_no1);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u0      = eval_test.shape_value_face(dof_u, face_no0, direction, cell_no0);
      const auto & grad_u0 = eval_test.shape_gradient_face(dof_u, face_no0, direction, cell_no0);
      const auto & u1      = eval_test.shape_value_face(dof_u, face_no1, direction, cell_no1);
      const auto & grad_u1 = eval_test.shape_gradient_face(dof_u, face_no1, direction, cell_no1);

      /// consistency + symmetry
      value_on_interface01 = -average_factor * (v0 * normal0 * grad_u1 + grad_v0 * u1 * normal1);
      value_on_interface10 = -average_factor * (v1 * normal1 * grad_u0 + grad_v1 * u0 * normal0);

      value_on_interface01 -= penalty * v0 * u1;
      value_on_interface10 -= penalty * v1 * u0;

      cell_matrix01(dof_v, dof_u) += value_on_interface01;
      cell_matrix10(dof_v, dof_u) += value_on_interface10;
    }
  }
}

} // end namespace FD

namespace MF
{
template<int dim, typename number>
void
adjust_ghost_range_if_necessary(const MatrixFree<dim, number> &                    data,
                                const LinearAlgebra::distributed::Vector<number> & vec)
{
  if(vec.get_partitioner().get() == data.get_dof_info(0).vector_partitioner.get())
    return;

  LinearAlgebra::distributed::Vector<number> copy_vec(vec);
  const_cast<LinearAlgebra::distributed::Vector<number> &>(vec).reinit(
    data.get_dof_info(0).vector_partitioner);
  const_cast<LinearAlgebra::distributed::Vector<number> &>(vec).copy_locally_owned_data_from(
    copy_vec);
}

template<int dim, int fe_degree, typename Number>
class Operator : public Subscriptor
{
public:
  using value_type = Number;

  Operator() = default;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> data);

  void
  clear();

  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  void
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> & vec) const;

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const;

  std::vector<TimeInfo>
  get_time_data() const;

  void
  vmult(LinearAlgebra::distributed::Vector<Number> &       dst,
        const LinearAlgebra::distributed::Vector<Number> & src) const;

  void
  Tvmult(LinearAlgebra::distributed::Vector<Number> &       dst,
         const LinearAlgebra::distributed::Vector<Number> & src) const;

  void
  vmult(const ArrayView<Number>, const ArrayView<const Number>) const
  {
    AssertThrow(false, ExcMessage("Dummy satisfying interface of MGCoarseSolver."));
  }

  Number
  get_penalty_factor() const
  {
    return 1.0 * IP::pre_factor * std::max((Number)1., (Number)fe_degree) * (fe_degree + 1);
  }

private:
  void
  apply_cell(const MatrixFree<dim, Number> &                    data,
             LinearAlgebra::distributed::Vector<Number> &       dst,
             const LinearAlgebra::distributed::Vector<Number> & src,
             const std::pair<unsigned int, unsigned int> &      cell_range) const;

  void
  apply_face(const MatrixFree<dim, Number> &                    data,
             LinearAlgebra::distributed::Vector<Number> &       dst,
             const LinearAlgebra::distributed::Vector<Number> & src,
             const std::pair<unsigned int, unsigned int> &      face_range) const;

  void
  apply_boundary(const MatrixFree<dim, Number> &                    data,
                 LinearAlgebra::distributed::Vector<Number> &       dst,
                 const LinearAlgebra::distributed::Vector<Number> & src,
                 const std::pair<unsigned int, unsigned int> &      face_range) const;

  std::shared_ptr<const MatrixFree<dim, Number>> data;
  mutable std::vector<TimeInfo>                  time_infos;
};

template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::clear()
{
  data.reset();
  time_infos.clear();
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize(std::shared_ptr<const MatrixFree<dim, Number>> data)
{
  this->data = data;
  time_infos = {TimeInfo{0., "[MF::Operator] vmult:", "[s]", 0}};
}



template<int dim, int fe_degree, typename Number>
std::shared_ptr<const MatrixFree<dim, Number>>
Operator<dim, fe_degree, Number>::get_matrix_free() const
{
  return data;
}



template<int dim, int fe_degree, typename Number>
std::vector<TimeInfo>
Operator<dim, fe_degree, Number>::get_time_data() const
{
  return time_infos;
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize_dof_vector(
  LinearAlgebra::distributed::Vector<Number> & vec) const
{
  data->initialize_dof_vector(vec);
}



template<int dim, int fe_degree, typename Number>
types::global_dof_index
Operator<dim, fe_degree, Number>::m() const
{
  Assert(data.get() != nullptr, ExcNotInitialized());
  return data->get_dof_handler().n_dofs();
}



template<int dim, int fe_degree, typename Number>
types::global_dof_index
Operator<dim, fe_degree, Number>::n() const
{
  return m();
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::vmult(
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src) const
{
  Timer timer; //{MPI_COMM_WORLD, /*sync procs?*/ true};
  timer.restart();
  adjust_ghost_range_if_necessary(*data, dst);
  adjust_ghost_range_if_necessary(*data, src);
  data->loop(&Operator::apply_cell,
             &Operator::apply_face,
             &Operator::apply_boundary,
             this,
             dst,
             src,
             /*zero_dst*/ true);
  //            MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
  //            MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
  time_infos[0].add_time(timer.wall_time());
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::Tvmult(
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src) const
{
  vmult(dst, src);
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_cell(
  const MatrixFree<dim, Number> &                    data,
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src,
  const std::pair<unsigned int, unsigned int> &      cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src, false, true);
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_gradient(phi.get_gradient(q), q);
    phi.integrate_scatter(false, true, dst);
  }
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_face(
  const MatrixFree<dim, Number> &                    data,
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src,
  const std::pair<unsigned int, unsigned int> &      face_range) const
{
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(data, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_outer(data, false);
  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    phi_inner.reinit(face);
    phi_inner.gather_evaluate(src, true, true);
    phi_outer.reinit(face);
    phi_outer.gather_evaluate(src, true, true);

    const VectorizedArray<Number> inverse_length_normal_to_face =
      0.5 * (std::abs((phi_inner.get_normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]) +
             std::abs((phi_outer.get_normal_vector(0) * phi_outer.inverse_jacobian(0))[dim - 1]));
    const VectorizedArray<Number> sigma = inverse_length_normal_to_face * get_penalty_factor();

    for(unsigned int q = 0; q < phi_inner.n_q_points; ++q)
    {
      const VectorizedArray<Number> solution_jump =
        (phi_inner.get_value(q) - phi_outer.get_value(q));
      const VectorizedArray<Number> average_normal_gradient =
        (phi_inner.get_normal_derivative(q) + phi_outer.get_normal_derivative(q)) * Number(0.5);
      const VectorizedArray<Number> test_by_value = solution_jump * sigma - average_normal_gradient;

      phi_inner.submit_value(test_by_value, q);
      phi_outer.submit_value(-test_by_value, q);

      phi_inner.submit_normal_derivative(-solution_jump * Number(0.5), q);
      phi_outer.submit_normal_derivative(-solution_jump * Number(0.5), q);
    }

    phi_inner.integrate_scatter(true, true, dst);
    phi_outer.integrate_scatter(true, true, dst);
  }
}


template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_boundary(
  const MatrixFree<dim, Number> &                    data,
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src,
  const std::pair<unsigned int, unsigned int> &      face_range) const
{
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(data, true);
  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    phi_inner.reinit(face);
    phi_inner.gather_evaluate(src, true, true);

    const VectorizedArray<Number> inverse_length_normal_to_face =
      std::abs((phi_inner.get_normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<Number> sigma = inverse_length_normal_to_face * get_penalty_factor();

    const bool is_dirichlet = true; //(data.get_boundary_id(face) == 0);

    for(unsigned int q = 0; q < phi_inner.n_q_points; ++q)
    {
      const VectorizedArray<Number> u_inner                 = phi_inner.get_value(q);
      const VectorizedArray<Number> u_outer                 = is_dirichlet ? -u_inner : u_inner;
      const VectorizedArray<Number> normal_derivative_inner = phi_inner.get_normal_derivative(q);
      const VectorizedArray<Number> normal_derivative_outer =
        is_dirichlet ? normal_derivative_inner : -normal_derivative_inner;
      const VectorizedArray<Number> solution_jump = (u_inner - u_outer);
      const VectorizedArray<Number> average_normal_gradient =
        (normal_derivative_inner + normal_derivative_outer) * Number(0.5);
      const VectorizedArray<Number> test_by_value = solution_jump * sigma - average_normal_gradient;
      phi_inner.submit_normal_derivative(-solution_jump * Number(0.5), q);
      phi_inner.submit_value(test_by_value, q);
    }
    phi_inner.integrate_scatter(true, true, dst);
  }
}



} // end namespace MF



template<int dim, int fe_degree, typename Number>
struct CombinedOperator : public MF::Operator<dim, fe_degree, Number>,
                          public FD::MatrixIntegrator<dim, fe_degree, Number>
{
  using MFOperator    = typename MF::Operator<dim, fe_degree, Number>;
  using FDOperator    = typename FD::MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename FDOperator::transfer_type;

  CombinedOperator() = default;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> data)
  {
    MFOperator::initialize(data);
  };

  void
  clear()
  {
    MFOperator::clear();
  };
};



/*
 * Linear operators describing the standard finite element discretization of
 * the negative Laplacian:
 *
 * (MF) MatrixFree
 * (FD) FastDiagonalization // TODO
 */
namespace CFEM
{
/*
 * MatrixFree operator based on MatrixFreeOperators interface
 */
namespace MF
{
template<int dim, int fe_degree, typename Number>
class Operator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<Number>>
{
public:
  using Base = MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<Number>>;
  using Base::size_type;
  using Base::value_type;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage);

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage,
             const MGConstrainedDoFs &                      mg_constrained_dofs,
             const unsigned int                             level);

  void
  clear();

  virtual void
  compute_diagonal() override
  {
    AssertThrow(false, ExcMessage("Assembly of diagonal is not implemented."));
  }

  std::vector<TimeInfo>
  get_time_data() const;

  Number
  get_penalty_factor() const
  {
    return IP::pre_factor * std::max((Number)1., (Number)fe_degree) * (fe_degree + 1);
  }

  using Base::vmult;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage("No MPI possible."));
    LinearAlgebra::distributed::Vector<Number> dst;
    Base::initialize_dof_vector(dst);
    LinearAlgebra::distributed::Vector<Number> src;
    Base::initialize_dof_vector(src);
    std::copy(src_view.begin(), src_view.end(), src.begin());
    Base::vmult(dst, src);
    std::copy(dst.begin(), dst.end(), dst_view.begin());
  }

  using Base::Tvmult;

protected:
  void
  apply_add(LinearAlgebra::distributed::Vector<Number> &       dst,
            const LinearAlgebra::distributed::Vector<Number> & src) const override final;

private:
  void
  apply_cell(const MatrixFree<dim, Number> &                    mf_storage,
             LinearAlgebra::distributed::Vector<Number> &       dst,
             const LinearAlgebra::distributed::Vector<Number> & src,
             const std::pair<unsigned int, unsigned int> &      cell_range) const;

  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
  mutable std::vector<TimeInfo>                  time_infos;
};



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::clear()
{
  mf_storage.reset();
  time_infos.clear();
  Base::clear();
}


template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize(
  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in)
{
  Base::initialize(mf_storage_in);
  mf_storage = mf_storage_in;
  time_infos = {TimeInfo{0., "[MF::Operator] vmult:", "[s]", 0}};
}


template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize(
  std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
  const MGConstrainedDoFs &                      mg_constrained_dofs,
  const unsigned int                             level)
{
  Base::initialize(mf_storage_in, mg_constrained_dofs, level);
  mf_storage = mf_storage_in;
  time_infos = {TimeInfo{0., "[MF::Operator] vmult:", "[s]", 0}};
}


template<int dim, int fe_degree, typename Number>
std::vector<TimeInfo>
Operator<dim, fe_degree, Number>::get_time_data() const
{
  return time_infos;
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_add(
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src) const
{
  Timer timer;
  timer.restart();
  mf_storage->cell_loop(&Operator::apply_cell,
                        this,
                        dst,
                        src,
                        /*zero_dst*/ true);
  time_infos[0].add_time(timer.wall_time());
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_cell(
  const MatrixFree<dim, Number> &                    mf_storage,
  LinearAlgebra::distributed::Vector<Number> &       dst,
  const LinearAlgebra::distributed::Vector<Number> & src,
  const std::pair<unsigned int, unsigned int> &      cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(mf_storage);
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.read_dof_values(src);
    phi.evaluate(false, true);
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_gradient(phi.get_gradient(q), q);
    phi.integrate(false, true);
    phi.distribute_local_to_global(dst);
  }
}



} // end namespace MF



namespace FD
{
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using This          = MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransfer<dim, Number, fe_degree>;

  // static constexpr int fe_order   = fe_degree + 1;
  static constexpr int macro_size = VectorizedArray<Number>::n_array_elements;

  template<typename TPMatrix, typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & subdomain_handler,
                             std::vector<TPMatrix> &               local_matrices,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    using Evaluator = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;

    Evaluator                           eval(subdomain_handler);
    Laplace::FD::CellMass<Evaluator>    cell_mass_operation;
    Laplace::FD::CellLaplace<Evaluator> cell_laplace_operation;

    for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
    {
      eval.reinit(patch);
      const auto mass_matrices    = eval.patch_action(cell_mass_operation);
      const auto laplace_matrices = eval.patch_action(cell_laplace_operation);

      /// initialize (fast diagonalized) matrix
      auto & local_solver = local_matrices[patch];
      local_solver.matrix.reinit(mass_matrices, laplace_matrices);

      /// set constraints
      for(auto lane = 0U; lane < macro_size; ++lane)
      {
        auto tmp_row = eval.get_constrained_dof_indices(lane);
        std::swap(local_solver.constrained_dof_indices_row[lane], tmp_row);
        auto tmp_col = eval.get_constrained_dof_indices(lane);
        std::swap(local_solver.constrained_dof_indices_col[lane], tmp_col);
      }
    }
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & patch_storage) const
  {
    return std::make_shared<transfer_type>(patch_storage);
  }
};



} // end namespace FD



template<int dim, int fe_degree, typename Number>
struct CombinedOperator : public CFEM::MF::Operator<dim, fe_degree, Number>,
                          public CFEM::FD::MatrixIntegrator<dim, fe_degree, Number>
{
  using MFOperator    = typename CFEM::MF::Operator<dim, fe_degree, Number>;
  using FDOperator    = typename CFEM::FD::MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename FDOperator::transfer_type;

  CombinedOperator() = default;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> mf_storage_in,
             const MGConstrainedDoFs &                      mg_constrained_dofs,
             const unsigned int                             level)
  {
    MFOperator::initialize(mf_storage_in, mg_constrained_dofs, level);
  };

  void
  clear()
  {
    MFOperator::clear();
  };
};



} // end namespace CFEM



} // end namespace Laplace


#endif /* TESTS_LAPLACEINTEGRATOR_H_ */
