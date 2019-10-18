/*
 * laplace_integrator.h
 *
 *  Created on: Aug 11, 2018
 *      Author: witte
 */

#ifndef TESTS_LAPLACEINTEGRATOR_H_
#define TESTS_LAPLACEINTEGRATOR_H_

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include "operators/linear_operator_base.h"
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
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using This          = MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransfer<dim, fe_degree, fe_degree + 1, 1, Number>;

  static constexpr int fe_order   = fe_degree + 1;
  static constexpr int macro_size = VectorizedArray<Number>::n_array_elements;

  template<typename Evaluator>
  struct CellMass
  {
    CellMass(const Table<2, VectorizedArray<Number>> & unit_mass_matrix)
      : cell_mass_unit(unit_mass_matrix)
    {
    }

    void
    operator()(const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const Table<2, VectorizedArray<Number>> cell_mass_unit;
  };

  template<typename Evaluator>
  struct CellLaplace
  {
    void
    operator()(const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;
  };

  template<typename Evaluator>
  struct FaceLaplace
  {
    void
    operator()(const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no,
               const std::bitset<macro_size>       bdry_mask) const;

    void
    operator()(const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           direction) const;

    // TODO consider penalty w.r.t. to real cell or surrogate cell ?!
    // template <typename Evaluator>
    VectorizedArray<Number> static compute_penalty(const Evaluator &             fd_eval,
                                                   const int                     direction,
                                                   const int                     cell_no,
                                                   const int                     cell_no_neighbor,
                                                   const std::bitset<macro_size> at_boundary_mask);
  };

  template<typename TPMatrix, typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & data,
                             std::vector<TPMatrix> &               inverses,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    const auto & assemble_subspace_inverses_impl = [&](auto && fd_eval) {
      using EVALUATOR = typename std::decay<decltype(fd_eval)>::type;

      Table<2, VectorizedArray<Number>> cell_mass_unit{fe_order, fe_order};
      fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
      // const auto cell_operation_mass = [&](const auto &                        fd_eval,
      //                                      Table<2, VectorizedArray<Number>> & cell_matrix,
      //                                      const int                           direction,
      //                                      const int                           cell_no) {
      //   AssertIndexRange(direction, static_cast<int>(dim));
      //   EVALUATOR::CellAssembler::scale_matrix(fd_eval.get_h(direction, cell_no),
      //                                          make_array_view(cell_mass_unit),
      //                                          make_array_view(cell_matrix));
      // };
      CellMass<EVALUATOR>    cell_mass_assembler{cell_mass_unit};
      CellLaplace<EVALUATOR> cell_laplace_assembler;

      for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
      {
        fd_eval.reinit(id);
        fd_eval.evaluate(true);

        const std::array<Table<2, VectorizedArray<Number>>, dim> & mass_matrices =
          fd_eval.patch_action(cell_mass_assembler);

        const std::array<Table<2, VectorizedArray<Number>>, dim> & laplace_matrices =
          fd_eval.patch_action(cell_laplace_assembler,
                               FaceLaplace<EVALUATOR>{},
                               FaceLaplace<EVALUATOR>{});

        inverses[id].reinit(mass_matrices, laplace_matrices);
      }
    };

    auto && evaluator = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>{data};
    assemble_subspace_inverses_impl(evaluator);
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & patch_storage) const
  {
    // if (patch_transfer == nullptr)
    return std::make_shared<transfer_type>(patch_storage);
  }
};

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++   inline definitions: MatrixIntegrator   ++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellMass<Evaluator>::
operator()(const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(cell_no, fd_eval.patch_variant == TPSS::PatchVariant::cell ? 1 : 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  Evaluator::CellAssembler::scale_matrix(fd_eval.get_h(direction, cell_no),
                                         make_array_view(cell_mass_unit),
                                         make_array_view(cell_matrix));
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellLaplace<Evaluator>::
operator()(const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(cell_no, fd_eval.patch_variant == TPSS::PatchVariant::cell ? 1 : 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  auto integral{make_vectorized_array<Number>(0.)};
  for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < Evaluator::n_q_points; ++q)
      {
        const auto & grad_u = fd_eval.shape_gradient(dof_u, q, direction, cell_no);
        const auto & grad_v = fd_eval.shape_gradient(dof_v, q, direction, cell_no);
        const auto & dx     = fd_eval.get_JxW(q, direction, cell_no);
        integral += grad_u * grad_v * dx;
      }
      cell_matrix(dof_v, dof_u) += integral;
    }

  // // FUTURE WORK efficiency gain with assembler?
  // auto&& gradients_view = make_array_view (
  //   &(fd_eval.shape_gradient (0, 0, direction, cell_no)),
  //   &(fd_eval.shape_gradient (fe_degree, n_q_points_1d - 1, direction, cell_no)) + 1);
  // auto&& dx_view =
  //   make_array_view (&(fd_eval.get_JxW (0, direction, cell_no)),
  //                            &(fd_eval.get_JxW (n_q_points_1d - 1, direction, cell_no)) + 1);
  // auto&& matrix_view =
  //   make_array_view (&(cell_matrix (0, 0)), &(cell_matrix (fe_degree, fe_degree)) + 1);

  // TPSS::MatrixEvaluator<fe_order, n_q_points_1d, VectorizedArray<Number>>::
  //   template assemble<false> (gradients_view, gradients_view, dx_view, matrix_view);
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::FaceLaplace<Evaluator>::
operator()(const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no,
           const int                           face_no,
           const std::bitset<macro_size>       bdry_mask) const
{
  using namespace dealii;

  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  const auto normal{make_vectorized_array<Number>(face_no == 0 ? -1. : 1.)};
  const auto penalty{compute_penalty(fd_eval, direction, cell_no, cell_no, bdry_mask)};

  /*** factor varies on interior and boundary cells ***/
  auto factor{make_vectorized_array<Number>(0.)};
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    factor[vv] = bdry_mask[vv] ? -1. : -0.5;

  auto value_on_face{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v      = fd_eval.shape_value_face(dof_v, face_no, direction, cell_no);
    const auto & grad_v = fd_eval.shape_gradient_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u      = fd_eval.shape_value_face(dof_u, face_no, direction, cell_no);
      const auto & grad_u = fd_eval.shape_gradient_face(dof_u, face_no, direction, cell_no);

      value_on_face =
        factor * (v * normal * grad_u + grad_v * u * normal); // consistency + symmetry
      value_on_face += penalty * v * u;                       // penalty
      cell_matrix(dof_v, dof_u) += value_on_face;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::FaceLaplace<Evaluator>::
operator()(const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix01,
           Table<2, VectorizedArray<Number>> & cell_matrix10,
           const int                           direction) const
{
  using namespace dealii;

  AssertIndexRange(direction, dim);
  AssertDimension(cell_matrix01.n_rows(), cell_matrix01.n_cols());
  AssertDimension(static_cast<int>(cell_matrix01.n_rows()), fe_order);
  AssertDimension(cell_matrix01.n_rows(), cell_matrix10.n_rows());
  AssertDimension(cell_matrix01.n_cols(), cell_matrix10.n_cols());

  /*** the outward normal on face 1 seen from cell 0 ***/
  const auto normal0{make_vectorized_array<Number>(1.)};
  /*** the outward normal on face 0 seen from cell 1 ***/
  const auto normal1{make_vectorized_array<Number>(-1.)};
  /*** boundary mask is obiviously 0(=all interior), cell_no = 0 and cell_no_neighbor = 1 ***/
  const auto penalty{compute_penalty(fd_eval, direction, 0, 1, 0)};

  auto value_on_interface01{make_vectorized_array<Number>(0.)};
  auto value_on_interface10{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz & v is test shape function
  {
    const auto & v0      = fd_eval.shape_value_face(dof_v, 1, direction, 0);
    const auto & grad_v0 = fd_eval.shape_gradient_face(dof_v, 1, direction, 0);
    const auto & v1      = fd_eval.shape_value_face(dof_v, 0, direction, 1);
    const auto & grad_v1 = fd_eval.shape_gradient_face(dof_v, 0, direction, 1);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u0      = fd_eval.shape_value_face(dof_u, 1, direction, 0);
      const auto & grad_u0 = fd_eval.shape_gradient_face(dof_u, 1, direction, 0);
      const auto & u1      = fd_eval.shape_value_face(dof_u, 0, direction, 1);
      const auto & grad_u1 = fd_eval.shape_gradient_face(dof_u, 0, direction, 1);

      value_on_interface01 =
        -0.5 * (v0 * normal0 * grad_u1 + grad_v0 * u1 * normal1); // consistency + symmetry
      value_on_interface10 =
        -0.5 * (v1 * normal1 * grad_u0 + grad_v1 * u0 * normal0); // consistency + symmetry

      value_on_interface01 -= penalty * v0 * u1; // penalty
      value_on_interface10 -= penalty * v1 * u0; // penalty

      cell_matrix01(dof_v, dof_u) += value_on_interface01;
      cell_matrix10(dof_v, dof_u) += value_on_interface10;
    }
  }
}

// TODO consider penalty w.r.t. to real cell or surrogate cell ?!
template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
VectorizedArray<Number>
MatrixIntegrator<dim, fe_degree, Number>::FaceLaplace<Evaluator>::compute_penalty(
  const Evaluator &             fd_eval,
  const int                     direction,
  const int                     cell_no,
  const int                     cell_no_neighbor,
  const std::bitset<macro_size> at_boundary_mask)
{
  using namespace dealii;

  constexpr auto          degree_factor = static_cast<Number>(fe_degree * (fe_degree + 1.));
  VectorizedArray<Number> penalty       = fe_degree == 0 ?
                                      make_vectorized_array<Number>(1.) :
                                      degree_factor / fd_eval.get_h(direction, cell_no);
  const VectorizedArray<Number> penalty_neighbor =
    fe_degree == 0 ? make_vectorized_array<Number>(1.) :
                     degree_factor / fd_eval.get_h(direction, cell_no_neighbor);

  penalty = 0.5 * (penalty + penalty_neighbor);
  for(unsigned int vv = 0; vv < VectorizedArray<Number>::n_array_elements; ++vv)
    penalty[vv] *= at_boundary_mask[vv] ? 2. : 1.;

  return IP::pre_factor * penalty;
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
class Operator : public LinearOperatorBase
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

  using MFOperator::Tvmult;
};

} // end namespace Laplace


#endif /* TESTS_LAPLACEINTEGRATOR_H_ */
