/*
 * common_integrator.h
 *
 *  Created on: May 26, 2019
 *      Author: witte
 */

#ifndef APPS_COMMONINTEGRATOR_H_
#define APPS_COMMONINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>

#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"



namespace Nitsche
{
/**
 * Standard penalty to obtain well-posedness of the Nitsche method. The
 * penalty is weighted for face integrals at the physical boundary (that is
 * h_left = h_right). In case of interior penalty methods, the penalty factor
 * for interfaces between a so-called "left" and "right" cell is obtained by
 * multiplying with 1/2.
 */
template<typename Number>
Number
compute_penalty_impl(const int degree, const Number h_left, const Number h_right)
{
  const auto one_over_h = (0.5 / h_left) + (0.5 / h_right);
  const auto gamma      = degree == 0 ? 1 : degree * (degree + 1);
  return 2.0 * gamma * one_over_h;
}
} // namespace Nitsche



namespace FD
{
using namespace dealii;

template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct CellOperationBase
{
  static constexpr unsigned int fe_degree_static        = fe_degree;
  static constexpr unsigned int ansatz_fe_degree_static = fe_degree;
  static constexpr unsigned int n_q_points_static       = n_q_points_1d;

  using value_type            = Number;
  using evaluator_type        = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;
  using ansatz_evaluator_type = FDEvaluation<dim, fe_degree_ansatz, n_q_points_1d, Number>;
  using matrix_type           = Table<2, VectorizedArray<Number>>;

  void
  operator()(const ansatz_evaluator_type & eval_ansatz,
             const evaluator_type &        eval_test,
             matrix_type &                 cell_matrix,
             const int                     direction,
             const int                     cell_no) const
  {
    AssertThrow(false, ExcMessage("Please, overload this operation in derived classes."));
  }
};



template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperationBase
{
  static constexpr unsigned int fe_degree_static        = fe_degree;
  static constexpr unsigned int ansatz_fe_degree_static = fe_degree;
  static constexpr unsigned int n_q_points_static       = n_q_points_1d;

  using value_type            = Number;
  using evaluator_type        = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;
  using ansatz_evaluator_type = FDEvaluation<dim, fe_degree_ansatz, n_q_points_1d, Number>;
  using matrix_type           = Table<2, VectorizedArray<Number>>;

  void
  operator()(const ansatz_evaluator_type & eval_ansatz,
             const evaluator_type &        eval_test,
             matrix_type &                 cell_matrix,
             const int                     direction,
             const int                     cell_no,
             const int                     face_no) const
  {
    AssertThrow(false, ExcMessage("Please, overload this operation in derived classes."));
  }

  void
  operator()(const ansatz_evaluator_type & eval_ansatz,
             const evaluator_type &        eval_test,
             matrix_type &                 cell_matrix01,
             matrix_type &                 cell_matrix10,
             const int                     cell_no0,
             const int                     direction)
  {
    AssertThrow(false, ExcMessage("Please, overload this operation in derived classes."));
  }
};



namespace Gradient
{
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct CellOperation : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const int n_q_points    = eval_test.n_q_points_1d(direction);

    VectorizedArray<Number> integral;
    for(int i = 0; i < n_dofs_test; ++i)
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        integral = 0.;
        for(int q = 0; q < n_q_points; ++q)
        {
          const auto & grad_u_j = eval_ansatz.shape_gradient(i, q, direction, cell_no);
          const auto & v_i      = eval_test.shape_value(j, q, direction, cell_no);
          const auto & dx       = eval_test.get_JxW(q, direction, cell_no);
          integral += grad_u_j * v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};
} // namespace Gradient



namespace Laplace
{
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct CellOperation : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const int n_q_points    = eval_test.n_q_points_1d(direction);

    VectorizedArray<Number> integral;
    for(int i = 0; i < n_dofs_test; ++i)
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        integral = 0.;
        for(int q = 0; q < n_q_points; ++q)
        {
          const auto & grad_u_j = eval_ansatz.shape_gradient(i, q, direction, cell_no);
          const auto & grad_v_i = eval_test.shape_gradient(j, q, direction, cell_no);
          const auto & dx       = eval_test.get_JxW(q, direction, cell_no);
          integral += grad_u_j * grad_v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};

namespace SIPG
{
template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperation
  : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no,
             const int                                    face_no) const
  {
    const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
    const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
    const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
    const auto normal         = eval_test.get_normal(face_no);

    const auto h = eval_test.get_h(direction, cell_no);
    const auto penalty =
      this->penalty_factor * average_factor * ::Nitsche::compute_penalty_impl(fe_degree, h, h);

    auto value_on_face = make_vectorized_array<Number>(0.);
    for(int i = 0; i < n_dofs_test; ++i)
    {
      const auto & v_i      = eval_test.shape_value_face(i, face_no, direction, cell_no);
      const auto & grad_v_i = eval_test.shape_gradient_face(i, face_no, direction, cell_no);
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        const auto & u_j      = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);
        const auto & grad_u_j = eval_ansatz.shape_gradient_face(j, face_no, direction, cell_no);

        value_on_face = -average_factor * (v_i * normal * grad_u_j + grad_v_i * u_j * normal);

        value_on_face += penalty * v_i * u_j * normal * normal;

        cell_matrix(i, j) += value_on_face;
      }
    }
  }

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix01,
             typename Base::matrix_type &                 cell_matrix10,
             const int                                    cell_no0,
             const int                                    direction) const
  {
    AssertDimension(cell_no0, 0);

    const int  n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int  n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const auto normal0       = eval_test.get_normal(1); // on cell 0
    const auto normal1       = eval_test.get_normal(0); // on cell 1 !!!

    const auto h0 = eval_test.get_h(direction, cell_no0);
    const auto h1 = eval_test.get_h(direction, cell_no0 + 1);
    const auto penalty =
      this->interior_penalty_factor * 0.5 * ::Nitsche::compute_penalty_impl(fe_degree, h0, h1);

    auto value_on_interface01 = make_vectorized_array<Number>(0.);
    auto value_on_interface10 = make_vectorized_array<Number>(0.);
    for(int i = 0; i < n_dofs_test; ++i) // u is ansatz & v is test shape function
    {
      const auto & v0_i      = eval_test.shape_value_face(i, 1, direction, 0);
      const auto & grad_v0_i = eval_test.shape_gradient_face(i, 1, direction, 0);
      const auto & v1_i      = eval_test.shape_value_face(i, 0, direction, 1);
      const auto & grad_v1_i = eval_test.shape_gradient_face(i, 0, direction, 1);
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        const auto & u0_j      = eval_ansatz.shape_value_face(j, 1, direction, 0);
        const auto & grad_u0_j = eval_ansatz.shape_gradient_face(j, 1, direction, 0);
        const auto & u1_j      = eval_ansatz.shape_value_face(j, 0, direction, 1);
        const auto & grad_u1_j = eval_ansatz.shape_gradient_face(j, 0, direction, 1);

        /// consistency + symmetry
        value_on_interface01 = -0.5 * (v0_i * normal0 * grad_u1_j + grad_v0_i * u1_j * normal1);
        value_on_interface10 = -0.5 * (v1_i * normal1 * grad_u0_j + grad_v1_i * u0_j * normal0);

        /// penalty
        value_on_interface01 += penalty * v0_i * u1_j * normal0 * normal1;
        value_on_interface10 += penalty * v1_i * u0_j * normal1 * normal0;

        cell_matrix01(i, j) += value_on_interface01;
        cell_matrix10(i, j) += value_on_interface10;
      }
    }
  }

  Number penalty_factor          = 1.;
  Number interior_penalty_factor = 1.;
};

} // namespace SIPG
} // namespace Laplace



namespace L2
{
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct CellOperation : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const int n_q_points    = eval_test.n_q_points_1d(direction);

    VectorizedArray<Number> integral;
    for(int i = 0; i < n_dofs_test; ++i)
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        integral = 0.;
        for(int q = 0; q < n_q_points; ++q)
        {
          const auto & u_j = eval_ansatz.shape_value(i, q, direction, cell_no);
          const auto & v_i = eval_test.shape_value(j, q, direction, cell_no);
          const auto & dx  = eval_test.get_JxW(q, direction, cell_no);
          integral += u_j * v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};

template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperation
  : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no,
             const int                                    face_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

    const auto average_factor = use_average_of_gradient ?
                                  eval_test.get_average_factor(direction, cell_no, face_no) :
                                  make_vectorized_array(1.);

    auto value_on_face = make_vectorized_array<Number>(0.);
    for(int i = 0; i < n_dofs_test; ++i)
    {
      const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        const auto & u_j = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);

        value_on_face = v_i * u_j;

        cell_matrix(i, j) += average_factor * value_on_face;
      }
    }
  }

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix01,
             typename Base::matrix_type &                 cell_matrix10,
             const int                                    cell_no0,
             const int                                    direction) const
  {
    AssertDimension(cell_no0, 0);

    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

    const auto average_factor = use_average_of_gradient ? 0.5 : 1.;

    auto value_on_interface01 = make_vectorized_array<Number>(0.);
    auto value_on_interface10 = make_vectorized_array<Number>(0.);
    for(int i = 0; i < n_dofs_test; ++i) // u is ansatz & v is test shape function
    {
      const auto & v0_i = eval_test.shape_value_face(i, 1, direction, 0);
      const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        const auto & u0_j = eval_ansatz.shape_value_face(j, 1, direction, 0);
        const auto & u1_j = eval_ansatz.shape_value_face(j, 0, direction, 1);

        value_on_interface01 = average_factor * v0_i * u1_j;
        value_on_interface10 = average_factor * v1_i * u0_j;

        cell_matrix01(i, j) += value_on_interface01;
        cell_matrix10(i, j) += value_on_interface10;
      }
    }
  }

  /**
   * If true we use the averaged gradient {{grad u}} instead of grad u. This
   * option is typically used for DG methods.
   */
  bool use_average_of_gradient = false;
};

} // namespace L2



namespace Void
{
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct CellOperation : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number>;

  void
  operator()(const typename Base::ansatz_evaluator_type &,
             const typename Base::evaluator_type &,
             typename Base::matrix_type &,
             const int,
             const int) const
  {
  }
};

template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperation
  : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type &,
             const typename Base::evaluator_type &,
             typename Base::matrix_type &,
             const int,
             const int,
             const int) const
  {
  }

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix01,
             typename Base::matrix_type &                 cell_matrix10,
             const int                                    cell_no0,
             const int                                    direction) const
  {
  }
};

} // namespace Void

} // namespace FD



namespace MW
{
template<int dim, bool is_multigrid>
struct IteratorSelector
{
  // static_assert(false, "No specialization has been found.");
};

template<int dim>
struct IteratorSelector<dim, false>
{
  using type = typename DoFHandler<dim>::active_cell_iterator;
};

template<int dim>
struct IteratorSelector<dim, true>
{
  using type = typename DoFHandler<dim>::level_cell_iterator;
};



template<int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe,
              const unsigned int         quadrature_degree,
              const UpdateFlags          update_flags,
              const UpdateFlags          interface_update_flags)
    : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags),
      fe_interface_values(mapping, fe, QGauss<dim - 1>(quadrature_degree), interface_update_flags)
  {
  }

  ScratchData(const ScratchData<dim> & scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags()),
      fe_interface_values(scratch_data.fe_values.get_mapping(),
                          scratch_data.fe_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {
  }

  FEValues<dim>          fe_values;
  FEInterfaceValues<dim> fe_interface_values;
};



struct CopyData
{
  struct FaceData
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
    Vector<double>                       cell_rhs;
  };

  CopyData(const unsigned int dofs_per_cell,
           const unsigned int level_in = numbers::invalid_unsigned_int)
    : level(level_in),
      cell_matrix(dofs_per_cell, dofs_per_cell),
      cell_rhs(dofs_per_cell),
      local_dof_indices(dofs_per_cell)
  {
  }

  CopyData(const CopyData &) = default;

  unsigned int                         level;
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<FaceData>                face_data;
};
} // namespace MW

#endif /* APPS_COMMONINTEGRATOR_H_ */
