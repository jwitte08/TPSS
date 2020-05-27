/*
 * common_integrator.h
 *
 *  Created on: May 26, 2019
 *      Author: witte
 */

#ifndef APPS_COMMONINTEGRATOR_H_
#define APPS_COMMONINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"



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
  operator()(const evaluator_type &        eval_ansatz,
             const ansatz_evaluator_type & eval_test,
             matrix_type &                 cell_matrix,
             const int                     direction,
             const int                     cell_no) const
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
    int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    int n_q_points    = eval_test.n_q_points_1d(direction);

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
    int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    int n_q_points    = eval_test.n_q_points_1d(direction);

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
    int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    int n_q_points    = eval_test.n_q_points_1d(direction);

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
} // namespace L2
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
