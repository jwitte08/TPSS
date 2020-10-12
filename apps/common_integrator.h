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
template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct CellOperation
  : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

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
          const auto & grad_u_j = eval_ansatz.shape_gradient(j, q, direction, cell_no);
          const auto & v_i      = eval_test.shape_value(i, q, direction, cell_no);
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
template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct CellOperation
  : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

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
          const auto & u_j = eval_ansatz.shape_value(j, q, direction, cell_no);
          const auto & v_i = eval_test.shape_value(i, q, direction, cell_no);
          const auto & dx  = eval_test.get_JxW(q, direction, cell_no);
          integral += u_j * v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};

// TODO !!! FaceOperation has not been tested
// template<int dim,
//          int fe_degree,
//          int n_q_points_1d,
//          typename Number,
//          int fe_degree_ansatz = fe_degree>
// struct FaceOperation
//   : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
// {
//   using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

//   void
//   operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
//              const typename Base::evaluator_type &        eval_test,
//              typename Base::matrix_type &                 cell_matrix,
//              const int                                    direction,
//              const int                                    cell_no,
//              const int                                    face_no) const
//   {
//     const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
//     const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

//     auto value_on_face = make_vectorized_array<Number>(0.);
//     for(int i = 0; i < n_dofs_test; ++i)
//     {
//       const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
//       for(int j = 0; j < n_dofs_ansatz; ++j)
//       {
//         const auto & u_j = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);

//         value_on_face = v_i * u_j;

//         cell_matrix(i, j) += value_on_face;
//       }
//     }
//   }

//   void
//   operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
//              const typename Base::evaluator_type &        eval_test,
//              typename Base::matrix_type &                 cell_matrix01,
//              typename Base::matrix_type &                 cell_matrix10,
//              const int                                    cell_no0,
//              const int                                    direction) const
//   {
//     AssertDimension(cell_no0, 0);

//     const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
//     const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

//     auto value_on_interface01 = make_vectorized_array<Number>(0.);
//     auto value_on_interface10 = make_vectorized_array<Number>(0.);
//     for(int i = 0; i < n_dofs_test; ++i) // u is ansatz & v is test shape function
//     {
//       const auto & v0_i = eval_test.shape_value_face(i, 1, direction, 0);
//       const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
//       for(int j = 0; j < n_dofs_ansatz; ++j)
//       {
//         const auto & u0_j = eval_ansatz.shape_value_face(j, 1, direction, 0);
//         const auto & u1_j = eval_ansatz.shape_value_face(j, 0, direction, 1);

//         value_on_interface01 = v0_i * u1_j;
//         value_on_interface10 = v1_i * u0_j;

//         cell_matrix01(i, j) += value_on_interface01;
//         cell_matrix10(i, j) += value_on_interface10;
//       }
//     }
//   }
// };

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
              const UpdateFlags          interface_update_flags = UpdateFlags::update_default)
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



/**
 * symgrad(phi)_{d,c} = 0.5 (\partial_d phi_{i;c} + \partial_c phi_{i;d})
 */
template<int dim, typename EvaluatorType>
SymmetricTensor<2, dim>
compute_symgrad_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  SymmetricTensor<2, dim> symgrad_of_phi;
  for(auto d = 0U; d < dim; ++d)
    for(auto c = d; c < dim; ++c)
      symgrad_of_phi[d][c] =
        0.5 * (phi.shape_grad_component(i, q, c)[d] + phi.shape_grad_component(i, q, d)[c]);
  return symgrad_of_phi;
}

template<int dim>
SymmetricTensor<2, dim>
compute_symgrad(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_symgrad_impl<dim, FEValues<dim>>(phi, i, q);
}



/**
 * symgrad(phi)_{d,c} = 0.5 (\partial_d phi_{i;c} + \partial_c phi_{i;d})
 */
template<int dim>
double
compute_divergence(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  double div_of_phi = 0.;
  for(auto d = 0U; d < dim; ++d)
    div_of_phi += phi.shape_grad_component(i, q, d)[d];
  return div_of_phi;
}



/**
 * {{ symgrad(phi_i) }} = 0.5 ({{ \partial_d phi_{i;c} }} + {{ \partial_c phi_{i;d} }})
 */
template<int dim, typename EvaluatorType>
SymmetricTensor<2, dim>
compute_average_symgrad_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  SymmetricTensor<2, dim> av_symgrad_of_phi;
  for(auto d = 0U; d < dim; ++d)
    for(auto c = d; c < dim; ++c)
      av_symgrad_of_phi[d][c] =
        0.5 * (phi.average_gradient(i, q, c)[d] + phi.average_gradient(i, q, d)[c]);
  return av_symgrad_of_phi;
}

template<int dim>
SymmetricTensor<2, dim>
compute_average_symgrad(const FEInterfaceValues<dim> & phi,
                        const unsigned int             i,
                        const unsigned int             q)
{
  return compute_average_symgrad_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}

/**
 * Assuming that the normal n is constant, i.e. the interface flat-sided, the
 * average of the symmetric gradient of the tangential vector field phit = phi -
 * (phi*n) n reads
 *
 *    {{ symgrad(phit_i) }} = {{ symgrad(phi_i) }} (I - n(x)n)
 *
 * where (x) is the dyadic product of two vectors and I the identity matrix.
 */
template<int dim>
Tensor<2, dim>
compute_average_symgrad_tangential(const FEInterfaceValues<dim> & phi,
                                   const unsigned int             i,
                                   const unsigned int             q)
{
  AssertThrow(false, ExcMessage("TODO seems to be wrong !!!"));
  Tensor<2, dim> av_symgrad_of_phi;
  for(auto d = 0U; d < dim; ++d)
    for(auto c = 0U; c < dim; ++c)
      av_symgrad_of_phi[d][c] =
        0.5 * (phi.average_gradient(i, q, c)[d] + phi.average_gradient(i, q, d)[c]);

  const Tensor<1, dim> & n              = phi.normal(q);
  Tensor<2, dim>         n_cross_n      = outer_product(n, n); // n(x)n
  Tensor<2, dim>         normal_contrib = contract<1, 0>(av_symgrad_of_phi, n_cross_n);

  return av_symgrad_of_phi - normal_contrib;
}



/**
 * {{ phi }} = 0.5 * (phi^+ + phi^-)
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vaverage_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> average_phi;
  for(auto c = 0; c < dim; ++c)
    average_phi[c] = phi.average(i, q, c);
  return average_phi;
}

template<int dim>
Tensor<1, dim>
compute_vaverage(const FEInterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vaverage_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}



/**
 * [[ phi ]] = phi^+ - phi^-
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vvalue_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> value_phi;
  for(auto c = 0; c < dim; ++c)
    value_phi[c] = phi.shape_value_component(i, q, c);
  return value_phi;
}

template<int dim>
Tensor<1, dim>
compute_vvalue(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vvalue_impl<dim, FEValues<dim>>(phi, i, q);
}



/**
 * [[ phi ]] = phi^+ - phi^-
 */
template<int dim>
Tensor<1, dim>
compute_vvalue(const FEFaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> value_phi;
  for(auto c = 0; c < dim; ++c)
    value_phi[c] = phi.shape_value_component(i, q, c);
  return value_phi;
}



/**
 * [[ phi ]] = phi^+ - phi^-
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vjump_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> jump_phi;
  for(auto c = 0; c < dim; ++c)
    jump_phi[c] = phi.jump(i, q, c);
  return jump_phi;
}

template<int dim>
Tensor<1, dim>
compute_vjump(const FEInterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vjump_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}



/**
 * The jump of the tangential vector field phit = phi - (phi*n) n reads
 *
 *    [[ phit ]] = [[ phi ]] - ([[ phi ]]*n) n
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vjump_tangential_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  const Tensor<1, dim> & n = phi.normal(q);
  Tensor<1, dim>         jump_phi;
  for(auto c = 0; c < dim; ++c)
    jump_phi[c] = phi.jump(i, q, c);
  return jump_phi - (jump_phi * n) * n;
}

template<int dim>
Tensor<1, dim>
compute_vjump_tangential(const FEInterfaceValues<dim> & phi,
                         const unsigned int             i,
                         const unsigned int             q)
{
  return compute_vjump_tangential_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}



/**
 * [[ phi ]] (x) n
 */
template<int dim>
Tensor<2, dim>
compute_vjump_cross_normal(const FEInterfaceValues<dim> & phi,
                           const unsigned int             i,
                           const unsigned int             q)
{
  const Tensor<1, dim> & n          = phi.normal(q);
  const Tensor<1, dim> & jump_value = compute_vjump(phi, i, q);
  return outer_product(jump_value, n);
}



/**
 *    [[ phit ]] (x) n
 *
 * where n defines the normal and phit the tangential vector field (phi -
 * (phi*n) n). The jump of [[ phit ]] is defined in compute_vjump_tangential.
 */
template<int dim>
Tensor<2, dim>
compute_vjump_cross_normal_tangential(const FEInterfaceValues<dim> & phi,
                                      const unsigned int             i,
                                      const unsigned int             q)
{
  const Tensor<1, dim> & n          = phi.normal(q);
  const Tensor<1, dim> & jump_value = compute_vjump_tangential(phi, i, q);
  return outer_product(jump_value, n);
}



/**
 * [[ phi ]] . n
 */
template<int dim>
double
compute_vjump_dot_normal(const FEInterfaceValues<dim> & phi,
                         const unsigned int             i,
                         const unsigned int             q)
{
  const Tensor<1, dim> & n          = phi.normal(q);
  const Tensor<1, dim> & jump_value = compute_vjump(phi, i, q);
  return jump_value * n;
}



/**
 * Vector curl in 2D:   curl(phi) = (d/dy phi, -d/dx phi)^T
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vcurl_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  AssertDimension(phi.get_fe().n_components(), 1U);
  static_assert(dim == 2, "vector curl is reasonable in 2D.");

  /// See Guido's MFEM script for the vector curl definition.
  Tensor<1, dim> vcurl_of_phi;
  const auto &   grad_of_phi = phi.shape_grad_component(i, q, 0U);
  vcurl_of_phi[0]            = grad_of_phi[1];
  vcurl_of_phi[1]            = -grad_of_phi[0];

  return vcurl_of_phi;
}

template<int dim>
Tensor<1, dim>
compute_vcurl(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vcurl_impl<dim, FEValues<dim>>(phi, i, q);
}

template<int dim>
Tensor<1, dim>
compute_vcurl(const FEFaceValuesBase<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vcurl_impl<dim, FEFaceValuesBase<dim>>(phi, i, q);
}



/**
 * Jump of vector curl in 2D:   [[ curl(phi) ]] = (d/dy [[ phi ]], -d/dx [[ phi ]])^T
 */
template<int dim>
Tensor<1, dim>
compute_jump_vcurl(const FEInterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  static_assert(dim == 2, "vector curl is reasonable in 2D.");

  /// See Guido's MFEM script for the vector curl definition.
  Tensor<1, dim> jump_vcurl_of_phi;
  const auto &   jump_grad_of_phi = phi.jump_gradient(i, q);
  jump_vcurl_of_phi[0]            = jump_grad_of_phi[1];
  jump_vcurl_of_phi[1]            = -jump_grad_of_phi[0];

  return jump_vcurl_of_phi;
}



namespace Mixed
{
template<int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe_test,
              const FiniteElement<dim> & fe_ansatz,
              const unsigned int         n_q_points_1d,
              const UpdateFlags          update_flags_test,
              const UpdateFlags          update_flags_ansatz,
              const UpdateFlags          interface_update_flags_test = UpdateFlags::update_default,
              const UpdateFlags interface_update_flags_ansatz        = UpdateFlags::update_default)
    : fe_values_test(mapping, fe_test, QGauss<dim>(n_q_points_1d), update_flags_test),
      fe_values_ansatz(mapping, fe_ansatz, QGauss<dim>(n_q_points_1d), update_flags_ansatz),
      fe_interface_values_test(mapping,
                               fe_test,
                               QGauss<dim - 1>(n_q_points_1d),
                               interface_update_flags_test),
      fe_interface_values_ansatz(mapping,
                                 fe_ansatz,
                                 QGauss<dim - 1>(n_q_points_1d),
                                 interface_update_flags_ansatz)
  {
  }

  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe,
              const unsigned int         n_q_points_1d,
              const UpdateFlags          update_flags,
              const UpdateFlags          interface_update_flags = UpdateFlags::update_default)
    : ScratchData(mapping,
                  fe,
                  fe,
                  n_q_points_1d,
                  update_flags,
                  update_flags,
                  interface_update_flags,
                  interface_update_flags)
  {
  }

  ScratchData(const ScratchData<dim> & scratch_data)
    : fe_values_test(scratch_data.fe_values_test.get_mapping(),
                     scratch_data.fe_values_test.get_fe(),
                     scratch_data.fe_values_test.get_quadrature(),
                     scratch_data.fe_values_test.get_update_flags()),
      fe_values_ansatz(scratch_data.fe_values_ansatz.get_mapping(),
                       scratch_data.fe_values_ansatz.get_fe(),
                       scratch_data.fe_values_ansatz.get_quadrature(),
                       scratch_data.fe_values_ansatz.get_update_flags()),
      fe_interface_values_test(scratch_data.fe_values_test.get_mapping(),
                               scratch_data.fe_values_test.get_fe(),
                               scratch_data.fe_interface_values_test.get_quadrature(),
                               scratch_data.fe_interface_values_test.get_update_flags()),
      fe_interface_values_ansatz(scratch_data.fe_values_ansatz.get_mapping(),
                                 scratch_data.fe_values_ansatz.get_fe(),
                                 scratch_data.fe_interface_values_ansatz.get_quadrature(),
                                 scratch_data.fe_interface_values_ansatz.get_update_flags())
  {
  }

  FEValues<dim>          fe_values_test;
  FEValues<dim>          fe_values_ansatz;
  FEInterfaceValues<dim> fe_interface_values_test;
  FEInterfaceValues<dim> fe_interface_values_ansatz;
};

struct CopyData
{
  struct FaceData
  {
    FullMatrix<double>                   cell_matrix;
    FullMatrix<double>                   cell_matrix_flipped;
    std::vector<types::global_dof_index> joint_dof_indices_test;
    std::vector<types::global_dof_index> joint_dof_indices_ansatz;
    Vector<double>                       cell_rhs_test;
    Vector<double>                       cell_rhs_ansatz;
  };

  CopyData(const unsigned int n_dofs_per_cell_test,
           const unsigned int n_dofs_per_cell_ansatz,
           const unsigned int level_in = numbers::invalid_unsigned_int)
    : level(level_in),
      cell_matrix(n_dofs_per_cell_test, n_dofs_per_cell_ansatz),
      cell_matrix_flipped(n_dofs_per_cell_ansatz, n_dofs_per_cell_test),
      cell_rhs_test(n_dofs_per_cell_test),
      cell_rhs_ansatz(n_dofs_per_cell_ansatz),
      local_dof_indices_test(n_dofs_per_cell_test),
      local_dof_indices_ansatz(n_dofs_per_cell_ansatz)
  {
  }

  CopyData(const unsigned int n_dofs_per_cell)
    : CopyData(n_dofs_per_cell, n_dofs_per_cell, numbers::invalid_unsigned_int)
  {
  }

  CopyData(const CopyData &) = default;

  unsigned int                         level;
  FullMatrix<double>                   cell_matrix;
  FullMatrix<double>                   cell_matrix_flipped;
  Vector<double>                       cell_rhs_test;
  Vector<double>                       cell_rhs_ansatz;
  std::vector<types::global_dof_index> local_dof_indices_test;
  std::vector<types::global_dof_index> local_dof_indices_ansatz;
  std::vector<FaceData>                face_data;
};

} // namespace Mixed

} // namespace MW

#endif /* APPS_COMMONINTEGRATOR_H_ */
