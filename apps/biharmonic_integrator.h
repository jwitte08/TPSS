/*
 * biharmonic_integrator.h
 *
 *  Created on: Jan 01, 2020
 *      Author: witte
 */

#ifndef APPS_BIHARMONICINTEGRATOR_H_
#define APPS_BIHARMONICINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>


#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"
#include "solvers_and_preconditioners/TPSS/tensor_product_matrix.h"


#include "equation_data.h"
#include "laplace_integrator.h"



namespace Biharmonic
{
using namespace dealii;

/**
 * Linear operators associated to the C0 interior penalty formulation for the
 * biharmonic problem with clamped boundary conditions
 *
 * (MW) MeshWorker
 * (FD) FastDiagonalization
 */
namespace C0IP
{
// /**
//  * Standard (interior) penalty to obtain well-posedness of the Nitsche
//  * method. The penalty is weighted for face integrals at the physical
//  * boundary. The interior penalty is obtained by multiplying with 1/2.
//  */
// template<typename Number>
// Number
// compute_penalty_impl(const int degree, const Number h_left, const Number h_right)
// {
//   const auto one_over_h = (0.5 / h_left) + (0.5 / h_right);
//   const auto gamma      = degree == 0 ? 1 : degree * (degree + 1);
//   return 2.0 * gamma * one_over_h;
// }
using ::Nitsche::compute_penalty_impl;



namespace MW
{
using ::MW::ScratchData;

using ::MW::DoF::CopyData;

using ::MW::compute_vcurl;



template<int dim, bool is_multigrid = false, bool is_stream = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *                              load_function_in,
                   const Function<dim> *                              analytical_solution_in,
                   const LinearAlgebra::distributed::Vector<double> * particular_solution,
                   const EquationData &                               equation_data_in)
    : load_function(load_function_in),
      analytical_solution(analytical_solution_in),
      discrete_solution(particular_solution),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  face_worker(const IteratorType & cell,
              const unsigned int & f,
              const unsigned int & sf,
              const IteratorType & ncell,
              const unsigned int & nf,
              const unsigned int & nsf,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  boundary_worker(const IteratorType & cell,
                  const unsigned int & face_no,
                  ScratchData<dim> &   scratch_data,
                  CopyData &           copy_data) const;

  const Function<dim> *                              load_function;
  const Function<dim> *                              analytical_solution;
  const LinearAlgebra::distributed::Vector<double> * discrete_solution;
  const EquationData                                 equation_data;

  const unsigned int proc_no = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
};

template<int dim, bool is_multigrid, bool is_stream>
void
MatrixIntegrator<dim, is_multigrid, is_stream>::cell_worker(const IteratorType & cell,
                                                            ScratchData<dim> &   scratch_data,
                                                            CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);
  copy_data.cell_data.clear();

  FEValues<dim> & fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  const unsigned int n_dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;

  auto & cell_data = copy_data.cell_data.emplace_back(n_dofs_per_cell);

  cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

  for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < n_dofs_per_cell; ++i)
    {
      const Tensor<2, dim> hessian_i = fe_values.shape_hessian(i, q);

      for(unsigned int j = 0; j < n_dofs_per_cell; ++j)
      {
        const Tensor<2, dim> hessian_j = fe_values.shape_hessian(j, q);

        cell_data.matrix(i, j) += scalar_product(hessian_i,   // nabla^2 phi_i(x)
                                                 hessian_j) * // nabla^2 phi_j(x)
                                  fe_values.JxW(q);           // dx
      }

      if(!is_multigrid)
      {
        if(!is_stream)
        {
          AssertDimension(load_function->n_components, 1U);
          cell_data.rhs(i) += fe_values.shape_value(i, q) *                         // phi_i(x)
                              load_function->value(fe_values.quadrature_point(q)) * // f(x)
                              fe_values.JxW(q);                                     // dx
        }
        else
        {
          AssertDimension(load_function->n_components, dim);
          const auto &   curl_phi_i = compute_vcurl(fe_values, i, q);
          Tensor<1, dim> f;
          for(auto c = 0U; c < dim; ++c)
            f[c] = load_function->value(fe_values.quadrature_point(q), c);

          cell_data.rhs(i) += f * curl_phi_i * fe_values.JxW(q);
        }
      }
    }
  }

  /// For non-zero boundary conditions we compute the negative residual, namely
  /// -(A u0 - b) with u0 being a particular solution that satisfies (strong)
  /// boundary conditions, and store it as right hand side vector. Then, we are
  /// left with finding a homogeneous solution u based on this right hand side.
  if(!is_multigrid)
  {
    Vector<double> u0(cell_data.dof_indices.size());
    for(auto i = 0U; i < u0.size(); ++i)
      u0(i) = (*discrete_solution)(cell_data.dof_indices[i]);
    Vector<double> w0(cell_data.dof_indices.size());
    cell_data.matrix.vmult(w0, u0);
    cell_data.rhs -= w0;
  }
}


template<int dim, bool is_multigrid, bool is_stream>
void
MatrixIntegrator<dim, is_multigrid, is_stream>::face_worker(const IteratorType & cell,
                                                            const unsigned int & f,
                                                            const unsigned int & sf,
                                                            const IteratorType & ncell,
                                                            const unsigned int & nf,
                                                            const unsigned int & nsf,
                                                            ScratchData<dim> &   scratch_data,
                                                            CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
  fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

  const unsigned int   n_interface_dofs = fe_interface_values.n_current_interface_dofs();
  CopyData::FaceData & copy_data_face   = copy_data.face_data.emplace_back(n_interface_dofs);

  copy_data_face.dof_indices = fe_interface_values.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   nh        = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  const auto   fe_degree = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h =
    0.5 * equation_data.ip_factor * C0IP::compute_penalty_impl(fe_degree, h, nh);

  for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
  {
    const auto & n = fe_interface_values.normal(q);

    for(unsigned int i = 0; i < n_interface_dofs; ++i)
    {
      const double av_hessian_i_dot_n_dot_n = (fe_interface_values.average_hessian(i, q) * n * n);
      const double jump_grad_i_dot_n        = (fe_interface_values.jump_gradient(i, q) * n);

      for(unsigned int j = 0; j < n_interface_dofs; ++j)
      {
        const double av_hessian_j_dot_n_dot_n = (fe_interface_values.average_hessian(j, q) * n * n);
        const double jump_grad_j_dot_n        = (fe_interface_values.jump_gradient(j, q) * n);

        copy_data_face.matrix(i, j) += (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n
                                                                   //
                                          * jump_grad_j_dot_n      // [grad u n]
                                        - av_hessian_j_dot_n_dot_n // - {grad^2 u n n
                                                                   //
                                            * jump_grad_i_dot_n    // [grad v n]
                                        +                          // +
                                        gamma_over_h *             // gamma/h
                                          jump_grad_i_dot_n *      // [grad v n]
                                          jump_grad_j_dot_n) *     // [grad u n]
                                       fe_interface_values.JxW(q); // dx
      }
    }
  }

  /// For non-zero boundary conditions we compute the negative residual, namely
  /// -(A u0 - b) with u0 being a particular solution that satisfies (strong)
  /// boundary conditions, and store it as right hand side vector. Then, we are
  /// left with finding a homogeneous solution u based on this right hand side.
  if(!is_multigrid)
  {
    const bool cell_is_at_boundary     = cell->at_boundary();
    const bool neighbor_is_at_boundary = ncell->at_boundary();
    /// Particular solution u0 is only non-zero at the physical boundary.
    if(cell_is_at_boundary || neighbor_is_at_boundary)
    {
      AssertDimension(n_interface_dofs, copy_data_face.dof_indices.size());
      AssertDimension(n_interface_dofs, copy_data_face.rhs.size());
      Vector<double> u0(copy_data_face.dof_indices.size());
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solution)(copy_data_face.dof_indices[i]);
      Vector<double> w0(copy_data_face.dof_indices.size());
      copy_data_face.matrix.vmult(w0, u0);
      copy_data_face.rhs -= w0;
    }
  }
}


template<int dim, bool is_multigrid, bool is_stream>
void
MatrixIntegrator<dim, is_multigrid, is_stream>::boundary_worker(const IteratorType & cell,
                                                                const unsigned int & face_no,
                                                                ScratchData<dim> &   scratch_data,
                                                                CopyData & copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
  fe_interface_values.reinit(cell, face_no);

  const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();

  CopyData::FaceData & copy_data_face = copy_data.face_data.emplace_back(n_interface_dofs);

  copy_data_face.dof_indices = fe_interface_values.get_interface_dof_indices();

  const auto &                quadrature_points = fe_interface_values.get_quadrature_points();
  const std::vector<double> & JxW               = fe_interface_values.get_JxW_values();
  const std::vector<Tensor<1, dim>> & normals   = fe_interface_values.get_normal_vectors();

  std::vector<Tensor<1, dim>> exact_gradients(quadrature_points.size());
  if(!is_multigrid)
  {
    Assert(analytical_solution, ExcMessage("Did you initialize analytical_solution?"));
    analytical_solution->gradient_list(quadrature_points, exact_gradients);
  }

  const auto   h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
  const auto   fe_degree    = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h = equation_data.ip_factor * C0IP::compute_penalty_impl(fe_degree, h, h);

  for(unsigned int q = 0; q < quadrature_points.size(); ++q)
  {
    const auto & n = normals[q];

    for(unsigned int i = 0; i < n_interface_dofs; ++i)
    {
      const double av_hessian_i_dot_n_dot_n = (fe_interface_values.average_hessian(i, q) * n * n);
      const double jump_grad_i_dot_n        = (fe_interface_values.jump_gradient(i, q) * n);

      for(unsigned int j = 0; j < n_interface_dofs; ++j)
      {
        const double av_hessian_j_dot_n_dot_n = (fe_interface_values.average_hessian(j, q) * n * n);
        const double jump_grad_j_dot_n        = (fe_interface_values.jump_gradient(j, q) * n);

        copy_data_face.matrix(i, j) += (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n}
                                          * jump_grad_j_dot_n      //   [grad u n]
                                                                   //
                                        - av_hessian_j_dot_n_dot_n // - {grad^2 u n n}
                                            * jump_grad_i_dot_n    //   [grad v n]
                                                                   //
                                        + gamma_over_h             //  gamma/h
                                            * jump_grad_i_dot_n    // [grad v n]
                                            * jump_grad_j_dot_n    // [grad u n]
                                        ) *
                                       JxW[q]; // dx
      }

      if(!is_multigrid)
        copy_data_face.rhs(i) += (-av_hessian_i_dot_n_dot_n *  // - {grad^2 v n n }
                                    (exact_gradients[q] * n)   //   (grad u_exact . n)
                                  +                            // +
                                  gamma_over_h                 //  gamma/h
                                    * jump_grad_i_dot_n        // [grad v n]
                                    * (exact_gradients[q] * n) // (grad u_exact . n)
                                  ) *
                                 JxW[q]; // dx
    }
  }

  /// For non-zero boundary conditions we compute the negative residual, namely
  /// -(A u0 - b) with u0 being a particular solution that satisfies (strong)
  /// boundary conditions, and store it as right hand side vector. Then, we are
  /// left with finding a homogeneous solution u based on this right hand side.
  if(!is_multigrid)
  {
    Vector<double> u0(n_interface_dofs);
    for(auto i = 0U; i < n_interface_dofs; ++i)
      u0(i) = (*discrete_solution)(copy_data_face.dof_indices[i]);
    Vector<double> w0(u0.size());
    copy_data_face.matrix.vmult(w0, u0);
    copy_data_face.rhs -= w0;
  }
}

} // namespace MW



namespace FD
{
template<int dim, int fe_degree, typename Number = double>
class MatrixIntegrator
{
public:
  using This          = MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransfer<dim, Number>;
  /// TODO static bounds (template parameter)
  using matrix_type             = Tensors::TensorProductMatrix<dim, VectorizedArray<Number>>;
  using matrix_state            = typename matrix_type::State;
  using evaluator_type          = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;
  static constexpr int fe_order = fe_degree + 1;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & subdomain_handler,
                             std::vector<matrix_type> &            local_matrices,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    evaluator_type eval(subdomain_handler); // common evaluator for test + ansatz
    for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
    {
      eval.reinit(patch);

      std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>> rank1_tensors;

      if(equation_data.local_solver_variant == LocalSolverVariant::Bilaplacian)
      {
        rank1_tensors.emplace_back(assemble_mass_tensor(eval));
        rank1_tensors.emplace_back(assemble_bilaplace_tensor(eval));
      }
      else if(equation_data.local_solver_variant == LocalSolverVariant::Exact ||
              equation_data.local_solver_variant == LocalSolverVariant::KSVD)
      {
        /// compute 1D matrices
        const auto mass_matrices      = assemble_mass_tensor(eval);
        const auto bilaplace_matrices = assemble_bilaplace_tensor(eval);
        const auto laplace_matrices   = assemble_laplace_tensor(eval);

        /// store rank1 tensors of separable Kronecker representation
        /// BxMxM + MxBxM + MxMxB
        const auto & BxMxM = [&](const int direction) {
          std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
          for(auto d = 0; d < dim; ++d)
            kronecker_tensor[d] = d == direction ? bilaplace_matrices[direction] : mass_matrices[d];
          return kronecker_tensor;
        };
        for(auto direction = 0; direction < dim; ++direction)
          rank1_tensors.emplace_back(BxMxM(direction));

        /// store rank1 tensors of mixed derivatives
        /// 2(LxLxM + LxMxL + MxLxL)
        const auto & LxLxM = [&](const int direction1, const int direction2) {
          std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
          for(auto d = 0; d < dim; ++d)
            kronecker_tensor[d] =
              (d == direction1 || d == direction2) ? laplace_matrices[d] : mass_matrices[d];
          return kronecker_tensor;
        };
        for(auto direction1 = 0; direction1 < dim; ++direction1)
          for(auto direction2 = 0; direction2 < dim; ++direction2)
            if(direction1 != direction2)
              rank1_tensors.emplace_back(LxLxM(direction1, direction2));

        AssertDimension(rank1_tensors.size(), 2 * dim);
      }

      /// submit vector of rank1 Kronecker tensors
      switch(equation_data.local_solver_variant)
      {
        case LocalSolverVariant::Exact:
          local_matrices[patch].reinit(rank1_tensors);
          break;
        case LocalSolverVariant::Bilaplacian:
          local_matrices[patch].reinit(rank1_tensors, matrix_state::separable);
          break;
        case LocalSolverVariant::KSVD:
        {
          std::array<std::size_t, dim> rows, columns;
          for(auto d = 0U; d < dim; ++d)
          {
            const auto & A_d = rank1_tensors.front()[d];
            rows[d]          = A_d.size(0);
            columns[d]       = A_d.size(1);
          }


          const auto ksvd_rank = *(equation_data.ksvd_tensor_indices.rbegin()) + 1;
          AssertIndexRange(ksvd_rank, rank1_tensors.size() + 1);
          auto ksvd_tensors =
            Tensors::make_zero_rank1_tensors<dim, VectorizedArray<Number>>(ksvd_rank,
                                                                           rows,
                                                                           columns);
          const auto & ksvd_singular_values =
            compute_ksvd(rank1_tensors, ksvd_tensors, equation_data.n_lanczos_iterations);

          std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>> approximation;
          for(auto i = 0U; i < ksvd_tensors.size(); ++i)
            if(equation_data.ksvd_tensor_indices.find(i) !=
               equation_data.ksvd_tensor_indices.cend())
              approximation.emplace_back(ksvd_tensors[i]);

          AssertDimension(equation_data.ksvd_tensor_indices.size(), approximation.size());

          if(approximation.size() == 1U)
          {
            local_matrices[patch].reinit(approximation, matrix_state::rankone);
          }

          else if(approximation.size() == 2U)
          {
            /// first tensor must contain s.p.d. matrices ("mass matrices")
            typename matrix_type::AdditionalData additional_data;
            additional_data.state = matrix_state::ranktwo;

            local_matrices[patch].reinit(approximation, additional_data);

            const auto & tensor_of_eigenvalues = local_matrices[patch].get_eigenvalue_tensor();
            const auto   eigenvalues_ksvd1 =
              Tensors::kronecker_product<dim, VectorizedArray<Number>>(tensor_of_eigenvalues);

            /// if the rank-2 KSVD isn't positive definite we scale the second
            /// tensor of matrices by a factor \alpha (with 0 < \alpha < 1),
            /// thus obtaing an approximation that is better than the best
            /// rank-1 approximation but worse than the best rank-2
            /// approximation. \alpha is computed at negligible costs due to the
            /// specific eigendecomposition with tensor structure
            if(equation_data.ksvd_tensor_indices == std::set<unsigned int>{0U, 1U})
            {
              VectorizedArray<Number> alpha(1.);
              for(auto lane = 0U; lane < VectorizedArray<Number>::size(); ++lane)
              {
                // std::cout << "eigenvalues of KSVD[1]:\n"
                //           << vector_to_string(alignedvector_to_vector(eigenvalues_ksvd1, lane))
                //           << std::endl;
                const auto   min_elem   = std::min_element(eigenvalues_ksvd1.begin(),
                                                       eigenvalues_ksvd1.end(),
                                                       [&](const auto & lhs, const auto & rhs) {
                                                         return lhs[lane] < rhs[lane];
                                                       });
                const Number lambda_min = (*min_elem)[lane];

                /// \alpha = -1 / ((1 + \epsilon) * \lambda_{min})
                if(lambda_min < -0.99) // KSVD isn't positive definite
                  alpha[lane] /= -(1. + equation_data.addition_to_min_eigenvalue) * lambda_min;
                if(alpha[lane] > 1.)
                  alpha[lane] = 0.99;
              }

              // std::cout << "alpha: " << varray_to_string(alpha) << std::endl;
              Tensors::scaling<dim>(alpha, approximation.at(1U));
              local_matrices[patch].reinit(approximation, additional_data);
            }
          }

          else
          {
            local_matrices[patch].reinit(approximation);
          }

          // /// DEBUG
          // const auto & [is_posdef, eigenvalues] =
          //   LinAlg::is_positive_definite_impl(local_matrices[patch].as_inverse_table());
          // if(!is_posdef.all())
          // {
          //   std::cout << "patch: " << patch << " " << is_posdef << std::endl;
          //   for(auto lane = 0U; lane < VectorizedArray<Number>::size(); ++lane)
          //     std::cout << "singular values of KSVD:\n"
          //               << vector_to_string(alignedvector_to_vector(ksvd_singular_values, lane))
          //               << "eigenvalues of inverse:\n"
          //               << vector_to_string(alignedvector_to_vector(eigenvalues, lane))
          //               << std::endl;
          // }

          break;
        }
        default:
          AssertThrow(false, ExcMessage("Local solver isn't supported."));
      }
    }
  }

  template<bool c0ip = true>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  assemble_bilaplace_tensor(evaluator_type & eval) const
  {
    const auto cell_bilaplace = [](const evaluator_type &              eval_u,
                                   const evaluator_type &              eval_v,
                                   Table<2, VectorizedArray<Number>> & cell_matrix,
                                   const int                           direction,
                                   const int                           cell_no) {
      auto integral = make_vectorized_array<Number>(0.);
      for(int j = 0; j < fe_order; ++j)
        for(int i = 0; i < fe_order; ++i)
        {
          integral = 0.;
          for(unsigned int q = 0; q < evaluator_type::n_q_points_1d_static; ++q)
          {
            const auto & hess_ui = eval_u.shape_hessian(j, q, direction, cell_no);
            const auto & hess_vj = eval_v.shape_hessian(i, q, direction, cell_no);
            const auto & dx      = eval_v.get_JxW(q, direction, cell_no);
            integral += hess_ui * hess_vj * dx;
          }
          cell_matrix(i, j) += integral;
        }
    };

    const auto compute_penalty =
      [&](const evaluator_type & eval, const int direction, const int cell_no, const int ncell_no) {
        const auto h  = eval.get_h(direction, cell_no);
        const auto nh = eval.get_h(direction, ncell_no);
        return equation_data.ip_factor * compute_penalty_impl(fe_degree, h, nh);
      };

    const auto face_nitsche = [&](const evaluator_type &              eval_u,
                                  const evaluator_type &              eval_v,
                                  Table<2, VectorizedArray<Number>> & cell_matrix,
                                  const int                           direction,
                                  const int                           cell_no,
                                  const int                           face_no) {
      const auto normal         = eval_v.get_normal(face_no);
      const auto sqnormal       = normal * normal;
      const auto average_factor = eval_v.get_average_factor(direction, cell_no, face_no);
      const auto penalty = average_factor * compute_penalty(eval_v, direction, cell_no, cell_no);

      auto value_on_face = make_vectorized_array<Number>(0.);
      for(int i = 0; i < fe_degree + 1; ++i)
      {
        const auto & grad_vi = eval_v.shape_gradient_face(i, face_no, direction, cell_no);
        const auto & hess_vi = eval_v.shape_hessian_face(i, face_no, direction, cell_no);
        for(int j = 0; j < fe_degree + 1; ++j)
        {
          const auto & grad_uj = eval_u.shape_gradient_face(j, face_no, direction, cell_no);
          const auto & hess_uj = eval_u.shape_hessian_face(j, face_no, direction, cell_no);

          value_on_face = penalty * grad_vi * normal * grad_uj * normal;
          value_on_face -= (grad_vi * normal) * (average_factor * hess_uj * sqnormal);
          value_on_face -= (grad_uj * normal) * (average_factor * hess_vi * sqnormal);

          cell_matrix(i, j) += value_on_face;
        }
      }
    };

    const auto interface_nitsche = [&](const evaluator_type &              eval_u,
                                       const evaluator_type &              eval_v,
                                       Table<2, VectorizedArray<Number>> & cell_matrix01,
                                       Table<2, VectorizedArray<Number>> & cell_matrix10,
                                       const int                           cell_no0, // left cell
                                       const int                           direction) {
      const auto cell_no1 = cell_no0 + 1;                 // right cell
      const auto face_no0 = 1;                            // interface seen from left cell
      const auto face_no1 = 0;                            // interface seen from right cell
      AssertDimension(cell_no0, 0);                       // vertex patch has one interface
      const auto normal0   = eval_v.get_normal(face_no0); // on cell 0
      const auto normal1   = eval_v.get_normal(face_no1); // on cell 1
      const auto sqnormal0 = normal0 * normal0;
      const auto sqnormal1 = normal1 * normal1;
      const auto penalty   = 0.5 * compute_penalty(eval_v, direction, cell_no0, cell_no1);

      auto value_on_interface01{make_vectorized_array<Number>(0.)};
      auto value_on_interface10{make_vectorized_array<Number>(0.)};
      for(int i = 0; i < fe_degree + 1; ++i)
      {
        const auto & hess_v0i = eval_v.shape_hessian_face(i, face_no0, direction, cell_no0);
        const auto & grad_v0i = eval_v.shape_gradient_face(i, face_no0, direction, cell_no0);
        const auto & hess_v1i = eval_v.shape_hessian_face(i, face_no1, direction, cell_no1);
        const auto & grad_v1i = eval_v.shape_gradient_face(i, face_no1, direction, cell_no1);
        for(int j = 0; j < fe_degree + 1; ++j)
        {
          const auto & hess_u0j = eval_u.shape_hessian_face(j, face_no0, direction, cell_no0);
          const auto & grad_u0j = eval_u.shape_gradient_face(j, face_no0, direction, cell_no0);
          const auto & hess_u1j = eval_u.shape_hessian_face(j, face_no1, direction, cell_no1);
          const auto & grad_u1j = eval_u.shape_gradient_face(j, face_no1, direction, cell_no1);

          /// consistency + symmetry
          value_on_interface01 = penalty * grad_v0i * normal0 * grad_u1j * normal1;
          value_on_interface01 -= (grad_v0i * normal0) * (0.5 * hess_u1j * sqnormal1);
          value_on_interface01 -= (grad_u1j * normal1) * (0.5 * hess_v0i * sqnormal0);

          value_on_interface10 = penalty * grad_v1i * normal1 * grad_u0j * normal0;
          value_on_interface10 -= (grad_v1i * normal1) * (0.5 * hess_u0j * sqnormal0);
          value_on_interface10 -= (grad_u0j * normal0) * (0.5 * hess_v1i * sqnormal1);

          cell_matrix01(i, j) += value_on_interface01;
          cell_matrix10(i, j) += value_on_interface10;
        }
      }
    };

    if constexpr(c0ip)
      return eval.patch_action(cell_bilaplace, face_nitsche, interface_nitsche);
    (void)face_nitsche, (void)interface_nitsche;
    return eval.patch_action(cell_bilaplace);
  }

  template<bool sipg = false>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  assemble_laplace_tensor(evaluator_type & eval) const
  {
    if constexpr(sipg)
    {
      Laplace::DG::FD::CellLaplace<evaluator_type> cell_laplace;
      Laplace::EquationData                        laplace_data;
      laplace_data.ip_factor = equation_data.ip_factor;
      Laplace::DG::FD::FaceLaplace<evaluator_type> nitsche(laplace_data);
      return eval.patch_action(cell_laplace, nitsche, nitsche);
    }
    return eval.patch_action(Laplace::DG::FD::CellLaplace<evaluator_type>{});
  }

  std::array<Table<2, VectorizedArray<Number>>, dim>
  assemble_mass_tensor(evaluator_type & eval) const
  {
    return eval.patch_action(Laplace::DG::FD::CellMass<evaluator_type>{});
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};

} // end namespace FD

} // end namespace C0IP



namespace Pressure
{
using InterfaceId = typename std::pair<CellId, CellId>;

template<int dim>
struct InterfaceHandler
{
  void
  reinit(const DoFHandler<dim> & dof_handler_velocity)
  {
    const auto & triangulation = dof_handler_velocity.get_triangulation();

    std::set<CellId> marked_cells;

    /// Choose a first cell with fixed constant mode.
    const auto & first_cell = triangulation.begin_active();
    fixed_cell_id           = first_cell->id();
    interface_ids.emplace_back(first_cell->id(), first_cell->id());
    marked_cells.emplace(first_cell->id());

    /// For all remaining cells set one and only one "inflow"
    /// interface. "Inflow" means that the neighboring cell has already a
    /// "fixed" constant mode.
    while(marked_cells.size() < triangulation.n_global_active_cells())
      for(auto & cell : triangulation.active_cell_iterators())
      {
        const bool cell_is_marked = marked_cells.find(cell->id()) != marked_cells.cend();

        for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          const bool there_is_no_neighbor = cell->neighbor_index(face_no) == -1;
          if(there_is_no_neighbor)
            continue;

          const auto & ncell           = cell->neighbor(face_no);
          const bool   ncell_is_marked = marked_cells.find(ncell->id()) != marked_cells.cend();

          if(cell_is_marked && ncell_is_marked)
            continue;

          else if(cell_is_marked && !ncell_is_marked) // cell -> ncell
          {
            interface_ids.emplace_back(cell->id(), ncell->id());
            const auto & [dummy_, has_been_inserted] = marked_cells.emplace(ncell->id());
            (void)dummy_, (void)has_been_inserted;
            Assert(has_been_inserted, ExcMessage("Insertion failed."));
          }

          else if(!cell_is_marked && ncell_is_marked) // ncell -> cell
          {
            interface_ids.emplace_back(ncell->id(), cell->id());
            const auto & [dummy_, has_been_inserted] = marked_cells.emplace(cell->id());
            (void)dummy_, (void)has_been_inserted;
            Assert(has_been_inserted, ExcMessage("Insertion failed."));
          }
        }
      } // end loop over cells

    /// The second member of each interface id determines the new cell with
    /// "inflow" interface. We use CellIds ordering to sort the interface ids.
    std::sort(interface_ids.begin(), interface_ids.end());

    /// TODO find an explicit way to convert CellId into a unique integer
    std::copy(marked_cells.cbegin(), marked_cells.cend(), std::back_inserter(cell_ids));
    /// this std::sort is not needed as the set is ordered !!!
    // std::sort(cell_ids.begin(), cell_ids.end());

    AssertDimension(cell_ids.size(), triangulation.n_global_active_cells());
    AssertDimension(cell_ids.size(), interface_ids.size());

    // // DEBUG
    // std::cout << "marked cells: " << std::endl;
    // for(const auto & cell_id : cell_ids)
    //   std::cout << cell_id << " ";
    // std::cout << std::endl;
    // std::cout << "marked interfaces: " << std::endl;
    // for(const auto & interface_id : interface_ids)
    //   std::cout << interface_id << std::endl;
  }


  std::vector<InterfaceId>::const_iterator
  get_interface_iterator(const InterfaceId & id) const
  {
    Assert(!interface_ids.empty(), ExcMessage("Have you initialized this interface handler?"));
    const bool is_contained = std::binary_search(interface_ids.cbegin(), interface_ids.cend(), id);
    if(is_contained)
      return std::lower_bound(interface_ids.cbegin(), interface_ids.cend(), id);
    const InterfaceId flipped_id = {id.second, id.first};
    const bool        is_flipped =
      std::binary_search(interface_ids.cbegin(), interface_ids.cend(), flipped_id);
    if(is_flipped)
      return std::lower_bound(interface_ids.cbegin(), interface_ids.cend(), flipped_id);
    return interface_ids.cend();
  }

  unsigned int
  get_interface_index(const InterfaceId & id) const
  {
    const auto interface_iterator = get_interface_iterator(id);
    const bool id_isnt_contained  = interface_ids.cend() == interface_iterator;
    if(id_isnt_contained)
      return numbers::invalid_unsigned_int;
    const auto index = std::distance(interface_ids.cbegin(), interface_iterator);
    AssertIndexRange(index, interface_ids.size());
    return index;
  }

  unsigned int
  get_cell_index(const CellId & id) const
  {
    Assert(!cell_ids.empty(), ExcMessage("Have you initialized this interface handler?"));
    const auto it    = std::lower_bound(cell_ids.cbegin(), cell_ids.cend(), id);
    const auto index = std::distance(cell_ids.cbegin(), it);
    AssertIndexRange(index, cell_ids.size());
    return index;
  }

  std::pair<unsigned int, unsigned int>
  get_cell_index_pair(const InterfaceId & id) const
  {
    return {get_cell_index(id.first), get_cell_index(id.second)};
  }

  unsigned int
  get_fixed_cell_index() const
  {
    const auto index = get_cell_index(fixed_cell_id);
    Assert(index != numbers::invalid_unsigned_int, ExcMessage("Fixed cell isnt contained!?"));
    return index;
  }

  unsigned int
  get_fixed_interface_index() const
  {
    const auto index = get_interface_index({fixed_cell_id, fixed_cell_id});
    Assert(index != numbers::invalid_unsigned_int, ExcMessage("Fixed cell isnt contained!?"));
    return index;
  }

  unsigned int
  n_interfaces() const
  {
    return interface_ids.size();
  }

  std::vector<InterfaceId> interface_ids; // sorted
  std::vector<CellId>      cell_ids;      // sorted
  CellId                   fixed_cell_id;
};



namespace Interface
{
namespace MW
{
using ::MW::TestFunction::ScratchData;

using ::MW::Mixed::CopyData;



template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;


  /// This integrator assumes that the coefficients of the discrete pressure
  /// associated to the constant modes are set to zero!
  MatrixIntegrator(const Function<dim> *         load_function_in,
                   const Vector<double> *        stream_function_solution,
                   const Vector<double> *        pressure_solution,
                   const InterfaceHandler<dim> * interface_handler_in,
                   const Stokes::EquationData &  equation_data_in)
    : load_function(load_function_in),
      discrete_velocity(stream_function_solution),
      discrete_pressure(pressure_solution),
      interface_handler(interface_handler_in),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  face_worker(const IteratorType & cell,
              const IteratorType & cellP,
              const unsigned int & face_no,
              const unsigned int & sface_no,
              const IteratorType & ncell,
              const IteratorType & ncellP,
              const unsigned int & nface_no,
              const unsigned int & nsface_no,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  const Function<dim> *         load_function;
  const Vector<double> *        discrete_velocity;
  const Vector<double> *        discrete_pressure;
  const InterfaceHandler<dim> * interface_handler;
  const Stokes::EquationData    equation_data;
};



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cell,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_rhs_test.size(), GeometryInfo<dim>::faces_per_cell);
  copy_data.cell_rhs_test = 0.;

  for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
  {
    const bool this_is_no_interface = cell->neighbor_index(face_no) == -1;
    if(this_is_no_interface)
      continue;

    const auto ncell = cell->neighbor(face_no);

    InterfaceId        interface_id{cell->id(), ncell->id()};
    const unsigned int interface_index       = interface_handler->get_interface_index(interface_id);
    const bool this_interface_isnt_contained = interface_index == numbers::invalid_unsigned_int;
    if(this_interface_isnt_contained)
      continue;

    copy_data.local_dof_indices_test[face_no] = interface_index;
  }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker(const IteratorType & cell,
                                                 const IteratorType & cellP,
                                                 const unsigned int & face_no,
                                                 const unsigned int & sface_no,
                                                 const IteratorType & ncell,
                                                 const IteratorType & ncellP,
                                                 const unsigned int & nface_no,
                                                 const unsigned int & nsface_no,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  InterfaceId        interface_id{cellP->id(), ncellP->id()};
  const unsigned int interface_index       = interface_handler->get_interface_index(interface_id);
  const bool this_interface_isnt_contained = interface_index == numbers::invalid_unsigned_int;
  if(this_interface_isnt_contained)
    return;

  const auto cell_index  = interface_handler->get_cell_index(cell->id());
  const auto ncell_index = interface_handler->get_cell_index(ncell->id());

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  FEInterfaceValues<dim> & phiP = scratch_data.fe_interface_values_ansatz;
  phiP.reinit(cellP, face_no, sface_no, ncellP, nface_no, nsface_no);

  copy_data_face.joint_dof_indices_ansatz.resize(2, 0U);
  copy_data_face.joint_dof_indices_ansatz[0U] = cell_index;
  copy_data_face.joint_dof_indices_ansatz[1U] = ncell_index;

  std::vector<std::array<unsigned int, 2>> testfunc_indices;
  testfunc_indices.push_back({face_no, nface_no});

  auto & phiV = scratch_data.test_interface_values;
  phiV.reinit(cell, face_no, sface_no, ncell, nface_no, nsface_no, testfunc_indices);

  const unsigned int n_interface_dofs_p  = phiP.n_current_interface_dofs();
  const auto &       joint_dof_indices_p = phiP.get_interface_dof_indices();

  copy_data_face.joint_dof_indices_test.resize(GeometryInfo<dim>::faces_per_cell, 0U);
  copy_data_face.joint_dof_indices_test[face_no] = interface_index;

  copy_data_face.cell_matrix.reinit(GeometryInfo<dim>::faces_per_cell, 2U);
  copy_data_face.cell_rhs_test.reinit(GeometryInfo<dim>::faces_per_cell);

  std::vector<double> joint_dof_values_p;
  for(const auto i : joint_dof_indices_p)
    joint_dof_values_p.push_back((*discrete_pressure)(i));

  /// As long as the constant pressure mode is set to zero looping over the
  /// whole set of dof indices is valid.
  const auto & compute_jump_pn = [&](const unsigned int q) {
    double               value = 0.;
    const Tensor<1, dim> n     = phiP.normal(q);
    for(auto j = 0U; j < n_interface_dofs_p; ++j) // constant mode is zero !
      value += joint_dof_values_p[j] * phiP.jump(j, q);
    return value * n;
  };

  double alpha_left  = 0.;
  double alpha_right = 0.;
  double pn_dot_v    = 0.;
  for(unsigned int q = 0; q < phiP.n_quadrature_points; ++q)
  {
    const auto &           dx      = phiV.JxW(q);
    const Tensor<1, dim> & v_face  = compute_vaverage(phiV, 0, q);
    const Tensor<1, dim> & jump_pn = compute_jump_pn(q);
    const Tensor<1, dim> & n_left  = phiP.normal(q);
    const Tensor<1, dim> & n_right = -n_left;

    pn_dot_v += jump_pn * v_face * dx;

    alpha_left += -1. * n_left * v_face * dx;
    alpha_right += -1. * n_right * v_face * dx;
  }

  copy_data_face.cell_rhs_test[face_no] += pn_dot_v;
  copy_data_face.cell_matrix(face_no, 0U) = alpha_left;
  copy_data_face.cell_matrix(face_no, 1U) = alpha_right;
}

} // namespace MW

} // namespace Interface

} // namespace Pressure

} // end namespace Biharmonic



#endif /* APPS_BIHARMONICINTEGRATOR_H_ */
