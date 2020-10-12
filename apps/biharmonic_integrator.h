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

using ::MW::CopyData;

using ::MW::compute_vcurl;



template<int dim, bool is_multigrid = false, bool is_stream = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *  load_function_in,
                   const Function<dim> *  analytical_solution_in,
                   const Vector<double> * particular_solution,
                   const EquationData &   equation_data_in)
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

  const Function<dim> *  load_function;
  const Function<dim> *  analytical_solution;
  const Vector<double> * discrete_solution;
  const EquationData     equation_data;
};

template<int dim, bool is_multigrid, bool is_stream>
void
MatrixIntegrator<dim, is_multigrid, is_stream>::cell_worker(const IteratorType & cell,
                                                            ScratchData<dim> &   scratch_data,
                                                            CopyData &           copy_data) const
{
  copy_data.cell_matrix = 0.;
  copy_data.cell_rhs    = 0.;

  FEValues<dim> & fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);

  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;

  for(unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points; ++qpoint)
  {
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const Tensor<2, dim> hessian_i = fe_values.shape_hessian(i, qpoint);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const Tensor<2, dim> hessian_j = fe_values.shape_hessian(j, qpoint);

        copy_data.cell_matrix(i, j) += scalar_product(hessian_i,   // nabla^2 phi_i(x)
                                                      hessian_j) * // nabla^2 phi_j(x)
                                       fe_values.JxW(qpoint);      // dx
      }

      if(!is_multigrid)
      {
        if(!is_stream)
        {
          AssertDimension(load_function->n_components, 1U);
          copy_data.cell_rhs(i) +=
            fe_values.shape_value(i, qpoint) *                         // phi_i(x)
            load_function->value(fe_values.quadrature_point(qpoint)) * // f(x)
            fe_values.JxW(qpoint);                                     // dx
        }
        else
        {
          AssertDimension(load_function->n_components, dim);
          const auto &   curl_phi_i = compute_vcurl(fe_values, i, qpoint);
          Tensor<1, dim> f;
          for(auto c = 0U; c < dim; ++c)
            f[c] = load_function->value(fe_values.quadrature_point(qpoint), c);

          copy_data.cell_rhs(i) += f * curl_phi_i * fe_values.JxW(qpoint);
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
    Vector<double> u0(copy_data.local_dof_indices.size());
    for(auto i = 0U; i < u0.size(); ++i)
      u0(i) = (*discrete_solution)(copy_data.local_dof_indices[i]);
    Vector<double> w0(copy_data.local_dof_indices.size());
    copy_data.cell_matrix.vmult(w0, u0);
    copy_data.cell_rhs -= w0;
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

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

  const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();
  copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

  const auto h  = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto nh = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  // !!!
  const auto fe_degree = scratch_data.fe_values.get_fe().degree - (is_stream ? 1U : 0U);
  // const auto   fe_degree = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h =
    0.5 * equation_data.ip_factor * C0IP::compute_penalty_impl(fe_degree, h, nh);
  std::cout << "bi:face:gamma(: " << gamma_over_h << " " << fe_degree << " " << h << std::endl;

  for(unsigned int qpoint = 0; qpoint < fe_interface_values.n_quadrature_points; ++qpoint)
  {
    const auto & n = fe_interface_values.normal(qpoint);

    for(unsigned int i = 0; i < n_interface_dofs; ++i)
    {
      const double av_hessian_i_dot_n_dot_n =
        (fe_interface_values.average_hessian(i, qpoint) * n * n);
      const double jump_grad_i_dot_n = (fe_interface_values.jump_gradient(i, qpoint) * n);

      for(unsigned int j = 0; j < n_interface_dofs; ++j)
      {
        const double av_hessian_j_dot_n_dot_n =
          (fe_interface_values.average_hessian(j, qpoint) * n * n);
        const double jump_grad_j_dot_n = (fe_interface_values.jump_gradient(j, qpoint) * n);

        copy_data_face.cell_matrix(i, j) += (-av_hessian_i_dot_n_dot_n       // - {grad^2 v n n
                                                                             //
                                               * jump_grad_j_dot_n           // [grad u n]
                                             - av_hessian_j_dot_n_dot_n      // - {grad^2 u n n
                                                                             //
                                                 * jump_grad_i_dot_n         // [grad v n]
                                             +                               // +
                                             gamma_over_h *                  // gamma/h
                                               jump_grad_i_dot_n *           // [grad v n]
                                               jump_grad_j_dot_n) *          // [grad u n]
                                            fe_interface_values.JxW(qpoint); // dx
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
      AssertDimension(n_interface_dofs, copy_data_face.joint_dof_indices.size());
      Vector<double> u0(copy_data_face.joint_dof_indices.size());
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solution)(copy_data_face.joint_dof_indices[i]);
      copy_data_face.cell_rhs.reinit(u0.size());
      copy_data_face.cell_matrix.vmult(copy_data_face.cell_rhs, u0);
      copy_data_face.cell_rhs *= -1.;
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
  const auto & q_points = fe_interface_values.get_quadrature_points();

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  const unsigned int n_dofs        = fe_interface_values.n_current_interface_dofs();
  copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

  copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

  const std::vector<double> &         JxW     = fe_interface_values.get_JxW_values();
  const std::vector<Tensor<1, dim>> & normals = fe_interface_values.get_normal_vectors();

  std::vector<Tensor<1, dim>> exact_gradients(q_points.size());
  analytical_solution->gradient_list(q_points, exact_gradients);

  const auto h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
  // !!!
  const auto fe_degree = scratch_data.fe_values.get_fe().degree - (is_stream ? 1U : 0U);
  // const auto   fe_degree = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h = equation_data.ip_factor * C0IP::compute_penalty_impl(fe_degree, h, h);
  std::cout << "bi:bdry:gamma: " << gamma_over_h << " " << fe_degree << " " << h << std::endl;

  for(unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
  {
    const auto & n = normals[qpoint];

    for(unsigned int i = 0; i < n_dofs; ++i)
    {
      const double av_hessian_i_dot_n_dot_n =
        (fe_interface_values.average_hessian(i, qpoint) * n * n);
      const double jump_grad_i_dot_n = (fe_interface_values.jump_gradient(i, qpoint) * n);

      for(unsigned int j = 0; j < n_dofs; ++j)
      {
        const double av_hessian_j_dot_n_dot_n =
          (fe_interface_values.average_hessian(j, qpoint) * n * n);
        const double jump_grad_j_dot_n = (fe_interface_values.jump_gradient(j, qpoint) * n);

        copy_data_face.cell_matrix(i, j) += (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n}
                                               * jump_grad_j_dot_n      //   [grad u n]
                                                                        //
                                             - av_hessian_j_dot_n_dot_n // - {grad^2 u n n}
                                                 * jump_grad_i_dot_n    //   [grad v n]
                                                                        //
                                             + gamma_over_h             //  gamma/h
                                                 * jump_grad_i_dot_n    // [grad v n]
                                                 * jump_grad_j_dot_n    // [grad u n]
                                             ) *
                                            JxW[qpoint]; // dx
      }

      if(!is_multigrid)
        copy_data.cell_rhs(i) += (-av_hessian_i_dot_n_dot_n *       // - {grad^2 v n n }
                                    (exact_gradients[qpoint] * n)   //   (grad u_exact . n)
                                  +                                 // +
                                  gamma_over_h                      //  gamma/h
                                    * jump_grad_i_dot_n             // [grad v n]
                                    * (exact_gradients[qpoint] * n) // (grad u_exact . n)
                                  ) *
                                 JxW[qpoint]; // dx
    }
  }

  /// For non-zero boundary conditions we compute the negative residual, namely
  /// -(A u0 - b) with u0 being a particular solution that satisfies (strong)
  /// boundary conditions, and store it as right hand side vector. Then, we are
  /// left with finding a homogeneous solution u based on this right hand side.
  if(!is_multigrid)
  {
    AssertDimension(n_dofs, copy_data.cell_rhs.size());
    AssertDimension(n_dofs, copy_data.local_dof_indices.size());
    Vector<double> u0(n_dofs);
    for(auto i = 0U; i < n_dofs; ++i)
      u0(i) = (*discrete_solution)(copy_data.local_dof_indices[i]);
    Vector<double> w0(u0.size());
    copy_data_face.cell_matrix.vmult(w0, u0);
    copy_data.cell_rhs -= w0;
  }
}

} // namespace MW



namespace FD
{
template<int dim, int fe_degree, typename Number = double>
class MatrixIntegrator
{
public:
  using This                    = MatrixIntegrator<dim, fe_degree, Number>;
  using value_type              = Number;
  using transfer_type           = typename TPSS::PatchTransfer<dim, Number>;
  using matrix_type             = Tensors::TensorProductMatrix<dim, VectorizedArray<Number>>;
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
      /// compute 1D matrices
      eval.reinit(patch);
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
      std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>> rank1_tensors;
      for(auto direction = 0; direction < dim; ++direction)
        rank1_tensors.emplace_back(BxMxM(direction));

      /// store rank1 tensors of mixed derivatives
      /// 2(LxLxM + LxMxL + MxLxL)
      if(equation_data.local_solver_variant == LocalSolverVariant::Exact)
      {
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
      }

      /// submit vector of rank1 Kronecker tensors
      local_matrices[patch].reinit(rank1_tensors);
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
    const auto   n_cells       = triangulation.n_global_active_cells();

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
            (void)dummy_;
            Assert(has_been_inserted, ExcMessage("Insertion failed."));
          }

          else if(!cell_is_marked && ncell_is_marked) // ncell -> cell
          {
            interface_ids.emplace_back(ncell->id(), cell->id());
            const auto & [dummy_, has_been_inserted] = marked_cells.emplace(cell->id());
            (void)dummy_;
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

    AssertDimension(cell_ids.size(), n_cells);
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



template<int dim>
struct TestFunctionInterfaceValues
{
  static_assert(dim == 2, "Implemented for 2D only.");

  TestFunctionInterfaceValues(const FEValues<dim> &          fe_values_in,
                              const FEInterfaceValues<dim> & fe_interface_values_in,
                              const FullMatrix<double> &     shape_to_test_functions_left_in,
                              const FullMatrix<double> &     shape_to_test_functions_right_in)
    : fe_values(fe_values_in.get_mapping(),
                fe_values_in.get_fe(),
                fe_values_in.get_quadrature(),
                fe_values_in.get_update_flags()),
      fe_interface_values(fe_values_in.get_mapping(),
                          fe_values_in.get_fe(),
                          fe_interface_values_in.get_quadrature(),
                          fe_interface_values_in.get_update_flags()),
      shape_to_test_functions_left(shape_to_test_functions_left_in),
      shape_to_test_functions_right(shape_to_test_functions_right_in),
      n_quadrature_points(fe_interface_values_in.n_quadrature_points)
  {
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType &                         cell,
         const unsigned int                               face_no,
         const unsigned int                               subface_no,
         const CellIteratorType &                         ncell,
         const unsigned int                               nface_no,
         const unsigned int                               nsubface_no,
         const std::vector<std::array<unsigned int, 2>> & joint_to_cell_dof_indices_in)
  {
    fe_interface_values.reinit(cell, face_no, subface_no, ncell, nface_no, nsubface_no);
    joint_to_cell_dof_indices = joint_to_cell_dof_indices_in;
    std::cout << "TFInterface:";
    for(const auto liri : joint_to_cell_dof_indices_in)
      std::cout << " (" << liri[0] << "," << liri[1] << ")";
    std::cout << std::endl;

    std::vector<std::array<unsigned int, 2>> jtc_rt;
    for(auto i = 0; i < fe_interface_values.n_current_interface_dofs(); ++i)
      jtc_rt.push_back(fe_interface_values.interface_dof_to_dof_indices(i));
    std::cout << "TFInterface::RT:";
    for(const auto liri : jtc_rt)
      std::cout << " (" << liri[0] << "," << liri[1] << ")";
    std::cout << std::endl;
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType &                         cell,
         const unsigned int                               face_no,
         const std::vector<std::array<unsigned int, 2>> & joint_to_cell_dof_indices_in)
  {
    fe_interface_values.reinit(cell, face_no);
    joint_to_cell_dof_indices = joint_to_cell_dof_indices_in;
    std::cout << "TFface:";
    for(const auto liri : joint_to_cell_dof_indices_in)
      std::cout << " (" << liri[0] << "," << liri[1] << ")";
    std::cout << std::endl;
  }

  unsigned int
  n_current_interface_dofs() const
  {
    return joint_to_cell_dof_indices.size();
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return fe_values.get_fe();
  }

  const std::vector<Point<dim>> &
  get_quadrature_points() const
  {
    return fe_interface_values.get_quadrature_points();
  }

  const std::vector<Tensor<1, dim>> &
  get_normal_vectors() const
  {
    return fe_interface_values.get_normal_vectors();
  }

  // double
  // get_coefficient(const unsigned int i, const unsigned int j) const
  // {
  //   const auto [li, ri] = index_helper.multi_index(i);
  //   AssertIndexRange(li, shape_to_test_functions_left.m());
  //   AssertIndexRange(ri, shape_to_test_functions_right.m());

  //   AssertIndexRange(j, fe_interface_values.n_current_interface_dofs());
  //   const auto [lj, rj] = fe_interface_values.interface_dof_to_dof_indices(j);
  //   if(lj == numbers::invalid_unsigned_int)
  //     return shape_to_test_functions_right(ri, rj);

  //   return shape_to_test_functions_left(li, lj);
  // }

  double
  JxW(const unsigned int q) const
  {
    return fe_interface_values.JxW(q);
  }

  Tensor<1, dim>
  normal(const unsigned int q) const
  {
    return fe_interface_values.normal(q);
  }

  double
  shape_value_component_left(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    // if(i == numbers::invalid_unsigned_int)
    //   return 0.;

    AssertIndexRange(i, shape_to_test_functions_left.m());
    const auto & fe_face_values_left = fe_interface_values.get_fe_face_values(0);

    double value = 0.;
    for(auto j = 0U; j < shape_to_test_functions_left.n(); ++j)
      value +=
        shape_to_test_functions_left(i, j) * fe_face_values_left.shape_value_component(j, q, c);
    return value;
  }

  double
  shape_value_component_right(const unsigned int i,
                              const unsigned int q,
                              const unsigned int c) const
  {
    // if(i == numbers::invalid_unsigned_int)
    //   return 0.;

    AssertIndexRange(i, shape_to_test_functions_right.m());
    const auto & fe_face_values_right = fe_interface_values.get_fe_face_values(1);

    AssertDimension(fe_face_values_right.dofs_per_cell, shape_to_test_functions_right.n());
    double value = 0.;
    for(auto j = 0U; j < shape_to_test_functions_right.n(); ++j)
      value +=
        shape_to_test_functions_right(i, j) * fe_face_values_right.shape_value_component(j, q, c);
    return value;
  }

  Tensor<1, dim>
  shape_grad_component_left(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    // if(i == numbers::invalid_unsigned_int)
    //   return Tensor<1, dim>{};

    AssertIndexRange(i, shape_to_test_functions_left.m());
    const auto &   fe_face_values_left = fe_interface_values.get_fe_face_values(0);
    Tensor<1, dim> grad;
    for(auto j = 0U; j < fe_face_values_left.dofs_per_cell; ++j)
      grad +=
        shape_to_test_functions_left(i, j) * fe_face_values_left.shape_grad_component(j, q, c);
    return grad;
  }

  Tensor<1, dim>
  shape_grad_component_right(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    // if(i == numbers::invalid_unsigned_int)
    //   return Tensor<1, dim>{};

    AssertIndexRange(i, shape_to_test_functions_right.m());
    const auto &   fe_face_values_right = fe_interface_values.get_fe_face_values(1);
    Tensor<1, dim> grad;
    for(auto j = 0U; j < fe_face_values_right.dofs_per_cell; ++j)
      grad +=
        shape_to_test_functions_right(i, j) * fe_face_values_right.shape_grad_component(j, q, c);
    return grad;
  }

  double
  average(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    AssertIndexRange(i, joint_to_cell_dof_indices.size());
    const auto [li, ri] = joint_to_cell_dof_indices[i];

    // const double value_left = shape_value_component_left(li, q, c);
    // const double value_right =
    //   fe_interface_values.at_boundary() ? value_left : shape_value_component_right(ri, q, c);
    // double value = 0.5 * (value_left + value_right);
    // return value;

    if(fe_interface_values.at_boundary())
    {
      Assert(li != numbers::invalid_unsigned_int,
             ExcMessage("invalid test function index on the left cell"));
      return shape_value_component_left(li, q, c);
    }

    double value = 0.;

    if(li != numbers::invalid_unsigned_int)
      value += 0.5 * shape_value_component_left(li, q, c);
    if(ri != numbers::invalid_unsigned_int)
      value += 0.5 * shape_value_component_right(ri, q, c);

    return value;
  }

  Tensor<1, dim>
  average_gradient(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    AssertIndexRange(i, joint_to_cell_dof_indices.size());
    const auto [li, ri] = joint_to_cell_dof_indices[i];

    // const Tensor<1, dim> & grad_left = shape_grad_component_left(li, q, c);
    // const Tensor<1, dim> & grad_right =
    //   fe_interface_values.at_boundary() ? grad_left : shape_grad_component_right(ri, q, c);
    // Tensor<1, dim> av_grad = 0.5 * (grad_left + grad_right);
    // return av_grad;

    if(fe_interface_values.at_boundary())
    {
      Assert(li != numbers::invalid_unsigned_int,
             ExcMessage("invalid test function index on the left cell"));
      return shape_grad_component_left(li, q, c);
    }

    Tensor<1, dim> av_grad;

    if(li != numbers::invalid_unsigned_int)
      av_grad += 0.5 * shape_grad_component_left(li, q, c);
    if(ri != numbers::invalid_unsigned_int)
      av_grad += 0.5 * shape_grad_component_right(ri, q, c);

    return av_grad;
  }

  double
  jump(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    AssertIndexRange(i, joint_to_cell_dof_indices.size());
    const auto [li, ri] = joint_to_cell_dof_indices[i];

    // const double value_left = shape_value_component_left(li, q, c);
    // const double value_right =
    //   fe_interface_values.at_boundary() ? 0. : shape_value_component_right(ri, q, c);
    // double value = value_left - value_right;
    // return value;

    if(fe_interface_values.at_boundary())
    {
      Assert(li != numbers::invalid_unsigned_int,
             ExcMessage("invalid test function index on the left cell"));
      return shape_value_component_left(li, q, c);
    }

    double jump = 0.;

    if(li != numbers::invalid_unsigned_int)
      jump += shape_value_component_left(li, q, c);
    if(ri != numbers::invalid_unsigned_int)
      jump -= shape_value_component_right(ri, q, c);

    return jump;
  }

  FEValues<dim>                            fe_values;
  FEInterfaceValues<dim>                   fe_interface_values;
  const FullMatrix<double> &               shape_to_test_functions_left;
  const FullMatrix<double> &               shape_to_test_functions_right;
  unsigned int                             n_quadrature_points;
  std::vector<std::array<unsigned int, 2>> joint_to_cell_dof_indices;
};

template<int dim>
Tensor<1, dim>
compute_vaverage(const TestFunctionInterfaceValues<dim> & phi,
                 const unsigned int                       i,
                 const unsigned int                       q)
{
  return ::MW::compute_vaverage_impl<dim, TestFunctionInterfaceValues<dim>>(phi, i, q);
}

template<int dim>
SymmetricTensor<2, dim>
compute_average_symgrad(const TestFunctionInterfaceValues<dim> & phi,
                        const unsigned int                       i,
                        const unsigned int                       q)
{
  return ::MW::compute_average_symgrad_impl<dim, TestFunctionInterfaceValues<dim>>(phi, i, q);
}

template<int dim>
Tensor<1, dim>
compute_vjump(const TestFunctionInterfaceValues<dim> & phi,
              const unsigned int                       i,
              const unsigned int                       q)
{
  return ::MW::compute_vjump_impl<dim, TestFunctionInterfaceValues<dim>>(phi, i, q);
}

template<int dim>
Tensor<1, dim>
compute_vjump_tangential(const TestFunctionInterfaceValues<dim> & phi,
                         const unsigned int                       i,
                         const unsigned int                       q)
{
  return ::MW::compute_vjump_tangential_impl<dim, TestFunctionInterfaceValues<dim>>(phi, i, q);
}



namespace Interior
{
namespace MW
{
using ::MW::ScratchData;

using ::MW::Mixed::CopyData;



template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *            load_function_in,
                   const Vector<double> *           stream_function_solution,
                   const LAPACKFullMatrix<double> * transformation_matrix,
                   const InterfaceHandler<dim> *    interface_handler_in,
                   const Stokes::EquationData &     equation_data_in)
    : load_function(load_function_in),
      discrete_velocity(stream_function_solution),
      interior_rt_to_test_functions(transformation_matrix),
      interface_handler(interface_handler_in),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              const IteratorType & cellP,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  const Function<dim> *            load_function;
  const Vector<double> *           discrete_velocity;
  const LAPACKFullMatrix<double> * interior_rt_to_test_functions;
  const InterfaceHandler<dim> *    interface_handler;
  const Stokes::EquationData       equation_data;
};



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cell,
                                                 const IteratorType & cellP,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  copy_data.cell_rhs_test = 0.;

  cellP->get_active_or_mg_dof_indices(copy_data.local_dof_indices_test);
  const auto n_dofs_per_cell_p            = copy_data.local_dof_indices_test.size();
  const auto n_interior_nodes_by_pressure = n_dofs_per_cell_p - 1;

  const unsigned int cell_index = interface_handler->get_cell_index(cell->id());
  AssertDimension(copy_data.local_dof_indices_ansatz.size(), 1U);
  copy_data.local_dof_indices_ansatz[0U] = cell_index;

  FEValues<dim> & phi = scratch_data.fe_values;
  phi.reinit(cell);

  const auto n_dofs_per_cell_v = phi.get_fe().dofs_per_cell;
  AssertThrow(dim == 2, ExcMessage("dofs_per_quad only correct in 2D"));
  const auto n_interior_dofs_v = phi.get_fe().dofs_per_quad;
  const auto n_face_dofs_v     = GeometryInfo<dim>::faces_per_cell * phi.get_fe().dofs_per_face;
  AssertDimension(n_interior_dofs_v + n_face_dofs_v, n_dofs_per_cell_v);
  (void)n_dofs_per_cell_v;
  AssertDimension(interior_rt_to_test_functions->m(), n_interior_nodes_by_pressure);
  AssertDimension(interior_rt_to_test_functions->n(), n_interior_dofs_v);

  std::vector<Tensor<1, dim>> load_values;
  {
    Assert(load_function, ExcMessage("load_function is not set."));
    AssertDimension(load_function->n_components, dim);
    const auto & q_points = phi.get_quadrature_points();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(load_values),
                   [&](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = load_function->value(x_q, c);
                     return value;
                   });
  }

  /// Evaluate test function v_i (which is generated by grad p_i) at quadrature
  /// point x_q.
  const auto & compute_v_i = [&](const unsigned int i, const unsigned int q) {
    AssertIndexRange(i, n_interior_nodes_by_pressure);
    Tensor<1, dim> value;
    for(auto j = 0U; j < n_interior_dofs_v; ++j)
    {
      const auto jj = n_face_dofs_v + j; // shift to interior dofs
      for(auto d = 0U; d < dim; ++d)
        value[d] += (*interior_rt_to_test_functions)(i, j) * phi.shape_value_component(jj, q, d);
    }
    return value;
  };

  double integral_iq;
  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
  {
    const auto & f  = load_values[q];
    const auto & dx = phi.JxW(q);

    for(auto i = 0U; i < n_interior_nodes_by_pressure; ++i)
    {
      const auto & v_i = compute_v_i(i, q);
      integral_iq      = v_i * f * dx;

      AssertIndexRange(i + 1, copy_data.cell_rhs_test.size());
      copy_data.cell_rhs_test(i + 1) += integral_iq; // skip constant mode
    }
  }
}

} // namespace MW

} // namespace Interior

namespace Interface
{
namespace MW
{
using ::MW::Mixed::ScratchData;

using ::MW::Mixed::CopyData;



template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;


  /// This integrator assumes that the coefficients of the discrete pressure
  /// associated to the constant modes are set to zero!
  MatrixIntegrator(const Function<dim> *            load_function_in,
                   const Vector<double> *           stream_function_solution,
                   const Vector<double> *           pressure_solution,
                   const LAPACKFullMatrix<double> * transformation_matrix,
                   const InterfaceHandler<dim> *    interface_handler_in,
                   const Stokes::EquationData &     equation_data_in)
    : load_function(load_function_in),
      discrete_velocity(stream_function_solution),
      discrete_pressure(pressure_solution),
      face_rt_to_test_functions(transformation_matrix),
      interface_handler(interface_handler_in),
      equation_data(equation_data_in)
  {
    AssertDimension(face_rt_to_test_functions->m(), GeometryInfo<dim>::faces_per_cell);
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

  template<typename EvaluatorType>
  Tensor<1, dim>
  compute_v_face(const EvaluatorType & phi, const unsigned int face_no, const unsigned int q) const;

  const Function<dim> *            load_function;
  const Vector<double> *           discrete_velocity;
  const Vector<double> *           discrete_pressure;
  const LAPACKFullMatrix<double> * face_rt_to_test_functions;
  const InterfaceHandler<dim> *    interface_handler;
  const Stokes::EquationData       equation_data;
};



template<int dim, bool is_multigrid>
template<typename EvaluatorType>
Tensor<1, dim>
MatrixIntegrator<dim, is_multigrid>::compute_v_face(const EvaluatorType & phi,
                                                    const unsigned int    face_no,
                                                    const unsigned int    q) const
{
  AssertIndexRange(face_no, GeometryInfo<dim>::faces_per_cell);
  Tensor<1, dim> value;
  const auto     n_face_dofs_v = face_rt_to_test_functions->n();
  for(auto j = 0U; j < n_face_dofs_v; ++j)
  {
    for(auto d = 0U; d < dim; ++d)
      value[d] += (*face_rt_to_test_functions)(face_no, j) * phi.shape_value_component(j, q, d);
  }
  return value;
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cell,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_rhs_test.size(), GeometryInfo<dim>::faces_per_cell);
  copy_data.cell_rhs_test = 0.;
  // copy_data.local_dof_indices_test.resize(GeometryInfo<dim>::faces_per_cell, 0);

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

  //   FEValues<dim> & phi = scratch_data.fe_values_test;
  //   phi.reinit(cell);

  //   std::vector<Tensor<1, dim>> load_values;
  //   {
  //     Assert(load_function, ExcMessage("load_function is not set."));
  //     AssertDimension(load_function->n_components, dim);
  //     const auto & q_points = phi.get_quadrature_points();
  //     std::transform(q_points.cbegin(),
  //                    q_points.cend(),
  //                    std::back_inserter(load_values),
  //                    [&](const auto & x_q) {
  //                      Tensor<1, dim> value;
  //                      for(auto c = 0U; c < dim; ++c)
  //                        value[c] = load_function->value(x_q, c);
  //                      return value;
  //                    });
  //   }
  //   AssertDimension(load_values.size(), phi.n_quadrature_points);

  //   double residual = 0.;
  //   for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
  //   {
  //     {
  //       const auto & f      = load_values[q];
  //       const auto & dx     = phi.JxW(q);
  //       const auto & v_face = compute_v_face(phi, face_no, q);

  //       residual += v_face * f * dx;
  //     }
  //   }

  //   copy_data.cell_rhs_test[face_no] = residual;
  // }
  // std::cout << "Bi::cell_worker  ";
  // copy_data.cell_rhs_test.print(std::cout);
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

  FEInterfaceValues<dim> & phiP = scratch_data.fe_interface_values_ansatz;
  phiP.reinit(cellP, face_no, sface_no, ncellP, nface_no, nsface_no);

  FullMatrix<double> shape_to_test_functions(GeometryInfo<dim>::faces_per_cell,
                                             cell->get_fe().dofs_per_cell);
  FullMatrix<double> tmp(face_rt_to_test_functions->m(), face_rt_to_test_functions->n());
  tmp = *face_rt_to_test_functions;
  shape_to_test_functions.fill(tmp, 0U, 0U, 0U, 0U);

  TestFunctionInterfaceValues              phiV(scratch_data.fe_values_test,
                                   scratch_data.fe_interface_values_test,
                                   shape_to_test_functions,
                                   shape_to_test_functions);
  std::vector<std::array<unsigned int, 2>> testfunc_indices;
  testfunc_indices.push_back({face_no, nface_no});
  phiV.reinit(cell, face_no, sface_no, ncell, nface_no, nsface_no, testfunc_indices);

  const unsigned int n_interface_dofs_p  = phiP.n_current_interface_dofs();
  const auto &       joint_dof_indices_p = phiP.get_interface_dof_indices();

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  copy_data_face.cell_matrix.reinit(GeometryInfo<dim>::faces_per_cell, 2U);
  copy_data_face.cell_rhs_test.reinit(GeometryInfo<dim>::faces_per_cell);
  copy_data_face.joint_dof_indices_test.resize(GeometryInfo<dim>::faces_per_cell, 0U);
  copy_data_face.joint_dof_indices_ansatz.resize(2, 0U);

  copy_data_face.joint_dof_indices_test[face_no] = interface_index;
  copy_data_face.joint_dof_indices_ansatz[0U]    = cell_index;
  copy_data_face.joint_dof_indices_ansatz[1U]    = ncell_index;

  std::vector<double> joint_dof_values_p;
  for(const auto i : joint_dof_indices_p)
    joint_dof_values_p.push_back((*discrete_pressure)(i));

  /// Note that the constant mode should be currently set to zero, thus,
  /// looping over the whole set of dof indices is valid.
  const auto & compute_jump_pn = [&](const unsigned int q) {
    double               value = 0.;
    const Tensor<1, dim> n     = phiP.normal(q);
    for(auto j = 0U; j < n_interface_dofs_p; ++j) // skip constant mode
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

  // // DEBUG
  // std::cout << "interface index " << interface_index << " pn_dot_v " << pn_dot_v << std::endl;
  // std::cout << "interface index " << interface_index << " alpha_left " << alpha_left <<
  // std::endl; std::cout << "interface index " << interface_index << " alpha_right " << alpha_right
  // << std::endl;

  copy_data_face.cell_rhs_test[face_no] += pn_dot_v;
  copy_data_face.cell_matrix(face_no, 0U) = alpha_left;
  copy_data_face.cell_matrix(face_no, 1U) = alpha_right;
}

} // namespace MW

} // namespace Interface

} // namespace Pressure

} // end namespace Biharmonic



#endif /* APPS_BIHARMONICINTEGRATOR_H_ */
