/*
 * laplace_integrator.h
 *
 *  Created on: Aug 11, 2018
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
// #include "solvers_and_preconditioners/TPSS/matrix_helper.h"
// #include "solvers_and_preconditioners/TPSS/patch_transfer.h"
// #include "solvers_and_preconditioners/preconditioner/schwarz_preconditioner.h"
// #include "solvers_and_preconditioners/smoother/schwarz_smoother.h"


#include "equation_data.h"
#include "laplace_integrator.h"



using namespace dealii;

namespace Biharmonic
{
/*
 * Linear operators associated to the C0 interior penalty formulation for the
 * biharmonic problem with clamped boundary conditions
 *
 * (FD) FastDiagonalization
 */

namespace C0IP
{
namespace FD
{
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using This                    = MatrixIntegrator<dim, fe_degree, Number>;
  using value_type              = Number;
  using transfer_type           = typename TPSS::PatchTransfer<dim, Number, fe_degree>;
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
      if(equation_data.local_solver_variant == EquationData::LocalSolverVariant::Exact)
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
        constexpr Number gamma = fe_degree * (fe_degree + 1);
        const auto       h     = eval.get_h(direction, cell_no);
        const auto       nh    = eval.get_h(direction, ncell_no);
        return 2. * std::max(gamma / h, gamma / nh); // default: at boundary
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
      Laplace::DG::FD::FaceLaplace<evaluator_type> nitsche(Laplace::EquationData{});
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

} // end namespace Biharmonic



#endif /* APPS_BIHARMONICINTEGRATOR_H_ */
