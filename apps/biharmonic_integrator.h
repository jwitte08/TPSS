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

    Laplace::DG::FD::CellMass<evaluator_type>    cell_mass;
    Laplace::DG::FD::CellLaplace<evaluator_type> cell_laplace;

    evaluator_type eval(subdomain_handler); // common evaluator for test + ansatz
    for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
    {
      eval.reinit(patch);

      /// compute 1D matrices
      const auto mass_matrices      = eval.patch_action(cell_mass);
      const auto bilaplace_matrices = eval.patch_action(cell_bilaplace);
      const auto laplace_matrices   = eval.patch_action(cell_laplace);

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

      /// submit vector of rank1 Kronecker tensors
      local_matrices[patch].reinit(rank1_tensors);
    }
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
