/*
 * stokes_integrator.h
 *
 *  Created on: May 19, 2020
 *      Author: witte
 */

#ifndef APPS_STOKESINTEGRATOR_H_
#define APPS_STOKESINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>


#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"
#include "solvers_and_preconditioners/TPSS/tensor_product_matrix.h"


#include "biharmonic_integrator.h"
#include "common_integrator.h"
#include "equation_data.h"
#include "laplace_integrator.h"

namespace Stokes
{
using namespace dealii;

/**
 * Linear operators associated to the SIPG formulation (symmetric gradient) for
 * the stokes velocity with heterogeneous Dirichlet boundary conditions
 *
 * (MW) MeshWorker
 * (FD) FastDiagonalization
 */
namespace Velocity
{
namespace SIPG
{
/**
 * Standard (interior) penalty to obtain well-posedness of the Nitsche
 * method. The penalty is weighted for face integrals at the physical
 * boundary. The interior penalty is obtained by multiplying with 1/2.
 */
template<typename Number>
Number
compute_penalty_impl(const int degree, const Number h_left, const Number h_right)
{
  const auto one_over_h = (0.5 / h_left) + (0.5 / h_right);
  const auto gamma      = degree == 0 ? 1 : degree * (degree + 1);
  return 2.0 * gamma * one_over_h;
}



namespace MW
{
using ::MW::ScratchData;

using ::MW::CopyData;

template<int dim, bool is_multigrid = false>
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

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cell,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  copy_data.cell_matrix = 0.;
  copy_data.cell_rhs    = 0.;

  FEValues<dim> & phi = scratch_data.fe_values;
  phi.reinit(cell);
  cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);

  const unsigned int               dofs_per_cell = phi.get_fe().dofs_per_cell;
  const FEValuesExtractors::Vector velocities(0);
  /// symgrad_phi_{d,c} = 0.5 (\partial_d phi_{i;c} + \partial_c phi_{i;d})
  const auto symgrad_phi = [&](const unsigned int i, const unsigned int q) {
    SymmetricTensor<2, dim> symgrad_of_phi;
    for(auto d = 0U; d < dim; ++d)
      for(auto c = d; c < dim; ++c)
        symgrad_of_phi[d][c] =
          0.5 * (phi.shape_grad_component(i, q, c)[d] + phi.shape_grad_component(i, q, d)[c]);
    return symgrad_of_phi;
  };

  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const SymmetricTensor<2, dim> symgrad_phi_i =
        symgrad_phi(i, q); // phi[velocities].symmetric_gradient(i, q);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const SymmetricTensor<2, dim> symgrad_phi_j = phi[velocities].symmetric_gradient(j, q);

        copy_data.cell_matrix(i, j) += 2. *
                                       scalar_product(symgrad_phi_i,   // symgrad phi_i(x)
                                                      symgrad_phi_j) * // symgrad phi_j(x)
                                       phi.JxW(q);                     // dx
      }
    }
  }
}


template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker(const IteratorType & cell,
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

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   nh        = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  const auto   fe_degree = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h = 0.5 * compute_penalty_impl(fe_degree, h, nh);

  /// average_symgrad(phi) = 0.5 ({{ \partial_d phi_{i;c} }} + {{ \partial_c phi_{i;d} }})
  const auto average_symgrad_phi = [&](const unsigned int i, const unsigned int q) {
    SymmetricTensor<2, dim> av_symgrad_of_phi;
    for(auto d = 0U; d < dim; ++d)
      for(auto c = d; c < dim; ++c)
        av_symgrad_of_phi[d][c] = 0.5 * (fe_interface_values.average_gradient(i, q, c)[d] +
                                         fe_interface_values.average_gradient(i, q, d)[c]);
    return av_symgrad_of_phi;
  };

  /// jump(phi) = [[ phi ]] = phi^+ - phi^-
  const auto jump_phi = [&](const unsigned int i, const unsigned int q) {
    Tensor<1, dim> jump_phi;
    for(auto c = 0; c < dim; ++c)
      jump_phi[c] = fe_interface_values.jump(i, q, c);
    return jump_phi;
  };

  /// jump_cross_normal(phi) = [[ phi ]] (x) n
  const auto jump_phi_cross_normal = [&](const unsigned int i, const unsigned int q) {
    const Tensor<1, dim> & n = fe_interface_values.normal(q);
    return outer_product(jump_phi(i, q), n);
  };

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
  {
    // const auto & n = fe_interface_values.normal(q);

    for(unsigned int i = 0; i < n_interface_dofs; ++i)
    {
      const auto & av_symgrad_phi_i   = average_symgrad_phi(i, q);
      const auto & jump_phi_i_cross_n = jump_phi_cross_normal(i, q);

      for(unsigned int j = 0; j < n_interface_dofs; ++j)
      {
        const auto & av_symgrad_phi_j   = average_symgrad_phi(j, q);
        const auto & jump_phi_j_cross_n = jump_phi_cross_normal(j, q);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi(j, q) * jump_phi(i, q);
        integral_ijq *= 2. * fe_interface_values.JxW(q);

        copy_data_face.cell_matrix(i, j) += integral_ijq;
      }
    }
  }
}

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker(const IteratorType & cell,
                                                     const unsigned int & f,
                                                     ScratchData<dim> &   scratch_data,
                                                     CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values;
  fe_interface_values.reinit(cell, f);

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

  const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();
  copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   fe_degree = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h = compute_penalty_impl(fe_degree, h, h);

  /// average_symgrad(phi) = 0.5 ({{ \partial_d phi_{i;c} }} + {{ \partial_c phi_{i;d} }})
  const auto average_symgrad_phi = [&](const unsigned int i, const unsigned int q) {
    SymmetricTensor<2, dim> av_symgrad_of_phi;
    for(auto d = 0U; d < dim; ++d)
      for(auto c = d; c < dim; ++c)
        av_symgrad_of_phi[d][c] = 0.5 * (fe_interface_values.average_gradient(i, q, c)[d] +
                                         fe_interface_values.average_gradient(i, q, d)[c]);
    return av_symgrad_of_phi;
  };

  /// jump(phi) = [[ phi ]] = phi^+ - phi^-
  const auto jump_phi = [&](const unsigned int i, const unsigned int q) {
    Tensor<1, dim> jump_phi;
    for(auto c = 0; c < dim; ++c)
      jump_phi[c] = fe_interface_values.jump(i, q, c);
    return jump_phi;
  };

  /// jump_cross_normal(phi) = [[ phi ]] (x) n
  const auto jump_phi_cross_normal = [&](const unsigned int i, const unsigned int q) {
    const Tensor<1, dim> & n = fe_interface_values.normal(q);
    return outer_product(jump_phi(i, q), n);
  };

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
  {
    // const auto & n = fe_interface_values.normal(q);

    for(unsigned int i = 0; i < n_interface_dofs; ++i)
    {
      const auto & av_symgrad_phi_i   = average_symgrad_phi(i, q);
      const auto & jump_phi_i_cross_n = jump_phi_cross_normal(i, q);

      for(unsigned int j = 0; j < n_interface_dofs; ++j)
      {
        const auto & av_symgrad_phi_j   = average_symgrad_phi(j, q);
        const auto & jump_phi_j_cross_n = jump_phi_cross_normal(j, q);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi(j, q) * jump_phi(i, q);
        integral_ijq *= 2. * fe_interface_values.JxW(q);

        copy_data_face.cell_matrix(i, j) += integral_ijq;
      }
    }
  }
}

} // namespace MW

namespace FD
{
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using This = MatrixIntegrator<dim, fe_degree, Number>;

  static constexpr int fe_order = fe_degree + 1;

  using value_type     = Number;
  using transfer_type  = typename TPSS::PatchTransfer<dim, Number>;
  using matrix_type_1d = Table<2, VectorizedArray<Number>>;
  using matrix_type    = Tensors::BlockMatrix<dim, VectorizedArray<Number>, -1, -1>;
  using evaluator_type = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;

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

    for(auto comp_test = 0U; comp_test < dim; ++comp_test)
    {
      evaluator_type eval_test(subdomain_handler, /*dofh_index*/ 0, comp_test);
      for(auto comp_ansatz = comp_test; comp_ansatz < dim; ++comp_ansatz) // assuming isotropy !
      {
        evaluator_type eval_ansatz(subdomain_handler, /*dofh_index*/ 0, comp_ansatz);
        for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
        {
          auto & Aloc = local_matrices[patch];
          if(Aloc.n_block_rows() == 0 && Aloc.n_block_cols() == 0)
            Aloc.resize(dim, dim);

          eval_test.reinit(patch);
          eval_ansatz.reinit(patch);

          const auto mass_matrices = assemble_mass_tensor(eval_test, eval_ansatz);

          if(comp_test == comp_ansatz)
          {
            const auto laplace_matrices = assemble_laplace_tensor(eval_test, eval_ansatz);

            const auto & MxMxL = [&](const unsigned int direction_of_L) {
              /// for example, we obtain MxMxL for direction_of_L = 0 (dimension
              /// 0 is rightmost!)
              std::array<matrix_type_1d, dim> kronecker_tensor;
              /// if direction_of_L equals the velocity component we scale by two
              AssertDimension(comp_test, comp_ansatz);
              const auto factor = direction_of_L == comp_ansatz ? 2. : 1.;
              for(auto d = 0U; d < dim; ++d)
                kronecker_tensor[d] = d == direction_of_L ?
                                        factor * laplace_matrices[direction_of_L] :
                                        mass_matrices[d];
              return kronecker_tensor;
            };

            /// (0,0)-block: LxMxM + MxLxM + MxMx2L
            std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
            for(auto direction_of_L = 0; direction_of_L < dim; ++direction_of_L)
              rank1_tensors.emplace_back(MxMxL(direction_of_L));
            Aloc.get_block(comp_test, comp_ansatz).reinit(rank1_tensors);
          }

          else
          {
            const auto gradient_matrices = assemble_gradient_tensor(eval_test, eval_ansatz);

            const auto & MxGxGT = [&](const int deriv_index_ansatz, const int deriv_index_test) {
              /// for example, we obtain MxGxGT for deriv_index_ansatz = 1 and
              /// deriv_index_test = 0 (dimension 0 is rightmost!)
              Assert(deriv_index_ansatz != deriv_index_test,
                     ExcMessage("This case is not well-defined."));
              std::array<matrix_type_1d, dim> kronecker_tensor;
              for(auto d = 0; d < dim; ++d)
                kronecker_tensor[d] = d == deriv_index_ansatz ?
                                        gradient_matrices[deriv_index_ansatz] :
                                        (d == deriv_index_test ?
                                           Tensors::transpose(gradient_matrices[deriv_index_test]) :
                                           mass_matrices[d]);
              return kronecker_tensor;
            };

            /// (0,1)-block: MxGxGT
            {
              const auto                                   derivative_index_ansatz = comp_test;
              const auto                                   derivative_index_test   = comp_ansatz;
              std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
              rank1_tensors.emplace_back(
                MxGxGT(/*G*/ derivative_index_ansatz, /*GT*/ derivative_index_test));
              Aloc.get_block(comp_test, comp_ansatz).reinit(rank1_tensors);
            }

            /// (1,0)-block: MxGTxG
            {
              const auto comp_ansatz_flipped     = comp_test;   // assuming isotropy !
              const auto comp_test_flipped       = comp_ansatz; // assuming isotropy !
              const auto derivative_index_ansatz = comp_test_flipped;
              const auto derivative_index_test   = comp_ansatz_flipped;
              std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
              rank1_tensors.emplace_back(
                MxGxGT(/*G*/ derivative_index_ansatz, /*GT*/ derivative_index_test));
              Aloc.get_block(comp_test_flipped, comp_ansatz_flipped).reinit(rank1_tensors);
            }
          }
        }
      }
    }
  }

  template<bool is_sipg = false>
  std::array<matrix_type_1d, dim>
  assemble_laplace_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    if constexpr(is_sipg)
    {
      Laplace::DG::FD::CellLaplace<evaluator_type> cell_laplace;
      Laplace::DG::FD::FaceLaplace<evaluator_type> nitsche(Laplace::EquationData{});
      return eval_test.patch_action(eval_ansatz, cell_laplace, nitsche, nitsche);
    }
    using CellLaplace = ::FD::Laplace::CellOperation<dim, fe_degree, fe_degree + 1, Number>;
    return eval_test.patch_action(eval_ansatz, CellLaplace{});
  }

  std::array<matrix_type_1d, dim>
  assemble_mass_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellMass = ::FD::L2::CellOperation<dim, fe_degree, fe_degree + 1, Number>;
    return eval_test.patch_action(eval_ansatz, CellMass{});
  }

  std::array<matrix_type_1d, dim>
  assemble_gradient_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellGradient = ::FD::Gradient::CellOperation<dim, fe_degree, fe_degree + 1, Number>;
    return eval_test.patch_action(eval_ansatz, CellGradient{});
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};

} // end namespace FD

} // end namespace SIPG

} // namespace Velocity



namespace VelocityPressure
{
namespace MW
{
using ::MW::ScratchData;

using ::MW::CopyData;

template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *       load_function_in,
                   const Function<dim> *       analytical_solution_in,
                   const BlockVector<double> * particular_solution,
                   const EquationData &        equation_data_in)
    : load_function(load_function_in),
      analytical_solution(analytical_solution_in),
      discrete_solution(particular_solution),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const
  {
    copy_data.cell_matrix = 0.;
    copy_data.cell_rhs    = 0.;

    FEValues<dim> & fe_values = scratch_data.fe_values;
    fe_values.reinit(cell);
    cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);
    const auto &       fe            = fe_values.get_fe();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = fe_values.n_quadrature_points;

    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    load_function->vector_value_list(fe_values.get_quadrature_points(), rhs_values);

    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
        div_phi_u[k]     = fe_values[velocities].divergence(k, q);
        phi_p[k]         = fe_values[pressure].value(k, q);
      }

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for(unsigned int j = 0; j <= i; ++j)
        {
          copy_data.cell_matrix(i, j) +=
            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) - div_phi_u[i] * phi_p[j] -
             phi_p[i] * div_phi_u[j] +
             (equation_data.assemble_pressure_mass_matrix ? phi_p[i] * phi_p[j] : 0)) *
            fe_values.JxW(q);
        }

        const unsigned int component_i = fe.system_to_component_index(i).first;
        copy_data.cell_rhs(i) +=
          fe_values.shape_value(i, q) * rhs_values[q](component_i) * fe_values.JxW(q);
      }
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = i + 1; j < dofs_per_cell; ++j)
        copy_data.cell_matrix(i, j) = copy_data.cell_matrix(j, i);

    if(discrete_solution)
    {
      Vector<double> u0(copy_data.local_dof_indices.size());
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solution)(copy_data.local_dof_indices[i]);
      Vector<double> w0(copy_data.local_dof_indices.size());
      copy_data.cell_matrix.vmult(w0, u0);
      copy_data.cell_rhs -= w0;
    }
  }

  const Function<dim> *       load_function;
  const Function<dim> *       analytical_solution;
  const BlockVector<double> * discrete_solution;
  const EquationData          equation_data;
};
} // namespace MW

} // namespace VelocityPressure

} // namespace Stokes

#endif /* APPS_STOKESINTEGRATOR_H_ */
