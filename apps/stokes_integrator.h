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



using Biharmonic::C0IP::MW::IteratorSelector;

template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename IteratorSelector<dim, is_multigrid>::type;

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

    // evaluator_type eval(subdomain_handler); // common evaluator for test + ansatz
    for(auto d_test = 0U; d_test < dim; ++d_test)
    {
      evaluator_type eval_test(subdomain_handler, /*dofh_index*/ 0, /*component*/ d_test);
      for(auto d_ansatz = 0U; d_ansatz < dim; ++d_ansatz)
      {
        evaluator_type eval_ansatz(subdomain_handler, /*dofh_index*/ 0, /*component*/ d_ansatz);
        for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
        {
          /// compute 1D matrices
          eval_test.reinit(patch);
          eval_ansatz.reinit(patch);
          //   const auto mass_matrices      = assemble_mass_tensor(eval);
          //   const auto bilaplace_matrices = assemble_bilaplace_tensor(eval);
          //   const auto laplace_matrices   = assemble_laplace_tensor(eval);

          //   /// store rank1 tensors of separable Kronecker representation
          //   /// BxMxM + MxBxM + MxMxB
          //   const auto & BxMxM = [&](const int direction) {
          //     std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
          //     for(auto d = 0; d < dim; ++d)
          //       kronecker_tensor[d] = d == direction ? bilaplace_matrices[direction] :
          //       mass_matrices[d];
          //     return kronecker_tensor;
          //   };
          //   std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>> rank1_tensors;
          //   for(auto direction = 0; direction < dim; ++direction)
          //     rank1_tensors.emplace_back(BxMxM(direction));

          //   /// store rank1 tensors of mixed derivatives
          //   /// 2(LxLxM + LxMxL + MxLxL)
          //   if(equation_data.local_solver_variant == EquationData::LocalSolverVariant::Exact)
          //   {
          //     const auto & LxLxM = [&](const int direction1, const int direction2) {
          //       std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
          //       for(auto d = 0; d < dim; ++d)
          //         kronecker_tensor[d] =
          //           (d == direction1 || d == direction2) ? laplace_matrices[d] :
          //           mass_matrices[d];
          //       return kronecker_tensor;
          //     };
          //     for(auto direction1 = 0; direction1 < dim; ++direction1)
          //       for(auto direction2 = 0; direction2 < dim; ++direction2)
          //         if(direction1 != direction2)
          //           rank1_tensors.emplace_back(LxLxM(direction1, direction2));
          //   }

          //   /// submit vector of rank1 Kronecker tensors
          //   local_matrices[patch].reinit(rank1_tensors);
        }
      }
    }
  }

  // template<bool c0ip = true>
  // std::array<Table<2, VectorizedArray<Number>>, dim>
  // assemble_bilaplace_tensor(evaluator_type & eval) const
  // {
  //   const auto cell_bilaplace = [](const evaluator_type &              eval_u,
  //                                  const evaluator_type &              eval_v,
  //                                  Table<2, VectorizedArray<Number>> & cell_matrix,
  //                                  const int                           direction,
  //                                  const int                           cell_no) {
  //     auto integral = make_vectorized_array<Number>(0.);
  //     for(int j = 0; j < fe_order; ++j)
  //       for(int i = 0; i < fe_order; ++i)
  //       {
  //         integral = 0.;
  //         for(unsigned int q = 0; q < evaluator_type::n_q_points_1d_static; ++q)
  //         {
  //           const auto & hess_ui = eval_u.shape_hessian(j, q, direction, cell_no);
  //           const auto & hess_vj = eval_v.shape_hessian(i, q, direction, cell_no);
  //           const auto & dx      = eval_v.get_JxW(q, direction, cell_no);
  //           integral += hess_ui * hess_vj * dx;
  //         }
  //         cell_matrix(i, j) += integral;
  //       }
  //   };

  //   const auto compute_penalty =
  //     [&](const evaluator_type & eval, const int direction, const int cell_no, const int
  //     ncell_no) {
  //       const auto h  = eval.get_h(direction, cell_no);
  //       const auto nh = eval.get_h(direction, ncell_no);
  //       return compute_penalty_impl(fe_degree, h, nh);
  //     };

  //   const auto face_nitsche = [&](const evaluator_type &              eval_u,
  //                                 const evaluator_type &              eval_v,
  //                                 Table<2, VectorizedArray<Number>> & cell_matrix,
  //                                 const int                           direction,
  //                                 const int                           cell_no,
  //                                 const int                           face_no) {
  //     const auto normal         = eval_v.get_normal(face_no);
  //     const auto sqnormal       = normal * normal;
  //     const auto average_factor = eval_v.get_average_factor(direction, cell_no, face_no);
  //     const auto penalty = average_factor * compute_penalty(eval_v, direction, cell_no, cell_no);

  //     auto value_on_face = make_vectorized_array<Number>(0.);
  //     for(int i = 0; i < fe_degree + 1; ++i)
  //     {
  //       const auto & grad_vi = eval_v.shape_gradient_face(i, face_no, direction, cell_no);
  //       const auto & hess_vi = eval_v.shape_hessian_face(i, face_no, direction, cell_no);
  //       for(int j = 0; j < fe_degree + 1; ++j)
  //       {
  //         const auto & grad_uj = eval_u.shape_gradient_face(j, face_no, direction, cell_no);
  //         const auto & hess_uj = eval_u.shape_hessian_face(j, face_no, direction, cell_no);

  //         value_on_face = penalty * grad_vi * normal * grad_uj * normal;
  //         value_on_face -= (grad_vi * normal) * (average_factor * hess_uj * sqnormal);
  //         value_on_face -= (grad_uj * normal) * (average_factor * hess_vi * sqnormal);

  //         cell_matrix(i, j) += value_on_face;
  //       }
  //     }
  //   };

  //   const auto interface_nitsche = [&](const evaluator_type &              eval_u,
  //                                      const evaluator_type &              eval_v,
  //                                      Table<2, VectorizedArray<Number>> & cell_matrix01,
  //                                      Table<2, VectorizedArray<Number>> & cell_matrix10,
  //                                      const int                           cell_no0, // left cell
  //                                      const int                           direction) {
  //     const auto cell_no1 = cell_no0 + 1;                 // right cell
  //     const auto face_no0 = 1;                            // interface seen from left cell
  //     const auto face_no1 = 0;                            // interface seen from right cell
  //     AssertDimension(cell_no0, 0);                       // vertex patch has one interface
  //     const auto normal0   = eval_v.get_normal(face_no0); // on cell 0
  //     const auto normal1   = eval_v.get_normal(face_no1); // on cell 1
  //     const auto sqnormal0 = normal0 * normal0;
  //     const auto sqnormal1 = normal1 * normal1;
  //     const auto penalty   = 0.5 * compute_penalty(eval_v, direction, cell_no0, cell_no1);

  //     auto value_on_interface01{make_vectorized_array<Number>(0.)};
  //     auto value_on_interface10{make_vectorized_array<Number>(0.)};
  //     for(int i = 0; i < fe_degree + 1; ++i)
  //     {
  //       const auto & hess_v0i = eval_v.shape_hessian_face(i, face_no0, direction, cell_no0);
  //       const auto & grad_v0i = eval_v.shape_gradient_face(i, face_no0, direction, cell_no0);
  //       const auto & hess_v1i = eval_v.shape_hessian_face(i, face_no1, direction, cell_no1);
  //       const auto & grad_v1i = eval_v.shape_gradient_face(i, face_no1, direction, cell_no1);
  //       for(int j = 0; j < fe_degree + 1; ++j)
  //       {
  //         const auto & hess_u0j = eval_u.shape_hessian_face(j, face_no0, direction, cell_no0);
  //         const auto & grad_u0j = eval_u.shape_gradient_face(j, face_no0, direction, cell_no0);
  //         const auto & hess_u1j = eval_u.shape_hessian_face(j, face_no1, direction, cell_no1);
  //         const auto & grad_u1j = eval_u.shape_gradient_face(j, face_no1, direction, cell_no1);

  //         /// consistency + symmetry
  //         value_on_interface01 = penalty * grad_v0i * normal0 * grad_u1j * normal1;
  //         value_on_interface01 -= (grad_v0i * normal0) * (0.5 * hess_u1j * sqnormal1);
  //         value_on_interface01 -= (grad_u1j * normal1) * (0.5 * hess_v0i * sqnormal0);

  //         value_on_interface10 = penalty * grad_v1i * normal1 * grad_u0j * normal0;
  //         value_on_interface10 -= (grad_v1i * normal1) * (0.5 * hess_u0j * sqnormal0);
  //         value_on_interface10 -= (grad_u0j * normal0) * (0.5 * hess_v1i * sqnormal1);

  //         cell_matrix01(i, j) += value_on_interface01;
  //         cell_matrix10(i, j) += value_on_interface10;
  //       }
  //     }
  //   };

  //   if constexpr(c0ip)
  //     return eval.patch_action(cell_bilaplace, face_nitsche, interface_nitsche);
  //   (void)face_nitsche, (void)interface_nitsche;
  //   return eval.patch_action(cell_bilaplace);
  // }

  template<bool is_sipg = false>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  assemble_laplace_tensor(evaluator_type & eval) const
  {
    if constexpr(is_sipg)
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

} // end namespace SIPG

} // namespace Velocity

} // namespace Stokes

#endif /* APPS_STOKESINTEGRATOR_H_ */
