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
using ::MW::compute_symgrad;

using ::MW::compute_average_symgrad;

using ::MW::compute_vvalue;

using ::MW::compute_vjump;

using ::MW::compute_vjump_cross_normal;

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

  const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

  std::vector<Tensor<1, dim>> load_values;
  if(!is_multigrid)
  {
    Assert(load_function, ExcMessage("load_function is not set."));
    AssertDimension(load_function->n_components, dim);
    const auto & q_points = phi.get_quadrature_points();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(load_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = load_function->value(x_q, c);
                     return value;
                   });
  }

  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const SymmetricTensor<2, dim> symgrad_phi_i = compute_symgrad(phi, i, q);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const SymmetricTensor<2, dim> symgrad_phi_j = compute_symgrad(phi, j, q);

        copy_data.cell_matrix(i, j) += 2. *
                                       scalar_product(symgrad_phi_i,   // symgrad phi_i(x)
                                                      symgrad_phi_j) * // symgrad phi_j(x)
                                       phi.JxW(q);                     // dx
      }

      if(!is_multigrid)
      {
        const auto & phi_i = compute_vvalue(phi, i, q);
        copy_data.cell_rhs(i) += phi_i * load_values[q] * phi.JxW(q);
      }
    }
  }

  /// Subtract the particular solution @p discrete_solution from the right hand
  /// side, thus, as usual moving essential boundary conditions to the right
  /// hand side.
  ///
  /// If @p discrete_solution is not set (for example for a DG method) we skip
  /// here.
  if(!is_multigrid)
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
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < n_interface_dofs; ++i)
    {
      const auto & av_symgrad_phi_i   = compute_average_symgrad(fe_interface_values, i, q);
      const auto & jump_phi_i_cross_n = compute_vjump_cross_normal(fe_interface_values, i, q);
      const auto & jump_phi_i         = compute_vjump(fe_interface_values, i, q);

      for(unsigned int j = 0; j < n_interface_dofs; ++j)
      {
        const auto & av_symgrad_phi_j   = compute_average_symgrad(fe_interface_values, j, q);
        const auto & jump_phi_j_cross_n = compute_vjump_cross_normal(fe_interface_values, j, q);
        const auto & jump_phi_j         = compute_vjump(fe_interface_values, j, q);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi_j * jump_phi_i;
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

  const unsigned int n_dofs = fe_interface_values.n_current_interface_dofs();
  copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   fe_degree = scratch_data.fe_values.get_fe().degree;
  const double gamma_over_h = equation_data.ip_factor * compute_penalty_impl(fe_degree, h, h);

  std::vector<Tensor<1, dim>> solution_values;
  std::vector<Tensor<2, dim>> solution_cross_normals;
  if(!is_multigrid)
  {
    Assert(analytical_solution, ExcMessage("analytical_solution is not set."));
    AssertDimension(analytical_solution->n_components, dim);
    const auto &                        q_points = fe_interface_values.get_quadrature_points();
    const std::vector<Tensor<1, dim>> & normals  = fe_interface_values.get_normal_vectors();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(solution_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = analytical_solution->value(x_q, c);
                     return value;
                   });
    AssertDimension(normals.size(), solution_values.size());
    std::transform(solution_values.cbegin(),
                   solution_values.cend(),
                   normals.cbegin(),
                   std::back_inserter(solution_cross_normals),
                   [](const auto & u_q, const auto & normal) {
                     return outer_product(u_q, normal);
                   });
  }

  AssertDimension(copy_data.cell_rhs.size(), n_dofs);
  Assert(copy_data.local_dof_indices == copy_data_face.joint_dof_indices,
         ExcMessage(
           "copy_data.cell_rhs is incompatible compared to copy_data_face.joint_dof_indices."));

  double integral_ijq = 0.;
  double nitsche_iq   = 0.;
  for(unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < n_dofs; ++i)
    {
      const auto & av_symgrad_phi_i   = compute_average_symgrad(fe_interface_values, i, q);
      const auto & jump_phi_i_cross_n = compute_vjump_cross_normal(fe_interface_values, i, q);
      const auto & jump_phi_i         = compute_vjump(fe_interface_values, i, q);

      for(unsigned int j = 0; j < n_dofs; ++j)
      {
        const auto & av_symgrad_phi_j   = compute_average_symgrad(fe_interface_values, j, q);
        const auto & jump_phi_j_cross_n = compute_vjump_cross_normal(fe_interface_values, j, q);
        const auto & jump_phi_j         = compute_vjump(fe_interface_values, j, q);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi_j * jump_phi_i;
        integral_ijq *= 2. * fe_interface_values.JxW(q);

        copy_data_face.cell_matrix(i, j) += integral_ijq;
      }

      /// Nitsche method (weak Dirichlet conditions)
      if(!is_multigrid)
      {
        const auto & u         = solution_values[q];
        const auto & u_cross_n = solution_cross_normals[q];

        nitsche_iq = -scalar_product(u_cross_n, av_symgrad_phi_i);
        nitsche_iq += gamma_over_h * u * jump_phi_i;
        nitsche_iq *= 2. * fe_interface_values.JxW(q);

        copy_data.cell_rhs(i) += nitsche_iq;
      }
    }
  }
}

} // namespace MW

namespace FD
{
template<int dim,
         int fe_degree,
         typename Number            = double,
         TPSS::DoFLayout dof_layout = TPSS::DoFLayout::Q>
class MatrixIntegrator
{
public:
  using This = MatrixIntegrator<dim, fe_degree, Number>;

  static constexpr int n_q_points_1d = fe_degree + 1;

  using value_type     = Number;
  using transfer_type  = typename TPSS::PatchTransfer<dim, Number>;
  using matrix_type_1d = Table<2, VectorizedArray<Number>>;
  using matrix_type    = Tensors::BlockMatrix<dim, VectorizedArray<Number>, -1, -1>;
  using evaluator_type = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;

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
    /// TODO tangential components only for RT !!!
    constexpr bool is_sipg =
      TPSS::DoFLayout::DGQ == dof_layout || TPSS::DoFLayout::RT == dof_layout;

    // const auto zero_out = [](std::vector<std::array<matrix_type_1d, dim>> & rank1_tensors) {
    //   for(auto & tensor : rank1_tensors)
    //   {
    //     tensor.front() = 0. * tensor.front();
    //   }
    // };

    for(auto comp_test = 0U; comp_test < dim; ++comp_test)
    {
      evaluator_type eval_test(subdomain_handler, /*dofh_index*/ 0, comp_test);
      for(auto comp_ansatz = comp_test; comp_ansatz < dim; ++comp_ansatz) // assuming isotropy !
      {
        evaluator_type eval_ansatz(subdomain_handler, /*dofh_index*/ 0, comp_ansatz);
        for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
        {
          auto & velocity_matrix = local_matrices[patch];
          if(velocity_matrix.n_block_rows() != dim && velocity_matrix.n_block_cols() != dim)
            velocity_matrix.resize(dim, dim);

          eval_test.reinit(patch);
          eval_ansatz.reinit(patch);

          const auto mass_matrices = assemble_mass_tensor(eval_test, eval_ansatz);

          if(comp_test == comp_ansatz)
          {
            const auto laplace_matrices = assemble_laplace_tensor<is_sipg>(eval_test, eval_ansatz);

            const auto & MxMxL = [&](const unsigned int direction_of_L) {
              /// For example, we obtain MxMxL for direction_of_L = 0 (dimension
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
            velocity_matrix.get_block(comp_test, comp_ansatz).reinit(rank1_tensors);
          }

          else
          {
            /// The factor 2 arising from 2 * e(u) : grad v is implicitly
            /// equalized by the factor 1/2 from the symmetrized gradient
            /// e(u). First, we emphasize that for off-diagonal blocks there
            /// are no penalty contributions. For the remaing contributions,
            /// namely consistency and symmetry terms, again the factor 2 is
            /// implicitly equalized. Nevertheless, we have to consider the
            /// factor 1/2 arising from average operators {{e(u)}} and
            /// {{e(v)}}, respectively.
            const auto gradient_matrices = assemble_gradient_tensor(eval_test, eval_ansatz);

            const auto & MxGxGT = [&](const auto component_test, const auto component_ansatz) {
              const int deriv_index_ansatz = component_test;
              const int deriv_index_test   = component_ansatz;
              Assert(deriv_index_ansatz != deriv_index_test,
                     ExcMessage("This case is not well-defined."));
              std::array<matrix_type_1d, dim> kronecker_tensor;
              for(auto d = 0; d < dim; ++d)
              {
                if(d == deriv_index_ansatz)
                  kronecker_tensor[d] = gradient_matrices[deriv_index_ansatz];
                else if(d == deriv_index_test)
                  kronecker_tensor[d] = Tensors::transpose(gradient_matrices[deriv_index_test]);
                else
                  kronecker_tensor[d] = mass_matrices[d];
              }
              return kronecker_tensor;
            };

            std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
            /// (0,1)-block: MxGxGT + MxPxG + MxGTxP
            {
              /// MxGxGT
              rank1_tensors.emplace_back(MxGxGT(comp_test, comp_ansatz));

              /// Factor 1/2 of average operator {{e(u)}} and {{e(v)}} is used
              /// within assemble_mixed_nitsche_tensor
              const auto point_mass_matrices =
                assemble_mixed_nitsche_tensor(eval_test, eval_ansatz);

              { /// MxPxG
                std::array<matrix_type_1d, dim> kronecker_tensor;
                for(auto d = 0U; d < dim; ++d)
                {
                  if(d == comp_test)
                    kronecker_tensor[d] = gradient_matrices[comp_test];
                  else if(d == comp_ansatz)
                    kronecker_tensor[d] = point_mass_matrices[comp_ansatz];
                  else
                    kronecker_tensor[d] = mass_matrices[d];
                }
                rank1_tensors.emplace_back(kronecker_tensor);
              }

              { /// MxGTxP
                std::array<matrix_type_1d, dim> kronecker_tensor;
                for(auto d = 0U; d < dim; ++d)
                {
                  if(d == comp_test)
                    kronecker_tensor[d] = point_mass_matrices[comp_test];
                  else if(d == comp_ansatz)
                    kronecker_tensor[d] = Tensors::transpose(gradient_matrices[comp_ansatz]);
                  else
                    kronecker_tensor[d] = mass_matrices[d];
                }
                rank1_tensors.emplace_back(kronecker_tensor);
              }

              velocity_matrix.get_block(comp_test, comp_ansatz).reinit(rank1_tensors);
            }

            /// (1,0)-block: transpose of (0,1)-block
            {
              for(auto & tensor : rank1_tensors)
                Tensors::transpose_tensor<dim>(tensor);
              velocity_matrix.get_block(comp_ansatz, comp_test).reinit(rank1_tensors);
            }
          }
        }
      }
    }
  }

  std::array<matrix_type_1d, dim>
  assemble_mixed_nitsche_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellVoid = ::FD::Void::CellOperation<dim, fe_degree, fe_degree + 1, Number>;

    const auto face_point_mass = [&](const evaluator_type &              eval_ansatz,
                                     const evaluator_type &              eval_test,
                                     Table<2, VectorizedArray<Number>> & cell_matrix,
                                     const int                           direction,
                                     const int                           cell_no,
                                     const int                           face_no) {
      const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
      const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
      const auto normal_vector  = eval_ansatz.get_normal_vector(face_no, direction);
      const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
      const auto comp_u         = eval_ansatz.vector_component();
      const auto comp_v         = eval_test.vector_component();

      for(int i = 0; i < n_dofs_test; ++i)
      {
        const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
        for(int j = 0; j < n_dofs_ansatz; ++j)
        {
          const auto & u_j           = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);
          const auto & value_on_face = -average_factor * (v_i * normal_vector[comp_u] * u_j +
                                                          v_i * u_j * normal_vector[comp_v]);
          cell_matrix(i, j) += value_on_face;
        }
      }
    };

    const auto interface_point_mass = [&](const evaluator_type &              eval_ansatz,
                                          const evaluator_type &              eval_test,
                                          Table<2, VectorizedArray<Number>> & cell_matrix01,
                                          Table<2, VectorizedArray<Number>> & cell_matrix10,
                                          const int                           cell_no_left,
                                          const int                           direction) {
      (void)cell_no_left;
      AssertDimension(cell_no_left, 0);
      const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
      const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
      const auto normal_vector0 = eval_test.get_normal_vector(1, direction); // on cell 0
      const auto normal_vector1 = eval_test.get_normal_vector(0, direction); // on cell 1
      const auto comp_u         = eval_ansatz.vector_component();
      const auto comp_v         = eval_test.vector_component();

      auto value_on_interface01{make_vectorized_array<Number>(0.)};
      auto value_on_interface10{make_vectorized_array<Number>(0.)};
      for(int i = 0; i < n_dofs_test; ++i)
      {
        const auto & v0_i = eval_test.shape_value_face(i, /*face_no*/ 1, direction, /*cell_no*/ 0);
        const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
        for(int j = 0; j < n_dofs_ansatz; ++j)
        {
          const auto & u0_j = eval_ansatz.shape_value_face(j, 1, direction, 0);
          const auto & u1_j = eval_ansatz.shape_value_face(j, 0, direction, 1);

          /// consistency + symmetry
          value_on_interface01 =
            -0.5 * (v0_i * normal_vector0[comp_u] * u1_j + v0_i * u1_j * normal_vector1[comp_v]);
          value_on_interface10 =
            -0.5 * (v1_i * normal_vector1[comp_u] * u0_j + v1_i * u0_j * normal_vector0[comp_v]);

          cell_matrix01(i, j) += value_on_interface01;
          cell_matrix10(i, j) += value_on_interface10;
        }
      }
    };

    return eval_test.patch_action(eval_ansatz, CellVoid{}, face_point_mass, interface_point_mass);
  }

  template<bool is_sipg = false>
  std::array<matrix_type_1d, dim>
  assemble_laplace_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellLaplace = ::FD::Laplace::CellOperation<dim, fe_degree, fe_degree + 1, Number>;
    CellLaplace cell_laplace;

    if constexpr(is_sipg)
    {
      using FaceLaplace = ::FD::Laplace::SIPG::FaceOperation<dim, fe_degree, fe_degree + 1, Number>;
      FaceLaplace nitsche;
      nitsche.penalty_factor          = equation_data.ip_factor;
      nitsche.interior_penalty_factor = equation_data.ip_factor;

      const auto face_nitsche_plus_penalty = [&](const evaluator_type & eval_ansatz,
                                                 const evaluator_type & eval_test,
                                                 matrix_type_1d &       cell_matrix,
                                                 const int              direction,
                                                 const int              cell_no,
                                                 const int              face_no) {
        nitsche(eval_ansatz, eval_test, cell_matrix, direction, cell_no, face_no);

        const int vector_component = eval_test.vector_component();
        AssertDimension(vector_component, static_cast<int>(eval_ansatz.vector_component()));

        if(vector_component != direction)
        {
          const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
          const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
          const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
          const auto normal         = eval_test.get_normal(face_no);

          const auto h       = eval_test.get_h(direction, cell_no);
          const auto penalty = nitsche.penalty_factor * average_factor *
                               ::Nitsche::compute_penalty_impl(fe_degree, h, h);

          auto value_on_face = make_vectorized_array<Number>(0.);
          for(int i = 0; i < n_dofs_test; ++i)
          {
            const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
            for(int j = 0; j < n_dofs_ansatz; ++j)
            {
              const auto & u_j = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);
              value_on_face    = penalty * v_i * u_j * normal * normal;
              cell_matrix(i, j) += value_on_face;
            }
          }
        }
      };

      const auto interface_nitsche_plus_penalty = [&](const evaluator_type & eval_ansatz,
                                                      const evaluator_type & eval_test,
                                                      matrix_type_1d &       cell_matrix01,
                                                      matrix_type_1d &       cell_matrix10,
                                                      const int              cell_no0,
                                                      const int              direction) {
        nitsche(eval_ansatz, eval_test, cell_matrix01, cell_matrix10, cell_no0, direction);

        const int vector_component = eval_test.vector_component();
        AssertDimension(vector_component, static_cast<int>(eval_ansatz.vector_component()));

        if(vector_component != direction)
        {
          const int  n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
          const int  n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
          const auto normal0       = eval_test.get_normal(1); // on cell 0
          const auto normal1       = eval_test.get_normal(0); // on cell 1

          const auto h0      = eval_test.get_h(direction, cell_no0);
          const auto h1      = eval_test.get_h(direction, cell_no0 + 1);
          const auto penalty = nitsche.interior_penalty_factor * 0.5 *
                               ::Nitsche::compute_penalty_impl(fe_degree, h0, h1);

          auto value_on_interface01 = make_vectorized_array<Number>(0.);
          auto value_on_interface10 = make_vectorized_array<Number>(0.);
          for(int i = 0; i < n_dofs_test; ++i)
          {
            const auto & v0_i = eval_test.shape_value_face(i, 1, direction, 0);
            const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
            for(int j = 0; j < n_dofs_ansatz; ++j)
            {
              const auto & u0_j    = eval_ansatz.shape_value_face(j, 1, direction, 0);
              const auto & u1_j    = eval_ansatz.shape_value_face(j, 0, direction, 1);
              value_on_interface01 = penalty * v0_i * u1_j * normal0 * normal1;
              value_on_interface10 = penalty * v1_i * u0_j * normal1 * normal0;
              cell_matrix01(i, j) += value_on_interface01;
              cell_matrix10(i, j) += value_on_interface10;
            }
          }
        }
      };

      return eval_test.patch_action(eval_ansatz,
                                    cell_laplace,
                                    face_nitsche_plus_penalty,
                                    interface_nitsche_plus_penalty);
    }

    return eval_test.patch_action(eval_ansatz, cell_laplace);
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
    CellGradient cell_gradient;
    return eval_test.patch_action(eval_ansatz, cell_gradient);
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



namespace Pressure
{
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
  cell_mass_worker(const IteratorType & cell,
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

  const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

  AssertDimension(load_function->n_components, 1U); // !!!

  const auto & quadrature_points = phi.get_quadrature_points();
  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
  {
    const auto & x_q = quadrature_points[q];
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      if(!is_multigrid)
      {
        const auto load_value = load_function->value(x_q, 0);
        copy_data.cell_rhs(i) += phi.shape_value(i, q) * load_value * phi.JxW(q);
      }
    }
  }
}

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_mass_worker(const IteratorType & cell,
                                                      ScratchData<dim> &   scratch_data,
                                                      CopyData &           copy_data) const
{
  copy_data.cell_matrix = 0.;
  copy_data.cell_rhs    = 0.;

  FEValues<dim> & phi = scratch_data.fe_values;
  phi.reinit(cell);
  cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);

  const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        copy_data.cell_matrix(i, j) += phi.shape_value(i, q) * phi.shape_value(j, q) * phi.JxW(q);
}

} // namespace MW

} // namespace Pressure



namespace VelocityPressure
{
namespace MW
{
using ::MW::ScratchData;

using ::MW::CopyData;



template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  static_assert(!is_multigrid, "not implemented.");

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



namespace Mixed
{
using ::MW::compute_vjump;

using ::MW::compute_vjump_dot_normal;

using ::MW::compute_divergence;

using ::MW::Mixed::ScratchData;

using ::MW::Mixed::CopyData;

template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Vector<double> * particular_solutionU,
                   const Vector<double> * particular_solutionP,
                   const Function<dim> *  analytical_solutionU_in,
                   const Function<dim> *  analytical_solutionP_in,
                   const EquationData &   equation_data_in)
    : discrete_solutionU(particular_solutionU),
      discrete_solutionP(particular_solutionP),
      analytical_solutionU(analytical_solutionU_in),
      analytical_solutionP(analytical_solutionP_in),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cellU,
              const IteratorType & cellP,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  face_worker(const IteratorType & cellU,
              const IteratorType & cellP,
              const unsigned int & f,
              const unsigned int & sf,
              const IteratorType & ncellU,
              const IteratorType & ncellP,
              const unsigned int & nf,
              const unsigned int & nsf,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  boundary_worker(const IteratorType & cellU,
                  const IteratorType & cellP,
                  const unsigned int & f,
                  ScratchData<dim> &   scratch_data,
                  CopyData &           copy_data) const;

  const Vector<double> * discrete_solutionU;
  const Vector<double> * discrete_solutionP;
  const Function<dim> *  analytical_solutionU;
  const Function<dim> *  analytical_solutionP;
  const EquationData     equation_data;
};

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cellU,
                                                 const IteratorType & cellP,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  copy_data.cell_matrix         = 0.;
  copy_data.cell_matrix_flipped = 0.;
  copy_data.cell_rhs_test       = 0.;
  copy_data.cell_rhs_ansatz     = 0.;

  /// Velocity "U" takes test function role (in flipped mode ansatz function)
  auto & phiU = scratch_data.fe_values_test;
  phiU.reinit(cellU);
  cellU->get_active_or_mg_dof_indices(copy_data.local_dof_indices_test);
  const auto n_dofs_per_cellU = phiU.get_fe().dofs_per_cell;
  AssertDimension(n_dofs_per_cellU, copy_data.local_dof_indices_test.size());

  /// Pressure "P" takes ansatz function role (in flipped mode test function)
  auto & phiP = scratch_data.fe_values_ansatz;
  phiP.reinit(cellP);
  cellP->get_active_or_mg_dof_indices(copy_data.local_dof_indices_ansatz);
  const auto n_dofs_per_cellP = phiP.get_fe().dofs_per_cell;
  AssertDimension(n_dofs_per_cellP, copy_data.local_dof_indices_ansatz.size());

  const unsigned int n_q_points = phiU.n_quadrature_points;
  AssertDimension(n_q_points, phiP.n_quadrature_points);

  for(unsigned int q = 0; q < n_q_points; ++q)
  {
    for(unsigned int i = 0; i < n_dofs_per_cellU; ++i)
    {
      const auto div_phiU_i = compute_divergence(phiU, i, q);
      for(unsigned int j = 0; j < n_dofs_per_cellP; ++j)
      {
        const auto phiP_j = phiP.shape_value(j, q);

        /// assign to velocity-pressure block
        copy_data.cell_matrix(i, j) += -div_phiU_i * phiP_j * phiP.JxW(q);
      }
    }
  }

  /// pressure-velocity block ("flipped") is the transpose of the
  /// velocity-pressure block
  for(unsigned int i = 0; i < n_dofs_per_cellU; ++i)
    for(unsigned int j = 0; j < n_dofs_per_cellP; ++j)
      copy_data.cell_matrix_flipped(j, i) = copy_data.cell_matrix(i, j);

  if(!is_multigrid && discrete_solutionU)
  {
    Vector<double> u0(n_dofs_per_cellU);
    for(auto i = 0U; i < u0.size(); ++i)
      u0(i) = (*discrete_solutionU)(copy_data.local_dof_indices_test[i]);
    Vector<double> w0(n_dofs_per_cellP);
    copy_data.cell_matrix_flipped.vmult(w0, u0);
    copy_data.cell_rhs_ansatz -= w0;
  }

  /// There is no need for this as we have not imposed boundary conditions on
  /// the pressure !!!
  // if(!is_multigrid && discrete_solutionP)
  // {
  //   Vector<double> p0(n_dofs_per_cellP);
  //   for(auto i = 0U; i < p0.size(); ++i)
  //     p0(i) = (*discrete_solutionP)(copy_data.local_dof_indices_ansatz[i]);
  //   Vector<double> w0(n_dofs_per_cellU);
  //   copy_data.cell_matrix.vmult(w0, p0);
  //   copy_data.cell_rhs_test -= w0;
  // }
}

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker(const IteratorType & cellU,
                                                 const IteratorType & cellP,
                                                 const unsigned int & f,
                                                 const unsigned int & sf,
                                                 const IteratorType & ncellU,
                                                 const IteratorType & ncellP,
                                                 const unsigned int & nf,
                                                 const unsigned int & nsf,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  /// Velocity "U" takes test function role (in flipped mode ansatz function)
  auto & phiU = scratch_data.fe_interface_values_test;
  phiU.reinit(cellU, f, sf, ncellU, nf, nsf);
  const auto n_dofsU = phiU.n_current_interface_dofs();

  /// Pressure "P" takes ansatz function role (in flipped mode test function)
  auto & phiP = scratch_data.fe_interface_values_ansatz;
  phiP.reinit(cellP, f, sf, ncellP, nf, nsf);
  const auto n_dofsP = phiP.n_current_interface_dofs();

  AssertDimension(phiU.n_quadrature_points, phiP.n_quadrature_points);

  copy_data.face_data.emplace_back();
  CopyData::FaceData & copy_data_face = copy_data.face_data.back();

  copy_data_face.joint_dof_indices_test   = phiU.get_interface_dof_indices();
  copy_data_face.joint_dof_indices_ansatz = phiP.get_interface_dof_indices();

  AssertDimension(n_dofsU, copy_data_face.joint_dof_indices_test.size());
  AssertDimension(n_dofsP, copy_data_face.joint_dof_indices_ansatz.size());
  copy_data_face.cell_matrix.reinit(n_dofsU, n_dofsP);

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < phiU.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < n_dofsU; ++i)
    {
      const auto & jump_phiU_i_dot_n = compute_vjump_dot_normal(phiU, i, q);

      for(unsigned int j = 0; j < n_dofsP; ++j)
      {
        const auto & av_phiP_j = phiP.average(j, q);

        integral_ijq = av_phiP_j * jump_phiU_i_dot_n * phiU.JxW(q);

        copy_data_face.cell_matrix(i, j) += integral_ijq;
      }
    }
  }

  /// pressure-velocity block ("flipped") is the transpose of the
  /// velocity-pressure block
  copy_data_face.cell_matrix_flipped.reinit(n_dofsP, n_dofsU);
  for(unsigned int i = 0; i < n_dofsU; ++i)
    for(unsigned int j = 0; j < n_dofsP; ++j)
      copy_data_face.cell_matrix_flipped(j, i) = copy_data_face.cell_matrix(i, j);
}

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker(const IteratorType & cellU,
                                                     const IteratorType & cellP,
                                                     const unsigned int & f,
                                                     ScratchData<dim> &   scratch_data,
                                                     CopyData &           copy_data) const
{
  /// Velocity "U" takes test function role (in flipped mode ansatz function)
  auto & phiU = scratch_data.fe_interface_values_test;
  phiU.reinit(cellU, f);
  const auto n_dofsU = phiU.n_current_interface_dofs();
  AssertDimension(n_dofsU, copy_data.local_dof_indices_test.size());

  /// Pressure "P" takes ansatz function role (in flipped mode test function)
  auto & phiP = scratch_data.fe_interface_values_ansatz;
  phiP.reinit(cellP, f);
  const auto n_dofsP = phiP.n_current_interface_dofs();
  AssertDimension(n_dofsP, copy_data.local_dof_indices_ansatz.size());

  AssertDimension(phiU.n_quadrature_points, phiP.n_quadrature_points);
  std::vector<double> velocity_solution_dot_normals;
  if(!is_multigrid)
  {
    Assert(analytical_solutionU, ExcMessage("analytical_solutionU is not set."));
    AssertDimension(analytical_solutionU->n_components, dim);
    const auto &                        q_points = phiU.get_quadrature_points();
    const std::vector<Tensor<1, dim>> & normals  = phiU.get_normal_vectors();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   normals.cbegin(),
                   std::back_inserter(velocity_solution_dot_normals),
                   [this](const auto & x_q, const auto & normal) {
                     Tensor<1, dim> u_q;
                     for(auto c = 0U; c < dim; ++c)
                       u_q[c] = analytical_solutionU->value(x_q, c);
                     return u_q * normal;
                   });
  }

  AssertDimension(n_dofsP, copy_data.cell_rhs_ansatz.size());
  AssertDimension(n_dofsU, copy_data.cell_matrix.m());
  AssertDimension(n_dofsP, copy_data.cell_matrix.n());
  Assert(
    copy_data.local_dof_indices_ansatz == phiP.get_interface_dof_indices(),
    ExcMessage(
      "the local dof indices in copy_data are incompatible with the interface dof indices of phiP."));
  Assert(
    copy_data.local_dof_indices_test == phiU.get_interface_dof_indices(),
    ExcMessage(
      "the local dof indices in copy_data are incompatible with the interface dof indices of phiU."));

  double integral_ijq = 0.;
  double integral_jq  = 0.;
  for(unsigned int q = 0; q < phiP.n_quadrature_points; ++q)
  {
    for(unsigned int j = 0; j < n_dofsP; ++j)
    {
      const auto & av_phiP_j = phiP.average(j, q);

      /// Nitsche method (weak Dirichlet conditions)
      if(!is_multigrid) // here P is test function
      {
        const auto & u_dot_n = velocity_solution_dot_normals[q];

        integral_jq = u_dot_n * av_phiP_j * phiP.JxW(q);

        copy_data.cell_rhs_ansatz(j) += integral_jq;
      }

      for(unsigned int i = 0; i < n_dofsU; ++i)
      {
        /// IP method
        const auto & jump_phiU_i_dot_n = compute_vjump_dot_normal(phiU, i, q);

        integral_ijq = av_phiP_j * jump_phiU_i_dot_n * phiU.JxW(q);

        copy_data.cell_matrix(i, j) += integral_ijq;
      }
    }
  }

  /// The pressure-velocity block ("flipped") is the transpose of the
  /// velocity-pressure block.
  AssertDimension(n_dofsP, copy_data.cell_matrix_flipped.m());
  AssertDimension(n_dofsU, copy_data.cell_matrix_flipped.n());
  for(unsigned int i = 0; i < n_dofsU; ++i)
    for(unsigned int j = 0; j < n_dofsP; ++j)
      copy_data.cell_matrix_flipped(j, i) = copy_data.cell_matrix(i, j);
}

} // namespace Mixed

} // namespace MW



namespace FD
{
/**
 * Assembles the (exact) local matrices/solvers by exploiting the tensor
 * structure of each scalar-valued shape function, that is each block of a
 * patch matrix involving a component of the velocity vector-field and/or a
 * pressure function have a low-rank Kronecker product decomposition (representation?).
 *
 * However, each block matrix itself has no low-rank Kronecker decomposition
 * (representation?), thus, local matrices are stored and inverted in a
 * standard (vectorized) fashion.
 */
template<int dim,
         int fe_degree_p,
         typename Number              = double,
         TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q,
         int             fe_degree_v  = fe_degree_p + 1>
class MatrixIntegratorTensor
{
public:
  using This = MatrixIntegratorTensor<dim, fe_degree_p, Number>;

  static constexpr int n_q_points_1d = fe_degree_v + 1;

  using value_type              = Number;
  using transfer_type           = typename TPSS::PatchTransferBlock<dim, Number>;
  using matrix_type_1d          = Table<2, VectorizedArray<Number>>;
  using matrix_type             = MatrixAsTable<VectorizedArray<Number>>;
  using matrix_type_mixed       = Tensors::BlockMatrix<dim, VectorizedArray<Number>, -1, -1>;
  using velocity_evaluator_type = FDEvaluation<dim, fe_degree_v, n_q_points_1d, Number>;
  using pressure_evaluator_type = FDEvaluation<dim, fe_degree_p, n_q_points_1d, Number>;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> &       subdomain_handler,
                             std::vector<matrix_type> &                  local_matrices,
                             const OperatorType &                        dummy_operator,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    using MatrixIntegratorVelocity =
      Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree_v, Number, dof_layout_v>;
    static_assert(std::is_same<typename MatrixIntegratorVelocity::evaluator_type,
                               velocity_evaluator_type>::value,
                  "Velocity evaluator types mismatch.");
    using matrix_type_velocity = typename MatrixIntegratorVelocity::matrix_type;

    /// Assemble local matrices for the local velocity-velocity block.
    std::vector<matrix_type_velocity> local_matrices_velocity(local_matrices.size());
    {
      MatrixIntegratorVelocity matrix_integrator;
      matrix_integrator.initialize(equation_data);

      matrix_integrator.template assemble_subspace_inverses<OperatorType>(subdomain_handler,
                                                                          local_matrices_velocity,
                                                                          dummy_operator,
                                                                          subdomain_range);
    }

    /// Assemble local matrices for the local pressure-pressure block
    {
      /// This block is zero.
    }

    /// Assemble local matrices for the local velocity-pressure block
    std::vector<matrix_type_mixed> local_matrices_velocity_pressure(local_matrices.size());
    {
      assemble_mixed_subspace_inverses<OperatorType>(subdomain_handler,
                                                     local_matrices_velocity_pressure,
                                                     dummy_operator,
                                                     subdomain_range);
    }

    AssertDimension(local_matrices_velocity.size(), local_matrices.size());
    const auto patch_transfer = get_patch_transfer(subdomain_handler);
    for(auto patch_index = subdomain_range.first; patch_index < subdomain_range.second;
        ++patch_index)
    {
      const auto & local_block_velocity          = local_matrices_velocity[patch_index];
      const auto & local_block_velocity_pressure = local_matrices_velocity_pressure[patch_index];

      patch_transfer->reinit(patch_index);
      const auto n_dofs          = patch_transfer->n_dofs_per_patch();
      const auto n_dofs_velocity = local_block_velocity.m();
      const auto n_dofs_pressure = local_block_velocity_pressure.n();
      AssertDimension(patch_transfer->n_dofs_per_patch(0), n_dofs_velocity);
      (void)n_dofs_pressure;
      AssertDimension(patch_transfer->n_dofs_per_patch(1), n_dofs_pressure);

      auto & local_matrix = local_matrices[patch_index];
      local_matrix.as_table().reinit(n_dofs, n_dofs);

      /// velocity-velocity
      local_matrix.fill_submatrix(local_block_velocity.as_table(), 0U, 0U);

      /// velocity-pressure
      local_matrix.fill_submatrix(local_block_velocity_pressure.as_table(), 0U, n_dofs_velocity);

      /// pressure-velocity
      local_matrix.template fill_submatrix<true>(local_block_velocity_pressure.as_table(),
                                                 n_dofs_velocity,
                                                 0U);

      local_matrix.invert({equation_data.local_kernel_size, equation_data.local_kernel_threshold});
      // {
      //   for(auto b = 0U; b < dim; ++b)
      //   {
      //     auto lane = 0U;
      //     std::cout << "block: " << b << ", " << 0 << "   patch: " << patch_index
      //               << "   lane: " << lane << std::endl;
      //     const auto & rank1_tensors =
      //       local_block_velocity_pressure.get_block(b, 0).get_elementary_tensors();
      //     for(auto d = 0U; d < dim; ++d)
      //     {
      //       const auto & matrix_d = rank1_tensors[0][d];
      //       std::cout << "direction: " << d << std::endl;
      //       table_to_fullmatrix(matrix_d, lane).print_formatted(std::cout);
      //     }
      //   }
      // }
    }
  }

  template<typename OperatorType>
  void
  assemble_mixed_subspace_inverses(
    const SubdomainHandler<dim, Number> & subdomain_handler,
    std::vector<matrix_type_mixed> &      local_matrices,
    const OperatorType &,
    const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    for(auto compU = 0U; compU < dim; ++compU)
    {
      velocity_evaluator_type eval_velocity(subdomain_handler, /*dofh_index*/ 0, compU);
      pressure_evaluator_type eval_pressure(subdomain_handler, /*dofh_index*/ 1, /*component*/ 0);

      for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
      {
        auto & velocity_pressure_matrix = local_matrices[patch];
        if(velocity_pressure_matrix.n_block_rows() == 0 &&
           velocity_pressure_matrix.n_block_cols() == 0)
          velocity_pressure_matrix.resize(dim, 1);

        eval_velocity.reinit(patch);
        eval_pressure.reinit(patch);

        const auto mass_matrices =
          assemble_mass_tensor(/*test*/ eval_velocity, /*ansatz*/ eval_pressure);
        /// Note that we have flipped ansatz and test functions roles. The
        /// divergence of the velocity test functions is obtained by
        /// transposing gradient matrices.
        const auto gradient_matrices =
          assemble_gradient_tensor(/*test*/ eval_pressure, /*ansatz*/ eval_velocity);

        const auto & MxMxGT = [&](const unsigned int direction_of_div) {
          /// For example, we obtain MxMxGT for direction_of_div = 0 (dimension
          /// 0 is rightmost!)
          std::array<matrix_type_1d, dim> kronecker_tensor;
          for(auto d = 0U; d < dim; ++d)
            kronecker_tensor[d] = d == direction_of_div ?
                                    -1. * Tensors::transpose(gradient_matrices[direction_of_div]) :
                                    mass_matrices[d];
          return kronecker_tensor;
        };

        std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
        rank1_tensors.emplace_back(MxMxGT(compU));
        velocity_pressure_matrix.get_block(compU, 0).reinit(rank1_tensors);
      }
    }
  }

  std::array<matrix_type_1d, dim>
  assemble_mass_tensor(velocity_evaluator_type & eval_test,
                       pressure_evaluator_type & eval_ansatz) const
  {
    using CellMass = ::FD::L2::CellOperation<dim, fe_degree_v, n_q_points_1d, Number, fe_degree_p>;
    return eval_test.patch_action(eval_ansatz, CellMass{});
  }

  /**
   * We remark that the velocity takes the ansatz function role here
   * (Gradient::CellOperation derives the ansatz function) although in
   * assemble_mixed_subspace_inverses() we require the divergence of the
   * velocity test functions. Therefore, we transpose the returned tensor of
   * matrices.
   */
  std::array<matrix_type_1d, dim>
  assemble_gradient_tensor(pressure_evaluator_type & eval_test,
                           velocity_evaluator_type & eval_ansatz) const
  {
    using CellGradient =
      ::FD::Gradient::CellOperation<dim, fe_degree_p, n_q_points_1d, Number, fe_degree_v>;
    CellGradient cell_gradient;
    return eval_test.patch_action(eval_ansatz, cell_gradient);
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};



/**
 * This class is actual not an "integration-type" struct for local matrices. It
 * simply uses PatchTransferBlock w.r.t. the velocity-pressure block system to
 * extract the local matrices/solvers from the (global) level matrix. The level
 * matrix is passed as argument to assemble_subspace_inverses().
 *
 * Therefore, all local matrices are stored and inverted in a classical
 * fashion.
 */
template<int dim,
         int fe_degree_p,
         typename Number              = double,
         TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q,
         int             fe_degree_v  = fe_degree_p + 1>
class MatrixIntegratorCut
{
public:
  using This = MatrixIntegratorCut<dim, fe_degree_p, Number>;

  static constexpr int n_q_points_1d = fe_degree_v + 1;

  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransferBlock<dim, Number>;
  using matrix_type   = Tensors::BlockMatrixBasic2x2<MatrixAsTable<VectorizedArray<Number>>>;
  using operator_type = BlockSparseMatrix<Number>;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> &       subdomain_handler,
                             std::vector<matrix_type> &                  local_matrices,
                             const operator_type &                       level_matrix,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    const auto                           patch_transfer = get_patch_transfer(subdomain_handler);
    typename matrix_type::AdditionalData additional_data;
    additional_data.basic_inverse = {equation_data.local_kernel_size,
                                     equation_data.local_kernel_threshold};

    FullMatrix<double> tmp_v_v;
    FullMatrix<double> tmp_p_p;
    FullMatrix<double> tmp_v_p;
    FullMatrix<double> tmp_p_v;

    for(auto patch_index = subdomain_range.first; patch_index < subdomain_range.second;
        ++patch_index)
    {
      patch_transfer->reinit(patch_index);
      const auto   n_dofs                = patch_transfer->n_dofs_per_patch();
      const auto   n_dofs_velocity       = patch_transfer->n_dofs_per_patch(0);
      const auto   n_dofs_pressure       = patch_transfer->n_dofs_per_patch(1);
      const auto & patch_worker_velocity = patch_transfer->get_patch_dof_worker(0);

      matrix_type & patch_matrix = local_matrices[patch_index];

      auto & local_block_velocity = patch_matrix.get_block(0U, 0U);
      local_block_velocity.as_table().reinit(n_dofs_velocity, n_dofs_velocity);
      tmp_v_v.reinit(n_dofs_velocity, n_dofs_velocity);

      auto & local_block_pressure = patch_matrix.get_block(1U, 1U);
      local_block_pressure.as_table().reinit(n_dofs_pressure, n_dofs_pressure);
      tmp_p_p.reinit(n_dofs_pressure, n_dofs_pressure);

      auto & local_block_velocity_pressure = patch_matrix.get_block(0U, 1U);
      local_block_velocity_pressure.as_table().reinit(n_dofs_velocity, n_dofs_pressure);
      tmp_v_p.reinit(n_dofs_velocity, n_dofs_pressure);

      auto & local_block_pressure_velocity = patch_matrix.get_block(1U, 0U);
      local_block_pressure_velocity.as_table().reinit(n_dofs_pressure, n_dofs_velocity);
      tmp_p_v.reinit(n_dofs_pressure, n_dofs_velocity);

      for(auto lane = 0U; lane < patch_worker_velocity.n_lanes_filled(patch_index); ++lane)
      {
        /// Patch-wise local and global dof indices of velocity block.
        const auto & patch_transfer_velocity = patch_transfer->get_patch_transfer(0);
        std::vector<types::global_dof_index> velocity_dof_indices_on_patch;
        {
          const auto view = patch_transfer_velocity.get_dof_indices(lane);
          std::copy(view.cbegin(), view.cend(), std::back_inserter(velocity_dof_indices_on_patch));
        }

        /// Patch-wise local and global dof indices of pressure block.
        const auto & patch_transfer_pressure = patch_transfer->get_patch_transfer(1);
        std::vector<types::global_dof_index> pressure_dof_indices_on_patch;
        {
          const auto view = patch_transfer_pressure.get_dof_indices(lane);
          std::copy(view.cbegin(), view.cend(), std::back_inserter(pressure_dof_indices_on_patch));
        }

        /// velocity block
        if(equation_data.local_solver == LocalSolver::Vdiag)
        {
          for(auto comp = 0U; comp < dim; ++comp)
          {
            std::vector<types::global_dof_index> velocity_dof_indices_per_comp;
            const auto view = patch_transfer_velocity.get_dof_indices(lane, comp);
            std::copy(view.cbegin(),
                      view.cend(),
                      std::back_inserter(velocity_dof_indices_per_comp));
            const auto n_velocity_dofs_per_comp = velocity_dof_indices_per_comp.size();

            tmp_v_v.reinit(n_velocity_dofs_per_comp, n_velocity_dofs_per_comp);
            tmp_v_v.extract_submatrix_from(level_matrix.block(0U, 0U),
                                           velocity_dof_indices_per_comp,
                                           velocity_dof_indices_per_comp);

            const auto start = comp * n_velocity_dofs_per_comp;
            local_block_velocity.fill_submatrix(tmp_v_v, start, start, lane);
          }
        }
        else
        {
          tmp_v_v.extract_submatrix_from(level_matrix.block(0U, 0U),
                                         velocity_dof_indices_on_patch,
                                         velocity_dof_indices_on_patch);
          local_block_velocity.fill_submatrix(tmp_v_v, 0U, 0U, lane);
        }

        /// pressure block
        tmp_p_p.extract_submatrix_from(level_matrix.block(1U, 1U),
                                       pressure_dof_indices_on_patch,
                                       pressure_dof_indices_on_patch);
        local_block_pressure.fill_submatrix(tmp_p_p, 0U, 0U, lane);

        /// velocity-pressure block
        tmp_v_p.extract_submatrix_from(level_matrix.block(0U, 1U),
                                       velocity_dof_indices_on_patch,
                                       pressure_dof_indices_on_patch);
        local_block_velocity_pressure.fill_submatrix(tmp_v_p, 0U, 0U, lane);

        /// pressure-velocity block
        tmp_p_v.extract_submatrix_from(level_matrix.block(1U, 0U),
                                       pressure_dof_indices_on_patch,
                                       velocity_dof_indices_on_patch);
        local_block_pressure_velocity.fill_submatrix(tmp_p_v, 0U, 0U, lane);
      }

      (void)n_dofs;
      AssertDimension(patch_matrix.m(), n_dofs);
      AssertDimension(patch_matrix.n(), n_dofs);

      patch_matrix.invert(additional_data);
    }
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};



/**
 * Selects the MatrixIntegrator at compile time w.r.t. the LocalAssembly
 * template. Hence, the generic class is empty and we select by template
 * specialization.
 */
template<LocalAssembly local_assembly,
         int           dim,
         int           fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v>
struct MatrixIntegratorSelector
{
};

template<int dim, int fe_degree_p, typename Number, TPSS::DoFLayout dof_layout_v, int fe_degree_v>
struct MatrixIntegratorSelector<LocalAssembly::Tensor,
                                dim,
                                fe_degree_p,
                                Number,
                                dof_layout_v,
                                fe_degree_v>
{
  using type = MatrixIntegratorTensor<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;
};

template<int dim, int fe_degree_p, typename Number, TPSS::DoFLayout dof_layout_v, int fe_degree_v>
struct MatrixIntegratorSelector<LocalAssembly::Cut,
                                dim,
                                fe_degree_p,
                                Number,
                                dof_layout_v,
                                fe_degree_v>
{
  using type = MatrixIntegratorCut<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;
};


/**
 * Aliasing MatrixIntegrator w.r.t. the choice for LocalAssembly.
 */
template<int dim,
         int fe_degree_p,
         typename Number                = double,
         TPSS::DoFLayout dof_layout_v   = TPSS::DoFLayout::Q,
         int             fe_degree_v    = fe_degree_p + 1,
         LocalAssembly   local_assembly = LocalAssembly::Tensor>
using MatrixIntegrator = typename MatrixIntegratorSelector<local_assembly,
                                                           dim,
                                                           fe_degree_p,
                                                           Number,
                                                           dof_layout_v,
                                                           fe_degree_v>::type;

} // end namespace FD

} // namespace VelocityPressure

} // namespace Stokes

#endif /* APPS_STOKESINTEGRATOR_H_ */
