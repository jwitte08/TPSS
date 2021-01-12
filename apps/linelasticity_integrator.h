/*
 * linelasticity_integrator.h
 *
 *  Created on: Jun 25, 2019
 *      Author: witte
 */

#ifndef TESTS_LINELASTICITYINTEGRATOR_H_
#define TESTS_LINELASTICITYINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

#include "solvers_and_preconditioners/TPSS/block_matrix.h"

#include "equation_data.h"
#include "laplace_integrator.h"

using namespace dealii;



namespace LinElasticity
{
namespace MW
{
template<int dim>
class MatrixIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  MatrixIntegrator(const EquationData & equation_data_in)
    : equation_data(equation_data_in),
      mu(equation_data_in.mu),
      lambda(equation_data_in.lambda),
      ip_factor(equation_data_in.ip_factor)
  {
  }

  void
  cell(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
       typename dealii::MeshWorker::IntegrationInfo<dim> & info) const;
  void
  boundary(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
           typename dealii::MeshWorker::IntegrationInfo<dim> & info) const;
  void
  face(dealii::MeshWorker::DoFInfo<dim> &                  dinfo1,
       dealii::MeshWorker::DoFInfo<dim> &                  dinfo2,
       typename dealii::MeshWorker::IntegrationInfo<dim> & info1,
       typename dealii::MeshWorker::IntegrationInfo<dim> & info2) const;

private:
  EquationData equation_data;
  double       mu;
  double       lambda;
  double       ip_factor;
};

template<int dim>
inline void
MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
  if(equation_data.integrator_variant != EquationData::IntegratorVariant::graddiv)
    dealii::LocalIntegrators::Elasticity::cell_matrix(dinfo.matrix(0, false).matrix,
                                                      info.fe_values(0),
                                                      2. * mu);
  if(equation_data.integrator_variant != EquationData::IntegratorVariant::strain)
    dealii::LocalIntegrators::GradDiv::cell_matrix(dinfo.matrix(0, false).matrix,
                                                   info.fe_values(0),
                                                   lambda);
}

template<int dim>
inline void
MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
                                typename dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if(equation_data.integrator_variant != EquationData::IntegratorVariant::graddiv)
    dealii::LocalIntegrators::Elasticity::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      ip_factor * dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * mu);
  if(equation_data.integrator_variant != EquationData::IntegratorVariant::strain)
    dealii::LocalIntegrators::GradDiv::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      ip_factor * dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      lambda);
}

template<int dim>
inline void
MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &                  dinfo1,
                            dealii::MeshWorker::DoFInfo<dim> &                  dinfo2,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info1,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  if(equation_data.integrator_variant != EquationData::IntegratorVariant::graddiv)
    dealii::LocalIntegrators::Elasticity::ip_matrix(
      dinfo1.matrix(0, false).matrix,
      dinfo1.matrix(0, true).matrix,
      dinfo2.matrix(0, true).matrix,
      dinfo2.matrix(0, false).matrix,
      info1.fe_values(0),
      info2.fe_values(0),
      ip_factor * dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      2. * mu);
  if(equation_data.integrator_variant != EquationData::IntegratorVariant::strain)
    dealii::LocalIntegrators::GradDiv::ip_matrix(
      dinfo1.matrix(0, false).matrix,
      dinfo1.matrix(0, true).matrix,
      dinfo2.matrix(0, true).matrix,
      dinfo2.matrix(0, false).matrix,
      info1.fe_values(0),
      info2.fe_values(0),
      ip_factor * dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      lambda);
}

} // end namespace MW



namespace FD
{
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using value_type                     = Number;
  using transfer_type                  = typename TPSS::PatchTransferBlock<dim, Number>;
  static constexpr int n_patch_dofs_1d = -1;
  using BlockMatrixDiagonal =
    typename Tensors::BlockMatrixDiagonal<dim, VectorizedArray<Number>, n_patch_dofs_1d>;
  using BlockMatrix =
    typename Tensors::BlockMatrix<dim, VectorizedArray<Number>, -1, n_patch_dofs_1d>;
  using BlockMatrixFast =
    typename Tensors::BlockMatrix<dim, VectorizedArray<Number>, 0, n_patch_dofs_1d>;
  using EvaluatorType        = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;
  using VectorizedMatrixType = Table<2, VectorizedArray<Number>>;

private:
  using This                            = MatrixIntegrator<dim, fe_degree, Number>;
  static constexpr int fe_order         = fe_degree + 1;
  static constexpr int macro_size       = VectorizedArray<Number>::size();
  static constexpr int n_blocks_offdiag = dim * (dim - 1);

private:
  EquationData equation_data;

public:
  MatrixIntegrator() = default;

  MatrixIntegrator(const EquationData & equation_data_in)
  {
    initialize(equation_data_in);
  }

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  VectorizedArray<Number>
  compute_penalty(const EvaluatorType & eval_test,
                  const int             direction,
                  const int             cell_no,
                  const int             cell_no_neighbor) const
  {
    return Laplace::DG::FD::compute_penalty(eval_test, direction, cell_no, cell_no_neighbor);
  }

  double
  ip_factor_strain() const
  {
    if(equation_data.ip_variant == EquationData::PenaltyVariant::tensor)
      return equation_data.ip_factor * equation_data.lambda;
    return equation_data.ip_factor;
  }

  double
  ip_factor_graddiv() const
  {
    if(equation_data.ip_variant == EquationData::PenaltyVariant::tensor)
      return equation_data.ip_factor;
    return equation_data.ip_factor;
  }

  const EquationData &
  get_equation_data() const
  {
    return equation_data;
  }

  template<typename Evaluator>
  struct Void
  {
    void
    operator()(const Evaluator &,
               const Evaluator &,
               Table<2, VectorizedArray<Number>> &,
               const int,
               const int) const
    {
      // *** does nothing
    }

    void
    operator()(const Evaluator &,
               const Evaluator &,
               Table<2, VectorizedArray<Number>> &,
               const int,
               const int,
               const int,
               const std::bitset<macro_size>) const
    {
      /// does nothing
    }

    void
    operator()(const Evaluator &,
               const Evaluator &,
               Table<2, VectorizedArray<Number>> &,
               Table<2, VectorizedArray<Number>> &,
               const int) const
    {
      /// does nothing
    }
  };

  template<typename Evaluator>
  struct CellStrain
  {
    CellStrain(const MatrixIntegrator * integrator_in, const int component_in)
      : integrator(integrator_in), component(component_in)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const MatrixIntegrator * integrator;
    const int                component;
  };

  template<typename Evaluator>
  struct CellMass
  {
    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;
  };

  // d/dx_i u_j * d/dx_j v_i   with i =/= j
  template<typename Evaluator>
  struct CellDerivative
  {
    CellDerivative(const int deriv_index_u, const int deriv_index_v)
      : partial_derivative_index_u(deriv_index_u), partial_derivative_index_v(deriv_index_v)
    {
      AssertIndexRange(partial_derivative_index_u, dim);
      AssertIndexRange(partial_derivative_index_v, dim);
      Assert(partial_derivative_index_u != partial_derivative_index_v,
             ExcMessage("Implemented mixed derivatives only"));
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const int partial_derivative_index_u; // i
    const int partial_derivative_index_v; // j
  };

  template<typename Evaluator>
  struct CellGradDiv
  {
    CellGradDiv(const MatrixIntegrator * integrator_in, const int component_in)
      : integrator(integrator_in), component(component_in)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const MatrixIntegrator * integrator;
    const int                component;
  };

  template<typename Evaluator>
  struct NitscheStrain
  {
    NitscheStrain(const MatrixIntegrator * integrator_in, const int component_in)
      : integrator(integrator_in), component(component_in)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no) const;
    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           cell_no_left,
               const int                           direction) const;

    const MatrixIntegrator * integrator;
    const int                component;
  };

  template<typename Evaluator>
  struct NitscheStrainMixed
  {
    NitscheStrainMixed(const MatrixIntegrator * integrator_in, const int comp_u, const int comp_v)
      : integrator(integrator_in), component_u(comp_u), component_v(comp_v)
    {
      AssertIndexRange(component_u, dim);
      AssertIndexRange(component_v, dim);
      Assert(component_u != component_v, ExcInvalidState());
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no) const;

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           cell_no_left,
               const int                           direction) const;

    const MatrixIntegrator * integrator;
    const int                component_u;
    const int                component_v;
  };

  template<typename Evaluator>
  struct NitscheGradDiv
  {
    NitscheGradDiv(const MatrixIntegrator * integrator_in, const int component_in)
      : integrator(integrator_in), component(component_in)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no) const;

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           cell_no_left,
               const int                           direction) const;

    const MatrixIntegrator * integrator;
    const int                component;
  };

  template<typename Evaluator>
  struct NitscheGradDivMixed
  {
    NitscheGradDivMixed(const MatrixIntegrator * integrator_in, const int comp_u, const int comp_v)
      : integrator(integrator_in), component_u(comp_u), component_v(comp_v)
    {
      AssertIndexRange(component_u, dim);
      AssertIndexRange(component_v, dim);
      Assert(component_u != component_v, ExcInvalidState());
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no) const;

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           cell_no_left,
               const int                           direction) const;

    const MatrixIntegrator * integrator;
    const int                component_u;
    const int                component_v;
  };

  std::array<std::array<VectorizedMatrixType, dim>, dim>
  assemble_mass_tensors(std::vector<std::shared_ptr<EvaluatorType>> & eval_tests,
                        const std::vector<CellMass<EvaluatorType>> &  cell_mass_operations,
                        const unsigned int                            patch_id) const
  {
    AssertDimension(eval_tests.size(), dim);
    AssertDimension(cell_mass_operations.size(), dim);
    std::array<std::array<VectorizedMatrixType, dim>, dim> tensors;
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      auto & matrices  = tensors[comp];
      auto & eval_test = *(eval_tests[comp]);
      eval_test.reinit(patch_id);
      eval_test.evaluate(true, false);

      matrices = eval_test.patch_action(cell_mass_operations[comp]);
    }
    return tensors;
  }

  std::array<std::array<VectorizedMatrixType, dim>, dim>
  assemble_strain_tensors(
    std::vector<std::shared_ptr<EvaluatorType>> &     eval_tests,
    const std::vector<CellStrain<EvaluatorType>> &    cell_strain_operations,
    const std::vector<NitscheStrain<EvaluatorType>> & nitsche_strain_operations,
    const unsigned int                                patch_id) const
  {
    AssertDimension(eval_tests.size(), dim);
    AssertDimension(cell_strain_operations.size(), dim);
    AssertDimension(nitsche_strain_operations.size(), dim);
    std::array<std::array<VectorizedMatrixType, dim>, dim> tensors;
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      auto & matrices  = tensors[comp];
      auto & eval_test = *(eval_tests[comp]);
      eval_test.reinit(patch_id);
      eval_test.evaluate(true, false);

      matrices = eval_test.patch_action(cell_strain_operations[comp],
                                        nitsche_strain_operations[comp],
                                        nitsche_strain_operations[comp]);
      // /// DEBUG
      // const Void<EvaluatorType>                          void_op;
      // matrices = eval_test.patch_action(cell_strain_operations[comp],
      //                              void_op,
      // 				   void_op);
    }
    return tensors;
  }

  std::vector<std::array<VectorizedMatrixType, dim>>
  assemble_mixed_block(std::vector<std::shared_ptr<EvaluatorType>> & eval_tests,
                       const EquationData &                          equation_data,
                       const int                                     component_v,
                       const int                                     component_u,
                       const unsigned int                            patch_id) const
  {
    AssertThrow(dim == 2, ExcMessage("TODO"));
    AssertIndexRange(component_v, dim);
    AssertIndexRange(component_u, dim);

    const Void<EvaluatorType>                          void_op;
    std::vector<std::array<VectorizedMatrixType, dim>> elementary_tensors;

    {                                                     /// LINEAR STRAIN
      const int partial_derivative_index_u = component_v; // i
      const int partial_derivative_index_v = component_u; // j
      /// compute the univariate matrices subject to the mixed gradients
      /// (d/dx_j v_i) * (d/dx_i u_j)
      const CellDerivative<EvaluatorType> derivative_op{partial_derivative_index_u,
                                                        partial_derivative_index_v};
      /// compute the complementing matrices subject to the nitsche-strain
      /// contributions (point evaluations)
      const NitscheStrainMixed<EvaluatorType> nitsche_op(this, component_u, component_v);

      auto & eval_test   = *(eval_tests[component_v]);
      auto & eval_ansatz = *(eval_tests[component_u]);
      eval_ansatz.reinit(patch_id);
      eval_ansatz.evaluate(true, false);
      eval_test.reinit(patch_id);
      eval_test.evaluate(true, false);

      const auto & tensor_derivative = eval_test.patch_action(eval_ansatz, derivative_op);
      const auto & tensor_nitsche =
        eval_test.patch_action(eval_ansatz, void_op, nitsche_op, nitsche_op);

      /// (mu *  G(1)^T + N(1)) x G(0)   NOTE: mu is included in Nitsche N(1)
      const auto & mu_derivativeT = LinAlg::scaling(equation_data.mu, tensor_derivative[1]);
      elementary_tensors.emplace_back(
        std::array<VectorizedMatrixType, dim>{tensor_derivative[0], mu_derivativeT});
      elementary_tensors.emplace_back(
        std::array<VectorizedMatrixType, dim>{tensor_derivative[0], tensor_nitsche[1]});

      /// G(1)^T x N(0)                  NOTE: mu is included in Nitsche N(0)
      elementary_tensors.emplace_back(
        std::array<VectorizedMatrixType, dim>{tensor_nitsche[0], tensor_derivative[1]});
    }

    {                                                     /// GRAD-DIV
      const int partial_derivative_index_v = component_v; // i
      const int partial_derivative_index_u = component_u; // j
      /// computes the univariate matrices subject to the mixed divergence
      /// (d/dx_i v_i) * (d/dx_j u_j) ***/
      const CellDerivative<EvaluatorType> derivative_op{partial_derivative_index_u,
                                                        partial_derivative_index_v};
      /// computes the complementing matrices of the nitsche - grad-div
      /// contributions (point eval_testuations)
      const NitscheGradDivMixed<EvaluatorType> nitsche_op(this, component_u, component_v);

      auto & eval_test   = *(eval_tests[component_v]);
      auto & eval_ansatz = *(eval_tests[component_u]);
      eval_ansatz.reinit(patch_id);
      eval_ansatz.evaluate(true, false);
      eval_test.reinit(patch_id);
      eval_test.evaluate(true, false);

      const auto & tensor_derivative = eval_test.patch_action(eval_ansatz, derivative_op);
      const auto & tensor_nitsche =
        eval_test.patch_action(eval_ansatz, void_op, nitsche_op, nitsche_op);

      /// (lambda * G(1) + N(1)) x G(0)^T    NOTE: lambda is included in Nitsche N(1)
      const auto & lambda_derivative = LinAlg::scaling(equation_data.lambda, tensor_derivative[1]);
      elementary_tensors.emplace_back(
        std::array<VectorizedMatrixType, dim>{tensor_derivative[0], lambda_derivative});
      elementary_tensors.emplace_back(
        std::array<VectorizedMatrixType, dim>{tensor_derivative[0], tensor_nitsche[1]});

      /// G(1) x N(0)                        NOTE: lambda is included in Nitsche N(0)
      elementary_tensors.emplace_back(
        std::array<VectorizedMatrixType, dim>{tensor_nitsche[0], tensor_derivative[1]});
    }

    return elementary_tensors;
  }

  void
  assemble_graddiv_tensors(
    std::array<std::array<VectorizedMatrixType, dim>, dim> & tensors,
    std::vector<std::shared_ptr<EvaluatorType>> &            eval_tests,
    const std::vector<CellGradDiv<EvaluatorType>> &          cell_graddiv_operations,
    const std::vector<NitscheGradDiv<EvaluatorType>> &       nitsche_graddiv_operations,
    const unsigned int                                       patch_id) const
  {
    AssertDimension(eval_tests.size(), dim);
    AssertDimension(cell_graddiv_operations.size(), dim);
    AssertDimension(nitsche_graddiv_operations.size(), dim);
    const auto & sum_matrices = [](auto & A, const auto & B) {
      for(unsigned int d = 0; d < dim; ++d)
        A[d] = LinAlg::sum(A[d], B[d]);
    };

    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      auto & eval_test = *(eval_tests[comp]);
      eval_test.reinit(patch_id);
      eval_test.evaluate(true, false);

      const auto & matrices = eval_test.patch_action(cell_graddiv_operations[comp],
                                                     nitsche_graddiv_operations[comp],
                                                     nitsche_graddiv_operations[comp]);
      // /// DEBUG
      // const Void<EvaluatorType>                          void_op;
      // const auto & matrices = eval_test.patch_action(cell_graddiv_operations[comp],
      // 						void_op, void_op);

      sum_matrices(tensors[comp], matrices);
    }
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & data,
                             std::vector<BlockMatrixDiagonal> &    inverses,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    std::vector<std::shared_ptr<EvaluatorType>> eval_tests;
    for(unsigned int comp = 0; comp < dim; ++comp)
      eval_tests.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    std::vector<CellMass<EvaluatorType>> cell_mass_operations(dim);

    std::vector<CellStrain<EvaluatorType>> cell_strain_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      cell_strain_operations.emplace_back(this, comp);
    std::vector<NitscheStrain<EvaluatorType>> nitsche_strain_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      nitsche_strain_operations.emplace_back(/*equation_data*/ this, comp);

    std::vector<CellGradDiv<EvaluatorType>> cell_graddiv_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      cell_graddiv_operations.emplace_back(this, comp);
    std::vector<NitscheGradDiv<EvaluatorType>> nitsche_graddiv_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      nitsche_graddiv_operations.emplace_back(this, comp);

    for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    {
      auto & inverse = inverses[id];
      inverse.resize(dim);
      const auto mass_matrices = assemble_mass_tensors(eval_tests, cell_mass_operations, id);
      auto       elasticity_matrices =
        assemble_strain_tensors(eval_tests, cell_strain_operations, nitsche_strain_operations, id);
      assemble_graddiv_tensors(
        elasticity_matrices, eval_tests, cell_graddiv_operations, nitsche_graddiv_operations, id);

      for(unsigned int comp = 0; comp < dim; ++comp)
      {
        auto & blockmatrix = inverse.get_block(comp);
        blockmatrix.reinit(mass_matrices[comp], elasticity_matrices[comp]);
      }
    }
  }

  template<typename OperatorType, typename BlockMatrixType>
  void
  assemble_subspace_inverses_impl(const SubdomainHandler<dim, Number> & data,
                                  std::vector<BlockMatrixType> &        inverses,
                                  const OperatorType &,
                                  const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    using TensorProductState = typename BlockMatrixType::matrix_type::State;
    static_assert(dim == 2, "TODO");

    std::vector<std::shared_ptr<EvaluatorType>> eval_tests;
    for(unsigned int comp = 0; comp < dim; ++comp)
      eval_tests.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    std::vector<CellMass<EvaluatorType>> cell_mass_op(dim);

    std::vector<CellStrain<EvaluatorType>> cell_strain_op;
    for(unsigned int comp = 0; comp < dim; ++comp)
      cell_strain_op.emplace_back(this, comp);
    std::vector<NitscheStrain<EvaluatorType>> nitsche_strain_op;
    for(unsigned int comp = 0; comp < dim; ++comp)
      nitsche_strain_op.emplace_back(/*equation_data*/ this, comp);

    std::vector<CellGradDiv<EvaluatorType>> cell_graddiv_op;
    for(unsigned int comp = 0; comp < dim; ++comp)
      cell_graddiv_op.emplace_back(this, comp);
    std::vector<NitscheGradDiv<EvaluatorType>> nitsche_graddiv_op;
    for(unsigned int comp = 0; comp < dim; ++comp)
      nitsche_graddiv_op.emplace_back(this, comp);

    for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
    {
      auto & block_matrix = inverses[patch];
      block_matrix.resize(dim);
      block_matrix.set_lambda_rank(equation_data.lambda_rank);
      block_matrix.set_kronecker_rank(equation_data.kronecker_rank);
      block_matrix.set_schur_factor(equation_data.factor);
      auto mass_matrices = assemble_mass_tensors(eval_tests, cell_mass_op, patch);

      std::array<std::array<VectorizedMatrixType, dim>, dim> elasticity_matrices;
      elasticity_matrices =
        assemble_strain_tensors(eval_tests, cell_strain_op, nitsche_strain_op, patch);
      assemble_graddiv_tensors(
        elasticity_matrices, eval_tests, cell_graddiv_op, nitsche_graddiv_op, patch);

      /// block diagonal
      for(unsigned int comp = 0; comp < dim; ++comp)
      {
        std::vector<std::array<VectorizedMatrixType, dim>> elementary_tensors;
        elementary_tensors.emplace_back(mass_matrices[comp]);
        elementary_tensors.emplace_back(elasticity_matrices[comp]);
        block_matrix.get_block(comp, comp)
          .reinit(elementary_tensors, TensorProductState::separable);
      }

      /// block off-diagonals
      {
        auto mixed_tensors = assemble_mixed_block(
          eval_tests, equation_data, /*component_v*/ 1U, /*component_u*/ 0U, patch);
        block_matrix.get_block(1U, 0U).reinit(mixed_tensors);

        std::vector<std::array<VectorizedMatrixType, dim>> mixed_tensorsT;
        std::transform(mixed_tensors.cbegin(),
                       mixed_tensors.cend(),
                       std::back_inserter(mixed_tensorsT),
                       [](const auto & tensor) {
                         std::array<VectorizedMatrixType, dim> tensorT;
                         std::transform(tensor.cbegin(),
                                        tensor.cend(),
                                        tensorT.begin(),
                                        [](const auto & matrix) {
                                          return LinAlg::transpose(matrix);
                                        });
                         return tensorT;
                       });
        block_matrix.get_block(0U, 1U).reinit(mixed_tensorsT);
      }
    }
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> &       data,
                             std::vector<BlockMatrix> &                  inverses,
                             const OperatorType &                        op,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    assemble_subspace_inverses_impl<OperatorType, BlockMatrix>(data, inverses, op, subdomain_range);
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> &       data,
                             std::vector<BlockMatrixFast> &              inverses,
                             const OperatorType &                        op,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    assemble_subspace_inverses_impl<OperatorType, BlockMatrixFast>(data,
                                                                   inverses,
                                                                   op,
                                                                   subdomain_range);
  }
};

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellStrain<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no) const
{
  const Number            symgrad_factor = component == direction ? 1. : 0.5;
  const auto              mu             = integrator->equation_data.mu;
  VectorizedArray<Number> integral;
  for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < Evaluator::n_q_points_1d_static; ++q)
      {
        const auto & grad_u = eval_ansatz.shape_gradient(dof_u, q, direction, cell_no);
        const auto & grad_v = eval_test.shape_gradient(dof_v, q, direction, cell_no);
        const auto & dx     = eval_test.get_JxW(q, direction, cell_no);
        integral += grad_u * grad_v * dx;
      }
      cell_matrix(dof_v, dof_u) += 2. * mu * symgrad_factor * integral;
    }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellMass<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no) const
{
  VectorizedArray<Number> integral;
  for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < Evaluator::n_q_points_1d_static; ++q)
      {
        const auto & value_u = eval_ansatz.shape_value(dof_u, q, direction, cell_no);
        const auto & value_v = eval_test.shape_value(dof_v, q, direction, cell_no);
        const auto & dx      = eval_test.get_JxW(q, direction, cell_no);
        integral += value_u * value_v * dx;
      }
      cell_matrix(dof_v, dof_u) += integral;
    }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellDerivative<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no) const
{
  VectorizedArray<Number> integral;
  const bool              flag_derive_u = (direction == partial_derivative_index_u);
  const bool              flag_derive_v = (direction == partial_derivative_index_v);
  for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < Evaluator::n_q_points_1d_static; ++q)
      {
        const auto & value_or_grad_u = flag_derive_u ?
                                         eval_ansatz.shape_gradient(dof_u, q, direction, cell_no) :
                                         eval_ansatz.shape_value(dof_u, q, direction, cell_no);
        const auto & value_or_grad_v = flag_derive_v ?
                                         eval_test.shape_gradient(dof_v, q, direction, cell_no) :
                                         eval_test.shape_value(dof_v, q, direction, cell_no);
        const auto & dx = eval_test.get_JxW(q, direction, cell_no);
        integral += value_or_grad_u * value_or_grad_v * dx;
      }
      cell_matrix(dof_v, dof_u) += integral;
    }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellGradDiv<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no) const
{
  const auto lambda = integrator->equation_data.lambda;
  if(component == direction)
  {
    VectorizedArray<Number> integral;
    for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
      for(int dof_v = 0; dof_v < fe_order; ++dof_v)
      {
        integral = 0.;
        for(unsigned int q = 0; q < Evaluator::n_q_points_1d_static; ++q)
        {
          const auto & grad_u = eval_ansatz.shape_gradient(dof_u, q, direction, cell_no);
          const auto & grad_v = eval_test.shape_gradient(dof_v, q, direction, cell_no);
          const auto & dx     = eval_test.get_JxW(q, direction, cell_no);
          integral += grad_u * grad_v * dx;
        }
        cell_matrix(dof_v, dof_u) += lambda * integral;
      }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrain<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no,
  const int                           face_no) const
{
  const auto & equation_data  = integrator->equation_data;
  const auto   mu             = equation_data.mu;
  const auto   normal         = eval_test.get_normal(face_no);
  const Number symgrad_factor = (component == direction) ? 1. : 0.5;
  const auto   average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
  const auto   penalty        = integrator->ip_factor_strain() * average_factor *
                       integrator->compute_penalty(eval_test, direction, cell_no, cell_no);

  auto value_on_face = make_vectorized_array<Number>(0.);
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v      = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
    const auto & grad_v = eval_test.shape_gradient_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u      = eval_ansatz.shape_value_face(dof_u, face_no, direction, cell_no);
      const auto & grad_u = eval_ansatz.shape_gradient_face(dof_u, face_no, direction, cell_no);

      value_on_face =
        -average_factor * symgrad_factor * (v * normal * grad_u + grad_v * u * normal);

      if(equation_data.ip_variant == EquationData::PenaltyVariant::tensor)
        value_on_face += symgrad_factor * penalty * v * u * normal * normal;
      else
        value_on_face += penalty * v * u * normal * normal;

      cell_matrix(dof_v, dof_u) += 2. * mu * value_on_face;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrain<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix01,
  Table<2, VectorizedArray<Number>> & cell_matrix10,
  const int                           cell_no_left,
  const int                           direction) const
{
  (void)cell_no_left;
  AssertDimension(cell_no_left, 0);
  const auto & equation_data  = integrator->equation_data;
  const auto   mu             = equation_data.mu;
  const auto   average_factor = eval_test.get_average_factor(direction, 0, 1);
  const auto   normal0        = eval_test.get_normal(1); // on cell 0
  const auto   normal1        = eval_test.get_normal(0); // on cell 1
  const auto   penalty        = integrator->ip_factor_strain() * average_factor *
                       integrator->compute_penalty(eval_test, direction, 0, 1);
  /*** diagonal term of grad(u)^T : v ^ n ***/
  const Number symgrad_factor = (component == direction) ? 1. : 0.5;

  auto value_on_interface01 = make_vectorized_array<Number>(0.);
  auto value_on_interface10 = make_vectorized_array<Number>(0.);
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz & v is test shape function
  {
    const auto & v0      = eval_test.shape_value_face(dof_v, 1, direction, 0);
    const auto & grad_v0 = eval_test.shape_gradient_face(dof_v, 1, direction, 0);
    const auto & v1      = eval_test.shape_value_face(dof_v, 0, direction, 1);
    const auto & grad_v1 = eval_test.shape_gradient_face(dof_v, 0, direction, 1);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u0      = eval_ansatz.shape_value_face(dof_u, 1, direction, 0);
      const auto & grad_u0 = eval_ansatz.shape_gradient_face(dof_u, 1, direction, 0);
      const auto & u1      = eval_ansatz.shape_value_face(dof_u, 0, direction, 1);
      const auto & grad_u1 = eval_ansatz.shape_gradient_face(dof_u, 0, direction, 1);

      value_on_interface01 =
        -average_factor * symgrad_factor *
        (v0 * normal0 * grad_u1 + grad_v0 * u1 * normal1); // consistency + symmetry
      value_on_interface10 =
        -average_factor * symgrad_factor *
        (v1 * normal1 * grad_u0 + grad_v1 * u0 * normal0); // consistency + symmetry

      if(equation_data.ip_variant == EquationData::PenaltyVariant::tensor)
      {
        value_on_interface01 += symgrad_factor * penalty * v0 * u1 * normal0 * normal1; // penalty
        value_on_interface10 += symgrad_factor * penalty * v1 * u0 * normal1 * normal0; // penalty
      }
      value_on_interface01 += penalty * v0 * u1 * normal0 * normal1; // penalty
      value_on_interface10 += penalty * v1 * u0 * normal1 * normal0; // penalty

      cell_matrix01(dof_v, dof_u) += 2. * mu * value_on_interface01;
      cell_matrix10(dof_v, dof_u) += 2. * mu * value_on_interface10;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrainMixed<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no,
  const int                           face_no) const
{
  const auto & equation_data  = integrator->equation_data;
  const auto   mu             = equation_data.mu;
  const auto   normal_vector  = eval_ansatz.get_normal_vector(face_no, direction);
  const auto   average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
  const Number symgrad_factor = 0.5;

  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u = eval_ansatz.shape_value_face(dof_u, face_no, direction, cell_no);
      const auto & value_on_face =
        -symgrad_factor * average_factor *
        (v * normal_vector[component_u] * u + v * u * normal_vector[component_v]);
      cell_matrix(dof_v, dof_u) += 2. * mu * value_on_face;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrainMixed<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix01,
  Table<2, VectorizedArray<Number>> & cell_matrix10,
  const int                           cell_no_left,
  const int                           direction) const
{
  (void)cell_no_left;
  AssertDimension(cell_no_left, 0);
  const auto & equation_data  = integrator->equation_data;
  const auto   mu             = equation_data.mu;
  const auto   normal_vector0 = eval_test.get_normal_vector(1, direction); // on cell 0
  const auto   normal_vector1 = eval_test.get_normal_vector(0, direction); // on cell 1
  const auto   average_factor = eval_test.get_average_factor(direction, 0, 1);
  const Number symgrad_factor = 0.5;

  auto value_on_interface01{make_vectorized_array<Number>(0.)};
  auto value_on_interface10{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v0 = eval_test.shape_value_face(dof_v, /*face_no*/ 1, direction, /*cell_no*/ 0);
    const auto & v1 = eval_test.shape_value_face(dof_v, 0, direction, 1);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u0 = eval_ansatz.shape_value_face(dof_u, 1, direction, 0);
      const auto & u1 = eval_ansatz.shape_value_face(dof_u, 0, direction, 1);

      /// consistency + symmetry
      value_on_interface01 =
        -average_factor * symgrad_factor *
        (v0 * normal_vector0[component_u] * u1 + v0 * u1 * normal_vector1[component_v]);
      value_on_interface10 =
        -average_factor * symgrad_factor *
        (v1 * normal_vector1[component_u] * u0 + v1 * u0 * normal_vector0[component_v]);
      cell_matrix01(dof_v, dof_u) += 2. * mu * value_on_interface01;
      cell_matrix10(dof_v, dof_u) += 2. * mu * value_on_interface10;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDiv<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no,
  const int                           face_no) const
{
  const auto & equation_data  = integrator->equation_data;
  const auto   lambda         = equation_data.lambda;
  const auto   normal         = eval_test.get_normal(face_no);
  const auto   average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
  const auto   penalty        = integrator->ip_factor_graddiv() * average_factor *
                       integrator->compute_penalty(eval_test, direction, cell_no, cell_no);

  /*** non-zero normal if component coincides with direction (Cartesian!)***/
  if(component == direction)
  {
    auto value_on_face{make_vectorized_array<Number>(0.)};
    for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
    {
      const auto & v      = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
      const auto & grad_v = eval_test.shape_gradient_face(dof_v, face_no, direction, cell_no);
      for(int dof_u = 0; dof_u < fe_order; ++dof_u)
      {
        const auto & u      = eval_ansatz.shape_value_face(dof_u, face_no, direction, cell_no);
        const auto & grad_u = eval_ansatz.shape_gradient_face(dof_u, face_no, direction, cell_no);

        value_on_face = -average_factor * (v * normal * grad_u + grad_v * u * normal);
        value_on_face += penalty * v * u * normal * normal;
        cell_matrix(dof_v, dof_u) += lambda * value_on_face;
      }
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDiv<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix01,
  Table<2, VectorizedArray<Number>> & cell_matrix10,
  const int                           cell_no_left,
  const int                           direction) const
{
  (void)cell_no_left;
  AssertDimension(cell_no_left, 0);
  const auto & equation_data  = integrator->equation_data;
  const auto   lambda         = equation_data.lambda;
  const auto   average_factor = eval_test.get_average_factor(direction, 0, 1);
  const auto   normal0        = eval_test.get_normal(1); // on cell 0
  const auto   normal1        = eval_test.get_normal(0); // on cell 1
  const auto   penalty        = integrator->ip_factor_graddiv() * average_factor *
                       integrator->compute_penalty(eval_test, direction, 0, 1);

  /*** non-zero normal if component coincides with direction (Cartesian!)***/
  if(component == direction)
  {
    auto value_on_interface01{make_vectorized_array<Number>(0.)};
    auto value_on_interface10{make_vectorized_array<Number>(0.)};
    for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz & v is test shape function
    {
      const auto & v0      = eval_test.shape_value_face(dof_v, 1, direction, 0);
      const auto & grad_v0 = eval_test.shape_gradient_face(dof_v, 1, direction, 0);
      const auto & v1      = eval_test.shape_value_face(dof_v, 0, direction, 1);
      const auto & grad_v1 = eval_test.shape_gradient_face(dof_v, 0, direction, 1);
      for(int dof_u = 0; dof_u < fe_order; ++dof_u)
      {
        const auto & u0      = eval_ansatz.shape_value_face(dof_u, 1, direction, 0);
        const auto & grad_u0 = eval_ansatz.shape_gradient_face(dof_u, 1, direction, 0);
        const auto & u1      = eval_ansatz.shape_value_face(dof_u, 0, direction, 1);
        const auto & grad_u1 = eval_ansatz.shape_gradient_face(dof_u, 0, direction, 1);

        /// consistency + symmetry
        value_on_interface01 =
          -1. * average_factor * (v0 * normal0 * grad_u1 + grad_v0 * u1 * normal1);
        value_on_interface10 =
          -1. * average_factor * (v1 * normal1 * grad_u0 + grad_v1 * u0 * normal0);

        value_on_interface01 += penalty * v0 * u1 * normal0 * normal1;
        value_on_interface10 += penalty * v1 * u0 * normal1 * normal0;

        cell_matrix01(dof_v, dof_u) += lambda * value_on_interface01;
        cell_matrix10(dof_v, dof_u) += lambda * value_on_interface10;
      }
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDivMixed<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix,
  const int                           direction,
  const int                           cell_no,
  const int                           face_no) const
{
  const auto & equation_data  = integrator->equation_data;
  const auto   lambda         = equation_data.lambda;
  const auto   normal_vector  = eval_ansatz.get_normal_vector(face_no, direction);
  const auto   average_factor = eval_test.get_average_factor(direction, cell_no, face_no);

  auto value_on_face{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u = eval_ansatz.shape_value_face(dof_u, face_no, direction, cell_no);

      value_on_face = -1. * average_factor *
                      (v * normal_vector[component_v] * u +
                       v * u * normal_vector[component_u]); // consistency + symmetry
      cell_matrix(dof_v, dof_u) += lambda * value_on_face;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDivMixed<Evaluator>::operator()(
  const Evaluator &                   eval_ansatz,
  const Evaluator &                   eval_test,
  Table<2, VectorizedArray<Number>> & cell_matrix01,
  Table<2, VectorizedArray<Number>> & cell_matrix10,
  const int                           cell_no_left,
  const int                           direction) const
{
  (void)cell_no_left;
  AssertDimension(cell_no_left, 0);
  const auto & equation_data  = integrator->equation_data;
  const auto   lambda         = equation_data.lambda;
  const auto   normal_vector0 = eval_ansatz.get_normal_vector(1, direction); // on cell 0
  const auto   normal_vector1 = eval_ansatz.get_normal_vector(0, direction); // on cell 1
  const auto   average_factor = eval_test.get_average_factor(direction, 0, 1);

  for(int dof_v = 0; dof_v < fe_order; ++dof_v)
  {
    const auto & v0 = eval_test.shape_value_face(dof_v, /*face_no*/ 1, direction, /*cell_no*/ 0);
    const auto & v1 = eval_test.shape_value_face(dof_v, /*face_no*/ 0, direction, /*cell_no*/ 1);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u0 = eval_ansatz.shape_value_face(dof_u, 1, direction, 0);
      const auto & u1 = eval_ansatz.shape_value_face(dof_u, 0, direction, 1);

      /// consistency + symmetry
      const auto & value_on_interface01 = -average_factor * (v0 * normal_vector0[component_v] * u1 +
                                                             v0 * u1 * normal_vector1[component_u]);
      const auto & value_on_interface10 = -average_factor * (v1 * normal_vector1[component_v] * u0 +
                                                             v1 * u0 * normal_vector0[component_u]);
      cell_matrix01(dof_v, dof_u) += lambda * value_on_interface01;
      cell_matrix10(dof_v, dof_u) += lambda * value_on_interface10;
    }
  }
}

} // end namespace FD


namespace MF
{
// template<int dim, typename number>
// void
// adjust_ghost_range_if_necessary(const MatrixFree<dim, number> &                    data,
//                                 const LinearAlgebra::distributed::Vector<number> & vec)
// {
//   if(vec.get_partitioner().get() == data.get_dof_info(0).vector_partitioner.get())
//     return;

//   LinearAlgebra::distributed::Vector<number> copy_vec(vec);
//   const_cast<LinearAlgebra::distributed::Vector<number> &>(vec).reinit(
//     data.get_dof_info(0).vector_partitioner);
//   const_cast<LinearAlgebra::distributed::Vector<number> &>(vec).copy_locally_owned_data_from(
//     copy_vec);
// }

template<int dim, int fe_degree, typename Number>
class Operator : public Subscriptor
{
public:
  using value_type  = Number;
  using vector_type = LinearAlgebra::distributed::BlockVector<Number>;
  using gradient_type =
    typename FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>::gradient_type;
  static constexpr unsigned int n_components = dim;
  static constexpr unsigned int macro_size   = VectorizedArray<Number>::size();

private:
  std::shared_ptr<const MatrixFree<dim, Number>> data;
  VectorizedArray<Number>                        mu;
  VectorizedArray<Number>                        lambda;
  EquationData::PenaltyVariant                   ip_variant;
  Number                                         ip_factor;
  mutable std::vector<TimeInfo>                  time_infos;
  bool                                           is_valid = false;

public:
  Operator() = default;

  Operator(const std::shared_ptr<const MatrixFree<dim, Number>> data_in,
           const EquationData &                                 equation_data_in)
  {
    initialize(data_in, equation_data_in);
  }

  void
  initialize(const std::shared_ptr<const MatrixFree<dim, Number>> data_in,
             const EquationData &                                 equation_data_in);

  void
  clear();

  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  void
  initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> & vec) const;

  std::shared_ptr<const MatrixFree<dim, Number>>
  get_matrix_free() const;

  std::vector<TimeInfo>
  get_time_data() const;

  void
  vmult(LinearAlgebra::distributed::BlockVector<Number> &       dst,
        const LinearAlgebra::distributed::BlockVector<Number> & src) const;

  void
  Tvmult(LinearAlgebra::distributed::BlockVector<Number> &       dst,
         const LinearAlgebra::distributed::BlockVector<Number> & src) const;

  void
  vmult(const ArrayView<Number> dst, const ArrayView<const Number> src) const
  {
    AssertThrow(1 == Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
                ExcMessage("Not implemented."));
    LinearAlgebra::distributed::BlockVector<Number> src_vector;
    initialize_dof_vector(src_vector);
    AssertDimension(src_vector.size(), src.size());
    std::copy(src.cbegin(), src.cend(), src_vector.begin());
    LinearAlgebra::distributed::BlockVector<Number> dst_vector;
    initialize_dof_vector(dst_vector);
    AssertDimension(dst_vector.size(), dst.size());
    vmult(dst_vector, src_vector);
    std::copy(dst_vector.begin(), dst_vector.end(), dst.begin());
  }

  operator const FullMatrix<Number> &() const
  {
    if(!fullmatrix)
    {
      const auto & tmp = Tensors::matrix_to_table(*this);
      fullmatrix       = std::make_shared<FullMatrix<Number>>(table_to_fullmatrix(tmp));
    }
    return *fullmatrix;
  }

  Number
  get_penalty_factor() const
  {
    return ip_factor * std::max((Number)1., (Number)fe_degree) * (fe_degree + 1);
  }

  // private:
  template<typename FEEvalType>
  Tensor<1, dim, VectorizedArray<Number>>
  get_value(const std::vector<std::shared_ptr<FEEvalType>> & phi, const unsigned q) const
  {
    AssertDimension(phi.size(), dim);
    Tensor<1, dim, VectorizedArray<Number>> value;
    for(unsigned i = 0; i < dim; ++i)
      value[i] = phi[i]->get_value(q);
    return value;
  }

  template<typename FEEvalType>
  void
  submit_value(std::vector<std::shared_ptr<FEEvalType>> &      phi,
               const Tensor<1, dim, VectorizedArray<Number>> & value_in,
               const unsigned                                  q) const
  {
    AssertDimension(phi.size(), dim);
    for(unsigned i = 0; i < dim; ++i)
      phi[i]->submit_value(value_in[i], q);
  }

  template<typename FEEvalType>
  Tensor<2, dim, VectorizedArray<Number>>
  get_symmetric_gradient(const std::vector<std::shared_ptr<FEEvalType>> & phi,
                         const unsigned                                   q) const
  {
    AssertDimension(phi.size(), dim);
    Tensor<2, dim, VectorizedArray<Number>> e;
    for(unsigned i = 0; i < dim; ++i)
      for(unsigned j = 0; j < dim; ++j)
        e[i][j] = 0.5 * (phi[i]->get_gradient(q)[j] + phi[j]->get_gradient(q)[i]);
    return e;
  }

  template<typename FEEvalType>
  void
  submit_symmetric_gradient(std::vector<std::shared_ptr<FEEvalType>> & phi,
                            Tensor<2, dim, VectorizedArray<Number>>    tensor_in,
                            const unsigned                             q) const
  {
    AssertDimension(phi.size(), dim);
    for(unsigned i = 0; i < dim; ++i)
    {
      Tensor<1, dim, VectorizedArray<Number>> symmetric_tensor_in;
      for(unsigned j = 0; j < dim; ++j)
        symmetric_tensor_in[j] = 0.5 * (tensor_in[i][j] + tensor_in[j][i]);
      phi[i]->submit_gradient(symmetric_tensor_in, q);
    }
  }

  template<typename FEEvalType>
  VectorizedArray<Number>
  get_divergence(const std::vector<std::shared_ptr<FEEvalType>> & phi, const unsigned q) const
  {
    AssertDimension(phi.size(), dim);
    VectorizedArray<Number> div = make_vectorized_array<Number>(0.);
    for(unsigned i = 0; i < dim; ++i)
      div += (phi[i]->get_gradient(q))[i];
    return div;
  }

  template<typename FEEvalType>
  void
  submit_divergence(std::vector<std::shared_ptr<FEEvalType>> & phi,
                    const VectorizedArray<Number> &            value_in,
                    const unsigned                             q) const
  {
    AssertDimension(phi.size(), dim);
    const auto & gradtype_extension = [&](const VectorizedArray<Number> & scalar,
                                          const unsigned                  comp) {
      Tensor<1, dim, VectorizedArray<Number>> scalar_extended;
      for(unsigned d = 0; d < dim; ++d)
        scalar_extended[d] = (d == comp) ? scalar : make_vectorized_array<Number>(0.);
      return scalar_extended;
    };
    for(unsigned i = 0; i < dim; ++i)
    {
      phi[i]->submit_gradient(gradtype_extension(value_in, i), q);
    }
  }

private:
  Tensor<1, dim, VectorizedArray<Number>>
  get_value_mod(const Tensor<1, dim, VectorizedArray<Number>> & value,
                const Tensor<1, dim, VectorizedArray<Number>> & normal) const
  {
    auto value_mod = value;
    for(unsigned comp = 0; comp < dim; ++comp)
    {
      Tensor<1, dim, Number> face_identifier;
      face_identifier[comp]              = 1.;
      const auto              inner_prod = abs(face_identifier * normal);
      std::bitset<macro_size> flag;
      for(auto lane = 0U; lane < macro_size; ++lane)
        flag[lane] = inner_prod[lane] < 1.e-6;
      for(auto lane = 0U; lane < macro_size; ++lane)
        if(flag[lane])
          value_mod[comp][lane] = 0.5 * value[comp][lane];
    }
    return value_mod;
  }

  void
  apply_cell(const MatrixFree<dim, Number> &                         data,
             LinearAlgebra::distributed::BlockVector<Number> &       dst,
             const LinearAlgebra::distributed::BlockVector<Number> & src,
             const std::pair<unsigned int, unsigned int> &           cell_range) const;

  void
  apply_face(const MatrixFree<dim, Number> &                         data,
             LinearAlgebra::distributed::BlockVector<Number> &       dst,
             const LinearAlgebra::distributed::BlockVector<Number> & src,
             const std::pair<unsigned int, unsigned int> &           face_range) const;

  void
  apply_boundary(const MatrixFree<dim, Number> &                         data,
                 LinearAlgebra::distributed::BlockVector<Number> &       dst,
                 const LinearAlgebra::distributed::BlockVector<Number> & src,
                 const std::pair<unsigned int, unsigned int> &           face_range) const;

  mutable std::shared_ptr<FullMatrix<Number>> fullmatrix;
};



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::clear()
{
  fullmatrix.reset();
  data.reset();
  time_infos.clear();
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize(
  const std::shared_ptr<const MatrixFree<dim, Number>> data_in,
  const EquationData &                                 equation_data_in)
{
  this->data = data_in;
  mu         = make_vectorized_array<Number>(equation_data_in.mu);
  lambda     = make_vectorized_array<Number>(equation_data_in.lambda);
  ip_variant = equation_data_in.ip_variant;
  ip_factor  = equation_data_in.ip_factor;
  time_infos = {TimeInfo{0., "[MF::Operator] vmult:", "[s]", 0}};
  is_valid   = static_cast<bool>(data);
}



template<int dim, int fe_degree, typename Number>
std::shared_ptr<const MatrixFree<dim, Number>>
Operator<dim, fe_degree, Number>::get_matrix_free() const
{
  return data;
}



template<int dim, int fe_degree, typename Number>
std::vector<TimeInfo>
Operator<dim, fe_degree, Number>::get_time_data() const
{
  return time_infos;
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize_dof_vector(
  LinearAlgebra::distributed::BlockVector<Number> & vec) const
{
  AssertThrow(data, ExcMessage("Matrix-free storage is uninitialized."));
  vec.reinit(n_components);
  for(unsigned int comp = 0; comp < n_components; ++comp)
    data->initialize_dof_vector(vec.block(comp), comp);
  vec.collect_sizes();
}



template<int dim, int fe_degree, typename Number>
types::global_dof_index
Operator<dim, fe_degree, Number>::m() const
{
  unsigned int n_dofs = 0;
  for(unsigned int comp = 0; comp < dim; ++comp)
  {
    const auto level = data->get_mg_level();
    if(level == numbers::invalid_dof_index)
      n_dofs += data->get_dof_handler(comp).n_dofs();
    else
      n_dofs += data->get_dof_handler(comp).n_dofs(level);
  }
  return n_dofs;
}
template<int dim, int fe_degree, typename Number>
types::global_dof_index
Operator<dim, fe_degree, Number>::n() const
{
  return m();
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::vmult(
  LinearAlgebra::distributed::BlockVector<Number> &       dst,
  const LinearAlgebra::distributed::BlockVector<Number> & src) const
{
  // adjust_ghost_range_if_necessary(*data, dst);
  // adjust_ghost_range_if_necessary(*data, src);
  AssertThrow(is_valid, ExcNotInitialized());
  Timer timer;
  timer.restart();

  data->loop(&Operator::apply_cell,
             &Operator::apply_face,
             &Operator::apply_boundary,
             this,
             dst,
             src,
             /*zero_dst =*/true);
  // MatrixFree<dim,Number>::DataAccessOnFaces::unspecified,
  // MatrixFree<dim,Number>::DataAccessOnFaces::unspecified);

  time_infos[0].add_time(timer.wall_time());
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::Tvmult(
  LinearAlgebra::distributed::BlockVector<Number> &       dst,
  const LinearAlgebra::distributed::BlockVector<Number> & src) const
{
  vmult(dst, src);
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_cell(
  const MatrixFree<dim, Number> &                         data,
  LinearAlgebra::distributed::BlockVector<Number> &       dst,
  const LinearAlgebra::distributed::BlockVector<Number> & src,
  const std::pair<unsigned int, unsigned int> &           cell_range) const
{
  std::vector<std::shared_ptr<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> u;
  std::vector<std::shared_ptr<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> v;
  for(unsigned int comp = 0; comp < dim; ++comp)
  {
    u.push_back(
      std::make_shared<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(data, comp));
    v.push_back(
      std::make_shared<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(data, comp));
  }

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // *** evaluate ansatz functions u in all quadrature points
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      u[comp]->reinit(cell);
      u[comp]->gather_evaluate(src.block(comp), false, true);
      v[comp]->reinit(cell);
    }

    const unsigned int n_qpoints = u[0]->n_q_points;
    Assert(std::all_of(u.cbegin(),
                       u.cend(),
                       [n_qpoints](const auto & phi) { return phi->n_q_points == n_qpoints; }),
           ExcMessage("Quadrature is not isotropic."));

    // *** linear strain: e(u) : e(v)
    for(unsigned int q = 0; q < n_qpoints; ++q)
    {
      submit_symmetric_gradient(v, 2. * mu * get_symmetric_gradient(u, q), q);
    }
    for(unsigned comp = 0; comp < dim; ++comp)
      v[comp]->integrate_scatter(false, true, dst.block(comp));

    // *** grad div: div(u) * div(v)
    for(unsigned int q = 0; q < n_qpoints; ++q)
    {
      submit_divergence(v, lambda * get_divergence(u, q), q);
    }
    for(unsigned comp = 0; comp < dim; ++comp)
      v[comp]->integrate_scatter(false, true, dst.block(comp));
  }
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_face(
  const MatrixFree<dim, Number> &                         data,
  LinearAlgebra::distributed::BlockVector<Number> &       dst,
  const LinearAlgebra::distributed::BlockVector<Number> & src,
  const std::pair<unsigned int, unsigned int> &           face_range) const
{
  std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> u_inner;
  std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> u_outer;
  std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> v_inner;
  std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> v_outer;
  for(unsigned int comp = 0; comp < dim; ++comp)
  {
    u_inner.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(
      data, true, comp));
    u_outer.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(
      data, false, comp));
    v_inner.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(
      data, true, comp));
    v_outer.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(
      data, false, comp));
  }

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // *** evaluate ansatz functions u in all quadrature points
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      u_inner[comp]->reinit(face);
      u_inner[comp]->gather_evaluate(src.block(comp), true, true);
      u_outer[comp]->reinit(face);
      u_outer[comp]->gather_evaluate(src.block(comp), true, true);
      v_inner[comp]->reinit(face);
      v_outer[comp]->reinit(face);
    }

    // *** compute penalty
    const VectorizedArray<Number> inverse_length_normal_to_face =
      std::abs((u_inner[0]->get_normal_vector(0) * u_inner[0]->inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<Number> sigma = inverse_length_normal_to_face * get_penalty_factor();

    const unsigned int n_qpoints = u_inner[0]->n_q_points; // TODO assert isotropic quadrature
    Assert(std::all_of(u_inner.cbegin(),
                       u_inner.cend(),
                       [n_qpoints](const auto & phi) { return phi->n_q_points == n_qpoints; }),
           ExcMessage("Quadrature is not isotropic."));

    /*
     * Nitsche terms of the linear strain tested by v:
     *    2mu * { sigma * (u ^ normal_u) : (v * normal_v) - 0.5 * e(u) : (v ^ normal_v) }
     * where ^ is the outer/dyadic product
     */
    const auto test_by_value_strain =
      [this, sigma](const auto &                                    u,
                    const Tensor<1, dim, VectorizedArray<Number>> & normal_u,
                    const Tensor<1, dim, VectorizedArray<Number>> & normal_v,
                    const unsigned                                  q) {
        const auto & e_u     = get_symmetric_gradient(u, q);
        const auto & value_u = get_value(u, q);

        if(EquationData::PenaltyVariant::basic == ip_variant)
          return 2. * mu *
                 (sigma * contract<1, 0>(outer_product(value_u, normal_u), normal_v) -
                  contract<1, 0>(0.5 * e_u, normal_v));
        else if(EquationData::PenaltyVariant::tensor == ip_variant)
        {
          const auto value_u_mod = get_value_mod(value_u, normal_u);
          return 2. * mu *
                 (lambda * sigma * contract<1, 0>(outer_product(value_u_mod, normal_u), normal_v) -
                  contract<1, 0>(0.5 * e_u, normal_v));
        }
        return Tensor<1, dim, VectorizedArray<Number>>{};
      };

    /*
     * Nitsche terms of the linear strain tested by v:
     *    - 2mu * (u ^ normal_u) : (0.5 * e(v))
     * where ^ is the outer/dyadic product
     */
    const auto test_by_symmetric_gradient =
      [this](const auto &                                    u,
             const Tensor<1, dim, VectorizedArray<Number>> & normal_u,
             const unsigned                                  q) {
        const auto & value_u = get_value(u, q);
        return -0.5 * 2. * mu * outer_product(value_u, normal_u);
      };

    /*
     * Nitsche terms of the grad-div-operator tested by v:
     *    lambda * { sigma * u . normal_u - 0.5 * div(u) } * normal_v
     * where ^ is the outer/dyadic product
     */
    const auto test_by_value_graddiv =
      [this, sigma](const auto &                                    u,
                    const Tensor<1, dim, VectorizedArray<Number>> & normal_u,
                    const Tensor<1, dim, VectorizedArray<Number>> & normal_v,
                    const unsigned                                  q) {
        const auto & value_u = get_value(u, q);
        return lambda * (sigma * value_u * normal_u - 0.5 * get_divergence(u, q)) * normal_v;
      };

    /*
     * Nitsche terms of the grad-div-operator tested by v:
     *    - lambda * u . normal_u * 0.5 *div(u)
     * where ^ is the outer/dyadic product
     */
    const auto test_by_divergence = [this](const auto &                                    u,
                                           const Tensor<1, dim, VectorizedArray<Number>> & normal_u,
                                           const unsigned                                  q) {
      const auto & value_u = get_value(u, q);
      return -0.5 * lambda * value_u * normal_u;
    };

    // *** integrate against test functions v (LINEAR STRAIN)
    for(unsigned int q = 0; q < n_qpoints; ++q)
    {
      const auto & normal_inner = u_inner[0]->get_normal_vector(q);
      const auto & normal_outer = -normal_inner;

      submit_value(v_inner,
                   test_by_value_strain(u_inner, /*u*/ normal_inner, /*v*/ normal_inner, q) +
                     test_by_value_strain(u_outer, /*u*/ normal_outer, /*v*/ normal_inner, q),
                   q);
      submit_symmetric_gradient(v_inner,
                                test_by_symmetric_gradient(u_inner, /*u*/ normal_inner, q) +
                                  test_by_symmetric_gradient(u_outer, /*u*/ normal_outer, q),
                                q);
      submit_value(v_outer,
                   test_by_value_strain(u_outer, /*u*/ normal_outer, /*v*/ normal_outer, q) +
                     test_by_value_strain(u_inner, /*u*/ normal_inner, /*v*/ normal_outer, q),
                   q);
      submit_symmetric_gradient(v_outer,
                                test_by_symmetric_gradient(u_outer, normal_outer, q) +
                                  test_by_symmetric_gradient(u_inner, normal_inner, q),
                                q);
    }
    for(unsigned comp = 0; comp < dim; ++comp)
    {
      v_inner[comp]->integrate_scatter(true, true, dst.block(comp));
      v_outer[comp]->integrate_scatter(true, true, dst.block(comp));
    }

    // *** integrate against test functions v (GRAD DIV)
    for(unsigned int q = 0; q < n_qpoints; ++q)
    {
      const auto & normal_inner = u_inner[0]->get_normal_vector(q);
      const auto & normal_outer = -normal_inner;

      submit_value(v_inner,
                   test_by_value_graddiv(u_inner, normal_inner, normal_inner, q) +
                     test_by_value_graddiv(u_outer, normal_outer, normal_inner, q),
                   q);
      submit_divergence(v_inner,
                        test_by_divergence(u_inner, /*u*/ normal_inner, q) +
                          test_by_divergence(u_outer, /*u*/ normal_outer, q),
                        q);
      submit_value(v_outer,
                   test_by_value_graddiv(u_outer, normal_outer, normal_outer, q) +
                     test_by_value_graddiv(u_inner, normal_inner, normal_outer, q),
                   q);
      submit_divergence(v_outer,
                        test_by_divergence(u_outer, normal_outer, q) +
                          test_by_divergence(u_inner, normal_inner, q),
                        q);
    }
    for(unsigned comp = 0; comp < dim; ++comp)
    {
      v_inner[comp]->integrate_scatter(true, true, dst.block(comp));
      v_outer[comp]->integrate_scatter(true, true, dst.block(comp));
    }
  }
}

template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::apply_boundary(
  const MatrixFree<dim, Number> &                         data,
  LinearAlgebra::distributed::BlockVector<Number> &       dst,
  const LinearAlgebra::distributed::BlockVector<Number> & src,
  const std::pair<unsigned int, unsigned int> &           face_range) const
{
  std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> u;
  std::vector<std::shared_ptr<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>> v;
  for(unsigned int comp = 0; comp < dim; ++comp)
  {
    u.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(data,
                                                                                             true,
                                                                                             comp));
    v.push_back(std::make_shared<FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>>(data,
                                                                                             true,
                                                                                             comp));
  }

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // *** evaluate ansatz functions u in all quadrature points
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      u[comp]->reinit(face);
      u[comp]->gather_evaluate(src.block(comp), true, true);
      v[comp]->reinit(face);
    }

    // *** compute penalty
    const VectorizedArray<Number> inverse_length_normal_to_face =
      std::abs((u[0]->get_normal_vector(0) * u[0]->inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<Number> sigma = 2. * inverse_length_normal_to_face * get_penalty_factor();

    const unsigned int n_qpoints = u[0]->n_q_points; // TODO assert isotropic quadrature
    Assert(std::all_of(u.cbegin(),
                       u.cend(),
                       [n_qpoints](const auto & phi) { return phi->n_q_points == n_qpoints; }),
           ExcMessage("Quadrature is not isotropic."));

    // *** integrate against test functions and derivates v (LINEAR STRAIN)
    for(unsigned int q = 0; q < n_qpoints; ++q)
    {
      const auto e_u     = get_symmetric_gradient(u, q);
      const auto normal  = u[0]->get_normal_vector(q);
      const auto value_u = get_value(u, q);

      if(EquationData::PenaltyVariant::basic == ip_variant)
        submit_value(v, 2. * mu * (sigma * value_u - contract<0, 0>(normal, e_u)), q);
      else if(EquationData::PenaltyVariant::tensor == ip_variant)
      {
        const auto value_u_mod = get_value_mod(value_u, normal);
        submit_value(v, 2. * mu * (lambda * sigma * value_u_mod - contract<0, 0>(normal, e_u)), q);
      }
      submit_symmetric_gradient(v, -2. * mu * outer_product(value_u, normal), q);
    }
    for(unsigned comp = 0; comp < dim; ++comp)
      v[comp]->integrate_scatter(true, true, dst.block(comp));

    // *** integrate against test functions and derivates v (GRAD DIV)
    for(unsigned int q = 0; q < n_qpoints; ++q)
    {
      const auto normal  = u[0]->get_normal_vector(q);
      const auto value_u = get_value(u, q);

      submit_value(v, lambda * (sigma * value_u * normal - get_divergence(u, q)) * normal, q);
      submit_divergence(v, -1. * lambda * value_u * normal, q);
    }
    for(unsigned comp = 0; comp < dim; ++comp)
      v[comp]->integrate_scatter(true, true, dst.block(comp));
  }
}

} // end namespace MF

template<int dim, int fe_degree, typename Number>
struct CombinedOperator : public MF::Operator<dim, fe_degree, Number>,
                          public FD::MatrixIntegrator<dim, fe_degree, Number>
{
  using MFOperator = typename MF::Operator<dim, fe_degree, Number>;
  using FDOperator = typename FD::MatrixIntegrator<dim, fe_degree, Number>;
  using value_type = Number;
  using FDOperator::get_equation_data;
  using FDOperator::transfer_type;
  using MFOperator::vector_type;

  CombinedOperator() = default;

  CombinedOperator(const std::shared_ptr<const MatrixFree<dim, Number>> data_in,
                   const EquationData &                                 equation_data_in)
  {
    initialize(data_in, equation_data_in);
  }

  void
  initialize(const std::shared_ptr<const MatrixFree<dim, Number>> data_in,
             const EquationData &                                 equation_data_in)
  {
    MFOperator::initialize(data_in, equation_data_in);
    FDOperator::initialize(equation_data_in);
  };

  void
  clear()
  {
    MFOperator::clear();
  };

  using MFOperator::Tvmult;
};

} // namespace LinElasticity


#endif /* TESTS_LINELASTICITYINTEGRATOR_H_ */
