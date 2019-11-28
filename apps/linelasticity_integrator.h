/*
 * linelasticity_integrator.h
 *
 *  Created on: Jun 25, 2019
 *      Author: witte
 */

#ifndef TESTS_LINELASTICITYINTEGRATOR_H_
#define TESTS_LINELASTICITYINTEGRATOR_H_

#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
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
    : mu(equation_data_in.mu), lambda(equation_data_in.lambda)
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
  double mu;     //= 1.;
  double lambda; //= 1.;
};

template<int dim>
inline void
MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &                  dinfo,
                            typename dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
  dealii::LocalIntegrators::Elasticity::cell_matrix(dinfo.matrix(0, false).matrix,
                                                    info.fe_values(0),
                                                    2. * mu);
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
  dealii::LocalIntegrators::Elasticity::nitsche_matrix(
    dinfo.matrix(0, false).matrix,
    info.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
    2. * mu);
  dealii::LocalIntegrators::GradDiv::nitsche_matrix(
    dinfo.matrix(0, false).matrix,
    info.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
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
  dealii::LocalIntegrators::Elasticity::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    2. * mu);
  dealii::LocalIntegrators::GradDiv::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    lambda);
}

} // end namespace MW



namespace FD
{
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransferBlock<dim, fe_degree, fe_degree + 1, Number>;
  using EvaluatorType = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;
  using VectorizedMatrixType = Table<2, VectorizedArray<Number>>;

private:
  using This                            = MatrixIntegrator<dim, fe_degree, Number>;
  static constexpr int fe_order         = fe_degree + 1;
  static constexpr int macro_size       = VectorizedArray<Number>::n_array_elements;
  static constexpr int n_blocks_offdiag = dim * (dim - 1);

public:
  using LaplaceIntegrator = typename Laplace::FD::MatrixIntegrator<dim, fe_degree, Number>;
  using CellMass          = typename LaplaceIntegrator::template CellMass<EvaluatorType>;
  // using CellLaplace       = typename LaplaceIntegrator::template CellLaplace<EvaluatorType>;
  // using FaceLaplace       = typename LaplaceIntegrator::template FaceLaplace<EvaluatorType>;

private:
  EquationData equation_data;
  bool         is_valid = false;

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
    is_valid      = true;
  }

  template<typename Evaluator>
  struct CellVoid
  {
    void
    operator()(const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const
    {
      // *** does nothing
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const
    {
      // *** does nothing
    }
  };


  template<typename Evaluator>
  struct CellStrain
  {
    CellStrain(const EquationData & equation_data_in, const unsigned int component_in)
      : component(component_in), mu(equation_data_in.mu)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator & /*fd_eval_ansatz*/,
               const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const unsigned int component;
    const double       mu;
  };

  template<typename Evaluator>
  struct CellGradMixed
  {
    CellGradMixed(const EquationData & equation_data_in, const int dir_u, const int dir_v)
      : Ddirection_u(dir_u), Ddirection_v(dir_v), mu(equation_data_in.mu)
    {
      AssertIndexRange(Ddirection_u, dim);
      AssertIndexRange(Ddirection_v, dim);
      Assert(Ddirection_u != Ddirection_v, ExcInvalidState());
    }

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const int    Ddirection_u;
    const int    Ddirection_v;
    const double mu;
  };

  template<typename Evaluator>
  struct CellGradDiv
  {
    CellGradDiv(const EquationData & equation_data_in, const unsigned int component_in)
      : component(component_in), lambda(equation_data_in.lambda)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator & /*fd_eval_ansatz*/,
               const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no) const;

    const unsigned int component;
    const double       lambda;
  };

  template<typename Evaluator>
  struct NitscheStrain
  {
    NitscheStrain(const EquationData & equation_data_in, const int component_in)
      : mu(equation_data_in.mu), component(component_in)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator & /*fd_eval_ansatz*/,
               const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no,
               const std::bitset<macro_size>       bdry_mask) const;

    void
    operator()(const Evaluator & /*fd_eval_ansatz*/,
               const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           direction) const;

    VectorizedArray<Number>
    compute_penalty(const Evaluator &             fd_eval,
                    const int                     direction,
                    const int                     cell_no,
                    const int                     cell_no_neighbor,
                    const std::bitset<macro_size> at_boundary_mask) const
    {
      using FaceLaplace = typename LaplaceIntegrator::template FaceLaplace<Evaluator>;
      return FaceLaplace::compute_penalty(
        fd_eval, direction, cell_no, cell_no_neighbor, at_boundary_mask);
    }

    const double mu;
    const int    component;
  };

  template<typename Evaluator>
  struct NitscheStrainMixed
  {
    NitscheStrainMixed(const EquationData & equation_data_in, const int comp_u, const int comp_v)
      : component_u(comp_u), component_v(comp_v), mu(equation_data_in.mu)
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
               const int                           face_no,
               const std::bitset<macro_size>       bdry_mask) const;

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           direction) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    VectorizedArray<Number>
    compute_penalty(const Evaluator &             fd_eval,
                    const int                     direction,
                    const int                     cell_no,
                    const int                     cell_no_neighbor,
                    const std::bitset<macro_size> at_boundary_mask) const
    {
      using FaceLaplace = typename LaplaceIntegrator::template FaceLaplace<Evaluator>;
      return FaceLaplace::compute_penalty(
        fd_eval, direction, cell_no, cell_no_neighbor, at_boundary_mask);
    }

    const int    component_u;
    const int    component_v;
    const double mu;
  };

  template<typename Evaluator>
  struct NitscheGradDiv
  {
    NitscheGradDiv(const EquationData & equation_data_in, const int component_in)
      : lambda(equation_data_in.lambda), component(component_in)
    {
      AssertIndexRange(component, dim);
    }

    void
    operator()(const Evaluator & /*fd_eval_ansatz*/,
               const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix,
               const int                           direction,
               const int                           cell_no,
               const int                           face_no,
               const std::bitset<macro_size>       bdry_mask) const;

    void
    operator()(const Evaluator & /*fd_eval_ansatz*/,
               const Evaluator &                   fd_eval,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           direction) const;

    VectorizedArray<Number>
    compute_penalty(const Evaluator &             fd_eval,
                    const int                     direction,
                    const int                     cell_no,
                    const int                     cell_no_neighbor,
                    const std::bitset<macro_size> at_boundary_mask) const
    {
      using FaceLaplace = typename LaplaceIntegrator::template FaceLaplace<Evaluator>;
      return FaceLaplace::compute_penalty(
        fd_eval, direction, cell_no, cell_no_neighbor, at_boundary_mask);
    }

    const double lambda;
    const int    component;
  };

  template<typename Evaluator>
  struct NitscheGradDivMixed
  {
    NitscheGradDivMixed(const EquationData & equation_data_in, const int comp_u, const int comp_v)
      : lambda(equation_data_in.lambda), component_u(comp_u), component_v(comp_v)
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
               const int                           face_no,
               const std::bitset<macro_size>       bdry_mask) const;

    void
    operator()(const Evaluator &                   eval_ansatz,
               const Evaluator &                   eval_test,
               Table<2, VectorizedArray<Number>> & cell_matrix01,
               Table<2, VectorizedArray<Number>> & cell_matrix10,
               const int                           direction) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    VectorizedArray<Number>
    compute_penalty(const Evaluator &             fd_eval,
                    const int                     direction,
                    const int                     cell_no,
                    const int                     cell_no_neighbor,
                    const std::bitset<macro_size> at_boundary_mask) const
    {
      using FaceLaplace = typename LaplaceIntegrator::template FaceLaplace<Evaluator>;
      return FaceLaplace::compute_penalty(
        fd_eval, direction, cell_no, cell_no_neighbor, at_boundary_mask);
    }

    const double lambda;
    const int    component_u;
    const int    component_v;
  };

  static std::array<std::array<VectorizedMatrixType, dim>, dim>
  assemble_mass_tensors(std::vector<std::shared_ptr<EvaluatorType>> & fd_evals,
                        const std::vector<CellMass> &                 cell_mass_operations,
                        const unsigned int                            patch_id)
  {
    AssertDimension(fd_evals.size(), dim);
    AssertDimension(cell_mass_operations.size(), dim);
    std::array<std::array<VectorizedMatrixType, dim>, dim> tensors;
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      auto & matrices = tensors[comp];
      auto & fd_eval  = *(fd_evals[comp]);
      fd_eval.reinit(patch_id);
      fd_eval.evaluate(true);

      matrices = fd_eval.patch_action(cell_mass_operations[comp]);
    }
    return tensors;
  }

  static std::array<std::array<VectorizedMatrixType, dim>, dim>
  assemble_strain_tensors(
    std::vector<std::shared_ptr<EvaluatorType>> &     fd_evals,
    const std::vector<CellStrain<EvaluatorType>> &    cell_strain_operations,
    const std::vector<NitscheStrain<EvaluatorType>> & nitsche_strain_operations,
    const unsigned int                                patch_id)
  {
    AssertDimension(fd_evals.size(), dim);
    AssertDimension(cell_strain_operations.size(), dim);
    AssertDimension(nitsche_strain_operations.size(), dim);
    std::array<std::array<VectorizedMatrixType, dim>, dim> tensors;
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      auto & matrices = tensors[comp];
      auto & fd_eval  = *(fd_evals[comp]);
      fd_eval.reinit(patch_id);
      fd_eval.evaluate(true);

      matrices = fd_eval.patch_action(cell_strain_operations[comp],
                                      nitsche_strain_operations[comp],
                                      nitsche_strain_operations[comp]);
    }
    return tensors;
  }

  static std::array<VectorizedMatrixType, dim>
  assemble_gradmixed_tensor(EvaluatorType &                      eval_ansatz,
                            EvaluatorType &                      eval_test,
                            const CellGradMixed<EvaluatorType> & cell_gradmixed_operation,
                            const unsigned int                   patch_id)
  {
    std::array<VectorizedMatrixType, dim> tensor;

    eval_ansatz.reinit(patch_id);
    eval_ansatz.evaluate(true);
    eval_test.reinit(patch_id);
    eval_test.evaluate(true);
    tensor = eval_test.patch_action(eval_ansatz, cell_gradmixed_operation);

    return tensor;
  }

  static VectorizedMatrixType
  assemble_strain_mixed(std::vector<std::shared_ptr<EvaluatorType>> & fd_evals,
                        const EquationData &                          equation_data,
                        const int                                     component_u,
                        const int                                     component_v,
                        const unsigned int                            patch_id)
  {
    AssertThrow(dim == 2, ExcNotImplemented());                         // TODO
    const int                               Ddirection_u = component_v; // i
    const int                               Ddirection_v = component_u; // j
    const CellGradMixed<EvaluatorType>      grad{equation_data, Ddirection_u, Ddirection_v};
    const CellVoid<EvaluatorType>           void_op;
    const NitscheStrainMixed<EvaluatorType> nitsche{equation_data, component_u, component_v};

    auto & eval_v = *(fd_evals[component_v]);
    auto & eval_u = *(fd_evals[component_u]);
    eval_u.reinit(patch_id);
    eval_u.evaluate(true);
    eval_v.reinit(patch_id);
    eval_v.evaluate(true);

    /*** compute the 1D integrals over    (d/dx_j v_i) * (d/dx_i u_j) ***/
    const auto & tensor_grad = eval_v.patch_action(eval_u, grad);
    /*** compute the  nitsche matrices (point evaluations) given component_u/v ***/
    const auto & tensor_nitsche = eval_v.patch_action(eval_u, void_op, nitsche, nitsche);

    /*** assemble cell integrals (strain) ***/
    const auto & cell_strain = Tensors::kronecker_product(tensor_grad[1], tensor_grad[0]);
    auto         block       = Tensors::scale(equation_data.mu, cell_strain);
    /*** assemble nitsche integrals (strain) ***/
    const auto & tmp1         = Tensors::kronecker_product(tensor_nitsche[1], tensor_grad[0]);
    const auto & tmp2         = Tensors::kronecker_product(tensor_grad[1], tensor_nitsche[0]);
    const auto & face_nitsche = Tensors::sum(tmp1, tmp2);
    block                     = Tensors::sum(block, face_nitsche);

    return block;
  }

  static VectorizedMatrixType
  assemble_graddiv_mixed(std::vector<std::shared_ptr<EvaluatorType>> & fd_evals,
                         const EquationData &                          equation_data,
                         const unsigned int                            component_u,
                         const unsigned int                            component_v,
                         const unsigned int                            patch_id)
  {
    AssertThrow(dim == 2, ExcNotImplemented());                          // TODO
    const int                                Ddirection_v = component_v; // i
    const int                                Ddirection_u = component_u; // j
    const CellGradMixed<EvaluatorType>       grad{equation_data, Ddirection_u, Ddirection_v};
    const CellVoid<EvaluatorType>            void_op;
    const NitscheGradDivMixed<EvaluatorType> nitsche{equation_data, component_u, component_v};

    auto & eval_v = *(fd_evals[component_v]);
    auto & eval_u = *(fd_evals[component_u]);
    eval_u.reinit(patch_id);
    eval_u.evaluate(true);
    eval_v.reinit(patch_id);
    eval_v.evaluate(true);

    /*** compute the 1D integrals over    (d/dx_i v_i) * (d/dx_j u_j) ***/
    const auto & tensor_grad = eval_v.patch_action(eval_u, grad);
    /*** compute the  nitsche matrices (point evaluations) given component_u/v ***/
    const auto & tensor_nitsche = eval_v.patch_action(eval_u, void_op, nitsche, nitsche);

    /*** assemble cell integrals (graddiv) ***/
    const auto & cell_graddiv = Tensors::kronecker_product(tensor_grad[1], tensor_grad[0]);
    auto         block        = Tensors::scale(equation_data.lambda, cell_graddiv);
    /*** assemble nitsche integrals (graddiv) ***/
    const auto & tmp1         = Tensors::kronecker_product(tensor_nitsche[1], tensor_grad[0]);
    const auto & tmp2         = Tensors::kronecker_product(tensor_grad[1], tensor_nitsche[0]);
    const auto & face_nitsche = Tensors::sum(tmp1, tmp2);
    block                     = Tensors::sum(block, face_nitsche);

    return block;
    // AssertThrow(dim == 2, ExcNotImplemented());                    // TODO
    // const int                          Ddirection_v = component_v; // i
    // const int                          Ddirection_u = component_u; // j
    // const CellGradMixed<EvaluatorType> gradmixed{equation_data, Ddirection_u, Ddirection_v};
    // auto &                             eval_v = *(fd_evals[component_v]);
    // auto &                             eval_u = *(fd_evals[component_u]);

    // /*** compute the 1D integrals over    (d/dx_i v_i) * (d/dx_j u_j) ***/
    // auto tensor = assemble_gradmixed_tensor(/*u*/ eval_u, /*v*/ eval_v, gradmixed, patch_id);
    // const auto & kronecker_product = Tensors::kronecker_product(tensor[1], tensor[0]);
    // /*** scale the kronecker product ***/
    // const auto & block = Tensors::scale(equation_data.lambda, kronecker_product);

    // return block;
  }

  static void
  assemble_graddiv_tensors(
    std::array<std::array<VectorizedMatrixType, dim>, dim> & tensors,
    std::vector<std::shared_ptr<EvaluatorType>> &            fd_evals,
    const std::vector<CellGradDiv<EvaluatorType>> &          cell_graddiv_operations,
    const std::vector<NitscheGradDiv<EvaluatorType>> &       nitsche_graddiv_operations,
    const unsigned int                                       patch_id)
  {
    AssertDimension(fd_evals.size(), dim);
    AssertDimension(cell_graddiv_operations.size(), dim);
    AssertDimension(nitsche_graddiv_operations.size(), dim);
    const auto & sum_matrices = [](auto & A, const auto & B) {
      for(unsigned int d = 0; d < dim; ++d)
        A[d] = Tensors::sum(A[d], B[d]);
    };

    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      // auto & matrices   = tensors_graddiv[comp];
      auto & fd_eval = *(fd_evals[comp]);
      fd_eval.reinit(patch_id);
      fd_eval.evaluate(true);

      const auto & matrices = fd_eval.patch_action(cell_graddiv_operations[comp],
                                                   nitsche_graddiv_operations[comp],
                                                   nitsche_graddiv_operations[comp]);
      // // DEBUG
      // tensors[comp] = matrices;
      sum_matrices(tensors[comp], matrices);
    }
  }

  template<typename TPBlockMatrix, typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & data,
                             std::vector<TPBlockMatrix> &          inverses,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertThrow(is_valid, ExcNotInitialized());
    std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
    for(unsigned int comp = 0; comp < dim; ++comp)
      fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));

    std::vector<CellMass> cell_mass_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
    {
      auto &               fd_eval = *(fd_evals[comp]);
      VectorizedMatrixType cell_mass_unit(fe_order, fe_order);
      fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
      cell_mass_operations.emplace_back(cell_mass_unit);
      // // DEBUG
      // for (unsigned int lane = 0; lane < macro_size; ++lane)
      //   {
      //     const auto& mass_lane = Tensors::vectorized_table_to_fullmatrix (cell_mass_unit,
      // lane);
      //     std::cout << "unit mass sizes: " << mass_lane.m() << ", " << mass_lane.n() <<
      //     std::endl; mass_lane.print_formatted (std::cout);
      //   }
    }

    std::vector<CellStrain<EvaluatorType>> cell_strain_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      cell_strain_operations.emplace_back(equation_data, comp);
    std::vector<NitscheStrain<EvaluatorType>> nitsche_strain_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      nitsche_strain_operations.emplace_back(equation_data, comp);

    std::vector<CellGradDiv<EvaluatorType>> cell_graddiv_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      cell_graddiv_operations.emplace_back(equation_data, comp);
    std::vector<NitscheGradDiv<EvaluatorType>> nitsche_graddiv_operations;
    for(unsigned int comp = 0; comp < dim; ++comp)
      nitsche_graddiv_operations.emplace_back(equation_data, comp);

    for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    {
      auto & inverse = inverses[id];
      inverse.resize(dim);
      const auto mass_matrices = assemble_mass_tensors(fd_evals, cell_mass_operations, id);
      auto       elasticity_matrices =
        assemble_strain_tensors(fd_evals, cell_strain_operations, nitsche_strain_operations, id);
      assemble_graddiv_tensors(
        elasticity_matrices, fd_evals, cell_graddiv_operations, nitsche_graddiv_operations, id);

      for(unsigned int comp = 0; comp < dim; ++comp)
      {
        auto & blockmatrix = inverse.get_block(comp);
        blockmatrix.reinit(mass_matrices[comp], elasticity_matrices[comp]);

        //     // // DEBUG
        //     // if (id == subdomain_range.first)
        //     //   {
        //     //     std::cout << "comp: " << comp << dim << std::endl;
        //     //     for (const auto& mass : mass_matrices)
        //     // 	for (unsigned int lane = 0; lane < macro_size; ++lane)
        //     // 	  {
        //     // 	    const auto& mass_lane = Tensors::vectorized_table_to_fullmatrix (mass,
        //     lane);
        //     // 	    std::cout << lane << "mass sizes: " << mass_lane.m() << ", " <<
        //     mass_lane.n()
        //     <<
        //     // std::endl; 	    mass_lane.print_formatted (std::cout);
        //     // 	  }
        //     //     const auto mat = Tensors::assemble_separableKD (mass_matrices,
        //     laplace_matrices);
        //     //     for (unsigned int lane = 0; lane < macro_size; ++lane)
        //     //     	{
        //     //     	  const auto& mat_lane = Tensors::vectorized_table_to_fullmatrix (mat,
        //     lane);
        //     // 	  std::cout << lane << "kdmatrix sizes: " << mat_lane.m() << ", " <<
        //     mat_lane.n()
        //     <<
        //     // std::endl;
        //     //     	  mat_lane.print_formatted (std::cout);
        //     //     	}
        //     //   }
      }
    }
  }
};

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellStrain<Evaluator>::
operator()(const Evaluator & /*fd_eval_ansatz*/,
           const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(cell_no, fd_eval.patch_variant == TPSS::PatchVariant::cell ? 1 : 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  auto         integral{make_vectorized_array<Number>(0.)};
  const Number factor{static_cast<int>(component) == direction ? 2. * mu : mu};
  for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < Evaluator::n_q_points; ++q)
      {
        const auto & grad_u = fd_eval.shape_gradient(dof_u, q, direction, cell_no);
        const auto & grad_v = fd_eval.shape_gradient(dof_v, q, direction, cell_no);
        const auto & dx     = fd_eval.get_JxW(q, direction, cell_no);
        integral += grad_u * grad_v * dx;
      }
      cell_matrix(dof_v, dof_u) += factor * integral;
    }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellGradMixed<Evaluator>::
operator()(const Evaluator &                   fd_eval_u,
           const Evaluator &                   fd_eval_v,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(cell_no, fd_eval_v.patch_variant == TPSS::PatchVariant::cell ? 1 : 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  auto       integral{make_vectorized_array<Number>(0.)};
  const bool flag_derive_u = (direction == Ddirection_u);
  const bool flag_derive_v = (direction == Ddirection_v);
  for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
    for(int dof_v = 0; dof_v < fe_order; ++dof_v)
    {
      integral = 0.;
      for(unsigned int q = 0; q < Evaluator::n_q_points; ++q)
      {
        const auto & value_or_grad_u = flag_derive_u ?
                                         fd_eval_u.shape_gradient(dof_u, q, direction, cell_no) :
                                         fd_eval_u.shape_value(dof_u, q, direction, cell_no);
        const auto & value_or_grad_v = flag_derive_v ?
                                         fd_eval_v.shape_gradient(dof_v, q, direction, cell_no) :
                                         fd_eval_v.shape_value(dof_v, q, direction, cell_no);
        const auto & dx = fd_eval_v.get_JxW(q, direction, cell_no);
        integral += value_or_grad_u * value_or_grad_v * dx;
      }
      cell_matrix(dof_v, dof_u) += integral;
    }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::CellGradDiv<Evaluator>::
operator()(const Evaluator & /*fd_eval_ansatz*/,
           const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(cell_no, fd_eval.patch_variant == TPSS::PatchVariant::cell ? 1 : 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  if(static_cast<int>(component) == direction)
  {
    auto         integral{make_vectorized_array<Number>(0.)};
    const Number factor{lambda};
    for(int dof_u = 0; dof_u < fe_order; ++dof_u) // u is ansatz function & v is test function
      for(int dof_v = 0; dof_v < fe_order; ++dof_v)
      {
        integral = 0.;
        for(unsigned int q = 0; q < Evaluator::n_q_points; ++q)
        {
          const auto & grad_u = fd_eval.shape_gradient(dof_u, q, direction, cell_no);
          const auto & grad_v = fd_eval.shape_gradient(dof_v, q, direction, cell_no);
          const auto & dx     = fd_eval.get_JxW(q, direction, cell_no);
          integral += grad_u * grad_v * dx;
        }
        cell_matrix(dof_v, dof_u) += factor * integral;
      }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrain<Evaluator>::
operator()(const Evaluator & /*fd_eval_ansatz*/,
           const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no,
           const int                           face_no,
           const std::bitset<macro_size>       bdry_mask) const
{
  using namespace dealii;

  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  const auto normal{make_vectorized_array<Number>(face_no == 0 ? -1. : 1.)};
  const auto penalty{compute_penalty(fd_eval, direction, cell_no, cell_no, bdry_mask)};

  /*** factor varies on interior and boundary cells ***/
  auto factor{make_vectorized_array<Number>(0.)};
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    factor[vv] = bdry_mask[vv] ? -1. : -0.5;
  factor = (component == direction) ? factor : 0.5 * factor; // diagonal term of grad(u)^T : v ^ n

  auto value_on_face{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v      = fd_eval.shape_value_face(dof_v, face_no, direction, cell_no);
    const auto & grad_v = fd_eval.shape_gradient_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u      = fd_eval.shape_value_face(dof_u, face_no, direction, cell_no);
      const auto & grad_u = fd_eval.shape_gradient_face(dof_u, face_no, direction, cell_no);

      value_on_face =
        factor * (v * normal * grad_u + grad_v * u * normal); // consistency + symmetry
      value_on_face += penalty * v * u;                       // penalty
      cell_matrix(dof_v, dof_u) += 2. * mu * value_on_face;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrain<Evaluator>::
operator()(const Evaluator & /*fd_eval_ansatz*/,
           const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix01,
           Table<2, VectorizedArray<Number>> & cell_matrix10,
           const int                           direction) const
{
  AssertIndexRange(direction, dim);
  AssertDimension(cell_matrix01.n_rows(), cell_matrix01.n_cols());
  AssertDimension(static_cast<int>(cell_matrix01.n_rows()), fe_order);
  AssertDimension(cell_matrix01.n_rows(), cell_matrix10.n_rows());
  AssertDimension(cell_matrix01.n_cols(), cell_matrix10.n_cols());

  /*** the outward normal on face 1 seen from cell 0 ***/
  const auto normal0{make_vectorized_array<Number>(1.)};
  /*** the outward normal on face 0 seen from cell 1 ***/
  const auto normal1{make_vectorized_array<Number>(-1.)};
  /*** boundary mask is obiviously 0(=all interior), cell_no = 0 and cell_no_neighbor = 1 ***/
  const auto penalty{compute_penalty(fd_eval, direction, 0, 1, 0)};
  /*** diagonal term of grad(u)^T : v ^ n ***/
  const auto factor = make_vectorized_array<Number>((component == direction) ? 1. : 0.5);

  auto value_on_interface01{make_vectorized_array<Number>(0.)};
  auto value_on_interface10{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz & v is test shape function
  {
    const auto & v0      = fd_eval.shape_value_face(dof_v, 1, direction, 0);
    const auto & grad_v0 = fd_eval.shape_gradient_face(dof_v, 1, direction, 0);
    const auto & v1      = fd_eval.shape_value_face(dof_v, 0, direction, 1);
    const auto & grad_v1 = fd_eval.shape_gradient_face(dof_v, 0, direction, 1);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u0      = fd_eval.shape_value_face(dof_u, 1, direction, 0);
      const auto & grad_u0 = fd_eval.shape_gradient_face(dof_u, 1, direction, 0);
      const auto & u1      = fd_eval.shape_value_face(dof_u, 0, direction, 1);
      const auto & grad_u1 = fd_eval.shape_gradient_face(dof_u, 0, direction, 1);

      value_on_interface01 =
        -0.5 * factor * (v0 * normal0 * grad_u1 + grad_v0 * u1 * normal1); // consistency + symmetry
      value_on_interface10 =
        -0.5 * factor * (v1 * normal1 * grad_u0 + grad_v1 * u0 * normal0); // consistency + symmetry

      value_on_interface01 -= penalty * v0 * u1; // penalty
      value_on_interface10 -= penalty * v1 * u0; // penalty

      cell_matrix01(dof_v, dof_u) += 2. * mu * value_on_interface01;
      cell_matrix10(dof_v, dof_u) += 2. * mu * value_on_interface10;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheStrainMixed<Evaluator>::
operator()(const Evaluator &                   eval_ansatz,
           const Evaluator &                   eval_test,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no,
           const int                           face_no,
           const std::bitset<macro_size>       bdry_mask) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  const bool   is_normal_nonzero_consistency = (component_u == direction);
  const bool   is_normal_nonzero_symmetry    = (component_v == direction);
  const Number sign_of_normal                = (face_no == 0 ? -1. : 1.);

  const auto normal_consistency =
    sign_of_normal * make_vectorized_array<Number>(is_normal_nonzero_consistency ? 1. : 0.);
  const auto normal_symmetry =
    sign_of_normal * make_vectorized_array<Number>(is_normal_nonzero_symmetry ? 1. : 0.);

  /*** factor varies on interior and boundary cells ***/
  auto chi_bdry{make_vectorized_array<Number>(0.)};
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    chi_bdry[vv] = bdry_mask[vv] ? 1. : 0.5;
  const auto & factor{-1. * mu * chi_bdry};

  auto value_on_face{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u = eval_ansatz.shape_value_face(dof_u, face_no, direction, cell_no);

      value_on_face =
        (v * u * normal_consistency + v * normal_symmetry * u); // consistency + symmetry
      cell_matrix(dof_v, dof_u) += factor * value_on_face;
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDiv<Evaluator>::
operator()(const Evaluator & /*fd_eval_ansatz*/,
           const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no,
           const int                           face_no,
           const std::bitset<macro_size>       bdry_mask) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  const auto normal{make_vectorized_array<Number>(face_no == 0 ? -1. : 1.)};
  const auto penalty{compute_penalty(fd_eval, direction, cell_no, cell_no, bdry_mask)};

  /*** factor varies on interior and boundary cells ***/
  auto factor{make_vectorized_array<Number>(0.)};
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    factor[vv] = bdry_mask[vv] ? -1. : -0.5;

  /*** non-zero normal if component coincides with direction ***/
  if(component == direction)
  {
    auto value_on_face{make_vectorized_array<Number>(0.)};
    for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
    {
      const auto & v      = fd_eval.shape_value_face(dof_v, face_no, direction, cell_no);
      const auto & grad_v = fd_eval.shape_gradient_face(dof_v, face_no, direction, cell_no);
      for(int dof_u = 0; dof_u < fe_order; ++dof_u)
      {
        const auto & u      = fd_eval.shape_value_face(dof_u, face_no, direction, cell_no);
        const auto & grad_u = fd_eval.shape_gradient_face(dof_u, face_no, direction, cell_no);

        value_on_face =
          factor * (v * normal * grad_u + grad_v * u * normal); // consistency + symmetry
        value_on_face += penalty * v * u;                       // penalty
        cell_matrix(dof_v, dof_u) += lambda * value_on_face;
      }
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDiv<Evaluator>::
operator()(const Evaluator & /*fd_eval_ansatz*/,
           const Evaluator &                   fd_eval,
           Table<2, VectorizedArray<Number>> & cell_matrix01,
           Table<2, VectorizedArray<Number>> & cell_matrix10,
           const int                           direction) const
{
  AssertIndexRange(direction, dim);
  AssertDimension(cell_matrix01.n_rows(), cell_matrix01.n_cols());
  AssertDimension(static_cast<int>(cell_matrix01.n_rows()), fe_order);
  AssertDimension(cell_matrix01.n_rows(), cell_matrix10.n_rows());
  AssertDimension(cell_matrix01.n_cols(), cell_matrix10.n_cols());

  /*** the outward normal on face 1 seen from cell 0 ***/
  const auto normal0{make_vectorized_array<Number>(1.)};
  /*** the outward normal on face 0 seen from cell 1 ***/
  const auto normal1{make_vectorized_array<Number>(-1.)};
  /*** boundary mask is obiviously 0(=all interior), cell_no = 0 and cell_no_neighbor = 1 ***/
  const auto penalty{compute_penalty(fd_eval, direction, 0, 1, 0)};

  /*** non-zero normal if component coincides with direction ***/
  if(component == direction)
  {
    auto value_on_interface01{make_vectorized_array<Number>(0.)};
    auto value_on_interface10{make_vectorized_array<Number>(0.)};
    for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz & v is test shape function
    {
      const auto & v0      = fd_eval.shape_value_face(dof_v, 1, direction, 0);
      const auto & grad_v0 = fd_eval.shape_gradient_face(dof_v, 1, direction, 0);
      const auto & v1      = fd_eval.shape_value_face(dof_v, 0, direction, 1);
      const auto & grad_v1 = fd_eval.shape_gradient_face(dof_v, 0, direction, 1);
      for(int dof_u = 0; dof_u < fe_order; ++dof_u)
      {
        const auto & u0      = fd_eval.shape_value_face(dof_u, 1, direction, 0);
        const auto & grad_u0 = fd_eval.shape_gradient_face(dof_u, 1, direction, 0);
        const auto & u1      = fd_eval.shape_value_face(dof_u, 0, direction, 1);
        const auto & grad_u1 = fd_eval.shape_gradient_face(dof_u, 0, direction, 1);

        value_on_interface01 =
          -0.5 * (v0 * normal0 * grad_u1 + grad_v0 * u1 * normal1); // consistency + symmetry
        value_on_interface10 =
          -0.5 * (v1 * normal1 * grad_u0 + grad_v1 * u0 * normal0); // consistency + symmetry

        value_on_interface01 -= penalty * v0 * u1; // penalty
        value_on_interface10 -= penalty * v1 * u0; // penalty

        cell_matrix01(dof_v, dof_u) += lambda * value_on_interface01;
        cell_matrix10(dof_v, dof_u) += lambda * value_on_interface10;
      }
    }
  }
}

template<int dim, int fe_degree, typename Number>
template<typename Evaluator>
inline void
MatrixIntegrator<dim, fe_degree, Number>::NitscheGradDivMixed<Evaluator>::
operator()(const Evaluator &                   eval_ansatz,
           const Evaluator &                   eval_test,
           Table<2, VectorizedArray<Number>> & cell_matrix,
           const int                           direction,
           const int                           cell_no,
           const int                           face_no,
           const std::bitset<macro_size>       bdry_mask) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  AssertDimension(cell_matrix.n_rows(), cell_matrix.n_cols());
  AssertDimension(static_cast<int>(cell_matrix.n_rows()), fe_order);

  const bool   is_normal_nonzero_consistency = (component_v == direction);
  const bool   is_normal_nonzero_symmetry    = (component_u == direction);
  const Number sign_of_normal                = (face_no == 0 ? -1. : 1.);

  const auto normal_consistency =
    sign_of_normal * make_vectorized_array<Number>(is_normal_nonzero_consistency ? 1. : 0.);
  const auto normal_symmetry =
    sign_of_normal * make_vectorized_array<Number>(is_normal_nonzero_symmetry ? 1. : 0.);

  const auto penalty{compute_penalty(eval_test, direction, cell_no, cell_no, bdry_mask)};
  /*** factor varies on interior and boundary cells ***/
  auto chi_bdry{make_vectorized_array<Number>(0.)};
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    chi_bdry[vv] = bdry_mask[vv] ? 1. : 0.5;
  const auto & factor{-1. * lambda * chi_bdry};

  auto value_on_face{make_vectorized_array<Number>(0.)};
  for(int dof_v = 0; dof_v < fe_order; ++dof_v) // u is ansatz function & v is test function
  {
    const auto & v = eval_test.shape_value_face(dof_v, face_no, direction, cell_no);
    for(int dof_u = 0; dof_u < fe_order; ++dof_u)
    {
      const auto & u = eval_ansatz.shape_value_face(dof_u, face_no, direction, cell_no);

      value_on_face =
        (v * normal_consistency) * u + v * (u * normal_symmetry); // consistency + symmetry
      cell_matrix(dof_v, dof_u) += factor * value_on_face;
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
class Operator : public LinearOperatorBase
{
public:
  using value_type = Number;
  using gradient_type =
    typename FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>::gradient_type;
  static constexpr unsigned int n_components = dim;

private:
  std::shared_ptr<const MatrixFree<dim, Number>> data;
  VectorizedArray<Number>                        mu;     // = make_vectorized_array(1.);
  VectorizedArray<Number>                        lambda; // = make_vectorized_array(1.);
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

  Number
  get_penalty_factor() const
  {
    return 1.0 * IP::pre_factor * std::max((Number)1., (Number)fe_degree) * (fe_degree + 1);
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
};

template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::clear()
{
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
    n_dofs += data->get_dof_handler(comp).n_dofs(); // TODO: works only on active cells
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
        return 2. * mu *
               (sigma * contract<1, 0>(outer_product(value_u, normal_u), normal_v) -
                contract<1, 0>(0.5 * e_u, normal_v));
      };

    /*
     * Nitsche terms of the linear strain tested by v:
     *    - 2mu * (u ^ normal_u) : (0.5 * e(v))
     * where ^ is the outer/dyadic product
     */
    const auto test_by_symmetric_gradient =
      [this, sigma](const auto &                                    u,
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
    const auto test_by_divergence =
      [this, sigma](const auto &                                    u,
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
    const unsigned int            n_qpoints = u[0]->n_q_points; // TODO assert isotropic quadrature
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

      submit_value(v, 2. * mu * (sigma * value_u - contract<0, 0>(normal, e_u)), q);
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
  using MFOperator    = typename MF::Operator<dim, fe_degree, Number>;
  using FDOperator    = typename FD::MatrixIntegrator<dim, fe_degree, Number>;
  using value_type    = Number;
  using transfer_type = typename FDOperator::transfer_type;

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
