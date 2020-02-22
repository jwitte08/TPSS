/*
 * mlaplace_integrator.h
 *
 *  Created on: Aug 11, 2018
 *      Author: witte
 */

#ifndef TESTS_MLAPLACEINTEGRATOR_H_
#define TESTS_MLAPLACEINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include "laplace_integrator.h"

using namespace dealii;

namespace MultiLaplace
{
namespace FD
{
template<int dim, int fe_degree, typename Number>
class MatrixIntegrator
{
public:
  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransferBlock<dim, fe_degree, fe_degree + 1, Number>;
  using EvaluatorType = FDEvaluation<dim, fe_degree, fe_degree + 1, Number>;

private:
  using This                      = MatrixIntegrator<dim, fe_degree, Number>;
  static constexpr int fe_order   = fe_degree + 1;
  static constexpr int macro_size = VectorizedArray<Number>::n_array_elements;

  using LaplaceIntegrator = typename Laplace::FD::MatrixIntegrator<dim, fe_degree, Number>;
  using CellMass          = typename LaplaceIntegrator::template CellMass<EvaluatorType>;
  using CellLaplace       = typename LaplaceIntegrator::template CellLaplace<EvaluatorType>;
  using FaceLaplace       = typename LaplaceIntegrator::template FaceLaplace<EvaluatorType>;

public:
  void
  initialize(const unsigned int n_components_in = 1)
  {
    n_components = n_components_in;
  }

  bool
  is_valid()
  {
    const bool n_components_is_valid = (n_components != static_cast<unsigned>(-1));
    return n_components_is_valid;
  }

  template<typename TPBlockMatrix, typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & data,
                             std::vector<TPBlockMatrix> &          inverses,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    std::vector<std::shared_ptr<EvaluatorType>> fd_evals;
    using VectorizedMatrixType = Table<2, VectorizedArray<Number>>;
    std::vector<CellMass> cell_mass_operations;
    // std::vector<VectorizedMatrixType> cell_masses_unit;
    for(unsigned int comp = 0; comp < n_components; ++comp)
    {
      fd_evals.emplace_back(std::make_shared<EvaluatorType>(data, /*dofh_index*/ comp));
      // auto & cell_mass_unit = cell_masses_unit[comp];
      auto &               fd_eval = *(fd_evals[comp]);
      VectorizedMatrixType cell_mass_unit(fe_order, fe_order);
      fd_eval.compute_unit_mass(make_array_view(cell_mass_unit));
      cell_mass_operations.emplace_back(cell_mass_unit);
      // // DEBUG
      // for (unsigned int lane = 0; lane < macro_size; ++lane)
      //   {
      //     const auto& mass_lane = table_to_fullmatrix (cell_mass_unit, lane);
      //     std::cout << "unit mass sizes: " << mass_lane.m() << ", " << mass_lane.n() <<
      //     std::endl; mass_lane.print_formatted (std::cout);
      //   }
    }

    CellLaplace cell_laplace_operation;
    FaceLaplace face_laplace_operation;
    for(unsigned int id = subdomain_range.first; id < subdomain_range.second; ++id)
    {
      auto & inverse = inverses[id];
      inverse.resize(n_components);
      for(unsigned int comp = 0; comp < n_components; ++comp)
      {
        auto & blockmatrix = inverse.get_block(comp);
        auto & fd_eval     = *(fd_evals[comp]);
        fd_eval.reinit(id);
        fd_eval.evaluate(true);

        const std::array<VectorizedMatrixType, dim> & mass_matrices =
          fd_eval.patch_action(cell_mass_operations[comp]);
        const std::array<Table<2, VectorizedArray<Number>>, dim> & laplace_matrices =
          fd_eval.patch_action(cell_laplace_operation,
                               face_laplace_operation,
                               face_laplace_operation);
        blockmatrix.reinit(mass_matrices, laplace_matrices);

        // // DEBUG
        // if (id == subdomain_range.first)
        //   {
        //     std::cout << "comp: " << comp << n_components << std::endl;
        //     for (const auto& mass : mass_matrices)
        // 	for (unsigned int lane = 0; lane < macro_size; ++lane)
        // 	  {
        // 	    const auto& mass_lane = table_to_fullmatrix (mass, lane);
        // 	    std::cout << lane << "mass sizes: " << mass_lane.m() << ", " << mass_lane.n() <<
        // std::endl; 	    mass_lane.print_formatted (std::cout);
        // 	  }
        //     const auto mat = Tensors::assemble_separableKD (mass_matrices, laplace_matrices);
        //     for (unsigned int lane = 0; lane < macro_size; ++lane)
        //     	{
        //     	  const auto& mat_lane = table_to_fullmatrix (mat, lane);
        // 	  std::cout << lane << "kdmatrix sizes: " << mat_lane.m() << ", " << mat_lane.n() <<
        // std::endl;
        //     	  mat_lane.print_formatted (std::cout);
        //     	}
        //   }
      }
    }
  }

private:
  unsigned int n_components = static_cast<unsigned>(-1);
};

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
  using value_type = Number;

  Operator() = default;

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> data,
             const unsigned int                             n_components = 1);

  void
  clear();

  bool
  is_valid()
  {
    const bool data_is_valid         = data;
    const bool n_components_is_valid = (n_components != static_cast<unsigned>(-1));
    return data_is_valid && n_components_is_valid;
  }

  types::global_dof_index
  m() const;
  types::global_dof_index
  n() const;

  // void
  // initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> & vec) const;

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

  std::shared_ptr<const MatrixFree<dim, Number>> data;
  unsigned int                                   n_components = static_cast<unsigned int>(-1);
  mutable std::vector<TimeInfo>                  time_infos;
};

template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::clear()
{
  data.reset();
  n_components = static_cast<unsigned int>(-1);
  time_infos.clear();
}



template<int dim, int fe_degree, typename Number>
void
Operator<dim, fe_degree, Number>::initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_in,
                                             const unsigned int n_components_in)
{
  this->data         = data_in;
  this->n_components = n_components_in;
  time_infos         = {TimeInfo{0., "[MF::Operator] vmult:", "[s]", 0}};
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



// template<int dim, int fe_degree, typename Number>
// void
// Operator<dim, fe_degree, Number>::initialize_dof_vector(
//   LinearAlgebra::distributed::BlockVector<Number> & vec) const
// {
//   data->initialize_dof_vector(vec);
// }



template<int dim, int fe_degree, typename Number>
types::global_dof_index
Operator<dim, fe_degree, Number>::m() const
{
  Assert(is_valid(), ExcNotInitialized());
  const unsigned int n_dofs = 0;
  for(unsigned int comp = 0; comp < n_components; ++comp)
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
  for(unsigned int comp = 0; comp < n_components; ++comp)
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data, comp);
    const auto                                             src_comp = src.block(comp);
    auto &                                                 dst_comp = dst.block(comp);
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.gather_evaluate(src_comp, false, true);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(phi.get_gradient(q), q);
      phi.integrate_scatter(false, true, dst_comp);
    }
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
  for(unsigned int comp = 0; comp < n_components; ++comp)
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(data, true, comp);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_outer(data, false, comp);
    const auto                                                 src_comp = src.block(comp);
    auto &                                                     dst_comp = dst.block(comp);
    for(unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      phi_inner.reinit(face);
      phi_inner.gather_evaluate(src_comp, true, true);
      phi_outer.reinit(face);
      phi_outer.gather_evaluate(src_comp, true, true);

      const VectorizedArray<Number> inverse_length_normal_to_face =
        0.5 * (std::abs((phi_inner.get_normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]) +
               std::abs((phi_outer.get_normal_vector(0) * phi_outer.inverse_jacobian(0))[dim - 1]));
      const VectorizedArray<Number> sigma = inverse_length_normal_to_face * get_penalty_factor();

      for(unsigned int q = 0; q < phi_inner.n_q_points; ++q)
      {
        const VectorizedArray<Number> solution_jump =
          (phi_inner.get_value(q) - phi_outer.get_value(q));
        const VectorizedArray<Number> average_normal_gradient =
          (phi_inner.get_normal_derivative(q) + phi_outer.get_normal_derivative(q)) * Number(0.5);
        const VectorizedArray<Number> test_by_value =
          solution_jump * sigma - average_normal_gradient;

        phi_inner.submit_value(test_by_value, q);
        phi_outer.submit_value(-test_by_value, q);

        phi_inner.submit_normal_derivative(-solution_jump * Number(0.5), q);
        phi_outer.submit_normal_derivative(-solution_jump * Number(0.5), q);
      }

      phi_inner.integrate_scatter(true, true, dst_comp);
      phi_outer.integrate_scatter(true, true, dst_comp);
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
  for(unsigned int comp = 0; comp < n_components; ++comp)
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(data, true, comp);
    const auto                                                 src_comp = src.block(comp);
    auto &                                                     dst_comp = dst.block(comp);
    for(unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      phi_inner.reinit(face);
      phi_inner.gather_evaluate(src_comp, true, true);

      const VectorizedArray<Number> inverse_length_normal_to_face =
        std::abs((phi_inner.get_normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]);
      const VectorizedArray<Number> sigma = inverse_length_normal_to_face * get_penalty_factor();

      const bool is_dirichlet = true; //(data.get_boundary_id(face) == 0);

      for(unsigned int q = 0; q < phi_inner.n_q_points; ++q)
      {
        const VectorizedArray<Number> u_inner                 = phi_inner.get_value(q);
        const VectorizedArray<Number> u_outer                 = is_dirichlet ? -u_inner : u_inner;
        const VectorizedArray<Number> normal_derivative_inner = phi_inner.get_normal_derivative(q);
        const VectorizedArray<Number> normal_derivative_outer =
          is_dirichlet ? normal_derivative_inner : -normal_derivative_inner;
        const VectorizedArray<Number> solution_jump = (u_inner - u_outer);
        const VectorizedArray<Number> average_normal_gradient =
          (normal_derivative_inner + normal_derivative_outer) * Number(0.5);
        const VectorizedArray<Number> test_by_value =
          solution_jump * sigma - average_normal_gradient;
        phi_inner.submit_normal_derivative(-solution_jump * Number(0.5), q);
        phi_inner.submit_value(test_by_value, q);
      }
      phi_inner.integrate_scatter(true, true, dst_comp);
    }
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

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> data,
             const unsigned int                             n_components = 1)
  {
    MFOperator::initialize(data, n_components);
    FDOperator::initialize(n_components);
  };

  void
  clear()
  {
    MFOperator::clear();
  };

  using MFOperator::Tvmult;
};

} // end namespace MultiLaplace


#endif /* TESTS_MLAPLACEINTEGRATOR_H_ */
