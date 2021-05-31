/*
 * move_to_deal_ii.h
 *
 *  Created on: Jun 10, 2020
 *      Author: witte
 */

#ifndef MOVE_TO_DEAL_II_H_
#define MOVE_TO_DEAL_II_H_

#include <deal.II/matrix_free/tensor_product_kernels.h>

// VECTORTOOLSFIX BEGIN
#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/mapping_collection.h>

#include <functional>
#include <map>
#include <set>
#include <vector>

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/filtered_matrix.h> // ???
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/matrix_tools.h>
// #include <deal.II/numerics/vector_tools.h>

#include <boost/range/iterator_range.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <list>
#include <numeric>
#include <set>
#include <typeinfo>
#include <vector>
// VECTORTOOLSFIX END



namespace dealii
{
namespace VectorToolsFix
{
/**
 * Compute constraints that correspond to boundary conditions of the form
 * $\vec{n}^T\vec{u}=\vec{n}^T\vec{f}$, i.e. the normal components of the
 * solution $u$ and a given $f$ shall coincide. The function $f$ is given by
 * @p boundary_function and the resulting constraints are added to @p
 * constraints for faces with boundary indicator @p boundary_component.
 *
 * This function is explicitly written to use with the FE_RaviartThomas
 * elements. Thus it throws an exception, if it is called with other finite
 * elements.
 *
 * If the AffineConstraints object @p constraints contained values or other
 * constraints before, the new ones are added or the old ones overwritten,
 * if a node of the boundary part to be used was already in the list of
 * constraints. This is handled by using inhomogeneous constraints. Please
 * note that when combining adaptive meshes and this kind of constraints,
 * the Dirichlet conditions should be set first, and then completed by
 * hanging node constraints, in order to make sure that the discretization
 * remains consistent. See the discussion on conflicting constraints in the
 * module on
 * @ref constraints.
 *
 * The argument @p first_vector_component denotes the first vector component
 * in the finite element that corresponds to the vector function $\vec{u}$
 * that you want to constrain. Vectors are implicitly assumed to have
 * exactly <code>dim</code> components that are ordered in the same way as
 * we usually order the coordinate directions, i.e., $x$-, $y$-, and finally
 * $z$-component.
 *
 * The parameter @p boundary_component corresponds to the @p boundary_id of
 * the faces where the boundary conditions are applied.
 * numbers::internal_face_boundary_id is an illegal value, since it is
 * reserved for interior faces. The @p mapping is used to compute the normal
 * vector $\vec{n}$ at the boundary points.
 *
 * <h4>Computing constraints</h4>
 *
 * To compute the constraints we use interpolation operator proposed in
 * Brezzi, Fortin (Mixed and Hybrid (Finite Element Methods, Springer, 1991)
 * on every face located at the boundary.
 *
 * @ingroup constraints
 *
 * @see
 * @ref GlossBoundaryIndicator "Glossary entry on boundary indicators"
 */
template<int dim>
void
project_boundary_values_div_conforming(
  const DoFHandler<dim> &       dof_handler,
  const unsigned int            first_vector_component,
  const Function<dim, double> & boundary_function,
  const types::boundary_id      boundary_component,
  AffineConstraints<double> &   constraints,
  const Mapping<dim> &          mapping = StaticMappingQ1<dim>::mapping);



namespace internals
{
// This function computes the projection of the boundary function on the
// boundary in 2d.
template<typename cell_iterator>
void
compute_face_projection_div_conforming(const cell_iterator &   cell,
                                       const unsigned int      face_no,
                                       const FEFaceValues<2> & fe_values,
                                       const unsigned int      first_vector_component,
                                       const Function<2> &     boundary_function,
                                       const std::vector<DerivativeForm<1, 2, 2>> & jacobians,
                                       AffineConstraints<double> &                  constraints)
{
  // Compute the integral over the product of the normal components of
  // the boundary function times the normal components of the shape
  // functions supported on the boundary.
  const FEValuesExtractors::Vector vec(first_vector_component);
  const FiniteElement<2> &         fe = cell->get_fe();

  const unsigned int unit_tangential_direction[GeometryInfo<2>::faces_per_cell] = {1, 1, 0, 0};

  std::vector<Vector<double>> solution_values(fe_values.n_quadrature_points, Vector<double>(2));
  // Get the values of the boundary function at the quadrature points.
  {
    const std::vector<Point<2>> & quadrature_points = fe_values.get_quadrature_points();
    boundary_function.vector_value_list(quadrature_points, solution_values);
  }

  const std::vector<Tensor<1, 2>> & normals = fe_values.get_normal_vectors();
  Vector<double>                    dof_values(fe.dofs_per_face);
  for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
  {
    /// u * n
    double u_times_n = 0.;
    for(unsigned int d = 0; d < 2; ++d)
      u_times_n += normals[q][d] * solution_values[q](d);

    /// ||DF(x_q) * t||_2   where t is the unit tangential at unit hyperface
    const double & norm_of_DFx_times_t =
      std::sqrt(jacobians[q][0][unit_tangential_direction[face_no]] *
                  jacobians[q][0][unit_tangential_direction[face_no]] +
                jacobians[q][1][unit_tangential_direction[face_no]] *
                  jacobians[q][1][unit_tangential_direction[face_no]]);
    const auto & dx = fe_values.JxW(q);

    for(unsigned int i = 0; i < fe.dofs_per_face; ++i)
    {
      const auto ii = fe.face_to_cell_index(i,
                                            face_no,
                                            cell->face_orientation(face_no),
                                            cell->face_flip(face_no),
                                            cell->face_rotation(face_no));

      /// phi_i * n
      const double & phi_i_times_n = (normals[q] * fe_values[vec].value(ii, q));
      dof_values(i) += u_times_n * phi_i_times_n * norm_of_DFx_times_t * dx;
    }
  }

  std::vector<types::global_dof_index> face_dof_indices(fe.dofs_per_face);
  cell->face(face_no)->get_dof_indices(face_dof_indices, cell->active_fe_index());

  // Copy the computed values in the AffineConstraints only, if the degree
  // of freedom is not already constrained.
  for(unsigned int i = 0; i < fe.dofs_per_face; ++i)
  {
    const auto ii                 = fe.face_to_cell_index(i,
                                          face_no,
                                          cell->face_orientation(face_no),
                                          cell->face_flip(face_no),
                                          cell->face_rotation(face_no));
    const auto global_dof_index   = face_dof_indices[i];
    const bool is_not_constrained = !(constraints.is_constrained(global_dof_index));
    /// NOTE FE_RaviartThomas returns true for all vector components regardless
    /// of the hyperface, thus it suffices to query the first vector component
    const bool has_nonzero_component = fe.get_nonzero_components(ii)[first_vector_component];
    if(is_not_constrained && has_nonzero_component)
    {
      constraints.add_line(global_dof_index);
      if(std::abs(dof_values(i)) > 1e-14)
        constraints.set_inhomogeneity(global_dof_index, dof_values(i));
    }
  }
}

// dummy implementation of above function for all other dimensions
template<int dim, typename cell_iterator>
void
compute_face_projection_div_conforming(const cell_iterator &,
                                       const unsigned int,
                                       const FEFaceValues<dim> &,
                                       const unsigned int,
                                       const Function<dim> &,
                                       const std::vector<DerivativeForm<1, dim, dim>> &,
                                       AffineConstraints<double> &)
{
  Assert(false, ExcNotImplemented());
}

// This function computes the projection of the boundary function on the
// boundary in 3d.
template<typename cell_iterator>
void
compute_face_projection_div_conforming(const cell_iterator &   cell,
                                       const unsigned int      face,
                                       const FEFaceValues<3> & fe_values,
                                       const unsigned int      first_vector_component,
                                       const Function<3> &     boundary_function,
                                       const std::vector<DerivativeForm<1, 3, 3>> & jacobians,
                                       std::vector<double> &                        dof_values,
                                       std::vector<types::global_dof_index> &       projected_dofs)
{
  /// TODO
  AssertThrow(false, ExcMessage("Parallel impl. missing..."));

  // Compute the intergral over the product of the normal components of
  // the boundary function times the normal components of the shape
  // functions supported on the boundary.
  const FEValuesExtractors::Vector  vec(first_vector_component);
  const FiniteElement<3> &          fe      = cell->get_fe();
  const std::vector<Tensor<1, 3>> & normals = fe_values.get_normal_vectors();
  const unsigned int face_coordinate_directions[GeometryInfo<3>::faces_per_cell][2] = {
    {1, 2}, {1, 2}, {2, 0}, {2, 0}, {0, 1}, {0, 1}};
  std::vector<Vector<double>> values(fe_values.n_quadrature_points, Vector<double>(3));
  Vector<double>              dof_values_local(fe.dofs_per_face);

  {
    const std::vector<Point<3>> & quadrature_points = fe_values.get_quadrature_points();

    boundary_function.vector_value_list(quadrature_points, values);
  }

  for(unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
  {
    double tmp = 0.0;

    for(unsigned int d = 0; d < 3; ++d)
      tmp += normals[q_point][d] * values[q_point](d);

    tmp *= fe_values.JxW(q_point) *
           std::sqrt((jacobians[q_point][0][face_coordinate_directions[face][0]] *
                        jacobians[q_point][0][face_coordinate_directions[face][0]] +
                      jacobians[q_point][1][face_coordinate_directions[face][0]] *
                        jacobians[q_point][1][face_coordinate_directions[face][0]] +
                      jacobians[q_point][2][face_coordinate_directions[face][0]] *
                        jacobians[q_point][2][face_coordinate_directions[face][0]]) *
                     (jacobians[q_point][0][face_coordinate_directions[face][1]] *
                        jacobians[q_point][0][face_coordinate_directions[face][1]] +
                      jacobians[q_point][1][face_coordinate_directions[face][1]] *
                        jacobians[q_point][1][face_coordinate_directions[face][1]] +
                      jacobians[q_point][2][face_coordinate_directions[face][1]] *
                        jacobians[q_point][2][face_coordinate_directions[face][1]]));

    for(unsigned int i = 0; i < fe.dofs_per_face; ++i)
      dof_values_local(i) +=
        tmp *
        (normals[q_point] * fe_values[vec].value(fe.face_to_cell_index(i,
                                                                       face,
                                                                       cell->face_orientation(face),
                                                                       cell->face_flip(face),
                                                                       cell->face_rotation(face)),
                                                 q_point));
  }

  std::vector<types::global_dof_index> face_dof_indices(fe.dofs_per_face);

  cell->face(face)->get_dof_indices(face_dof_indices, cell->active_fe_index());

  for(unsigned int i = 0; i < fe.dofs_per_face; ++i)
    if(projected_dofs[face_dof_indices[i]] < fe.degree &&
       fe.get_nonzero_components(
         fe.face_to_cell_index(i,
                               face,
                               cell->face_orientation(face),
                               cell->face_flip(face),
                               cell->face_rotation(face)))[first_vector_component])
    {
      dof_values[face_dof_indices[i]]     = dof_values_local(i);
      projected_dofs[face_dof_indices[i]] = fe.degree;
    }
}

// dummy implementation of above
// function for all other
// dimensions
template<int dim, typename cell_iterator>
void
compute_face_projection_div_conforming(const cell_iterator &,
                                       const unsigned int,
                                       const FEFaceValues<dim> &,
                                       const unsigned int,
                                       const Function<dim> &,
                                       const std::vector<DerivativeForm<1, dim, dim>> &,
                                       std::vector<double> &,
                                       std::vector<types::global_dof_index> &)
{
  Assert(false, ExcNotImplemented());
}
} // namespace internals


template<int dim>
void
project_boundary_values_div_conforming(const DoFHandler<dim> &     dof_handler,
                                       const unsigned int          first_vector_component,
                                       const Function<dim> &       boundary_function,
                                       const types::boundary_id    boundary_component,
                                       AffineConstraints<double> & constraints,
                                       const Mapping<dim> &        mapping)
{
  constexpr auto spacedim = dim;

  /// the face integrals computed in compute_face_projection_div_conforming()
  /// are one-to-one to the Raviart-Thomas node functionals associated with
  /// hyperfaces: thus it suffices to constrain only those shape functions
  /// associated with a given hyperface

  const auto      n_q_points_1d = dof_handler.get_fe().degree + 2; // !!! fe.degree+1
  QGauss<dim - 1> face_quadrature(n_q_points_1d);

  const auto update_flags_face =
    update_JxW_values | update_normal_vectors | update_quadrature_points | update_values;
  FEFaceValues<dim> fe_face_values(mapping,
                                   dof_handler.get_fe(),
                                   face_quadrature,
                                   update_flags_face);

  hp::FECollection<dim>            fe_collection(dof_handler.get_fe());
  const hp::MappingCollection<dim> mapping_collection(mapping);
  hp::QCollection<dim>             quadrature_collection;
  /// project (dim-1)-dimensional quadrature points to dim-dimensional points
  /// for each hyperface
  for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
    quadrature_collection.push_back(QProjector<dim>::project_to_face(
      dof_handler.get_fe().reference_cell(), face_quadrature, face_no));
  const auto        update_flags_cell = update_jacobians;
  hp::FEValues<dim> fe_values(mapping_collection,
                              fe_collection,
                              quadrature_collection,
                              update_flags_cell);

  switch(dim)
  {
    case 2:
    {
      const auto locally_owned_or_ghost_cell_iterators =
        filter_iterators(dof_handler.active_cell_iterators(), [&](const auto & cell) {
          return cell->is_locally_owned() || cell->is_ghost();
        });

      for(const auto & cell : locally_owned_or_ghost_cell_iterators)
        if(cell->at_boundary())
          for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
            if(cell->face(face_no)->boundary_id() == boundary_component)
            {
              // if the FE is a
              // FE_Nothing object
              // there is no work to
              // do
              if(dynamic_cast<const FE_Nothing<dim> *>(&cell->get_fe()) != nullptr)
                return;

              // This is only
              // implemented, if the
              // FE is a Raviart-Thomas
              // element. If the FE is
              // a FESystem we cannot
              // check this.
              if(dynamic_cast<const FESystem<dim> *>(&cell->get_fe()) == nullptr)
              {
                const bool is_raviartthomas =
                  dynamic_cast<const FE_RaviartThomas<dim> *>(&cell->get_fe()) != nullptr;
                const bool is_raviartthomas_new =
                  dynamic_cast<const FE_RaviartThomas_new<dim> *>(&cell->get_fe()) != nullptr;
                const bool is_raviartthomasnodal_new =
                  dynamic_cast<const FE_RaviartThomasNodal_new<dim> *>(&cell->get_fe()) != nullptr;
                AssertThrow(is_raviartthomas || is_raviartthomas_new || is_raviartthomasnodal_new,
                            typename FiniteElement<dim>::ExcInterpolationNotImplemented());
              }

              AssertDimension(cell->active_fe_index(), 0); // hp-case has spec.
              fe_values.reinit(cell, face_no);

              const std::vector<DerivativeForm<1, dim, spacedim>> & jacobians =
                fe_values.get_present_fe_values().get_jacobians();

              fe_face_values.reinit(cell, face_no);
              internals::compute_face_projection_div_conforming(cell,
                                                                face_no,
                                                                fe_face_values,
                                                                first_vector_component,
                                                                boundary_function,
                                                                jacobians,
                                                                constraints);
            }

      break;
    }

    case 3:
    {
      /// TODO
      AssertThrow(false, ExcMessage("Parallel impl. missing..."));

      // In three dimensions the edges between two faces are treated
      // twice. Therefore we store the computed values in a vector
      // and copy them over in the AffineConstraints after all values
      // have been computed. If we have two values for one edge, we
      // choose the one, which was computed with the higher order
      // element. If both elements are of the same order, we just
      // keep the first value and do not compute a second one.
      const unsigned int &                 n_dofs = dof_handler.n_dofs();
      std::vector<double>                  dof_values(n_dofs);
      std::vector<types::global_dof_index> projected_dofs(n_dofs);

      for(unsigned int dof = 0; dof < n_dofs; ++dof)
        projected_dofs[dof] = 0;

      for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
          cell != dof_handler.end();
          ++cell)
        if(cell->at_boundary())
          for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if(cell->face(face)->boundary_id() == boundary_component)
            {
              // This is only implemented, if the FE is a
              // Raviart-Thomas element. If the FE is a FESystem we
              // cannot check this.
              if(dynamic_cast<const FESystem<dim> *>(&cell->get_fe()) == nullptr)
              {
                AssertThrow(dynamic_cast<const FE_RaviartThomas<dim> *>(&cell->get_fe()) != nullptr,
                            typename FiniteElement<dim>::ExcInterpolationNotImplemented());
              }

              fe_values.reinit(cell,
                               face + cell->active_fe_index() * GeometryInfo<dim>::faces_per_cell);

              const std::vector<DerivativeForm<1, dim, spacedim>> & jacobians =
                fe_values.get_present_fe_values().get_jacobians();

              fe_face_values.reinit(cell, face);
              internals::compute_face_projection_div_conforming(cell,
                                                                face,
                                                                fe_face_values,
                                                                first_vector_component,
                                                                boundary_function,
                                                                jacobians,
                                                                dof_values,
                                                                projected_dofs);
            }

      for(unsigned int dof = 0; dof < n_dofs; ++dof)
        if((projected_dofs[dof] != 0) && !(constraints.is_constrained(dof)))
        {
          constraints.add_line(dof);

          if(std::abs(dof_values[dof]) > 1e-14)
            constraints.set_inhomogeneity(dof, dof_values[dof]);
        }

      break;
    }

    default:
      Assert(false, ExcNotImplemented());
  }
}


// template<int dim>
// void
// project_boundary_values_div_conforming(const hp::DoFHandler<dim> & dof_handler,
//                                        const unsigned int          first_vector_component,
//                                        const Function<dim> &       boundary_function,
//                                        const types::boundary_id    boundary_component,
//                                        AffineConstraints<double> & constraints,
//                                        const hp::MappingCollection<dim, dim> &
//                                        mapping_collection)
// {
//   const unsigned int            spacedim      = dim;
//   const hp::FECollection<dim> & fe_collection = dof_handler.get_fe_collection();
//   hp::QCollection<dim - 1>      face_quadrature_collection;
//   hp::QCollection<dim>          quadrature_collection;

//   for(unsigned int i = 0; i < fe_collection.size(); ++i)
//   {
//     const QGauss<dim - 1> quadrature(fe_collection[i].degree + 1);

//     face_quadrature_collection.push_back(quadrature);

//     for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//       quadrature_collection.push_back(QProjector<dim>::project_to_face(quadrature, face));
//   }

//   hp::FEFaceValues<dim> fe_face_values(mapping_collection,
//                                        fe_collection,
//                                        face_quadrature_collection,
//                                        update_JxW_values | update_normal_vectors |
//                                          update_quadrature_points | update_values);
//   hp::FEValues<dim>     fe_values(mapping_collection,
//                               fe_collection,
//                               quadrature_collection,
//                               update_jacobians);

//   switch(dim)
//   {
//     case 2:
//     {
//       for(typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
//           cell != dof_handler.end();
//           ++cell)
//         if(cell->at_boundary())
//           for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//             if(cell->face(face)->boundary_id() == boundary_component)
//             {
//               // This is only
//               // implemented, if the
//               // FE is a Raviart-Thomas
//               // element. If the FE is
//               // a FESystem we cannot
//               // check this.
//               if(dynamic_cast<const FESystem<dim> *>(&cell->get_fe()) == nullptr)
//               {
//                 AssertThrow(dynamic_cast<const FE_RaviartThomas<dim> *>(&cell->get_fe()) !=
//                 nullptr,
//                             typename FiniteElement<dim>::ExcInterpolationNotImplemented());
//               }

//               fe_values.reinit(cell,
//                                face + cell->active_fe_index() *
//                                GeometryInfo<dim>::faces_per_cell);

//               const std::vector<DerivativeForm<1, dim, spacedim>> & jacobians =
//                 fe_values.get_present_fe_values().get_jacobians();

//               fe_face_values.reinit(cell, face);
//               internals::compute_face_projection_div_conforming(
//                 cell,
//                 face,
//                 fe_face_values.get_present_fe_values(),
//                 first_vector_component,
//                 boundary_function,
//                 jacobians,
//                 constraints);
//             }

//       break;
//     }

//     case 3:
//     {
//       const unsigned int &                 n_dofs = dof_handler.n_dofs();
//       std::vector<double>                  dof_values(n_dofs);
//       std::vector<types::global_dof_index> projected_dofs(n_dofs);

//       for(unsigned int dof = 0; dof < n_dofs; ++dof)
//         projected_dofs[dof] = 0;

//       for(typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
//           cell != dof_handler.end();
//           ++cell)
//         if(cell->at_boundary())
//           for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//             if(cell->face(face)->boundary_id() == boundary_component)
//             {
//               // This is only
//               // implemented, if the
//               // FE is a Raviart-Thomas
//               // element. If the FE is
//               // a FESystem we cannot
//               // check this.
//               if(dynamic_cast<const FESystem<dim> *>(&cell->get_fe()) == nullptr)
//               {
//                 AssertThrow(dynamic_cast<const FE_RaviartThomas<dim> *>(&cell->get_fe()) !=
//                 nullptr,
//                             typename FiniteElement<dim>::ExcInterpolationNotImplemented());
//               }

//               fe_values.reinit(cell,
//                                face + cell->active_fe_index() *
//                                GeometryInfo<dim>::faces_per_cell);

//               const std::vector<DerivativeForm<1, dim, spacedim>> & jacobians =
//                 fe_values.get_present_fe_values().get_jacobians();

//               fe_face_values.reinit(cell, face);
//               internals::compute_face_projection_div_conforming(
//                 cell,
//                 face,
//                 fe_face_values.get_present_fe_values(),
//                 first_vector_component,
//                 boundary_function,
//                 jacobians,
//                 dof_values,
//                 projected_dofs);
//             }

//       for(unsigned int dof = 0; dof < n_dofs; ++dof)
//         if((projected_dofs[dof] != 0) && !(constraints.is_constrained(dof)))
//         {
//           constraints.add_line(dof);

//           if(std::abs(dof_values[dof]) > 1e-14)
//             constraints.set_inhomogeneity(dof, dof_values[dof]);
//         }

//       break;
//     }

//     default:
//       Assert(false, ExcNotImplemented());
//   }
// }
} // namespace VectorToolsFix



namespace internal
{
namespace My
{
/**
 * This function computes the d-mode product @p out for the tensor of vectors
 * @p in with matrix @p shape_data. The direction of contraction, that is the
 * 'd' in 'd-mode product', is @p direction.
 *
 * The destination and source tensor of vectors @p out and @p in are passed as
 * flat arrays, respectively, thus reading from @p in or writing into @out
 * requires the mode sizes of previous and subsequent dimensions. Fortunately,
 * the d-mode product is reduced to a 1-mode product of order-3 tensors, where
 * the new 0th mode is obtained by collapsing all previous modes to dimension
 * @p direction and the new 2nd mode by collapsing all subsequent modes. The
 * size of collapsed modes is @p n_pre or @p n_post, respectively.
 *
 * For the d-mode product with the transpose of @p shape_data, which is
 * triggered by @p contract_over_rows equal to true, we simply contract over
 * columns instead of explicitly transposing the values in @p shape_data (note
 * that the size of mode @p direction of the source tensor has to be @p n_rows
 * instead of @p n_columns and vice versa for the destination tensor). By
 * template parameter @p add we switch between adding values into @p out or
 * assigning values (that is overwritting previous values).
 */
template<int dim,
         typename Number,
         int  n_rows,
         int  n_columns,
         int  direction,
         int  n_pre,
         int  n_post,
         bool contract_over_rows,
         bool add,
         typename Number2 = Number>
void
contract_general_impl(const Number2 * DEAL_II_RESTRICT shape_data,
                      const Number *                   in,
                      Number *                         out,
                      const int                        n_rows_dynamic    = -1,
                      const int                        n_columns_dynamic = -1,
                      const int                        n_pre_dynamic     = -1,
                      const int                        n_post_dynamic    = -1)
{
  Assert(shape_data != nullptr,
         ExcMessage("The given array shape_data must not be the null pointer!"));
  AssertIndexRange(direction, dim);

  constexpr bool is_dynamic_mode = n_rows == -1 && n_columns == -1 && n_pre == -1 && n_post == -1;

  if(is_dynamic_mode) /// TODO static if-else
  {
    Assert(n_rows_dynamic > 0, ExcMessage("Any reasonable n_rows_dynamic is at least one."));
    Assert(n_columns_dynamic > 0, ExcMessage("Any reasonable n_columns_dynamic is at least one."));
    Assert(n_pre_dynamic > 0, ExcMessage("Any reasonable n_pre_dynamic is at least one."));
    Assert(n_post_dynamic > 0, ExcMessage("Any reasonable n_post_dynamic is at least one."));
  }
  else
  {
    /// TODO static_asserts if static if-else is used
    Assert(n_rows > 0, ExcMessage("Any reasonable n_rows is at least one."));
    Assert(n_columns > 0, ExcMessage("Any reasonable n_columns is at least one."));
    Assert(n_pre > 0, ExcMessage("Any reasonable n_pre is at least one."));
    Assert(n_post > 0, ExcMessage("Any reasonable n_post is at least one."));
  }

  constexpr int mm_static = contract_over_rows ? n_rows : n_columns;
  constexpr int nn_static = contract_over_rows ? n_columns : n_rows;

  const int mm =
    is_dynamic_mode ? (contract_over_rows ? n_rows_dynamic : n_columns_dynamic) : mm_static;
  const int nn =
    is_dynamic_mode ? (contract_over_rows ? n_columns_dynamic : n_rows_dynamic) : nn_static;
  const int n_columns_actual = is_dynamic_mode ? n_columns_dynamic : n_columns;
  const int stride           = is_dynamic_mode ? n_pre_dynamic : n_pre;
  const int n_blocks1        = is_dynamic_mode ? n_pre_dynamic : n_pre;
  const int n_blocks2        = is_dynamic_mode ? n_post_dynamic : n_post;

  Assert(n_columns_actual > 0, ExcMessage("Any reasonable n_columns_actual is at least one."));
  Assert(mm <= 128,
         ExcMessage("For dynamic mm intermediate array x is initialized with static size 128."));

  for(int i2 = 0; i2 < n_blocks2; ++i2)
  {
    for(int i1 = 0; i1 < n_blocks1; ++i1)
    {
      Number x[is_dynamic_mode ? 128 : mm_static];
      for(int i = 0; i < mm; ++i)
        x[i] = in[stride * i];
      for(int col = 0; col < nn; ++col)
      {
        Number2 val0;
        if(contract_over_rows == true)
          val0 = shape_data[col];
        else
          val0 = shape_data[col * n_columns_actual];
        Number res0 = val0 * x[0];
        for(int i = 1; i < mm; ++i)
        {
          if(contract_over_rows == true)
            val0 = shape_data[i * n_columns_actual + col];
          else
            val0 = shape_data[col * n_columns_actual + i];
          res0 += val0 * x[i];
        }
        if(add == false)
          out[stride * col] = res0;
        else
          out[stride * col] += res0;
      }

      // if (one_line == false)
      {
        ++in;
        ++out;
      }
    }
    // if (one_line == false)
    {
      in += stride * (mm - 1);
      out += stride * (nn - 1);
    }
  }
}



/**
 * A purely dynamic version of contract_general_impl() above.
 */
template<int dim,
         typename Number,
         int  direction,
         bool contract_over_rows,
         bool add,
         typename Number2 = Number>
void
contract_general_impl(const Number2 * DEAL_II_RESTRICT shape_data,
                      const Number *                   in,
                      Number *                         out,
                      const int                        n_rows_dynamic,
                      const int                        n_columns_dynamic,
                      const int                        n_pre_dynamic,
                      const int                        n_post_dynamic)
{
  Assert(shape_data != nullptr,
         ExcMessage("The given array shape_data must not be the null pointer!"));
  AssertIndexRange(direction, dim);

  Assert(n_rows_dynamic > 0, ExcMessage("Any reasonable n_rows_dynamic is at least one."));
  Assert(n_columns_dynamic > 0, ExcMessage("Any reasonable n_columns_dynamic is at least one."));
  Assert(n_pre_dynamic > 0, ExcMessage("Any reasonable n_pre_dynamic is at least one."));
  Assert(n_post_dynamic > 0, ExcMessage("Any reasonable n_post_dynamic is at least one."));

  const int mm               = contract_over_rows ? n_rows_dynamic : n_columns_dynamic;
  const int nn               = contract_over_rows ? n_columns_dynamic : n_rows_dynamic;
  const int n_columns_actual = n_columns_dynamic;
  const int stride           = n_pre_dynamic;
  const int n_blocks1        = n_pre_dynamic;
  const int n_blocks2        = n_post_dynamic;

  Assert(n_columns_actual > 0, ExcMessage("Any reasonable n_columns_actual is at least one."));

  AlignedVector<Number> x(mm);
  for(int i2 = 0; i2 < n_blocks2; ++i2)
  {
    for(int i1 = 0; i1 < n_blocks1; ++i1)
    {
      for(int i = 0; i < mm; ++i)
        x[i] = in[stride * i];
      for(int col = 0; col < nn; ++col)
      {
        Number2 val0;
        if(contract_over_rows == true)
          val0 = shape_data[col];
        else
          val0 = shape_data[col * n_columns_actual];
        Number res0 = val0 * x[0];
        for(int i = 1; i < mm; ++i)
        {
          if(contract_over_rows == true)
            val0 = shape_data[i * n_columns_actual + col];
          else
            val0 = shape_data[col * n_columns_actual + i];
          res0 += val0 * x[i];
        }
        if(add == false)
          out[stride * col] = res0;
        else
          out[stride * col] += res0;
      }

      // if (one_line == false)
      {
        ++in;
        ++out;
      }
    }
    // if (one_line == false)
    {
      in += stride * (mm - 1);
      out += stride * (nn - 1);
    }
  }
}



/**
 * Tensor product evaluator with static loop bounds (assuming isotropic tensors by default).
 */
template<int dim, typename Number, int n_rows, int n_columns, typename Number2 = Number>
struct EvaluatorTensorProduct
{
  EvaluatorTensorProduct(const int /*dummy*/ = -1, const int /*dummy*/ = -1){};

  template<int  direction,
           bool contract_over_rows,
           bool add,
           int  n_pre            = -1,
           int  n_post           = -1,
           int  n_rows_actual    = n_rows,
           int  n_columns_actual = n_columns>
  void
  apply(const Number2 * DEAL_II_RESTRICT matrix,
        const Number *                   tensor_of_vectors_src,
        Number *                         tensor_of_vectors_dst,
        const int /*n_rows_dynamic*/    = -1,
        const int /*n_columns_dynamic*/ = -1,
        const int /*n_pre_dynamic*/     = -1,
        const int /*n_post_dynamic*/    = -1)
  {
    constexpr bool is_isotropic_path = n_pre == -1 && n_post == -1;
    constexpr int  reverse_direction = dim - direction - 1;

    constexpr int n_pre_actual =
      is_isotropic_path ? Utilities::pow(n_columns_actual, direction) : n_pre;
    constexpr int n_post_actual =
      is_isotropic_path ? Utilities::pow(n_rows_actual, reverse_direction) : n_post;

    contract_general_impl<dim,
                          Number,
                          n_rows_actual,
                          n_columns_actual,
                          direction,
                          n_pre_actual,
                          n_post_actual,
                          contract_over_rows,
                          add,
                          Number2>(matrix, tensor_of_vectors_src, tensor_of_vectors_dst);
  }
};

/**
 * Tensor product evaluator with dynamic loop bounds.
 */
template<int dim, typename Number, typename Number2>
struct EvaluatorTensorProduct<dim, Number, -1, -1, Number2>
{
  static_assert(1 <= dim && dim <= 3, "This struct is implemented for up to 3 dimensions.");

  EvaluatorTensorProduct(const int n_rows_in, const int n_columns_in)
    : n_rows(n_rows_in), n_columns(n_columns_in)
  {
    Assert(n_rows > 0, ExcMessage("Any reasonable n_rows is at least one."));
    Assert(n_columns > 0, ExcMessage("Any reasonable n_columns is at least one."));
  }

  template<int  direction,
           bool contract_over_rows,
           bool add,
           int /*n_pre*/            = -1,
           int /*n_post*/           = -1,
           int /*n_rows_actual*/    = -1,
           int /*n_columns_actual*/ = -1>
  void
  apply(const Number2 * DEAL_II_RESTRICT matrix,
        const Number *                   tensor_of_vectors_src,
        Number *                         tensor_of_vectors_dst,
        const int                        n_pre        = -1,
        const int                        n_post       = -1,
        const int                        n_rows_in    = -1,
        const int                        n_columns_in = -1) const
  {
    const int n_rows_actual    = n_rows_in == -1 ? n_rows : n_rows_in;
    const int n_columns_actual = n_columns_in == -1 ? n_columns : n_columns_in;

    const bool    is_isotropic_path = n_pre == -1 && n_post == -1;
    constexpr int reverse_direction = dim - direction - 1;

    const int n_pre_actual =
      is_isotropic_path ? Utilities::pow(n_columns_actual, direction) : n_pre;
    const int n_post_actual =
      is_isotropic_path ? Utilities::pow(n_rows_actual, reverse_direction) : n_post;

    // const int n_pre_actual = collapse_sizes_pre(direction, size0_src, size1_src, size2_src);
    // const int n_post_actual = collapse_sizes_post(direction, size0_dst, size1_dst, size2_dst);

    /// TODO use static bounds whenever possible !!!
    contract_general_impl<dim, Number, direction, contract_over_rows, add, Number2>(
      matrix,
      tensor_of_vectors_src,
      tensor_of_vectors_dst,
      n_rows_actual,
      n_columns_actual,
      n_pre_actual,
      n_post_actual);
  }

  const int n_rows;
  const int n_columns;
};

} // namespace My

} // namespace internal

} // namespace dealii

#endif // MOVE_TO_DEAL_II_H_
