/*
 * common_integrator.h
 *
 *  Created on: May 26, 2019
 *      Author: witte
 */

#ifndef APPS_COMMONINTEGRATOR_H_
#define APPS_COMMONINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/assemble_flags.h>
#include <deal.II/meshworker/mesh_loop.h>

#include "solvers_and_preconditioners/TPSS/block_matrix.h"
#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"



namespace dealii
{
namespace MeshWorker
{
namespace m2d2
{
template<
  class CellIteratorType,
  class ScratchData,
  class CopyData,
  class CellIteratorBaseType = typename internal::CellIteratorBaseType<CellIteratorType>::type>
void
mesh_loop(
  const CellIteratorType &                          begin,
  const typename identity<CellIteratorType>::type & end,

  const typename identity<
    std::function<void(const CellIteratorBaseType &, ScratchData &, CopyData &)>>::type &
                                                                         cell_worker,
  const typename identity<std::function<void(const CopyData &)>>::type & copier,

  const ScratchData & sample_scratch_data,
  const CopyData &    sample_copy_data,

  const AssembleFlags flags = assemble_own_cells,

  const bool assemble_relevant_faces = false,

  const typename identity<std::function<
    void(const CellIteratorBaseType &, const unsigned int, ScratchData &, CopyData &)>>::type &
    boundary_worker = std::function<
      void(const CellIteratorBaseType &, const unsigned int, ScratchData &, CopyData &)>(),

  const typename identity<std::function<void(const CellIteratorBaseType &,
                                             const unsigned int,
                                             const unsigned int,
                                             const CellIteratorBaseType &,
                                             const unsigned int,
                                             const unsigned int,
                                             ScratchData &,
                                             CopyData &)>>::type & face_worker =
    std::function<void(const CellIteratorBaseType &,
                       const unsigned int,
                       const unsigned int,
                       const CellIteratorBaseType &,
                       const unsigned int,
                       const unsigned int,
                       ScratchData &,
                       CopyData &)>(),

  const unsigned int queue_length = 2 * MultithreadInfo::n_threads(),
  const unsigned int chunk_size   = 8)
{
  Assert(
    (!cell_worker) == !(flags & work_on_cells),
    ExcMessage(
      "If you specify a cell_worker, you need to set assemble_own_cells or assemble_ghost_cells."));

  Assert(
    (flags & (assemble_own_interior_faces_once | assemble_own_interior_faces_both)) !=
      (assemble_own_interior_faces_once | assemble_own_interior_faces_both),
    ExcMessage(
      "You can only specify assemble_own_interior_faces_once OR assemble_own_interior_faces_both."));

  Assert((flags & (assemble_ghost_faces_once | assemble_ghost_faces_both)) !=
           (assemble_ghost_faces_once | assemble_ghost_faces_both),
         ExcMessage(
           "You can only specify assemble_ghost_faces_once OR assemble_ghost_faces_both."));

  Assert(!(flags & cells_after_faces) || (flags & (assemble_own_cells | assemble_ghost_cells)),
         ExcMessage("The option cells_after_faces only makes sense if you assemble on cells."));

  Assert((!face_worker) == !(flags & work_on_faces),
         ExcMessage("If you specify a face_worker, assemble_face_* needs to be set."));

  Assert((!boundary_worker) == !(flags & assemble_boundary_faces),
         ExcMessage("If you specify a boundary_worker, assemble_boundary_faces needs to be set."));

  /// DEBUG
  // std::ofstream                       ofs;
  // ofs.open("debug.txt", std::ios_base::app);
  // const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  // const auto pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

  auto cell_action =
    [&](const CellIteratorBaseType & cell, ScratchData & scratch, CopyData & copy) {
      // First reset the CopyData class to the empty copy_data given by the
      // user.
      copy = sample_copy_data;

      // Store the dimension in which we are working for later use
      const auto dim = cell->get_triangulation().dimension;

      const bool ignore_subdomain =
        (cell->get_triangulation().locally_owned_subdomain() == numbers::invalid_subdomain_id);

      types::subdomain_id current_subdomain_id =
        (cell->is_level_cell() ? cell->level_subdomain_id() : cell->subdomain_id());

      const bool own_cell =
        ignore_subdomain ||
        (current_subdomain_id == cell->get_triangulation().locally_owned_subdomain());

      if((!ignore_subdomain) && (current_subdomain_id == numbers::artificial_subdomain_id))
        return;

      if(!(flags & (cells_after_faces)) && (((flags & (assemble_own_cells)) && own_cell) ||
                                            ((flags & assemble_ghost_cells) && !own_cell)))
        cell_worker(cell, scratch, copy);

      if(flags & (work_on_faces | work_on_boundary))
        for(const unsigned int face_no : cell->face_indices())
        {
          if(cell->at_boundary(face_no) && !cell->has_periodic_neighbor(face_no))
          {
            // only integrate boundary faces of own cells
            if((flags & assemble_boundary_faces) && (own_cell || assemble_relevant_faces))
              boundary_worker(cell, face_no, scratch, copy);
          }
          else
          {
            // interior face, potentially assemble
            TriaIterator<typename CellIteratorBaseType::AccessorType> neighbor =
              cell->neighbor_or_periodic_neighbor(face_no);

            types::subdomain_id neighbor_subdomain_id = numbers::artificial_subdomain_id;
            if(neighbor->is_level_cell())
              neighbor_subdomain_id = neighbor->level_subdomain_id();
            // subdomain id is only valid for active cells
            else if(neighbor->is_active())
              neighbor_subdomain_id = neighbor->subdomain_id();

            const bool own_neighbor =
              ignore_subdomain ||
              (neighbor_subdomain_id == cell->get_triangulation().locally_owned_subdomain());

            /// DEBUG
            // if(!own_cell && !own_neighbor)
            // {
            //   const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);
            //   const unsigned int neighbor_face_no = periodic_neighbor ?
            //                                           cell->periodic_neighbor_face_no(face_no) :
            //                                           cell->neighbor_face_no(face_no);
            //   *pcout << "A " << "l" << cell->level() << "c" << cell->index() << "f" << face_no <<
            //   "l"
            //             << neighbor->level() << "c" << neighbor->index() << "f" <<
            //             neighbor_face_no
            //             << std::endl;
            // }

            // skip all faces between two ghost cells
            if((!own_cell && !own_neighbor) && !assemble_relevant_faces)
              continue;

            // skip if the user doesn't want faces between own cells
            if(own_cell && own_neighbor &&
               !(flags & (assemble_own_interior_faces_both | assemble_own_interior_faces_once)))
              continue;

            // skip face to ghost
            if(own_cell != own_neighbor &&
               !(flags & (assemble_ghost_faces_both | assemble_ghost_faces_once)))
              continue;

            // Deal with refinement edges from the refined side. Assuming
            // one-irregular meshes, this situation should only occur if
            // both cells are active.
            const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);

            if(dim > 1 && ((!periodic_neighbor && cell->neighbor_is_coarser(face_no)) ||
                           (periodic_neighbor && cell->periodic_neighbor_is_coarser(face_no))))
            {
              Assert(cell->is_active(), ExcInternalError());
              Assert(neighbor->is_active(), ExcInternalError());

              // skip if only one processor needs to assemble the face
              // to a ghost cell and the fine cell is not ours.
              if(!own_cell && (flags & assemble_ghost_faces_once))
                continue;

              const std::pair<unsigned int, unsigned int> neighbor_face_no =
                periodic_neighbor ? cell->periodic_neighbor_of_coarser_periodic_neighbor(face_no) :
                                    cell->neighbor_of_coarser_neighbor(face_no);

              face_worker(cell,
                          face_no,
                          numbers::invalid_unsigned_int,
                          neighbor,
                          neighbor_face_no.first,
                          neighbor_face_no.second,
                          scratch,
                          copy);

              if(flags & assemble_own_interior_faces_both)
              {
                // If own faces are to be assembled from both sides,
                // call the faceworker again with swapped arguments.
                // This is because we won't be looking at an adaptively
                // refined edge coming from the other side.
                face_worker(neighbor,
                            neighbor_face_no.first,
                            neighbor_face_no.second,
                            cell,
                            face_no,
                            numbers::invalid_unsigned_int,
                            scratch,
                            copy);
              }
            }
            else if(dim == 1 && cell->level() > neighbor->level())
            {
              // In one dimension, there is no other check to do
              const unsigned int neighbor_face_no = periodic_neighbor ?
                                                      cell->periodic_neighbor_face_no(face_no) :
                                                      cell->neighbor_face_no(face_no);
              Assert(periodic_neighbor || neighbor->face(neighbor_face_no) == cell->face(face_no),
                     ExcInternalError());

              face_worker(cell,
                          face_no,
                          numbers::invalid_unsigned_int,
                          neighbor,
                          neighbor_face_no,
                          numbers::invalid_unsigned_int,
                          scratch,
                          copy);

              if(flags & assemble_own_interior_faces_both)
              {
                // If own faces are to be assembled from both sides,
                // call the faceworker again with swapped arguments.
                face_worker(neighbor,
                            neighbor_face_no,
                            numbers::invalid_unsigned_int,
                            cell,
                            face_no,
                            numbers::invalid_unsigned_int,
                            scratch,
                            copy);
              }
            }
            else
            {
              // If iterator is active and neighbor is refined, skip
              // internal face.
              if(dealii::internal::is_active_iterator(cell) && neighbor->has_children())
                continue;

              // Now neighbor is on same level, double-check this:
              Assert(cell->level() == neighbor->level(), ExcInternalError());

              /// DEBUG
              // if(!own_cell && !own_neighbor)
              // {
              //   const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);
              //   const unsigned int neighbor_face_no = periodic_neighbor ?
              //                                           cell->periodic_neighbor_face_no(face_no)
              //                                           : cell->neighbor_face_no(face_no);
              //   *pcout << "B " << "l" << cell->level() << "c" << cell->index() << "f" << face_no
              //   << "l"
              //             << neighbor->level() << "c" << neighbor->index() << "f" <<
              //             neighbor_face_no
              //             << std::endl;
              // }

              // If we own both cells only do faces from one side (unless
              // AssembleFlags says otherwise). Here, we rely on cell
              // comparison that will look at cell->index().
              if(own_cell && own_neighbor && (flags & assemble_own_interior_faces_once) &&
                 (neighbor < cell))
                continue;

              /// DEBUG
              // if(!own_cell && !own_neighbor)
              // {
              //   const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);
              //   const unsigned int neighbor_face_no = periodic_neighbor ?
              //                                           cell->periodic_neighbor_face_no(face_no)
              //                                           : cell->neighbor_face_no(face_no);
              //   *pcout << "C " << "l" << cell->level() << "c" << cell->index() << "f" << face_no
              //   << "l"
              //             << neighbor->level() << "c" << neighbor->index() << "f" <<
              //             neighbor_face_no
              //             << std::endl;
              // }

              // We only look at faces to ghost on the same level once
              // (only where own_cell=true and own_neighbor=false)
              if(!own_cell && !assemble_relevant_faces)
                continue;

              /// DEBUG
              // if(!own_cell && !own_neighbor)
              // {
              //   const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);
              //   const unsigned int neighbor_face_no = periodic_neighbor ?
              //                                           cell->periodic_neighbor_face_no(face_no)
              //                                           : cell->neighbor_face_no(face_no);
              //   *pcout << "D " << "l" << cell->level() << "c" << cell->index() << "f" << face_no
              //   << "l"
              //             << neighbor->level() << "c" << neighbor->index() << "f" <<
              //             neighbor_face_no
              //             << std::endl;
              // }

              // now only one processor assembles faces_to_ghost. We let
              // the processor with the smaller (level-)subdomain id
              // assemble the face.
              if(own_cell && !own_neighbor && (flags & assemble_ghost_faces_once) &&
                 (neighbor_subdomain_id < current_subdomain_id))
                continue;

              /// DEBUG
              // if(!own_cell && !own_neighbor)
              // {
              //   const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);
              //   const unsigned int neighbor_face_no = periodic_neighbor ?
              //                                           cell->periodic_neighbor_face_no(face_no)
              //                                           : cell->neighbor_face_no(face_no);
              //   *pcout << "E " << "l" << cell->level() << "c" << cell->index() << "f" << face_no
              //   << "l"
              //             << neighbor->level() << "c" << neighbor->index() << "f" <<
              //             neighbor_face_no
              //             << std::endl;
              // }

              const unsigned int neighbor_face_no = periodic_neighbor ?
                                                      cell->periodic_neighbor_face_no(face_no) :
                                                      cell->neighbor_face_no(face_no);
              Assert(periodic_neighbor || neighbor->face(neighbor_face_no) == cell->face(face_no),
                     ExcInternalError());

              face_worker(cell,
                          face_no,
                          numbers::invalid_unsigned_int,
                          neighbor,
                          neighbor_face_no,
                          numbers::invalid_unsigned_int,
                          scratch,
                          copy);
            }
          }
        } // faces

      // Execute the cell_worker if faces are handled before cells
      if((flags & cells_after_faces) && (((flags & assemble_own_cells) && own_cell) ||
                                         ((flags & assemble_ghost_cells) && !own_cell)))
        cell_worker(cell, scratch, copy);
    };

  // Submit to workstream
  WorkStream::run(begin,
                  end,
                  cell_action,
                  copier,
                  sample_scratch_data,
                  sample_copy_data,
                  queue_length,
                  chunk_size);
}



template<
  class CellIteratorType,
  class ScratchData,
  class CopyData,
  class CellIteratorBaseType = typename internal::CellIteratorBaseType<CellIteratorType>::type>
void
mesh_loop(
  IteratorRange<CellIteratorType> iterator_range,
  const typename identity<
    std::function<void(const CellIteratorBaseType &, ScratchData &, CopyData &)>>::type &
                                                                         cell_worker,
  const typename identity<std::function<void(const CopyData &)>>::type & copier,

  const ScratchData & sample_scratch_data,
  const CopyData &    sample_copy_data,

  const AssembleFlags flags = assemble_own_cells,

  const bool assemble_relevant_faces = false,

  const typename identity<std::function<
    void(const CellIteratorBaseType &, const unsigned int, ScratchData &, CopyData &)>>::type &
    boundary_worker = std::function<
      void(const CellIteratorBaseType &, const unsigned int, ScratchData &, CopyData &)>(),

  const typename identity<std::function<void(const CellIteratorBaseType &,
                                             const unsigned int,
                                             const unsigned int,
                                             const CellIteratorBaseType &,
                                             const unsigned int,
                                             const unsigned int,
                                             ScratchData &,
                                             CopyData &)>>::type & face_worker =
    std::function<void(const CellIteratorBaseType &,
                       const unsigned int,
                       const unsigned int,
                       const CellIteratorBaseType &,
                       const unsigned int,
                       const unsigned int,
                       ScratchData &,
                       CopyData &)>(),

  const unsigned int queue_length = 2 * MultithreadInfo::n_threads(),
  const unsigned int chunk_size   = 8)
{
  // Call the function above
  mesh_loop<typename IteratorRange<CellIteratorType>::IteratorOverIterators,
            ScratchData,
            CopyData,
            CellIteratorBaseType>(iterator_range.begin(),
                                  iterator_range.end(),
                                  cell_worker,
                                  copier,
                                  sample_scratch_data,
                                  sample_copy_data,
                                  flags,
                                  assemble_relevant_faces,
                                  boundary_worker,
                                  face_worker,
                                  queue_length,
                                  chunk_size);
}

} // namespace m2d2

} // namespace MeshWorker

} // namespace dealii



namespace Nitsche
{
/**
 * Standard penalty to obtain well-posedness of the Nitsche method. The
 * penalty is weighted for face integrals at the physical boundary (that is
 * h_left = h_right). In case of interior penalty methods, the penalty factor
 * for interfaces between a so-called "left" and "right" cell is obtained by
 * multiplying with 1/2.
 */
template<typename Number>
Number
compute_penalty_impl(const int degree, const Number h_left, const Number h_right)
{
  const auto one_over_h = (0.5 / h_left) + (0.5 / h_right);
  const auto gamma      = degree == 0 ? 1 : degree * (degree + 1);
  return 2.0 * gamma * one_over_h;
}
} // namespace Nitsche



namespace FD
{
using namespace dealii;

template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct CellOperationBase
{
  static constexpr unsigned int fe_degree_static        = fe_degree;
  static constexpr unsigned int ansatz_fe_degree_static = fe_degree;
  static constexpr unsigned int n_q_points_static       = n_q_points_1d;

  using value_type            = Number;
  using evaluator_type        = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;
  using ansatz_evaluator_type = FDEvaluation<dim, fe_degree_ansatz, n_q_points_1d, Number>;
  using matrix_type           = Table<2, VectorizedArray<Number>>;

  void
  operator()(const ansatz_evaluator_type & eval_ansatz,
             const evaluator_type &        eval_test,
             matrix_type &                 cell_matrix,
             const int                     direction,
             const int                     cell_no) const
  {
    AssertThrow(false, ExcMessage("Please, overload this operation in derived classes."));
  }
};



template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperationBase
{
  static constexpr unsigned int fe_degree_static        = fe_degree;
  static constexpr unsigned int ansatz_fe_degree_static = fe_degree;
  static constexpr unsigned int n_q_points_static       = n_q_points_1d;

  using value_type            = Number;
  using evaluator_type        = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;
  using ansatz_evaluator_type = FDEvaluation<dim, fe_degree_ansatz, n_q_points_1d, Number>;
  using matrix_type           = Table<2, VectorizedArray<Number>>;

  void
  operator()(const ansatz_evaluator_type & eval_ansatz,
             const evaluator_type &        eval_test,
             matrix_type &                 cell_matrix,
             const int                     direction,
             const int                     cell_no,
             const int                     face_no) const
  {
    AssertThrow(false, ExcMessage("Please, overload this operation in derived classes."));
  }

  void
  operator()(const ansatz_evaluator_type & eval_ansatz,
             const evaluator_type &        eval_test,
             matrix_type &                 cell_matrix01,
             matrix_type &                 cell_matrix10,
             const int                     cell_no0,
             const int                     direction)
  {
    AssertThrow(false, ExcMessage("Please, overload this operation in derived classes."));
  }
};



namespace Gradient
{
template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct CellOperation
  : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const int n_q_points    = eval_test.n_q_points_1d(direction);

    VectorizedArray<Number> integral;
    for(int i = 0; i < n_dofs_test; ++i)
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        integral = 0.;
        for(int q = 0; q < n_q_points; ++q)
        {
          const auto & grad_u_j = eval_ansatz.shape_gradient(j, q, direction, cell_no);
          const auto & v_i      = eval_test.shape_value(i, q, direction, cell_no);
          const auto & dx       = eval_test.get_JxW(q, direction, cell_no);
          integral += grad_u_j * v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};
} // namespace Gradient



namespace Laplace
{
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct CellOperation : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const int n_q_points    = eval_test.n_q_points_1d(direction);

    VectorizedArray<Number> integral;
    for(int i = 0; i < n_dofs_test; ++i)
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        integral = 0.;
        for(int q = 0; q < n_q_points; ++q)
        {
          const auto & grad_u_j = eval_ansatz.shape_gradient(i, q, direction, cell_no);
          const auto & grad_v_i = eval_test.shape_gradient(j, q, direction, cell_no);
          const auto & dx       = eval_test.get_JxW(q, direction, cell_no);
          integral += grad_u_j * grad_v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};

namespace SIPG
{
template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperation
  : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no,
             const int                                    face_no) const
  {
    const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
    const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
    const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
    const auto normal         = eval_test.get_normal(face_no);

    const auto h = eval_test.get_h(direction, cell_no);
    const auto penalty =
      this->penalty_factor * average_factor * ::Nitsche::compute_penalty_impl(fe_degree, h, h);

    auto value_on_face = make_vectorized_array<Number>(0.);
    for(int i = 0; i < n_dofs_test; ++i)
    {
      const auto & v_i      = eval_test.shape_value_face(i, face_no, direction, cell_no);
      const auto & grad_v_i = eval_test.shape_gradient_face(i, face_no, direction, cell_no);
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        const auto & u_j      = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);
        const auto & grad_u_j = eval_ansatz.shape_gradient_face(j, face_no, direction, cell_no);

        value_on_face = -average_factor * (v_i * normal * grad_u_j + grad_v_i * u_j * normal);

        value_on_face += penalty * v_i * u_j * normal * normal;

        cell_matrix(i, j) += value_on_face;
      }
    }
  }

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix01,
             typename Base::matrix_type &                 cell_matrix10,
             const int                                    cell_no0,
             const int                                    direction) const
  {
    AssertDimension(cell_no0, 0);

    const int  n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int  n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const auto normal0       = eval_test.get_normal(1); // on cell 0
    const auto normal1       = eval_test.get_normal(0); // on cell 1 !!!

    const auto h0 = eval_test.get_h(direction, cell_no0);
    const auto h1 = eval_test.get_h(direction, cell_no0 + 1);
    const auto penalty =
      this->interior_penalty_factor * 0.5 * ::Nitsche::compute_penalty_impl(fe_degree, h0, h1);

    auto value_on_interface01 = make_vectorized_array<Number>(0.);
    auto value_on_interface10 = make_vectorized_array<Number>(0.);
    for(int i = 0; i < n_dofs_test; ++i) // u is ansatz & v is test shape function
    {
      const auto & v0_i      = eval_test.shape_value_face(i, 1, direction, 0);
      const auto & grad_v0_i = eval_test.shape_gradient_face(i, 1, direction, 0);
      const auto & v1_i      = eval_test.shape_value_face(i, 0, direction, 1);
      const auto & grad_v1_i = eval_test.shape_gradient_face(i, 0, direction, 1);
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        const auto & u0_j      = eval_ansatz.shape_value_face(j, 1, direction, 0);
        const auto & grad_u0_j = eval_ansatz.shape_gradient_face(j, 1, direction, 0);
        const auto & u1_j      = eval_ansatz.shape_value_face(j, 0, direction, 1);
        const auto & grad_u1_j = eval_ansatz.shape_gradient_face(j, 0, direction, 1);

        /// consistency + symmetry
        value_on_interface01 = -0.5 * (v0_i * normal0 * grad_u1_j + grad_v0_i * u1_j * normal1);
        value_on_interface10 = -0.5 * (v1_i * normal1 * grad_u0_j + grad_v1_i * u0_j * normal0);

        /// penalty
        value_on_interface01 += penalty * v0_i * u1_j * normal0 * normal1;
        value_on_interface10 += penalty * v1_i * u0_j * normal1 * normal0;

        cell_matrix01(i, j) += value_on_interface01;
        cell_matrix10(i, j) += value_on_interface10;
      }
    }
  }

  Number penalty_factor          = 1.;
  Number interior_penalty_factor = 1.;
};

} // namespace SIPG
} // namespace Laplace



namespace L2
{
template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct CellOperation
  : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix,
             const int                                    direction,
             const int                                    cell_no) const
  {
    const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
    const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
    const int n_q_points    = eval_test.n_q_points_1d(direction);

    VectorizedArray<Number> integral;
    for(int i = 0; i < n_dofs_test; ++i)
      for(int j = 0; j < n_dofs_ansatz; ++j)
      {
        integral = 0.;
        for(int q = 0; q < n_q_points; ++q)
        {
          const auto & u_j = eval_ansatz.shape_value(j, q, direction, cell_no);
          const auto & v_i = eval_test.shape_value(i, q, direction, cell_no);
          const auto & dx  = eval_test.get_JxW(q, direction, cell_no);
          integral += u_j * v_i * dx;
        }
        cell_matrix(i, j) += integral;
      }
  }
};

// TODO !!! FaceOperation has not been tested
// template<int dim,
//          int fe_degree,
//          int n_q_points_1d,
//          typename Number,
//          int fe_degree_ansatz = fe_degree>
// struct FaceOperation
//   : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
// {
//   using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

//   void
//   operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
//              const typename Base::evaluator_type &        eval_test,
//              typename Base::matrix_type &                 cell_matrix,
//              const int                                    direction,
//              const int                                    cell_no,
//              const int                                    face_no) const
//   {
//     const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
//     const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

//     auto value_on_face = make_vectorized_array<Number>(0.);
//     for(int i = 0; i < n_dofs_test; ++i)
//     {
//       const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
//       for(int j = 0; j < n_dofs_ansatz; ++j)
//       {
//         const auto & u_j = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);

//         value_on_face = v_i * u_j;

//         cell_matrix(i, j) += value_on_face;
//       }
//     }
//   }

//   void
//   operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
//              const typename Base::evaluator_type &        eval_test,
//              typename Base::matrix_type &                 cell_matrix01,
//              typename Base::matrix_type &                 cell_matrix10,
//              const int                                    cell_no0,
//              const int                                    direction) const
//   {
//     AssertDimension(cell_no0, 0);

//     const int n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
//     const int n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);

//     auto value_on_interface01 = make_vectorized_array<Number>(0.);
//     auto value_on_interface10 = make_vectorized_array<Number>(0.);
//     for(int i = 0; i < n_dofs_test; ++i) // u is ansatz & v is test shape function
//     {
//       const auto & v0_i = eval_test.shape_value_face(i, 1, direction, 0);
//       const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
//       for(int j = 0; j < n_dofs_ansatz; ++j)
//       {
//         const auto & u0_j = eval_ansatz.shape_value_face(j, 1, direction, 0);
//         const auto & u1_j = eval_ansatz.shape_value_face(j, 0, direction, 1);

//         value_on_interface01 = v0_i * u1_j;
//         value_on_interface10 = v1_i * u0_j;

//         cell_matrix01(i, j) += value_on_interface01;
//         cell_matrix10(i, j) += value_on_interface10;
//       }
//     }
//   }
// };

} // namespace L2



namespace Void
{
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct CellOperation : public CellOperationBase<dim, fe_degree, n_q_points_1d, Number>
{
  using Base = CellOperationBase<dim, fe_degree, n_q_points_1d, Number>;

  void
  operator()(const typename Base::ansatz_evaluator_type &,
             const typename Base::evaluator_type &,
             typename Base::matrix_type &,
             const int,
             const int) const
  {
  }
};

template<int dim,
         int fe_degree,
         int n_q_points_1d,
         typename Number,
         int fe_degree_ansatz = fe_degree>
struct FaceOperation
  : public FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>
{
  using Base = FaceOperationBase<dim, fe_degree, n_q_points_1d, Number, fe_degree_ansatz>;

  void
  operator()(const typename Base::ansatz_evaluator_type &,
             const typename Base::evaluator_type &,
             typename Base::matrix_type &,
             const int,
             const int,
             const int) const
  {
  }

  void
  operator()(const typename Base::ansatz_evaluator_type & eval_ansatz,
             const typename Base::evaluator_type &        eval_test,
             typename Base::matrix_type &                 cell_matrix01,
             typename Base::matrix_type &                 cell_matrix10,
             const int                                    cell_no0,
             const int                                    direction) const
  {
  }
};

} // namespace Void

} // namespace FD



namespace MW
{
template<int dim, bool is_multigrid>
struct IteratorSelector
{
  // static_assert(false, "No specialization has been found.");
};

template<int dim>
struct IteratorSelector<dim, false>
{
  using type = typename DoFHandler<dim>::active_cell_iterator;
};

template<int dim>
struct IteratorSelector<dim, true>
{
  using type = typename DoFHandler<dim>::level_cell_iterator;
};



template<int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe,
              const unsigned int         quadrature_degree,
              const UpdateFlags          update_flags,
              const UpdateFlags          interface_update_flags = UpdateFlags::update_default)
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



/**
 * symgrad(phi)_{d,c} = 0.5 (\partial_d phi_{i;c} + \partial_c phi_{i;d})
 */
template<int dim, typename EvaluatorType>
SymmetricTensor<2, dim>
compute_symgrad_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  SymmetricTensor<2, dim> symgrad_of_phi;
  for(auto d = 0U; d < dim; ++d)
    for(auto c = d; c < dim; ++c)
      symgrad_of_phi[d][c] =
        0.5 * (phi.shape_grad_component(i, q, c)[d] + phi.shape_grad_component(i, q, d)[c]);
  return symgrad_of_phi;
}

/// TODO obsolete due to FEValuesBase variant?
template<int dim>
SymmetricTensor<2, dim>
compute_symgrad(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_symgrad_impl<dim, FEValues<dim>>(phi, i, q);
}

template<int dim>
SymmetricTensor<2, dim>
compute_symgrad(const FEValuesBase<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_symgrad_impl<dim, FEValuesBase<dim>>(phi, i, q);
}



/**
 * symgrad(phi)_{d,c} = 0.5 (\partial_d phi_{i;c} + \partial_c phi_{i;d})
 */
template<int dim>
double
compute_divergence(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  double div_of_phi = 0.;
  for(auto d = 0U; d < dim; ++d)
    div_of_phi += phi.shape_grad_component(i, q, d)[d];
  return div_of_phi;
}



/**
 * {{ symgrad(phi_i) }} = 0.5 ({{ \partial_d phi_{i;c} }} + {{ \partial_c phi_{i;d} }})
 */
template<int dim, typename EvaluatorType>
SymmetricTensor<2, dim>
compute_average_symgrad_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  SymmetricTensor<2, dim> av_symgrad_of_phi;
  for(auto d = 0U; d < dim; ++d)
    for(auto c = d; c < dim; ++c)
      av_symgrad_of_phi[d][c] =
        0.5 * (phi.average_gradient(i, q, c)[d] + phi.average_gradient(i, q, d)[c]);
  return av_symgrad_of_phi;
}

template<int dim>
SymmetricTensor<2, dim>
compute_average_symgrad(const FEInterfaceValues<dim> & phi,
                        const unsigned int             i,
                        const unsigned int             q)
{
  return compute_average_symgrad_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}

/**
 * Assuming that the normal n is constant, i.e. the interface flat-sided, the
 * average of the symmetric gradient of the tangential vector field phit = phi -
 * (phi*n) n reads
 *
 *    {{ symgrad(phit_i) }} = {{ symgrad(phi_i) }} (I - n(x)n)
 *
 * where (x) is the dyadic product of two vectors and I the identity matrix.
 */
template<int dim>
Tensor<2, dim>
compute_average_symgrad_tangential(const FEInterfaceValues<dim> & phi,
                                   const unsigned int             i,
                                   const unsigned int             q)
{
  AssertThrow(false, ExcMessage("TODO seems to be wrong !!!"));
  Tensor<2, dim> av_symgrad_of_phi;
  for(auto d = 0U; d < dim; ++d)
    for(auto c = 0U; c < dim; ++c)
      av_symgrad_of_phi[d][c] =
        0.5 * (phi.average_gradient(i, q, c)[d] + phi.average_gradient(i, q, d)[c]);

  const Tensor<1, dim> & n              = phi.normal(q);
  Tensor<2, dim>         n_cross_n      = outer_product(n, n); // n(x)n
  Tensor<2, dim>         normal_contrib = contract<1, 0>(av_symgrad_of_phi, n_cross_n);

  return av_symgrad_of_phi - normal_contrib;
}



/**
 * {{ phi }} = 0.5 * (phi^+ + phi^-)
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vaverage_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> average_phi;
  for(auto c = 0; c < dim; ++c)
    average_phi[c] = phi.average(i, q, c);
  return average_phi;
}

template<int dim>
Tensor<1, dim>
compute_vaverage(const FEInterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vaverage_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}



/**
 * [[ phi ]] = phi^+ - phi^-
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vvalue_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> value_phi;
  for(auto c = 0; c < dim; ++c)
    value_phi[c] = phi.shape_value_component(i, q, c);
  return value_phi;
}



/// TODO obsolete by FEValuesBase variant?
template<int dim>
Tensor<1, dim>
compute_vvalue(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vvalue_impl<dim, FEValues<dim>>(phi, i, q);
}



/// TODO use ..._impl
/// TODO obsolete by FEValuesBase variant?
template<int dim>
Tensor<1, dim>
compute_vvalue(const FEFaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> value_phi;
  for(auto c = 0; c < dim; ++c)
    value_phi[c] = phi.shape_value_component(i, q, c);
  return value_phi;
}



template<int dim>
Tensor<1, dim>
compute_vvalue(const FEValuesBase<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vvalue_impl<dim, FEValuesBase<dim>>(phi, i, q);
}



/**
 * The tangential vector field phit = phi - (phi*n) n is returned.
 */
template<int dim>
Tensor<1, dim>
compute_vvalue_tangential(const FEValuesBase<dim> & phi, const unsigned int i, const unsigned int q)
{
  const Tensor<1, dim> & n         = phi.normal_vector(q);
  const Tensor<1, dim> & value_phi = compute_vvalue(phi, i, q);
  return value_phi - (value_phi * n) * n;
}



/**
 * [[ phi ]] = phi^+ - phi^-
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vjump_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  Tensor<1, dim> jump_phi;
  for(auto c = 0; c < dim; ++c)
    jump_phi[c] = phi.jump(i, q, c);
  return jump_phi;
}

template<int dim>
Tensor<1, dim>
compute_vjump(const FEInterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vjump_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}



/**
 * The jump of the tangential vector field phit = phi - (phi*n) n reads
 *
 *    [[ phit ]] = [[ phi ]] - ([[ phi ]]*n) n
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vjump_tangential_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  const Tensor<1, dim> & n = phi.normal(q);
  Tensor<1, dim>         jump_phi;
  for(auto c = 0; c < dim; ++c)
    jump_phi[c] = phi.jump(i, q, c);
  return jump_phi - (jump_phi * n) * n;
}

template<int dim>
Tensor<1, dim>
compute_vjump_tangential(const FEInterfaceValues<dim> & phi,
                         const unsigned int             i,
                         const unsigned int             q)
{
  return compute_vjump_tangential_impl<dim, FEInterfaceValues<dim>>(phi, i, q);
}



/**
 * [[ phi ]] (x) n
 */
template<int dim>
Tensor<2, dim>
compute_vjump_cross_normal(const FEInterfaceValues<dim> & phi,
                           const unsigned int             i,
                           const unsigned int             q)
{
  const Tensor<1, dim> & n          = phi.normal(q);
  const Tensor<1, dim> & jump_value = compute_vjump(phi, i, q);
  return outer_product(jump_value, n);
}



/**
 *    [[ phit ]] (x) n
 *
 * where n defines the normal and phit the tangential vector field (phi -
 * (phi*n) n). The jump of [[ phit ]] is defined in compute_vjump_tangential.
 */
template<int dim>
Tensor<2, dim>
compute_vjump_cross_normal_tangential(const FEInterfaceValues<dim> & phi,
                                      const unsigned int             i,
                                      const unsigned int             q)
{
  const Tensor<1, dim> & n          = phi.normal(q);
  const Tensor<1, dim> & jump_value = compute_vjump_tangential(phi, i, q);
  return outer_product(jump_value, n);
}



/**
 * [[ phi ]] . n
 */
template<int dim>
double
compute_vjump_dot_normal(const FEInterfaceValues<dim> & phi,
                         const unsigned int             i,
                         const unsigned int             q)
{
  const Tensor<1, dim> & n          = phi.normal(q);
  const Tensor<1, dim> & jump_value = compute_vjump(phi, i, q);
  return jump_value * n;
}



/**
 * Vector curl in 2D:   curl(phi) = (d/dy phi, -d/dx phi)^T
 */
template<int dim, typename EvaluatorType>
Tensor<1, dim>
compute_vcurl_impl(const EvaluatorType & phi, const unsigned int i, const unsigned int q)
{
  AssertDimension(phi.get_fe().n_components(), 1U);
  static_assert(dim == 2, "vector curl is reasonable in 2D.");

  /// See Guido's MFEM script for the vector curl definition.
  Tensor<1, dim> vcurl_of_phi;
  const auto &   grad_of_phi = phi.shape_grad_component(i, q, 0U);
  vcurl_of_phi[0]            = grad_of_phi[1];
  vcurl_of_phi[1]            = -grad_of_phi[0];

  return vcurl_of_phi;
}

template<int dim>
Tensor<1, dim>
compute_vcurl(const FEValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vcurl_impl<dim, FEValues<dim>>(phi, i, q);
}

template<int dim>
Tensor<1, dim>
compute_vcurl(const FEFaceValuesBase<dim> & phi, const unsigned int i, const unsigned int q)
{
  return compute_vcurl_impl<dim, FEFaceValuesBase<dim>>(phi, i, q);
}



/**
 * Jump of vector curl in 2D:   [[ curl(phi) ]] = (d/dy [[ phi ]], -d/dx [[ phi ]])^T
 */
template<int dim>
Tensor<1, dim>
compute_jump_vcurl(const FEInterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  static_assert(dim == 2, "vector curl is reasonable in 2D.");

  /// See Guido's MFEM script for the vector curl definition.
  Tensor<1, dim> jump_vcurl_of_phi;
  const auto &   jump_grad_of_phi = phi.jump_gradient(i, q);
  jump_vcurl_of_phi[0]            = jump_grad_of_phi[1];
  jump_vcurl_of_phi[1]            = -jump_grad_of_phi[0];

  return jump_vcurl_of_phi;
}



namespace Mixed
{
template<int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe_test,
              const FiniteElement<dim> & fe_ansatz,
              const unsigned int         n_q_points_1d,
              const UpdateFlags          update_flags_test,
              const UpdateFlags          update_flags_ansatz,
              const UpdateFlags          interface_update_flags_test = UpdateFlags::update_default,
              const UpdateFlags interface_update_flags_ansatz        = UpdateFlags::update_default)
    : fe_values_test(mapping, fe_test, QGauss<dim>(n_q_points_1d), update_flags_test),
      fe_values_ansatz(mapping, fe_ansatz, QGauss<dim>(n_q_points_1d), update_flags_ansatz),
      fe_interface_values_test(mapping,
                               fe_test,
                               QGauss<dim - 1>(n_q_points_1d),
                               interface_update_flags_test),
      fe_interface_values_ansatz(mapping,
                                 fe_ansatz,
                                 QGauss<dim - 1>(n_q_points_1d),
                                 interface_update_flags_ansatz)
  {
  }

  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe,
              const unsigned int         n_q_points_1d,
              const UpdateFlags          update_flags,
              const UpdateFlags          interface_update_flags = UpdateFlags::update_default)
    : ScratchData(mapping,
                  fe,
                  fe,
                  n_q_points_1d,
                  update_flags,
                  update_flags,
                  interface_update_flags,
                  interface_update_flags)
  {
  }

  ScratchData(const ScratchData<dim> & scratch_data)
    : fe_values_test(scratch_data.fe_values_test.get_mapping(),
                     scratch_data.fe_values_test.get_fe(),
                     scratch_data.fe_values_test.get_quadrature(),
                     scratch_data.fe_values_test.get_update_flags()),
      fe_values_ansatz(scratch_data.fe_values_ansatz.get_mapping(),
                       scratch_data.fe_values_ansatz.get_fe(),
                       scratch_data.fe_values_ansatz.get_quadrature(),
                       scratch_data.fe_values_ansatz.get_update_flags()),
      fe_interface_values_test(scratch_data.fe_values_test.get_mapping(),
                               scratch_data.fe_values_test.get_fe(),
                               scratch_data.fe_interface_values_test.get_quadrature(),
                               scratch_data.fe_interface_values_test.get_update_flags()),
      fe_interface_values_ansatz(scratch_data.fe_values_ansatz.get_mapping(),
                                 scratch_data.fe_values_ansatz.get_fe(),
                                 scratch_data.fe_interface_values_ansatz.get_quadrature(),
                                 scratch_data.fe_interface_values_ansatz.get_update_flags())
  {
  }

  FEValues<dim>          fe_values_test;
  FEValues<dim>          fe_values_ansatz;
  FEInterfaceValues<dim> fe_interface_values_test;
  FEInterfaceValues<dim> fe_interface_values_ansatz;
};

struct CopyData
{
  struct FaceData
  {
    FullMatrix<double>                   cell_matrix;
    FullMatrix<double>                   cell_matrix_flipped;
    std::vector<types::global_dof_index> joint_dof_indices_test;
    std::vector<types::global_dof_index> joint_dof_indices_ansatz;
    Vector<double>                       cell_rhs_test;
    Vector<double>                       cell_rhs_ansatz;
  };

  CopyData(const unsigned int n_dofs_per_cell_test,
           const unsigned int n_dofs_per_cell_ansatz,
           const unsigned int level_in = numbers::invalid_unsigned_int)
    : level(level_in),
      cell_matrix(n_dofs_per_cell_test, n_dofs_per_cell_ansatz),
      cell_matrix_flipped(n_dofs_per_cell_ansatz, n_dofs_per_cell_test),
      cell_rhs_test(n_dofs_per_cell_test),
      cell_rhs_ansatz(n_dofs_per_cell_ansatz),
      local_dof_indices_test(n_dofs_per_cell_test),
      local_dof_indices_ansatz(n_dofs_per_cell_ansatz)
  {
  }

  CopyData(const unsigned int n_dofs_per_cell)
    : CopyData(n_dofs_per_cell, n_dofs_per_cell, numbers::invalid_unsigned_int)
  {
  }

  CopyData(const CopyData &) = default;

  unsigned int                         level;
  FullMatrix<double>                   cell_matrix;
  FullMatrix<double>                   cell_matrix_flipped;
  Vector<double>                       cell_rhs_test;
  Vector<double>                       cell_rhs_ansatz;
  std::vector<types::global_dof_index> local_dof_indices_test;
  std::vector<types::global_dof_index> local_dof_indices_ansatz;
  std::vector<FaceData>                face_data;
};

} // namespace Mixed



namespace StreamFunction
{
/**
 * A helper struct which provides a FEValues-like interface for stream
 * functions. A stream function is the vector curl of scalar polynomial shape
 * functions in two space dimensions.
 */
template<int dim>
struct Values
{
  static_assert(dim == 2, "Implemented for 2D only.");

  Values(const FEValues<dim> & fe_values_in)
    : fe_values(fe_values_in.get_mapping(),
                fe_values_in.get_fe(),
                fe_values_in.get_quadrature(),
                fe_values_in.get_update_flags()),
      n_quadrature_points(fe_values_in.n_quadrature_points)
  {
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType & cell)
  {
    fe_values.reinit(cell);
  }

  unsigned int
  n_dofs_per_cell() const
  {
    return get_fe().dofs_per_cell;
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return fe_values.get_fe();
  }

  const std::vector<Point<dim>> &
  get_quadrature_points() const
  {
    return fe_values.get_quadrature_points();
  }

  double
  JxW(const unsigned int q) const
  {
    return fe_values.JxW(q);
  }

  double
  shape_value_component(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    const auto & curl_phi_i = compute_vcurl(fe_values, i, q);
    return curl_phi_i[c];
  }

  Tensor<1, dim>
  shape_grad_component(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    AssertIndexRange(c, dim);
    const auto &   hess_phi_i = fe_values.shape_hessian(i, q);
    Tensor<1, dim> grad;
    if(c == 0U)
    {
      grad[0] = hess_phi_i[0][1];
      grad[1] = hess_phi_i[1][1];
    }
    else if(c == 1U)
    {
      grad[0] = -hess_phi_i[0][0];
      grad[1] = -hess_phi_i[1][0];
    }
    return grad;
  }

  FEValues<dim> fe_values;
  unsigned int  n_quadrature_points;
};



template<int dim>
SymmetricTensor<2, dim>
compute_symgrad(const Values<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_symgrad_impl<dim, Values<dim>>(phi, i, q);
}



template<int dim>
Tensor<1, dim>
compute_vvalue(const Values<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_vvalue_impl<dim, Values<dim>>(phi, i, q);
}



/**
 * A helper struct which provides a FEInterfaceValues-like interface for
 * stream functions. A stream function is the vector curl of scalar polynomial
 * shape functions in two space dimensions.
 */
template<int dim>
struct InterfaceValues
{
  static_assert(dim == 2, "Implemented for 2D only.");

  // TODO !!! in more recent deal.II versions FEInterfaceValues provides the
  // interfaces get_mapping() and get_fe(), such that fe_values_in becomes
  // obsolete
  InterfaceValues(const FEValues<dim> &          fe_values_in,
                  const FEInterfaceValues<dim> & fe_interface_values_in)
    : fe_values(fe_values_in.get_mapping(),
                fe_values_in.get_fe(),
                fe_values_in.get_quadrature(),
                fe_values_in.get_update_flags()),
      fe_interface_values(fe_values_in.get_mapping(),
                          fe_values_in.get_fe(),
                          fe_interface_values_in.get_quadrature(),
                          fe_interface_values_in.get_update_flags()),
      n_quadrature_points(fe_interface_values_in.n_quadrature_points)
  {
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType & cell,
         const unsigned int       face_no,
         const unsigned int       subface_no,
         const CellIteratorType & ncell,
         const unsigned int       nface_no,
         const unsigned int       nsubface_no)
  {
    fe_interface_values.reinit(cell, face_no, subface_no, ncell, nface_no, nsubface_no);
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType & cell, const unsigned int face_no)
  {
    fe_interface_values.reinit(cell, face_no);
  }

  unsigned int
  n_current_interface_dofs() const
  {
    return fe_interface_values.n_current_interface_dofs();
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return fe_values.get_fe();
  }

  std::vector<types::global_dof_index>
  get_interface_dof_indices() const
  {
    return fe_interface_values.get_interface_dof_indices();
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
    const auto & phi_left   = fe_interface_values.get_fe_face_values(0);
    const auto & curl_phi_i = compute_vcurl(phi_left, i, q);
    return curl_phi_i[c];
  }

  double
  shape_value_component_right(const unsigned int i,
                              const unsigned int q,
                              const unsigned int c) const
  {
    const auto & phi_right  = fe_interface_values.get_fe_face_values(1);
    const auto & curl_phi_i = compute_vcurl(phi_right, i, q);
    return curl_phi_i[c];
  }

  Tensor<1, dim>
  shape_grad_component_left(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    const auto & phi_left   = fe_interface_values.get_fe_face_values(0);
    const auto & hess_phi_i = phi_left.shape_hessian(i, q);

    Tensor<1, dim> grad;
    if(c == 0U)
    {
      grad[0] = hess_phi_i[0][1];
      grad[1] = hess_phi_i[1][1];
    }
    else if(c == 1U)
    {
      grad[0] = -hess_phi_i[0][0];
      grad[1] = -hess_phi_i[1][0];
    }

    return grad;
  }

  Tensor<1, dim>
  shape_grad_component_right(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    const auto & phi_right  = fe_interface_values.get_fe_face_values(1);
    const auto & hess_phi_i = phi_right.shape_hessian(i, q);

    Tensor<1, dim> grad;
    if(c == 0U)
    {
      grad[0] = hess_phi_i[0][1];
      grad[1] = hess_phi_i[1][1];
    }
    else if(c == 1U)
    {
      grad[0] = -hess_phi_i[0][0];
      grad[1] = -hess_phi_i[1][0];
    }

    return grad;
  }

  double
  jump(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    const auto [li, ri] = fe_interface_values.interface_dof_to_dof_indices(i);

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

    // const auto & jump_curl_phi_i = ::MW::compute_jump_vcurl(fe_interface_values, i, q);
    // return jump_curl_phi_i[c];
  }

  Tensor<1, dim>
  average_gradient(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    const auto [li, ri] = fe_interface_values.interface_dof_to_dof_indices(i);

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

    // AssertIndexRange(c, dim);
    // const auto &   av_hess_phi_i = fe_interface_values.average_hessian(i, q);
    // Tensor<1, dim> grad;
    // if(c == 0U)
    // {
    //   grad[0] = av_hess_phi_i[0][1];
    //   grad[1] = av_hess_phi_i[1][1];
    // }
    // else if(c == 1U)
    // {
    //   grad[0] = -av_hess_phi_i[0][0];
    //   grad[1] = -av_hess_phi_i[1][0];
    // }
    // return grad;
  }

  FEValues<dim>          fe_values;
  FEInterfaceValues<dim> fe_interface_values;
  unsigned int           n_quadrature_points;
};



template<int dim>
SymmetricTensor<2, dim>
compute_average_symgrad(const InterfaceValues<dim> & phi,
                        const unsigned int           i,
                        const unsigned int           q)
{
  return ::MW::compute_average_symgrad_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
Tensor<1, dim>
compute_vjump(const InterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_vjump_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
Tensor<1, dim>
compute_vjump_tangential(const InterfaceValues<dim> & phi,
                         const unsigned int           i,
                         const unsigned int           q)
{
  return ::MW::compute_vjump_tangential_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
struct ScratchData : public Mixed::ScratchData<dim>
{
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe_test,
              const FiniteElement<dim> & fe_ansatz,
              const unsigned int         n_q_points_1d,
              const UpdateFlags          update_flags_test,
              const UpdateFlags          update_flags_ansatz,
              const UpdateFlags          interface_update_flags_test = UpdateFlags::update_default,
              const UpdateFlags interface_update_flags_ansatz        = UpdateFlags::update_default)
    : Mixed::ScratchData<dim>(mapping,
                              fe_test,
                              fe_ansatz,
                              n_q_points_1d,
                              update_flags_test,
                              update_flags_ansatz,
                              interface_update_flags_test,
                              interface_update_flags_ansatz),
      stream_values(Mixed::ScratchData<dim>::fe_values_test),
      stream_interface_values(Mixed::ScratchData<dim>::fe_values_test,
                              Mixed::ScratchData<dim>::fe_interface_values_test),
      stream_values_ansatz(Mixed::ScratchData<dim>::fe_values_ansatz),
      stream_interface_values_ansatz(Mixed::ScratchData<dim>::fe_values_ansatz,
                                     Mixed::ScratchData<dim>::fe_interface_values_ansatz)
  {
  }

  ScratchData(const ScratchData<dim> & scratch_data_in)
    : Mixed::ScratchData<dim>(scratch_data_in),
      stream_values(scratch_data_in.fe_values_test),
      stream_interface_values(scratch_data_in.fe_values_test,
                              scratch_data_in.fe_interface_values_test),
      stream_values_ansatz(scratch_data_in.fe_values_ansatz),
      stream_interface_values_ansatz(scratch_data_in.fe_values_ansatz,
                                     scratch_data_in.fe_interface_values_ansatz)
  {
  }

  StreamFunction::Values<dim>          stream_values;
  StreamFunction::InterfaceValues<dim> stream_interface_values;
  StreamFunction::Values<dim>          stream_values_ansatz;
  StreamFunction::InterfaceValues<dim> stream_interface_values_ansatz;
};

} // namespace StreamFunction



namespace TestFunction
{
/**
 * A helper struct which provides a FEValues-like interface for arbitrary test
 * functions. The test functions are determined by the basis transformation
 * matrix @p shape_to_test_functions mapping shape to test functions. By a call
 * to reinit() one might select a subset of those test functions.
 */
template<int dim>
struct Values
{
  static_assert(dim == 2, "Implemented for 2D only.");

  Values(const FEValues<dim> & fe_values_in, const FullMatrix<double> & shape_to_test_functions_in)
    : fe_values(fe_values_in.get_mapping(),
                fe_values_in.get_fe(),
                fe_values_in.get_quadrature(),
                fe_values_in.get_update_flags()),
      shape_to_test_functions(shape_to_test_functions_in),
      n_quadrature_points(fe_values_in.n_quadrature_points)
  {
    AssertDimension(shape_to_test_functions.n(), fe_values_in.dofs_per_cell);
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType &          cell,
         const std::vector<unsigned int> & local_dof_indices_in = std::vector<unsigned int>{})
  {
    AssertDimension(shape_to_test_functions.n(), fe_values.dofs_per_cell);

    fe_values.reinit(cell);
    if(local_dof_indices_in.empty())
    {
      local_dof_indices.resize(shape_to_test_functions.m());
      std::iota(local_dof_indices.begin(), local_dof_indices.end(), 0U);
    }
    else
      local_dof_indices = local_dof_indices_in;

    AssertIndexRange(*std::max_element(local_dof_indices.cbegin(), local_dof_indices.cend()),
                     shape_to_test_functions.m());
  }

  unsigned int
  n_dofs_per_cell() const
  {
    return local_dof_indices.size();
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return fe_values.get_fe();
  }

  const std::vector<Point<dim>> &
  get_quadrature_points() const
  {
    return fe_values.get_quadrature_points();
  }

  double
  JxW(const unsigned int q) const
  {
    return fe_values.JxW(q);
  }

  double
  shape_value_component(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    AssertIndexRange(i, local_dof_indices.size());
    const auto ii = local_dof_indices[i];

    double value = 0.;
    for(auto j = 0U; j < shape_to_test_functions.n(); ++j)
    {
      value += shape_to_test_functions(ii, j) * fe_values.shape_value_component(j, q, c);
    }
    return value;
  }

  Tensor<1, dim>
  shape_grad_component(const unsigned int i, const unsigned int q, const unsigned int c) const
  {
    AssertIndexRange(i, local_dof_indices.size());
    const auto ii = local_dof_indices[i];

    Tensor<1, dim> grad;
    for(auto j = 0U; j < shape_to_test_functions.n(); ++j)
      grad += shape_to_test_functions(ii, j) * fe_values.shape_grad_component(j, q, c);
    return grad;
  }

  FEValues<dim>              fe_values;
  const FullMatrix<double> & shape_to_test_functions;
  unsigned int               n_quadrature_points;
  std::vector<unsigned int>  local_dof_indices;
};



template<int dim>
SymmetricTensor<2, dim>
compute_symgrad(const Values<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_symgrad_impl<dim, Values<dim>>(phi, i, q);
}



template<int dim>
Tensor<1, dim>
compute_vvalue(const Values<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_vvalue_impl<dim, Values<dim>>(phi, i, q);
}



std::vector<std::vector<unsigned int>>
make_test_to_active_shape_function_index_map(const FullMatrix<double> & trafomatrix)
{
  std::vector<std::vector<unsigned int>> map;
  map.reserve(trafomatrix.m());
  for(auto t = 0U; t < trafomatrix.m(); ++t)
  {
    std::vector<unsigned int> & active_indices = map.emplace_back();
    for(auto s = 0U; s < trafomatrix.n(); ++s)
      if(!has_nearly_zero_abs(trafomatrix(t, s)))
        active_indices.emplace_back(s);
  }
  AssertDimension(map.size(), trafomatrix.m());
  return map;
}



/**
 * A helper struct which provides a FEInterfaceValues-like interface for
 * arbitrary test functions. The test functions are determined by the basis
 * transformation matrices @p shape_to_test_functions_left and @p
 * shape_to_test_functions_right mapping shape to test functions on the left and
 * right cell, respectively.
 *
 * By passing the mapping of joint dof indices to the pair of indices on the
 * left and right cell, respectively, to the reinit() call it is possible to
 * activate only a subset of test functions. This mapping @p
 * joint_to_cell_dof_indices_in is mandatory and has to be passed by the user to
 * identify joint test functions.
 */
template<int dim>
struct InterfaceValues
{
  static_assert(dim == 2, "Implemented for 2D only.");

  InterfaceValues(const FEValues<dim> &          fe_values_in,
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
      test_to_active_shape_function_indices_left(
        make_test_to_active_shape_function_index_map(shape_to_test_functions_left_in)),
      test_to_active_shape_function_indices_right(
        make_test_to_active_shape_function_index_map(shape_to_test_functions_right_in)),
      active_shape_function_indices_left([&]() {
        std::vector<unsigned int> all_indices;
        for(const auto & active_indices : test_to_active_shape_function_indices_left)
          std::copy(active_indices.cbegin(),
                    active_indices.cend(),
                    std::back_inserter(all_indices));
        const auto unique_end = std::unique(all_indices.begin(), all_indices.end());
        return std::vector<unsigned int>(all_indices.begin(), unique_end);
      }()),
      active_shape_function_indices_right([&]() {
        std::vector<unsigned int> all_indices;
        for(const auto & active_indices : test_to_active_shape_function_indices_right)
          std::copy(active_indices.cbegin(),
                    active_indices.cend(),
                    std::back_inserter(all_indices));
        const auto unique_end = std::unique(all_indices.begin(), all_indices.end());
        return std::vector<unsigned int>(all_indices.begin(), unique_end);
      }()),
      n_quadrature_points(fe_interface_values_in.n_quadrature_points)
  {
    /// DEBUG
    // std::cout << "left: " << shape_to_test_functions_left.m() << "x"
    //           << shape_to_test_functions_left.n() << std::endl;
    // shape_to_test_functions_left.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-12);
    // for(const auto indices : test_to_active_shape_function_indices_left)
    //   std::cout << vector_to_string(indices) << std::endl;
    // std::cout << "right: " << shape_to_test_functions_right.m() << "x"
    //           << shape_to_test_functions_right.n() << std::endl;
    // shape_to_test_functions_right.print_formatted(std::cout, 3, true, 0, " ", 1., 1.e-12);
    // for(const auto indices : test_to_active_shape_function_indices_right)
    //   std::cout << vector_to_string(indices) << std::endl;
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType &                         cell,
         const unsigned int                               face_no,
         const unsigned int                               subface_no,
         const CellIteratorType &                         ncell,
         const unsigned int                               nface_no,
         const unsigned int                               nsubface_no,
         const std::vector<std::array<unsigned int, 2>> & joint_to_cell_dof_indices_in =
           std::vector<std::array<unsigned int, 2>>{})
  {
    fe_interface_values.reinit(cell, face_no, subface_no, ncell, nface_no, nsubface_no);

    if(joint_to_cell_dof_indices_in.empty())
    {
      std::vector<types::global_dof_index> global_dof_indices_left(
        shape_to_test_functions_left.n());
      cell->get_active_or_mg_dof_indices(global_dof_indices_left);
      std::vector<types::global_dof_index> global_dof_indices_right(
        shape_to_test_functions_right.n());
      ncell->get_active_or_mg_dof_indices(global_dof_indices_right);
      const auto joint_to_cell_dof_indices =
        make_joint_to_cell_dof_indices(global_dof_indices_left, global_dof_indices_right);

      /// DEBUG
      std::cout << "joint_to_cell_dof_indices: " << std::endl;
      for(const auto & [li, ri] : joint_to_cell_dof_indices)
        std::cout << " (" << li << "," << ri << ")";
      std::cout << std::endl;
    }

    else
    {
      joint_to_cell_dof_indices = joint_to_cell_dof_indices_in;

      /// DEBUG
      std::cout << "joint_to_cell_dof_indices_in: " << std::endl;
      for(const auto & [li, ri] : joint_to_cell_dof_indices_in)
        std::cout << " (" << li << "," << ri << ")";
      std::cout << std::endl;
    }
  }

  template<typename CellIteratorType>
  void
  reinit(const CellIteratorType &                         cell,
         const unsigned int                               face_no,
         const std::vector<std::array<unsigned int, 2>> & joint_to_cell_dof_indices_in)
  {
    fe_interface_values.reinit(cell, face_no);
    joint_to_cell_dof_indices = joint_to_cell_dof_indices_in;
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
    AssertIndexRange(i, shape_to_test_functions_left.m());

    const auto & fe_face_values_left = fe_interface_values.get_fe_face_values(0);
    double       value               = 0.;
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

    if(fe_interface_values.at_boundary())
    {
      Assert(li != numbers::invalid_unsigned_int,
             ExcMessage("Invalid test function index on the left cell"));
      return shape_value_component_left(li, q, c);
    }

    double jump = 0.;

    if(li != numbers::invalid_unsigned_int)
      jump += shape_value_component_left(li, q, c);
    if(ri != numbers::invalid_unsigned_int)
      jump -= shape_value_component_right(ri, q, c);

    return jump;
  }

  /**
   * For both cells (left and right) the test functions represented by this
   * class are linear combinations of shape functions of the underlying finite
   * element. We assume a test function is joint between cells if all involved
   * shape functions are joint, that is the coefficients for both transformation
   * matrices are nonzero???
   */
  /// TODO rename dof_indices -> test_indices
  std::vector<std::array<unsigned int, 2>>
  make_joint_to_cell_dof_indices(
    const std::vector<types::global_dof_index> & global_dof_indices_left,
    const std::vector<types::global_dof_index> & global_dof_indices_right) const
  {
    AssertDimension(shape_to_test_functions_left.n(), global_dof_indices_left.size());
    AssertDimension(shape_to_test_functions_right.n(), global_dof_indices_right.size());

    std::vector<std::array<unsigned int, 2>> map;

    std::vector<types::global_dof_index> active_global_dof_indices_both;
    std::transform(active_shape_function_indices_left.cbegin(),
                   active_shape_function_indices_left.cend(),
                   std::back_inserter(active_global_dof_indices_both),
                   [&](const auto s) { return global_dof_indices_left[s]; });
    std::transform(active_shape_function_indices_right.cbegin(),
                   active_shape_function_indices_right.cend(),
                   std::back_inserter(active_global_dof_indices_both),
                   [&](const auto s) { return global_dof_indices_right[s]; });
    std::sort(active_global_dof_indices_both.begin(), active_global_dof_indices_both.end());

    const bool no_joint_dof_indices =
      std::unique(active_global_dof_indices_both.begin(), active_global_dof_indices_both.end()) ==
      active_global_dof_indices_both.end();

    if(no_joint_dof_indices)
    {
      map.reserve(shape_to_test_functions_left.m() + shape_to_test_functions_right.m());
      for(auto li = 0U; li < shape_to_test_functions_left.m(); ++li)
        map.emplace_back(std::array<unsigned int, 2>{li, numbers::invalid_unsigned_int});
      for(auto ri = 0U; ri < shape_to_test_functions_right.m(); ++ri)
        map.emplace_back(std::array<unsigned int, 2>{numbers::invalid_unsigned_int, ri});
    }

    else
    {
      std::vector<std::array<unsigned int, 2>> joint_indices;
      for(auto li = 0U; li < shape_to_test_functions_left.m(); ++li)
      {
        const auto & active_indices_left = test_to_active_shape_function_indices_left[li];
        std::vector<types::global_dof_index> active_global_dof_indices_left;
        std::transform(active_indices_left.cbegin(),
                       active_indices_left.cend(),
                       std::back_inserter(active_global_dof_indices_left),
                       [&](const auto s) { return global_dof_indices_left[s]; });
        std::sort(active_global_dof_indices_left.begin(), active_global_dof_indices_left.end());

        for(auto ri = 0U; ri < shape_to_test_functions_right.m(); ++ri)
        {
          const auto & active_indices_right = test_to_active_shape_function_indices_right[ri];
          std::vector<types::global_dof_index> active_global_dof_indices_right;
          std::transform(active_indices_right.cbegin(),
                         active_indices_right.cend(),
                         std::back_inserter(active_global_dof_indices_right),
                         [&](const auto s) { return global_dof_indices_right[s]; });
          std::sort(active_global_dof_indices_right.begin(), active_global_dof_indices_right.end());

          const bool is_joint = active_global_dof_indices_left == active_global_dof_indices_right;

          if(is_joint)
          {
            joint_indices.emplace_back(std::array<unsigned int, 2>{li, ri});
            break;
          }
        }
      }

      std::vector<unsigned int> joint_indices_left;
      std::transform(joint_indices.cbegin(),
                     joint_indices.cend(),
                     std::back_inserter(joint_indices_left),
                     [&](const auto & liri) { return liri[0]; });
      std::sort(joint_indices_left.begin(), joint_indices_left.end());
      Assert(std::unique(joint_indices_left.begin(), joint_indices_left.end()) ==
               joint_indices_left.end(),
             ExcMessage("Duplicates..."));

      std::vector<unsigned int> joint_indices_right;
      std::transform(joint_indices.cbegin(),
                     joint_indices.cend(),
                     std::back_inserter(joint_indices_right),
                     [&](const auto & liri) { return liri[1]; });
      std::sort(joint_indices_right.begin(), joint_indices_right.end());
      Assert(std::unique(joint_indices_right.begin(), joint_indices_right.end()) ==
               joint_indices_right.end(),
             ExcMessage("Duplicates..."));

      for(auto li = 0U; li < shape_to_test_functions_left.m(); ++li)
        if(std::find(joint_indices_left.cbegin(), joint_indices_left.cend(), li) ==
           joint_indices_left.cend())
          map.emplace_back(std::array<unsigned int, 2>{li, numbers::invalid_unsigned_int});

      std::copy(joint_indices.cbegin(), joint_indices.cend(), std::back_inserter(map));

      for(auto ri = 0U; ri < shape_to_test_functions_right.m(); ++ri)
        if(std::find(joint_indices_right.cbegin(), joint_indices_right.cend(), ri) ==
           joint_indices_right.cend())
          map.emplace_back(std::array<unsigned int, 2>{numbers::invalid_unsigned_int, ri});
    }

    return map;
  }

  FEValues<dim>                            fe_values;
  FEInterfaceValues<dim>                   fe_interface_values;
  const FullMatrix<double> &               shape_to_test_functions_left;
  const FullMatrix<double> &               shape_to_test_functions_right;
  std::vector<std::vector<unsigned int>>   test_to_active_shape_function_indices_left;
  std::vector<std::vector<unsigned int>>   test_to_active_shape_function_indices_right;
  std::vector<unsigned int>                active_shape_function_indices_left;
  std::vector<unsigned int>                active_shape_function_indices_right;
  unsigned int                             n_quadrature_points;
  std::vector<std::array<unsigned int, 2>> joint_to_cell_dof_indices;
};



template<int dim>
Tensor<1, dim>
compute_vaverage(const InterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_vaverage_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
SymmetricTensor<2, dim>
compute_average_symgrad(const InterfaceValues<dim> & phi,
                        const unsigned int           i,
                        const unsigned int           q)
{
  return ::MW::compute_average_symgrad_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
Tensor<1, dim>
compute_vjump(const InterfaceValues<dim> & phi, const unsigned int i, const unsigned int q)
{
  return ::MW::compute_vjump_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
Tensor<1, dim>
compute_vjump_tangential(const InterfaceValues<dim> & phi,
                         const unsigned int           i,
                         const unsigned int           q)
{
  return ::MW::compute_vjump_tangential_impl<dim, InterfaceValues<dim>>(phi, i, q);
}



template<int dim>
struct ScratchData : public StreamFunction::ScratchData<dim>
{
  using Base = typename StreamFunction::ScratchData<dim>;

  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe_test,
              const FiniteElement<dim> & fe_ansatz,
              const unsigned int         n_q_points_1d,
              const FullMatrix<double> & shape_to_test_functions_in,
              const UpdateFlags          update_flags_test,
              const UpdateFlags          update_flags_ansatz,
              const UpdateFlags          interface_update_flags_test = UpdateFlags::update_default,
              const UpdateFlags interface_update_flags_ansatz        = UpdateFlags::update_default)
    : StreamFunction::ScratchData<dim>(mapping,
                                       fe_test,
                                       fe_ansatz,
                                       n_q_points_1d,
                                       update_flags_test,
                                       update_flags_ansatz,
                                       interface_update_flags_test,
                                       interface_update_flags_ansatz),
      test_values(StreamFunction::ScratchData<dim>::fe_values_test, shape_to_test_functions_in),
      test_interface_values(StreamFunction::ScratchData<dim>::fe_values_test,
                            StreamFunction::ScratchData<dim>::fe_interface_values_test,
                            shape_to_test_functions_in,
                            shape_to_test_functions_in)
  {
  }

  ScratchData(const ScratchData<dim> & scratch_data_in)
    : StreamFunction::ScratchData<dim>(scratch_data_in),
      test_values(scratch_data_in.fe_values_test,
                  scratch_data_in.test_values.shape_to_test_functions),
      test_interface_values(scratch_data_in.fe_values_test,
                            scratch_data_in.fe_interface_values_test,
                            scratch_data_in.test_values.shape_to_test_functions,
                            scratch_data_in.test_values.shape_to_test_functions)
  {
  }

  TestFunction::Values<dim>          test_values;
  TestFunction::InterfaceValues<dim> test_interface_values;
};

} // namespace TestFunction



namespace DoF
{
/**
 * This struct locally stores matrix and right-hand side contributions on cells and/or
 * (inter)faces. It is designed to be used in MeshWorker routines.
 */
struct CopyData
{
  struct CellData
  {
    CellData(const unsigned int n_rows, const unsigned int n_columns)
      : matrix(n_rows, n_columns), rhs(n_rows), dof_indices(n_rows), dof_indices_column(n_columns)
    {
    }

    CellData(const unsigned int n_local_dofs)
      : matrix(n_local_dofs, n_local_dofs), rhs(n_local_dofs), dof_indices(n_local_dofs)
    {
    }

    CellData() : CellData(0U)
    {
    }

    CellData(const CellData &) = default;

    CellData(CellData &&) = default;

    CellData &
    operator=(const CellData &) = default;

    CellData &
    operator=(CellData &&) = default;

    FullMatrix<double>                   matrix;
    Vector<double>                       rhs;
    std::vector<types::global_dof_index> dof_indices;
    std::vector<types::global_dof_index> dof_indices_column;
  };

  struct FaceData
  {
    FaceData(const unsigned int n_rows, const unsigned int n_columns)
      : matrix(n_rows, n_columns), rhs(n_rows), dof_indices(n_rows), dof_indices_column(n_columns)
    {
    }

    FaceData(const unsigned int n_local_dofs)
      : matrix(n_local_dofs, n_local_dofs), rhs(n_local_dofs), dof_indices(n_local_dofs)
    {
    }

    FaceData() : FaceData(0U)
    {
    }

    FaceData(const FaceData &) = default;

    FaceData(FaceData &&) = default;

    FaceData &
    operator=(const FaceData &) = default;

    FaceData &
    operator=(FaceData &&) = default;

    FullMatrix<double>                   matrix;
    Vector<double>                       rhs;
    std::vector<types::global_dof_index> dof_indices;
    std::vector<types::global_dof_index> dof_indices_column;
  };

  std::vector<CellData> cell_data;
  std::vector<FaceData> face_data;
};

} // namespace DoF



namespace Cell
{
/**
 * This struct stores local contributions for each cell and/or
 * (inter)face. It is designed to be used in MeshWorker routines.
 */
struct CopyData
{
  struct CellData
  {
    CellData(const unsigned int n_indices) : values(n_indices), cell_indices(n_indices)
    {
    }

    CellData() : CellData(0U)
    {
    }

    CellData(const CellData &) = default;

    CellData(CellData &&) = default;

    CellData &
    operator=(const CellData &) = default;

    CellData &
    operator=(CellData &&) = default;

    Vector<double>                       values;
    std::vector<types::global_dof_index> cell_indices;
  };

  struct FaceData
  {
    FaceData(const unsigned int n_indices) : values(n_indices), cell_indices(n_indices)
    {
    }

    FaceData() : FaceData(0U)
    {
    }

    FaceData(const FaceData &) = default;

    FaceData(FaceData &&) = default;

    FaceData &
    operator=(const FaceData &) = default;

    FaceData &
    operator=(FaceData &&) = default;

    Vector<double>                       values;
    std::vector<types::global_dof_index> cell_indices;
  };

  std::vector<CellData> cell_data;
  std::vector<FaceData> face_data;
};

} // namespace Cell

} // namespace MW

#endif /* APPS_COMMONINTEGRATOR_H_ */
