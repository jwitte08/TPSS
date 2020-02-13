#ifndef FDEVALUATION_H
#define FDEVALUATION_H

#include "TPSS.h"
#include "subdomain_handler.h"

#include <array>
#include <memory>



/**
 * This class provides an interface to compute integrals regarding univariate
 * shape functions on intervals in real space. The intervals stem from Cartesian
 * patches of cells (currently a single cell or vertex patch).  ... TODO
 *
 * The @p patch_action() functionality allows computation of univariate finite
 * element matrices for each spatial dimension (here refered to as @p
 * direction). For example, one might compute univariate mass matrices M_d and
 * FEM discretizations of the one-dimensional Laplace operator L_d in dimensions
 * d = 0, 1. Then, thanks to separability of the d-dimensional Laplace operator
 * the associated-dimensional FEM discretization L is given by the rank-d
 * Kronecker decomposition:
 *
 *    L = L_1 x M_2 + M_1 x L_2
 *
 * The tensor decomposition facilitates a fast inversion of L at the cost of
 * one-dimensional (generalized) eigenvalue problems known as fast
 * diagonalization method (FDM), which breaks the curse of dimensionality. This
 * deciphers the name of this class.  ... TODO
 */

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
class FDEvaluation
{
public:
  using CellAssembler =
    typename TPSS::MatrixEvaluator<fe_degree + 1, n_q_points_1d, VectorizedArray<Number>>;
  using value_type                         = Number;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  static constexpr unsigned int fe_order               = fe_degree + 1;
  static constexpr unsigned int n_q_points_1d_static   = n_q_points_1d;
  static constexpr unsigned int n_dofs_per_cell_static = Utilities::pow(fe_order, dim);

  FDEvaluation(const SubdomainHandler<dim, Number> & sd_handler_in,
               const unsigned int                    dofh_index = 0);

  ~FDEvaluation();

  FDEvaluation(const FDEvaluation & other) = delete;

  FDEvaluation &
  operator=(const FDEvaluation & other) = delete;

  void
  reinit(const unsigned int patch);

  void
  evaluate(const bool gradients);

  // void
  // evaluate(const bool gradients);

  VectorizedArray<Number>
  get_average_factor(const int direction, const int cell_no, const int face_no) const;

  std::bitset<macro_size>
  get_boundary_mask(const int direction, const int cell_no, const int face_no) const;

  TPSS::DoFLayout
  get_dof_layout() const;

  const VectorizedArray<Number> &
  get_h(const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  get_JxW(const int q_point_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  get_q_weight(const int q_point_no, const int direction) const;

  VectorizedArray<Number>
  get_normal(const int face_no) const;

  Tensor<1, dim, VectorizedArray<Number>>
  get_normal_vector(const int face_no, const int direction) const;

  const SubdomainHandler<dim, Number> &
  get_subdomain_handler() const;

  const VectorizedArray<Number> &
  shape_value(const int dof, const int q_point_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  shape_value_face(const int dof, const int face_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  shape_gradient(const int dof, const int q_point_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  shape_gradient_face(const int dof,
                      const int face_no,
                      const int direction,
                      const int cell_no) const;

  const ArrayView<VectorizedArray<Number>>
  acquire_scratch_chunk(const std::size_t size);

  void
  release_scratch_chunks();

  const unsigned int       level;
  const unsigned int       n_subdomains;
  const unsigned int       n_colors;
  const TPSS::PatchVariant patch_variant;
  const unsigned int       n_cells_per_direction;

  void
  compute_unit_mass(const ArrayView<VectorizedArray<Number>> & matrix) const;

  template<typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const FDEvaluation & eval_ansatz, CellOperation && cell_operation) const
  {
    if(patch_variant == TPSS::PatchVariant::cell)
    {
      constexpr unsigned int n_cells_per_direction =
        TPSS::UniversalInfo<dim>::n_cells_per_direction(TPSS::PatchVariant::cell);
      return patch_action_impl<n_cells_per_direction>(eval_ansatz,
                                                      std::forward<CellOperation>(cell_operation));
    }
    else if(patch_variant == TPSS::PatchVariant::vertex)
    {
      constexpr auto n_cells_per_direction =
        TPSS::UniversalInfo<dim>::n_cells_per_direction(TPSS::PatchVariant::vertex);
      return patch_action_impl<n_cells_per_direction>(eval_ansatz,
                                                      std::forward<CellOperation>(cell_operation));
    }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template<typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(CellOperation && cell_operation) const
  {
    return patch_action(*this, std::forward<CellOperation>(cell_operation));
  }

  template<typename CellOperation, typename FaceOperation, typename InterfaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const FDEvaluation &       eval_ansatz,
               const CellOperation &      cell_operation,
               const FaceOperation &      face_operation,
               const InterfaceOperation & interface_operation) const
  {
    if(patch_variant == TPSS::PatchVariant::cell)
    {
      return patch_action_dgcp_impl(eval_ansatz, cell_operation, face_operation);
    }
    else if(patch_variant == TPSS::PatchVariant::vertex)
    {
      return patch_action_dgvp_impl(eval_ansatz,
                                    cell_operation,
                                    face_operation,
                                    interface_operation);
    }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template<typename CellOperation, typename FaceOperation, typename InterfaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const CellOperation &      cell_operation,
               const FaceOperation &      face_operation,
               const InterfaceOperation & interface_operation) const
  {
    return patch_action(*this, cell_operation, face_operation, interface_operation);
  }

private:
  using This = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;

  template<TPSS::PatchVariant variant>
  void
  malloc_fedata()
  {
    Assert(this->scratch_fedata != nullptr, ExcInternalError());

    constexpr unsigned int scratch_pad_length  = fe_order * fe_order * dim * 2;
    constexpr unsigned int unit_weights_length = n_q_points_1d * dim;
    constexpr unsigned int values_length       = n_q_points_1d * fe_order * dim;
    constexpr unsigned int values_length_face  = fe_order * 2 * dim;
    constexpr unsigned int n_cells_per_direction =
      TPSS::UniversalInfo<dim>::n_cells_per_direction(variant);
    constexpr unsigned int gradients_length =
      n_q_points_1d * fe_order * n_cells_per_direction * dim;
    constexpr unsigned int gradients_length_face = fe_order * 2 * n_cells_per_direction * dim;
    constexpr unsigned int JxWs_length           = n_q_points_1d * n_cells_per_direction * dim;
    constexpr unsigned int size_to_be_allocated =
      scratch_pad_length + unit_weights_length + values_length + values_length_face +
      gradients_length + gradients_length_face + JxWs_length; // total size

    // *** allocate memory
    this->scratch_fedata->resize(size_to_be_allocated);
    this->q_weights_unit = this->scratch_fedata->begin();
    this->values         = this->q_weights_unit + unit_weights_length;
    this->values_face    = this->values + values_length;
    for(unsigned int d = 0; d < dim; ++d)
      this->gradients[d] = this->values_face + values_length_face +
                           n_q_points_1d * fe_order * n_cells_per_direction * d;
    for(unsigned int d = 0; d < dim; ++d)
      this->gradients_face[d] =
        this->gradients[0] + gradients_length + fe_order * 2 * n_cells_per_direction * d;
    this->JxWs                  = this->gradients_face[0] + gradients_length_face;
    this->scratch_fedata_end    = this->JxWs + JxWs_length;
    this->scratch_pad           = this->scratch_fedata_end;
    this->scratch_pad_remainder = this->scratch_pad;
    this->scratch_pad_end       = this->scratch_pad + scratch_pad_length;

    AssertDimension(std::distance(this->scratch_fedata->begin(), this->scratch_pad_end),
                    size_to_be_allocated); // check allocation
  }

  VectorizedArray<Number> &
  get_JxW_impl(const int q_point_no, const int direction, const int cell_no) const;

  VectorizedArray<Number> &
  get_q_weight_impl(const int q_point_no, const int direction) const;

  VectorizedArray<Number> &
  shape_value_impl(const int dof,
                   const int q_point_no,
                   const int direction,
                   const int cell_no) const;

  VectorizedArray<Number> &
  shape_value_face_impl(const int dof,
                        const int face_no,
                        const int direction,
                        const int cell_no) const;

  VectorizedArray<Number> &
  shape_gradient_impl(const int dof,
                      const int q_point_no,
                      const int direction,
                      const int cell_no) const;

  VectorizedArray<Number> &
  shape_gradient_face_impl(const int dof,
                           const int face_no,
                           const int direction,
                           const int cell_no) const;

  template<int n_cells_per_direction, typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action_impl(const FDEvaluation & eval_ansatz, CellOperation && cell_operation) const
  {
    using MatrixType                            = Table<2, VectorizedArray<Number>>;
    constexpr unsigned int n_dofs_per_direction = fe_order * n_cells_per_direction;

    std::array<MatrixType, dim> matrices;
    for(int direction = 0; direction < dim; ++direction)
    {
      MatrixType & matrix = matrices[direction];
      matrix.reinit(n_dofs_per_direction, n_dofs_per_direction);
      for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
      {
        MatrixType cell_matrix{fe_order, fe_order};
        std::forward<decltype(cell_operation)>(
          cell_operation)(eval_ansatz, *this, cell_matrix, direction, cell_no);
        submit_cell_matrix(matrix, cell_matrix, cell_no, cell_no);
      }
    }
    AssertDimension(n_dofs_per_direction, matrices.front().n_rows());
    AssertDimension(n_dofs_per_direction, matrices.back().n_cols());
    return matrices;
  }

  template<typename CellOperation, typename FaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action_dgcp_impl(const FDEvaluation &  eval_ansatz,
                         const CellOperation & cell_operation,
                         const FaceOperation & face_operation) const
  {
    constexpr unsigned int n_dofs_per_direction = fe_order;
    using MatrixType                            = Table<2, VectorizedArray<Number>>;

    std::array<MatrixType, dim> matrices;
    for(int direction = 0; direction < dim; ++direction)
    {
      auto & matrix = matrices[direction];
      matrix.reinit(n_dofs_per_direction, n_dofs_per_direction);
      MatrixType cell_matrix{fe_order, fe_order};

      cell_operation(eval_ansatz, *this, cell_matrix, direction, 0 /*cell_no*/);

      for(const int face_no : {0, 1})
      {
        face_operation(eval_ansatz, *this, cell_matrix, direction, /*cell_no*/ 0, face_no);
      }
      submit_cell_matrix(matrix, cell_matrix, 0 /*cell_no*/, 0 /* cell_no*/);
    }

    AssertDimension(n_dofs_per_direction, matrices.front().n_rows());
    AssertDimension(n_dofs_per_direction, matrices.back().n_cols());
    return matrices;
  }

  template<typename CellOperation, typename FaceOperation, typename InterfaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action_dgvp_impl(const FDEvaluation &       eval_ansatz,
                         const CellOperation &      cell_operation,
                         const FaceOperation &      face_operation,
                         const InterfaceOperation & interface_operation) const
  {
    using MatrixType                            = Table<2, VectorizedArray<Number>>;
    constexpr unsigned int n_dofs_per_direction = fe_order * 2;

    std::array<MatrixType, dim> matrices;
    for(int direction = 0; direction < dim; ++direction)
    {
      auto & matrix = matrices[direction];
      matrix.reinit(n_dofs_per_direction, n_dofs_per_direction);
      MatrixType cell_matrix0{fe_order, fe_order}, cell_matrix1{fe_order, fe_order};

      // *** CELL integral on cell 0 & 1
      cell_operation(eval_ansatz, *this, cell_matrix0, direction, 0);
      cell_operation(eval_ansatz, *this, cell_matrix1, direction, 1);

      // *** FACE integrals at patch boundary, i.e. face 0 on cell 0 & face 1 on cell 1
      face_operation(eval_ansatz, *this, cell_matrix0, direction, /*cell_no*/ 0, /*face_no*/ 0);
      face_operation(eval_ansatz, *this, cell_matrix1, direction, /*cell_no*/ 1, /*face_no*/ 1);

      // *** FACE integrals at interior face: face 1 on cell 0 & face 0 on cell 1
      face_operation(eval_ansatz, *this, cell_matrix0, direction, 0, 1);
      face_operation(eval_ansatz, *this, cell_matrix1, direction, 1, 0);

      submit_cell_matrix(matrix, cell_matrix0, 0, 0);
      submit_cell_matrix(matrix, cell_matrix1, 1, 1);

      // *** INTERFACE between cell 0 (seen as face 1) & cell 1 (seen as face 0) ***/
      MatrixType cell_matrix01{fe_order, fe_order},
        cell_matrix10{fe_order, fe_order}; // first parameter represents the cell_no associated to
                                           // the test functions and second parameter is the
                                           // cell_no associated to the ansatz functions
      interface_operation(eval_ansatz, *this, cell_matrix01, cell_matrix10, direction);
      submit_cell_matrix(matrix, cell_matrix01, 0, 1);
      submit_cell_matrix(matrix, cell_matrix10, 1, 0);
    }

    AssertDimension(n_dofs_per_direction, matrices.front().n_rows());
    AssertDimension(n_dofs_per_direction, matrices.back().n_cols());
    return matrices;
  }

  static void submit_cell_matrix(Table<2, VectorizedArray<Number>> &       subdomain_matrix,
                                 const Table<2, VectorizedArray<Number>> & cell_matrix,
                                 const unsigned int                        cell_no_row,
                                 const unsigned int                        cell_no_col);

  const SubdomainHandler<dim, Number> & sd_handler;

  const TPSS::PatchDoFWorker<dim, Number> patch_worker;

  const TPSS::MappingInfo<dim, Number> & mapping_info;

  const MatrixFree<dim, Number> & mf_storage;

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> & shape_info;

  unsigned int patch_id = numbers::invalid_unsigned_int;

  /**
   * Indicates which face of the macro patch is at the physical boundary. Local
   * face numbering follows the face numbering in GeometryInfo.
   */
  std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> at_boundary_masks;

  /**
   * Boundary id for each face of the macro patch.
   */
  std::array<std::array<types::boundary_id, macro_size>, GeometryInfo<dim>::faces_per_cell>
    boundary_ids;

  /**
   * quadrature weights on the unit cell
   */
  VectorizedArray<Number> * q_weights_unit = nullptr;

  /**
   * lexicographical ordering: cell_no_1d < direction
   */
  const VectorizedArray<Number> * h_lengths = nullptr;

  /**
   * lexicographical ordering: q_point_index < cell_no_1d < direction
   */
  VectorizedArray<Number> * JxWs = nullptr;

  /**
   * lexicographical ordering: q_point_index < dof_index < direction
   */
  VectorizedArray<Number> * values = nullptr;

  /**
   * lexicographical ordering: dof_index < face_no_1d < direction
   */
  VectorizedArray<Number> * values_face = nullptr;

  /**
   * lexicographical ordering: q_point_index < dof_index < cell_no
   */
  std::array<VectorizedArray<Number> *, dim> gradients;

  /**
   * lexicographical ordering: dof_index < face_no < cell_no
   */
  std::array<VectorizedArray<Number> *, dim> gradients_face;

  /**
   * scratch data array provided by underlying MatrixFree
   */
  AlignedVector<VectorizedArray<Number>> * scratch_fedata;

  /**
   * end of memory occupied by shape functions and cell-mapping
   */
  VectorizedArray<Number> * scratch_fedata_end = nullptr;

  /**
   * begin of memory allocated for user/small calculations
   */
  VectorizedArray<Number> * scratch_pad = nullptr;

  /**
   * begin of remaining memory of the scratch_pad
   */
  VectorizedArray<Number> * scratch_pad_remainder = nullptr;

  /**
   * end of memory allocated for user/small calculations
   */
  VectorizedArray<Number> * scratch_pad_end = nullptr;

  bool values_filled    = false;
  bool gradients_filled = false;
};



// ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::FDEvaluation(
  const SubdomainHandler<dim, Number> & sd_handler_in,
  const unsigned int                    dofh_index)
  : level(sd_handler_in.get_additional_data().level),
    n_subdomains(sd_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(sd_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    patch_variant(sd_handler_in.get_additional_data().patch_variant),
    n_cells_per_direction(TPSS::UniversalInfo<dim>::n_cells_per_direction(patch_variant)),
    sd_handler(sd_handler_in),
    patch_worker(sd_handler_in.get_dof_info(dofh_index)),
    mapping_info(sd_handler_in.get_mapping_info()),
    mf_storage(sd_handler_in.get_matrix_free()),
    shape_info(patch_worker.get_shape_info(0)),
    scratch_fedata(mf_storage.acquire_scratch_data())
{
  if(patch_variant == TPSS::PatchVariant::cell)
    malloc_fedata<TPSS::PatchVariant::cell>();
  else if(patch_variant == TPSS::PatchVariant::vertex)
    malloc_fedata<TPSS::PatchVariant::vertex>();
  else
    AssertThrow(false, ExcNotImplemented());

  /// Extract one-dimensional quadrature weights of reference interval
  for(auto d = 0U; d < dim; ++d)
  {
    const auto & shape_info = patch_worker.get_shape_info(d);
    const auto & quadrature = shape_info.quadrature;
    AssertIndexRange(static_cast<int>(quadrature.size()), n_q_points_1d + 1);
    const auto & unit_weights = quadrature.get_weights();
    auto         W            = &(get_q_weight_impl(0, d));
    std::copy(unit_weights.cbegin(), unit_weights.cend(), W);

    /// Extract one-dimensional shape function values on cell and face quadratures
    auto values_in_begin = shape_info.shape_values.begin();
    auto values_begin    = &(shape_value_impl(0, 0, d, 0));
    std::copy_n(values_in_begin, fe_order * n_q_points_1d, values_begin);
    for(const auto face_no : {0, 1})
    {
      auto values_face_in_begin = shape_info.shape_data_on_face[face_no].begin();
      auto values_face_begin    = &(shape_value_face_impl(0, face_no, d, 0));
      std::copy_n(values_face_in_begin, fe_order, values_face_begin);
    }
  }
  this->values_filled = true;
}

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::~FDEvaluation()
{
  try
  {
    mf_storage.release_scratch_data(scratch_fedata);
  }
  catch(...)
  {
  }
  scratch_fedata_end    = nullptr;
  scratch_fedata        = nullptr;
  values_face           = nullptr;
  values                = nullptr;
  JxWs                  = nullptr;
  h_lengths             = nullptr;
  q_weights_unit        = nullptr;
  scratch_pad           = nullptr;
  scratch_pad_remainder = nullptr;
  scratch_pad_end       = nullptr;
}

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::reinit(const unsigned int patch)
{
  AssertIndexRange(patch, n_subdomains);
  patch_id = patch;

  const auto & at_boundary_masks_in = patch_worker.get_at_boundary_masks(patch);
  std::copy_n(at_boundary_masks_in.cbegin(),
              GeometryInfo<dim>::faces_per_cell,
              at_boundary_masks.begin());

  if(TPSS::DoFLayout::Q == get_dof_layout())
  {
    const auto & boundary_ids_in = patch_worker.get_boundary_ids(patch);
    std::copy_n(boundary_ids_in.cbegin(), GeometryInfo<dim>::faces_per_cell, boundary_ids.begin());
  }

  h_lengths = mapping_info.template h_lengths_begin(patch);

  gradients_filled = false;
  evaluate(true);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const ArrayView<VectorizedArray<Number>>
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::acquire_scratch_chunk(const std::size_t size)
{
  auto && chunk =
    make_array_view(this->scratch_pad_remainder, (this->scratch_pad_remainder += size));
  Assert(std::distance(this->scratch_pad_remainder, this->scratch_pad_end) >= 0,
         ExcMessage("Acquired scratch chunk exceeds scratch pad!"));
  return chunk;
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::compute_unit_mass(
  const ArrayView<VectorizedArray<Number>> & matrix) const
{
  AssertDimension(matrix.size(), fe_order * fe_order);
  CellAssembler::template assemble<false>(
    make_array_view(this->values, this->values + fe_order * n_q_points_1d),
    make_array_view(this->values, this->values + fe_order * n_q_points_1d),
    make_array_view(this->q_weights_unit, this->q_weights_unit + n_q_points_1d),
    matrix);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number>
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_average_factor(const int direction,
                                                                        const int cell_no,
                                                                        const int face_no) const
{
  auto         factor      = make_vectorized_array<Number>(0.5);
  const auto & at_boundary = get_boundary_mask(direction, cell_no, face_no);
  for(auto lane = 0U; lane < macro_size; ++lane)
    if(at_boundary[lane])
      factor[lane] = 1.;
  return factor;
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline std::bitset<FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::macro_size>
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_boundary_mask(const int direction,
                                                                       const int cell_no,
                                                                       const int face_no) const
{
  AssertIndexRange(direction, dim);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(face_no, 2);
  if(patch_variant == TPSS::PatchVariant::cell)
    return at_boundary_masks[direction * 2 + face_no];
  else if(patch_variant == TPSS::PatchVariant::vertex)
  {
    const bool is_interior_face = (cell_no == 0 && face_no == 1) || (cell_no == 1 && face_no == 0);
    if(!is_interior_face)
      return at_boundary_masks[direction * 2 + face_no];
  }
  else
    AssertThrow(false, ExcNotImplemented());
  return std::bitset<macro_size>{0};
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline TPSS::DoFLayout
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_dof_layout() const
{
  return patch_worker.get_dof_info().get_dof_layout();
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_h(const int direction,
                                                           const int cell_no) const
{
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->h_lengths + cell_no + direction * n_cells_per_direction);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_JxW(const int q_point_no,
                                                             const int direction,
                                                             const int cell_no) const
{
  return get_JxW_impl(q_point_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_JxW_impl(const int q_point_no,
                                                                  const int direction,
                                                                  const int cell_no) const
{
  AssertIndexRange(q_point_no, n_q_points_1d);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->JxWs + q_point_no + cell_no * n_q_points_1d +
           direction * n_q_points_1d * n_cells_per_direction);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_q_weight(const int q_point_no,
                                                                  const int direction) const
{
  return get_q_weight_impl(q_point_no, direction);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_q_weight_impl(const int q_point_no,
                                                                       const int direction) const
{
  AssertIndexRange(q_point_no, n_q_points_1d);
  AssertIndexRange(direction, dim);
  return *(this->q_weights_unit + q_point_no + direction * n_q_points_1d);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number>
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_normal(const int face_no) const
{
  return make_vectorized_array<Number>(face_no == 0 ? -1. : 1.);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline Tensor<1, dim, VectorizedArray<Number>>
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_normal_vector(const int face_no,
                                                                       const int direction) const
{
  Tensor<1, dim, VectorizedArray<Number>> normal_vector;
  normal_vector *= 0.;
  normal_vector[direction] = get_normal(face_no);
  return normal_vector;
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const SubdomainHandler<dim, Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::get_subdomain_handler() const
{
  return sd_handler;
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::release_scratch_chunks()
{
  this->scratch_pad_remainder = this->scratch_pad;
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_value(const int dof,
                                                                 const int q_point_no,
                                                                 const int direction,
                                                                 const int cell_no) const
{
  return shape_value_impl(dof, q_point_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_value_impl(const int dof,
                                                                      const int q_point_no,
                                                                      const int direction,
                                                                      const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(q_point_no, n_q_points_1d);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  (void)cell_no;
  constexpr auto n_values_1d = fe_order * n_q_points_1d;
  return *(this->values + q_point_no + dof * n_q_points_1d + direction * n_values_1d);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_value_face(const int dof,
                                                                      const int face_no,
                                                                      const int direction,
                                                                      const int cell_no) const
{
  return shape_value_face_impl(dof, face_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_value_face_impl(const int dof,
                                                                           const int face_no,
                                                                           const int direction,
                                                                           const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  (void)cell_no;
  constexpr auto n_values_on_faces_1d = fe_order * 2;
  return *(this->values_face + dof + fe_order * face_no + direction * n_values_on_faces_1d);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_gradient(const int dof,
                                                                    const int q_point_no,
                                                                    const int direction,
                                                                    const int cell_no) const
{
  return shape_gradient_impl(dof, q_point_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_gradient_impl(const int dof,
                                                                         const int q_point_no,
                                                                         const int direction,
                                                                         const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(q_point_no, n_q_points_1d);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->gradients[direction] + q_point_no + dof * n_q_points_1d +
           cell_no * n_q_points_1d * fe_order);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_gradient_face(const int dof,
                                                                         const int face_no,
                                                                         const int direction,
                                                                         const int cell_no) const
{
  return shape_gradient_face_impl(dof, face_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::shape_gradient_face_impl(
  const int dof,
  const int face_no,
  const int direction,
  const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  return *(this->gradients_face[direction] + dof + face_no * fe_order +
           cell_no * n_cells_per_direction * fe_order);
}



#include "fd_evaluation.templates.h"

#endif // end inclusion guard
