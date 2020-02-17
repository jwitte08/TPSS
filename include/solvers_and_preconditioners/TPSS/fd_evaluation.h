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

template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
class FDEvaluation
{
public:
  using CellAssembler =
    typename TPSS::MatrixEvaluator<fe_degree + 1, n_q_points_1d_, VectorizedArray<Number>>;
  using value_type                         = Number;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

  static constexpr unsigned int fe_order               = fe_degree + 1;
  static constexpr unsigned int n_q_points_1d_static   = n_q_points_1d_;
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

  VectorizedArray<Number>
  get_average_factor(const int direction, const int cell_no, const int face_no) const;

  std::bitset<macro_size>
  get_boundary_mask(const int direction, const int cell_no, const int face_no) const;

  std::set<unsigned int>
  get_constrained_dof_indices(const unsigned int lane) const;

  TPSS::DoFLayout
  get_dof_layout() const;

  const TPSS::PatchLocalTensorHelper<dim> &
  get_dof_tensor() const;

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

  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
  get_shape_info(const unsigned int dimension) const;

  const SubdomainHandler<dim, Number> &
  get_subdomain_handler() const;

  unsigned int
  n_cells_1d(const unsigned int dimension) const;

  unsigned int
  n_dofs_1d(const unsigned int dimension) const;

  unsigned int
  n_dofs_per_cell_1d(const unsigned int dimension) const;

  unsigned int
  n_q_points_1d(const unsigned int dimension) const;

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

  void
  compute_unit_mass(const ArrayView<VectorizedArray<Number>> & matrix) const;

  template<int fe_degree_ansatz, int n_q_points_ansatz, typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
               CellOperation && cell_operation) const;

  template<typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(CellOperation && cell_operation) const;


  template<int fe_degree_ansatz,
           int n_q_points_ansatz,
           typename CellOperation,
           typename FaceOperation,
           typename InterfaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
               const CellOperation &      cell_operation,
               const FaceOperation &      face_operation,
               const InterfaceOperation & interface_operation) const;

  template<typename CellOperation, typename FaceOperation, typename InterfaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const CellOperation &      cell_operation,
               const FaceOperation &      face_operation,
               const InterfaceOperation & interface_operation) const;

private:
  using This = FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>;

  template<TPSS::PatchVariant variant>
  void
  malloc_fedata();

  VectorizedArray<Number> &
  get_JxW_impl(const int q_point_no, const int direction, const int cell_no) const;

  VectorizedArray<Number> &
  get_q_weight_impl(const int q_point_no, const int direction) const;

  void
  evaluate_gradients();

  template<int fe_degree_ansatz, int n_q_points_ansatz>
  void post_process_constraints(
    std::array<Table<2, VectorizedArray<Number>>, dim> &                   matrices,
    const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz) const;

  void submit_constraints(Table<2, VectorizedArray<Number>> & subdomain_matrix,
                          const std::set<unsigned int> &      constrained_dof_indices_row,
                          const std::set<unsigned int> &      constrained_dof_indices_col,
                          const bool                          at_block_diagonal) const
  {
    /// Clear rows and columns of constrained dof indices. If this matrix is
    /// part of the block diagonal of the local patch matrix (in case of
    /// vector-valued problems) the diagonal entry is set to one such that @p
    /// subdomain_matrix is non-singular.
    for(const auto dof_index_test : constrained_dof_indices_row)
    {
      AssertIndexRange(dof_index_test, subdomain_matrix.n_rows());
      for(auto j = 0U; j < subdomain_matrix.n_cols(); ++j)
        subdomain_matrix(dof_index_test, j) = 0.;
      if(at_block_diagonal)
      {
        AssertIndexRange(dof_index_test, subdomain_matrix.n_cols());
        subdomain_matrix(dof_index_test, dof_index_test) = 1.;
      }
    }
    for(const auto dof_index_ansatz : constrained_dof_indices_col)
    {
      AssertIndexRange(dof_index_ansatz, subdomain_matrix.n_cols());
      for(auto i = 0U; i < subdomain_matrix.n_rows(); ++i)
        subdomain_matrix(i, dof_index_ansatz) = 0.;
      if(at_block_diagonal)
      {
        AssertIndexRange(dof_index_ansatz, subdomain_matrix.n_rows());
        subdomain_matrix(dof_index_ansatz, dof_index_ansatz) = 1.;
      }
    }
  }

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

  template<int fe_degree_ansatz, int n_q_points_ansatz, typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action_impl(
    const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
    CellOperation &&                                                       cell_operation) const;

  template<int fe_degree_ansatz,
           int n_q_points_ansatz,
           typename CellOperation,
           typename FaceOperation,
           typename InterfaceOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action_impl(
    const FDEvaluation<dim, fe_degree_ansatz, n_q_points_ansatz, Number> & eval_ansatz,
    CellOperation &&                                                       cell_operation,
    FaceOperation &&                                                       face_operation,
    InterfaceOperation && interface_operation) const;

  void submit_cell_matrix(
    Table<2, VectorizedArray<Number>> &       subdomain_matrix,
    const Table<2, VectorizedArray<Number>> & cell_matrix,
    const unsigned int                        cell_no_row,
    const unsigned int                        cell_no_col,
    const unsigned int                        dimension               = 0,
    const TPSS::PatchLocalTensorHelper<dim> * patch_dof_tensor_ansatz = nullptr) const;

  const SubdomainHandler<dim, Number> & sd_handler;

  const TPSS::PatchDoFWorker<dim, Number> patch_worker;

  const TPSS::MappingInfo<dim, Number> & mapping_info;

  const MatrixFree<dim, Number> & mf_storage;

  const unsigned int       component;
  const unsigned int       level;
  const unsigned int       n_subdomains;
  const unsigned int       n_colors;
  const TPSS::PatchVariant patch_variant;
  const unsigned int       n_cells_per_direction;
  unsigned int             patch_id = numbers::invalid_unsigned_int;

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
   * Stores for each component (array position) of d-dimensional shape function
   * gradient the univariate shape function gradients (i.e. first derivatives)
   * of coinciding spatial coordinate @p direction in a flat data field with
   * lexicographical ordering:
   *
   *   q_point_index < dof_index < cell_no
   *
   * Evaluations at univariate quadrature points are in real space subject to
   * Cartesian mapping. For illustration, let
   *
   *   PHI(x_1,x_2,x_3) = phi_1(x_1) phi_2(x_2) phi_3(x_3)
   *
   * define a tensor product shape function, then, @p gradients[i] stores the
   * first derivatives of d/dx_i phi_i(x_i) in real space.
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



template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::FDEvaluation(
  const SubdomainHandler<dim, Number> & sd_handler_in,
  const unsigned int                    dofh_index)
  : sd_handler(sd_handler_in),
    patch_worker(sd_handler_in.get_dof_info(dofh_index)),
    mapping_info(sd_handler_in.get_mapping_info()),
    mf_storage(sd_handler_in.get_matrix_free()),
    component(dofh_index),
    level(sd_handler_in.get_additional_data().level),
    n_subdomains(sd_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(sd_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    patch_variant(sd_handler_in.get_additional_data().patch_variant),
    n_cells_per_direction(TPSS::UniversalInfo<dim>::n_cells_per_direction(patch_variant)),
    scratch_fedata(mf_storage.acquire_scratch_data())
{
  AssertIndexRange(dofh_index, sd_handler.n_components());
  for(auto d = 0U; d < dim; ++d)
  {
    /// static variables fe_order and n_q_points_1d_static have to define the maximum
    /// number for any dimension
    AssertIndexRange(patch_worker.get_dof_tensor().n_dofs_per_cell_1d(d), fe_order + 1);
    AssertIndexRange(patch_worker.get_shape_info(d).quadrature.size(), n_q_points_1d_static + 1);
    /// currently assume isotropy ... TODO
    AssertDimension(n_dofs_per_cell_1d(d), fe_order);
    AssertDimension(n_q_points_1d(d), n_q_points_1d_static);
    AssertDimension(n_cells_1d(d), n_cells_per_direction);
  }

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
    AssertIndexRange(static_cast<int>(quadrature.size()), n_q_points_1d_static + 1);
    const auto & unit_weights = quadrature.get_weights();
    auto         W            = &(get_q_weight_impl(0, d));
    std::copy(unit_weights.cbegin(), unit_weights.cend(), W);

    /// Extract one-dimensional shape function values on cell and face quadratures
    auto values_in_begin = shape_info.shape_values.begin();
    auto values_begin    = &(shape_value_impl(0, 0, d, 0));
    std::copy_n(values_in_begin, fe_order * n_q_points_1d_static, values_begin);
    for(const auto face_no : {0, 1})
    {
      auto values_face_in_begin = shape_info.shape_data_on_face[face_no].begin();
      auto values_face_begin    = &(shape_value_face_impl(0, face_no, d, 0));
      std::copy_n(values_face_in_begin, fe_order, values_face_begin);
    }
  }
  this->values_filled = true;
}

template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::~FDEvaluation()
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

template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::reinit(const unsigned int patch)
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

template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
template<TPSS::PatchVariant variant>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::malloc_fedata()
{
  Assert(this->scratch_fedata != nullptr, ExcInternalError());

  constexpr unsigned int scratch_pad_length  = fe_order * fe_order * dim * 2;
  constexpr unsigned int unit_weights_length = n_q_points_1d_static * dim;
  constexpr unsigned int values_length       = n_q_points_1d_static * fe_order * dim;
  constexpr unsigned int values_length_face  = fe_order * 2 * dim;
  constexpr unsigned int n_cells_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(variant);
  constexpr unsigned int gradients_length =
    n_q_points_1d_static * fe_order * n_cells_per_direction * dim;
  constexpr unsigned int gradients_length_face = fe_order * 2 * n_cells_per_direction * dim;
  constexpr unsigned int JxWs_length           = n_q_points_1d_static * n_cells_per_direction * dim;
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
                         n_q_points_1d_static * fe_order * n_cells_per_direction * d;
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


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const ArrayView<VectorizedArray<Number>>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::acquire_scratch_chunk(const std::size_t size)
{
  auto && chunk =
    make_array_view(this->scratch_pad_remainder, (this->scratch_pad_remainder += size));
  Assert(std::distance(this->scratch_pad_remainder, this->scratch_pad_end) >= 0,
         ExcMessage("Acquired scratch chunk exceeds scratch pad!"));
  return chunk;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::compute_unit_mass(
  const ArrayView<VectorizedArray<Number>> & matrix) const
{
  AssertDimension(matrix.size(), fe_order * fe_order);
  CellAssembler::template assemble<false>(
    make_array_view(this->values, this->values + fe_order * n_q_points_1d_static),
    make_array_view(this->values, this->values + fe_order * n_q_points_1d_static),
    make_array_view(this->q_weights_unit, this->q_weights_unit + n_q_points_1d_static),
    matrix);
}



template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::evaluate(const bool do_gradients)
{
  /// univariate Jacobian, that is h_d, times quadrature weight
  const VectorizedArray<Number> * weight = this->q_weights_unit;
  for(unsigned int d = 0; d < dim; ++d)
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      const auto h = get_h(d, cell_no);
      for(unsigned int q = 0; q < n_q_points_1d_static; ++q)
        get_JxW_impl(q, d, cell_no) = h * weight[q]; // JxW
    }

  if(do_gradients)
    evaluate_gradients();
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::evaluate_gradients()
{
  gradients_filled = false;

  /// scale univariate reference gradients with h_d^{-1}
  for(unsigned int d = 0; d < dim; ++d)
  {
    const auto & shape_info         = get_shape_info(d);
    const auto   n_q_points_1d      = this->n_q_points_1d(d);
    const auto   n_dofs_per_cell_1d = this->n_dofs_per_cell_1d(d);
    const auto * unit_grads_begin   = shape_info.shape_gradients.begin();
    for(unsigned int cell_no = 0; cell_no < n_cells_per_direction; ++cell_no)
    {
      const auto h_inv = 1. / get_h(d, cell_no);
      for(unsigned int dof = 0; dof < n_dofs_per_cell_1d; ++dof)
      {
        const auto * unit_grad_begin = unit_grads_begin + dof * n_q_points_1d;
        auto *       grad            = &(shape_gradient_impl(dof, 0, d, cell_no));
        std::transform(unit_grad_begin,
                       unit_grad_begin + n_q_points_1d,
                       grad,
                       [h_inv](const auto & unit_grad) { return unit_grad * h_inv; });
      }

      for(const int face_no : {0, 1})
      {
        const auto * unit_grads_on_face = shape_info.shape_data_on_face[face_no].begin() + fe_order;
        auto *       grad_on_face       = &(shape_gradient_face_impl(0, face_no, d, cell_no));
        std::transform(unit_grads_on_face,
                       unit_grads_on_face + fe_order,
                       grad_on_face,
                       [h_inv](const auto & unit_grad) { return unit_grad * h_inv; });
      }
    }
  }

  gradients_filled = true;
}



template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_average_factor(const int direction,
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



template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline std::bitset<FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::macro_size>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_boundary_mask(const int direction,
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


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline TPSS::DoFLayout
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_dof_layout() const
{
  return patch_worker.get_dof_info().get_dof_layout();
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const TPSS::PatchLocalTensorHelper<dim> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_dof_tensor() const
{
  return patch_worker.get_dof_tensor();
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_h(const int direction,
                                                            const int cell_no) const
{
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->h_lengths + cell_no + direction * n_cells_per_direction);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_JxW(const int q_point_no,
                                                              const int direction,
                                                              const int cell_no) const
{
  return get_JxW_impl(q_point_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_JxW_impl(const int q_point_no,
                                                                   const int direction,
                                                                   const int cell_no) const
{
  AssertIndexRange(q_point_no, n_q_points_1d(direction));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->JxWs + q_point_no + cell_no * n_q_points_1d_static +
           direction * n_q_points_1d_static * n_cells_per_direction);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_q_weight(const int q_point_no,
                                                                   const int direction) const
{
  return get_q_weight_impl(q_point_no, direction);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_q_weight_impl(const int q_point_no,
                                                                        const int direction) const
{
  AssertIndexRange(q_point_no, n_q_points_1d(direction));
  AssertIndexRange(direction, dim);
  return *(this->q_weights_unit + q_point_no + direction * n_q_points_1d_static);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_normal(const int face_no) const
{
  return make_vectorized_array<Number>(face_no == 0 ? -1. : 1.);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline Tensor<1, dim, VectorizedArray<Number>>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_normal_vector(const int face_no,
                                                                        const int direction) const
{
  Tensor<1, dim, VectorizedArray<Number>> normal_vector;
  normal_vector *= 0.;
  normal_vector[direction] = get_normal(face_no);
  return normal_vector;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline std::set<unsigned int>
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_constrained_dof_indices(
  const unsigned int lane) const
{
  return patch_worker.get_constrained_local_dof_indices(patch_id, lane);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_shape_info(
  const unsigned int dimension) const
{
  return patch_worker.get_shape_info(dimension);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const SubdomainHandler<dim, Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::get_subdomain_handler() const
{
  return sd_handler;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline unsigned int
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::n_cells_1d(const unsigned int dimension) const
{
  return patch_worker.get_dof_tensor().n_cells_1d(dimension);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline unsigned int
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::n_dofs_1d(const unsigned int dimension) const
{
  return patch_worker.get_dof_tensor().n_dofs_1d(dimension);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline unsigned int
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::n_dofs_per_cell_1d(
  const unsigned int dimension) const
{
  return patch_worker.get_dof_tensor().n_dofs_per_cell_1d(dimension);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline unsigned int
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::n_q_points_1d(
  const unsigned int dimension) const
{
  return get_shape_info(dimension).quadrature.size();
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::release_scratch_chunks()
{
  this->scratch_pad_remainder = this->scratch_pad;
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_value(const int dof,
                                                                  const int q_point_no,
                                                                  const int direction,
                                                                  const int cell_no) const
{
  return shape_value_impl(dof, q_point_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_value_impl(const int dof,
                                                                       const int q_point_no,
                                                                       const int direction,
                                                                       const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(n_dofs_per_cell_1d(direction)));
  AssertIndexRange(q_point_no, n_q_points_1d(direction));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  (void)cell_no;
  constexpr auto n_values_1d = fe_order * n_q_points_1d_static;
  return *(this->values + q_point_no + dof * n_q_points_1d_static + direction * n_values_1d);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_value_face(const int dof,
                                                                       const int face_no,
                                                                       const int direction,
                                                                       const int cell_no) const
{
  return shape_value_face_impl(dof, face_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_value_face_impl(const int dof,
                                                                            const int face_no,
                                                                            const int direction,
                                                                            const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(n_dofs_per_cell_1d(direction)));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  (void)cell_no;
  constexpr auto n_values_on_faces_1d = fe_order * 2;
  return *(this->values_face + dof + fe_order * face_no + direction * n_values_on_faces_1d);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_gradient(const int dof,
                                                                     const int q_point_no,
                                                                     const int direction,
                                                                     const int cell_no) const
{
  return shape_gradient_impl(dof, q_point_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_gradient_impl(const int dof,
                                                                          const int q_point_no,
                                                                          const int direction,
                                                                          const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(n_dofs_per_cell_1d(direction)));
  AssertIndexRange(q_point_no, n_q_points_1d(direction));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  constexpr auto n_dofs_per_cell_1d_static = fe_order * n_q_points_1d_static;
  return *(this->gradients[direction] + q_point_no + dof * n_q_points_1d_static +
           cell_no * n_dofs_per_cell_1d_static);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline const VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_gradient_face(const int dof,
                                                                          const int face_no,
                                                                          const int direction,
                                                                          const int cell_no) const
{
  return shape_gradient_face_impl(dof, face_no, direction, cell_no);
}


template<int dim, int fe_degree, int n_q_points_1d_, typename Number>
inline VectorizedArray<Number> &
FDEvaluation<dim, fe_degree, n_q_points_1d_, Number>::shape_gradient_face_impl(
  const int dof,
  const int face_no,
  const int direction,
  const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(n_dofs_per_cell_1d(direction)));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  return *(this->gradients_face[direction] + dof + face_no * fe_order +
           cell_no * n_cells_per_direction * fe_order);
}



#include "fd_evaluation.templates.h"

#endif // end inclusion guard
