#ifndef FDEVALUATION_H
#define FDEVALUATION_H

#include "TPSS.h"
#include "subdomain_handler.h"

#include <array>
#include <memory>

using namespace dealii;

/**
 * Base evaluation class providing finite element and (surrogate) mapping data
 * regarding the current patch is is associated to. It supports the
 * user written integration kernels with all the data needed to assemble the
 * tensor products of one dimensional subproblems. The obtained Kronecker decompositions
 * may provide the data required in the Fast Diagonalization Method.
 */
// TODO merge Base and Derivation ?
// TODO no need for n_comp template !
template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
class FDEvaluationBase
{
public:
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;
  static constexpr unsigned int fe_order   = fe_degree + 1;

  void
  reinit(const unsigned int patch);

  // TODO replace bool by UpdateFlags ?!
  void
  evaluate(const bool gradients);

  const VectorizedArray<Number> &
  get_JxW(const int qpoint_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  get_h(const int direction, const int cell_no) const;

  VectorizedArray<Number>
  get_normal(const int face_no) const
  {
    return make_vectorized_array<Number>(face_no == 0 ? -1. : 1.);
  }

  Tensor<1, dim, VectorizedArray<Number>>
  get_normal_vector(const int face_no, const int direction) const
  {
    Tensor<1, dim, VectorizedArray<Number>> normal_vector;
    normal_vector *= 0.;
    normal_vector[direction] = get_normal(face_no);
    return normal_vector;
  }

  std::bitset<macro_size>
  get_boundary_mask(const int direction, const int cell_no, const int face_no) const
  {
    AssertIndexRange(direction, dim);
    AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
    AssertIndexRange(face_no, 2);
    if(patch_variant == TPSS::PatchVariant::cell)
      return std::bitset<macro_size>{bdry_mask_id[direction * 2 + face_no]};
    else if(patch_variant == TPSS::PatchVariant::vertex)
    {
      const bool is_interior_face =
        (cell_no == 0 && face_no == 1) || (cell_no == 1 && face_no == 0);
      if(!is_interior_face)
        return std::bitset<macro_size>{bdry_mask_id[direction * 2 + face_no]};
    }
    else
      AssertThrow(false, ExcNotImplemented());
    return std::bitset<macro_size>{0};
  }

  VectorizedArray<Number>
  get_average_factor(const int direction, const int cell_no, const int face_no) const
  {
    auto         factor      = make_vectorized_array<Number>(0.5);
    const auto & at_boundary = get_boundary_mask(direction, cell_no, face_no);
    for(auto lane = 0U; lane < macro_size; ++lane)
      if(at_boundary[lane])
        factor[lane] = 1.;
    return factor;
  }

  const SubdomainHandler<dim, Number> &
  get_subdomain_handler() const
  {
    return sd_handler;
  }

  // constexpr unsigned int
  // n_dofs_per_cell(int direction = -1, int cell_no = -1) const
  // {
  //   // assuming isotropy ... TODO
  //   if(direction == -1 && cell_no == -1)
  //     return fe_order;
  //   AssertIndexRange(direction, dim);
  //   AssertIndexRange(cell_no, n_cells_per_direction);
  //   AssertThrow(false, ExcMessage("TODO..."));
  //   return 0U;
  // }

  const VectorizedArray<Number> &
  shape_value(const int dof, const int qpoint_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  shape_value_face(const int dof, const int face_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  shape_gradient(const int dof, const int qpoint_no, const int direction, const int cell_no) const;

  const VectorizedArray<Number> &
  shape_gradient_face(const int dof,
                      const int face_no,
                      const int direction,
                      const int cell_no) const;

  const ArrayView<VectorizedArray<Number>>
  acquire_scratch_chunk(const std::size_t size)
  {
    auto && chunk =
      make_array_view(this->scratch_pad_remainder, (this->scratch_pad_remainder += size));
    Assert(std::distance(this->scratch_pad_remainder, this->scratch_pad_end) >= 0,
           ExcMessage("Acquired scratch chunk exceeds scratch pad!"));
    return chunk;
  }

  void
  release_scratch_chunks()
  {
    this->scratch_pad_remainder = this->scratch_pad;
  }

  const unsigned int       level;
  const unsigned int       n_subdomains;
  const unsigned int       n_colors;
  const TPSS::PatchVariant patch_variant;
  const unsigned int       n_cells_per_direction;

protected:
  FDEvaluationBase(const SubdomainHandler<dim, Number> & sd_handler,
                   const unsigned int                    dofh_id = 0,
                   const unsigned int                    quad_id = 0);

  FDEvaluationBase(const SubdomainHandler<dim, Number> & sd_handler,
                   const TPSS::PatchVariant              patch_variant,
                   const unsigned int                    dofh_id = 0,
                   const unsigned int                    quad_id = 0);

  ~FDEvaluationBase();

  FDEvaluationBase(const FDEvaluationBase & other) = delete;

  FDEvaluationBase &
  operator=(const FDEvaluationBase & other) = delete;

  static void submit_cell_matrix(Table<2, VectorizedArray<Number>> &       subdomain_matrix,
                                 const Table<2, VectorizedArray<Number>> & cell_matrix,
                                 const unsigned int                        cell_no_row,
                                 const unsigned int                        cell_no_col);

  const SubdomainHandler<dim, Number> &                                     sd_handler;
  const TPSS::MappingInfo<dim, Number> &                                    mapping_info;
  const MatrixFree<dim, Number> &                                           mf_storage;
  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> & shape_info;

  unsigned int patch_id = numbers::invalid_unsigned_int;

  /**
   * lexicographical ordering: face_no_1d < direction
   */
  std::array<unsigned short, GeometryInfo<dim>::faces_per_cell> bdry_mask_id;

  /**
   * quadrature weights on the unit cell
   */
  VectorizedArray<Number> * quad_weights_unit = nullptr;

  /**
   * lexicographical ordering: cell_no_1d < direction
   */
  const VectorizedArray<Number> * h_lengths = nullptr;

  /**
   * lexicographical ordering: quad_index < cell_no_1d < direction
   */
  VectorizedArray<Number> * JxWs = nullptr;

  /**
   * lexicographical ordering: quad_index < dof_index ( < n_components? < direction?)
   */
  VectorizedArray<Number> * values = nullptr;

  /**
   * lexicographical ordering: dof_index < face_no_1d ( < n_components? < direction?)
   */
  VectorizedArray<Number> * values_face = nullptr;

  /**
   * lexicographical ordering: quad_index < dof_index < cell_no
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
  bool weights_filled   = false;
  bool gradients_filled = false;
};

/**
 * new implementation ...
 */
// TODO merge Base and Derivation ?
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
class FDEvaluation : public FDEvaluationBase<dim, fe_degree, n_q_points_1d, 1, Number>
{
public:
  using CellAssembler =
    typename TPSS::MatrixEvaluator<fe_degree + 1, n_q_points_1d, VectorizedArray<Number>>;
  using value_type                              = Number;
  static constexpr unsigned int fe_order        = fe_degree + 1;
  static constexpr unsigned int n_q_points      = n_q_points_1d;
  static constexpr unsigned int n_dofs_per_cell = Utilities::pow(fe_order, dim);

private:
  using Base = FDEvaluationBase<dim, fe_degree, n_q_points_1d, 1, Number>;
  static constexpr unsigned int macro_size = VectorizedArray<Number>::n_array_elements;

public:
  FDEvaluation(const SubdomainHandler<dim, Number> & sd_handler_in,
               const unsigned int                    dofh_index = 0,
               const unsigned int                    quad_index = 0)
    : FDEvaluationBase<dim, fe_degree, n_q_points_1d, 1, Number>(sd_handler_in,
                                                                 dofh_index,
                                                                 quad_index)
  {
    const auto patch_variant = sd_handler_in.get_additional_data().patch_variant;
    if(patch_variant == TPSS::PatchVariant::cell)
      malloc_fedata<TPSS::PatchVariant::cell>();
    else if(patch_variant == TPSS::PatchVariant::vertex)
      malloc_fedata<TPSS::PatchVariant::vertex>();
    else
      AssertThrow(false, ExcNotImplemented());

    // *** extract 1d quadrature weights on reference interval
    const auto & quadrature = this->shape_info.quadrature; // isotropic quadrature assumed
    AssertDimension(static_cast<int>(quadrature.size()), n_q_points_1d);
    auto W = this->quad_weights_unit;
    for(auto in = quadrature.get_weights().cbegin(); in != quadrature.get_weights().cend();
        ++W, ++in)
      *W = make_vectorized_array<Number>(*in);

    // *** extract 1d shape values on cell and faces
    std::copy_n(this->shape_info.shape_values.begin(), fe_order * n_q_points_1d, this->values);
    for(const std::size_t face_no_1d : {0, 1})
    {
      const auto shape_values_face = this->shape_info.shape_data_on_face[face_no_1d].begin();
      const auto this_values_face  = this->values_face + face_no_1d * fe_order;
      std::copy_n(shape_values_face, fe_order, this_values_face);
    }
    this->values_filled = true;
  }

  ~FDEvaluation() = default;

  FDEvaluation(const FDEvaluation & other) = delete;

  FDEvaluation &
  operator=(const FDEvaluation & other) = delete;

  void
  compute_unit_mass(const ArrayView<VectorizedArray<Number>> & matrix) const;

  template<typename CellOperation>
  std::array<Table<2, VectorizedArray<Number>>, dim>
  patch_action(const FDEvaluation & eval_ansatz, CellOperation && cell_operation) const
  {
    const auto patch_variant = Base::patch_variant;
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
    const auto patch_variant = Base::patch_variant;
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
  template<TPSS::PatchVariant variant>
  void
  malloc_fedata()
  {
    Assert(this->scratch_fedata != nullptr, ExcInternalError());

    constexpr unsigned int scratch_pad_length  = fe_order * fe_order * dim * 2;
    constexpr unsigned int unit_weights_length = n_q_points_1d;
    constexpr unsigned int values_length       = n_q_points_1d * fe_order;
    constexpr unsigned int values_length_face  = fe_order * 2;
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
    this->scratch_fedata->resize_fast(size_to_be_allocated);
    this->quad_weights_unit = this->scratch_fedata->begin();
    this->values            = this->quad_weights_unit + unit_weights_length;
    this->values_face       = this->values + values_length;
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
        Base::submit_cell_matrix(matrix, cell_matrix, cell_no, cell_no);
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
      Base::submit_cell_matrix(matrix, cell_matrix, 0 /*cell_no*/, 0 /* cell_no*/);
    }

    AssertDimension(n_dofs_per_direction, matrices.front().n_rows());
    AssertDimension(n_dofs_per_direction, matrices.back().n_cols());
    return matrices;
  }

  // template<typename CellOperation, typename FaceOperation, typename InterfaceOperation>
  // std::array<Table<2, VectorizedArray<Number>>, dim>
  // patch_action_dgvp_impl(const CellOperation &      cell_operation,
  //                        const FaceOperation &      face_operation,
  //                        const InterfaceOperation & interface_operation) const
  // {
  //   return patch_action_dgvp_impl(*this, cell_operation, face_operation, interface_operation);
  // }

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

      Base::submit_cell_matrix(matrix, cell_matrix0, 0, 0);
      Base::submit_cell_matrix(matrix, cell_matrix1, 1, 1);

      // *** INTERFACE between cell 0 (seen as face 1) & cell 1 (seen as face 0) ***/
      MatrixType cell_matrix01{fe_order, fe_order},
        cell_matrix10{fe_order, fe_order}; // first parameter represents the cell_no associated to
                                           // the test functions and second parameter is the
                                           // cell_no associated to the ansatz functions
      interface_operation(eval_ansatz, *this, cell_matrix01, cell_matrix10, direction);
      Base::submit_cell_matrix(matrix, cell_matrix01, 0, 1);
      Base::submit_cell_matrix(matrix, cell_matrix10, 1, 0);
    }

    AssertDimension(n_dofs_per_direction, matrices.front().n_rows());
    AssertDimension(n_dofs_per_direction, matrices.back().n_cols());
    return matrices;
  }
};



// ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++

// ______________________________   FDEvaluationBase   ______________________________



template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::FDEvaluationBase(
  const SubdomainHandler<dim, Number> & sd_handler_in,
  const unsigned int                    dofh_id,
  const unsigned int                    quad_id)
  : level(sd_handler_in.get_additional_data().level),
    n_subdomains(sd_handler_in.get_patch_info().subdomain_partition_data.n_subdomains()),
    n_colors(sd_handler_in.get_patch_info().subdomain_partition_data.n_colors()),
    patch_variant(sd_handler_in.get_additional_data().patch_variant),
    n_cells_per_direction(TPSS::UniversalInfo<dim>::n_cells_per_direction(patch_variant)),
    sd_handler(sd_handler_in),
    mapping_info(sd_handler.get_mapping_info()),
    mf_storage(sd_handler.get_matrix_free()),
    shape_info(mf_storage.get_shape_info(dofh_id, quad_id)),
    scratch_fedata(mf_storage.acquire_scratch_data())
{
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::~FDEvaluationBase()
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
  quad_weights_unit     = nullptr;
  scratch_pad           = nullptr;
  scratch_pad_remainder = nullptr;
  scratch_pad_end       = nullptr;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline void
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::reinit(const unsigned int patch)
{
  AssertIndexRange(patch, n_subdomains);
  patch_id                = patch;
  const auto & patch_info = sd_handler.get_patch_info();
  std::copy_n(patch_info.at_boundary_mask.data() + GeometryInfo<dim>::faces_per_cell * patch,
              GeometryInfo<dim>::faces_per_cell,
              bdry_mask_id.begin());

  h_lengths = mapping_info.template h_lengths_begin(patch);

  gradients_filled = false;
  evaluate(true);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline const VectorizedArray<Number> &
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::get_JxW(const int qpoint_no,
                                                                         const int direction,
                                                                         const int cell_no) const
{
  AssertIndexRange(qpoint_no, n_q_points_1d);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->JxWs + qpoint_no + cell_no * n_q_points_1d +
           direction * n_q_points_1d * n_cells_per_direction);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline const VectorizedArray<Number> &
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::get_h(const int direction,
                                                                       const int cell_no) const
{
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->h_lengths + cell_no + direction * n_cells_per_direction);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline const VectorizedArray<Number> &
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::shape_value(
  const int dof,
  const int qpoint_no,
  const int direction,
  const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(qpoint_no, n_q_points_1d);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  (void)cell_no;
  (void)direction; // only isotropic elements implemented
  return *(this->values + qpoint_no + dof * n_q_points_1d);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline const VectorizedArray<Number> &
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::shape_value_face(
  const int dof,
  const int face_no,
  const int direction,
  const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  AssertIndexRange(face_no, 2);
  (void)cell_no;
  (void)direction; // only isotropic elements implemented
  return *(this->values_face + dof + fe_order * face_no);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline const VectorizedArray<Number> &
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::shape_gradient(
  const int dof,
  const int qpoint_no,
  const int direction,
  const int cell_no) const
{
  AssertIndexRange(dof, static_cast<int>(fe_order));
  AssertIndexRange(qpoint_no, n_q_points_1d);
  AssertIndexRange(cell_no, static_cast<int>(n_cells_per_direction));
  AssertIndexRange(direction, dim);
  return *(this->gradients[direction] + qpoint_no + dof * n_q_points_1d +
           cell_no * n_q_points_1d * fe_order);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
inline const VectorizedArray<Number> &
FDEvaluationBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::shape_gradient_face(
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

// ______________________________   FDEvaluation   ______________________________

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline void
FDEvaluation<dim, fe_degree, n_q_points_1d, Number>::compute_unit_mass(
  const ArrayView<VectorizedArray<Number>> & matrix) const
{
  AssertDimension(matrix.size(), fe_order * fe_order);
  CellAssembler::template assemble<false>(
    make_array_view(this->values, this->values + fe_order * n_q_points_1d),
    make_array_view(this->values, this->values + fe_order * n_q_points_1d),
    make_array_view(this->quad_weights_unit, this->quad_weights_unit + n_q_points_1d),
    matrix);
}

#include "fd_evaluation.templates.h"

#endif // end inclusion guard
