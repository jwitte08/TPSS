/*
 * matrix_helper.h
 *
 *  Created on: Feb 17, 2020
 *      Author: witte
 */

#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_



#include <deal.II/base/aligned_vector.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>


#include "alignedlinalg.h"
#include "generic_functionalities.h"
#include "tensors.h"
#include "vectorization.h"

using namespace dealii;



/**
 * A wrapper class representing an inverse matrix based on dealii::Table type,
 * thus allowing vectorized arithmetic type @p Number.
 *
 * Internally, each lane is inverted by means of LAPACKFullMatrix's inverse
 * singular value decomposition.
 */
template<typename Number>
struct InverseTable
{
  using value_type        = Number;
  using scalar_value_type = typename ExtractScalarType<Number>::type;

  static constexpr unsigned int macro_size = get_macro_size<Number>();

  struct AdditionalData
  {
    unsigned int kernel_size = numbers::invalid_unsigned_int;
    /// all singular values s_i with s_i/s_max < threshold are set to zero
    double threshold = 0.;
  };

  InverseTable() = default;

  InverseTable(const Table<2, Number> & matrix_in,
               const AdditionalData &   additional_data_in = AdditionalData{});

  void
  reinit(const Table<2, Number> & matrix_in, const AdditionalData & additional_data_in);

  void
  clear();

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_impl(const ArrayView<scalar_value_type> &       dst_view,
             const ArrayView<const scalar_value_type> & src_view,
             const unsigned int                         lane = 0) const;

  void
  vmult_impl(const ArrayView<VectorizedArray<scalar_value_type>> &       dst_view,
             const ArrayView<const VectorizedArray<scalar_value_type>> & src_view) const;

  std::shared_ptr<const std::array<LAPACKFullMatrix<scalar_value_type>, macro_size>> inverses;
  /// ALTERNATIVE: FullMatrix
  // std::shared_ptr<const std::array<FullMatrix<scalar_value_type>, macro_size>> inverses;
};



/**
 * A matrix wrapper class around a dealii::Table class providing a basic feature
 * set one expects from a matrix. In particular, this class enables vectorized
 * arithmetic type @p Number in contrast to dealii::FullMatrix or
 * dealii::LAPACKFullMatrix.
 */
template<typename Number>
class MatrixAsTable
{
public:
  using value_type  = Number;
  using matrix_type = Table<2, Number>;

  static constexpr unsigned int macro_size = get_macro_size<Number>();

  MatrixAsTable() = default;

  ~MatrixAsTable() = default;

  MatrixAsTable(const MatrixAsTable & other);

  MatrixAsTable(MatrixAsTable && other);

  MatrixAsTable &
  operator=(const MatrixAsTable & other);

  MatrixAsTable &
  operator=(MatrixAsTable && other);

  MatrixAsTable &
  operator=(const Table<2, Number> & other);

  MatrixAsTable &
  operator=(Table<2, Number> && other);

  operator const Table<2, Number> &() const;

  operator Table<2, Number> &();

  unsigned int
  m() const;

  unsigned int
  n() const;

  /// TODO private?
  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> &       dst_view,
             const Number &                  factor,
             const ArrayView<const Number> & src_view) const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  void
  vmult_sadd(const ArrayView<Number> &       dst_view,
             const Number &                  scalar,
             const ArrayView<const Number> & src_view) const;

  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult_add(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  /**
   * TODO...
   */
  void
  shuffle(const std::array<std::size_t, 2> & size_of_blockmatrix,
          const std::array<std::size_t, 2> & size_of_blocks);

  /**
   * Compute the inverse of the current matrix. The inverse matrix type is
   * InverseTable and more details can be found there.
   */
  void
  invert(const typename InverseTable<Number>::AdditionalData & additional_data =
           typename InverseTable<Number>::AdditionalData{});

  /**
   * Note that invert() has to be explicitly called before to compute the
   * inverse of the underlying matrix. For efficiency reasons, the inverse is
   * not computed during initialization of this class.
   */
  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  /**
   * Read & write access to the underlying Table storing the matrix entries.
   */
  Table<2, Number> &
  as_table();

  /**
   * Read access to the underlying Table storing the matrix entries.
   */
  const Table<2, Number> &
  as_table() const;

  /**
   * Return the inverse matrix as Table by applying apply_inverse() to all
   * canonical vectors.
   */
  Table<2, Number>
  as_inverse_table() const;

  /**
   * Fills the rectangular submatrix of this matrix with first row index @p
   * row_start and @p column_start with the values of rectangular matrix @p
   * other. The "rectangle size" is determined by the size of @p other.
   *
   * If the template parameter @p fill_transpose is true we fill the submatrix
   * with the values of the transpose of @p other.
   */
  template<bool fill_transpose = false>
  void
  fill_submatrix(const Table<2, Number> & other,
                 const unsigned int       row_start,
                 const unsigned int       column_start);

  /**
   * Same as above except that value types might vary and we read from or write
   * to one specific vectorization lane @p lane. If Number or OtherNumber
   * defines a scalar value type the vectorization lane is suppressed by
   * scalar_value(), consequently, accessing only the scalar value independent
   * of the index @p lane.
   */
  template<typename OtherNumber = Number, bool fill_transpose = false>
  void
  fill_submatrix(const Table<2, OtherNumber> & other,
                 const unsigned int            row_start,
                 const unsigned int            column_start,
                 const unsigned int            lane);

  /**
   * Same as above but adding the matrix block @p other to this matrix.
   */
  template<typename OtherNumber = Number, bool fill_transpose = false>
  void
  add_submatrix(const Table<2, OtherNumber> & other,
                const unsigned int            row_start,
                const unsigned int            column_start,
                const unsigned int            lane);

  std::shared_ptr<const InverseTable<Number>>
  share_basic_inverse() const
  {
    return inverse_matrix;
  }

private:
  Table<2, Number> matrix;

  std::shared_ptr<InverseTable<Number>> inverse_matrix;

  template<typename OtherNumber, bool fill_transpose, bool add>
  void
  fill_or_add_submatrix_impl(const Table<2, OtherNumber> & other,
                             const unsigned int            row_start,
                             const unsigned int            column_start,
                             const unsigned int            lane);
};



/**
 * TODO...
 */
template<typename Number>
class DiagonalMatrixWrap
{
public:
  using value_type = Number;

  static constexpr unsigned int macro_size = get_macro_size<Number>();

  DiagonalMatrixWrap();

  DiagonalMatrixWrap(const AlignedVector<Number> &    diagonal_in,
                     const std::array<std::size_t, 2> size_in = {static_cast<std::size_t>(-1),
                                                                 static_cast<std::size_t>(-1)});

  void
  reinit(const AlignedVector<Number> &    diagonal_in,
         const std::array<std::size_t, 2> size_in = {static_cast<std::size_t>(-1),
                                                     static_cast<std::size_t>(-1)});

  void
  clear();

  unsigned int
  m() const;

  unsigned int
  n() const;

  template<bool add, bool transpose>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  Tvmult_add(AlignedVector<Number> & dst, const AlignedVector<Number> & src) const;

  /**
   * TODO...
   */
  void
  shuffle(const std::array<std::size_t, 2> & size_of_blockmatrix_in,
          const std::array<std::size_t, 2> & size_of_blocks_in);

  const AlignedVector<Number> &
  get_diagonal() const;

  /**
   * Return the current diagonal matrix as table. Note that after a call to
   * shuffle() this matrix is no longer in diagonal form.
   */
  Table<2, Number>
  as_table() const;

private:
  std::array<std::size_t, 2> size;
  AlignedVector<Number>      diagonal;
  std::array<std::size_t, 2> size_of_blockmatrix;
  std::array<std::size_t, 2> size_of_blocks;
  bool                       is_shuffled;
};



/**
 * TODO...
 */
template<typename PatchOperator>
struct PatchMatrix
{
  using scalar_value_type = typename PatchOperator::value_type;
  using value_type        = VectorizedArray<scalar_value_type>;

  static constexpr unsigned int macro_size = VectorizedArray<scalar_value_type>::size();

  void
  reinit(const PatchOperator * patch_operator_in, const unsigned int patch_id);

  unsigned int
  m() const;

  unsigned int
  n() const;

  void
  vmult(const ArrayView<value_type> & dst_view, const ArrayView<const value_type> & src_view) const;

  void
  apply_inverse(const ArrayView<value_type> &       dst_view,
                const ArrayView<const value_type> & src_view) const;

  Table<2, value_type>
  as_table() const;

  FullMatrix<scalar_value_type>
  as_fullmatrix(const unsigned int lane);

  const PatchOperator * patch_operator   = nullptr;
  unsigned int          current_patch_id = numbers::invalid_unsigned_int;
};



/**
 * Matrix wrapper that treats constraints arising from strong homogeneous
 * Dirichlet boundary conditions on a local index set.
 */
template<typename MatrixType, typename Number = typename MatrixType::value_type>
struct ConstrainedMatrix
{
  using value_type        = Number;
  using scalar_value_type = typename ExtractScalarType<Number>::type;

  static constexpr unsigned int macro_size = get_macro_size<Number>();

  ConstrainedMatrix() = default;

  ~ConstrainedMatrix() = default;

  ConstrainedMatrix &
  operator=(const ConstrainedMatrix & other) = delete;

  unsigned int
  m() const;

  unsigned int
  n() const;

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const;

  Table<2, Number>
  as_table() const;

  Table<2, Number>
  as_inverse_table() const;

  void
  zero_out_constrained_dofs_row(const ArrayView<Number> & dst_view) const;

  void
  zero_out_constrained_dofs_col(const ArrayView<Number> & dst_view) const;

  template<bool is_row>
  void
  zero_out_constrained_dofs_impl(const ArrayView<Number> & dst_view) const;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;

  MatrixType                                     matrix;
  std::array<std::set<unsigned int>, macro_size> constrained_dof_indices_row;
  std::array<std::set<unsigned int>, macro_size> constrained_dof_indices_col;
  scalar_value_type                              diagonal_factor = 1.;
};

#include "matrix_helper.templates.h"

#endif // MATRIX_HELPER_H_
