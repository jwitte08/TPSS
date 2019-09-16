
/*
 * tensors.h: test the functionality of BlockMatrixDiagonal
 *
 *  Created on: May 18, 2019
 *      Author: witte
 */

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"

template<int dim, typename Number = double, int size = -1>
void
reinit_laplace(TensorProductMatrixSymmetricSum<dim, VectorizedArray<Number>, size> & tpmatrix)
{
  // *** initialize 1D Laplace and mass matrix
  Table<2, VectorizedArray<Number>> init_mass(size, size);
  Table<2, VectorizedArray<Number>> init_laplace(size, size);
  for(unsigned int i = 0; i < size; ++i)
  {
    init_mass(i, i) = 2. / 3.;
    if(i > 0)
      init_mass(i, i - 1) = 1. / 6.;
    if(i < size - 1)
      init_mass(i, i + 1) = 1. / 6.;
    init_laplace(i, i) = 2.;
    if(i > 0)
      init_laplace(i, i - 1) = -1.;
    if(i < size - 1)
      init_laplace(i, i + 1) = -1.;
  }

  tpmatrix.reinit(init_mass, init_laplace); // !!!
}

template<typename Number>
Number
l2_error(const ArrayView<const Number> array1, const ArrayView<const Number> array2)
{
  AssertThrow(array1.size() == array2.size(), ExcNotImplemented());
  auto   val2 = array2.cbegin();
  Number error;
  error = 0.;
  for(auto val1 = array1.cbegin(); val1 != array1.cend(); ++val1, ++val2)
    error += (*val1 - *val2) * (*val1 - *val2);
  error = std::sqrt(error);
  return error;
}

int
main() // int argc, char* argv[])
{
  using namespace dealii;

  constexpr unsigned int dim  = 3;
  constexpr unsigned int size = 10;

  // *** init blockmatrix
  using BlockMatrix = typename Tensors::BlockMatrixDiagonal<dim, VectorizedArray<double>, size>;
  using MatrixType  = typename BlockMatrix::matrix_type;
  BlockMatrix bmatrix;
  bmatrix.resize(dim);
  for(std::size_t b = 0; b < dim; ++b)
    reinit_laplace(bmatrix.get_block(b));
  std::cout << "initialized " << bmatrix.m() << "x" << bmatrix.n() << " (block-diagonal) matrix\n";

  // *** init vectors
  const std::size_t   block_vector_size = dealii::Utilities::pow(size, dim);
  std::vector<double> random_input;
  random_input.resize(block_vector_size);
  std::generate(random_input.begin(), random_input.end(), []() {
    return (double)rand() / RAND_MAX;
  });
  const std::size_t                      vector_size = dim * block_vector_size;
  AlignedVector<VectorizedArray<double>> dst;
  dst.resize(vector_size);
  for(std::size_t b = 0; b < dim; ++b)
  {
    const std::size_t stride = b * block_vector_size;
    std::transform(random_input.cbegin(),
                   random_input.cend(),
                   dst.begin() + stride,
                   [](const auto & val) { return make_vectorized_array<double>(val); });
  }
  const auto src{dst};

  {
    std::cout << "test BlockMatrixDiagonal::vmult\n";
    // *** vmult
    const auto src_view = make_array_view<const VectorizedArray<double>>(src.begin(), src.end());
    const auto dst_view = make_array_view<VectorizedArray<double>>(dst.begin(), dst.end());
    bmatrix.vmult(dst_view, src_view);

    // *** compare results
    const ArrayView<const VectorizedArray<double>> dst_first{dst.begin(), block_vector_size};
    for(std::size_t b = 1; b < dim; ++b)
    {
      const std::size_t                              stride = b * block_vector_size;
      const ArrayView<const VectorizedArray<double>> dst_b{dst.begin() + stride, block_vector_size};
      std::cout << "l2 error: " << varray_to_string(l2_error(dst_first, dst_b)) << std::endl;
    }
  }

  {
    std::cout << "test BlockMatrixDiagonal::apply_inverse\n";
    // *** apply inverse
    const auto src_view = make_array_view<const VectorizedArray<double>>(src.begin(), src.end());
    const auto dst_view = make_array_view<VectorizedArray<double>>(dst.begin(), dst.end());
    bmatrix.apply_inverse(dst_view, src_view);

    // *** compare results
    const ArrayView<const VectorizedArray<double>> dst_first{dst.begin(), block_vector_size};
    for(std::size_t b = 1; b < dim; ++b)
    {
      const std::size_t                              stride = b * block_vector_size;
      const ArrayView<const VectorizedArray<double>> dst_b{dst.begin() + stride, block_vector_size};
      std::cout << "l2 error: " << varray_to_string(l2_error(dst_first, dst_b)) << std::endl;
    }
  }

  return 0;
}
