#ifndef ALIGNEDLINALG_H_
#define ALIGNEDLINALG_H_

#include <deal.II/base/table.h>
#include <deal.II/lac/lapack_full_matrix.h>
using namespace dealii;

// Calculate inner product of two AlignedVectors
template<typename Number>
Number
inner_product(const AlignedVector<Number> & in1, const AlignedVector<Number> & in2)
{
	AssertDimension(in1.size(), in2.size());
	Number ret = Number(0);
	for(std::size_t i = 0; i < in1.size(); i++)
		ret += in1[i] * in2[i];
	return ret;
}


// Add two Tables
template<typename Number>
Table<2,Number>
matrix_addition(const Table<2,Number> & in1, const Table<2,Number> & in2)
{
	AssertDimension(in1.size()[0], in2.size()[0]);
	AssertDimension(in1.size()[1], in2.size()[1]);
	Table<2,Number> ret = Table<2,Number>(in1);
	for(std::size_t i = 0; i < in1.size()[0]; i++)
		for(std::size_t j = 0; j < in1.size()[1]; j++)
			ret(i,j) = in1(i,j) + in2(i,j);
	return ret;
}
// Add two AlignedVectors
template<typename Number>
AlignedVector<Number>
vector_addition(const AlignedVector<Number> & in1, const AlignedVector<Number> & in2)
{
	AssertDimension(in1.size(), in2.size());
	AlignedVector<Number> ret = AlignedVector<Number>(in1);
	for(std::size_t i = 0; i < in1.size(); i++)
		ret[i] = in1[i] + in2[i];
	return ret;
}


// Multiply AlignedVector with scalar
template<typename Number>
AlignedVector<Number>
vector_scaling(const AlignedVector<Number> & in, const Number & scalar)
{
	AlignedVector<Number> ret = AlignedVector<Number>(in);
	for(std::size_t i = 0; i < in.size(); i++)
		ret[i] = in[i] * scalar;
	return ret;
}


/*
  Divide AlignedVector by scalar, if vector is zero allow scalar to be zero.
  Alowing this makes it possible to simultaneously execute a Lanczos algorithm
  on matrices with different Kronecker rank. We check for zero, so for
  vectorizedarrays we need to do it comonent wise
*/
template<typename Number>
AlignedVector<VectorizedArray<Number>>
vector_inverse_scaling(const AlignedVector<VectorizedArray<Number>> & in,
                       const VectorizedArray<Number> &                scalar)
{
	AlignedVector<VectorizedArray<Number>> ret(in.size());

	constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
	for(std::size_t lane = 0; lane < macro_size; lane++)
	{
		for(std::size_t i = 0; i < in.size(); i++)
		{
			if(std::abs(scalar[lane]) >=
			   std::numeric_limits<Number>::epsilon() * std::abs(scalar[lane] + in[i][lane]))
				ret[i][lane] = in[i][lane] / scalar[lane];
			else
			{
				ret[i][lane] = 0;
			}
		}
	}
	return ret;
}


// Divide AlignedVector by scalar, if vector is zero allow scalar to be zero
template<typename Number>
AlignedVector<Number>
vector_inverse_scaling(const AlignedVector<Number> & in, const Number & scalar)
{
	AlignedVector<Number> ret(in.size());
	for(std::size_t i = 0; i < in.size(); i++)
	{
		if(std::abs(scalar) >= std::numeric_limits<Number>::epsilon() * std::abs(scalar + in[i]))
			ret[i] = in[i] / scalar;
		else
		{
			ret[i] = 0;
		}
	}
	return ret;
}


// Multiply Matrix by Matrix
template<typename Number>
Table<2, Number>
matrix_multiplication(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
	AssertDimension(in1.size()[1], in2.size()[0]);
	Table<2, Number> ret(in1.size()[0], in2.size()[1]);
	for(std::size_t i = 0; i < in1.size()[0]; i++)
		for(std::size_t j = 0; j < in2.size()[1]; j++)
			for(std::size_t k = 0; k < in2.size()[0]; k++)
				ret(i, j) += in1(i, k) * in2(k, j);
	return ret;
}


// Multiply transpose of Matrix by Matrix
template<typename Number>
Table<2, Number>
matrix_transpose_multiplication(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
	AssertDimension(in1.size()[0], in2.size()[0]);
	Table<2, Number> ret(in1.size()[1], in2.size()[1]);
	for(std::size_t i = 0; i < in1.size()[1]; i++)
		for(std::size_t j = 0; j < in2.size()[1]; j++)
			for(std::size_t k = 0; k < in2.size()[0]; k++)
				ret(i, j) += in1(k, i) * in2(k, j);
	return ret;
}


// Flatten Table to AlignedVector
template<typename Number>
AlignedVector<Number>
vectorize_matrix(const Table<2, Number> & tab)
{
	std::size_t           m = tab.size()[0];
	std::size_t           n = tab.size()[1];
	AlignedVector<Number> ret(m * n);
	for(std::size_t k = 0; k < m; k++)
		for(std::size_t l = 0; l < n; l++)
			ret[k * n + l] = tab(k, l);
	return ret;
}


// For vectorizedarray check if > holds for all elements
template<typename Number>
bool
operator>(VectorizedArray<Number> a, VectorizedArray<Number> b)
{
	constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
	for(std::size_t lane = 0; lane < macro_size; lane++)
		if(a[lane] <= b[lane])
			return false;
	return true;
}


// For vectorizedarray check if > holds for all elements
template<typename Number>
bool
operator>(VectorizedArray<Number> a, Number b)
{
	constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
	for(std::size_t lane = 0; lane < macro_size; lane++)
		if(a[lane] <= b)
			return false;
	return true;
}


namespace std
{
	template<typename Number>
	class numeric_limits<VectorizedArray<Number>>
	{
	public:
		static Number
		epsilon()
			{
				return numeric_limits<Number>::epsilon();
			};
	};
} // namespace std

template<typename Number>
void
printTable(Table<2, Number> tab)
{
	std::size_t           m          = tab.size()[0];
	std::size_t           n          = tab.size()[1];
	std::cout << "------------------------------------\n";
	for(std::size_t i = 0; i < m; i++)
	{
		for(std::size_t j = 0; j < n; j++)
			std::cout << ((int)(tab(i, j) * 100 + 0.5)) / 100.0 << "\t";
		std::cout << "\n";
	}
	std::cout << "------------------------------------\n";
}

template<typename Number>
void
printTable(Table<2, VectorizedArray<Number>> tab)
{
	constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
	std::size_t           m          = tab.size()[0];
	std::size_t           n          = tab.size()[1];
	std::cout << "------------------------------------\n";
	for(std::size_t lane = 0; lane < macro_size; lane++)
	{
		std::cout << "-----------------\n";
		for(std::size_t i = 0; i < m; i++)
		{
			for(std::size_t j = 0; j < n; j++)
				std::cout << ((int)(tab(i, j)[lane] * 100 + 0.5)) / 100.0 << "\t";
			std::cout << "\n";
		}
		std::cout << "-----------------\n";
	}
	std::cout << "------------------------------------\n";
}
template<typename Number>
void
printAlignedVector(AlignedVector<Number> vec)
{
	std::size_t           m          = vec.size();
	std::cout << "######################################\n";
	for(std::size_t i = 0; i < m; i++)
		std::cout << ((int)(vec[i] * 100 + 0.5)) / 100.0 << "\t";
	std::cout << "\n######################################\n";
}

template<typename Number>
void
printAlignedVector(AlignedVector<VectorizedArray<Number>> vec)
{
	constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
	std::size_t           m          = vec.size();
	std::cout << "######################################";
	for(std::size_t lane = 0; lane < macro_size; lane++)
	{
		std::cout << "\n-----------------\n";
		for(std::size_t i = 0; i < m; i++)
			std::cout << ((int)(vec[i] * 100 + 0.5)) / 100.0 << "\t";
	}
	std::cout << "\n######################################\n";
}
template<typename Number>
void
svd(const Number * matrix_begin,
    const std::size_t m,
    const std::size_t n,
    Number *            U_begin,
    Number *       singular_values_begin,
    Number *            VT_begin)
{
	LAPACKFullMatrix<Number> mat(m,n);
	for (unsigned int mm = 0; mm < m; ++mm)
		for (unsigned int nn = 0; nn < n; ++nn)
			mat(mm,nn) = *(matrix_begin++);
	mat.compute_svd();
	LAPACKFullMatrix<Number> U_  = mat.get_svd_u();
	LAPACKFullMatrix<Number> VT_ = mat.get_svd_vt();
	for(std::size_t i = 0; i < U_.size()[0]; i++)
		for(std::size_t j = 0; j < U_.size()[1]; j++)
			*(U_begin++) = U_(i, j);
	for(std::size_t i = 0; i < VT_.size()[0]; i++)
	{
		*(singular_values_begin++) = mat.singular_value(i);
		for(std::size_t j = 0; j < VT_.size()[1]; j++)
		{
			*(VT_begin++)= VT_(i, j);
		}
	}
        
}

template<typename Number>
void
svd(const Table<2, Number> matrix,
    Table<2, Number> &            U,
    AlignedVector<Number> &       singular_values,
    Table<2, Number> &            VT)
{
	svd<Number>(&(matrix(0,0)),matrix.size()[0],matrix.size()[1],&(U(0,0)),&(singular_values[0]), &(VT(0,0)));        
}
template<typename Number>
void
svd(const Table<2, VectorizedArray<Number>> & matrix,
    Table<2, VectorizedArray<Number>> &            U,
    AlignedVector<VectorizedArray<Number>> &       singular_values,
    Table<2, VectorizedArray<Number>> &            VT)
{
	constexpr std::size_t macro_size = VectorizedArray<Number>::n_array_elements;
	for(std::size_t lane = 0; lane < macro_size; lane++)
	{
		Table <2,Number> lane_matrix(matrix.size()[0],matrix.size()[1]);
		Table<2, Number>             lane_U(U.size()[0],U.size()[1]);
		AlignedVector<Number>        lane_singular_values(singular_values.size());
		Table<2, Number>             lane_VT(VT.size()[0],VT.size()[1]);
		for(std::size_t i = 0; i < matrix.size()[0]; i++)
			for(std::size_t j = 0; j < matrix.size()[1];j++)
				lane_matrix(i,j) = matrix(i,j)[lane];
				
		svd<Number>(lane_matrix, lane_U,lane_singular_values,lane_VT);
		for(std::size_t i = 0; i < U.size()[0]; i++)
			for(std::size_t j = 0; j < U.size()[1];j++)
				U(i,j)[lane] = lane_U(i,j);
		for(std::size_t i = 0; i < VT.size()[0]; i++)
		{
			singular_values[i][lane] = lane_singular_values[i];
			for(std::size_t j = 0; j < VT.size()[1];j++)
				VT(i,j)[lane] = lane_VT(i,j);
		}
	}
        
}




#endif
