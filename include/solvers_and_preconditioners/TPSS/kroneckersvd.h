/*
 * kroneckersvd.h
 *
 *  Created on: Nov 14, 2019
 *      Author: schmidt
 */

#ifndef KRONECKERSVD_H_
#define KRONECKERSVD_H_

#include <iostream>
#include <deal.II/base/table.h>
#include <deal.II/lac/lapack_full_matrix.h>


using namespace dealii;

//Calculate inner product of two AlignedVectors
template<typename Number>
Number vecMvec(const AlignedVector<Number> &in1, const AlignedVector<Number> &in2)
{
	AssertDimension(in1.size(),in2.size());
	Number ret = Number(0);
	for(std::size_t i=0; i < in1.size(); i++)
		ret += in1[i]*in2[i];
	return ret;
}
//Add two AlignedVectors
template<typename Number>
AlignedVector<Number>vecPvec(const AlignedVector<Number> &in1, const AlignedVector<Number> &in2)
{
	AssertDimension(in1.size(),in2.size());
	AlignedVector<Number> ret = AlignedVector<Number>(in1);
	for(std::size_t i=0; i < in1.size(); i++)
		ret[i] = in1[i]+in2[i];
	return ret;
}
//Multiply AlignedVector with scalar
template<typename Number>
AlignedVector<Number> vecMscal(const AlignedVector<Number> &in, const Number &scalar)
{
	AlignedVector<Number> ret = AlignedVector<Number>(in);
	for(std::size_t i=0; i < in.size(); i++)
		ret[i] = in[i]*scalar;
	return ret;
}
//Divide AlignedVector by scalar, if vector is zero allow scalar to be zero
template<typename Number>
AlignedVector<Number> vecMscali(const AlignedVector<Number> &in, const Number &scalar)
{
	AlignedVector<Number> ret = AlignedVector<Number>(in);
	for(std::size_t i=0; i < in.size(); i++)
	{
		if(scalar!=0)
			ret[i] = in[i]/scalar;
		else
			Assert(std::abs(vecMvec(in,in)) < 1e-13,ExcZero());
			ret[i] = 0;
	}
	return ret;
}
//Flatten Table to AlignedVector
template <typename Number>
AlignedVector<Number> vectorizeMatrix(const Table<2,Number> &tab)
{
	std::size_t m = tab.size()[0];
	std::size_t n = tab.size()[1];	
	AlignedVector<Number> ret(m*n);
	for(std::size_t k = 0; k < m; k++)
		for(std::size_t l = 0; l < n; l++)
			ret[k*m + l] = tab(k,l);
	return ret;
}
//Revert vectorizeMatrix
template <typename Number>
Table<2,Number> MatricizeVector(const AlignedVector<Number> &vec, std::size_t m, std::size_t n)
{
	AssertDimension(vec.size(),m*n);
	Table<2,Number>ret(m,n);
	for(std::size_t k = 0; k < m; k++)
		for(std::size_t l = 0; l < n; l++)
			ret(k,l) = vec[k*m+l];
	return ret;
}

//calculates the product of a Rank 1 matrix with a vector: (in⊗inT)vec
//where ⊗ is the dyadic product
template <typename Number>
AlignedVector<Number> rank1Mvec(const AlignedVector<Number> &in, const AlignedVector<Number> &inT, const AlignedVector<Number> &vec)
{
	AssertDimension(inT.size(),vec.size());
	Number inTMvec = vecMvec(inT,vec);
	return vecMscal(in,inTMvec);
}
//calculates the product of a rank k matrix with a vector
template <typename Number>
AlignedVector<Number> rankkMvec(const std::vector<AlignedVector<Number>> &in, const std::vector<AlignedVector<Number>> &inT, const AlignedVector<Number> &vec)
{
	AlignedVector<Number> ret(in[0].size());
	for(std::size_t i=0; i < in.size();i++)
		ret = vecPvec(ret,rank1Mvec(in[i],inT[i],vec));
	return ret;
}
//transform the last vector of a family of vectors such that the family becomes orthogonal
template <typename Number>
void orthogonalize(std::vector<AlignedVector<Number>> &vectors)
{
	for(std::size_t j = 0; j < vectors.size()-1; j++) //orthogonalize r
		vectors.back() = vecPvec(vectors.back(),vecMscal(vectors[j],-vecMvec(vectors.back(),vectors[j])/vecMvec(vectors[j],vectors[j])));
}



/*
  computes the low Kronecker rank approximation, i.e. the ksvd, of a matrix of the form
  M = Σ A_i⊗B_i
  where A_i is called the big matrix and B_i is called the small matrix.
 We first reshuffle M and then compute the first few singular vectors by using the Lanczos algorithm.
 The matricization of these singular vectors then are the low Kronecker approximation.

the matrix M is passed in "in", and the low rank approximation is passed in "out"
*/
template <int dim, typename Number, int out_rank>
void compute_ksvd(std::vector<std::array<Table<2,Number>, dim > > in, std::array<std::array<Table<2,Number>,dim>,out_rank> &out)
{
	std::size_t in_rank = in.size();
	std::size_t big_m = in[0][0].size()[0];
	std::size_t big_n = in[0][0].size()[1];
	std::size_t small_m = in[0][1].size()[0];
	std::size_t small_n = in[0][1].size()[1];
	if(dim == 2)
	{
		std::vector<AlignedVector<Number>> big_matrices_vectorized;
		std::vector<AlignedVector<Number>> small_matrices_vectorized;
		for(std::size_t i=0; i < in_rank; i++)
		{
			small_matrices_vectorized.push_back(vectorizeMatrix(in[i][1]));
			big_matrices_vectorized.push_back(vectorizeMatrix(in[i][0]));
		}
		std::vector<Number> beta = {Number(1)};  //just to define beta.back()                 
		std::vector<Number> alpha;
		std::vector<AlignedVector<Number>> r; 
		std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(small_m * small_n)}; 
		p.back()[0] = 1;
		std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(big_m * big_n)};
		std::vector<AlignedVector<Number>> v; 
		for(int i = 0; i < out_rank*out_rank+15 && std::abs(beta.back()) > 1e-6; i++)
		{
			beta.push_back(std::sqrt(vecMvec(p.back(),p.back())));
			v.push_back(vecMscali(p.back(),beta.back()));
			r.push_back(vecMscal(u.back(),-beta.back()));
			r.back() = vecPvec(r.back(),rankkMvec(big_matrices_vectorized,small_matrices_vectorized,v.back()));           
		        orthogonalize(r);
			alpha.push_back(std::sqrt(vecMvec(r.back(),r.back()))); 
			u.push_back(vecMscali(r.back(),alpha.back()));
			p.push_back(vecMscal(v.back(),-alpha.back()));
			p.back() = vecPvec(p.back(),rankkMvec(small_matrices_vectorized, big_matrices_vectorized, u.back())); 
			orthogonalize(p);
		}
		std::size_t base_len = alpha.size()-1;
		LAPACKFullMatrix <Number> R(base_len,base_len);
		LAPACKFullMatrix <Number> U(base_len,big_m*big_n);
		LAPACKFullMatrix <Number> V(base_len,small_m*small_n);
		for(std::size_t i = 0; i < base_len; i++)
		{
			R(i,i) = alpha[i];
			
			for(std::size_t j = 0; j < big_m*big_n;j++)
				U(i,j) = u[i+1][j];
			for(std::size_t j = 0; j < small_m*small_n;j++)
				V(i,j) = v[i][j];
			if(i < base_len - 1)
				R(i,i+1) = beta[i+2];
		}
		R.compute_svd();
		std::array<Number,out_rank> singular_values;
		for(std::size_t i = 0; i < out_rank; i++)
		{
			if(i < base_len)
				singular_values[i] = R.singular_value(i);
			else
				singular_values[i] = 0;
			std::cout << singular_values[i]<<"\n";
		}
		LAPACKFullMatrix<Number> tildeU = R.get_svd_u();
		LAPACKFullMatrix<Number> tildeV = R.get_svd_vt();
		LAPACKFullMatrix<Number> left_singular_vectors(base_len,big_m*big_n);
		LAPACKFullMatrix<Number> right_singular_vectors(base_len,small_m*small_n);
		tildeU.Tmmult(left_singular_vectors,U);
		tildeV.mmult(right_singular_vectors,V);
		for(std::size_t i=0; i < out_rank; i++)
		{
			for(std::size_t k = 0; k < big_m; k++)
				for(std::size_t l = 0; l < big_n; l++)
					out[i][0](k,l) = left_singular_vectors(i,k*big_m+l)*std::sqrt(singular_values[i]);
			for(std::size_t k = 0; k < small_m; k++)
				for(std::size_t l = 0; l < small_n; l++)
					out[i][1](k,l) = right_singular_vectors(i,k*small_m+l)*std::sqrt(singular_values[i]);
		}
	}
}



#endif
