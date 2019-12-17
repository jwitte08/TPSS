/*
 * kroneckersvd.h
 *
 *  Created on: Nov 14, 2019
 *      Author: schmidt
 */

#ifndef KRONECKERSVD_H_
#define KRONECKERSVD_H_

#include <deal.II/base/table.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include "alignedlinalg.h"

using namespace dealii;

// calculate the new index (the index in the reshuffled matrix) of the i-th diagonal entry, having a
// diagonal matrix only makes sense if kronecker factors are square
std::array<std::size_t, 2>
reshuffle_diag_index(std::size_t i, std::size_t big_m, std::size_t small_m)
{
	AssertIndexRange(i, big_m * small_m);
	std::size_t block_i = i / small_m;
	std::size_t sub_i   = i % small_m;
	return {block_i * (big_m + 1), sub_i * (small_m + 1)};
}

// Calculate the product of a Rank 1 matrix, given as dyadic product of two
// vectors, with a vector: (in⊗inT)vec where ⊗ is the dyadic product
template<typename Number>
AlignedVector<Number>
rank1_vector_multiplication(const AlignedVector<Number> & in,
                            const AlignedVector<Number> & inT,
                            const AlignedVector<Number> & vec)
{
	AssertDimension(inT.size(), vec.size());
	Number inTMvec = inner_product(inT, vec);
	return vector_scaling(in, inTMvec);
}


// calculates the product of a rank k matrix, given as k dyadic products of two
// vectors, with a vector
template<typename Number>
AlignedVector<Number>
rankk_vector_multiplication(const std::vector<AlignedVector<Number>> & in,
                            const std::vector<AlignedVector<Number>> & inT,
                            const AlignedVector<Number> &              vec)
{
	AlignedVector<Number> ret(in[0].size());
	for(std::size_t i = 0; i < in.size(); i++)
		ret = vector_addition(ret, rank1_vector_multiplication(in[i], inT[i], vec));
	return ret;
}
//calculates matrix-vector product of a reshuffled diagonal matrix with a vector, big_m and small_m of the wanted kronecker structure are passed to have correct reshuffling
template<typename Number>
AlignedVector<Number>
reshuffled_diag_vector_multiplication(const AlignedVector<Number> & diagonal,
                                      const AlignedVector<Number>   vec,
                                      std::size_t                   big_m,
                                      std::size_t                   small_m)
{
	AssertDimension(vec.size(), small_m * small_m);
	AlignedVector<Number> ret(big_m * big_m);
	for(std::size_t i = 0; i < diagonal.size(); i++)
	{
		std::array<std::size_t, 2> reshuffled_index = reshuffle_diag_index(i, big_m, small_m);
		ret[reshuffled_index[0]] += diagonal[i] * vec[reshuffled_index[1]];
	}
	return ret;
}
//calculates matrix-vector product of a transposed reshuffled diagonal matrix with a vector, big_m and small_m of the wanted kronecker structure are passed to have correct reshuffling

template<typename Number>
AlignedVector<Number>
transpose_reshuffled_diag_vector_multiplication(const AlignedVector<Number> & diagonal,
                                                const AlignedVector<Number>   vec,
                                                std::size_t                   big_m,
                                                std::size_t                   small_m)
{
	AssertDimension(vec.size(), big_m * big_m);
	AlignedVector<Number> ret(small_m * small_m);
	for(std::size_t i = 0; i < diagonal.size(); i++)
	{
		std::array<std::size_t, 2> reshuffled_index = reshuffle_diag_index(i, big_m, small_m);
		ret[reshuffled_index[1]] += diagonal[i] * vec[reshuffled_index[0]];
	}
	return ret;
}
// transform the last vector of a family of vectors such that the family becomes
// orthogonal
template<typename Number>
void
orthogonalize(std::vector<AlignedVector<Number>> & vectors)
{
	for(std::size_t j = 0; j < vectors.size() - 1; j++) // orthogonalize r
		vectors.back() = vector_addition(vectors.back(),
						 vector_scaling(vectors[j],
								-inner_product(vectors.back(), vectors[j]) /
								inner_product(vectors[j], vectors[j])));
}



// compute svd of bidiagonal matrix
template<typename Number>
void
bidiagonal_svd(const AlignedVector<Number> & diagonal,
               const AlignedVector<Number> & upper_diagonal,
               Table<2, Number> &            U,
               AlignedVector<Number> &       singular_values,
               Table<2, Number> &            VT)
{
	std::size_t base_len = diagonal.size();
	AssertDimension(base_len, upper_diagonal.size() + 1);
	AssertDimension(base_len, U.size()[0]);
	AssertDimension(base_len, U.size()[1]);
	AssertDimension(base_len, singular_values.size());
	AssertDimension(base_len, VT.size()[0]);
	AssertDimension(base_len, VT.size()[1]);
	Table<2, Number> bidiag_mat(base_len, base_len);
	for(std::size_t i = 0; i < base_len; i++)
	{
		bidiag_mat(i, i) = diagonal[i];
		if(i < base_len - 1)
			bidiag_mat(i, i + 1) = upper_diagonal[i];
	}
	svd(bidiag_mat,U,singular_values,VT);
}

template<typename Number>
Table<2, Number>
khatri_rao(const Table<2, Number> & in1, const Table<2, Number> & in2)
{
	AssertDimension(in1.size()[1], in2.size()[1]);
	Table<2, Number> ret(in1.size()[0]*in1.size()[0], in1.size()[1]);
	for(std::size_t i = 0; i < in1.size()[0]; i++)
		for(std::size_t j = 0; j < in2.size()[0]; j++)
			for(std::size_t k = 0; k < in1.size()[1]; k++)
				ret(i*in2.size()[0]+j,k) = in1(i,k)*in2(j,k);
	return ret;
}
template<typename Number>
Table<2, Number>
unfold_rank1(std::array<AlignedVector<Number>, 3> polyadic_factors, std::size_t direction)
{
	AlignedVector<Number> first;
	AlignedVector<Number> second;
	AlignedVector<Number> third;
	if(direction == 0)
	{
		first = polyadic_factors[0];
		second = polyadic_factors[2];
		third = polyadic_factors[1];
	}
	if(direction == 1)
	{
		first = polyadic_factors[1];
		second = polyadic_factors[2];
		third = polyadic_factors[0];
	}
	if(direction == 2)
	{
		first = polyadic_factors[2];
		second = polyadic_factors[1];
		third = polyadic_factors[0];
	}
	Table<2, Number> ret(first.size(),second.size()*third.size());
	for(std::size_t i=0; i < first.size(); i++)
		for(std::size_t j=0; j < second.size(); j++)
			for(std::size_t k=0; k< third.size(); k++)
				ret(i,j*third.size()+k) = first[i]*second[j]*third[k];
	return ret;
}
template<typename Number>
Table<2, Number>
unfold_rankk(std::vector<std::array<AlignedVector<Number>, 3>> polyadic_factors, std::size_t direction)
{
	Table<2, Number> ret = unfold_rank1(polyadic_factors[0], direction);
	for(std::size_t i = 1; i < polyadic_factors.size(); i++)
		ret = matrix_addition(ret, unfold_rank1(polyadic_factors[i], direction));
	return ret;
}


/*
  Compute the low Kronecker rank approximation, i.e. the ksvd, of a matrix of
  the form M = Σ A_i⊗B_i where A_i is called the big matrix and B_i is called
  the small matrix. We first reshuffle M and then compute the first few singular
  values/vectors by using the Lanczos algorithm. The matricization of these
  singular vectors then is the low Kronecker rank approximation. The matrix M is
  passed in "in", and the low rank approximation is passed in "out"
*/
template<typename Number>
void
compute_ksvd(const std::vector<std::array<Table<2, Number>, 2>> &    in,
             std::vector<std::array<Table<2, Number>, 2>> & out,
             std::size_t lanczos_iterations = -1)
{
	std::size_t in_rank = in.size();
	std::size_t out_rank = out.size();
	if(lanczos_iterations == (std::size_t)-1)
	{
		lanczos_iterations = out_rank*out_rank+10;
	}
	
	std::size_t                        big_m   = in[0][0].size()[0];
	std::size_t                        big_n   = in[0][0].size()[1];
	std::size_t                        small_m = in[0][1].size()[0];
	std::size_t                        small_n = in[0][1].size()[1];
	std::vector<AlignedVector<Number>> big_matrices_vectorized;
	std::vector<AlignedVector<Number>> small_matrices_vectorized;
	for(std::size_t i = 0; i < in_rank; i++)
	{
		small_matrices_vectorized.push_back(vectorize_matrix(in[i][1]));
		big_matrices_vectorized.push_back(vectorize_matrix(in[i][0]));
	}
	AlignedVector<Number> beta;
	beta.push_back(Number(1)); // we artificially introduce a first value for
	// beta to define beta.back()
	AlignedVector<Number>              alpha;
	std::vector<AlignedVector<Number>> r;
	std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(small_m * small_n)};
	p.back()[0]                          = Number(1);
	std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(big_m * big_n)};
	std::vector<AlignedVector<Number>> v;
	for(std::size_t i = 0;
	    i < lanczos_iterations && std::abs(beta.back()) > std::numeric_limits<Number>::epsilon();
	    i++)
	{
		beta.push_back(std::sqrt(inner_product(p.back(), p.back())));
		v.push_back(vector_inverse_scaling(p.back(), beta.back()));
		r.push_back(vector_scaling(u.back(), -beta.back()));
		r.back() = vector_addition(r.back(),
					   rankk_vector_multiplication(big_matrices_vectorized,
								       small_matrices_vectorized,
								       v.back()));
		orthogonalize(r);
		alpha.push_back(std::sqrt(inner_product(r.back(), r.back())));
		u.push_back(vector_inverse_scaling(r.back(), alpha.back()));
		p.push_back(vector_scaling(v.back(), -alpha.back()));
		p.back() = vector_addition(p.back(),
					   rankk_vector_multiplication(small_matrices_vectorized,
								       big_matrices_vectorized,
								       u.back()));
		orthogonalize(p);
	}
	std::size_t           base_len = alpha.size() - 1;
	Table<2, Number>      U(base_len, big_m * big_n); // discard first value of u since it is zero
	Table<2, Number>      V(base_len,
				small_m * small_n); // discard last value of v since it is zero
	AlignedVector<Number> real_beta(base_len -
					1); // discard first two values of beta, first is artificially
	// introduced, second only depends on inital guess
	AlignedVector<Number> real_alpha(base_len); // discard last value of alpha since it is zero
	for(std::size_t i = 0; i < base_len; i++)
	{
		real_alpha[i] = alpha[i];
		if(i < base_len - 1)
			real_beta[i] = beta[i + 2];
		for(std::size_t j = 0; j < big_m * big_n; j++)
			U(i, j) = u[i + 1][j];
		for(std::size_t j = 0; j < small_m * small_n; j++)
			V(i, j) = v[i][j];
	}

	AlignedVector<Number> singular_values(base_len);
	Table<2, Number>      tildeU(base_len, base_len);
	Table<2, Number>      tildeVT(base_len, base_len);

	bidiagonal_svd(real_alpha, real_beta, tildeU, singular_values, tildeVT);

	Table<2, Number> left_singular_vectors  = matrix_transpose_multiplication(tildeU, U);
	Table<2, Number> right_singular_vectors = matrix_multiplication(tildeVT, V);

	for(std::size_t i = 0; i < out_rank; i++)
	{
		for(std::size_t k = 0; k < big_m; k++)
			for(std::size_t l = 0; l < big_n; l++)
				out[i][0](k, l) = left_singular_vectors(i, k * big_n + l) * std::sqrt(singular_values[i]);
		for(std::size_t k = 0; k < small_m; k++)
			for(std::size_t l = 0; l < small_n; l++)
				out[i][1](k, l) =
					right_singular_vectors(i, k * small_n + l) * std::sqrt(singular_values[i]);
	}
	
}
/*
  Compute the low Kronecker rank approximation, i.e. the ksvd, of a diagonal matrix.
  We compute the first few singular
  values/vectors by using the Lanczos algorithm. The matricization of these
  singular vectors then is the low Kronecker rank approximation. The matrix M is
  passed in "in", and the low rank approximation is passed in "out"
*/

template<typename Number>
void
compute_ksvd(AlignedVector<Number> &                                   in,
             std::vector<std::array<Table<2, Number>,2>> & out,
             std::size_t lanczos_iterations = -1)
{
	
	std::size_t out_rank = out.size();
	if(lanczos_iterations == (std::size_t)-1)
	{
		lanczos_iterations = out_rank*out_rank+10;
	}
	
	std::size_t           big_m   = out[0][0].size()[0];
	std::size_t           small_m = out[0][1].size()[0];
	AlignedVector<Number> beta;
	beta.push_back(Number(1)); // we artificially introduce a first value for
	// beta to define beta.back()
	AlignedVector<Number>              alpha;
	std::vector<AlignedVector<Number>> r;
	std::vector<AlignedVector<Number>> p = {AlignedVector<Number>(small_m * small_m)};
	p.back()[0]                          = Number(1);
	std::vector<AlignedVector<Number>> u = {AlignedVector<Number>(big_m * big_m)};
	std::vector<AlignedVector<Number>> v;
	for(std::size_t i = 0;
	    i < lanczos_iterations && std::abs(beta.back()) > std::numeric_limits<Number>::epsilon();
	    i++)
	{
		beta.push_back(std::sqrt(inner_product(p.back(), p.back())));
		v.push_back(vector_inverse_scaling(p.back(), beta.back()));
		r.push_back(vector_scaling(u.back(), -beta.back()));
		r.back() =
			vector_addition(r.back(),
					reshuffled_diag_vector_multiplication(in, v.back(), big_m, small_m));
		orthogonalize(r);
		alpha.push_back(std::sqrt(inner_product(r.back(), r.back())));
		u.push_back(vector_inverse_scaling(r.back(), alpha.back()));
		p.push_back(vector_scaling(v.back(), -alpha.back()));
		p.back() = vector_addition(
			p.back(), transpose_reshuffled_diag_vector_multiplication(in, u.back(), big_m, small_m));
		orthogonalize(p);
	}
	std::size_t           base_len = alpha.size() - 1;
	Table<2, Number>      U(base_len, big_m * big_m); // discard first value of u since it is zero
	Table<2, Number>      V(base_len,
				small_m * small_m); // discard last value of v since it is zero
	AlignedVector<Number> real_beta(base_len -
					1); // discard first two values of beta, first is artificially
	// introduced, second only depends on inital guess
	AlignedVector<Number> real_alpha(base_len); // discard last value of alpha since it is zero
	for(std::size_t i = 0; i < base_len; i++)
	{
		real_alpha[i] = alpha[i];
		if(i < base_len - 1)
			real_beta[i] = beta[i + 2];
		for(std::size_t j = 0; j < big_m * big_m; j++)
			U(i, j) = u[i + 1][j];
		for(std::size_t j = 0; j < small_m * small_m; j++)
			V(i, j) = v[i][j];
	}

	AlignedVector<Number> singular_values(base_len);
	Table<2, Number>      tildeU(base_len, base_len);
	Table<2, Number>      tildeVT(base_len, base_len);

	bidiagonal_svd(real_alpha, real_beta, tildeU, singular_values, tildeVT);

	Table<2, Number> left_singular_vectors  = matrix_transpose_multiplication(tildeU, U);
	Table<2, Number> right_singular_vectors = matrix_multiplication(tildeVT, V);

	for(std::size_t i = 0; i < out_rank; i++)
	{
		for(std::size_t k = 0; k < big_m; k++)
			for(std::size_t l = 0; l < big_m; l++)
				out[i][0](k, l) = left_singular_vectors(i, k * big_m + l) * std::sqrt(singular_values[i]);
		for(std::size_t k = 0; k < small_m; k++)
			for(std::size_t l = 0; l < small_m; l++)
				out[i][1](k, l) =
					right_singular_vectors(i, k * small_m + l) * std::sqrt(singular_values[i]);
	}
}


template<typename Number>
void
compute_kcp(const std::vector<std::array<Table<2, Number>, 3>> &    in,
	    std::vector<std::array<Table<2, Number>, 3>> & out,
	    std::size_t als_iterations = -1)
{
	
	std::size_t in_rank = in.size();

	std::size_t out_rank = out.size();
	if(als_iterations == (std::size_t)-1)
	{
		als_iterations = out_rank*out_rank+10;
	}
	std::vector<std::array<AlignedVector<Number>, 3>> matrices_vectorized;
	for(std::size_t i = 0; i < in_rank; i++)
	{
		std::array<AlignedVector<Number>, 3> matrix_vect = {vectorize_matrix(in[i][0]),vectorize_matrix(in[i][1]),vectorize_matrix(in[i][2])};
		matrices_vectorized.push_back(matrix_vect);
	}
	for(std::size_t i = 0; i < als_iterations; i++)
	{
		//TODO
	}
	

}

#endif
