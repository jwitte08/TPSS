#include "solvers_and_preconditioners/TPSS/kroneckersvd.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"
#include <deal.II/base/table.h>

void printTable(Table<2,double> tab)
{
	std::size_t m = tab.size()[0];
	std::size_t n = tab.size()[1];
	std::cout << "-----------------\n";
	for(std::size_t i = 0; i < m; i++)
	{
		for(std::size_t j = 0; j < n; j++)
			std::cout << ((int)(tab(i,j)*100+0.5))/100.0 << "\t";
		std::cout << "\n";
	}
	std::cout << "-----------------\n";
	
}
void printTable(Table<2,VectorizedArray<double>> tab)
{
	
	constexpr std::size_t macro_size = VectorizedArray<double>::n_array_elements;
	std::size_t m = tab.size()[0];
	std::size_t n = tab.size()[1];
	std::cout << "------------------------------------\n";
        for(std::size_t lane = 0; lane < macro_size; lane++)
	{
		
		std::cout << "-----------------\n";
		for(std::size_t i = 0; i < m; i++)
		{
			for(std::size_t j = 0; j < n; j++)
				std::cout << ((int)(tab(i,j)[lane]*100+0.5))/100.0 << "\t";
			std::cout << "\n";
		}
		std::cout << "-----------------\n";
	}
	std::cout << "------------------------------------\n";
}
using namespace dealii;
int main()
{
	Table<2,double> t1 = Table<2,double>(4,3);
	Table<2,double> t2 = Table<2,double>(2,2);
	t1(0,0) = 1;
	t1(1,1) = 1;
	t1(0,1) = 0;
	t1(1,0) = 0;
	t1(0,2) = 1;
	t1(1,2) = 0.1;
	t1(2,0) = 1;
	t1(2,1) = 0.1;
	t1(2,2) = -7;
	t2(0,0) = 1;
	t2(1,1) = 4;
	t2(0,1) = 2;
	t2(1,0) = 2;
	Table<2,double> t3 = Table<2,double>(4,3);
	Table<2,double> t4 = Table<2,double>(2,2);
	t3(0,0) = 0.5;
	t3(1,1) = 1;
	t3(0,1) = 3;
	t3(1,0) = 3.4;
	t3(0,2) = 2;
	t3(1,2) = -0.4;
	t3(2,0) = 2;
	t3(2,1) = -0.5;
	t3(2,2) = 0;
	t4(0,0) = 1;
	t4(1,1) = 0;
	t4(0,1) = 1;
	t4(1,0) = 1;
	std::array<Table<2,double>,2> kp1 = {t1,t2};
	std::array<Table<2,double>,2> kp2 = {t3,t4};
	std::vector<std::array<Table<2,double>,2>>  mat1 =  {kp1,kp2};
	std::array<std::array<Table<2,double>,2>,3>  approx =  {kp1,kp1,kp1};
	compute_ksvd<2,double,3>(mat1,approx);
	printTable(Tensors::sum(Tensors::kronecker_product(approx[0][0],approx[0][1]),Tensors::kronecker_product(approx[1][0],approx[1][1])));
	printTable(Tensors::sum(Tensors::kronecker_product(t1,t2),Tensors::kronecker_product(t3,t4)));




	Table<2,VectorizedArray<double>> t5 = Table<2,VectorizedArray<double>>(2,3);
	Table<2,VectorizedArray<double>> t6= Table<2,VectorizedArray<double>>(2,2);
	t5(0,0) = make_vectorized_array<double>(1);
	t5(1,1) = make_vectorized_array<double>(1);
	t5(0,1) = make_vectorized_array<double>(0);
	t5(1,0) = make_vectorized_array<double>(0);
	t6(0,0) = make_vectorized_array<double>(1);
	t6(1,1) = make_vectorized_array<double>(4);
	t6(1,1)[2] = 3.1415;
	t6(0,1) = make_vectorized_array<double>(2);
	t6(1,0) = make_vectorized_array<double>(2);
	Table<2,VectorizedArray<double>> t7 = Table<2,VectorizedArray<double>>(2,3);
	Table<2,VectorizedArray<double>> t8 = Table<2,VectorizedArray<double>>(2,2);
	t7(0,0) = make_vectorized_array<double>(0.5);
	t7(1,1) = make_vectorized_array<double>(1);
	t7(0,1) = make_vectorized_array<double>(3);
	t7(0,1)[1] = -2;
	t7(1,0) = make_vectorized_array<double>(3.4);
	t8(0,0) = make_vectorized_array<double>(1);
	t8(1,1) = make_vectorized_array<double>(0);
	t8(0,1) = make_vectorized_array<double>(1);
	t8(0,0)[0] = -2;
	t8(0,0)[3] = -200;
	t8(1,0) = make_vectorized_array<double>(1);
	std::array<Table<2,VectorizedArray<double>>,2> kp3 = {t5,t6};
	std::array<Table<2,VectorizedArray<double>>,2> kp4 = {t7,t8};
	std::vector<std::array<Table<2,VectorizedArray<double>>,2>>  mat2 =  {kp3,kp4};
	std::array<std::array<Table<2,VectorizedArray<double>>,2>,3>  approx2 =  {kp3,kp3,kp3};
	compute_ksvd<2,VectorizedArray<double>,3>(mat2,approx2);
	printTable(Tensors::sum(Tensors::kronecker_product(approx2[0][0],approx2[0][1]),Tensors::kronecker_product(approx2[1][0],approx2[1][1])));
	printTable(Tensors::sum(Tensors::kronecker_product(t5,t6),Tensors::kronecker_product(t7,t8)));
	
}

