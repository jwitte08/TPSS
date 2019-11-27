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
			std::cout << ((int)(tab(i,j)*1000+0.5))/1000.0 << "  \t";
		std::cout << "\n";
	}
	std::cout << "-----------------\n";
	
}
using namespace dealii;
int main()
{
	Table<2,double> t1 = Table<2,double>(3,3);
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
	Table<2,double> t3 = Table<2,double>(3,3);
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
	std::vector<std::array<Table<2,double>,2>>  mat2 =  {kp1,kp2};
	std::array<std::array<Table<2,double>,2>,3>  approx =  {kp1,kp1,kp1};
	compute_ksvd<2,double,3>(mat2,approx);
	printTable(Tensors::sum(Tensors::kronecker_product(approx[0][0],approx[0][1]),Tensors::kronecker_product(approx[1][0],approx[1][1])));
	printTable(Tensors::sum(Tensors::kronecker_product(t1,t2),Tensors::kronecker_product(t3,t4)));
}

