
/*
 * tensors.h: test the functionality of transforming multi- into
 * uni-index and vice versa
 *
 *  Created on: Dec 07, 2018
 *      Author: witte
 */

#include <deal.II/base/utilities.h>

#include "solvers_and_preconditioners/TPSS/tensors.h"

int
main() // int argc, char* argv[])
{
  using namespace dealii;

  constexpr int      order   = 3;
  const unsigned int size    = 5;
  const unsigned int n_elems = Utilities::pow(size, order);

  for(unsigned int i = 0; i < n_elems; ++i)
  {
    const auto i_multi = Tensors::uni_to_multiindex<order>(i, size);
    const auto ii      = Tensors::multi_to_uniindex<order>(i_multi, size);
    // std::cout << i << "   " << Tensors::multiindex_to_string<order> (i_multi) << "   " << ii <<
    // std::endl;
    AssertThrow(i == ii, ExcMessage("uni->multi->uni transformation failed"));
  }

  return 0;
}
