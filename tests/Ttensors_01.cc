
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

  { // isotropic index set
    constexpr int      order   = 3;
    const unsigned int size    = 3;
    const unsigned int n_elems = Utilities::pow(size, order);
    for(unsigned int i = 0; i < n_elems; ++i)
    {
      const auto                  multii = Tensors::uni_to_multiindex<order>(i, size);
      std::array<unsigned, order> sizes;
      sizes.fill(size);
      const auto multii_ani = Tensors::uni_to_multiindex<order>(i, sizes);
      const auto ii         = Tensors::multi_to_uniindex<order>(multii, size);

      std::cout << i << "   " << Tensors::multiindex_to_string<order>(multii) << "   "
                << Tensors::multiindex_to_string<order>(multii_ani) << "   " << ii << std::endl;
      AssertThrow(multii == multii_ani,
                  ExcMessage("Functions do not coincide for isotropic index sets."));
      AssertThrow(i == ii, ExcMessage("uni->multi->uni transformation failed"));
    }
  }

  { // anisotropic index set
    constexpr int               order = 3;
    std::array<unsigned, order> sizes = {2, 3, 4};
    const unsigned int          n_elems =
      std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<unsigned>());
    for(unsigned int i = 0; i < n_elems; ++i)
    {
      const auto multii = Tensors::uni_to_multiindex<order>(i, sizes);
      const auto ii     = Tensors::multi_to_uniindex<order>(multii, sizes);

      std::cout << i << "   " << Tensors::multiindex_to_string<order>(multii) << "   " << ii
                << std::endl;
      AssertThrow(i == ii, ExcMessage("uni->multi->uni transformation failed"));
    }
  }

  return 0;
}
