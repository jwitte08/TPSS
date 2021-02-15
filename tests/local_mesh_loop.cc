/*
 * Tests for ...TODO...
 *
 *  Created on: Feb 12, 2021
 *      Author: witte
 */

#include <gtest/gtest.h>

#include "poisson_problem.h"

#include "test_utilities.h"

using namespace dealii;



using TestParamsLinear2D    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic2D = testing::Types<Util::NonTypeParams<2, 2>>;



//   template<typename BaseIterator>
//   const BaseIterator & get_base_at_position_impl(const std::vector<BaseIterator> * collection,
//   const size_type position)
//   {
//     Assert(collection, ExcMessage("Collection of cell iterators is not initialized."));
//     Assert(!collection->empty(), ExcMessage("collection of cells is empty!"));
//     if(position == collection->size())
//       return std::next(collection->back());
//     AssertIndexRange(position, collection->size());
//     return (*collection)[position];
//   }



// template<typename BaseIterator>
// struct CellOnPatchIterator<BaseIterator> : public BaseIterator
// {
//   using value_type = BaseIterator;
//   using size_type = typename std::vector<BaseIterator>::size_type;
//   static const invalid_size = static_cast<size_type>(-1);

//   CellOnPatchIterator(const std::vector<BaseIterator> & cells_on_patch, const size_type position)
//     : BaseIterator(get_base_at_position_impl(&cells_on_patch, position)),
//       collection(&cells_on_patch),
//       pos(position)
//   {
//   }

//   CellOnPatchIterator(const CellOnPatchIterator & other)

//     CellOnPatchIterator &
//     operator=(const CellOnPatchIterator & other)

//     CellOnPatchIterator &
//     operator=(const BaseIterator & other_base)

//     const std::vector<BaseIterator> * collection = nullptr;
//   size_type pos = invalid_size;
// };



//  template <typename BaseIterator>
//  inline bool
//  CellOnPatchIterator<BaseIterator>::operator==(const CellOnPatchIterator &fi) const
//  {
//    return (static_cast<const BaseIterator &>(*this) ==
//            static_cast<const BaseIterator &>(fi));
//  }



//  template <typename BaseIterator>
//  inline bool
//  CellOnPatchIterator<BaseIterator>::operator!=(const CellOnPatchIterator &fi) const
//  {
//    return (static_cast<const BaseIterator &>(*this) !=
//            static_cast<const BaseIterator &>(fi));
//  }



//  template <typename BaseIterator>
//  inline bool
//  CellOnPatchIterator<BaseIterator>::operator<(const CellOnPatchIterator &fi) const
//  {
//    return (static_cast<const BaseIterator &>(*this) <
//            static_cast<const BaseIterator &>(fi));
//  }



//  template <typename BaseIterator>
//  inline bool
//  CellOnPatchIterator<BaseIterator>::operator==(const BaseIterator &bi) const
//  {
//    return (static_cast<const BaseIterator &>(*this) == bi);
//  }



//  template <typename BaseIterator>
//  inline bool
//  CellOnPatchIterator<BaseIterator>::operator!=(const BaseIterator &bi) const
//  {
//    return (static_cast<const BaseIterator &>(*this) != bi);
//  }



//  template <typename BaseIterator>
//  inline bool
//  CellOnPatchIterator<BaseIterator>::operator<(const BaseIterator &bi) const
//  {
//    return (static_cast<const BaseIterator &>(*this) < bi);
//  }


//  template <typename BaseIterator>
//  inline CellOnPatchIterator<BaseIterator> &
//  CellOnPatchIterator<BaseIterator>::operator++()
//  {
//    if (this->state() == IteratorState::valid)
//      do
//        BaseIterator::operator++();
//      while ((this->state() == IteratorState::valid) && !(*predicate)(*this));
//    return *this;
//  }



//  template <typename BaseIterator>
//  inline CellOnPatchIterator<BaseIterator>
//  CellOnPatchIterator<BaseIterator>::operator++(int)
//  {
//    const CellOnPatchIterator old_state = *this;

//    if (this->state() == IteratorState::valid)
//      do
//        BaseIterator::operator++();
//      while ((this->state() == IteratorState::valid) && !(*predicate)(*this));
//    return old_state;
//  }



//  template <typename BaseIterator>
//  inline CellOnPatchIterator<BaseIterator> &
//  CellOnPatchIterator<BaseIterator>::operator--()
//  {
//    if (this->state() == IteratorState::valid)
//      do
//        BaseIterator::operator--();
//      while ((this->state() == IteratorState::valid) && !(*predicate)(*this));
//    return *this;
//  }

template<typename Iterator>
struct IsLocalCell
{
  IsLocalCell(const std::vector<Iterator> & cell_collection_in)
    : cell_collection(cell_collection_in)
  {
  }

  bool
  operator()(const Iterator & this_bi) const
  {
    return std::any_of(cell_collection.cbegin(),
                       cell_collection.cend(),
                       [&](const Iterator & other_bi) { return other_bi == this_bi; });
  }

  const std::vector<Iterator> & cell_collection;
};

template<typename BaseIterator>
class LocalCellIterator : public FilteredIterator<BaseIterator>
{
  using Base = FilteredIterator<BaseIterator>;

public:
  LocalCellIterator(const std::vector<BaseIterator> & cell_collection_in,
                    const BaseIterator &              bi_in)
    : FilteredIterator<BaseIterator>(
        // [&](const BaseIterator & this_bi) {
        //   return std::any_of(cell_collection_in.cbegin(),
        //                      cell_collection_in.cend(),
        //                      [&](const BaseIterator & other_bi) { return other_bi == this_bi; });
        // },
        IsLocalCell<BaseIterator>(cell_collection_in),
        bi_in)
  {
  }

  /// TODO simply use copy constructor of FilteredIterator ?
  // LocalCellIterator(const LocalCellIterator & other)

  // LocalCellIterator & operator=(const LocalCellIterator & other)
  // {
  //   const FilteredIterator<BaseIterator> & other_fi = other;
  //   this->                                 operator =(other_fi);
  //   return *this;
  // }
};



// template<typename BaseIterator>
// LocalCellIterator<BaseIterator>
// make_local_cell_iterator(const std::vector<BaseIterator> & cell_collection_in,
//                          const BaseIterator &              bi_in)
// {
//   LocalCellIterator<BaseIterator> lci(cell_collection_in, bi_in);
//   lci.set_to_next_positive(bi_in);
//   return lci;
// }



template<typename BaseIterator>
IteratorRange<FilteredIterator<BaseIterator>>
make_local_cell_range(const std::vector<BaseIterator> & cell_collection_in)
{
  Assert(!cell_collection_in.empty(), ExcMessage("Collection is empty."));
  LocalCellIterator<BaseIterator> lci_begin(cell_collection_in, cell_collection_in.front());
  LocalCellIterator<BaseIterator> lci_end(cell_collection_in, cell_collection_in.back());
  return IteratorRange<FilteredIterator<BaseIterator>>(lci_begin, lci_end);
}



////////// TestLocalMeshLoopBase



template<typename T>
class TestLocalMeshLoopBase : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();

  virtual void
  SetUp() override
  {
    ofs.open("local_mesh_loop.log", std::ios_base::app);
    const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    pcout                    = std::make_shared<ConditionalOStream>(ofs, is_first_proc);

    /// base mesh is a vertex patch
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_repetitions    = 2;
    rt_parameters.mesh.n_refinements    = 0U;
  }

  virtual void
  TearDown() override
  {
    ofs.close();
  }

  std::ofstream                       ofs;
  std::shared_ptr<ConditionalOStream> pcout;
  RT::Parameter                       rt_parameters;
};



////////// TestLocalMeshLoop



template<typename T>
class TestLocalMeshLoop : public TestLocalMeshLoopBase<T>
{
protected:
  using Base = TestLocalMeshLoopBase<T>;

  static constexpr int dim       = Base::dim;
  static constexpr int fe_degree = Base::fe_degree;

  using Base::pcout;
  using Base::rt_parameters;

  virtual void
  SetUp() override final
  {
    Base::SetUp();
  }

  virtual void
  TearDown() override final
  {
    Base::TearDown();
  }

  template<TPSS::DoFLayout dof_layout>
  void
  test()
  {
    using PoissonProblem     = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;
    using PatchTransfer      = typename TPSS::PatchTransfer<dim, double>;
    using cell_iterator_type = typename PatchTransfer::CellIterator;

    const auto poisson_problem = std::make_shared<PoissonProblem>(rt_parameters);
    poisson_problem->pcout     = pcout;
    poisson_problem->create_triangulation();
    poisson_problem->distribute_dofs();

    // *** patch transfer
    const auto global_level = poisson_problem->level;
    const auto mf_storage   = poisson_problem->template build_mf_storage<double>(global_level);
    const auto subdomain_handler =
      poisson_problem->template build_patch_storage<double>(global_level, mf_storage);
    const auto   patch_transfer = std::make_shared<PatchTransfer>(*subdomain_handler);
    const auto & dof_handler    = poisson_problem->dof_handler;

    const auto & partition_data = subdomain_handler->get_partition_data();
    const auto   n_subdomains   = partition_data.n_subdomains();

    const auto & patch_dof_worker = patch_transfer->get_patch_dof_worker();

    for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
    {
      *pcout << "patch: " << patch_index << std::endl;
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
      {
        *pcout << "lane: " << lane << std::endl;
        const auto & cell_collection = patch_dof_worker.get_cell_collection(patch_index, lane);
        for(const auto & cell : cell_collection)
          *pcout << cell->index() << " ";
        *pcout << std::endl;

        /// iterator range
        IteratorRange<cell_iterator_type> range(cell_collection.front(), cell_collection.back());
        for(const auto & cell : range)
          *pcout << cell->index() << " ";
        *pcout << std::endl;

        /// local cell iterator range
        // const auto & local_cell_range = make_local_cell_range(cell_collection);
        const auto & local_cell_range =
          filter_iterators(dof_handler.mg_cell_iterators_on_level(global_level),
                           IsLocalCell<cell_iterator_type>(cell_collection));
        for(const auto & cell : local_cell_range)
          *pcout << cell->index() << " ";
        *pcout << std::endl;
      }
    }
  }
};

TYPED_TEST_SUITE_P(TestLocalMeshLoop);

TYPED_TEST_P(TestLocalMeshLoop, VertexPatchDGQ)
{
  using Fixture = TestLocalMeshLoop<TypeParam>;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::template test<TPSS::DoFLayout::DGQ>();

  Fixture::rt_parameters.mesh.n_refinements = 1U;
  Fixture::template test<TPSS::DoFLayout::DGQ>();
}

REGISTER_TYPED_TEST_SUITE_P(TestLocalMeshLoop, VertexPatchDGQ);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLocalMeshLoop, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestLocalMeshLoop, TestParamsQuadratic2D);



int
main(int argc, char ** argv)
{
  /// clear output file
  std::ofstream ofs("local_mesh_loop.log", std::ios_base::out);
  ofs.close();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
