/*
 * Tests for ...TODO...
 *
 *  Created on: Feb 12, 2021
 *      Author: witte
 */

#include <gtest/gtest.h>

#include "common_integrator.h"
#include "poisson_problem.h"

#include "test_utilities.h"

using namespace dealii;



using TestParamsLinear2D    = testing::Types<Util::NonTypeParams<2, 1>>;
using TestParamsQuadratic2D = testing::Types<Util::NonTypeParams<2, 2>>;



/**
 * A predicate returning true if the iterator passed belongs to the collection
 * passed at construction.
 */
template<typename BaseIterator>
struct BelongsToCollection
{
  BelongsToCollection(const std::vector<BaseIterator> & cell_collection_in)
    : cell_collection(cell_collection_in)
  {
  }

  bool
  operator()(const BaseIterator & this_bi) const
  {
    return std::any_of(cell_collection.cbegin(),
                       cell_collection.cend(),
                       [&](const BaseIterator & other_bi) { return other_bi == this_bi; });
  }

  const std::vector<BaseIterator> & cell_collection;
};



/**
 * Generate a filtered iterator range which reflects the collection of cell
 * iterators passed.
 */
template<typename BaseIterator>
IteratorRange<FilteredIterator<BaseIterator>>
make_local_cell_range(const std::vector<BaseIterator> & cell_collection_in)
{
  Assert(!cell_collection_in.empty(), ExcMessage("Collection is empty."));
  const auto & cell_begin =
    *(std::min_element(cell_collection_in.cbegin(), cell_collection_in.cend()));
  const auto & cell_end =
    std::next(*(std::max_element(cell_collection_in.cbegin(), cell_collection_in.cend())));
  IteratorRange<BaseIterator> unfiltered_range(cell_begin, cell_end);
  return filter_iterators(unfiltered_range, BelongsToCollection<BaseIterator>(cell_collection_in));
}



std::string
get_filename(const MPI_Comm & mpi_communicator)
{
  const unsigned int n_mpi_procs = Utilities::MPI::n_mpi_processes(mpi_communicator);
  std::ostringstream oss;
  oss << "local_mesh_loop";
  if(n_mpi_procs == 1U)
    oss << ".log";
  else
    oss << ".np" << n_mpi_procs << ".log";
  return oss.str();
}



FullMatrix<double>
extract_locally_relevant_matrix(
  const TrilinosWrappers::SparseMatrix &                   matrix,
  const std::shared_ptr<const Utilities::MPI::Partitioner> partitioner)
{
  const auto locally_relevant_size = partitioner->local_size() + partitioner->n_ghost_indices();

  FullMatrix<double> submatrix(locally_relevant_size, locally_relevant_size);

  LinearAlgebra::distributed::Vector<double> e_j(partitioner);
  LinearAlgebra::distributed::Vector<double> dst(partitioner);

  for(types::global_dof_index j = 0; j < e_j.size(); ++j)
  {
    e_j = 0.;
    dst = 0.;

    if(partitioner->in_local_range(j))
      e_j.local_element(partitioner->global_to_local(j)) = 1.;

    e_j.update_ghost_values();
    dst.zero_out_ghosts();

    /// DEBUG
    // {
    //   std::ostringstream oss;
    //   oss << "e_" << j << " : " << std::endl;
    //   e_j.print(oss);
    //   std::cout << oss.str();
    // }

    matrix.vmult(dst, e_j);

    /// DEBUG
    // {
    //   std::ostringstream oss;
    //   oss << "dst = A e_" << j << " : " << std::endl;
    //   dst.print(oss);
    //   std::cout << oss.str();
    // }

    dst.compress(VectorOperation::add);
    dst.update_ghost_values();

    /// DEBUG
    // {
    //   std::ostringstream oss;
    //   oss << "dst = A e_" << j << " : " << std::endl;
    //   dst.print(oss);
    //   std::cout << oss.str();
    // }

    if(partitioner->in_local_range(j) || partitioner->is_ghost_entry(j))
    {
      const auto jj = partitioner->global_to_local(j);
      for(auto ii = 0U; ii < locally_relevant_size; ++ii)
        submatrix(ii, jj) = dst.local_element(ii);
    }
  }

  return submatrix;
}



////////// TestLocalMeshLoopBase



template<typename T>
class TestLocalMeshLoopBase : public testing::Test
{
protected:
  static constexpr int dim       = T::template value<0>();
  static constexpr int fe_degree = T::template value<1>();

  enum class Variant
  {
    makelocalcellrange,
    assembleovercells
  };

  virtual void
  SetUp() override
  {
    ofs.open(get_filename(MPI_COMM_WORLD), std::ios_base::app);
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

  template<typename Number>
  void
  compare_matrix(const FullMatrix<Number> & matrix, const FullMatrix<Number> & other)
  {
    Util::compare_matrix(matrix, other, *pcout);
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

  using PatchTransfer      = typename TPSS::PatchTransfer<dim, double>;
  using cell_iterator_type = typename PatchTransfer::CellIterator;

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
  test(const typename Base::Variant test_variant)
  {
    using PoissonProblem = typename Poisson::ModelProblem<dim, fe_degree, dof_layout>;

    const auto poisson_problem = std::make_shared<PoissonProblem>(rt_parameters);
    poisson_problem->pcout     = pcout;
    poisson_problem->create_triangulation();
    poisson_problem->distribute_dofs();

    level   = poisson_problem->level;
    mapping = &(poisson_problem->mapping);

    mf_storage        = poisson_problem->template build_mf_storage<double>(level);
    subdomain_handler = poisson_problem->template build_patch_storage<double>(level, mf_storage);
    patch_transfer    = std::make_shared<PatchTransfer>(*subdomain_handler);

    switch(test_variant)
    {
      case Base::Variant::makelocalcellrange:
        test_makelocalcellrange();
        break;
      case Base::Variant::assembleovercells:
        test_assembleovercells();
        break;
      default:
        ASSERT_TRUE(false) << "test variant is not implemented";
    }
  }

  void
  test_makelocalcellrange()
  {
    ASSERT_TRUE(patch_transfer) << "patch_transfer not setup";

    const auto & partition_data   = subdomain_handler->get_partition_data();
    const auto & patch_dof_worker = patch_transfer->get_patch_dof_worker();

    const auto n_subdomains = partition_data.n_subdomains();

    for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
    {
      *pcout << "\npatch: " << patch_index << std::endl;
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
      {
        const auto & cell_collection = patch_dof_worker.get_cell_collection(patch_index, lane);

        auto sorted_cell_collection(cell_collection);
        std::sort(sorted_cell_collection.begin(), sorted_cell_collection.end());

        /// DEBUG sorted cell collection
        *pcout << "sorted cell collection on lane: " << lane << std::endl;
        for(const auto & cell : sorted_cell_collection)
          *pcout << cell->index() << " ";
        *pcout << std::endl;

        const auto & local_cell_range = make_local_cell_range<cell_iterator_type>(cell_collection);

        /// DEBUG make local cell range
        for(const auto & cell : local_cell_range)
          *pcout << cell->index() << " ";
        *pcout << std::endl;

        ASSERT_TRUE(std::equal(local_cell_range.begin(),
                               local_cell_range.end(),
                               sorted_cell_collection.begin()))
          << "The filtered iterator range obtained by make_local_cell_range() does not coincide with the (sorted) cell collection it originates from.";
      }
    }
  }

  void
  test_assembleovercells()
  {
    ASSERT_TRUE(patch_transfer) << "patch_transfer is not setup";
    ASSERT_TRUE(mapping) << "mapping is not set";

    const auto & dof_handler      = subdomain_handler->get_dof_handler();
    const auto & partition_data   = subdomain_handler->get_partition_data();
    const auto & patch_dof_worker = patch_transfer->get_patch_dof_worker();

    const std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner =
      patch_dof_worker.get_dof_info().vector_partitioner;

    const auto n_subdomains = partition_data.n_subdomains();

    //: setup
    const IndexSet & locally_owned_dof_indices = dof_handler.locally_owned_mg_dofs(level);
    IndexSet         locally_relevant_dof_indices;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, locally_relevant_dof_indices);

    AffineConstraints<double> empty_constraints;
    empty_constraints.clear();
    empty_constraints.reinit(locally_relevant_dof_indices);
    empty_constraints.close();

    TrilinosWrappers::SparsityPattern dsp(locally_owned_dof_indices,
                                          locally_owned_dof_indices,
                                          locally_relevant_dof_indices,
                                          MPI_COMM_WORLD);
    MGTools::make_flux_sparsity_pattern(dof_handler, dsp, level);
    dsp.compress();

    level_matrix.clear();
    level_matrix.reinit(dsp);

    //: assemble
    using MW::ScratchData;
    using MW::DoF::CopyData;

    auto cell_worker =
      [&](const auto & cell, ScratchData<dim> & scratch_data, CopyData & copy_data) {
        FEValues<dim> & phi = scratch_data.fe_values;
        phi.reinit(cell);

        const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

        auto & cell_data = copy_data.cell_data.emplace_back(dofs_per_cell);

        cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_data.matrix(i, j) +=
              static_cast<double>(cell_data.dof_indices[i] + cell_data.dof_indices[j]);

        /// DEBUG
        // *pcout << "visiting cell " << cell->index() << std::endl;
        // if(pcout->is_active())
        //   cell_data.matrix.print_formatted(pcout->get_stream());
      };

    const auto copier = [&](const CopyData & copy_data) {
      for(const auto & cd : copy_data.cell_data)
        empty_constraints.template distribute_local_to_global<TrilinosWrappers::SparseMatrix>(
          cd.matrix, cd.dof_indices, level_matrix);
    };

    ScratchData<dim> scratch_data(
      *mapping, dof_handler.get_fe(), fe_degree + 1, update_default, update_default);

    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_mg(level),
                          dof_handler.end_mg(level),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);

    level_matrix.compress(VectorOperation::add);

    const auto & locally_relevant_matrix =
      extract_locally_relevant_matrix(level_matrix, vector_partitioner);

    /// DEBUG
    // if(pcout->is_active())
    //   locally_relevant_matrix.print_formatted(pcout->get_stream());

    std::vector<MatrixAsTable<VectorizedArray<double>>> local_matrices_cut;
    {
      FullMatrix<double> tmp;
      for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
      {
        patch_transfer->reinit(patch_index);
        const auto n_dofs       = patch_transfer->n_dofs_per_patch();
        auto &     local_matrix = local_matrices_cut.emplace_back();
        local_matrix.as_table().reinit(n_dofs, n_dofs);
        tmp.reinit(n_dofs, n_dofs);
        for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
        {
          tmp = 0.;
          std::vector<types::global_dof_index> dof_indices =
            std::move(patch_transfer->get_global_dof_indices(lane));
          std::vector<unsigned int> proc_local_dof_indices;
          std::transform(dof_indices.begin(),
                         dof_indices.end(),
                         std::back_inserter(proc_local_dof_indices),
                         [&](const auto dof_index) {
                           return vector_partitioner->global_to_local(dof_index);
                         });
          tmp.extract_submatrix_from(locally_relevant_matrix,
                                     proc_local_dof_indices,
                                     proc_local_dof_indices);
          local_matrix.fill_submatrix(tmp, 0U, 0U, lane);
        }
      }
    }

    std::vector<MatrixAsTable<VectorizedArray<double>>> local_matrices_ass;
    {
      ScratchData<dim> scratch_data(
        *mapping, dof_handler.get_fe(), fe_degree + 1, update_default, update_default);

      CopyData copy_data;

      FullMatrix<double> tmp;
      for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
      {
        patch_transfer->reinit(patch_index);
        const auto n_dofs       = patch_transfer->n_dofs_per_patch();
        auto &     local_matrix = local_matrices_ass.emplace_back();
        local_matrix.as_table().reinit(n_dofs, n_dofs);
        tmp.reinit(n_dofs, n_dofs);

        for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
        {
          tmp = 0.;

          const auto & cell_collection = patch_dof_worker.get_cell_collection(patch_index, lane);

          const auto & local_cell_range = make_local_cell_range(cell_collection);

          /// DEBUG make local cell range
          // for(const auto & cell : local_cell_range)
          //   *pcout << cell->index() << " ";
          // *pcout << std::endl;

          const auto & g2l = patch_transfer->get_global_to_local_dof_indices(lane);

          /// DEBUG
          // for(const auto & [key, value] : g2l)
          //   *pcout << key << ":" << value << "  ";
          // *pcout << std::endl;

          const auto local_copier = [&](const CopyData & copy_data) {
            for(const auto & cd : copy_data.cell_data)
            {
              std::vector<unsigned int> local_dof_indices;
              std::transform(cd.dof_indices.begin(),
                             cd.dof_indices.end(),
                             std::back_inserter(local_dof_indices),
                             [&](const auto dof_index) {
                               const auto & local_index = g2l.find(dof_index);
                               return local_index != g2l.cend() ? local_index->second :
                                                                  numbers::invalid_unsigned_int;
                             });

              /// DEBUG
              // *pcout << "cd.dof_indices: " << vector_to_string(cd.dof_indices) << std::endl;
              // *pcout << "local_dof_indices: " << vector_to_string(local_dof_indices) <<
              // std::endl;

              ASSERT_EQ(local_dof_indices.size(), cd.matrix.m());
              ASSERT_EQ(local_dof_indices.size(), cd.matrix.n());

              for(auto i = 0U; i < cd.matrix.m(); ++i)
                if(local_dof_indices[i] != numbers::invalid_unsigned_int)
                  for(auto j = 0U; j < cd.matrix.n(); ++j)
                    if(local_dof_indices[j] != numbers::invalid_unsigned_int)
                      tmp(local_dof_indices[i], local_dof_indices[j]) += cd.matrix(i, j);
            }
          };

          MeshWorker::mesh_loop(local_cell_range,
                                cell_worker,
                                local_copier,
                                scratch_data,
                                copy_data,
                                MeshWorker::assemble_own_cells | MeshWorker::assemble_ghost_cells);

          /// DEBUG
          // if(pcout->is_active())
          //   tmp.print_formatted(pcout->get_stream());

          local_matrix.fill_submatrix(tmp, 0U, 0U, lane);
        }
      }
    }

    ASSERT_EQ(local_matrices_cut.size(), local_matrices_ass.size());
    for(auto patch_index = 0U; patch_index < n_subdomains; ++patch_index)
    {
      *pcout << "\npatch: " << patch_index << std::endl;
      const auto & local_matrix_cut = local_matrices_cut[patch_index];
      const auto & local_matrix_ass = local_matrices_ass[patch_index];
      for(auto lane = 0U; lane < patch_dof_worker.n_lanes_filled(patch_index); ++lane)
      {
        /// DEBUG make local cell range
        // {
        //   const auto & cell_collection  = patch_dof_worker.get_cell_collection(patch_index,
        //   lane); const auto & local_cell_range = make_local_cell_range(cell_collection);
        //   for(const auto & cell : local_cell_range)
        //     *pcout << cell->index() << " ";
        //   *pcout << std::endl;
        // }

        *pcout << "compare local matrices on lane " << lane << ": " << std::endl;
        const auto & fullmatrix_cut = table_to_fullmatrix(local_matrix_cut.as_table(), lane);
        const auto & fullmatrix_ass = table_to_fullmatrix(local_matrix_ass.as_table(), lane);
        this->compare_matrix(fullmatrix_ass, fullmatrix_cut);
      }
    }
  }

  std::shared_ptr<const MatrixFree<dim, double>>       mf_storage;
  std::shared_ptr<const SubdomainHandler<dim, double>> subdomain_handler;
  std::shared_ptr<PatchTransfer>                       patch_transfer;
  TrilinosWrappers::SparseMatrix                       level_matrix;

  unsigned int         level   = numbers::invalid_unsigned_int;
  const Mapping<dim> * mapping = nullptr;
};

TYPED_TEST_SUITE_P(TestLocalMeshLoop);

TYPED_TEST_P(TestLocalMeshLoop, VertexPatchDGQ_makelocalcellrange_MPI)
{
  using Fixture     = TestLocalMeshLoop<TypeParam>;
  using TestVariant = typename Fixture::Base::Variant;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::template test<TPSS::DoFLayout::DGQ>(TestVariant::makelocalcellrange);

  Fixture::rt_parameters.mesh.n_refinements = 2U;
  Fixture::template test<TPSS::DoFLayout::DGQ>(TestVariant::makelocalcellrange);
}

TYPED_TEST_P(TestLocalMeshLoop, VertexPatchDGQ_assembleovercells_MPI)
{
  using Fixture     = TestLocalMeshLoop<TypeParam>;
  using TestVariant = typename Fixture::Base::Variant;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::rt_parameters.mesh.n_refinements = 1U;
  Fixture::template test<TPSS::DoFLayout::DGQ>(TestVariant::assembleovercells);

  Fixture::rt_parameters.mesh.n_refinements = 2U;
  Fixture::template test<TPSS::DoFLayout::DGQ>(TestVariant::assembleovercells);
}

TYPED_TEST_P(TestLocalMeshLoop, VertexPatchQ_assembleovercells_MPI)
{
  using Fixture     = TestLocalMeshLoop<TypeParam>;
  using TestVariant = typename Fixture::Base::Variant;

  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.patch_variant = TPSS::PatchVariant::vertex;
  Fixture::rt_parameters.multigrid.pre_smoother.schwarz.smoother_variant =
    TPSS::SmootherVariant::additive;
  Fixture::rt_parameters.multigrid.post_smoother.schwarz =
    Fixture::rt_parameters.multigrid.pre_smoother.schwarz;

  Fixture::rt_parameters.mesh.n_refinements = 1U;
  Fixture::template test<TPSS::DoFLayout::Q>(TestVariant::assembleovercells);

  Fixture::rt_parameters.mesh.n_refinements = 2U;
  Fixture::template test<TPSS::DoFLayout::Q>(TestVariant::assembleovercells);
}

REGISTER_TYPED_TEST_SUITE_P(TestLocalMeshLoop,
                            VertexPatchDGQ_makelocalcellrange_MPI,
                            VertexPatchDGQ_assembleovercells_MPI,
                            VertexPatchQ_assembleovercells_MPI);

INSTANTIATE_TYPED_TEST_SUITE_P(Linear2D, TestLocalMeshLoop, TestParamsLinear2D);
INSTANTIATE_TYPED_TEST_SUITE_P(Quadratic2D, TestLocalMeshLoop, TestParamsQuadratic2D);



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);

  /// first mpi proc clears output file
  const bool is_first_proc = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  if(is_first_proc)
  {
    std::ofstream ofs(get_filename(MPI_COMM_WORLD), std::ios_base::out);
    ofs.close();
  }

  return RUN_ALL_TESTS();
}
