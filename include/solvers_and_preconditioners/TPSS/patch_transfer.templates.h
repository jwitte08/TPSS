namespace TPSS
{
template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
template<typename VectorType>
AlignedVector<VectorizedArray<Number>>
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::gather(
  const VectorType & src) const
{
  AssertDimension(src.size(), sd_handler.get_dof_handler().n_dofs(level));

  auto bid_count_pair = batch_count;
  auto triple_bvc     = batch_triple;

  AlignedVector<VectorizedArray<Number>> dst{n_dofs};
  for(auto & elem : dst)
    elem = 0.;

  for(unsigned int bb = 0; bb < n_batches; ++bb, ++bid_count_pair)
  {
    const unsigned int batch_id = bid_count_pair->first;
    // std::cout << "batch id: " << batch_id << ", count: " << bid_count_pair->second << std::endl;
    fe_eval.reinit(batch_id);
    fe_eval.read_dof_values(src);
    const VectorizedArray<Number> * values = fe_eval.begin_dof_values();
    for(unsigned int comp = 0; comp < bid_count_pair->second; ++comp, ++triple_bvc)
    {
      const auto bcomp   = (*triple_bvc)[0];
      const auto vcomp   = (*triple_bvc)[1];
      const auto cell_no = (*triple_bvc)[2];
      // std::cout << "bcomp: " << bcomp << "  vcomp: " << vcomp << "  cell_no: " << cell_no
      //           << std::endl;
      auto pindex = cell_to_patch_indices.data() + cell_no * n_dofs_per_cell;
      auto in     = values;
      if(is_incomplete_patch)
        for(unsigned int dof = 0; dof < n_dofs_per_cell; ++pindex, ++in, ++dof)
          dst[*pindex][vcomp] = (*in)[bcomp];
      else
        for(unsigned int dof = 0; dof < n_dofs_per_cell; ++pindex, ++in, ++dof)
          dst[*pindex][vcomp] += (*in)[bcomp];
    }
  }

  AssertDimension(dst.size(), n_dofs);
  return dst;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
template<typename VectorType>
void
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::gather_add(
  const ArrayView<VectorizedArray<Number>> dst,
  const VectorType &                       src) const
{
  AssertDimension(dst.size(), n_dofs);
  const auto & src_local = gather(src);
  std::transform(dst.begin(),
                 dst.end(),
                 src_local.begin(),
                 dst.begin(),
                 [](const auto & dst, const auto & src) { return dst + src; });
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
template<typename VectorType>
void
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::gather_add(
  AlignedVector<VectorizedArray<Number>> & dst,
  const VectorType &                       src) const
{
  AssertDimension(dst.size(), n_dofs);
  const auto dst_view = make_array_view<VectorizedArray<Number>>(dst.begin(), dst.end());
  gather_add(dst_view, src);
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
template<typename VectorType>
void
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::scatter_add(
  VectorType &                                   dst,
  const ArrayView<const VectorizedArray<Number>> src) const
{
  AssertDimension(dst.size(), sd_handler.get_dof_handler().n_dofs(level));
  AssertDimension(src.size(), n_dofs);

  auto bid_count_pair = batch_count;
  auto triple_bvc     = batch_triple;

  std::lock_guard<std::mutex> guard(write_mutex);
  // REQUIRE src vectors on incomplete batches are mutually equivalent per vectorization lane.
  // Therefore, an incomplete patch describes only one physical patch and duplicates computations
  // along vectorization lanes.
  AlignedVector<VectorizedArray<Number>> local_dst(n_dofs);
  for(unsigned int bb = 0; bb < n_batches; ++bb, ++bid_count_pair)
  {
    const unsigned int batch_id = bid_count_pair->first;
    // std::cout << "batch id: " << batch_id << ", count: " << bid_count_pair->second << std::endl;
    fe_eval.reinit(batch_id);
    std::fill_n(fe_eval.begin_dof_values(), n_dofs_per_cell, make_vectorized_array<Number>(0.));
    // std::cout << "init" << std::endl;

    for(unsigned int dummy = 0; dummy < bid_count_pair->second; ++dummy, ++triple_bvc)
    {
      const auto bcomp   = (*triple_bvc)[0];
      const auto vcomp   = (*triple_bvc)[1];
      const auto cell_no = (*triple_bvc)[2];
      // std::cout << is_incomplete_patch << std::endl;
      // std::cout << sd_handler.get_matrix_free().at_irregular_cell(bcomp);


      const auto begin = cell_to_patch_indices.data() + cell_no * n_dofs_per_cell;
      const ArrayView<const unsigned int> patch_dofs_on_cell{begin, n_dofs_per_cell};
      auto                                out = fe_eval.begin_dof_values();
      if(is_incomplete_patch)
        for(auto dof = patch_dofs_on_cell.cbegin(); dof != patch_dofs_on_cell.end(); ++out, ++dof)
          // for(unsigned int dof = 0; dof < n_dofs_per_cell; ++dof_index, ++out, ++dof)
          (*out)[bcomp] = src[*dof][vcomp];
      else
        for(auto dof = patch_dofs_on_cell.cbegin(); dof != patch_dofs_on_cell.end(); ++out, ++dof)
          //	  for(unsigned int dof = 0; dof < n_dofs_per_cell; ++dof_index, ++out, ++dof)
          (*out)[bcomp] += src[*dof][vcomp];
    }

    // std::cout << "write" << std::endl;
    // TODO debug this for 2 threads
    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_comp, typename Number>
template<typename VectorType>
void
PatchTransferBase<dim, fe_degree, n_q_points_1d, n_comp, Number>::scatter_add(
  VectorType &                                   dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  const auto src_view = make_array_view<const VectorizedArray<Number>>(src.begin(), src.end());
  scatter_add(dst, src_view);
}

} // end namespace TPSS
