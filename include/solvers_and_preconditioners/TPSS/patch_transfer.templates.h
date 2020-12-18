namespace TPSS
{
template<int dim, typename Number>
template<typename VectorType>
AlignedVector<VectorizedArray<Number>>
PatchTransfer<dim, Number>::gather(const VectorType & src) const
{
  AssertIndexRange(patch_id, n_subdomains);
  AssertDimension(src.size(), patch_dof_worker.n_global_dofs());

  AlignedVector<VectorizedArray<Number>> dst(n_dofs_per_patch());
  for(unsigned int lane = 0; lane < macro_size; ++lane)
  {
    const auto & global_dof_indices = get_dof_indices(lane);
    AssertDimension(dst.size(), global_dof_indices.size());
    auto dof_index = global_dof_indices.cbegin();
    for(auto dst_value = dst.begin(); dst_value != dst.end(); ++dof_index, ++dst_value)
      (*dst_value)[lane] = internal::local_element(src, *dof_index);
  }

  AssertDimension(dst.size(), n_dofs_per_patch());
  return dst;
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::gather_add(const ArrayView<VectorizedArray<Number>> dst,
                                       const VectorType &                       src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto & src_local = gather(src);
  std::transform(dst.begin(),
                 dst.end(),
                 src_local.begin(),
                 dst.begin(),
                 [](const auto & dst, const auto & src) { return dst + src; });
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::gather_add(AlignedVector<VectorizedArray<Number>> & dst,
                                       const VectorType &                       src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto dst_view = make_array_view<VectorizedArray<Number>>(dst.begin(), dst.end());
  gather_add(dst_view, src);
}


template<int dim, typename Number>
template<typename VectorType, bool is_restricted, bool do_add>
void
PatchTransfer<dim, Number>::scatter_add_impl(
  VectorType &                                   dst,
  const ArrayView<const VectorizedArray<Number>> src) const
{
  AssertIndexRange(patch_id, n_subdomains);
  AssertDimension(dst.size(), patch_dof_worker.n_global_dofs());
  AssertDimension(src.size(), n_dofs_per_patch());

  for(unsigned int lane = 0; lane < patch_dof_worker.n_lanes_filled(patch_id); ++lane)
  {
    const auto & global_dof_indices = get_dof_indices(lane);
    AssertDimension(src.size(), global_dof_indices.size());

    if(is_restricted)
    {
      const auto is_restricted_dof = patch_dof_worker.get_restricted_dof_flags(patch_id, lane);
      AssertDimension(is_restricted_dof.size(), global_dof_indices.size());
      auto i = 0U;
      for(auto src_value = src.cbegin(); src_value != src.cend(); ++i, ++src_value)
      {
        const auto dof_index = global_dof_indices[i];
        if(!is_restricted_dof[i])
        {
          if(do_add)
            internal::local_element(dst, dof_index) += (*src_value)[lane];
          else
            internal::local_element(dst, dof_index) = (*src_value)[lane];
        }
      }
    }
    else
    {
      auto dof_index = global_dof_indices.cbegin();
      for(auto src_value = src.cbegin(); src_value != src.cend(); ++dof_index, ++src_value)
      {
        if(do_add)
          internal::local_element(dst, *dof_index) += (*src_value)[lane];
        else
          internal::local_element(dst, *dof_index) = (*src_value)[lane];
      }
    }
  }
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::scatter_add(VectorType &                                   dst,
                                        const ArrayView<const VectorizedArray<Number>> src) const
{
  scatter_add_impl<VectorType, false>(dst, src);
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::scatter_add(VectorType &                                   dst,
                                        const AlignedVector<VectorizedArray<Number>> & src) const
{
  const auto src_view = make_array_view<const VectorizedArray<Number>>(src.begin(), src.end());
  scatter_add(dst, src_view);
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::scatter(VectorType &                                   dst,
                                    const ArrayView<const VectorizedArray<Number>> src) const
{
  scatter_add_impl<VectorType, false, false>(dst, src);
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::scatter(VectorType &                                   dst,
                                    const AlignedVector<VectorizedArray<Number>> & src) const
{
  const auto src_view = make_array_view<const VectorizedArray<Number>>(src.begin(), src.end());
  scatter(dst, src_view);
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::rscatter_add(VectorType &                                   dst,
                                         const ArrayView<const VectorizedArray<Number>> src) const
{
  scatter_add_impl<VectorType, true>(dst, src);
}


template<int dim, typename Number>
template<typename VectorType>
void
PatchTransfer<dim, Number>::rscatter_add(VectorType &                                   dst,
                                         const AlignedVector<VectorizedArray<Number>> & src) const
{
  const auto src_view = make_array_view<const VectorizedArray<Number>>(src.begin(), src.end());
  rscatter_add(dst, src_view);
}



// -----------------------------   PatchTransferBlock   ----------------------------



template<int dim, typename Number>
template<typename BlockVectorType>
AlignedVector<VectorizedArray<Number>>
PatchTransferBlock<dim, Number>::gather(const BlockVectorType & src) const
{
  AlignedVector<VectorizedArray<Number>> dst;
  reinit_local_vector(dst);
  auto begin = dst.begin();
  for(std::size_t b = 0; b < n_blocks; ++b)
  {
    const auto                         transfer = transfers[b];
    const auto                         size     = transfer->n_dofs_per_patch();
    ArrayView<VectorizedArray<Number>> dst_block{begin, size};
    transfer->gather_add(dst_block, src.block(b));
    begin += size;
  }
  AssertThrow(begin == dst.end(), ExcMessage("Inconsistent slicing."));
  return dst;
}


template<int dim, typename Number>
template<typename BlockVectorType>
void
PatchTransferBlock<dim, Number>::gather_add(AlignedVector<VectorizedArray<Number>> & dst,
                                            const BlockVectorType &                  src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto & src_local = gather(src);
  std::transform(dst.begin(),
                 dst.end(),
                 src_local.begin(),
                 dst.begin(),
                 [](const auto & dst, const auto & src) { return dst + src; });
}


template<int dim, typename Number>
template<typename BlockVectorType>
void
PatchTransferBlock<dim, Number>::scatter_add(
  BlockVectorType &                              dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  scatter_add_impl<BlockVectorType, false>(dst, src);
}


template<int dim, typename Number>
template<typename BlockVectorType>
void
PatchTransferBlock<dim, Number>::rscatter_add(
  BlockVectorType &                              dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  AssertThrow(false, ExcMessage("This method is untested!"));
  scatter_add_impl<BlockVectorType, true>(dst, src);
}


template<int dim, typename Number>
template<typename BlockVectorType, bool is_restricted>
void
PatchTransferBlock<dim, Number>::scatter_add_impl(
  BlockVectorType &                              dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  auto begin = src.begin();
  for(std::size_t b = 0; b < n_blocks; ++b)
  {
    const auto                               transfer = transfers[b];
    const auto                               size     = transfer->n_dofs_per_patch();
    ArrayView<const VectorizedArray<Number>> src_block{begin, size};
    if(is_restricted)
      transfer->rscatter_add(dst.block(b), src_block);
    else
      transfer->scatter_add(dst.block(b), src_block);
    begin += size;
  }
  Assert(begin == src.end(), ExcMessage("Inconsistent slicing."));
}



} // end namespace TPSS
