
namespace TPSS
{
template<int dim, typename number>
typename MappingInfo<dim, number>::LocalData
MappingInfo<dim, number>::extract_cartesian_scaling(
  const std::array<unsigned int, 3> * const     triple_begin,
  const std::pair<unsigned int, unsigned int> * bid_count_pair,
  const unsigned int                            n_batches) const
{
  using namespace dealii;
  Assert(n_batches != 0, ExcInternalError());
  Assert(patch_size != -1, ExcInternalError());
  Assert(n_cells_per_direction != -1, ExcInternalError());
  Assert(this->mf_mapping_info != nullptr, ExcInternalError());
  Assert(this->mf_mapping_data != nullptr, ExcInternalError());

  constexpr auto         regular_vpatch_size = UniversalInfo<dim>::n_cells(PatchVariant::vertex);
  constexpr unsigned int last_cell_no_vpatch = regular_vpatch_size - 1;

  auto                                         batch_triple = triple_begin;
  typename MappingInfo<dim, number>::LocalData local_data(n_cells_per_direction);
  for(unsigned int nn = 0; nn < n_batches; ++nn) // check if UNIFORM
  {
    const auto         batch_id  = bid_count_pair[nn].first;
    const auto         n_triples = bid_count_pair[nn].second;
    const unsigned int offsets   = mf_mapping_data->data_index_offsets[batch_id];
    const Tensor<2, dim, VectorizedArray<number>> * jacobian_invT =
      (&mf_mapping_data->jacobians[0][offsets]);

    for(unsigned int triple_no = 0; triple_no < n_triples; ++batch_triple, ++triple_no)
    {
      const auto & cell_no = batch_triple->at(2); // cell number within patch
      const auto & vcomp   = batch_triple->at(1); // lane within macro patch
      const auto & bcomp   = batch_triple->at(0); // lane within batch

      if(cell_no == 0) // first cell has scalings in x-, y-, z- directions for the first cell
        // per direction
        for(unsigned int d = 0; d < dim; ++d)
          local_data.h_inverses[d][0][vcomp] = jacobian_invT[0][d][d][bcomp];
      else if(cell_no == last_cell_no_vpatch) // last cell has
        // scalings in x-, y-,
        // z- directions for
        // the second cell per
        // direction (NOTE
        // this is on purpose
        // only performed for
        // vertex patches)
        for(unsigned int d = 0; d < dim; ++d)
          local_data.h_inverses[d][1][vcomp] = jacobian_invT[0][d][d][bcomp];
    }
  }
  AssertDimension(std::distance(triple_begin, batch_triple),
                  (patch_size == 1) ?
                    macro_size :
                    (patch_size == regular_vpatch_size) ? regular_vpatch_size * macro_size : false);

  return local_data;
}

template<int dim, typename number>
typename MappingInfo<dim, number>::LocalData
MappingInfo<dim, number>::compute_average_scaling(dealii::FEValues<dim> &          fe_values,
                                                  const PatchWorker<dim, number> & patch_worker,
                                                  const unsigned int               patch_id) const
{
  using namespace dealii;

  Assert(n_cells_per_direction != -1, ExcInternalError());

  const auto cell_collection{std::move(patch_worker.get_cell_collection(patch_id))};
  typename MappingInfo<dim, number>::LocalData local_data(n_cells_per_direction);

  // LAMBDA: computes the average (arc) length between opposite faces
  const auto n_qpoints  = fe_values.n_quadrature_points;
  const auto qmode_size = additional_data.n_q_points; // isotropic !!!
  AssertDimension(n_qpoints, Utilities::pow(qmode_size, dim));
  // // DEBUG
  // const auto& quad = fe_values.get_quadrature();
  // const auto qpoints_ref = quad.get_points();

  // TODO replace feface_values by face-quadrature
  const QGaussLobatto<dim - 1> fquadrature(qmode_size);
  const auto                   fweights{std::move(fquadrature.get_weights())};
  const auto                   n_qpoints_face = Utilities::pow(qmode_size, dim - 1);

  auto && compute_distance = [&](const auto & macro_cell) {
    // // DEBUG
    // std::cout << "Compute average arc length on cells ";
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> qpoints{n_qpoints};
    auto volume{make_vectorized_array<double>(0.)};
    for(unsigned int vv = 0; vv < macro_size; ++vv)
    {
      const auto & cell = macro_cell[vv];
      // // DEBUG
      // std::cout << cell->index() << " " << std::endl;

      // *** store quadrature points in vectorized form
      fe_values.reinit(cell);
      const auto temp = fe_values.get_quadrature_points();
      auto       qp   = qpoints.begin();
      for(auto point = temp.cbegin(); point != temp.cend(); ++point, ++qp)
        for(unsigned int d = 0; d < dim; ++d)
          (*qp)[d][vv] = (*point)[d];
      // // DEBUG
      // for (const auto& p : temp)
      //   std::cout << p << "   ";

      // *** compute the volume of the macro cell
      for(unsigned q = 0; q < n_qpoints; ++q)
        volume[vv] += fe_values.JxW(q);
    }
    // // DEBUG
    // std::cout << std::endl;

    // *** compute the average arc length between opposite faces
    std::array<VectorizedArray<double>, dim> cell_lengths;
    for(unsigned int d = 0; d < dim; ++d)
    {
      auto h = make_vectorized_array<double>(0.);
      if(additional_data.use_arc_length) // compute arc length between opposite faces
        for(unsigned int qf = 0; qf < n_qpoints_face; ++qf)
        {
          const auto qf_multi   = Tensors::uni_to_multiindex<dim - 1>(qf, qmode_size);
          const auto qfibre     = Tensors::index_fibre<dim>(qf_multi, /*mode*/ d, qmode_size);
          auto       arc_length = make_vectorized_array<double>(0.);
          for(unsigned int i = 1; i < qfibre.size(); ++i)
          {
            const auto & delta = qpoints[qfibre[i - 1]] - qpoints[qfibre[i]];
            arc_length += delta.norm();
          }
          h += make_vectorized_array<double>(fweights[qf]) * arc_length;
        }

      else // compute direct distance between opposite faces
      {
        for(unsigned int qf = 0; qf < n_qpoints_face; ++qf)
        {
          const auto   qf_multi = Tensors::uni_to_multiindex<dim - 1>(qf, qmode_size);
          const auto   qfibre   = Tensors::index_fibre<dim>(qf_multi, /*mode*/ d, qmode_size);
          const auto & delta    = qpoints[qfibre.front()] - qpoints[qfibre.back()];
          h += make_vectorized_array<double>(fweights[qf]) * delta.norm_square();
        }
        h = std::sqrt(h);
      }
      // // DEBUG
      // std::cout << "h = " << varray_to_string(h) << std::endl;

      cell_lengths[d] = h;
    }

    return std::make_pair(cell_lengths, volume);
  };

  // LAMBDA check if the standard orientation is satisfied
  auto && assert_standard_orientation = [](const auto & macro_cell) {
    for(const auto & cell : macro_cell)
      for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        (void)cell, void(face_no);
        Assert(cell->face_orientation(face_no), ExcMessage("Invalid face orientation."));
        Assert(!cell->face_flip(face_no), ExcMessage("Invalid face flip."));
        Assert(!cell->face_rotation(face_no), ExcMessage("Invalid face rotation."));
      }
  };

  // *** compute the average scaling of each cell
  std::array<AlignedVector<VectorizedArray<number>>, dim> avg_cell_lengths;
  avg_cell_lengths.fill(AlignedVector<VectorizedArray<number>>(cell_collection.size()));
  auto volume_patch = make_vectorized_array<double>(0.);
  for(unsigned int cell_no = 0; cell_no < cell_collection.size(); ++cell_no)
  {
    const auto & macro_cell = cell_collection[cell_no];
    assert_standard_orientation(macro_cell);

    // *** compute average cell lengths & the volume of the actual cell
    const auto   pair{std::move(compute_distance(macro_cell))};
    const auto & macro_lengths = pair.first;
    for(unsigned int d = 0; d < dim; ++d)
      avg_cell_lengths[/*direction*/ d][cell_no] = macro_lengths[d];

    // *** normalize with respect to the volume
    if(additional_data.normalize_patch)
    {
      const auto & volume_act = pair.second;
      volume_patch += volume_act;
      auto volume_sur = make_vectorized_array<double>(1.);
      for(unsigned int d = 0; d < dim; ++d)
        volume_sur *= avg_cell_lengths[d][cell_no];
      // // DEBUG
      // std::cout << "vol_actual = " << varray_to_string(volume_act) << std::endl;
      // std::cout << "volume_surrogate = " << varray_to_string(volume_sur) << std::endl;
      const auto alpha =
        std::pow(volume_act / volume_sur, 1. / static_cast<double>(dim)); // normalization
      for(unsigned int direction = 0; direction < dim; ++direction)
        avg_cell_lengths[direction][cell_no] *= alpha;
      auto volume_new = make_vectorized_array<double>(1.);
      for(unsigned int d = 0; d < dim; ++d)
        volume_new *= avg_cell_lengths[d][cell_no];
      // // DEBUG
      // std::cout << "volume_newrogate = " << varray_to_string(volume_new) << std::endl;
      // for (unsigned int d=0; d<dim; ++d)
      // 	std::cout << "h = " << varray_to_string(avg_cell_lengths[/*direction*/d][cell_no]) <<
      // std::endl;
    }
  }

  if(cell_collection.size() == 1) // CELL PATCHES
    for(unsigned int direction = 0; direction < dim; ++direction)
      local_data.h_inverses[direction][/*cell_no_1d*/ 0] = 1. / avg_cell_lengths[direction][0];

  else if(cell_collection.size() == 1 << dim) // VERTEX PATCHES
  {
    // *** compute average patch lengths from all cell lengths
    constexpr double                                        N = static_cast<number>(1 << (dim - 1));
    std::array<std::array<VectorizedArray<number>, 2>, dim> h;
    h.fill({make_vectorized_array<number>(0.), make_vectorized_array<number>(0.)});
    for(unsigned int cell_no = 0; cell_no < cell_collection.size(); ++cell_no)
      for(unsigned int direction = 0; direction < dim; ++direction)
      {
        std::bitset<dim> binary_cell_no(cell_no);
        const auto       cell_no_1d = static_cast<std::size_t>(binary_cell_no[direction]);
        h[direction][cell_no_1d] += avg_cell_lengths[direction][cell_no] / N;
      }
    for(unsigned int direction = 0; direction < dim; ++direction)
      for(unsigned int cell_no_1d = 0; cell_no_1d < /*n_cells_per_direction*/ 2; ++cell_no_1d)
        local_data.h_inverses[direction][cell_no_1d] = 1. / h[direction][cell_no_1d];

    // *** normalize patch lengths by volume
    if(additional_data.normalize_patch)
    {
      auto volume_patch_sur = make_vectorized_array<double>(1.);
      for(unsigned int d = 0; d < dim; ++d)
      {
        const auto & lengths_1d = h[d];
        volume_patch_sur *= std::accumulate(lengths_1d.cbegin(),
                                            lengths_1d.cend(),
                                            make_vectorized_array<double>(0.));
      }
      // // DEBUG
      // std::cout << "vol_actual = " << varray_to_string(volume_patch) << std::endl;
      // std::cout << "volume_surrogate = " << varray_to_string(volume_patch_sur) << std::endl;
      // for (unsigned int d=0; d<dim; ++d)
      // 	print_row_variable (std::cout, 10, "direction:", 5, d, 5, "h = ", 20,
      // varray_to_string(h[d][0]), 20, varray_to_string(h[d][1]));
      const auto alpha = std::pow(volume_patch / volume_patch_sur, 1. / static_cast<double>(dim));
      for(unsigned int direction = 0; direction < dim; ++direction)
      {
        const auto & lengths_1d  = h[direction];
        auto         h_old       = lengths_1d.cbegin();
        auto &       lengths_inv = local_data.h_inverses[direction];
        for(auto h_inv = lengths_inv.begin(); h_inv != lengths_inv.end(); ++h_old, ++h_inv)
          *h_inv = 1. / ((*h_old) * alpha);
      }
      // // DEBUG
      // auto volume_new = make_vectorized_array<double> (1.);
      // for (unsigned int d=0; d<dim; ++d)
      // 	{
      // 	  const auto& lengths_inv = local_data.h_inverses[d];
      // 	  volume_new *= 1./lengths_inv[0] + 1./lengths_inv[1];
      // 	}
      // std::cout << "vol_surrogate(new) = " << varray_to_string(volume_new) << std::endl;
    }
  }

  return local_data;
}

template<int dim, typename number>
void
MappingInfo<dim, number>::initialize_storage(const PatchInfo<dim> &                 patch_info,
                                             const MatrixFreeConnect<dim, number> & mf_connect,
                                             const AdditionalData &                 addit_data)
{
  using namespace dealii;

  const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
  if(is_mpi_parallel && patch_info.empty())
    return;
  else
    AssertThrow(!patch_info.empty(), ExcMessage("invalid PatchInfo"));

  // TODO we fail assert on single-cell meshes
  Assert(!mf_connect.batch_starts.empty(), ExcNotInitialized());
  Assert(!mf_connect.batch_count_per_id.empty(), ExcNotInitialized());
  Assert(!mf_connect.bcomp_vcomp_cindex.empty(), ExcNotInitialized());

  // *** check if patch_info's PartitionData is valid
  const auto   patch_variant  = patch_info.get_additional_data().patch_variant;
  const auto & partition_data = patch_info.subdomain_partition_data;
  typename std::decay<decltype(partition_data)>::type partition_data_tester;
  PatchWorker<dim, number>::compute_partition_data(partition_data_tester,
                                                   patch_info.get_internal_data());
  Assert(partition_data_tester.check_compatibility(patch_info.subdomain_partition_data),
         ExcMessage("Passed PatchInfo object is corrupt. Subdomain partitioning does not fit the "
                    "internal data!"));

  // *** set input information
  this->mf_storage        = mf_connect.mf_storage;
  const auto & mf_storage = *(this->mf_storage);
  additional_data         = addit_data;

  // *** extract domain decomposition related information
  n_subdomains = partition_data.n_subdomains();
  if(patch_variant == PatchVariant::cell)
    patch_size = 1;
  else if(patch_variant == PatchVariant::vertex)
    patch_size = 1 << dim;
  n_cells_per_direction =
    (patch_variant == PatchVariant::cell) ? 1 : (patch_variant == PatchVariant::vertex) ? 2 : -1;
  Assert(n_subdomains != static_cast<unsigned int>(-1), ExcInternalError());
  Assert(n_cells_per_direction != -1, ExcInternalError());
  Assert(patch_size != -1, ExcInternalError());

  // *** link to internal MatrixFree mapping storage
  this->mf_mapping_info        = &mf_storage.get_mapping_info();
  const Mapping<dim> & mapping = *(mf_mapping_info->mapping);
  this->mf_mapping_data        = internal::MatrixFreeFunctions::
    MappingInfoCellsOrFaces<dim, number, false /*is_face*/, VectorizedArrayType>::get(
      *mf_mapping_info, 0 /*quad_no*/);

  const unsigned int       n_q_points  = additional_data.n_q_points; // before:5
  const auto &             dof_handler = mf_storage.get_dof_handler();
  const QGaussLobatto<dim> cell_quadrature(n_q_points);
  FEValues<dim>            fe_values(mapping,
                          dof_handler.get_fe(),
                          cell_quadrature,
                          dealii::update_quadrature_points | dealii::update_JxW_values);

  mapping_data_starts.reserve(n_subdomains);
  for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    for(unsigned int part = 0; part < partition_data.n_partitions(color); ++part)
    {
      const auto                          range = partition_data.get_patch_range(part, color);
      const std::array<unsigned int, 3> * batch_triple             = nullptr;
      const std::pair<unsigned int, unsigned int> * bid_count_pair = nullptr;
      for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
      {
        const unsigned int n_batches =
          mf_connect.set_pointers_and_count(patch_id, batch_triple, bid_count_pair);

        PatchType patch_type = PatchType::cartesian;
        for(unsigned int nn = 0; nn < n_batches; ++nn) // check if UNIFORM
        {
          const auto batch_id  = bid_count_pair[nn].first;
          const auto cell_type = mf_mapping_info->get_cell_type(batch_id);
          // std::cout << "cell_type[" << patch_id << "]: " << cell_type << std::endl;
          patch_type = patch_type < cell_type ? cell_type : patch_type;
        }
        // std::cout << "patch_type[" << patch_id << "]: " << patch_type << std::endl;

        // TODO data compression ?!
        /*** we can extract the scalings from the jacobians stored matrix-free mapping data ***/
        if(patch_type == PatchType::cartesian)
        {
          mf_connect.set_pointers_and_count(patch_id, batch_triple, bid_count_pair);
          const auto local_data =
            extract_cartesian_scaling(batch_triple, bid_count_pair, n_batches);

          mapping_data_starts.emplace_back(internal_data.h_inverses.size());
          submit_local_data(local_data);
        } // patch_type == cartesian

        // *** we have to compute average scalings to gain a tensor product structure
        else
        {
          PatchWorker<dim, number> patch_worker{patch_info};
          auto && local_data = compute_average_scaling(fe_values, patch_worker, patch_id);
          mapping_data_starts.emplace_back(internal_data.h_inverses.size());
          submit_local_data(local_data);
        }
      } // patch loop
    }   // partition loop

  AssertDimension(internal_data.h_inverses.size(), internal_data.h_lengths.size());
  AssertDimension(internal_data.h_inverses.size() % (n_cells_per_direction * dim), 0);
  mapping_data_initialized = true;
}
} // end namespace TPSS
