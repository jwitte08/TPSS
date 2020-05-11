template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::initialize(
  const std::shared_ptr<const SubdomainHandler<
    dim,
    typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type>>
                         subdomain_handler_in,
  OperatorType &         operator_in,
  const AdditionalData & additional_data_in)
{
  // *** assert that each process has the same number of colors
  const unsigned int n_colors     = subdomain_handler_in->get_partition_data().n_colors();
  const unsigned int n_colors_max = Utilities::MPI::max(n_colors, MPI_COMM_WORLD);
  const unsigned int n_colors_min = Utilities::MPI::min(n_colors, MPI_COMM_WORLD);
  (void)n_colors_max, (void)n_colors_min;
  Assert(n_colors_max == n_colors_min, ExcMessage("Not the same number of colors on each proc."));

  // *** reset members to uninitialized state
  clear();

  // *** set time information
  time_data = {{0., "[SchwarzPrecond] Update residuals:", "[s]", 0},
               {0., "[SchwarzPrecond] Apply inverses:", "[s]", 0},
               {0., "[SchwarzPrecond] Compute inverses:", "[s]", 0},
               {0., "[SchwarzPrecond] Copy Locally Owned:", "[s]", 0},
               {0., "[SchwarzPrecond] Step:", "[s]", 0}};

  // *** initialization of members
  subdomain_handler = subdomain_handler_in;
  linear_operator   = &operator_in;
  transfer          = std::make_shared<typename OperatorType::transfer_type>(*subdomain_handler);
  additional_data   = additional_data_in;
  Assert(additional_data.relaxation > 0., ExcInvalidState());

  const auto & sh_data = subdomain_handler->get_additional_data();
  level                = sh_data.level;
  Assert(level != static_cast<unsigned int>(-1), ExcNotImplemented());
  patch_variant    = sh_data.patch_variant;
  smoother_variant = sh_data.smoother_variant;
  Assert(patch_variant == TPSS::PatchVariant::vertex || patch_variant == TPSS::PatchVariant::cell,
         dealii::ExcNotImplemented());

  // *** compute subproblem inverses
  Timer timer;
  timer.restart();
  compute_inverses();
  time_data[2].add_time(timer.wall_time());

  /// instantiate ghosted vectors (initialization is postponed to the actual
  /// smoothing step)
  solution_ghosted = std::make_shared<VectorType>();
  residual_ghosted = std::make_shared<VectorType>();

  // *** storing SubdomainHandler's timings
  const auto & sh_time_data = subdomain_handler->get_time_data();
  for(const auto & info : sh_time_data)
    time_data.emplace_back(info.time, info.description, info.unit);
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::initialize(
  const SchwarzPreconditioner & schwarz_preconditioner_in,
  const AdditionalData &        additional_data_in)
{
  // *** reset members to uninitialized state
  clear();

  // *** initialization of members
  time_data            = schwarz_preconditioner_in.time_data;
  subdomain_handler    = schwarz_preconditioner_in.subdomain_handler;
  transfer             = schwarz_preconditioner_in.transfer;
  linear_operator      = schwarz_preconditioner_in.linear_operator;
  subdomain_to_inverse = schwarz_preconditioner_in.subdomain_to_inverse;
  additional_data      = additional_data_in;
  level                = schwarz_preconditioner_in.level;
  patch_variant        = schwarz_preconditioner_in.patch_variant;
  smoother_variant     = schwarz_preconditioner_in.smoother_variant;
  solution_ghosted     = schwarz_preconditioner_in.solution_ghosted;
  residual_ghosted     = schwarz_preconditioner_in.residual_ghosted;
  Assert(additional_data.relaxation > 0., ExcMessage("Invalid relaxation factor."));
  Assert(patch_variant == TPSS::PatchVariant::vertex || patch_variant == TPSS::PatchVariant::cell,
         ExcMessage("Invalid patch variant."));
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::initialize(
  const OperatorType &   linear_operator,
  const AdditionalData & additional_data)
{
  // Does nothing.
  (void)linear_operator;
  (void)additional_data;
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::compute_inverses()
{
  const auto & partition_data      = subdomain_handler->get_partition_data();
  const auto   n_subdomain_batches = partition_data.n_subdomains();
  subdomain_to_inverse             = std::make_shared<std::vector<MatrixType>>();
  subdomain_to_inverse->resize(n_subdomain_batches);
  // TODO this is a naive way to bind member functions
  const auto make_assembling = [this](const SubdomainHandler<dim, value_type> &     data,
                                      std::vector<MatrixType> &                     inverses,
                                      const OperatorType &                          operator_,
                                      const std::pair<unsigned int, unsigned int> & range) {
    linear_operator->assemble_subspace_inverses(data, inverses, operator_, range);
  };

  // *** loop over all patches in parallel (regardless of color and constitution)
  subdomain_handler->template parloop<OperatorType, std::vector<MatrixType>>(
    std::ref(make_assembling), *subdomain_to_inverse, *linear_operator);
  AssertThrow(subdomain_to_inverse->size() == partition_data.n_subdomains(),
              ExcMessage("Mismatch."));
  subdomain_to_inverse->shrink_to_fit();
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::clear()
{
  time_data.clear();
  additional_data  = AdditionalData{};
  patch_variant    = TPSS::PatchVariant::invalid;
  smoother_variant = TPSS::SmootherVariant::invalid;
  level            = numbers::invalid_unsigned_int;
  residual_ghosted.reset();
  solution_ghosted.reset();
  subdomain_to_inverse.reset();
  linear_operator = nullptr;
  transfer.reset();
  subdomain_handler.reset();
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::vmult(
  VectorType &       dst,
  const VectorType & src) const
{
  dst = 0.;
  vmult_add(dst, src);
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::Tvmult(
  VectorType &       dst,
  const VectorType & src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::vmult_add(
  VectorType &       dst,
  const VectorType & src) const
{
  Timer timer;
  timer.restart();
  switch(smoother_variant)
  {
    case TPSS::SmootherVariant::additive:
    {
      additive_schwarz_operation(dst, src);
      break;
    }
    case TPSS::SmootherVariant::multiplicative:
    {
      multiplicative_schwarz_operation(dst, src);
      break;
    }
    default:
    {
      Assert(false, ExcMessage("Smoother variant is not supported!"));
      break;
    }
  } // end switch
  time_data.at(4).add_time(timer.wall_time());

  dst *= additional_data.relaxation;
  AssertIsFinite(dst.l2_norm());
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::Tvmult_add(
  VectorType &       dst,
  const VectorType & src) const
{
  switch(smoother_variant)
  {
    case TPSS::SmootherVariant::additive:
    {
      additive_schwarz_operation</*transpose?*/ true>(dst, src);
      break;
    }
    case TPSS::SmootherVariant::multiplicative:
    {
      multiplicative_schwarz_operation</*transpose?*/ true>(dst, src);
      break;
    }
    default:
    {
      Assert(false, ExcMessage("Smoother variant is not supported!"));
      break;
    }
  } // end switch (SmootherVariant)

  dst *= additional_data.relaxation;
  AssertIsFinite(dst.l2_norm());
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template<typename OtherVectorType>
bool
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::is_globally_compatible(
  const OtherVectorType &) const
{
  AssertThrow(false, ExcMessage("VectorType is not supported."));
  return false;
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
bool
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::is_globally_compatible(
  const LinearAlgebra::distributed::Vector<value_type> & vec) const
{
  const auto & partitioner   = subdomain_handler->get_vector_partitioner();
  const bool   is_compatible = vec.partitioners_are_globally_compatible(*partitioner);
  return is_compatible;
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
bool
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::is_globally_compatible(
  const LinearAlgebra::distributed::BlockVector<value_type> & vec) const
{
  bool         is_compatible = true;
  const auto & partitioners  = subdomain_handler->get_vector_partitioners();
  AssertThrow(partitioners.size() == subdomain_handler->n_components(),
              ExcMessage("There are no partitioners."));
  for(unsigned int b = 0; b < subdomain_handler->n_components(); ++b)
  {
    is_compatible =
      is_compatible && vec.block(b).partitioners_are_globally_compatible(*(partitioners[b]));
  }
  return is_compatible;
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template<typename OtherVectorType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::
  initialize_ghosted_vector_if_needed(OtherVectorType &) const
{
  AssertThrow(false, ExcMessage("VectorType is not supported."));
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::
  initialize_ghosted_vector_if_needed(LinearAlgebra::distributed::Vector<value_type> & vec) const
{
  const auto partitioner = subdomain_handler->get_vector_partitioner();
  if(vec.partitioners_are_compatible(*partitioner))
    return;
  vec.reinit(partitioner);
  // std::cout << "initialize ghost on level " << level << std::endl;
}



template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::
  initialize_ghosted_vector_if_needed(
    LinearAlgebra::distributed::BlockVector<value_type> & vec) const
{
  const auto & partitioners = subdomain_handler->get_vector_partitioners();
  if(vec.n_blocks() == subdomain_handler->n_components())
  {
    bool is_compatible = true;
    for(unsigned int b = 0; b < vec.n_blocks(); ++b)
      is_compatible &= vec.block(b).partitioners_are_compatible(*(partitioners[b]));
    if(is_compatible)
      return; //: skip initialization
  }

  //: initialize ghosted vector
  const unsigned int n_components = subdomain_handler->n_components();
  vec.reinit(n_components, /*block_size*/ 0, /*omit_zeroing_entries*/ false);
  for(unsigned int b = 0; b < n_components; ++b)
    vec.block(b).reinit(partitioners[b]);
  /// since the 'block size' has not been set in the reinit() call we have to
  /// finalize the initialization by updating the intrinsic block sizes,
  /// calling collect_sizes()
  vec.collect_sizes();
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template<typename OtherVectorType>
std::pair<OtherVectorType *, const OtherVectorType *>
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::preprocess_solution_and_residual(
  OtherVectorType &       solution_in,
  const OtherVectorType & residual_in) const
{
  Timer              timer;
  VectorType *       solution;
  const VectorType * residual;

  if(is_globally_compatible(solution_in))
  {
    solution = &solution_in;
    residual = &residual_in;
  }
  else // set ghosted vector with write access
  {
    timer.restart();
    initialize_ghosted_vector_if_needed(*solution_ghosted);
    solution_ghosted->zero_out_ghosts();
    TPSS::internal::copy_locally_owned_data(*solution_ghosted, solution_in);
    solution = solution_ghosted.get();

    initialize_ghosted_vector_if_needed(*residual_ghosted);
    TPSS::internal::copy_locally_owned_data(*residual_ghosted, residual_in);
    residual_ghosted->update_ghost_values();
    residual = residual_ghosted.get();
    time_data.at(3).add_time(timer.wall_time());
  }
  return {solution, residual};
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
std::pair<
  Vector<typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> *,
  const Vector<
    typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> *>
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::preprocess_solution_and_residual(
  Vector<typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> &
    solution_in,
  const Vector<
    typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> &
    residual_in) const
{
  return {&solution_in, &residual_in};
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::postprocess_solution(
  Vector<typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> &,
  const Vector<
    typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> &) const
{
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template<typename OtherVectorType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::postprocess_solution(
  OtherVectorType &       solution_dst,
  const OtherVectorType & solution_src) const
{
  Timer timer;
  if(&solution_src == solution_ghosted.get())
  {
    timer.restart();
    TPSS::internal::copy_locally_owned_data(solution_dst, solution_src);
    time_data.at(3).add_time(timer.wall_time());
  }
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::apply_local_solvers(
  VectorType &       solution_in,
  const VectorType & residual_in,
  const unsigned int color) const
{
  // *** pre-process solution and residual vectors (if needed)
  auto [solution, residual] = preprocess_solution_and_residual(solution_in, residual_in);

  // *** apply local inverses for all subdomains of the given color
  const auto apply_inverses_add = [this](const SubdomainHandler<dim, value_type> & data,
                                         VectorType &                              solution,
                                         const VectorType &                        residual,
                                         const std::pair<int, int> & subdomain_range) {
    (void)data;                                                // TODO
    AlignedVector<VectorizedArray<value_type>> local_residual; // r_j
    AlignedVector<VectorizedArray<value_type>> local_solution; // u_j

    for(int patch_id = subdomain_range.first; patch_id < subdomain_range.second; ++patch_id)
    {
      // restrict global to local residual and initialize local solution u_j
      transfer->reinit(patch_id);
      local_residual = std::move(transfer->gather(residual));
      transfer->reinit_local_vector(local_solution);

      // apply local solver u_j = A_loc^{-1} r_j
      const ArrayView<const VectorizedArray<value_type>> local_residual_view =
        make_array_view(local_residual.begin(), local_residual.end());
      const ArrayView<VectorizedArray<value_type>> local_solution_view =
        make_array_view(local_solution.begin(), local_solution.end());
      const auto & inverse = (*subdomain_to_inverse)[patch_id];
      inverse.apply_inverse(local_solution_view, local_residual_view);

      // apply local relaxation parameter
      for(auto & elem : local_solution)
        elem *= make_vectorized_array<value_type>(additional_data.local_relaxation);

      // prolongate and add local solution u_j, that is u += R_j^T u_j
      transfer->scatter_add(solution, local_solution);
    }
  };

  subdomain_handler->template loop<const VectorType, VectorType>(std::ref(apply_inverses_add),
                                                                 *solution,
                                                                 *residual,
                                                                 color);
  /// compress add, i.e. transfer ghost values to their owners
  solution->compress(VectorOperation::add);

  // *** post-process solution (if needed)
  postprocess_solution(solution_in, *solution);
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template<bool transpose>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::additive_schwarz_operation(
  VectorType &       solution,
  const VectorType & rhs) const
{
  AssertThrow(!transpose, ExcMessage("TODO transpose operation is not implemented."));

  // we separate the sum over subdomains with respect to colors avoiding race conditions
  const auto & partition_data = subdomain_handler->get_partition_data();
  const auto   n_colors       = partition_data.n_colors();
  Timer        timer;
  timer.restart();
  for(unsigned int color = 0; color < n_colors; ++color)
    apply_local_solvers(solution, rhs, color);
  time_data[1].add_time(timer.wall_time());
}


template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template<bool transpose>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::multiplicative_schwarz_operation(
  VectorType &       solution,
  const VectorType & rhs) const
{
  AssertThrow(!transpose, ExcMessage("TODO transpose operation is not implemented."));

  const std::vector<unsigned int> color_sequence = get_color_sequence(transpose);
  AssertThrow(!color_sequence.empty(), ExcMessage("There are no colors."));
  // std::cout << std::boolalpha << color_sequence.empty() << std::endl;
  VectorType residual(rhs);
  Timer      timer;

  // apply inverses of first color (no update of residual needed since the initial solution is zero)
  timer.restart();
  apply_local_solvers(solution, rhs, color_sequence.front());
  time_data[1].add_time(timer.wall_time());

  // iterate over remaining colors
  const auto colors_without_first =
    make_array_view(++color_sequence.cbegin(), color_sequence.cend());
  for(const auto color : colors_without_first)
  {
    // *** update residual
    timer.restart();
    this->linear_operator->vmult(residual, solution); // residual = A * u_n
    residual.sadd(-1., 1., rhs);                      // residual = b - A * u_n
    time_data[0].add_time(timer.wall_time());

    // *** apply inverses of given color
    timer.restart();
    apply_local_solvers(solution, residual, color);
    // NOTE a call of apply_inverses includes all colors, such that we add the time to the opened
    // call in additive_schwarz_operation
    time_data[1].time += timer.wall_time();
  }
}
