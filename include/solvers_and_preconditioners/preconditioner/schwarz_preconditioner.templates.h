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
  using namespace dealii;

  // *** reset members to uninitialized state
  clear();

  // *** set time information
  time_data = {TimeInfo{0., "[SchwarzPrecond] Update residuals:", "[s]", 0},
               TimeInfo{0., "[SchwarzPrecond] Apply inverses:", "[s]", 0},
               TimeInfo{0., "[SchwarzPrecond] Compute inverses:", "[s]", 0}};

  // *** initialization of members
  subdomain_handler     = subdomain_handler_in;
  differential_operator = &operator_in;
  transfer        = std::make_shared<typename OperatorType::transfer_type>(*subdomain_handler);
  additional_data = additional_data_in;
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
  using namespace dealii;

  // *** reset members to uninitialized state
  clear();

  // *** set time information
  time_data = {TimeInfo{0., "[SchwarzPrecond] Update residuals:", "[s]", 0},
               TimeInfo{0., "[SchwarzPrecond] Apply inverses:", "[s]", 0},
               TimeInfo{0., "[SchwarzPrecond] Compute inverses:", "[s]", 0}};

  // *** initialization of members
  subdomain_handler     = schwarz_preconditioner_in.subdomain_handler;
  transfer              = schwarz_preconditioner_in.transfer;
  differential_operator = schwarz_preconditioner_in.differential_operator;
  inverses_owned        = schwarz_preconditioner_in.inverses_owned;
  subdomain_to_inverse  = inverses_owned.get();
  additional_data       = additional_data_in;
  level                 = schwarz_preconditioner_in.level;
  patch_variant         = schwarz_preconditioner_in.patch_variant;
  smoother_variant      = schwarz_preconditioner_in.smoother_variant;
  Assert(additional_data.relaxation > 0., ExcInvalidState());
  Assert(patch_variant == TPSS::PatchVariant::vertex || patch_variant == TPSS::PatchVariant::cell,
         dealii::ExcNotImplemented());
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::compute_inverses()
{
  const auto & partition_data      = subdomain_handler->get_partition_data();
  const auto   n_subdomain_batches = partition_data.n_subdomains();
  inverses_owned                   = std::make_shared<std::vector<MatrixType>>();
  inverses_owned->resize(n_subdomain_batches);
  subdomain_to_inverse = inverses_owned.get();
  // TODO this is a naive way to bind member functions
  const auto make_assembling = [this](const SubdomainHandler<dim, value_type> &     data,
                                      std::vector<MatrixType> &                     inverses,
                                      const OperatorType &                          operator_,
                                      const std::pair<unsigned int, unsigned int> & range) {
    differential_operator->assemble_subspace_inverses(data, inverses, operator_, range);
  };

  // *** loop over all patches in parallel (regardless of color and constitution)
  AssertThrow(subdomain_to_inverse->size() == partition_data.n_subdomains(),
              ExcMessage("Mismatch."));
  subdomain_handler->template parloop<OperatorType, std::vector<MatrixType>>(
    std::ref(make_assembling), *inverses_owned, *differential_operator);
  AssertThrow(subdomain_to_inverse->size() == partition_data.n_subdomains(),
              ExcMessage("Mismatch."));
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::clear()
{
  subdomain_handler.reset();
  differential_operator = nullptr;
  inverses_owned.reset();
  subdomain_to_inverse = nullptr;
  additional_data      = AdditionalData{};
  level                = static_cast<unsigned int>(-1);
  time_data.clear();
}

// template<int dim, class OperatorType, typename VectorType, typename MatrixType>
// void
// SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::vmult(
//   LinearAlgebra::distributed::Vector<
//     typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> & dst,
//   const LinearAlgebra::distributed::Vector<
//     typename SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::value_type> & src)
//     const
// {
//   vmult<LinearAlgebra::distributed::Vector<value_type>>(dst, src);
// }

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
  using namespace dealii;

  adjust_ghost_range_if_necessary(dst);
  adjust_ghost_range_if_necessary(src);

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
  } // end switch (SmootherVariant)
  dst.compress(dealii::VectorOperation::add);

  dst *= additional_data.relaxation;
  AssertIsFinite(dst.l2_norm());
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::Tvmult_add(
  VectorType &       dst,
  const VectorType & src) const
{
  using namespace dealii;

  switch(smoother_variant)
  {
    case TPSS::SmootherVariant::additive:
    {
      additive_schwarz_operation</*VectorType,*/ /*transpose?*/ true>(dst, src);
      break;
    }
    case TPSS::SmootherVariant::multiplicative:
    {
      multiplicative_schwarz_operation</*VectorType,*/ /*transpose?*/ true>(dst, src);
      break;
    }
    default:
    {
      Assert(false, ExcMessage("Smoother variant is not supported!"));
      break;
    }
  } // end switch (SmootherVariant)
  dst.compress(dealii::VectorOperation::add);

  dst *= additional_data.relaxation;
  AssertIsFinite(dst.l2_norm());
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::update(
  LinearOperatorBase const * matrix_operator)
{
  OperatorType const * differential_operator = dynamic_cast<OperatorType const *>(matrix_operator);

  if(differential_operator)
  {
    // TODO differential_operator does update
    Assert(false, ExcNotImplemented());
  }
  else
    AssertThrow(false, ExcMessage("UnderlyingOperator and MatrixOperator are not compatible!"));
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
// template<typename VectorType>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::apply_subdomain_inverses_add(
  VectorType &       solution,
  const VectorType & residual,
  const unsigned int color) const
{
  using namespace dealii;

  Assert(this->patch_variant == TPSS::PatchVariant::vertex ||
           this->patch_variant == TPSS::PatchVariant::cell,
         dealii::ExcNotImplemented()); // supported: additive, multiplicative

  const auto apply_inverses_lambda = [this](const SubdomainHandler<dim, value_type> & data,
                                            VectorType &                              solution,
                                            const VectorType &                        residual,
                                            const std::pair<int, int> & subdomain_range) {
    (void)data;                                                // TODO
    AlignedVector<VectorizedArray<value_type>> local_residual; // r_loc
    AlignedVector<VectorizedArray<value_type>> local_solution; // x_loc

    for(int patch_id = subdomain_range.first; patch_id < subdomain_range.second; ++patch_id)
    {
      // *** restrict global residual (gather): r_loc = R_loc residual
      // ***                                    where R_loc is restriction operator
      transfer->reinit(patch_id);
      local_residual = std::move(transfer->gather(residual));

      // *** initialize local solution: x_loc
      transfer->reinit_local_vector(local_solution);

      // *** apply local inverse: x_loc = A_loc^{-1} r_loc
      const ArrayView<const VectorizedArray<value_type>> local_residual_view =
        make_array_view(local_residual.begin(), local_residual.end());
      const ArrayView<VectorizedArray<value_type>> local_solution_view =
        make_array_view(local_solution.begin(), local_solution.end());
      const auto & inverse = (*subdomain_to_inverse)[patch_id];
      inverse.apply_inverse(local_solution_view, local_residual_view);

      // *** apply local relaxation parameter
      for(auto & elem : local_solution)
        elem *= make_vectorized_array<value_type>(additional_data.local_relaxation);

      // *** prolongate and add local solution: solution += R_loc^T x_loc
      // ***                                    where R_loc^T is prolongation operator
      transfer->scatter_add(solution, local_solution);
    }
  };

  // *** loop over all patches of given color !
  subdomain_handler->template loop<const VectorType, VectorType>(std::ref(apply_inverses_lambda),
                                                                 solution,
                                                                 residual,
                                                                 color);
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template</*typename VectorType,*/ bool transpose>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::additive_schwarz_operation(
  VectorType &       solution,
  const VectorType & rhs) const
{
  AssertThrow(!transpose, ExcMessage("Transpose operation is not implemented."));

  Timer timer;
  timer.restart();

  const auto & partition_data = subdomain_handler->get_partition_data();
  for(unsigned int color = 0; color < partition_data.n_colors(); ++color)
    apply_subdomain_inverses_add(solution, rhs, color);

  time_data[1].add_time(timer.wall_time());
}

template<int dim, class OperatorType, typename VectorType, typename MatrixType>
template</*typename VectorType,*/ bool transpose>
void
SchwarzPreconditioner<dim, OperatorType, VectorType, MatrixType>::multiplicative_schwarz_operation(
  VectorType &       solution,
  const VectorType & rhs) const
{
  AssertThrow(!transpose, ExcMessage("Transpose operation is not implemented."));

  /*** LAMBDA implementation of the multiplicative algorithm dependent ***/
  const auto get_color_sequence = [this]() {
    std::vector<unsigned int> color_sequence;
    const auto & partition_data = subdomain_handler->get_patch_info().subdomain_partition_data;
    const auto   n_colors       = partition_data.n_colors();

    // *** fill color sequence forwards
    for(unsigned int color = 0; color < n_colors; ++color)
      color_sequence.push_back(color); // 0, 1, ..., n_colors-1

    // *** reverse the color sequence
    const bool revert_colors = transpose || additional_data.reverse;
    if(revert_colors)
      std::reverse(color_sequence.begin(), color_sequence.end()); // n_colors-1, ..., 1, 0

    // *** skip first color and traverse backwards through colors
    if(additional_data.symmetrized && !color_sequence.empty())
    {
      const auto temp = color_sequence;
      auto       c    = ++temp.rbegin(); // skip last color (avoids duplicate!)
      for(; c != temp.rend(); ++c)
        color_sequence.push_back(*c);
      // std::cout << "color_sequence: ";
      // for (const auto c : color_sequence)
      //   std::cout << c << "  ";
      // std::cout << std::endl;
    }

    AssertDimension(additional_data.symmetrized ? 2 * n_colors - 1 : n_colors,
                    color_sequence.size());

    return color_sequence;
  };

  auto       color_sequence = get_color_sequence();
  VectorType residual{rhs};
  Timer      timer;

  // // DEBUG
  // std::cout << "color seq: ";
  // for (const auto c : color_sequence)
  //   std::cout << c << " ";
  // std::cout << std::endl;

  // *** apply the patch inverses to the first color
  timer.restart();
  if(!color_sequence.empty())
  {
    apply_subdomain_inverses_add(solution, rhs, color_sequence.front());
    color_sequence.erase(color_sequence.begin());
  }
  time_data[1].add_time(timer.wall_time());

  for(const auto color : color_sequence)
  {
    timer.restart();

    // *** update residual
    this->differential_operator->vmult(residual, solution); // residual = A*u_n
    residual.sadd(-1., 1., rhs);                            // residual = b - A*u_n

    time_data[0].add_time(timer.wall_time());
    timer.restart();

    // *** apply inverses of given color
    apply_subdomain_inverses_add(solution,
                                 residual,
                                 color); // solution = solution + A_color^{-1} * residual

    // NOTE a call of apply_inverses includes all colors, such that
    // we add the time to the opened call in
    // additive_schwarz_operation
    time_data[1].time += timer.wall_time();
  }
}
