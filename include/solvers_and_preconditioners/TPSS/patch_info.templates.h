namespace TPSS
{
template<int dim>
void
PatchInfo<dim>::initialize(const dealii::DoFHandler<dim> * dof_handler,
                           const AdditionalData            additional_data_in)
{
  clear();
  additional_data = additional_data_in;

  Assert(!(dof_handler->get_triangulation().has_hanging_nodes()),
         ExcMessage("Not implemented for adaptive meshes!"));
  Assert(additional_data_in.level != static_cast<unsigned int>(-1),
         ExcMessage("Implemented for level cell iterators!"));
  AssertIndexRange(additional_data_in.level, dof_handler->get_triangulation().n_global_levels());

  // *** submit additional data
  internal_data.level = additional_data.level;

  // *** initialize depending on the patch variant
  if(additional_data.patch_variant == TPSS::PatchVariant::cell)
    initialize_cell_patches(dof_handler, additional_data);
  else if(additional_data.patch_variant == TPSS::PatchVariant::vertex)
    initialize_vertex_patches(dof_handler, additional_data);
  else
    AssertThrow(false, dealii::ExcNotImplemented());

  const auto n_colors_mpimin =
    Utilities::MPI::min(internal_data.n_interior_subdomains.size(), MPI_COMM_WORLD);
  const auto n_colors_mpimax =
    Utilities::MPI::max(internal_data.n_interior_subdomains.size(), MPI_COMM_WORLD);
  (void)n_colors_mpimin, (void)n_colors_mpimax;
  Assert(n_colors_mpimin == n_colors_mpimax,
         ExcMessage("No unified number of colors between mpi-procs."));
  Assert(!internal_data.empty_on_all(), ExcMessage("No mpi-proc owns a patch!"));
}


template<int dim>
void
PatchInfo<dim>::initialize_cell_patches(const dealii::DoFHandler<dim> * dof_handler,
                                        const AdditionalData            additional_data)
{
  const auto level        = additional_data.level;
  const auto color_scheme = additional_data.smoother_variant;

  Timer time;
  time.restart();

  /**
   * Gathering the locally owned cell iterators as collection of cells
   * (patch). Here, it is only one cell iterator per collection.
   */
  const auto locally_owned_range_mg =
    filter_iterators(dof_handler->mg_cell_iterators_on_level(level),
                     IteratorFilters::LocallyOwnedLevelCell());
  std::vector<std::vector<typename DoFHandler<dim>::level_cell_iterator>> cell_collections;
  for(const auto & cell : locally_owned_range_mg)
  {
    std::vector<typename DoFHandler<dim>::level_cell_iterator> patch;
    patch.push_back(cell);
    cell_collections.emplace_back(patch);
  }

  time.stop();
  time_data.emplace_back(time.wall_time(), "Cell-based gathering");
  time.restart();

  /**
   * Coloring of the "cell patches". For the additive operator, we only have one
   * color. However, we require a vector of PatchIterators to call
   * submit_patches.
   */
  std::vector<std::vector<PatchIterator>> colored_iterators;
  constexpr int regular_size = UniversalInfo<dim>::n_cells(PatchVariant::cell);
  if(color_scheme == TPSS::SmootherVariant::additive) // ADDITIVE
  {
    colored_iterators.resize(1);
    std::vector<PatchIterator> & patch_iterators = colored_iterators.front();
    for(auto patch = cell_collections.cbegin(); patch != cell_collections.cend(); ++patch)
      patch_iterators.emplace_back(patch);
  }

  /**
   * Coloring of the "cell patches". For the multiplicative algorithm,
   * one has to prevent that two local solvers sharing a (global)
   * degree of freedom are applied to the same residual vector. For
   * example, DG elements are coupled in terms of the face integrals
   * involving traces of both elements. Therefore, we cells are in
   * conflict if they share a common face. TODO other FE types!
   */
  else if(color_scheme == TPSS::SmootherVariant::multiplicative) // MULTIPLICATIVE
  {
    const bool do_graph_coloring = !additional_data.coloring_func;
    if(do_graph_coloring) // graph coloring
    {
      const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
      AssertThrow(!is_mpi_parallel,
                  ExcMessage("Graph coloring is not compatible with distributed triangulations."));
      colored_iterators = std::move(GraphColoring::make_graph_coloring(cell_collections.cbegin(),
                                                                       cell_collections.cend(),
                                                                       get_face_conflicts));
    }

    else // user-defined coloring
    {
      colored_iterators =
        std::move(additional_data.coloring_func(cell_collections, additional_data));
    }
  }

  else // color_scheme
    AssertThrow(false, ExcNotImplemented());

  time.stop();
  time_data.emplace_back(time.wall_time(), "Cell-based coloring");
  time.restart();

  const unsigned int n_colors = colored_iterators.size();
  for(unsigned int color = 0; color < n_colors; ++color)
    submit_patches<regular_size>(colored_iterators[color]);

  time.stop();
  time_data.emplace_back(time.wall_time(), "Submit cell-based patches");
  time.restart();

  // *** check if the InternalData is valid
  AssertDimension(internal_data.n_interior_subdomains.size(),
                  internal_data.n_boundary_subdomains.size());
  if(color_scheme == TPSS::SmootherVariant::additive)
    AssertDimension(internal_data.n_boundary_subdomains.size(), 1);
  const unsigned int n_interior_subdomains =
    std::accumulate(internal_data.n_interior_subdomains.cbegin(),
                    internal_data.n_interior_subdomains.cend(),
                    0);
  const unsigned int n_boundary_subdomains =
    std::accumulate(internal_data.n_boundary_subdomains.cbegin(),
                    internal_data.n_boundary_subdomains.cend(),
                    0);
  const unsigned int n_subdomains = n_interior_subdomains + n_boundary_subdomains;
  (void)n_subdomains;
  AssertDimension(n_subdomains, internal_data.cell_iterators.size());

  if(additional_data.visualize_coloring)
    additional_data.visualize_coloring(*dof_handler, colored_iterators, "cp_");

  // *** print detailed information
  if(additional_data.print_details)
  {
    print_row_variable(pcout, 45, "Coloring on level:", additional_data.level);
    print_row_variable(
      pcout, 5, "", 10, "color:", 30, "# of interior patches:", 30, "# of boundary patches:");
    const auto n_colors   = internal_data.n_interior_subdomains.size();
    auto       n_interior = internal_data.n_interior_subdomains.cbegin();
    auto       n_boundary = internal_data.n_boundary_subdomains.cbegin();
    for(unsigned c = 0; c < n_colors; ++c, ++n_interior, ++n_boundary)
      print_row_variable(pcout, 5, "", 10, c, 30, *n_interior, 30, *n_boundary);
    pcout << std::endl;
  }
}

template<int dim>
std::vector<std::vector<typename PatchInfo<dim>::CellIterator>>
PatchInfo<dim>::gather_vertex_patches(const DoFHandler<dim> & dof_handler,
                                      const AdditionalData &  additional_data) const
{
  const unsigned int level = additional_data.level;

  // LAMBDA checks if a vertex is at the physical boundary
  auto && is_boundary_vertex = [](const CellIterator & cell, const unsigned int vertex_id) {
    return std::any_of(std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       [&cell](const auto & face_no) { return cell->at_boundary(face_no); });
  };
  constexpr unsigned int regular_vpatch_size = 1 << dim;
  const auto &           tria                = dof_handler.get_triangulation();
  const auto             locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                     IteratorFilters::LocallyOwnedLevelCell());

  /**
   * A mapping @p global_to_local_map between the global vertex and
   * the pair containing the number of locally owned cells and the
   * number of all cells (including ghosts) is constructed
   */
  std::map<unsigned int, std::pair<unsigned int, unsigned int>> global_to_local_map;
  for(const auto & cell : locally_owned_range_mg)
  {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      if(!is_boundary_vertex(cell, v))
      {
        const unsigned int global_index = cell->vertex_index(v);
        const auto         element      = global_to_local_map.find(global_index);
        if(element != global_to_local_map.cend())
        {
          ++(element->second.first);
          ++(element->second.second);
        }
        else
        {
          const auto n_cells_pair = std::pair<unsigned, unsigned>{1, 1};
          const auto status =
            global_to_local_map.insert(std::make_pair(global_index, n_cells_pair));
          (void)status;
          Assert(status.second, ExcMessage("failed to insert key-value-pair"))
        }
      }
  }

  /**
   * Ghost patches are stored as the mapping @p global_to_ghost_id
   * between the global vertex index and GhostPatch. The number of
   * cells, book-kept in @p global_to_local_map, is updated taking the
   * ghost cells into account.
   */
  // TODO: is_ghost_on_level() missing
  const auto not_locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                     [](const auto & cell) { return !(cell->is_locally_owned_on_level()); });
  std::map<unsigned int, GhostPatch> global_to_ghost_id;
  for(const auto & cell : not_locally_owned_range_mg)
  {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      const unsigned int global_index = cell->vertex_index(v);
      const auto         element      = global_to_local_map.find(global_index);
      if(element != global_to_local_map.cend())
      {
        ++(element->second.second);
        const unsigned int subdomain_id_ghost = cell->level_subdomain_id();
        const auto         ghost              = global_to_ghost_id.find(global_index);
        if(ghost != global_to_ghost_id.cend())
          ghost->second.submit_id(subdomain_id_ghost, cell->id());
        else
        {
          const auto status =
            global_to_ghost_id.emplace(global_index, GhostPatch(subdomain_id_ghost, cell->id()));
          (void)status;
          Assert(status.second, ExcMessage("failed to insert key-value-pair"));
        }
      }
    }
  }

  { // ASSIGN GHOSTS
    const unsigned int my_subdomain_id = tria.locally_owned_subdomain();
    /**
     * logic: if the mpi-proc owns more than half of the cells within
     *        a ghost patch he takes ownership
     */
    {
      //: (1) add subdomain_ids of locally owned cells to GhostPatches
      for(const auto & cell : locally_owned_range_mg)
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const unsigned global_index = cell->vertex_index(v);
          const auto     ghost        = global_to_ghost_id.find(global_index);
          //: checks if the global vertex has ghost cells attached
          if(ghost != global_to_ghost_id.end())
            ghost->second.submit_id(my_subdomain_id, cell->id());
        }

      std::set<unsigned> to_be_owned;
      std::set<unsigned> to_be_erased;
      for(const auto key_value : global_to_ghost_id)
      {
        const unsigned int global_index     = key_value.first;
        const auto &       proc_to_cell_ids = key_value.second.proc_to_cell_ids;

        const auto & get_proc_with_most_cellids = [](const auto & lhs, const auto & rhs) {
          const std::vector<CellId> & cell_ids_lhs = lhs.second;
          const std::vector<CellId> & cell_ids_rhs = rhs.second;
          Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
          Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
          return (cell_ids_lhs.size() < cell_ids_rhs.size());
        };

        const auto         most                       = std::max_element(proc_to_cell_ids.cbegin(),
                                           proc_to_cell_ids.cend(),
                                           get_proc_with_most_cellids);
        const unsigned int subdomain_id_most          = most->first;
        const unsigned int n_locally_owned_cells_most = most->second.size();
        const auto         member                     = global_to_local_map.find(global_index);
        Assert(member != global_to_local_map.cend(), ExcMessage("must be listed as patch"));
        const unsigned int n_cells = member->second.second;
        if(my_subdomain_id == subdomain_id_most)
        {
          AssertDimension(member->second.first, n_locally_owned_cells_most);
          if(2 * n_locally_owned_cells_most > n_cells)
            to_be_owned.insert(global_index);
        }
        else
        {
          if(2 * n_locally_owned_cells_most > n_cells)
            to_be_erased.insert(global_index);
        }
      }

      for(const unsigned global_index : to_be_owned)
      {
        auto & my_patch = global_to_local_map[global_index];
        my_patch.first  = my_patch.second;
        global_to_ghost_id.erase(global_index);
      }
      for(const unsigned global_index : to_be_erased)
      {
        global_to_local_map.erase(global_index);
        global_to_ghost_id.erase(global_index);
      }
    }

    /**
     * logic: the owner of the cell with the lowest CellId takes ownership
     */
    {
      //: (2) determine mpi-proc with the minimal CellId for all GhostPatches
      std::set<unsigned> to_be_owned;
      for(const auto key_value : global_to_ghost_id)
      {
        const unsigned int global_index     = key_value.first;
        const auto &       proc_to_cell_ids = key_value.second.proc_to_cell_ids;

        const auto & get_proc_with_min_cellid = [](const auto & lhs, const auto & rhs) {
          std::vector<CellId> cell_ids_lhs = lhs.second;
          Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
          std::sort(cell_ids_lhs.begin(), cell_ids_lhs.end());
          const auto          min_cell_id_lhs = cell_ids_lhs.front();
          std::vector<CellId> cell_ids_rhs    = rhs.second;
          Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
          std::sort(cell_ids_rhs.begin(), cell_ids_rhs.end());
          const auto min_cell_id_rhs = cell_ids_rhs.front();
          return min_cell_id_lhs < min_cell_id_rhs;
        };

        const auto min = std::min_element(proc_to_cell_ids.cbegin(),
                                          proc_to_cell_ids.cend(),
                                          get_proc_with_min_cellid);

        const unsigned int subdomain_id_min = min->first;
        if(my_subdomain_id == subdomain_id_min)
          to_be_owned.insert(global_index);
      }

      //: (3) set owned GhostPatches in global_to_local_map and delete all remaining
      for(const unsigned global_index : to_be_owned)
      {
        auto & my_patch = global_to_local_map[global_index];
        my_patch.first  = my_patch.second;
        global_to_ghost_id.erase(global_index);
      }
      for(const auto key_value : global_to_ghost_id)
      {
        const unsigned int global_index = key_value.first;
        global_to_local_map.erase(global_index);
      }
    }
  }

  /**
   * Enumerate the patches contained in @p global_to_local_map by
   * replacing the former number of locally owned cells in terms of a
   * consecutive numbering. The local numbering is required for
   * gathering the level cell iterators into a collection @
   * cell_collections according to the global vertex index.
   */
  unsigned int local_index = 0;
  for(auto & key_value : global_to_local_map)
  {
    key_value.second.first = local_index++;
  }
  const unsigned n_subdomains = global_to_local_map.size();
  AssertDimension(n_subdomains, local_index);
  std::vector<std::vector<CellIterator>> cell_collections;
  cell_collections.resize(n_subdomains);
  for(auto & cell : dof_handler.mg_cell_iterators_on_level(level))
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      const unsigned int global_index = cell->vertex_index(v);
      const auto         element      = global_to_local_map.find(global_index);
      if(element != global_to_local_map.cend())
      {
        const unsigned int local_index = element->second.first;
        const unsigned int patch_size  = element->second.second;
        auto &             collection  = cell_collections[local_index];
        if(collection.empty())
          collection.resize(patch_size);
        if(patch_size == regular_vpatch_size) // regular patch
          collection[regular_vpatch_size - 1 - v] = cell;
        else // irregular patch
          AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
      }
    }

  return cell_collections;
}

template<int dim>
void
PatchInfo<dim>::initialize_vertex_patches(const dealii::DoFHandler<dim> * dof_handler,
                                          const AdditionalData            additional_data)
{
  constexpr auto regular_vpatch_size = UniversalInfo<dim>::n_cells(PatchVariant::vertex);
  const auto     color_scheme        = additional_data.smoother_variant;

  Timer time;
  time.restart();

  /**
   * Collecting the cell iterators attached to a vertex. See @p
   * gather_vertex_patches for more information.
   */
  std::vector<std::vector<CellIterator>> cell_collections;
  if(!additional_data.manual_gathering_func)
    cell_collections = std::move(gather_vertex_patches(*dof_handler, additional_data));
  else
    additional_data.manual_gathering_func(dof_handler, additional_data, cell_collections);

  time.stop();
  time_data.emplace_back(time.wall_time(), "Vertex patch gathering");
  time.restart();

  /**
   * Coloring of vertex patches. For the additive operator, we only
   * require one color as long as we do not use thread-parallelism
   * (TODO). In multi-threaded loops, race-conditions might occur due
   * to overlapping cells: two local solvers sharing a common cell
   * might simultaneously write to the same DoF entry in the
   * destination vector. Therefore, two vertex patches are in conflict
   * if they share a common cell.
   */
  AssertThrow(MultithreadInfo::n_threads() == 1, ExcMessage("TODO"));
  std::string                             str_coloring_algorithm = "TBA";
  std::vector<std::vector<PatchIterator>> colored_iterators;
  switch(color_scheme)
  {
    case TPSS::SmootherVariant::additive:
    {
      str_coloring_algorithm = "none";
      colored_iterators.resize(1);
      auto & patch_iterators = colored_iterators.front();
      for(auto it = cell_collections.cbegin(); it != cell_collections.cend(); ++it)
        patch_iterators.emplace_back(it);
      break;
    }
    case TPSS::SmootherVariant::multiplicative:
    {
      if(!additional_data.coloring_func) // graph coloring
      {
        const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
        AssertThrow(!is_mpi_parallel,
                    ExcMessage(
                      "Graph coloring is not compatible with distributed triangulations."));
        str_coloring_algorithm = "graph";
        colored_iterators = std::move(GraphColoring::make_graph_coloring(cell_collections.cbegin(),
                                                                         cell_collections.cend(),
                                                                         get_face_conflicts));
      }

      else // user-defined coloring
      {
        str_coloring_algorithm = "user";
        colored_iterators =
          std::move(additional_data.coloring_func(cell_collections, additional_data));
      }
      break;
    }
    default:
    {
      AssertThrow(false, ExcNotImplemented());
      break;
    }
  } // end switch

  // TODO
  // *** REORDER COLORING: work in progress ...
  // pair is (# of cells, initial color)
  std::vector<std::pair<unsigned int, unsigned int>> reordered_colors;
  if(/*do_color_reordering?*/ false)
    reordered_colors = reorder_colors(colored_iterators);
  else
    for(unsigned int color = 0; color < colored_iterators.size(); ++color)
      reordered_colors.emplace_back(colored_iterators[color].size(), color);
  AssertDimension(colored_iterators.size(), reordered_colors.size());

  std::ostringstream oss;
  oss << "Vertex patch coloring (" << str_coloring_algorithm << ")";
  time.stop();
  time_data.emplace_back(time.wall_time(), oss.str());
  time.restart();

  if(additional_data.visualize_coloring)
    additional_data.visualize_coloring(*dof_handler, colored_iterators, "vp_");

  /**
   * Submisson of the colored collections of CellIterators into the
   * InternalData.
   */
  const unsigned int n_colors = colored_iterators.size();
  for(unsigned int cc = 0; cc < n_colors; ++cc)
  {
    const unsigned int color = reordered_colors[cc].second;
    AssertDimension(colored_iterators[color].size(), reordered_colors[cc].first);
    submit_patches<regular_vpatch_size>(colored_iterators[color]);
  }

  time.stop();
  time_data.emplace_back(time.wall_time(), "Submit vertex patches");
  time.restart();

  // *** check if the InternalData is valid
  AssertDimension(internal_data.cell_iterators.size() % regular_vpatch_size, 0);
  AssertDimension(internal_data.n_interior_subdomains.size(),
                  internal_data.n_boundary_subdomains.size());
  if(color_scheme == TPSS::SmootherVariant::additive)
    AssertDimension(internal_data.n_boundary_subdomains.size(), 1);
  const unsigned int n_interior_subdomains =
    std::accumulate(internal_data.n_interior_subdomains.cbegin(),
                    internal_data.n_interior_subdomains.cend(),
                    0);
  const unsigned int n_boundary_subdomains =
    std::accumulate(internal_data.n_boundary_subdomains.cbegin(),
                    internal_data.n_boundary_subdomains.cend(),
                    0);
  const unsigned int n_subdomains = n_interior_subdomains + n_boundary_subdomains;
  (void)n_subdomains;
  AssertDimension(n_subdomains, internal_data.cell_iterators.size() / regular_vpatch_size);

  if(additional_data.print_details && color_scheme != TPSS::SmootherVariant::additive)
  {
    print_row_variable(pcout, 2, "", 43, oss.str(), additional_data.level);
    pcout << std::endl;

    print_row_variable(
      pcout, 5, "", 10, "color:", 30, "# of interior patches:", 30, "# of boundary patches:");
    const auto n_colors   = internal_data.n_interior_subdomains.size();
    auto       n_interior = internal_data.n_interior_subdomains.cbegin();
    auto       n_boundary = internal_data.n_boundary_subdomains.cbegin();
    for(unsigned c = 0; c < n_colors; ++c, ++n_interior, ++n_boundary)
      print_row_variable(pcout, 5, "", 10, c, 30, *n_interior, 30, *n_boundary);
    pcout << std::endl;
  }
}

template<int dim>
std::vector<std::pair<unsigned int, unsigned int>>
PatchInfo<dim>::reorder_colors(
  const std::vector<std::vector<typename std::vector<std::vector<CellIterator>>::const_iterator>> &
    colored_cells) const
{
  const unsigned int level = internal_data.level;
  (void)level;
  Assert(level != static_cast<unsigned int>(-1), dealii::ExcInvalidState());

  const auto n_colors = colored_cells.size();
  std::vector<std::pair<unsigned int, unsigned int>>
    count_per_color; // temporary field storing the amount of patches per color
  for(unsigned int color = 0; color < n_colors; ++color)
    count_per_color.emplace_back(colored_cells[color].size(), color);

  std::vector<unsigned int> new_order; // set new order
  // // if (!permuted_colors->empty() && colored_cells.size () == permuted_colors->at(level).size
  // ())
  // //   new_order = permuted_colors->at(level);
  // if (permuted_colors->size () == 1 && colored_cells.size () == permuted_colors->at (0).size ())
  //   new_order = permuted_colors->front ();
  // if (permuted_colors->size () == 2 && colored_cells.size () == permuted_colors->at (0).size ()
  //     && colored_cells.size () == permuted_colors->at (1).size ())
  // {
  //   if (*activate_post_smooth)
  //     new_order = permuted_colors->at (1);
  //   else
  //     new_order = permuted_colors->at (0);
  // }

  if(n_colors != new_order.size()) // leave if new_order does not match
    return count_per_color;

  AssertDimension(colored_cells.size() - 1,
                  *std::max_element(new_order.cbegin(), new_order.cend()));

  const auto & do_color_reorder = [&](const auto & p1, const auto & p2) {
    const auto dist1 =
      std::distance(new_order.cbegin(), std::find(new_order.cbegin(), new_order.cend(), p1.second));
    const auto dist2 =
      std::distance(new_order.cbegin(), std::find(new_order.cbegin(), new_order.cend(), p2.second));
    return dist1 < dist2;
  };
  std::sort(count_per_color.begin(), count_per_color.end(), do_color_reorder);

  return count_per_color;
}

template<int dim>
void
PatchInfo<dim>::write_visual_data(
  const dealii::DoFHandler<dim> &                            dof_handler,
  const std::vector<std::pair<unsigned int, unsigned int>> & reordered_colors) const
{
  using namespace dealii;

  constexpr auto     regular_vpatch_size = UniversalInfo<dim>::n_cells(PatchVariant::vertex);
  const auto &       tria                = dof_handler.get_triangulation();
  const unsigned int level               = internal_data.level;

  if(level == tria.n_levels() - 1)
  {
    GridOutFlags::Svg gridout_flags;
    gridout_flags.coloring           = GridOutFlags::Svg::Coloring::material_id;
    gridout_flags.label_level_number = false;
    gridout_flags.label_cell_index   = false;
    gridout_flags.label_material_id  = true;
    GridOut gridout;
    gridout.set_flags(gridout_flags);

    // *** set all material ids to void
    const unsigned int void_id = 999;
    CellIterator       cell = dof_handler.begin_mg(level), end_cell = dof_handler.end_mg(level);
    for(; cell != end_cell; ++cell)
      cell->set_material_id(void_id);

    auto               cell_it  = internal_data.cell_iterators.cbegin();
    const unsigned int n_colors = reordered_colors.size();
    for(unsigned int color = 0; color < n_colors; ++color)
    {
      std::string filename = "make_graph_coloring_";
      filename             = filename + "L" + Utilities::int_to_string(level) + "_COLOR" +
                 Utilities::int_to_string(color, 2);
      std::ofstream output((filename + ".svg").c_str());

      unsigned int n_colored_cells = reordered_colors[color].first * regular_vpatch_size;
      for(unsigned int c = 0; c < n_colored_cells; ++cell_it, ++c)
        (*cell_it)->set_material_id(color);
      gridout.write_svg(tria, output);

      // *** reset all material ids to void
      CellIterator cell = dof_handler.begin_mg(level), end_cell = dof_handler.end_mg(level);
      for(; cell != end_cell; ++cell)
        cell->set_material_id(void_id);
    }
    Assert(internal_data.cell_iterators.cend() == cell_it, ExcInternalError());
  }
}

template<int dim, typename number>
void
PatchWorker<dim, number>::partition_patches(PatchInfo<dim> & info)
{
  compute_partition_data(info.subdomain_partition_data, info.get_internal_data());
  const auto & subdomain_partition_data = info.subdomain_partition_data;

  const auto & additional_data = info.get_additional_data();
  const auto   patch_size      = UniversalInfo<dim>::n_cells(additional_data.patch_variant);
  const auto * internal_data   = info.get_internal_data();
  (void)internal_data;
  auto & patch_starts = info.patch_starts;
  patch_starts.clear();

  const unsigned int stride_incomp = patch_size;
  const unsigned int stride_comp   = patch_size * macro_size;
  unsigned int       start         = 0;
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(0, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second; ++pp)
      {
        patch_starts.emplace_back(start);
        start += stride_incomp;
      }
    }

    { // interior complete
      const auto patch_range = subdomain_partition_data.get_patch_range(2, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second; ++pp)
      {
        patch_starts.emplace_back(start);
        start += stride_comp;
      }
    }
  }
  AssertDimension(start, internal_data->cell_iterators.size());
  AssertDimension(patch_starts.size(), subdomain_partition_data.n_subdomains());
  patch_starts.emplace_back(start); // endpoint required by n_lanes_filled()

  info.n_lanes_filled.clear();
  info.n_lanes_filled.reserve(subdomain_partition_data.n_subdomains());
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(0, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
        info.n_lanes_filled.emplace_back(1);
    }
    { // boundary incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(1, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
        std::cout << "BDRY INCP" << std::endl;
    }
    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(2, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
        info.n_lanes_filled.emplace_back(macro_size);
    }
    { // boundary complete
      const auto patch_range = subdomain_partition_data.get_patch_range(3, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
        std::cout << "BDRY CP" << std::endl;
    }
  }
  AssertDimension(info.n_lanes_filled.size(), subdomain_partition_data.n_subdomains());

  // TODO treat all macro_cells at the patch boundary instead of one representative
  const auto get_mask = [](auto &&            macro_cell,
                           const unsigned int direction,
                           const unsigned int face_no_1d) /* -> unsigned int*/ {
    std::bitset<macro_size> bitset_mask;
    for(unsigned int vv = 0; vv < macro_size; ++vv)
      bitset_mask[vv] = macro_cell[vv]->face(2 * direction + face_no_1d)->at_boundary();
    return bitset_mask;
  };

  info.at_boundary_mask.clear();
  info.at_boundary_mask.reserve(subdomain_partition_data.n_subdomains() *
                                GeometryInfo<dim>::faces_per_cell);
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(0, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
      {
        //: face_no < direction
        std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> masks;
        const auto cell_collection{std::move(get_cell_collection(patch_id))};
        for(unsigned int d = 0; d < dim; ++d)
        {
          masks[d * 2]     = get_mask(cell_collection.front(), d, /*face_no*/ 0);
          masks[d * 2 + 1] = get_mask(cell_collection.back(), d, /*face_no*/ 1);
        }
        for(const auto & mask : masks)
          info.at_boundary_mask.emplace_back(static_cast<unsigned int>(mask.to_ulong()));
      }
    }

    { // interior complete
      const auto patch_range = subdomain_partition_data.get_patch_range(2, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
      {
        // TODO
        // info.n_lanes_filled.emplace_back(macro_size);
        // const auto cell_collection{std::move(get_cell_collection(patch_id))};

        //: face_no < direction
        std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> masks;
        const auto cell_collection{std::move(get_cell_collection(patch_id))};
        for(unsigned int d = 0; d < dim; ++d)
        {
          masks[d * 2]     = get_mask(cell_collection.front(), d, /*face_no*/ 0);
          masks[d * 2 + 1] = get_mask(cell_collection.back(), d, /*face_no*/ 1);
        }
        for(const auto & mask : masks)
          info.at_boundary_mask.emplace_back(static_cast<unsigned int>(mask.to_ulong()));
      }
    }
  }
  AssertDimension(info.at_boundary_mask.size(),
                  subdomain_partition_data.n_subdomains() * GeometryInfo<dim>::faces_per_cell);
}

template<int dim, typename number>
void
PatchWorker<dim, number>::connect_to_matrixfree(MatrixFreeConnect<dim, number> & mf_connect)
{
  Assert(patch_info != nullptr, ExcNotInitialized());

  const auto &       mf_storage     = *(mf_connect.mf_storage);
  const unsigned int n_cell_batches = mf_storage.n_cell_batches();
  const auto &       internal_data  = *(patch_info->get_internal_data());
  const unsigned     level          = internal_data.level;
  const auto &       dof_handler    = mf_storage.get_dof_handler();
  const auto &       tria           = dof_handler.get_triangulation();
  //: N of locally stored cell iterators (including ghost and artificial)
  const bool         proc_has_cells_on_level = (level < tria.n_levels());
  const unsigned int n_cells_stored          = proc_has_cells_on_level ? tria.n_cells(level) : 0;

  /**
   * Each process enumerates its owned, ghosted & artificial
   * cells. The cell index is accessible via
   * TriaIterator::index(level) on each level. The maximal cell index
   * is bounded by the number of cells, Triangulation::n_cells(level).
   *
   * In the MatrixFree framework cells are stored as batches due to
   * vectorization. In the following, we map cells identified by its
   * cell index to their counterpart in the MatrixFree object
   * identified by the pair of batch index (bindex) and vectorization
   * lane (bcomp).
   */
  std::vector<std::pair<unsigned int, unsigned int>> cindex_to_bindex_bcomp_pair;
  cindex_to_bindex_bcomp_pair.resize(n_cells_stored); // we don't care about the accurate size
  for(unsigned int bid = 0; bid < n_cell_batches; ++bid)
    for(unsigned int comp = 0; comp < macro_size; ++comp)
      if(comp < mf_storage.n_components_filled(bid))
      {
        const auto         cell = mf_storage.get_cell_iterator(bid, comp, /*dof_handler_index*/ 0);
        const unsigned int cindex = cell->index();
        AssertIndexRange(cindex, n_cells_stored);
        cindex_to_bindex_bcomp_pair[cindex] = std::make_pair(bid, comp);
      }

  /**
   * For each cell iterator we store the associated macro cell (batch)
   * and the vectorization lane representing the same cell in the
   * MatrixFree framework.
   */
  const auto & cell_iterators   = patch_info->get_internal_data()->cell_iterators;
  auto &       bindex_and_bcomp = mf_connect.batch_and_lane;
  bindex_and_bcomp.clear();
  bindex_and_bcomp.reserve(cell_iterators.size());
  for(const auto & cell : cell_iterators)
  {
    const auto cindex = cell->index();
    bindex_and_bcomp.emplace_back(cindex_to_bindex_bcomp_pair[cindex]);
  }
  AssertDimension(bindex_and_bcomp.size(), cell_iterators.size());
}
} // end namespace TPSS
