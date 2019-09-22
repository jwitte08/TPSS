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
  // std::ostringstream     oss;
  // oss << "process " << tria.locally_owned_subdomain() << " reports:\n";
  const auto locally_owned_range_mg =
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
          Assert(status.second, ExcMessage("failed to insert key-value-pair"));
        }
      }
    }
  }

  // oss << "ghost patches (raw):" << std::endl;
  // for(const auto key_value : global_to_ghost_id)
  //   oss << key_value.first /*<< ", " << key_value.second.str()*/ << "   ";
  // oss << std::endl;
  // oss << "global-to-local mapping (raw):" << std::endl;
  // for(const auto key_value : global_to_local_map)
  //   oss << key_value.first << ", (" << key_value.second.first << "," << key_value.second.second
  //       << ")   ";
  // oss << std::endl;

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

      // oss << "ghost patches (more than half):" << std::endl;
      // for(const auto key_value : global_to_ghost_id)
      //   oss << key_value.first /*<< ", " << key_value.second.str()*/ << "   ";
      // oss << std::endl;
      // oss << "global-to-local mapping (more than half):" << std::endl;
      // for(const auto key_value : global_to_local_map)
      //   oss << key_value.first << ", (" << key_value.second.first << "," <<
      //   key_value.second.second
      //       << ")   ";
      // oss << std::endl;
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

    // oss << "ghost patches (final):" << std::endl;
    // for(const auto key_value : global_to_ghost_id)
    //   oss << key_value.first /*<< ", " << key_value.second.str()*/ << "   ";
    // oss << std::endl;
    // oss << "global-to-local mapping (final):" << std::endl;
    // for(const auto key_value : global_to_local_map)
    //   oss << key_value.first << ", (" << key_value.second.first << "," << key_value.second.second
    //       << ")   ";
    // oss << std::endl;
  }

  // /**
  //  * Check if regular vertex patch exists otherwise exit. TODO irregular patch
  //  */
  // const auto & has_regular_size = [](const auto & key_value){
  //   const auto n_cells_per_patch = key_value.second.second;
  //   return (n_cells_per_patch == regular_vpatch_size);
  // };
  // const bool regular_patch_exists = std::any_of(global_to_local_map.cbegin(),
  // global_to_local_map.cend(), has_regular_size); if (!regular_patch_exists)
  //   {
  //     return std::vector<std::vector<CellIterator>>{};
  //   }

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

  // std::cout << oss.str() << "\n\n";
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
  auto &       patch_starts    = info.patch_starts;
  patch_starts.clear();

  const unsigned int stride_incomp  = patch_size;
  const unsigned int stride_comp    = patch_size * macro_size;
  unsigned int       start_interior = 0;
  unsigned int       start_boundary = 0;
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    start_boundary += internal_data->n_interior_subdomains[color] * patch_size;

    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(0, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_interior += stride_incomp)
        patch_starts.emplace_back(start_interior);
    }
    // std::cout << "interior start: " << start_interior << std::endl ;

    { // boundary incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(1, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_boundary += stride_incomp)
        patch_starts.emplace_back(start_boundary);
    }
    // std::cout << "boundary start: " << start_boundary  << std::endl ;

    { // interior complete
      const auto patch_range = subdomain_partition_data.get_patch_range(2, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_interior += stride_comp)
        patch_starts.emplace_back(start_interior);
    }
    // std::cout << "interior start: " << start_interior  << std::endl ;
    start_interior += internal_data->n_boundary_subdomains[color] * patch_size;

    { // boundary complete
      const auto patch_range = subdomain_partition_data.get_patch_range(3, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_boundary += stride_comp)
        patch_starts.emplace_back(start_boundary);
    }
    // std::cout << "boundary start: " << start_boundary  << std::endl ;

    AssertDimension(start_interior, start_boundary);
  }
  AssertDimension(info.patch_starts.size(), subdomain_partition_data.n_subdomains());

  /*** setup up the is_interior flag for the given PartitionData ***/
  const auto get_mask = [](auto &&            macro_cell,
                           const unsigned int direction,
                           const unsigned int face_no_1d) /* -> unsigned short*/ {
    // std::cout << "macro cell indices: ";
    // for (unsigned int vv = 0; vv < macro_size; ++vv)
    //   std::cout << macro_cell[vv]->index() << " ";
    // std::cout << std::endl;
    // std::cout << "faces: ";
    // for (unsigned int ff = 0; ff < GeometryInfo<dim>::faces_per_cell; ++ff)
    //   for (unsigned int vv = 0; vv < macro_size; ++vv)
    // 	std::cout << macro_cell[vv]->face(ff)->at_boundary() << " ";
    // std::cout << std::endl;

    std::bitset<macro_size> bitset_mask;
    for(unsigned int vv = 0; vv < macro_size; ++vv)
      bitset_mask[vv] = macro_cell[vv]->face(2 * direction + face_no_1d)->at_boundary();

    // std::cout << "bitset mask: direction/face_no  " << direction << "/" << face_no_1d << ":  ";
    // for (unsigned int vv = 0; vv < macro_size; ++vv)
    //   std::cout << bitset_mask[vv] << " ";
    // std::cout << std::endl;
    // std::cout << std::endl;

    //    return static_cast<unsigned short> (bitset_mask.to_ulong ());
    return bitset_mask;
  };

  info.is_interior_patch.reserve(subdomain_partition_data.n_subdomains());
  info.is_incomplete_patch.reserve(subdomain_partition_data.n_subdomains());
  info.at_boundary_mask.reserve(subdomain_partition_data.n_subdomains() *
                                GeometryInfo<dim>::faces_per_cell);
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(0, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
      {
        info.is_interior_patch.emplace_back(true);
        info.is_incomplete_patch.emplace_back(true);
        const auto cell_collection{std::move(get_cell_collection(patch_id))};

        for(unsigned int mm = 0; mm < GeometryInfo<dim>::faces_per_cell; ++mm)
          info.at_boundary_mask.emplace_back(0);

        // DEBUG
        for(unsigned int d = 0; d < dim; ++d)
        {
          AssertDimension(static_cast<unsigned long>(0),
                          get_mask(cell_collection.front(), d, 0).to_ulong());
          AssertDimension(static_cast<unsigned long>(0),
                          get_mask(cell_collection.back(), d, 1).to_ulong());
        }
      }
    }

    { // boundary incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(1, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
      {
        info.is_interior_patch.emplace_back(false);
        info.is_incomplete_patch.emplace_back(true);
        std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> local_data;
        const auto cell_collection{std::move(get_cell_collection(patch_id))};

        for(unsigned int d = 0; d < dim; ++d)
        {
          local_data[d * 2]     = get_mask(cell_collection.front(), d, 0 /*face_no*/);
          local_data[d * 2 + 1] = get_mask(cell_collection.back(), d, 1 /*face_no*/);
        }

        for(auto && mask : local_data)
        {
          info.at_boundary_mask.emplace_back(static_cast<unsigned short>(mask.to_ulong()));
          Assert(info.at_boundary_mask.back() == (Utilities::pow(2, macro_size) - 1) ||
                   info.at_boundary_mask.back() == 0,
                 ExcDimensionMismatch2(info.at_boundary_mask.back(),
                                       0,
                                       (Utilities::pow(2, macro_size) - 1)));
        }
      }
    }

    { // interior complete
      const auto patch_range = subdomain_partition_data.get_patch_range(2, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
      {
        info.is_interior_patch.emplace_back(true);
        info.is_incomplete_patch.emplace_back(false);
        const auto cell_collection{std::move(get_cell_collection(patch_id))};

        for(unsigned int mm = 0; mm < GeometryInfo<dim>::faces_per_cell; ++mm)
          info.at_boundary_mask.emplace_back(0);

        // DEBUG
        for(unsigned int d = 0; d < dim; ++d)
        {
          AssertDimension(static_cast<unsigned long>(0),
                          get_mask(cell_collection.front(), d, 0).to_ulong());
          AssertDimension(static_cast<unsigned long>(0),
                          get_mask(cell_collection.back(), d, 1).to_ulong());
        }
      }
    }

    { // boundary complete
      const auto patch_range = subdomain_partition_data.get_patch_range(3, color);
      for(unsigned int patch_id = patch_range.first; patch_id < patch_range.second; ++patch_id)
      {
        info.is_interior_patch.emplace_back(false);
        info.is_incomplete_patch.emplace_back(false);
        std::array<std::bitset<macro_size>, GeometryInfo<dim>::faces_per_cell> local_data;
        const auto cell_collection{std::move(get_cell_collection(patch_id))};

        for(unsigned int d = 0; d < dim; ++d)
        {
          local_data[d * 2]     = get_mask(cell_collection.front(), d, 0 /*face_no*/);
          local_data[d * 2 + 1] = get_mask(cell_collection.back(), d, 1 /*face_no*/);
        }

        for(auto && mask : local_data)
          info.at_boundary_mask.emplace_back(static_cast<unsigned short>(mask.to_ulong()));
      }
    }
  }
  AssertDimension(info.is_interior_patch.size(), subdomain_partition_data.n_subdomains());
  AssertDimension(info.is_incomplete_patch.size(), subdomain_partition_data.n_subdomains());
  AssertDimension(info.at_boundary_mask.size(),
                  subdomain_partition_data.n_subdomains() * GeometryInfo<dim>::faces_per_cell);
}

template<int dim, typename number>
void
PatchWorker<dim, number>::connect_to_matrixfree(MatrixFreeConnect<dim, number> & mf_connect)
{
  using namespace dealii;

  Assert(patch_info != nullptr, ExcNotInitialized());
  if(patch_info->empty())
    return; //: nothing to do

  // const auto & additional_data = patch_info->get_additional_data();
  // std::cout << patch_size << "  " << TPSS::Info<dim>::n_cells.at(additional_data.patch_variant)
  // << std::endl; patch_size = TPSS::Info<dim>::n_cells.at(additional_data.patch_variant);
  mf_connect.stride_triple = patch_size * macro_size;

  std::vector<std::pair<unsigned int, unsigned int>> mf_cell_indices;
  const auto &                                       mf_storage = *(mf_connect.mf_storage);
  const unsigned int                                 n_batches  = mf_storage.n_cell_batches();
  const auto &   internal_data = *(patch_info->get_internal_data());
  const unsigned level         = internal_data.level;
  const auto &   dof_handler   = mf_storage.get_dof_handler();
  const auto &   tria          = dof_handler.get_triangulation();
  //: N of locally stored cell iterators (including ghost and artificial)
  const unsigned int n_cells_stored = tria.n_cells(level);

  /*** map each cell-index to its MatrixFree counterpart, namely the bcomp'th lane of the batch
   * given by the batch-index bindex ***/
  std::vector<std::pair<unsigned int, unsigned int>> cindex_to_bindex_bcomp_pair;
  cindex_to_bindex_bcomp_pair.resize(n_cells_stored); // we don't care about the accurate size
  for(unsigned int bid = 0; bid < n_batches; ++bid)
    for(unsigned int comp = 0; comp < macro_size; ++comp)
      if(comp < mf_storage.n_components_filled(bid))
      {
        // TODO quicker access to cell-level-index field within MatrixFree object ??
        const unsigned int cindex{mf_storage.get_cell_index(bid, comp)};
        AssertIndexRange(cindex, n_cells_stored);
        cindex_to_bindex_bcomp_pair[cindex] = std::make_pair(bid, comp);
      }

  /*** map (batch-index,batch-lane) pairs to its corresponding CellIterator within a patch by
   * storing them in the same order we stride through the subdomain_partition_data. Note that within
   * a patch the cell-number runs faster than the vec-lane and so do the (batch-index, batch-lane)
   * pairs as well! ***/
  const auto & patch_starts             = patch_info->patch_starts;
  const auto & cell_iterators           = patch_info->get_internal_data()->cell_iterators;
  const auto & subdomain_partition_data = patch_info->subdomain_partition_data;
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    { // interior incomplete
      const auto range = subdomain_partition_data.get_patch_range(0, color);
      if(range.first < range.second)
      {
        auto start   = patch_starts[range.first];
        auto cell_it = cell_iterators.cbegin() + start;
        for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
          for(unsigned int cc = 0; cc < patch_size; ++cc, ++cell_it)
            mf_cell_indices.emplace_back(cindex_to_bindex_bcomp_pair[(*cell_it)->index()]);
      }
    }
    { // boundary incomplete
      const auto range = subdomain_partition_data.get_patch_range(1, color);
      if(range.first < range.second)
      {
        auto cell_it = std::next(cell_iterators.cbegin(), patch_starts.at(range.first));
        for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
          for(unsigned int cc = 0; cc < patch_size; ++cc, ++cell_it)
            mf_cell_indices.emplace_back(cindex_to_bindex_bcomp_pair[(*cell_it)->index()]);
      }
    }
    { // interior complete
      const auto range = subdomain_partition_data.get_patch_range(2, color);
      if(range.first < range.second)
      {
        auto cell_it = std::next(cell_iterators.cbegin(), patch_starts.at(range.first));
        for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
          for(unsigned int cc = 0; cc < macro_size * patch_size; ++cc, ++cell_it)
            mf_cell_indices.emplace_back(cindex_to_bindex_bcomp_pair[(*cell_it)->index()]);
      }
    }
    { // boundary complete
      const auto range = subdomain_partition_data.get_patch_range(3, color);
      if(range.first < range.second)
      {
        auto cell_it = std::next(cell_iterators.cbegin(), patch_starts.at(range.first));
        for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
          for(unsigned int cc = 0; cc < macro_size * patch_size; ++cc, ++cell_it)
            mf_cell_indices.emplace_back(cindex_to_bindex_bcomp_pair[(*cell_it)->index()]);
      }
    }
  }

  AssertDimension(mf_cell_indices.size(), cell_iterators.size());

  const unsigned int n_patches          = subdomain_partition_data.n_subdomains();
  auto &             batch_starts       = mf_connect.batch_starts;
  auto &             batch_count_per_id = mf_connect.batch_count_per_id;
  auto &             bcomp_vcomp_cindex = mf_connect.bcomp_vcomp_cindex;
  batch_starts.clear();
  batch_count_per_id.clear();
  bcomp_vcomp_cindex.clear();
  batch_starts.reserve(n_patches);
  batch_count_per_id.reserve(n_patches);
  bcomp_vcomp_cindex.reserve(macro_size * n_patches);

  /*** Last, we gather batch-indices that occur more than once within a patch. That is
   * each batch-index is mapped to its associated (batch-lane, cell-lane, local-cell-number)
   * triples. The count of those triples and the batch-index itself is stored in batch_count_per_id.
   * Similar to the patch_starts the batch_starts point to the first batch-index(batch_count_per_id)
   * of patch represented by patch-index. ***/
  auto mf_cell_index = mf_cell_indices.cbegin();
  for(unsigned int patch_id = 0; patch_id < n_patches; ++patch_id)
  {
    batch_starts.emplace_back(batch_count_per_id.size());
    std::vector<std::pair<unsigned int, std::array<unsigned int, 3>>> map_bindex_to_triple;

    // *** duplicate triples along the artificial lanes
    if(patch_info->is_incomplete_patch[patch_id]) // INCOMPLETE
      for(unsigned int cidx = 0; cidx < patch_size; ++cidx, ++mf_cell_index)
        for(unsigned int vcomp = 0; vcomp < macro_size; ++vcomp)
          map_bindex_to_triple.emplace_back(mf_cell_index->first,
                                            std::array<unsigned int, 3>{
                                              {mf_cell_index->second, vcomp, cidx}});
    else // COMPLETE
      for(unsigned int vcomp = 0; vcomp < macro_size; ++vcomp)
        for(unsigned int cidx = 0; cidx < patch_size; ++cidx, ++mf_cell_index)
          map_bindex_to_triple.emplace_back(mf_cell_index->first,
                                            std::array<unsigned int, 3>{
                                              {mf_cell_index->second, vcomp, cidx}});
    AssertDimension(patch_size * macro_size, map_bindex_to_triple.size());

    /*** gather triples associated to the same batch-index and store their counts ***/
    std::sort(map_bindex_to_triple.begin(),
              map_bindex_to_triple.end(),
              [](const auto & trip1, const auto & trip2) { return trip1.first < trip2.first; });
    auto mapped_triple = map_bindex_to_triple.cbegin();
    while(mapped_triple != map_bindex_to_triple.cend())
    {
      const unsigned int batch_id{mapped_triple->first};
      unsigned int       count{0};
      auto               it = mapped_triple;
      while(it != map_bindex_to_triple.cend())
      {
        bcomp_vcomp_cindex.emplace_back(it->second);
        ++count;
        ++it;
        ++mapped_triple;
        it = std::find_if(it, map_bindex_to_triple.cend(), [batch_id](const auto & pair) {
          return batch_id == pair.first;
        });
      }
      batch_count_per_id.emplace_back(batch_id, count);
    }

    const auto &   sum_counts = [](const auto val, const auto & p) { return val + p.second; };
    const unsigned n_counts_accumulated = std::accumulate(
      batch_count_per_id.cbegin() + batch_starts.back(), batch_count_per_id.cend(), 0, sum_counts);
    (void)n_counts_accumulated;
    AssertDimension(patch_size * macro_size, n_counts_accumulated);
  } // patch loop
  batch_starts.emplace_back(batch_count_per_id.size());

  // // DEBUG
  // for(auto e : batch_starts)
  //   std::cout << e << " ";
  // std::cout << std::endl;
  // for(auto e : batch_count_per_id)
  //   std::cout << e.first << "," << e.second << " ";
  // std::cout << std::endl;
  // for(auto e : bcomp_vcomp_cindex)
  //   std::cout << e[0] << "," << e[1] << "," << e[2] << " ";
  // std::cout << std::endl;

  // this->batch_starts       = &mf_connect.batch_starts;
  // this->batch_count_per_id = &mf_connect.batch_count_per_id;
  // this->bcomp_vcomp_cindex = &mf_connect.bcomp_vcomp_cindex;
  // Assert (mf_connect.batch_starts != nullptr, ExcNotInitialized ());
  // Assert (mf_connect.batch_count_per_id != nullptr, ExcNotInitialized ());
  // Assert (mf_connect.bcomp_vcomp_cindex != nullptr, ExcNotInitialized ());
}

template<int dim, typename number>
void
PatchWorker<dim, number>::initialize(const PatchInfo<dim> & info)
{
  using namespace dealii;

  typename PatchInfo<dim>::PartitionData subdomain_partition_data;
  compute_partition_data(subdomain_partition_data, info.get_internal_data());
  Assert(subdomain_partition_data.check_compatibility(info.subdomain_partition_data),
         ExcMessage(
           "The PatchInfo object passed is corrupt. Subdomain partitioning does not fit the "
           "InternalData!"));
}

template<int dim, typename number>
void
PatchWorker<dim, number>::clear_patch_info(PatchInfo<dim> & info)
{
  info.patch_starts.clear();
  info.is_interior_patch.clear();
  info.is_incomplete_patch.clear();
  info.at_boundary_mask.clear();
  info.subdomain_partition_data.clear();
}

template<int dim, typename number>
void
PatchWorker<dim, number>::clear_mf_connect(MatrixFreeConnect<dim, number> & mf_connect)
{
  mf_connect.bcomp_vcomp_cindex.clear();
  mf_connect.batch_count_per_id.clear();
  mf_connect.batch_starts.clear();
  mf_connect.stride_triple = -1;
}

// template<int dim, typename number>
// void
// PatchWorker<dim, number>::initialize(PatchInfo<dim> &                 info,
//                                                      MatrixFreeConnect<dim, number> & mf_connect)
// {
//   using namespace dealii;

//   //const auto & additional_data = info.get_additional_data();
//   //std::cout << patch_size << "  " << TPSS::Info<dim>::n_cells.at(additional_data.patch_variant)
//   << std::endl;
//   //patch_size = TPSS::Info<dim>::n_cells.at(additional_data.patch_variant);

//   // clear existing data in PatchInfo and MatrixFreeConnect
//   clear_patch_info(info);
//   clear_mf_connect(mf_connect);

//   // If we do not locally own cells nothing has to be initialized
//   auto internal_data = info.get_internal_data();
//   if(internal_data->cell_iterators.size() == 0)
//     return;

//   // Distribute the (non-vectorized) patches, stored in PatchInfo, and
//   // distribute them according to the given vectorization length.
//   // Partitions are stored in PatchInfo::subdomain_partition_data and
//   // the associated vectorized strides are set in
//   // PatchInfo::patch_starts
//   partition_patches(info);

//   // In a last step we provide a mapping between cells stored in the
//   // macro patches, distributed above, and its MatrixFree
//   // counterpart, the so-called cell batches.
//   connect_to_matrixfree(mf_connect);
// }

} // end namespace TPSS
