
namespace TPSS
{
template<int dim>
std::vector<std::pair<typename PatchInfo<dim>::CellIterator, unsigned int>>
PatchInfo<dim>::extract_relevant_cells(CellIterator cell, const CellIterator end_cell) const
{
  using namespace dealii;

  std::vector<std::pair<CellIterator, unsigned int>> range_storage;

  typedef typename CellIterator::AccessorType CellAccessorType;
  const unsigned int my_pid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // *** step-1: first range contains all locally owned cells
  const auto && is_locally_owned = [my_pid](const CellAccessorType & cell) {
    return (my_pid == cell.level_subdomain_id());
  };

  const CellIterator first_locally_owned_cell = std::find_if(cell, end_cell, is_locally_owned);
  const CellIterator end_locally_owned_cell =
    std::find_if_not(first_locally_owned_cell, end_cell, is_locally_owned);
  const unsigned int n_locally_owned_cells =
    std::distance(first_locally_owned_cell, end_locally_owned_cell);

  range_storage.emplace_back(first_locally_owned_cell, n_locally_owned_cells);

  // *** step-2: collect all ghost cells with a higher subdomain_id
  // TODO required for MPI parallel vertex patches ...

  return range_storage;
}

template<int dim>
void
PatchInfo<dim>::initialize(const dealii::DoFHandler<dim> * dof_handler,
                           const AdditionalData            additional_data_in)
{
  clear();

  // Assert(!(dof_handler->get_triangulation().has_hanging_nodes()),
  //        ExcMessage("Not implemented for adaptive meshes!"));
  Assert(additional_data_in.level != static_cast<unsigned int>(-1),
         ExcMessage("Only implemented for level-dependent meshes!"));
  Assert(additional_data_in.level != static_cast<unsigned int>(-1),
         ExcMessage("Implemented for multigrid meshes!"));

  additional_data    = additional_data_in;
  const auto   level = additional_data_in.level;
  CellIterator cell = dof_handler->begin_mg(level), end_cell = dof_handler->end_mg(level);
  const auto   range_storage      = extract_relevant_cells(cell, end_cell);
  const bool   is_mpi_parallel    = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
  const bool   locally_owns_cells = (range_storage.front().second > 0);
  if(is_mpi_parallel && !locally_owns_cells)
    return; //: nothing to do
  else
    AssertThrow(locally_owns_cells, ExcInternalError());

  // *** submit additional data
  internal_data.level         = level;
  internal_data.range_storage = range_storage;
  internal_data.end_cell_in_storage =
    std::next(range_storage.back().first, range_storage.back().second);

  // *** initialize depending on the patch variant
  if(additional_data.patch_variant == TPSS::PatchVariant::cell)
    initialize_cell_patches(dof_handler, additional_data);
  else if(additional_data.patch_variant == TPSS::PatchVariant::vertex)
    initialize_vertex_patches(dof_handler, additional_data);
  else
    AssertThrow(false, dealii::ExcNotImplemented());
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
      // *** GRAPH COLORING: cells sharing a common face get different colors
      const auto & get_face_conflicts =
        [this, level](const PatchIterator & patch) -> std::vector<types::global_dof_index> {
        AssertDimension(patch->size(), 1);
        const auto &                         cell = patch->front();
        std::vector<types::global_dof_index> conflicts;
        for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          const bool neighbor_has_same_level =
            (cell->neighbor_level(face_no) == static_cast<int>(level));
          const bool neighbor_doesnt_exist = (cell->neighbor_level(face_no) == -1);
          const bool non_adaptive          = neighbor_has_same_level || neighbor_doesnt_exist;
          (void)non_adaptive;
          Assert(non_adaptive, ExcNotImplemented());
          conflicts.emplace_back(cell->face(face_no)->index());
        }
        return conflicts;
      };
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
void
PatchInfo<dim>::initialize_vertex_patches(const dealii::DoFHandler<dim> * dof_handler,
                                          const AdditionalData            additional_data)
{
  using namespace dealii;

  constexpr auto regular_vpatch_size = UniversalInfo<dim>::n_cells(PatchVariant::vertex);
  const auto     level               = additional_data.level;
  const auto     color_scheme        = additional_data.smoother_variant;
  const auto &   range_storage       = internal_data.range_storage;

  // LAMBDA checks if a vertex is at the physical boundary
  auto && is_boundary_vertex = [](const CellIterator & cell, const unsigned int vertex_id) {
    return std::any_of(std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       [&cell](const auto & face_no) { return cell->at_boundary(face_no); });
  };

  // *** check if this proc has locally owned cells
  const unsigned int n_cells_stored =
    std::accumulate(range_storage.cbegin(),
                    range_storage.cend(),
                    0,
                    [](const auto val, const auto range) { return val + range.second; });

  Timer time;
  /**
   * CASE-1 Serial case ...
   */
  if(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    // TODO if we have one irregular patches?
    AssertThrow(n_cells_stored >= regular_vpatch_size,
                ExcMessage("There are not enough cells to construct a regular vertex patch."))
      Assert(range_storage.size() == 1, ExcInternalError()); // only locally owned patches

    std::vector<std::vector<CellIterator>> cell_collections; // temporary vector gathering all
                                                             // subdomains (represented as a
                                                             // vector of CellIterators), i.e.
                                                             // irregular and regular subdomains
    time.restart();

    if(!additional_data.manual_gathering_func)
    {
      const auto & tria     = dof_handler->get_triangulation();
      auto         cell     = range_storage.front().first;
      const auto   end_cell = std::next(range_storage.front().first, range_storage.front().second);


      // PRE-PROCESSING
      std::vector<unsigned int> cell_count(tria.n_vertices(), 0);
      std::vector<unsigned int> vloc_map(tria.n_vertices(), -1);
      unsigned int              vg_max       = 0;
      unsigned int              n_subdomains = 0;

      // *** map each interior vertex (vg, i.e. global index) onto a
      // *** local index (vloc) and count the amount of cells
      // *** belonging to each vertex
      for(; cell != end_cell; ++cell)
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          if(!is_boundary_vertex(cell, v))
          {
            const unsigned int vg = cell->vertex_index(v);
            vg_max                = std::max(vg_max, vg);
            if(vloc_map[vg] == static_cast<unsigned int>(-1))
              vloc_map[vg] = n_subdomains++;
            ++cell_count[vg];
          }
      std::pair<unsigned int, unsigned int> vg_range(
        0, vg_max + 1); // half-open range of global vertex indices to be considered
      cell_count.resize(vg_range.second);
      vloc_map.resize(vg_range.second);

      // *** count the amount of regular and irregular patches
      unsigned int n_regular_patches =
        std::count_if(cell_count.cbegin(), cell_count.cend(), [](const auto & n) {
          return regular_vpatch_size == n;
        });
      unsigned int n_irregular_patches =
        std::count_if(cell_count.cbegin(), cell_count.cbegin() + vg_max + 1, [](const auto & n) {
          return (regular_vpatch_size != n) && (0 != n);
        });
      (void)n_regular_patches, (void)n_irregular_patches;
      AssertDimension(n_subdomains, n_regular_patches + n_irregular_patches);
      // TODO treat irregular patches
      Assert(n_irregular_patches == 0, ExcNotImplemented());

      // *** we gather CellIterators into patches (cell_collections)
      cell_collections.clear();
      cell_collections.resize(n_subdomains);
      for(cell = dof_handler->begin(level); cell != end_cell; ++cell)
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          if(!is_boundary_vertex(cell, v))
          {
            const unsigned int vg   = cell->vertex_index(v);
            const unsigned int vloc = vloc_map[vg];
            Assert(vloc != static_cast<unsigned int>(-1), ExcInternalError());
            auto &     collection = cell_collections[vloc];
            const auto patch_size = cell_count[vg];

            if(collection.empty())
              collection.resize(patch_size);
            if(patch_size == regular_vpatch_size) // regular patch
              collection[regular_vpatch_size - 1 - v] = cell;
            else // irregular patch
              Assert(false, ExcNotImplemented());
          }
    }

    else
      additional_data.manual_gathering_func(dof_handler, additional_data, cell_collections);

    time_data.emplace_back(time.wall_time(), "[PatchInfo] Gathering vertex patches:");
    time.restart();

    // *** coloring for vertex patches
    std::string                             str_coloring_algorithm = "TBA";
    std::vector<std::vector<PatchIterator>> colored_patch_iterators;
    switch(color_scheme)
    {
        // NOTE the overhead we have to pay for a unified submission of
        // patches into the InternalData
      case TPSS::SmootherVariant::additive:
      {
        str_coloring_algorithm = "No";

        // *** fake a first color of PatchIterators
        colored_patch_iterators.clear();
        colored_patch_iterators.resize(1);
        for(auto it = cell_collections.cbegin(); it != cell_collections.cend(); ++it)
          colored_patch_iterators[0].emplace_back(it);

        break;
      }
      case TPSS::SmootherVariant::multiplicative:
      {
        if(!additional_data.manual_coloring_func) // graph coloring
        {
          str_coloring_algorithm = "Graph";

          // LAMBDA conflicts appear if a pair of vertex patches shares a common face
          const auto get_face_conflicts =
            [&](const auto & it) -> std::vector<types::global_dof_index> {
            (void)level;
            const auto & collection = *it;

            std::vector<types::global_dof_index> conflicts;
            for(const auto & cell : collection)
              for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
              {
                Assert(cell->neighbor_level(face_no) == static_cast<int>(level) ||
                         cell->neighbor_level(face_no) == -1,
                       ExcNotImplemented()); // no adaptive refinement allowed
                conflicts.emplace_back(cell->face(face_no)->index());
              }
            return conflicts;
          };

          // *** make graph coloring with respect to a given conflict funtion
          const auto conflict_func_wrapper = std::ref(get_face_conflicts);
          colored_patch_iterators          = std::move(GraphColoring::make_graph_coloring(
            cell_collections.cbegin(), cell_collections.cend(), conflict_func_wrapper));
        }

        else // manual coloring
        {
          str_coloring_algorithm = "Manual";

          colored_patch_iterators = std::move(
            additional_data.manual_coloring_func(dof_handler, additional_data, cell_collections));
        }

        break;
      }
      default:
      {
        AssertThrow(false, ExcNotImplemented());
        break;
      }
    } // end switch

    std::ostringstream ostream;
    ostream << "[PatchInfo] " << str_coloring_algorithm << " coloring (VP):";
    time_data.emplace_back(time.wall_time(), ostream.str());
    time.restart();

    // TODO
    // *** REORDER COLORING: work in progress ...

    // pair is (# of cells, initial color)
    std::vector<std::pair<unsigned int, unsigned int>> reordered_colors;
    if(/*do_color_reordering?*/ false)
      reordered_colors = reorder_colors(colored_patch_iterators);
    else
      for(unsigned int color = 0; color < colored_patch_iterators.size(); ++color)
        reordered_colors.emplace_back(colored_patch_iterators[color].size(), color);

    AssertDimension(colored_patch_iterators.size(), reordered_colors.size());

    // *** we submit the (colored) collections of CellIterators into the InternalData
    const unsigned int n_colors = colored_patch_iterators.size();
    for(unsigned int cc = 0; cc < n_colors; ++cc)
    {
      const unsigned int color = reordered_colors[cc].second;
      AssertDimension(colored_patch_iterators[color].size(), reordered_colors[cc].first);
      submit_patches<regular_vpatch_size>(colored_patch_iterators[color]);
    }

    // *** check if the InternalData is valid
    AssertDimension(internal_data.cell_iterators.size() % regular_vpatch_size, 0);
    AssertDimension(internal_data.n_interior_subdomains.size(),
                    internal_data.n_boundary_subdomains.size());
    AssertDimension(std::accumulate(internal_data.n_interior_subdomains.cbegin(),
                                    internal_data.n_interior_subdomains.cend(),
                                    static_cast<unsigned int>(0)) +
                      std::accumulate(internal_data.n_boundary_subdomains.cbegin(),
                                      internal_data.n_boundary_subdomains.cend(),
                                      static_cast<unsigned int>(0)),
                    internal_data.cell_iterators.size() / regular_vpatch_size);
    if(color_scheme == TPSS::SmootherVariant::additive)
      AssertDimension(internal_data.n_boundary_subdomains.size(), 1);

    if(additional_data.print_details && color_scheme != TPSS::SmootherVariant::additive)
    {
      if(additional_data.manual_coloring_func)
        print_row_variable(
          pcout, 2, "", 43, "Printing manual coloring on level:", additional_data.level);
      else
        print_row_variable(
          pcout, 2, "", 43, "Printing graph coloring on level:", additional_data.level);
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

    // *** OUTPUT PATCHES: output patches per color as SVG-files ***/
    if(false)
      write_visual_data(*dof_handler, reordered_colors);
  }

  /**
   * CASE-2: MPI-parallel case with more than one processor ...
   */
  else
    Assert(false, ExcNotImplemented());
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
  using namespace dealii;

  compute_partition_data(info.subdomain_partition_data, info.get_internal_data());
  const auto & subdomain_partition_data = info.subdomain_partition_data;

  const auto & additional_data = info.get_additional_data();
  const auto   patch_size      = UniversalInfo<dim>::n_cells(additional_data.patch_variant);
  // patch_info                 = &info;
  const auto * internal_data = info.get_internal_data();
  auto &       patch_starts  = info.patch_starts;
  patch_starts.clear();

  const unsigned int stride_irr     = patch_size;
  const unsigned int stride_reg     = patch_size * macro_size;
  unsigned int       start_interior = 0;
  unsigned int       start_boundary = 0;
  for(unsigned int color = 0; color < subdomain_partition_data.n_colors(); ++color)
  {
    start_boundary += internal_data->n_interior_subdomains[color] * patch_size;

    { // interior incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(0, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_interior += stride_irr)
        patch_starts.emplace_back(start_interior);
    }
    // std::cout << "interior start: " << start_interior << std::endl ;

    { // boundary incomplete
      const auto patch_range = subdomain_partition_data.get_patch_range(1, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_boundary += stride_irr)
        patch_starts.emplace_back(start_boundary);
    }
    // std::cout << "boundary start: " << start_boundary  << std::endl ;

    { // interior complete
      const auto patch_range = subdomain_partition_data.get_patch_range(2, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_interior += stride_reg)
        patch_starts.emplace_back(start_interior);
    }
    // std::cout << "interior start: " << start_interior  << std::endl ;
    start_interior += internal_data->n_boundary_subdomains[color] * patch_size;

    { // boundary complete
      const auto patch_range = subdomain_partition_data.get_patch_range(3, color);
      for(unsigned int pp = patch_range.first; pp < patch_range.second;
          ++pp, start_boundary += stride_reg)
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
