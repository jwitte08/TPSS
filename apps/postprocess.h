/**
 * postprocess.h
 *
 * collection of postprocess functions
 *
 *  Created on: Nov 13, 2019
 *      Author: witte
 */

#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

struct PostProcessData
{
  std::vector<unsigned int>            n_mg_levels;
  std::vector<unsigned int>            n_cells_global;
  std::vector<types::global_dof_index> n_dofs_global;
  std::vector<unsigned int>            n_colors_system;
  std::vector<double>                  solve_time;
  std::vector<double>                  setup_time;
  std::vector<double>                  n_iterations_system;
  std::vector<double>                  average_reduction_system;
  std::vector<double>                  L2_error;
  std::vector<double>                  H1semi_error;
};

template<int dim, typename VectorType>
void
visualize_dof_vector(const DoFHandler<dim> & dof_handler,
                     const VectorType &      dof_vector,
                     const std::string       name           = "tba",
                     const unsigned int      n_subdivisions = 1,
                     const Mapping<dim> &    mapping        = MappingQGeneric<dim>(1))
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  const auto & tria         = dof_handler.get_triangulation();
  const auto   global_level = tria.n_global_levels() - 1;
  std::string  filename     = name + "_" + Utilities::int_to_string(dim) + "D";

  data_out.add_data_vector(dof_vector, name, DataOut<dim>::type_dof_data);
  data_out.build_patches(mapping,
                         n_subdivisions,
                         DataOut<dim>::CurvedCellRegion::curved_inner_cells);

  const auto filename_per_proc = [filename, global_level](const unsigned int proc_id) {
    std::ostringstream oss;
    oss << filename << "_" << Utilities::int_to_string(proc_id, 4) << "_"
        << Utilities::int_to_string(global_level, 2) << ".vtu";
    return oss.str();
  };

  /// write process-specific paraview file
  {
    std::ofstream file(filename_per_proc(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    data_out.write_vtu(file);
  }

  /// process 0 writes master file
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
      filenames.push_back(filename_per_proc(i));
    std::ostringstream oss;
    oss << filename << "_" << Utilities::int_to_string(global_level, 2) << ".pvtu";
    std::ofstream master_file(oss.str());
    data_out.write_pvtu_record(master_file, filenames);
  }
}

#endif /* POSTPROCESS_H_ */
