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

#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/data_postprocessor.h>



struct PostProcessData
{
  std::vector<double>                  average_reduction_system;
  unsigned int                         n_dimensions = 0;
  std::vector<double>                  H2semi_error;
  std::vector<double>                  H1semi_error;
  std::vector<double>                  L2_error;
  std::vector<unsigned int>            n_cells_global;
  std::vector<unsigned int>            n_colors_system;
  std::vector<types::global_dof_index> n_dofs_global;
  std::vector<double>                  n_iterations_system;
  std::vector<unsigned int>            n_mg_levels;
  std::vector<double>                  setup_time;
  std::vector<double>                  solve_time;
};



/**
 * compute the average reduction rho over n iterations and the
 * fractional number of iterations to achieve the requested
 * reduction (relative stopping criterion)
 */
std::pair<double, double>
compute_fractional_steps(const ReductionControl & solver_control)
{
  const double residual_0 = solver_control.initial_value();
  const double residual_n = solver_control.last_value();
  const int    n          = solver_control.last_step(); // number of iterations
  const double reduction  = solver_control.reduction(); // relative tolerance

  // *** average reduction: r_n = rho^n * r_0
  const double rho = std::pow(residual_n / residual_0, static_cast<double>(1. / n));

  /**
   * since r_n <= reduction * r_0 we can compute the fractional
   * number of iterations n_frac that is sufficient to achieve the
   * desired reduction:
   *    rho^n_frac = reduction   <=>   n_frac = log(reduction)/log(rho)
   */
  const double n_frac = std::log(reduction) / std::log(rho);

  /// n_frac should definitely not be larger than n. this can happen if the
  /// iterative solver is stopped before the relative tolerance is reached.
  AssertThrow(residual_n / residual_0 < reduction,
              ExcLowerRangeType(reduction, residual_n / residual_0));
  AssertThrow(n_frac <= (double)n, ExcLowerRangeType((double)n, n_frac));
  /// if the reduction of the last step n is above average it might happen that
  /// n_frac is smaller than (n-1). the subsequent assert should warn us if
  /// n_frac is even smaller than (n-2).
  AssertThrow((double)(n - 2) <= n_frac, ExcLowerRangeType(n_frac, (double)(n - 2)));

  return std::make_pair(n_frac, rho);
}


template<int dim, typename VectorType>
void
visualize_dof_vector(const DoFHandler<dim> & dof_handler,
                     const VectorType &      dof_vector,
                     const std::string       prefix         = "tba",
                     const unsigned int      n_subdivisions = 1,
                     const Mapping<dim> &    mapping        = MappingQGeneric<dim>(1),
                     const std::string       suffix         = "")
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  const auto & tria         = dof_handler.get_triangulation();
  const auto   global_level = tria.n_global_levels() - 1;
  std::string  filename     = prefix + "_" + Utilities::int_to_string(dim) + "D";

  data_out.add_data_vector(dof_vector, prefix, DataOut<dim>::type_dof_data);
  data_out.build_patches(mapping,
                         n_subdivisions,
                         DataOut<dim>::CurvedCellRegion::curved_inner_cells);

  const auto filename_per_proc = [&](const unsigned int proc_id) {
    std::ostringstream oss;
    oss << filename << "_" << Utilities::int_to_string(proc_id, 4) << "_"
        << Utilities::int_to_string(global_level, 2) << suffix << ".vtu";
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
    oss << filename << "_" << Utilities::int_to_string(global_level, 2) << suffix << ".pvtu";
    std::ofstream master_file(oss.str());
    data_out.write_pvtu_record(master_file, filenames);
  }
}

template<int dim, typename VectorType>
void
visualize_dof_vector(const DoFHandler<dim> &        dof_handler,
                     const VectorType &             dof_vector,
                     const std::vector<std::string> names,
                     const std::string              prefix         = "tba",
                     const std::string              suffix         = "",
                     const unsigned int             n_subdivisions = 1,
                     std::vector<DataComponentInterpretation::DataComponentInterpretation>
                                          data_component_interpretation = {},
                     const Mapping<dim> & mapping                       = MappingQGeneric<dim>(1))
{
  AssertThrow(!names.empty(), ExcMessage("Insert names for each vector component."));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  const auto & tria         = dof_handler.get_triangulation();
  const auto   global_level = tria.n_global_levels() - 1;
  std::string  filename     = prefix + "_" + Utilities::int_to_string(dim) + "D";

  data_out.add_data_vector(dof_vector,
                           names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches(mapping,
                         n_subdivisions,
                         DataOut<dim>::CurvedCellRegion::curved_inner_cells);

  const auto filename_per_proc = [filename, global_level, suffix](const unsigned int proc_id) {
    std::ostringstream oss;
    oss << filename << "_" << Utilities::int_to_string(proc_id, 4) << "_"
        << Utilities::int_to_string(global_level, 2) << (suffix != "" ? "_" : "") << suffix
        << ".vtu";
    return oss.str();
  };

  /// write process-specific paraview file
  {
    std::ofstream file(filename_per_proc(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    data_out.write_vtu(file);
  }

  /// process 0 writes master file
  if(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::vector<std::string> filenames;
      for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
        filenames.push_back(filename_per_proc(i));
      std::ostringstream oss;
      oss << filename << "_" << Utilities::int_to_string(global_level, 2)
          << (suffix != "" ? "_" : "") << suffix << ".pvtu";
      std::ofstream master_file(oss.str());
      data_out.write_pvtu_record(master_file, filenames);
    }
}

template<int dim, typename VectorType>
void
visualize_cell_vector(const DoFHandler<dim> & dof_handler,
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

  data_out.add_data_vector(dof_vector, name, DataOut<dim>::type_cell_data);
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



namespace Biharmonic
{
/**
 * Cell-wise visualization of the vector curl of a stream function.
 */
template<int dim>
class StreamVelocityPP : public DataPostprocessorVector<dim>
{
  static_assert(dim == 2, "implemented for 2D");

public:
  StreamVelocityPP() : DataPostprocessorVector<dim>("stream_velocity", update_gradients)
  {
  }

  virtual void
  evaluate_scalar_field(const DataPostprocessorInputs::Scalar<dim> & input_data,
                        std::vector<Vector<double>> & computed_quantities) const override
  {
    AssertDimension(input_data.solution_gradients.size(), computed_quantities.size());

    const auto n_q_points = input_data.solution_gradients.size();
    for(auto q = 0U; q < n_q_points; ++q)
    {
      const auto & grad_phi = input_data.solution_gradients[q];
      auto &       curl_phi = computed_quantities[q];
      AssertDimension(curl_phi.size(), dim);
      curl_phi[0] = grad_phi[1];
      curl_phi[1] = -grad_phi[0];
    }
  }
};



/**
 * Cell-wise visualization of the (reconstructed) pressure.
 */
template<int dim>
class PressurePP : public DataPostprocessorScalar<dim>
{
public:
  PressurePP() : DataPostprocessorScalar<dim>("pressure", update_values)
  {
  }

  virtual void
  evaluate_scalar_field(const DataPostprocessorInputs::Scalar<dim> & input_data,
                        std::vector<Vector<double>> & computed_quantities) const override
  {
    AssertDimension(input_data.solution_values.size(), computed_quantities.size());
    const auto n_q_points = input_data.solution_values.size();
    for(auto q = 0U; q < n_q_points; ++q)
    {
      AssertDimension(input_data.solution_values.size(), 1U);
      AssertDimension(input_data.solution_values.size(), computed_quantities[q].size());
      computed_quantities[q] = input_data.solution_values[q];
    }
  }
};

} // namespace Biharmonic



#endif /* POSTPROCESS_H_ */
