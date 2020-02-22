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
  std::vector<unsigned int> n_mg_levels;
  std::vector<unsigned int> n_cells_global;
  std::vector<unsigned int> n_dofs_global;
  std::vector<unsigned int> n_colors_system;
  std::vector<double>       solve_time;
  std::vector<double>       setup_time;
  std::vector<double>       n_iterations_system;
  std::vector<double>       average_reduction_system;
  std::vector<double>       L2_error;
  std::vector<double>       H1semi_error;
};

#endif /* POSTPROCESS_H_ */
