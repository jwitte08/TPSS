
namespace TPSS
{
template<int dim, int fe_order, int n_q_points_1d, typename number>
template<bool add>
inline void
MatrixAssembler<dealii::internal::EvaluatorVariant::evaluate_general,
                dim,
                fe_order,
                n_q_points_1d,
                number>::apply(const number * ansatz_data,
                               const number * test_data,
                               const number * dx,
                               number *       matrix)
{
  const int stride = n_q_points_1d;

  // TODO do we need AlignedVector here?
  // number test_data_dx[n_q_points_1d*fe_order] ;
  dealii::AlignedVector<number> temp;
  temp.resize_fast(n_q_points_1d * fe_order);
  for(int i = 0; i < fe_order; ++i)
    for(int q = 0; q < n_q_points_1d; ++q)
      temp[i * stride + q] = test_data[i * stride + q] * dx[q];
  number * test_data_dx = temp.begin();

  for(int j = 0; j < fe_order; ++j)
    for(int i = 0; i < fe_order; ++i)
    {
      number m_ji;
      m_ji = 0.;
      for(int q = 0; q < n_q_points_1d; ++q)
        m_ji += ansatz_data[i * stride + q] * test_data_dx[j * stride + q];
      if(add)
        *matrix += m_ji;
      else
        *matrix = m_ji;
      ++matrix;
    }
}

} // end namespace TPSS
