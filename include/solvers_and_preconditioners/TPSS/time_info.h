/*
 * time_info.h
 *
 *  Created on: Aug 11, 2018
 *      Author: witte
 */

#ifndef TIMEINFO_H_
#define TIMEINFO_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/vectorization.h>

#include <iomanip>
#include <string>
#include <type_traits>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

using namespace dealii;

// TODO rename TimeInfo struct into Stamp within a namespace TimeInfo
struct TimeInfo
{
  TimeInfo() = default;

  TimeInfo(const double        time_in,
           const std::string & description_in,
           const std::string   unit_in    = "[s]",
           const unsigned int  n_calls_in = 1)
    : time(time_in), description(description_in), unit(unit_in), n_calls(n_calls_in)
  {
  }

  TimeInfo(const TimeInfo &) = default;

  bool
  is_active() const
  {
    return (time > static_cast<double>(0.));
  }

  void
  add_time(const double time_in)
  {
    time += time_in;
    ++n_calls;
  }

  TimeInfo &
  operator+=(const TimeInfo & other)
  {
    time += other.time;
    n_calls += other.n_calls;
    description = other.description;
    unit        = other.unit;
    return *this;
  }

  double       time        = 0.;
  std::string  description = "t.b.a.";
  std::string  unit        = "[s]";
  unsigned int n_calls     = 0;
};

// TODO rename TimeInfo struct into Stamp within a namespace TimeInfo
namespace TimeStat
{
TimeInfo
mean(const std::vector<TimeInfo> & timings)
{
  AssertThrow(!timings.empty(), ExcMessage("Empty vector of timings!"));
  TimeInfo     time_info_mean{timings.front()};
  const double time_sum =
    std::accumulate(timings.cbegin(), timings.cend(), 0., [](const auto & sum, const auto & tinfo) {
      return sum + tinfo.time;
    });
  time_info_mean.time = time_sum / (double)timings.size();
  const unsigned int n_calls_sum =
    std::accumulate(timings.cbegin(), timings.cend(), 0, [](const auto & sum, const auto & tinfo) {
      return sum + tinfo.n_calls;
    });
  time_info_mean.n_calls = n_calls_sum / timings.size();
  return time_info_mean;
}

/**
 * Formatted print function of a TimeInfo vector of different
 * sections. First, we print a table describing the code sections,
 * which wall time we have measured. Second, we print the raw time
 * data of each sample.
 */
void
fprint_sample(ConditionalOStream & pcout, const std::vector<std::vector<TimeInfo>> & timings)
{
  if(timings.empty())
    return;

  const std::size_t n_sections = timings.size();
  const std::size_t n_timings  = timings.front().size();
  AssertThrow(std::all_of(timings.cbegin(),
                          timings.cend(),
                          [n_timings](const auto & sect) { return sect.size() == n_timings; }),
              ExcMessage("Sample sizes are not equal in each section."));
  AssertThrow(n_timings > 0, ExcMessage("Empty section of timings."));

  // *** print descriptional table
  print_row_variable(pcout, 15, "Section No.", 50, "Description", "Unit");
  unsigned int no = 0;
  for(auto sect = timings.cbegin(); sect != timings.cend(); ++sect, ++no)
    print_row_variable(pcout, 15, no, 50, sect->front().description, sect->front().unit);
  pcout << std::endl;

  // *** print table of timings
  std::vector<unsigned int> no_header(n_sections);
  std::iota(no_header.begin(), no_header.end(), 0);
  print_row_variable(pcout, 15, "Section No.:", 12, no_header);
  for(std::size_t sample = 0; sample < n_timings; ++sample)
  {
    std::vector<double> times;
    for(const auto & sect : timings)
      times.emplace_back(sect[sample].time);
    print_row_variable(pcout, 15, "", 12, times);
  }
  pcout << std::endl;
}
} // namespace TimeStat

#endif /* TIMEINFO_H_ */
