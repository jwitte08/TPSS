/**
 * app_utilites.h
 *
 * collection of helper functions
 *
 *  Created on: Sep 15, 2020
 *      Author: witte
 */

#ifndef APP_UTILITIES_H_
#define APP_UTILITIES_H_


#include <algorithm>
#include <cstring>
#include <string>
#include <vector>



namespace Util
{
static constexpr char const * skipper = "o";

std::vector<char const *>
args_to_strings(const int argc_in, char * argv_in[])
{
  std::vector<char const *> tmp;
  std::copy_n(argv_in, argc_in, std::back_inserter(tmp));
  return tmp;
}

struct ConditionalAtoi
{
  ConditionalAtoi(const int argc_in, char * argv_in[]) : argv(args_to_strings(argc_in, argv_in))
  {
  }

  template<typename T>
  void
  operator()(T & prm, const std::size_t index)
  {
    if(argv.size() <= index)
      return;
    if(std::strcmp(argv[index], skipper) == 0)
      return;
    prm = std::atoi(argv[index]);
  }

  std::vector<char const *> argv;
};

struct ConditionalAtof
{
  ConditionalAtof(const int argc_in, char * argv_in[]) : argv(args_to_strings(argc_in, argv_in))
  {
  }

  template<typename T>
  void
  operator()(T & prm, const std::size_t index)
  {
    if(argv.size() <= index)
      return;
    if(std::strcmp(argv[index], skipper) == 0)
      return;
    prm = std::atof(argv[index]);
  }

  std::vector<char const *> argv;
};

} // end namespace Util

#endif /* APP_UTILITIES_H_ */
