// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/tools/pbesabstract.h
/// \brief add your file description here.

#ifndef MCRL2_PBES_TOOLS_PBESABSTRACT_H
#define MCRL2_PBES_TOOLS_PBESABSTRACT_H

#include "mcrl2/pbes/abstract.h"
#include "mcrl2/pbes/io.h"
#include "mcrl2/pbes/tools.h"
#include "mcrl2/pbes/detail/pbes_parameter_map.h"

namespace mcrl2 {

namespace pbes_system {

void pbesabstract(const std::string& input_filename,
                  const std::string& output_filename,
                  const std::string& parameter_selection,
                  bool value_true
                 )
{
  // load the pbes
  pbes<> p;
  load_pbes(p, input_filename);

  // run the algorithm
  pbes_abstract_algorithm algorithm;
  pbes_system::detail::pbes_parameter_map parameter_map = pbes_system::detail::parse_pbes_parameter_map(p, parameter_selection);
  algorithm.run(p, parameter_map, value_true);

  // save the result
  p.save(output_filename);
}

} // namespace pbes_system

} // namespace mcrl2

#endif // MCRL2_PBES_TOOLS_PBESABSTRACT_H
