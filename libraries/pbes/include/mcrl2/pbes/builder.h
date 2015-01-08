// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/builder.h
/// \brief add your file description here.

#ifndef MCRL2_PBES_PBES_H
#include "mcrl2/pbes/pbes.h"
#endif

#ifndef MCRL2_PBES_BUILDER_H
#define MCRL2_PBES_BUILDER_H

#include "mcrl2/core/builder.h"
#include "mcrl2/data/builder.h"
#include "mcrl2/pbes/pbes.h"

namespace mcrl2
{

namespace pbes_system
{

/// \brief Builder class
template <typename Derived>
struct pbes_expression_builder_base: public core::builder<Derived>
{
  typedef core::builder<Derived> super;
  using super::enter;
  using super::leave;
  using super::operator();
  	
  data::data_expression operator()(const data::data_expression& x)
  {
  	return x;
  }
};

// Adds sort expression traversal to a builder
//--- start generated add_sort_expressions code ---//
template <template <class> class Builder, class Derived>
struct add_sort_expressions: public Builder<Derived>
{
  typedef Builder<Derived> super;
  using super::enter;
  using super::leave;
#ifndef _MSC_VER
  using super::operator();
#endif

  pbes_system::propositional_variable operator()(const pbes_system::propositional_variable& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::propositional_variable result = pbes_system::propositional_variable(x.name(), static_cast<Derived&>(*this)(x.parameters()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  void operator()(pbes_system::pbes_equation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    x.variable() = static_cast<Derived&>(*this)(x.variable());
    x.formula() = static_cast<Derived&>(*this)(x.formula());
    static_cast<Derived&>(*this).leave(x);
  }

  void operator()(pbes_system::pbes& x)
  {
    static_cast<Derived&>(*this).enter(x);
    static_cast<Derived&>(*this)(x.equations());
    static_cast<Derived&>(*this)(x.global_variables());
    x.initial_state() = static_cast<Derived&>(*this)(x.initial_state());
    static_cast<Derived&>(*this).leave(x);
  }

  pbes_system::propositional_variable_instantiation operator()(const pbes_system::propositional_variable_instantiation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::propositional_variable_instantiation result = pbes_system::propositional_variable_instantiation(x.name(), static_cast<Derived&>(*this)(x.parameters()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::not_ operator()(const pbes_system::not_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::not_ result = pbes_system::not_(static_cast<Derived&>(*this)(x.operand()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::and_ operator()(const pbes_system::and_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::and_ result = pbes_system::and_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::or_ operator()(const pbes_system::or_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::or_ result = pbes_system::or_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::imp operator()(const pbes_system::imp& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::imp result = pbes_system::imp(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::forall operator()(const pbes_system::forall& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::forall result = pbes_system::forall(static_cast<Derived&>(*this)(x.variables()), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::exists operator()(const pbes_system::exists& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::exists result = pbes_system::exists(static_cast<Derived&>(*this)(x.variables()), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::pbes_expression operator()(const pbes_system::pbes_expression& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::pbes_expression result;
    if (data::is_data_expression(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::data_expression>(x));
    }
    else if (pbes_system::is_propositional_variable_instantiation(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::propositional_variable_instantiation>(x));
    }
    else if (pbes_system::is_not(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::not_>(x));
    }
    else if (pbes_system::is_and(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::and_>(x));
    }
    else if (pbes_system::is_or(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::or_>(x));
    }
    else if (pbes_system::is_imp(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::imp>(x));
    }
    else if (pbes_system::is_forall(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::forall>(x));
    }
    else if (pbes_system::is_exists(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::exists>(x));
    }
    else if (data::is_variable(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::variable>(x));
    }
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

};

/// \brief Builder class
template <typename Derived>
struct sort_expression_builder: public add_sort_expressions<data::sort_expression_builder, Derived>
{
  typedef add_sort_expressions<data::sort_expression_builder, Derived> super;
  using super::enter;
  using super::leave;
  using super::operator();
};
//--- end generated add_sort_expressions code ---//

// Adds data expression traversal to a builder
//--- start generated add_data_expressions code ---//
template <template <class> class Builder, class Derived>
struct add_data_expressions: public Builder<Derived>
{
  typedef Builder<Derived> super;
  using super::enter;
  using super::leave;
#ifndef _MSC_VER
  using super::operator();
#endif

  void operator()(pbes_system::pbes_equation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    x.formula() = static_cast<Derived&>(*this)(x.formula());
    static_cast<Derived&>(*this).leave(x);
  }

  void operator()(pbes_system::pbes& x)
  {
    static_cast<Derived&>(*this).enter(x);
    static_cast<Derived&>(*this)(x.equations());
    x.initial_state() = static_cast<Derived&>(*this)(x.initial_state());
    static_cast<Derived&>(*this).leave(x);
  }

  pbes_system::propositional_variable_instantiation operator()(const pbes_system::propositional_variable_instantiation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::propositional_variable_instantiation result = pbes_system::propositional_variable_instantiation(x.name(), static_cast<Derived&>(*this)(x.parameters()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::not_ operator()(const pbes_system::not_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::not_ result = pbes_system::not_(static_cast<Derived&>(*this)(x.operand()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::and_ operator()(const pbes_system::and_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::and_ result = pbes_system::and_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::or_ operator()(const pbes_system::or_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::or_ result = pbes_system::or_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::imp operator()(const pbes_system::imp& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::imp result = pbes_system::imp(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::forall operator()(const pbes_system::forall& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::forall result = pbes_system::forall(x.variables(), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::exists operator()(const pbes_system::exists& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::exists result = pbes_system::exists(x.variables(), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::pbes_expression operator()(const pbes_system::pbes_expression& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::pbes_expression result;
    if (data::is_data_expression(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::data_expression>(x));
    }
    else if (pbes_system::is_propositional_variable_instantiation(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::propositional_variable_instantiation>(x));
    }
    else if (pbes_system::is_not(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::not_>(x));
    }
    else if (pbes_system::is_and(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::and_>(x));
    }
    else if (pbes_system::is_or(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::or_>(x));
    }
    else if (pbes_system::is_imp(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::imp>(x));
    }
    else if (pbes_system::is_forall(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::forall>(x));
    }
    else if (pbes_system::is_exists(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::exists>(x));
    }
    else if (data::is_variable(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::variable>(x));
    }
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

};

/// \brief Builder class
template <typename Derived>
struct data_expression_builder: public add_data_expressions<data::data_expression_builder, Derived>
{
  typedef add_data_expressions<data::data_expression_builder, Derived> super;
  using super::enter;
  using super::leave;
  using super::operator();
};
//--- end generated add_data_expressions code ---//

//--- start generated add_variables code ---//
template <template <class> class Builder, class Derived>
struct add_variables: public Builder<Derived>
{
  typedef Builder<Derived> super;
  using super::enter;
  using super::leave;
#ifndef _MSC_VER
  using super::operator();
#endif

  pbes_system::propositional_variable operator()(const pbes_system::propositional_variable& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::propositional_variable result = pbes_system::propositional_variable(x.name(), static_cast<Derived&>(*this)(x.parameters()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  void operator()(pbes_system::pbes_equation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    x.variable() = static_cast<Derived&>(*this)(x.variable());
    x.formula() = static_cast<Derived&>(*this)(x.formula());
    static_cast<Derived&>(*this).leave(x);
  }

  void operator()(pbes_system::pbes& x)
  {
    static_cast<Derived&>(*this).enter(x);
    static_cast<Derived&>(*this)(x.equations());
    static_cast<Derived&>(*this)(x.global_variables());
    x.initial_state() = static_cast<Derived&>(*this)(x.initial_state());
    static_cast<Derived&>(*this).leave(x);
  }

  pbes_system::propositional_variable_instantiation operator()(const pbes_system::propositional_variable_instantiation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::propositional_variable_instantiation result = pbes_system::propositional_variable_instantiation(x.name(), static_cast<Derived&>(*this)(x.parameters()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::not_ operator()(const pbes_system::not_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::not_ result = pbes_system::not_(static_cast<Derived&>(*this)(x.operand()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::and_ operator()(const pbes_system::and_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::and_ result = pbes_system::and_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::or_ operator()(const pbes_system::or_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::or_ result = pbes_system::or_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::imp operator()(const pbes_system::imp& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::imp result = pbes_system::imp(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::forall operator()(const pbes_system::forall& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::forall result = pbes_system::forall(static_cast<Derived&>(*this)(x.variables()), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::exists operator()(const pbes_system::exists& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::exists result = pbes_system::exists(static_cast<Derived&>(*this)(x.variables()), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::pbes_expression operator()(const pbes_system::pbes_expression& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::pbes_expression result;
    if (data::is_data_expression(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::data_expression>(x));
    }
    else if (pbes_system::is_propositional_variable_instantiation(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::propositional_variable_instantiation>(x));
    }
    else if (pbes_system::is_not(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::not_>(x));
    }
    else if (pbes_system::is_and(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::and_>(x));
    }
    else if (pbes_system::is_or(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::or_>(x));
    }
    else if (pbes_system::is_imp(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::imp>(x));
    }
    else if (pbes_system::is_forall(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::forall>(x));
    }
    else if (pbes_system::is_exists(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::exists>(x));
    }
    else if (data::is_variable(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::variable>(x));
    }
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

};

/// \brief Builder class
template <typename Derived>
struct variable_builder: public add_variables<data::data_expression_builder, Derived>
{
  typedef add_variables<data::data_expression_builder, Derived> super;
  using super::enter;
  using super::leave;
  using super::operator();
};
//--- end generated add_variables code ---//

// Adds pbes expression traversal to a builder
//--- start generated add_pbes_expressions code ---//
template <template <class> class Builder, class Derived>
struct add_pbes_expressions: public Builder<Derived>
{
  typedef Builder<Derived> super;
  using super::enter;
  using super::leave;
#ifndef _MSC_VER
  using super::operator();
#endif

  void operator()(pbes_system::pbes_equation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    x.formula() = static_cast<Derived&>(*this)(x.formula());
    static_cast<Derived&>(*this).leave(x);
  }

  void operator()(pbes_system::pbes& x)
  {
    static_cast<Derived&>(*this).enter(x);
    static_cast<Derived&>(*this)(x.equations());
    static_cast<Derived&>(*this).leave(x);
  }

  pbes_system::propositional_variable_instantiation operator()(const pbes_system::propositional_variable_instantiation& x)
  {
    static_cast<Derived&>(*this).enter(x);
    // skip
    static_cast<Derived&>(*this).leave(x);
    return x;
  }

  pbes_system::not_ operator()(const pbes_system::not_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::not_ result = pbes_system::not_(static_cast<Derived&>(*this)(x.operand()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::and_ operator()(const pbes_system::and_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::and_ result = pbes_system::and_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::or_ operator()(const pbes_system::or_& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::or_ result = pbes_system::or_(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::imp operator()(const pbes_system::imp& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::imp result = pbes_system::imp(static_cast<Derived&>(*this)(x.left()), static_cast<Derived&>(*this)(x.right()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::forall operator()(const pbes_system::forall& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::forall result = pbes_system::forall(x.variables(), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::exists operator()(const pbes_system::exists& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::exists result = pbes_system::exists(x.variables(), static_cast<Derived&>(*this)(x.body()));
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

  pbes_system::pbes_expression operator()(const pbes_system::pbes_expression& x)
  {
    static_cast<Derived&>(*this).enter(x);
    pbes_system::pbes_expression result;
    if (data::is_data_expression(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::data_expression>(x));
    }
    else if (pbes_system::is_propositional_variable_instantiation(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::propositional_variable_instantiation>(x));
    }
    else if (pbes_system::is_not(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::not_>(x));
    }
    else if (pbes_system::is_and(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::and_>(x));
    }
    else if (pbes_system::is_or(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::or_>(x));
    }
    else if (pbes_system::is_imp(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::imp>(x));
    }
    else if (pbes_system::is_forall(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::forall>(x));
    }
    else if (pbes_system::is_exists(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<pbes_system::exists>(x));
    }
    else if (data::is_variable(x))
    {
      result = static_cast<Derived&>(*this)(atermpp::down_cast<data::variable>(x));
    }
    static_cast<Derived&>(*this).leave(x);
    return result;
  }

};

/// \brief Builder class
template <typename Derived>
struct pbes_expression_builder: public add_pbes_expressions<pbes_system::pbes_expression_builder_base, Derived>
{
  typedef add_pbes_expressions<pbes_system::pbes_expression_builder_base, Derived> super;
  using super::enter;
  using super::leave;
  using super::operator();
};
//--- end generated add_pbes_expressions code ---//

} // namespace pbes_system

} // namespace mcrl2

#endif // MCRL2_PBES_BUILDER_H
