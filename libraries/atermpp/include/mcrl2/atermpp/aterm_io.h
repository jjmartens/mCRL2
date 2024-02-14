// Author(s): Jan Friso Groote
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef MCRL2_ATERMPP_ATERM_IO_H
#define MCRL2_ATERMPP_ATERM_IO_H

#include "mcrl2/atermpp/aterm_list.h"
#include "mcrl2/atermpp/aterm_int.h"
#include "mcrl2/utilities/type_traits.h"

namespace atermpp
{

/// \brief A function that is applied to all terms. The resulting term should only use
///        a subset of the original arguments (i.e. not introduce new terms).
/// \details Typical usage is removing the index traits from function symbols that represent operators.
using aterm_transformer = aterm_appl(const aterm_appl&);

/// \brief The default transformer that maps each term to itself.
inline aterm_appl identity(const aterm_appl& x) { return x; }

/// \brief The general aterm_core stream interface, which enables the use of a transformer to
///        change the written/read terms.
class aterm_stream
{
public:
  virtual ~aterm_stream();

  /// \brief Sets the given transformer to be applied to following writes.
  void set_transformer(aterm_transformer transformer) { m_transformer = transformer; }

  /// \returns The currently assigned transformer function.
  aterm_transformer* get_transformer() const { return m_transformer; }

protected:
  aterm_transformer* m_transformer = identity;
};

/// \brief The interface for a class that writes aterm_core to a stream.
///        Every written term is retrieved by the corresponding aterm_istream::get() call.
class aterm_ostream : public aterm_stream
{
public:
  virtual ~aterm_ostream();

  /// \brief Write the given term to the stream.
  virtual void put(const aterm_core& term) = 0;
};

/// \brief The interface for a class that reads aterm_core from a stream.
///        The default constructed term aterm_core() indicates the end of the stream.
class aterm_istream : public aterm_stream
{
public:
  virtual ~aterm_istream();

  /// \brief Reads an aterm_core from this stream.
  virtual void get(aterm_core& t) = 0;
};

// These free functions provide input/output operators for these streams.

/// \brief Sets the given transformer to be applied to following reads.
inline aterm_istream& operator>>(aterm_istream& stream, aterm_transformer transformer) { stream.set_transformer(transformer); return stream; }
inline aterm_ostream& operator<<(aterm_ostream& stream, aterm_transformer transformer) { stream.set_transformer(transformer); return stream; }

/// \brief Write the given term to the stream.
inline aterm_ostream& operator<<(aterm_ostream& stream, const aterm_core& term) { stream.put(term); return stream; }

/// \brief Read the given term from the stream, but for aterm_list we want to use a specific one that performs validation (defined below).
inline aterm_istream& operator>>(aterm_istream& stream, aterm_core& term) { stream.get(term); return stream; }

// Utility functions

/// \brief A helper class to restore the state of the aterm_{i,o}stream objects upon destruction. Currently, onlt
///        preserves the transformer object.
class aterm_stream_state
{
public:
  aterm_stream_state(aterm_stream& stream)
    : m_stream(stream)
  {
    m_transformer = stream.get_transformer();
  }

  ~aterm_stream_state()
  {
    m_stream.set_transformer(m_transformer);
  }

private:
  aterm_stream& m_stream;
  aterm_transformer* m_transformer;
};

/// \brief Write any container (that is not an aterm_core itself) to the stream.
template<typename T,
  typename std::enable_if_t<mcrl2::utilities::is_iterable_v<T>, int> = 0,
  typename std::enable_if_t<!std::is_base_of<aterm_core, T>::value, int> = 0>
inline aterm_ostream& operator<<(aterm_ostream& stream, const T& container)
{
  // Write the number of elements, followed by each element in the container.
  stream << aterm_int(std::distance(container.begin(), container.end()));

  for (const auto& element : container)
  {
    stream << element;
  }

  return stream;
}

/// \brief Read any container (that is not an aterm_core itself) from the stream.
template<typename T,
  typename std::enable_if_t<mcrl2::utilities::is_iterable_v<T>, int> = 0,
  typename std::enable_if_t<!std::is_base_of<aterm_core, T>::value, int> = 0>
inline aterm_istream& operator>>(aterm_istream& stream, T& container)
{
  // Insert the next nof_elements into the container.
  aterm_int nof_elements;
  stream >> nof_elements;

  auto it = std::inserter(container, container.end());
  for (std::size_t i = 0; i < nof_elements.value(); ++i)
  {
    typename T::value_type element;
    stream >> element;
    it = element;
  }

  return stream;
}

template<typename T>
inline aterm_ostream& operator<<(aterm_ostream&& stream, const T& t) { stream << t; return stream; }

template<typename T>
inline aterm_istream& operator>>(aterm_istream&& stream, T& t) { stream >> t; return stream; }

/// \brief Sends the name of a function symbol to an ostream.
/// \param out The out stream.
/// \param f The function symbol to be output.
/// \return The stream.
inline
std::ostream& operator<<(std::ostream& out, const function_symbol& f)
{
  return out << f.name();
}

/// \brief Prints the name of a function symbol as a string.
/// \param f The function symbol.
/// \return The string representation of r.
inline const std::string& pp(const function_symbol& f)
{
  return f.name();
}

/// \brief Writes term t to a stream in binary aterm_core format.
void write_term_to_binary_stream(const aterm_core& t, std::ostream& os);

/// \brief Reads a term from a stream in binary aterm_core format.
void read_term_from_binary_stream(std::istream& is, aterm_core& t);

/// \brief Writes term t to a stream in textual format.
void write_term_to_text_stream(const aterm_core& t, std::ostream& os);

/// \brief Reads a term from a stream which contains the term in textual format.
void read_term_from_text_stream(std::istream& is, aterm_core& t);

/// \brief Reads an aterm_core from a string. The string can be in either binary or text format.
aterm_core read_term_from_string(const std::string& s);

/// \brief Reads an aterm_list from a string. The string can be in either binary or text format.
/// \details If the input is not a string, an aterm_core is returned of the wrong type.
/// \return The term corresponding to the string.
inline aterm_list read_list_from_string(const std::string& s)
{
  const aterm_list l = down_cast<aterm_list>(read_term_from_string(s));
  assert(l.type_is_list());
  return l;
}

/// \brief Reads an aterm_int from a string. The string can be in either binary or text format.
/// \details If the input is not an int, an aterm_core is returned of the wrong type.
/// \return The aterm_int corresponding to the string.
inline aterm_int read_int_from_string(const std::string& s)
{
  const aterm_int n = down_cast<aterm_int>(read_term_from_string(s));
  assert(n.type_is_int());
  return n;
}

/// \brief Reads an aterm_appl from a string. The string can be in either binary or text format.
/// \details If the input is not an aterm_appl, an aterm_core is returned of the wrong type.
/// \return The term corresponding to the string.
inline aterm_appl read_appl_from_string(const std::string& s)
{
  const aterm_appl a = down_cast<aterm_appl>(read_term_from_string(s));
  assert(a.type_is_appl());
  return a;
}


} // namespace atermpp

#endif // MCRL2_ATERMPP_ATERM_IO_H
