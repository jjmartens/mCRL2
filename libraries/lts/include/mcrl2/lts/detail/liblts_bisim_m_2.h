// Author(s): Jan Martens
//
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
/// \file lts/detail/liblts_bisim_m.h
///
/// \brief Partition refinement for Branching bisimularity reduction.
/// Inspired by Jules Jacobs, Thorsten Wiﬂmann,  "Fast coalgebraic bisimilarity minimization." - POPL2023.
///
/// \details 
#ifndef _LIBLTS_BISIM_MARTENS
#define _LIBLTS_BISIM_MARTENS
#include <fstream>
#include "mcrl2/modal_formula/state_formula.h"
#include "mcrl2/lts/lts_utilities.h"
#include "mcrl2/lts/detail/liblts_scc.h"
#include "mcrl2/lts/detail/liblts_merge.h"
#include "mcrl2/lts/lts_aut.h"
#include "mcrl2/lts/lts_fsm.h"
#include "mcrl2/lts/lts_dot.h"

//#define UORDERED
namespace mcrl2
{
namespace lts
{
namespace detail
{
template < class LTS_TYPE>
class bisim_partitioner_martens
{
public:
  /** \brief Creates a bisimulation partitioner for an LTS.
    *  \details Based on the paper "Fast coalgebraic bisimilarity minimization." - Jacobs, Jules, and Thorsten Wiﬂmann. Proceedings of the ACM on Programming Languages 7.POPL (2023): 1514-1541.
    *  \warning Experimental.
    *  \param[in] l Reference to the LTS. */
  bisim_partitioner_martens(
    LTS_TYPE& l)
    : max_state_index(0),
    aut(l)
  {
    auto start = std::chrono::high_resolution_clock::now();

    const std::vector<transition>& trans = aut.get_transitions();
    // We should do this much smarter

    //Initialize arrays for pred and suc, blocks and state2loc and loc2state
    pred = new std::vector<pred_bucket_type>[aut.num_states()];

    //TODO optimization: Derive these by partitioning the transition array
    // WE doing strong bisimulation now so not needed.
    /* sil_pred = new std::set<state_type>[aut.num_states()];
    sil_suc = new std::set<state_type>[aut.num_states()];
    pre_marked = new std::set<state_type>[aut.num_states()];
    marked = new std::set<state_type>[aut.num_states()];*/

    blocks = new block_type[aut.num_states()];
    state2loc = new state_type[aut.num_states()];
    loc2state = new state_type[aut.num_states()];
    std::vector<int> state2in = std::vector<int>(aut.num_states(), 0);
    std::vector<int> state2out = std::vector<int>(aut.num_states(), 0);
    mCRL2log(mcrl2::log::debug) << "start moving transitions " << std::endl;
    std::size_t num_labels = aut.num_action_labels();

    std::vector<std::vector<transition>> label2buckettrans(num_labels);
    
    //Count transitions per state
    for (auto r = trans.begin(); r != trans.end(); r++) {
      state2in[r->to()] += 1;
      state2out[r->to()] += 1;
      label2buckettrans[r->label()].push_back((*r));
    }

    for (label_type a = 0; a < num_labels; ++a)
    {
      std::set<state_type> dirty;
      // TODO: Inefficient, don't use maps find better way for this. 
      std::map<state_type, std::vector<state_type>> state2preds;
      for (auto r = label2buckettrans[a].begin(); r != label2buckettrans[a].end(); r++)
      {
        state2out[r->to()] -= 1;
        state2preds[r->to()].push_back(r->from());
        dirty.insert(r->to());
      }

      for (auto s : dirty)
      {
        std::vector<state_type> preds(state2preds[s].size());
        std::copy(state2preds[s].begin(), state2preds[s].end(), preds.begin()); 
        pred[s].push_back(std::make_pair(a, preds));
      }
    }
    mCRL2log(mcrl2::log::debug) << "moved all transitions" << std::endl;
    // Log predecessor structure for debug: 
    /* for (state_type i = 0; i < aut.num_states(); ++i)
    {
			mCRL2log(mcrl2::log::debug) << "State " << i << " has " << pred[i].size() << " predecessors:" << std::endl;
      for (auto p : pred[i])
      {
				mCRL2log(mcrl2::log::debug) << "\t -" << p.first << "-> ";
        for (auto s : p.second)
        {
					mCRL2log(mcrl2::log::debug) << s << " ";
				}
				mCRL2log(mcrl2::log::debug) << std::endl;
			}
		} */

    //Initialize block map
    worklist = std::queue<superpartition>();
    block_map = std::vector<block>();

    //Initialize blocks
    block b0 { 0, 0, 0,(unsigned int)aut.num_states() };
    superpartition c0 {0, (unsigned int)aut.num_states()};

    for (state_type i = 0; i < (unsigned int)aut.num_states(); ++i)
    {
      blocks[i] = 0;
    }
    block_map.push_back(b0);
    max_block_index = 0;

    //initialize worklist
    worklist.push(c0);
    splitOn(b0);
    mCRL2log(mcrl2::log::debug) << "Done initial split max_block:" << max_block_index << std::endl;
    //Initialize state2loc and loc2state
    for (std::size_t i = 0; i < (unsigned int) aut.num_states(); ++i)
    {
      state2loc[i] = i;
      loc2state[i] = i;
    }


    // Iterate refinement 
    //refine();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    mCRL2log(mcrl2::log::info) << "Done s:" << duration.count() << std::endl;
    //cleanup
    delete[] pred;
    delete[] state2loc;
    delete[] loc2state;
    delete[] blocks;
  }

  /** \brief Destroys this partitioner. */
  ~bisim_partitioner_martens() = default;

  /** \brief Gives the number of bisimulation equivalence classes of the LTS.
   *  \return The number of bisimulation equivalence classes of the LTS.
   */
  std::size_t num_eq_classes() const
  {
    return block_map.size();
  }


  /** \brief Gives the bisimulation equivalence class number of a state.
   *  \param[in] s A state number.
   *  \return The number of the bisimulation equivalence class to which \e s belongs. */
  std::size_t get_eq_class(const std::size_t s) const
  {
    return blocks[s];
  }

  /** \brief Returns whether two states are in the same bisimulation equivalence class.
   *  \param[in] s A state number.
   *  \param[in] t A state number.
   *  \retval true if \e s and \e t are in the same bisimulation equivalence class;
   *  \retval false otherwise. */
  bool in_same_class(const std::size_t s, const std::size_t t) const
  {
    mCRL2log(mcrl2::log::debug) << "in_same_class " << s << " " << t << " " << get_eq_class(s) << " " << get_eq_class(t) << std::endl;
    return get_eq_class(s) == get_eq_class(t);
  }

private:
  typedef std::size_t state_type;
  typedef std::size_t label_type;
  typedef std::size_t block_type;
  typedef std::size_t location_type;
  typedef std::pair<label_type, block_type> observation_type;
  typedef std::pair<label_type, state_type> custom_transition_type;
  typedef std::pair<label_type, std::vector<state_type>> pred_bucket_type;

  state_type max_state_index;
  LTS_TYPE& aut;
  block_type max_block_index;
  // Array of vectors of predeccessors and successors

  // Struct block with start, mid, end being pointers
  // [start, mid) is clean, [mid, end) is dirty
  // [mid, bottom) is dirty and bottom.
  struct superpartition
  {
    state_type start;
    state_type end;

    state_type size() const
		{
			return end - start;
		}
  };
  
  // This data structure is used to store a block that is split into three parts.
  // When we currently split based on -a-> B where B \in C (superpartition).  
  // [start, empty) reaches only B, 
  // [empty, mid) reaches B and C,
  // [remid, end) reaches only C.
struct block
{
  state_type start;
  state_type empty;
  state_type mid;
  state_type end;
  //state_type bottom;
  //std::unordered_sfmet<label_type> unstable_labels;
  block_type new_B1 = 0;
  block_type new_B2 = 0;
  state_type size() const { return end - start; }
};

// Data structures; representing the partition and the Predecessors. 
std::vector<block> block_map;
std::vector<pred_bucket_type>* pred;
block_type* blocks;
state_type* loc2state;
state_type* state2loc;

std::queue<superpartition> worklist;

// Split such that the partition is stable w.r.t. B, and C \ B. (Where C is the superpartition\constellation of B). 
bool splitOn(block B)
{
  bool splitted = false;
  std::vector<state_type> B_states = std::vector<state_type>(B.end - B.start);
  std::copy(loc2state + B.start, loc2state + B.end, B_states.begin());
  mCRL2log(mcrl2::log::debug) << "Splitting on block " << B.start << " " << B.end << " " << B_states.size() << std::endl;
  // Loop (labelled) through all incoming transitions.
  std::vector<std::vector<state_type>> pred_buckets;
  pred_buckets.reserve(aut.num_action_labels());
  std::vector<label_type> labels;
  for (state_type s : B_states)
  {
    for (auto& p : pred[s])
    {
      label_type a = p.first;
      if (pred_buckets[a].size() == 0)
      {
        labels.push_back(a);
      }
      // Add all a-predecessors to the corresponding buckets.
      // TODO: Maybe this is a performance hit? We might only copy the references to the correct buckets. 
      pred_buckets[a].insert(
        pred_buckets[a].end(),
        p.second.begin(),
        p.second.end());
    }
    mCRL2log(mcrl2::log::debug) << "State " << s << " has " << pred[s].size() << " predecessors" << std::endl;
  }
  mCRL2log(mcrl2::log::debug) << "Splitting on " << labels.size() << " labels " << std::endl;
  // For each label, split all blocks.
  for (label_type a : labels)
  {
    // For now psuedocode only.
    // TODO: make the correct splitting
    mCRL2log(mcrl2::log::debug) << "Splitting on label " << a << std::endl;
    std::vector<block> blocks_touched;
    for (state_type s : pred_buckets[a])
    {
      // Means s -a-> B
      mark_and_count(s); 
    }
    // Split the blocks we touched.
    for (block B : blocks_touched)
    {
      if (B.empty != B.start || B.mid != B.start)
      {
        split(B);
        splitted = true;
      }
    }
  }
  return splitted;
}

void mark_and_count(state_type s)
{ 
  state_type location = state2loc[s];
  block& B = block_map[blocks[location]];
  if (B.mid > location)
  {
    // Marked already.
    // Maybe we should do something with reference here
    return;
  }
  // Mark by swapping the location with mid.
  state_type stmp = loc2state[B.mid];
  loc2state[B.mid] = s;
  loc2state[location] = stmp;
  state2loc[s] = B.mid;
  state2loc[stmp] = location;
  if (B.mid == B.start)
  {
    mCRL2log(mcrl2::log::debug) << "Marked " << s << " in new block " << blocks[location] << std::endl;
    B.new_B1 = ++max_block_index;
  }

  blocks[s] = B.new_B1;
  B.mid += 1;
}

void split(block B) {
  //split the block B {start, empty, mid, end} into new blocks.
  block b0 = block{ B.start, B.empty, B.empty, B.empty };
  block b1 = block{ B.empty, B.mid, B.mid, B.mid };
  if (b0.size() != 0)
  {
    block_map[b0.start] = b0;
  }
  if (b1.size() != 0)
  {
    block_map[b1.start] = b1;
  }
  // Resize the block B.
  block_map[B.mid].start = B.mid;
  block_map[B.mid].empty = B.mid;
}

//Refine based on sigs
void refine()
{
  int iter = 0;
  mCRL2log(mcrl2::log::info) << "Start refinement" << std::endl;
  while (!worklist.empty())
  {
    superpartition C = worklist.front();
    if (blocks[C.start] == blocks[C.end - 1])
    {
      // Superpartition has only one block, hence stable.
      worklist.pop();
      continue;
    }
    block B1 = block_map[blocks[C.start]], B2 = block_map[blocks[C.end - 1]];
    block B = (B1.size() < B2.size()) ? B1 : B2;
    splitOn(B);
  }
  mCRL2log(mcrl2::log::info) << "Done total blocks: \"" << block_map.size() << "\"" << std::endl;
}
};
}
}
}
#endif

