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
/// Mainly doing Paige-Tarjan approach, but not partition transitions, and using a different data structure.
/// Using ideas by Jules Jacobs, Thorsten Wiﬂmann,  "Fast coalgebraic bisimilarity minimization." - POPL2023.
/// 
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
    transition2counter = std::vector<std::size_t>(aut.num_transitions());
    counter_map = std::vector<reference_counter>();

    for (state_type s = 0; s < aut.num_states(); s++)
    {
			pred[s] = std::vector<pred_bucket_type>();
		}

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
    transition_type num_trans = 0;
    for (label_type a = 0; a < num_labels; ++a)
    {
      std::set<state_type> dirty;
      // TODO: Inefficient, don't use maps find better way for this.
      std::map<state_type, std::vector<state_counter_pair>> state2preds;
      std::map<transition_type, transition_type> createdcountermap;

      for (auto r = label2buckettrans[a].begin(); r != label2buckettrans[a].end(); r++)
      {
        reference_counter rc(r->from());
        if (createdcountermap.find(r->from()) == createdcountermap.end())
        {
          // This seems very obfuscated.
          // The info should be reset in the first iteration, 0 might be colliding.
          createdcountermap[r->from()] = counter_map.size();
          transition2counter[num_trans] = counter_map.size(); 
          counter_map.push_back(rc);
        }
        else
        {
          transition2counter[num_trans] = createdcountermap[r->from()];
          counter_map[transition2counter[num_trans]].add_transition();
        }
        state2preds[r->to()].push_back(state_counter_pair(r->from(), num_trans));
        dirty.insert(r->to());
        num_trans += 1;
      }

      for (auto s : dirty)
      {
        std::vector<state_counter_pair> preds;
        preds.reserve(state2preds[s].size());
        for (auto sp : state2preds[s])
        {
          preds.push_back(sp);
        }
        pred[s].push_back(std::make_pair(a, preds));
      }
    }

    mCRL2log(mcrl2::log::debug) << "moved all transitions" << num_trans << std::endl;
    // Log predecessor structure for debug:

    if (false)
    {
      for (state_type i = 0; i < aut.num_states(); ++i)
      {
        mCRL2log(mcrl2::log::debug) << "State " << i << " has " << pred[i].size() << " predecessors:" << std::endl;
        for (auto p : pred[i])
        {
          mCRL2log(mcrl2::log::debug) << "\t -" << aut.action_labels()[p.first] << "-> :\n";
          for (auto sp : p.second)   
          {
            mCRL2log(mcrl2::log::debug) << "\t\t" << sp.first << "ref: ";
            mCRL2log(mcrl2::log::debug) << sp.second << ", " << counter_map[transition2counter[sp.second]].get_count() << "\n ";
          }
          mCRL2log(mcrl2::log::debug) << std::endl;
        }
      }
    }
    // Initialize block map
    worklist = std::queue<superpartition>();
    block_map = std::vector<block>();

    // Initialize blocks
    block b0(0, (unsigned int)aut.num_states());
    b0.parent_block = 0;

    superpartition c0{0, (unsigned int)aut.num_states()};

    for (state_type i = 0; i < (unsigned int)aut.num_states(); i++)
    {
      blocks[i] = 0;
      state2loc[i] = i;
      loc2state[i] = i;
    }
    block_map.push_back(b0);
    auto mid = std::chrono::high_resolution_clock::now();
    mCRL2log(mcrl2::log::info) << "init:"
                                << std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count()
                                << std::endl;
    // initialize worklist
    splitOn(b0);
    worklist.push(c0);

    mCRL2log(mcrl2::log::debug) << "Done initial split max_block:" << block_map.size() << std::endl;
    // Initialize state2loc and loc2state

    // Iterate refinement
    refine();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - mid);
    mCRL2log(mcrl2::log::info) << "refining:" << duration.count() << std::endl;
    std::set<block_type> block_set;
    for (state_type i = 0; i < aut.num_states(); i++)
    {
      block_set.insert(blocks[i]);
      // mCRL2log(mcrl2::log::debug) << "location " << i << " is in block " << blocks[loc2state[i]] << std::endl;
    }

    mCRL2log(mcrl2::log::info) << "Done total blocks: \"" << block_set.size() << "\"" << std::endl;

    worklist.push(c0);
    refine();

    block_set.clear();
    for (state_type i = 0; i < aut.num_states(); i++)
    {
      block_set.insert(blocks[i]);
      // mCRL2log(mcrl2::log::debug) << "location " << i << " is in block " << blocks[loc2state[i]] << std::endl;
    }
    mCRL2log(mcrl2::log::info) << "Done total blocks: \"" << block_set.size() << "\"" << std::endl;

    // cleanup
    delete[] pred;
    delete[] state2loc;
    delete[] loc2state;
    delete[] blocks;
  };

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
  typedef std::size_t transition_type;
  typedef std::size_t location_type;
  typedef int counter;
  typedef std::pair<state_type, transition_type> state_counter_pair;
  typedef std::pair<label_type, block_type> observation_type;
  typedef std::pair<label_type, std::vector<state_counter_pair>> pred_bucket_type;
  typedef std::pair<label_type, block_type> action_block_type; 

  state_type max_state_index;
  LTS_TYPE& aut;
  // Array of vectors of predeccessors and successors

  // Struct superpart with start, mid, end being pointers
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
  // state_type bottom;
  block_type parent_block;
  block_type id = 0;

  /* GOAL Structures:
    block* parent_type

  */
  block_type new_B1 = 0;
  block_type new_B2 = 0;
  bool fresh = false;
  state_type size() const { return end - start; }

  // B2 goes to both B and C
  block(location_type start, location_type end)
      : start{start},
        empty{start},
        mid{start},
        end{end},
        parent_block{0}
  {}

  // TODO: Implement this function, and use more references in order to clean up the code(?).
  //  Plan can be to make the blocks array into references (block* blocks). We can destroy blocks when size() == 0.
};

/*
 * TODO: implement this reference structs for counting.
*/
struct reference_counter
{
  state_type from;
  bool immutable = true; // if there is only one outgoing transition, we don't need to worry about the counter.
  counter count = 1;
  
  transition_type new_counter=0;
  observation_type new_split_obs;

  bool touch(observation_type splitter, std::vector<reference_counter>& counters)
  {
    if (immutable)
    {
      // We don't need to worry about the counter it's only 1 transition.
      return true;
    }
    count -= 1;
    if (new_counter == 0 || splitter != new_split_obs)
    {
      new_counter = counters.size(); // We might not need from !?
      counters.push_back(reference_counter(from));
      new_split_obs = splitter;
    }
    else
    {
      counters[new_counter].add_transition();
    }
    return (count == 0);
  }
  
  counter get_count()
  { return count; 
  };

  void add_transition()
  { 
    immutable = false;
    count += 1;
  };

  reference_counter(state_type s)
      : from(s)
  {}

  reference_counter() 
  {}
};

// Data structures; representing the partition and the Predecessors. 
std::vector<pred_bucket_type>* pred;
std::vector<transition_type> transition2counter;
block_type* blocks;
std::vector<block> block_map;
state_type* loc2state;
state_type* state2loc;
std::vector<reference_counter> counter_map;
std::queue<superpartition> worklist;


void create_initial_partition() {


}
// Split such that the partition is stable w.r.t. B, and C \ B. (Where C is the superpartition\constellation of B). 
bool splitOn(block B)
{
  bool splitted = false;
  block_type Bid = blocks[loc2state[B.start]];
  std::vector<state_type> B_states = std::vector<state_type>(B.end - B.start);
  std::copy(loc2state + B.start, loc2state + B.end, B_states.begin());

  mCRL2log(mcrl2::log::debug) << "Splitting on block " << B.start << " " << B.end << " " << B_states.size() << std::endl;
  // Loop (labelled) through all incoming transitions.
  std::vector<std::vector<state_counter_pair>> pred_buckets(aut.num_action_labels());
  std::set<label_type> labels;

  for (state_type s : B_states)
  {
    for (auto p : pred[s])
    {
      label_type a = p.first;
      labels.insert(a);
      if (pred_buckets[a].size() == 0)
      {
        pred_buckets[a] = std::vector<state_counter_pair>();
      }
      // Add all a-predecessors to the corresponding buckets.
      // TODO: Maybe this is a performance hit? We might only copy the references to the correct buckets. 
      for (auto& state : p.second)
      {
				pred_buckets[a].push_back(state);
			}
    }
  }
  // For each label, split all blocks.
  for (label_type a : labels)
  {
    // For now psuedocode only.
    // TODO: make the correct splitting
    std::vector<block_type> blocks_touched;
    observation_type splitter = std::make_pair(a, Bid);
    for (state_counter_pair scounterpair : pred_buckets[a])
    {
      // Means s -a-> B
      mark(scounterpair.first, blocks_touched);
      // The reference to s-a-> C should be decreased
      if (counter_map[transition2counter[scounterpair.second]].touch(splitter, counter_map))
      {
        double_mark(scounterpair.first);
      }
      if (!(counter_map[transition2counter[scounterpair.second]].immutable))
      {
        // We should split the block, and update the reference to the new counter.
        // This is a double mark.
        transition2counter[scounterpair.second] = counter_map[transition2counter[scounterpair.second]].new_counter;
      }
    }

    // Split the blocks we touched.
    for (block_type Bid : blocks_touched)
    {
      block Bprime = block_map[Bid];
      if (Bprime.empty != Bprime.start || Bprime.mid != Bprime.start)
      {
        split(Bprime);
        splitted = true;
      }
    }

  }

  return splitted;
}

void printState(state_type s) { 
  mCRL2log(mcrl2::log::debug) << "State: " << s << " Block " << blocks[s] << " loc:" << state2loc[s]
                              << std::endl;
  printBlock(blocks[s]);
 }

void printBlock(block_type Bid) {
  block B = block_map[Bid];
  if (B.parent_block != Bid)
  {
    block Bp = block_map[B.parent_block];
    mCRL2log(mcrl2::log::debug) << "StippelBlock: " << Bid << "in Block: " << B.parent_block << " " << Bp.start << " "
                                                                          << Bp.empty << " " << Bp.mid << " " << Bp.end 
                                                                          << std::endl;
  }
  else
  {
    mCRL2log(mcrl2::log::debug) << "Block: " << Bid << " s:" << B.start << " e:" << B.empty << " m:" << B.mid << " " << B.end << " f:" << B.fresh << std::endl;
  }
}

void mark(state_type s, std::vector<block_type>& blocks_touched)
{ 
  state_type location = state2loc[s];
  block_type Bid = blocks[s];
  //mCRL2log(mcrl2::log::debug) << "marking: ";
  //printState(s);
  block B = block_map[blocks[s]];
  if (B.start > location || B.end <= location)
  {
    printState(s);
    assert(false);
    mCRL2log(mcrl2::log::debug) << "Not in block" << std::endl;
	}

  if (B.mid > location || B.parent_block != blocks[s])
  {
    // Marked already.
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
    block_map[Bid].new_B1 = block_map.size();
    block new_B = block(B.start, B.end);
    new_B.fresh = B.fresh;
    new_B.parent_block = Bid;
    block_map.push_back(new_B); // dummy block untill we finish splitting. 
    blocks_touched.push_back(Bid);
  }
  blocks[s] = block_map[Bid].new_B1;
  block_map[Bid].mid += 1;
}

void double_mark(state_type s) {
  //means there are no more transitions form s to B, C. 
  // Split from B1 into B2
  block_type Bid = blocks[s]; // s is already marked, so we should look at parent
  block B = block_map[Bid];
  block Bp = block_map[B.parent_block];
  if (Bp.empty == Bp.start)
  {
    // New block
    block new_B = block(Bp.start, Bp.end);
    new_B.parent_block = B.parent_block;
    new_B.fresh = Bp.fresh;
    block_map[B.parent_block].new_B2 = block_map.size();
    block_map.push_back(new_B);
  }
  // Swap s and the emptypointer , and update empty
  state_type stmp = loc2state[Bp.empty];
  loc2state[Bp.empty] = s;
  loc2state[state2loc[s]] = stmp;
  state2loc[stmp] = state2loc[s];
  state2loc[s] = Bp.empty; 
  blocks[s] = block_map[B.parent_block].new_B2;
  block_map[B.parent_block].empty += 1;

  /* mCRL2log(mcrl2::log::debug) << "Double marked " << s << " now in location " << state2loc[s] << " in: \n" 
    << block_map[B.parent_block].start << " " << block_map[B.parent_block].empty << " "
                              << block_map[B.parent_block].mid << " " << block_map[B.parent_block].end << std::endl;*/
}

void split(block B) {
  //split the block B {start, empty, mid, end} into new blocks.
  block b2 = block(B.start, B.empty);
  block b1 = block(B.empty, B.mid);
  
  if (b2.size() != 0)
  {
    b2.parent_block = B.new_B2;
    block_map[B.new_B2] = b2;
    /* mCRL2log(mcrl2::log::debug) << "test double marked block: ";
    printBlock(B.new_B2);*/
  }
  if (b1.size() != 0)
  {
    b1.parent_block = B.new_B1;
    block_map[B.new_B1] = b1;
  }
  if (b1.size() != 0 && b2.size() != 0)
  {
    mCRL2log(mcrl2::log::info) << "DOUBLE BLOCK!!";
  }
  // If split performed and new block, add to worklist 
  if (blocks[loc2state[B.start]] != blocks[loc2state[B.end - 1]])
  {
    printBlock(blocks[loc2state[B.start]]);
    printBlock(blocks[loc2state[B.mid-1]]);
    if (B.mid != B.end)
    {
      printBlock(blocks[loc2state[B.mid]]);
    }
  }
  else
  {
    block_map[blocks[loc2state[B.start]]].fresh = B.fresh;
  }
  if (B.fresh && blocks[loc2state[B.start]] != blocks[loc2state[B.end - 1]])
  {
    block_map[blocks[loc2state[B.start]]].fresh = false;
    worklist.push(superpartition{B.start, B.end});
  }
  // Resize the block B.
  if (B.mid != B.end)
  {
    block_map[blocks[loc2state[B.mid]]].start = B.mid;
    block_map[blocks[loc2state[B.mid]]].empty = B.mid; // This could be empty?? (We don't care for now)
    block_map[blocks[loc2state[B.mid]]].parent_block = blocks[loc2state[B.mid]];
    B.new_B1 = 0;
    B.new_B2 = 0;
  }
}

void logSuperPartition(superpartition C)
{
  std::set<block_type> block_set;
  for (state_type i = C.start; i < C.end; i++)
  {
    if (block_set.insert(blocks[loc2state[i]]).second)
    {
      mCRL2log(mcrl2::log::debug) << i << " Block " << blocks[loc2state[i]] << " " << block_map[blocks[loc2state[i]]].start << " " << block_map[blocks[loc2state[i]]].end << std::endl;
    }
  }
  mCRL2log(mcrl2::log::debug) << "Superpartition " << C.start << " " << C.end << " has " << block_set.size()
                              << " blocks" << std::endl;
}

//Refine based on sigs
void refine()
{
  int iter = 0;
  mCRL2log(mcrl2::log::info) << "Start refinement" << std::endl;
  int stableinrow = 0;
  while (!worklist.empty())
  {
    iter++;
    superpartition C = worklist.front();
    mCRL2log(mcrl2::log::debug) << "Refining superpartition " << C.start << " " << C.end << std::endl;
    if (blocks[loc2state[C.start]] == blocks[loc2state[C.end - 1]])
    {
      // Superpartition has only one block, hence stable.
      // We get this block for free.
      worklist.pop();
      //splitOn(block_map[blocks[loc2state[C.start]]]);//TEMPORARY;
      block_map[blocks[loc2state[C.start]]].fresh = true; // If this block is split, we should add it to the worklist.
      //worklist.push(superpartition{C.start, C.end});
      continue;
    }

    //logSuperPartition(C);
    block B1 = block_map[blocks[loc2state[C.start]]], B2 = block_map[blocks[loc2state[C.end - 1]]];

    //mCRL2log(mcrl2::log::debug) << "sizeB1: " << B1.size() << " sizeB2:" << B2.size() << std::endl;

    block B = (B1.size() < B2.size()) ? B1 : B2;
    splitOn(B);
    // We did our work. for B. The blocks inside B should now be their own superpartition.
    //mCRL2log(mcrl2::log::debug) << "Done splitting on block " << B.start << " " << B.end << std::endl;
    //worklist.push(superpartition{B.start, B.end});
    
    if (blocks[loc2state[B.start]] != blocks[loc2state[B.end-1]])
    {
      // B is split on itself, hence we should add the range to the worklist.
      worklist.push(superpartition{B.start, B.end});
    }
    else
    {
      block_map[blocks[loc2state[B.start]]].fresh = true;
    }
    
    // Update C, by removing B from the superpartition.
    worklist.front().start = (B1.size() < B2.size()) ? B.end : C.start;
    worklist.front().end = (B1.size() < B2.size()) ? C.end : B.start;
  }
}
};
}
}
}
#endif

