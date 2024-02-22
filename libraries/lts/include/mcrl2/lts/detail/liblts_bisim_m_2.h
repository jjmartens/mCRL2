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
/// \brief Partition refinement for strong bisimularity reduction.
/// Based on the paper "Fast coalgebraic bisimilarity minimization." - Jacobs, Jules, and Thorsten Wiﬂmann.
/// counter-examples.
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

       //Initialize arrays for pred and suc, blocks and state2loc and loc2state
       pred = new std::vector<custom_transition_type>[aut.num_states()];
       suc  = new std::vector<custom_transition_type>[aut.num_states()];

       blocks = new block_type[aut.num_states()];
       state2loc = new state_type[aut.num_states()];
       loc2state = new state_type[aut.num_states()];
       
       std::vector<int> state2in = std::vector<int>(aut.num_states(), 0);
       std::vector<int> state2out = std::vector<int>(aut.num_states(), 0);
       mCRL2log(mcrl2::log::debug) << "start moving transitions " << std::endl;
       //Count transitions per state
       for (auto r = trans.begin(); r != trans.end(); r++) {
           state2in[(*r).to()] += 1;
           state2out[(*r).from()] += 1;
       }

       for (state_type s = 0; s < aut.num_states(); ++s)
	   {
           pred[s] = std::vector<custom_transition_type>(state2in[s]);
           suc[s] = std::vector<custom_transition_type>(state2out[s]);
	   }
	   mCRL2log(mcrl2::log::debug) << "now placing in correct place " << std::endl;
       for (auto r = trans.begin(); r != trans.end(); r++)
       {
           state2in[(*r).to()] -= 1;
           pred[(*r).to()][state2in[(*r).to()]] = std::make_pair((*r).label(), (*r).from());
           state2out[(*r).from()] -= 1;
           suc[(*r).from()][state2out[(*r).from()]] = std::make_pair((*r).label(), (*r).to());
       }
       mCRL2log(mcrl2::log::debug) << "moved all transitions" << std::endl;

        //Initialize block map
       worklist = std::queue<block_type>();
       block_map = std::map<block_type, block>();


       block b0 = block{0,0,(unsigned int) aut.num_states()};
        for (state_type i = 0; i < (unsigned int) aut.num_states(); ++i)
        {
            blocks[i] = 0; 
        }

        block_map.insert(std::pair(0, b0));
        //initialize worklist
        worklist.push(0);

        //Initialize state2loc and loc2state
        for(std::size_t i = 0; i < aut.num_states(); ++i)
		{
			state2loc[i] = i;
			loc2state[i] = i;
		}

        // Iterate refinement 
        refine();

        delete[] pred;
        delete[] suc;
        delete[] state2loc;
        delete[] loc2state;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        mCRL2log(mcrl2::log::info) << "Done s:" << duration.count() <<  std::endl;
    }

    /** \brief Destroys this partitioner. */
    ~bisim_partitioner_martens() = default;

    /** \brief Gives the number of bisimulation equivalence classes of the LTS.
     *  \return The number of bisimulation equivalence classes of the LTS.
     */
    std::size_t num_eq_classes() const
    {
        return max_state_index;
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
        mCRL2log(mcrl2::log::debug) << "in_same_class " << s<< " "<< t<< " " << get_eq_class(s) << " " << get_eq_class(t) << std::endl;
        return get_eq_class(s) == get_eq_class(t);
    }

private:
    typedef std::size_t state_type;
    typedef std::size_t label_type;
    typedef std::size_t block_type;
    typedef std::pair<label_type, block_type> observation_type;
    typedef std::pair<label_type, state_type> custom_transition_type;

    //Typedef for signature
#ifdef UORDERED
	typedef std::set<observation_type> signature_type;
#else
    typedef std::set<observation_type> signature_type;
#endif
    
    state_type max_state_index;
    LTS_TYPE& aut;
    // Array of vectors of predeccessors and successors
    std::vector<custom_transition_type>* pred; 
    std::vector<custom_transition_type>* suc;
    
    //Struct block with start, mid, end beingpointers
    struct block
	{
		state_type start;
		state_type mid;
		state_type end;
        //std::unordered_set<label_type> unstable_labels;
	};

    struct sigHash
    {
        std::size_t operator()(signature_type const& s) const
        {
            std::size_t res = 0;
            for (auto t : s)
			{
				res = res ^ t.first;
				res = res ^ t.second;
			}
            return res;
        }
    };

    block_type* blocks;
    std::map<block_type, block> block_map;
    state_type* loc2state;
    state_type* state2loc;
    std::queue<block_type> worklist;

    void mark_dirty(state_type s)
	{
        block_type Bid = blocks[s];
        block B = block_map[Bid];
        state_type loc = state2loc[s];
        //Add label to unstable labels
        /*if (B.unstable_labels.find(a) == B.unstable_labels.end()) {
            block_map[Bid].unstable_labels.insert(a);
        }*/

        if (loc < B.start || loc > B.end) {
            mCRL2log(mcrl2::log::info) << "KAPUT "<< loc << " " << B.start << " " << B.mid << " " << B.end << std::endl;
        }

        if (B.mid <= loc or B.start >= B.end-1) {
            //Already dirty or only 1 state
            return;
        }
        if (B.mid == B.end) {
            //First dirty state
            worklist.push(Bid);
        }
        //Swap last clean state with s
        state_type new_id = B.mid - 1;
        state_type tmp = loc2state[new_id];
        block_map[Bid].mid = new_id;

        state2loc[tmp] = loc;
        loc2state[loc] = tmp;
        state2loc[s] = new_id;
        loc2state[new_id] = s;
	}
    
    //Signature of a state
    void sig(const state_type& s, signature_type& retsignature)
    {
        for (auto t : suc[s])
        {
            auto sig = std::make_pair(t.first, blocks[t.second]);
            // Add observation of r to sig
            retsignature.insert(sig);
        }
    }

    //Split block based on signature
    int split(block_type Bid)
    {
		block B = block_map[Bid];
		if (B.mid == B.end)
		{
			//Block is clean (should not happen)
            mCRL2log(mcrl2::log::debug) << "Block clean but in worklist!? should not happen." << std::endl;
			return 0;
		}
#ifdef UORDERED
        std::unordered_map<signature_type, block_type, sigHash> sig2block;
#else  
        std::map<signature_type, block_type> sig2block;
#endif
        int j = 0;
        block_type* state2block = new block_type[B.end - B.mid];
        mCRL2log(mcrl2::log::debug) << "Computing signatures from here. :" << std::endl;

        //Add signature of one clean state
        signature_type signature;

        if (B.mid != B.start) {
			state_type s = loc2state[B.start];
            signature.clear();
            sig(loc2state[s], signature);
			sig2block.insert(std::make_pair(signature, j));
            j += 1;
		}
		//Add signatures of dirty states
		for (state_type s = B.mid; s < B.end; s++)
		{
            signature.clear();
			sig(loc2state[s], signature);
            auto ret = sig2block.insert(std::make_pair(signature, j));
            if (ret.second) {
                j += 1;
            } 
            state2block[s-B.mid] = (*ret.first).second;
		}
        if (j == 1) {
            //Only one signature, no need to split
            mCRL2log(mcrl2::log::debug) << "no new signatures." << B.end - B.start << B.mid - B.start << std::endl;
            block_map[Bid].mid = B.end;
            return 0;
        }
		//Count number of occurrences each signature
        int* Sizes = new int[j+1];
        for (int i = 0; i <= j; i++)
		{
			Sizes[i] = 0;
		}
        //Sizes[0] = B.mid - B.start;
        for (state_type s = B.mid; s < B.end; s++)
        {
            Sizes[state2block[s-B.mid]] += 1;
        }

        //Create new blocks
        // argmax Sizes
        state_type max = 0;
        int max_index = 0;
        Sizes[0] += B.mid - B.start;

        for (int i = 0; i < j; i++)
        {
            if (Sizes[i] > max) {
                max = Sizes[i];
                max_index = i;
            }
        }
        Sizes[0] -= B.mid - B.start;

        //Prefix sum Sizes
        for (int i = 0; i < j; i++)
        {
            Sizes[i+1] += Sizes[i];
        }

        int num_dirty = B.end - B.mid;
        state_type* dirty = new state_type[num_dirty];
        std::copy(loc2state + B.mid, loc2state + B.end, dirty);
        //Reorder states

        mCRL2log(mcrl2::log::debug) << "Rearranging states." << std::endl;

        for (state_type i = 0; i < num_dirty; i++) {
            state_type si = dirty[i];
            if(Sizes[state2block[i]] == 0) {
				//Impossible
                mCRL2log(mcrl2::log::info) << "Impossible" << si << " " << state2block[si] << std::endl;
				continue;
			}
            Sizes[state2block[i]] -= 1;
            int tmp = B.mid + Sizes[state2block[i]];
            loc2state[tmp] = si;
            state2loc[si] = tmp;
		}

        if (Sizes[0] != 0 || Sizes[j] != num_dirty) {
            mCRL2log(mcrl2::log::info) << "Assertion failed Sizes inccorrect" << std::endl;
        }
        // create new blocks
        state_type old_start = B.mid;
        state_type old_end = B.end;

        for (int i = 0; i < j; i++)
        {
            state_type newstart = old_start + Sizes[i];
            state_type newend = old_start + Sizes[i+1];
            if (i == 0) {
                newstart = B.start;
            }

            assert(newstart >= old_start);
            assert(newend <= old_end);
            if (i == max_index) {
                B.start = newstart;
                B.mid = newend;
                B.end = newend;
                // B.unstable_labels = std::unordered_set<label_type>();
                block_map[Bid] = B;
            }
            else {
                block_type newBid = block_map.size();
                block newB = block{ newstart, newend, newend };// std::unordered_set<label_type>()
                block_map.insert(std::pair<block_type, block>(newBid, newB));
                for (state_type locs = newB.start; locs < newB.end; locs++)
                {
                   blocks[loc2state[locs]] = newBid; 
                }
            }
		}

        delete[] Sizes;
        delete[] dirty;
        delete[] state2block;
        return block_map.size();
    }

    //Refine based on sigs
    void refine()
    {
        int iter = 0;
        int new_blocks = 0;
        int old_blocks = 0;
        mCRL2log(mcrl2::log::info) << "Start refinement" << std::endl;
		while (!worklist.empty())
		{
            iter += 1;
			block_type Bid = worklist.front();
			worklist.pop();

			block B = block_map[Bid];

            //Count states
			if (B.mid == B.end and B.end > B.start + 1)
			{
				//Block is clean (should not happen)
                mCRL2log(mcrl2::log::info) << "Block clean but in worklist!? should not happen." << std::endl;
            }
            else {
                mCRL2log(mcrl2::log::debug) << "start iter" << std::endl;

                old_blocks = block_map.size();
                split(Bid);
                mCRL2log(mcrl2::log::debug) << "new blocks created = " << block_map.size() - new_blocks << std::endl;
                for (int blockid = old_blocks; blockid < block_map.size(); blockid++)
                {
                    mCRL2log(mcrl2::log::debug) << "marking dirty " << blockid << std::endl;
                    block newB = block_map[blockid];

                    std::vector<observation_type> overlap;
                    for (state_type s = newB.start; s < newB.end; s++) {
                        for (custom_transition_type  t : pred[loc2state[s]]) {
                            if(blocks[t.second] == blockid) {
								overlap.push_back(t);
                            }
                            else {
                                mark_dirty(t.second);
                            }
                        }
                    }
                    for (auto s : overlap) {
						mark_dirty(s.second);
					}
                }
            }
        }
        mCRL2log(mcrl2::log::info) << "Done total blocks: \"" << block_map.size() << "\"" << std::endl;
    }
};
}
}
}
#endif

