import sys
import os
# look one directory up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from texasholdem import ActionType, PlayerState
from PokerBot import PokerBot
from JerryVersions.JerryHelpers import get_win_prob
from copy import deepcopy
from numpy import ndarray, empty, arange, argsort, argmax, array, append, \
  copy, delete, sum
from numpy.random import choice, exponential
from random import randint
from typing import Optional, Sequence

######################## RATIONAL JERRY CLASS DEFINITION #######################

# class JerryBotRational
# Rational Jerry is a basic but powerful RL agent that places adaptive decision
# bounds on an informative hand evaluation metric, allowing it to manipulate its
# play depending on recent successes and failures. Motivation for Jerry's
# algorithms can be found in JerryBotDocs.md for the especially curious.
class JerryBotRational(PokerBot):
  def __init__(self, adaptive:bool = True, maturity:int = 1000,
               max_memory:int = 10000, rationality:float = 20.0,
               num_bootstraps:int = 1000):
    #
    ########## LONG TERM PARAMETERS ##########
    self.adaptive = adaptive  # whether to recompute bounds and update memory
    self.maturity = maturity  # how many hands before decisions aren't random
    self.max_memory = max_memory    # maximum number of rows in ..._mem arrays
    self.rationality = rationality  # tuning parameter for noise in decisions
    self.num_bootstraps = num_bootstraps  # for hand strength estimation
    self.c_m_mem = empty(0, dtype = float)  # call/check metric memory
    self.c_o_mem = empty(0, dtype = int)    # call/check outcome memory
    self.r_m_mem = empty(0, dtype = float)  # raise metric memory
    self.r_o_mem = empty(0, dtype = int)    # raise outcome memory
    self.b1 = float(0.0)  # the metric bound between fold and call/check
    self.b2 = float(0.0)  # the metric bound between call/check and raise
    self.age = int(0) # how many hands the bot has played, determines maturity
    #
    ########## SHORT TERM MEMORY ##########
    self.c_m_rec = empty(0, dtype = float)  # call/check metric recents
    self.r_m_rec = empty(0, dtype = float)  # raise metric recents
    self.start_chips = int(0) # will hold num chips before each hand start
  
  # return a deep copy of the bot's parameters corresponding to the parameter
  # names passed in `params` argument
  def get_parameters(self, params:Sequence[str] = []):
    # matches names to params
    names_to_params = {"b1":self.b1, "b2":self.b2, "adaptive":self.adaptive,
                       "age":self.age, "maturity":self.maturity,
                       "max memory":self.max_memory,
                       "rationality":self.rationality,
                       "num boostraps":self.num_bootstraps,
                       "call/check metric mem":self.c_m_mem,
                       "call/check outcome mem":self.c_o_mem,
                       "raise metric mem":self.r_m_mem,
                       "raise outcome mem":self.r_o_mem}
    # cover singleton list, return one element
    if (len(params) == 1):
      if ({params[0]}.isdisjoint(names_to_params.keys())):
        raise ValueError(f"JerryBotRational has no parameter \"{params[0]}\"")
      return deepcopy(names_to_params[params[0]])
    # cover empty list, return all
    if (len(params) == 0):
      params = ["b1", "b2", "adaptive", "age", "maturity", "max memory",
                "rationality", "num boostraps", "call/check metric mem",
                "call/check outcome mem", "raise metric mem",
                "raise outcome mem"]
    # initialize returned collection
    output = ()
    # iterate through params, add elements to output
    for param in params:
      if ({param}.isdisjoint(names_to_params.keys())):
        raise ValueError(f"JerryBotRational has no parameter \"{param}\"")
      output = output + (names_to_params[param],)
    # return deep copy of output
    return deepcopy(output)
  
  # set the bot's parameters
  def set_parameters(self, b1:Optional[float] = None, b2:Optional[float] = None,
                     adaptive:Optional[bool] = None, age:Optional[int] = None,
                     maturity:Optional[int] = None,
                     max_memory:Optional[int] = None,
                     rationality:Optional[float] = None,
                     num_bootstraps:Optional[int] = None,
                     c_m_mem:Optional[ndarray] = None,
                     c_o_mem:Optional[ndarray] = None,
                     r_m_mem:Optional[ndarray] = None,
                     r_o_mem:Optional[ndarray] = None):
    if b1 is not None:
      self.b1 = b1
    if b2 is not None:
      self.b2 = b2
    if adaptive is not None:
      self.adaptive = adaptive
    if age is not None:
      self.age = age
    if maturity is not None:
      self.maturity = maturity
    if max_memory is not None:
      self.max_memory = max_memory
    if rationality is not None:
      self.rationality = rationality
    if num_bootstraps is not None:
      self.num_bootstraps = num_bootstraps
    if c_m_mem is not None:
      self.c_m_mem = c_m_mem
    if c_o_mem is not None:
      self.c_o_mem = c_o_mem
    if r_m_mem is not None:
      self.r_m_mem = r_m_mem
    if r_o_mem is not None:
      self.r_o_mem = r_o_mem
    return None
  
  # helper to re-compute the decision boundaries
  def _update_bounds(self):
    #
    ########## COMPUTE B1 ##########
    # initialize the outcome sums
    outcome_sums = copy(self.c_o_mem)
    # at all indices, set to sum of all outcomes in greater or equal indices
    for i in range(len(outcome_sums)):
      outcome_sums[i] = sum(outcome_sums[arange(i, outcome_sums.shape[0])])
    # set b1 as the metric value that maximized this sum
    self.b1 = self.c_m_mem[argmax(outcome_sums)]
    #
    ########## COMPUTE B2 ##########
    # initialize the outcome sums
    outcome_sums = copy(self.r_o_mem)
    # at all indices, set to sum of all outcomes in greater or equal indices
    for i in range(len(outcome_sums)):
      outcome_sums[i] = sum(outcome_sums[arange(i, outcome_sums.shape[0])])
    # set b2 as the metric value that maximized this sum
    self.b2 = self.r_m_mem[argmax(outcome_sums)]
    return None

  # helper to add elements to memory and sort by metric values
  def _log_memory(self, outcome:int):
    # duplicate outcomes for every decision to be appended to memory
    c_o_rec = array([outcome] * self.c_m_rec.shape[0], dtype = int)
    r_o_rec = array([outcome] * self.r_m_rec.shape[0], dtype = int)
    # add recent decisions and outcoms to memory
    self.c_m_mem = append(self.c_m_mem, self.c_m_rec)
    self.r_m_mem = append(self.r_m_mem, self.r_m_rec)
    self.c_o_mem = append(self.c_o_mem, c_o_rec)
    self.r_o_mem = append(self.r_o_mem, r_o_rec)
    # sort memory by metric values
    c_sort_idxs = argsort(self.c_m_mem)
    r_sort_idxs = argsort(self.r_m_mem)
    self.c_m_mem = self.c_m_mem[c_sort_idxs]
    self.r_m_mem = self.r_m_mem[r_sort_idxs]
    self.c_o_mem = self.c_o_mem[c_sort_idxs]
    self.r_o_mem = self.r_o_mem[r_sort_idxs]
    # trim random observations down to memory maximums, if needed
    if (self.c_m_mem.shape[0] > self.max_memory):
      # randomly choose observations to remove
      idxs_to_remove = choice(arange(self.c_m_mem.shape[0]),
                              size = self.c_m_mem.shape[0] - self.max_memory,
                              replace = False)
      # remove the observations
      self.c_m_mem = delete(self.c_m_mem, idxs_to_remove)
      self.c_o_mem = delete(self.c_o_mem, idxs_to_remove)
    if (self.r_m_mem.shape[0] > self.max_memory):
      # randomly choose observations to remove
      idxs_to_remove = choice(arange(self.r_m_mem.shape[0]),
                              size = self.r_m_mem.shape[0] - self.max_memory,
                              replace = False)
      # remove the observations
      self.r_m_mem = delete(self.r_m_mem, idxs_to_remove)
      self.r_o_mem = delete(self.r_o_mem, idxs_to_remove)
    return None

  # receives "new handler" flag passed by MatchHandler
  def new_handler(self):
    pass # JerryBotRational has no "new handler" operations to perform

  # receives "hand start" flag passed by MatchHandler
  def hand_start(self):
    # record number of chips from before the hand
    self.start_chips = self.game.players[self.player_num].chips
    # recompute decision boundaries, if needed
    if (self.adaptive and self.age >= self.maturity):
      self._update_bounds()

  # receives "hand end" flag passed by MatchHandler
  def hand_end(self):
    # add hand data into long-term memory with associated outcome
    if (self.adaptive):
      self._log_memory(self.game.players[self.player_num].chips - \
                       self.start_chips)
    # reset short-term memory
    self.c_m_rec = empty(0, dtype = float)
    self.r_m_rec = empty(0, dtype = float)
    # update age
    self.age += 1
  
  # make a decision in the current game as the assigned player number
  def make_decision(self):
    #
    ########## COVER EXCEPTIONS ##########
    self._check_integrity()
    #
    ########## MAKE DECISION ##########
    # get hand evaluation metric irregardless of maturity
    num_players_in = int(0)
    for i in range(self.game.max_players):
      if (not (self.game.players[i].state == PlayerState.OUT or
               self.game.players[i].state == PlayerState.SKIP)):
        num_players_in += 1
    hand_strength = \
      get_win_prob(self.game, self.player_num, self.num_bootstraps) - \
        (1/num_players_in)
    # determine whether to make random decision or inteligent decision
    if (self.age >= self.maturity):
      #
      ########## MAKE INTELLIGENT DECISION ##########
      # add some rightwards bias so leftward movement of bounds is possible
      bias = exponential(1/self.rationality)
      if (hand_strength + bias < self.b1):
        ##### TRY TO CHECK #####
        if (self.game.validate_move(action = ActionType.CHECK)):
          # record decision, perform check, and return
          self.c_m_rec = append(self.c_m_rec, [hand_strength])
          self.game.take_action(ActionType.CHECK)
          return None
        ##### TRY TO FOLD #####
        if (self.game.validate_move(action = ActionType.FOLD)):
          # perform fold and return
          self.game.take_action(ActionType.FOLD)
          return None
        else:
          # cover when folding is invalid (this should be impossible)
          raise Exception("JerryBotRational's attempted fold is invalid " \
                          "(this should be impossible, if you're seeing " \
                            "this, something went seriously wrong)")
      elif (hand_strength + bias < self.b2):
        ##### TRY TO CALL/CHECK #####
        if (self.game.validate_move(action = ActionType.CALL)):
          # record decision, perform call, and return
          self.c_m_rec = append(self.c_m_rec, [hand_strength])
          self.game.take_action(ActionType.CALL)
          return None
        elif (self.game.validate_move(action = ActionType.CHECK)):
          # record decision, perform check, and return
          self.c_m_rec = append(self.c_m_rec, [hand_strength])
          self.game.take_action(ActionType.CHECK)
          return None
        else:
          # cover when neither are allowed (should be impossible)
          raise Exception("JerryBotRational is trying to call or check but " \
                          "cannot (this should be impossible, if you're " \
                            "seeing this, something went seriously wrong)")
      else:
        ##### TRY TO RAISE #####
        min_raise = self.game.get_available_moves().raise_range.start
        max_raise = min(10 + self.game.get_available_moves().raise_range.start,
                        self.game.players[self.game.current_player].chips,
                        self.game.get_available_moves().raise_range.stop-1)
        # cover when min_raise exceeds max_raise
        if (min_raise > max_raise):
          # this means we don't have any chips left, so we can only call/check
          if (self.game.validate_move(action = ActionType.CALL)):
            # record decision, perform call, and return
            self.c_m_rec = append(self.c_m_rec, [hand_strength])
            self.game.take_action(ActionType.CALL)
            return None
          else:
            # record decision, perform check, and return
            self.c_m_rec = append(self.c_m_rec, [hand_strength])
            self.game.take_action(ActionType.CHECK)
            return None
        # if we get here, we can raise normally
        raise_amount = randint(min_raise, max_raise)
        # ensure raise is valid (should always be true)
        if (self.game.validate_move(action = ActionType.RAISE,
                                    value = raise_amount)):
          # record decision, perform raise, and return
          self.r_m_rec = append(self.r_m_rec, [hand_strength])
          self.game.take_action(ActionType.RAISE, raise_amount)
          return None
        else:
          # cover when attempted raise is invalid (should be impossible)
          raise Exception("JerryBotRational's attempted raise is invalid " \
                          "(this should be impossible, if you're seeing " \
                            "this, something went seriously wrong)")
    else:
      #
      ########## MAKE "RANDOM" DECISION (COPIED FROM TOMBOT) ##########
      # determine decision
      if (randint(0, 1) == 1):
        ##### TRY TO RAISE #####
        min_raise = self.game.get_available_moves().raise_range.start
        max_raise = min(5 + self.game.get_available_moves().raise_range.start,
                        self.game.players[self.game.current_player].chips,
                        self.game.get_available_moves().raise_range.stop-1)
        # cover when min_raise exceeds max_raise
        if (min_raise > max_raise):
          # this means we don't have any chips left, so we can only call/check
          if (self.game.validate_move(action = ActionType.CALL)):
            # record decision, perform call, and return
            self.c_m_rec = append(self.c_m_rec, [hand_strength])
            self.game.take_action(ActionType.CALL)
            return None
          else:
            # record decision, perform check, and return
            self.c_m_rec = append(self.c_m_rec, [hand_strength])
            self.game.take_action(ActionType.CHECK)
            return None
        # if we get here, we can raise normally
        raise_amount = randint(min_raise, max_raise)
        # ensure raise is valid (should always be true)
        if (self.game.validate_move(action = ActionType.RAISE,
                                    value = raise_amount)):
          # record decision, perform raise, and return
          self.r_m_rec = append(self.r_m_rec, [hand_strength])
          self.game.take_action(ActionType.RAISE, raise_amount)
          return None
        else:
          # cover when attempted raise is invalid (should be impossible)
          raise Exception("JerryBotRational's attempted raise is invalid " \
                          "(this should be impossible, if you're seeing " \
                            "this, something went seriously wrong)")
      else:
        ##### TRY TO CALL/CHECK #####
        if (self.game.validate_move(action = ActionType.CALL)):
          # record decision, perform call, and return
          self.c_m_rec = append(self.c_m_rec, [hand_strength])
          self.game.take_action(ActionType.CALL)
          return None
        elif (self.game.validate_move(action = ActionType.CHECK)):
          # record decision, perform check, and return
          self.c_m_rec = append(self.c_m_rec, [hand_strength])
          self.game.take_action(ActionType.CHECK)
          return None
        else:
          # cover when neither are allowed (should be impossible)
          raise Exception("JerryBotRational is trying to call or check but " \
                          "cannot (this should be impossible, if you're " \
                            "seeing this, something went seriously wrong)")
