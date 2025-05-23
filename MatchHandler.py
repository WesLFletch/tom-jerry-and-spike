from texasholdem import TexasHoldEm
from typing import Sequence
from PokerBot import PokerBot

######################## MATCH HANDLER CLASS DEFINITION ########################

# class that automates running Texas Holdem' matches and recording match
# histories. should be useful regarding the training and evaluations
# of our agents, as well as letting users play against them.
class MatchHandler:
  def __init__(self, bots:Sequence[PokerBot]):
    # cover too few bots (at least two required)
    if (len(bots) < 2):
      raise Exception("too few bots passed in, at least two required")
    self.bots = bots
    self.num_bots = len(self.bots)
    self.match_histories = None # TODO: IMPLEMENT ME.
    # send all bots a "new handler" flag
    for i in range(self.num_bots):
      self.bots[i].new_handler()
  
  # create a TexasHoldEm object using arguments as match parameters and assign
  # all bots to the match. only useful in conjunction with run_hand() method
  # for step-by-step matches or when called internally by run_match() method.
  def start_game(self, buyin:int, big_blind:int, small_blind:int):
    self.game = TexasHoldEm(buyin, big_blind, small_blind, self.num_bots)
    for i in range(self.num_bots):
      self.bots[i].set_game(self.game, i)
  
  # run through a single hand of play in the current match, if it exists and
  # hasn't already ended. can only be used after calling start_game() method,
  # or when called internally by run_match() method.
  def run_hand(self):
    #
    ########## COVER EXCEPTIONS ##########
    if (self.game is None):
      raise Exception("no game has been created, call start_game() method")
    if (not self.game.is_game_running()):
      raise Exception("current game has ended, call start_game() method")
    if (self.game.is_hand_running()):
      raise Exception("there is a hand already running, if you see this, " \
                      "something went seriously wrong")
    #
    ########## RUN THE HAND ##########
    # start the hand
    self.game.start_hand()
    # safeguard against the last hand being the one that ended the match
    if (not self.game.is_game_running()):
      return None
    # the hand is now actually underway, send all bots the "hand start" flag
    for i in range(self.num_bots):
      self.bots[i].hand_start()
    # actually run through the hand, bots make decisions until the hand ends
    while (self.game.is_hand_running()):
      self.bots[self.game.current_player].make_decision()
    # now that the hand has ended, send all bots the "hand end" flag
    for i in range(self.num_bots):
      self.bots[i].hand_end()
    return None
  
  # create a TexasHoldEm object using arguments as match parameters, and run
  # through the entire match using the bots passed in at the handler's
  # instanciation. records the match history if desired.
  # TODO: IMPLEMENT RECORDING MATCH HISTORY
  def run_match(self, buyin:int, big_blind:int, small_blind:int,
                record_history:bool = False):
    # instanciate game and assign all bots to the game
    self.start_game(buyin, big_blind, small_blind)
    # run hands until match ends
    while (self.game.is_game_running()):
      self.run_hand()
    return None
  
  # perform run_match num_matches times. records the match history if desired.
  # TODO: IMPLEMENT RECORDING MATCH HISTORY
  def run_matches(self, num_matches:int, buyin:int, big_blind:int,
                  small_blind:int, record_history:bool = False):
    for i in range(num_matches):
      self.run_match(buyin, big_blind, small_blind, record_history)
    return None
