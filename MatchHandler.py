from texasholdem import TexasHoldEm, Card, ActionType, PlayerAction
from copy import deepcopy
from typing import Optional, Sequence
from PokerBot import PokerBot

######################## MATCH HISTORY CLASS DEFINITION ########################

class MatchHistory:
  def __init__(self, game:TexasHoldEm):
    self.game = game # the TexasHodEm game that this history is recording
    self.num_players = game.max_players
    self.buyin = game.buyin
    self.big_blind = game.big_blind
    self.small_blind = game.small_blind
    self.hands = [] # HandHistory objects will be appended to this list
    self.winner = None # will be set after match conclusion
  
  def close_match(self): # must only be called after match conclusion
    # determine match winner
    for i in range(self.num_players):
      if (self.game.players[i].chips > 0):
        self.winner = i
        break

  # TODO: WRITE METHODS FOR OUTPUTTING HISTORY IN A MEANINGFUL WAY (e.g. to
  # strings) and for extracting certain features, maybe add automatic plotting
  # methods, etc.

######################### HAND HISTORY CLASS DEFINITION ########################

class HandHistory:
  def __init__(self, match:MatchHistory, game:TexasHoldEm):
    # IMPORTANT: Notice how we're passing in a pointer to the MatchHistory
    # object that will contain this instance. This will help users navigate the
    # history objects, and allow for both upwards and downwards navigation of
    # them.
    self.match = match # the MatchHistory object that will contain this object
    self.game = game # the TexasHoldEm game that this history is recording
    self.big_blind = game.bb_loc
    self.small_blind = game.sb_loc
    self.comm_cards = [] # Card objects will be appended to this list
    self.preflop = [] # PlayerDecision objects will be appended to this list
    self.flop = [] # PlayerDecision objects will be appended to this list
    self.turn = [] # PlayerDecision objects will be appended to this list
    self.river = [] # PlayerDecision objects will be appended to this list
    self.pockets = [[]] * game.max_players # list of all players' pockets
    self.chips_start = [None] * game.max_players # list of chips at hand start
    self.chips_end = [None] * game.max_players # list of chips at hand end
    # populate lists that are eligible for population
    for i in range(game.max_players):
      self.pockets[i] = deepcopy(game.get_hand(i))
      self.chips_start[i] = game.players[i].chips
  
  def close_hand(self): # must only be called after hand conclusion
    # record post-hand metadata
    self.comm_cards = deepcopy(self.game.board)
    for i in range(self.game.max_players):
      self.chips_end[i] = self.game.players[i].chips
    # record preflop, flop, turn, and river actions
    if (self.game.hand_history.preflop is not None):
      for i in range(len(self.game.hand_history.preflop.actions)):
        self.preflop.append(PlayerDecision(
          self, self.game.hand_history.preflop.actions[i]))
    if (self.game.hand_history.flop is not None):
      for i in range(len(self.game.hand_history.flop.actions)):
        self.flop.append(PlayerDecision(
          self, self.game.hand_history.flop.actions[i]))
    if (self.game.hand_history.turn is not None):
      for i in range(len(self.game.hand_history.turn.actions)):
        self.turn.append(PlayerDecision(
          self, self.game.hand_history.turn.actions[i]))
    if (self.game.hand_history.river is not None):
      for i in range(len(self.game.hand_history.river.actions)):
        self.river.append(PlayerDecision(
          self, self.game.hand_history.river.actions[i]))

####################### PLAYER DECISION CLASS DEFINITION #######################

class PlayerDecision:
  def __init__(self, hand:HandHistory, action:PlayerAction):
    self.hand = hand # the HandHistory object that will contain this object
    self.player_num = action.player_id # who made the decision
    self.decision = action.action_type.name # what decision was made
    self.amount = action.total # if decision is raise, records amount of raise

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
    self.match_hist = None # will be modified by class methods
    self.match_histories = []
    # send all bots a "new handler" flag
    for i in range(self.num_bots):
      self.bots[i].new_handler()
  
  # create a TexasHoldEm object using arguments as match parameters and assign
  # all bots to the match. only useful in conjunction with run_hand() method
  # for step-by-step matches or when called internally by run_match() method.
  def start_game(self, buyin:int, big_blind:int, small_blind:int):
    # instanciate poker table, and assign bots to it
    self.game = TexasHoldEm(buyin, big_blind, small_blind, self.num_bots)
    for i in range(self.num_bots):
      self.bots[i].set_game(self.game, i)
    # instanciate match history object
    self.match_hist = MatchHistory(self.game)
  
  # run through a single hand of play in the current match, if it exists and
  # hasn't already ended. can only be used after calling start_game() method,
  # or when called internally by run_match() method. records hand history, and
  # appends it to the current match history.
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
      # now that the match is done, close and append the MatchHistory object
      self.match_hist.close_match()
      self.match_histories.append(self.match_hist)
      self.match_hist = None
      return None
    # instanciate HandHistory object
    hand_hist = HandHistory(self.match_hist, self.game)
    # the hand is now actually underway, send all bots the "hand start" flag
    for i in range(self.num_bots):
      self.bots[i].hand_start()
    # actually run through the hand, bots make decisions until the hand ends
    while (self.game.is_hand_running()):
      self.bots[self.game.current_player].make_decision()
    # now that the hand has ended, send all bots the "hand end" flag
    for i in range(self.num_bots):
      self.bots[i].hand_end()
    # also close and append the HandHistory object
    hand_hist.close_hand()
    self.match_hist.hands.append(hand_hist)
    return None
  
  # create a TexasHoldEm object using arguments as match parameters, and run
  # through the entire match using the bots passed in at the handler's
  # instanciation. records match history.
  def run_match(self, buyin:int, big_blind:int, small_blind:int):
    # instanciate game and assign all bots to the game
    self.start_game(buyin, big_blind, small_blind)
    # run hands until match ends
    while (self.game.is_game_running()):
      self.run_hand()
    return None
  
  # perform run_match num_matches times. records match histories.
  def run_matches(self, num_matches:int, buyin:int, big_blind:int,
                  small_blind:int):
    for i in range(num_matches):
      self.run_match(buyin, big_blind, small_blind)
    return None
