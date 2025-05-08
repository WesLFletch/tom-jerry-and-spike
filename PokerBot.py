from abc import ABC, abstractmethod
from texasholdem import TexasHoldEm

########################## POKER BOT CLASS DEFINITION ##########################

# PokerBot is the parent class to all our other bots. this way, MatchHandler can
# operate off subclasses of this parent class (hopefully) and we can have a
# standardized and modular way to implement our other bots.
class PokerBot(ABC):
  def __init__(self):
    pass

  # helper function that should be called by child classes before performing
  # methods that relying upon having a game object loaded. returns nothing,
  # raises exception if game object is not loaded.
  def _check_integrity(self):
    # cover self.game not being set
    if (self.game is None):
      raise Exception( \
        "TexasHoldEm object not set, call set_game() method first")
    # cover self.game not having hand running
    if (not (self.game.is_game_running() and self.game.is_hand_running())):
      raise Exception( \
        f"player {self.player_num}'s game object does not have hand running")
    # cover being called to make decision when not bot's turn
    if (self.game.current_player != self.player_num):
      raise Exception(\
        f"player {self.player_num} called to play out of turn order (it is " \
          f"player {self.game.current_player}'s turn)")
    return None
  
  # sets the match that the bot is playing, as well as which player number it
  # is. allows bots to switch what game they're playing in, or switch which
  # player number they are.
  def set_game(self, game:TexasHoldEm, player_num:int):
    # cover player_num too large or too small
    if (player_num + 1 > game.max_players or player_num < 0):
      raise ValueError("argument player_num not applicable for game")
    # set game and player num
    self.game = game
    self.player_num = player_num
    return None
  
  # shorthand helper function to get the bot's chips for those curious users
  def get_num_chips(self):
    # cover self.game not being set
    if (self.game is None):
      raise Exception( \
        "TexasHoldEm object not set, call set_game() method first")
    return self.game.players[self.player_num].chips
  
  # shorthand helper function to get the bot's cards for those curious users
  def get_cards(self):
    # cover self.game not being set
    if (self.game is None):
      raise Exception( \
        "TexasHoldEm object not set, call set_game() method first")
    return self.game.get_hand(self.player_num)
  
  # returns the bot's parameters, so that saving is more efficient. must be
  # implemented by child classes.
  @abstractmethod
  def get_parameters(self):
    pass
  
  # sets the bot's parameters, so that loading is more efficient. must be
  # implemented by child classes.
  @abstractmethod
  def set_parameters(self):
    pass

  # receives "round start" flag passed by MatchHandler to perform start-of-round
  # updates, if needed. must be implemented by child classes.
  @abstractmethod
  def _round_start(self, start_chips:int):
    pass

  # receives "round end" flag passed by MatchHandler to perform end-of-round
  # updates, if needed. must be implemented by child classes.
  @abstractmethod
  def _round_end(self, end_chips:int):
    pass
  
  # called to have the bot make a deision. must be implemented by child classes.
  @abstractmethod
  def make_decision(self):
    pass
