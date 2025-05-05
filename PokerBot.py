from abc import ABC, abstractmethod
from texasholdem import TexasHoldEm

########################## POKER BOT CLASS DEFINITION ##########################

# PokerBot is the parent class to all our other bots. this way, MatchHandler can
# operate off subclasses of this parent class (hopefully) and we can have a
# standardized and modular way to implement our other bots.
class PokerBot(ABC):
  def __init__(self, game:TexasHoldEm, player_num:int):
    if (player_num + 1 > game.max_players):
      raise ValueError("argument player_num too large for game")
    self.game = game
    self.player_num = player_num
  
  # sets the match that the bot is playing, as well as which player number it
  # is. allows bots to switch what game they're playing in, or switch which
  # player number they are.
  def set_game(self, game:TexasHoldEm, player_num:int):
    # cover player_num too large
    if (player_num + 1 > game.max_players):
      raise ValueError("argument player_num too large for game")
    # set game and player num
    self.game = game
    self.player_num = player_num
    return None
  
  # shorthand helper function to get the bot's chips for those curious users
  def get_num_chips(self):
    return self.game.players[self.player_num].chips
  
  # shorthand helper function to get the bot's cards for those curious users
  def get_cards(self):
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
  
  # called to have the bot make a deision. must be implemented by child classes.
  @abstractmethod
  def make_decision(self):
    pass