from texasholdem import ActionType
from random import random, randint
from PokerBot import PokerBot

############################# TOM CLASS DEFINITION #############################

# class TomBot
# Tom is the least complex agent to be employed in our project. During play,
# he will always either call/check (50% chance) or make a small raise (50%
# chance). This restricted moveset allows JerryBot to train as efficiently as
# possible, and train well, when playing against TomBot.
class TomBot(PokerBot):
  # get the bot's parameters, useful for saving
  def get_parameters(self): # TomBot has no parameters to get
    pass
  
  # set the bot's parameters, useful for loading
  def set_parameters(self): # TomBot has no parameters to set
    pass

  # receives "round start" flag passed by MatchHandler
  def _round_start(self, start_chips:int):
    pass # TomBot has no "round start" operations to perform

  # receives "round end" flag passed by MatchHandler
  def _round_end(self, end_chips:int):
    pass # TomBot has no "round end" operations to perform
  
  # make a decision in the current game as the assigned player number
  def make_decision(self):
    #
    ########## COVER EXCEPTIONS ##########
    self._check_integrity()
    #
    ########## MAKE DECISION ##########
    # flip coin to determine decision
    if (random() > 0.5):
      ##### TRY TO RAISE #####
      min_raise = self.game.get_available_moves().raise_range.start
      max_raise = min(10 + self.game.get_available_moves().raise_range.start,
                      self.game.players[self.game.current_player].chips,
                      self.game.get_available_moves().raise_range.stop-1)
      # cover when min_raise exceeds max_raise
      if (min_raise > max_raise):
        # this means we don't have any chips left, so we can only call/check
        if (self.game.validate_move(action = ActionType.CALL)):
          self.game.take_action(ActionType.CALL)
          return None
        else:
          self.game.take_action(ActionType.CHECK)
          return None
      # if we get here, we can raise normally
      raise_amount = randint(min_raise, max_raise)
      # ensure raise is valid (should always be true)
      if (self.game.validate_move(action = ActionType.RAISE,
                                  value = raise_amount)):
        # raise is valid, perform raise and return
        self.game.take_action(ActionType.RAISE, raise_amount)
        return None
      else:
        # cover when attempted raise is invalid (should be impossible)
        raise Exception("TomBot's attempted raise is invalid (this should be " \
                        "impossible, if you're seeing this, something went " \
                          "seriously wrong)")
    else:
      ##### TRY TO CALL/CHECK #####
      if (self.game.validate_move(action = ActionType.CALL)):
        # call is valid, perform call and return
        self.game.take_action(ActionType.CALL)
        return None
      elif (self.game.validate_move(action = ActionType.CHECK)):
        # check is valid, perform check and return
        self.game.take_action(ActionType.CHECK)
        return None
      else:
        # cover when neither are allowed (should be impossible)
        raise Exception("TomBot is trying to call or check but cannot (this " \
                        "should be impossible, if you're seeing this, " \
                          "something went seriously wrong)")
