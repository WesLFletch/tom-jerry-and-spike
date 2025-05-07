from texasholdem import ActionType
from PokerBot import PokerBot

########################## USER BOTS CLASS DEFINITION ##########################

# class that prompts user for decision whenever a decision needs to be made.
# allows real people to play against our agents in real matches.
class UserBot(PokerBot):
  # get the bot's parameters, useful for saving
  def get_parameters(self): # UserBot has no parameters to get
    pass
  
  # set the bot's parameters, useful for loading
  def set_parameters(self): # UserBot has no parameters to set
    pass

  # receives "round end" flag passed by MatchHandler
  def _round_end(self):
    pass # UserBot has no "round end" operations to perform

  # prompt to user to make a decision
  def make_decision(self):
    #
    ########## COVER EXCEPTIONS ##########
    self._check_integrity()
    #
    ########## PROMPT USER TO MAKE DECISION ##########
    # show user game state information and prompt to input decision
    opponent_chips = []
    for i in range(self.game.max_players):
      if (i != self.player_num):
        opponent_chips.append(self.game.players[i].chips)
    print()
    print(f"Your Chips: {self.game.players[self.player_num].chips}")
    print(f"Opponents' Chips: {opponent_chips}")
    print(f"Chips to Call: {self.game.chips_to_call(self.player_num)}")
    print(f"Minimum Raise: {self.game.get_available_moves().raise_range.start}")
    print(f"Your Cards: {self.game.get_hand(self.player_num)}")
    print(f"Community Cards: {self.game.board}")
    user_decision = str(input("What will you do (c/r/f/a)?:"))
    #
    ########## TRY TO MAKE USER DECISION ##########
    # branch on user decision
    if (user_decision == "c"):
      ##### TRY TO CALL/CHECK #####
      if (self.game.validate_move(action = ActionType.CALL)):
        self.game.take_action(ActionType.CALL)
        return None
      elif (self.game.validate_move(action = ActionType.CHECK)):
        self.game.take_action(ActionType.CHECK)
        return None
      else:
        raise Exception("illegal decision made")
    elif (user_decision == "r"):
      ##### TRY TO RAISE #####
      user_raise_amount = int(input("How much to raise by?:"))
      if (self.game.validate_move(action = ActionType.RAISE,
                                  value = user_raise_amount)):
        self.game.take_action(ActionType.RAISE, user_raise_amount)
        return None
      else:
        raise Exception("illegal decision made")
    elif (user_decision == "f"):
      ##### TRY TO FOLD #####
      if (self.game.validate_move(action = ActionType.FOLD)):
        self.game.take_action(ActionType.FOLD)
        return None
      else:
        raise Exception("illegal decision made")
    elif (user_decision == "a"):
      ##### TRY TO GO ALL IN #####
      if (self.game.validate_move(action = ActionType.ALL_IN)):
        self.game.take_action(ActionType.ALL_IN)
        return None
      else:
        raise Exception("illegal decision made")
    else:
      raise ValueError("invalid user input")
