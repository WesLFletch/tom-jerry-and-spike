from texasholdem import ActionType
from PokerBot import PokerBot

########################## USER BOTS CLASS DEFINITION ##########################

# class that prompts user for decision whenever a decision needs to be made.
# allows real people to play against our agents in real matches.
class UserBot(PokerBot):
  # get the bot's parameters, useful for saving
  def get_parameters(self): # User Bot has no parameters to get
    pass
  
  # set the bot's parameters, useful for loading
  def set_parameters(self): # User Bot has no parameters to set
    pass

  # prompt to user to make a decision
  def make_decision(self):
    #
    ########## COVER EXCEPTIONS ##########
    # cover self.game not having hand running
    if (not (self.game.is_game_running() and self.game.is_hand_running())):
      raise Exception("TexasHoldEm object does not have hand running")
    # cover being called to make decision when not TomBot's turn
    if (self.game.current_player != self.player_num):
      raise Exception(\
        f"TomBot (player {self.player_num}) called to play out of turn order " \
          f"(it is player {self.game.current_player}'s turn)")
    #
    ########## MAKE DECISION ##########
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
    # try to make user_decision
    if (user_decision == "c"):
      # try to call/check
      if (self.game.validate_move(action = ActionType.CALL)):
        self.game.take_action(ActionType.CALL)
        return None
      elif (self.game.validate_move(action = ActionType.CHECK)):
        self.game.take_action(ActionType.CHECK)
        return None
      else:
        raise Exception("illegal decision made")
    elif (user_decision == "r"):
      # try to raise
      user_raise_amount = int(input("How much to raise by?:"))
      if (self.game.validate_move(action = ActionType.RAISE,
                                  value = user_raise_amount)):
        self.game.take_action(ActionType.RAISE, user_raise_amount)
        return None
      else:
        raise Exception("illegal decision made")
    elif (user_decision == "f"):
      # try to fold
      if (self.game.validate_move(action = ActionType.FOLD)):
        self.game.take_action(ActionType.FOLD)
        return None
      else:
        raise Exception("illegal decision made")
    elif (user_decision == "a"):
      # try to go all in
      if (self.game.validate_move(action = ActionType.ALL_IN)):
        self.game.take_action(ActionType.ALL_IN)
        return None
      else:
        raise Exception("illegal decision made")
    else:
      raise ValueError("invalid user input")
