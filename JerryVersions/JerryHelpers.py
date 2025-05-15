from texasholdem import TexasHoldEm, Card, PlayerState
from texasholdem.evaluator import evaluate
from numpy import zeros, mean
from random import randint

##################### HELPERS FOR JERRY'S VARIOUS VERSIONS #####################

# helper to convert Card to int for hand strength calculation
def card_to_int(card:Card):
  suit_dict = {1:0, 2:13, 4:26, 8:39}
  return suit_dict[card.suit] + card.rank

# helper to convert int to Card for hand strength calculation
def int_to_card(num:int):
  rank_dict = {0:"2", 1:"3", 2:"4", 3:"5", 4:"6", 5:"7", 6:"8", 7:"9", 8:"T",
                9:"J", 10:"Q", 11:"K", 12:"A"}
  suit_dict = {0:"s", 1:"h", 2:"d", 3:"c"}
  return Card(rank_dict[num % 13] + suit_dict[num // 13])

# helper to get the bootstrapped probability of winning the hand at showdown
def get_win_prob(game:TexasHoldEm, player_num:int, num_bootstraps:int):
  # initialize vector to hold bootstrap results
  bootstrap_results = zeros(num_bootstraps)
  # get bot's hand
  my_hand = game.get_hand(player_num)
  # perform the bootstrap loop
  for i in range(num_bootstraps):
    #
    ########## BOOTSTRAP COMMUNITY CARDS ##########
    # populate community cards with known community cards
    comm_cards = []
    for card in game.board:
      comm_cards.append(card)
    # populate known_cards with all known cards
    known_cards = set()
    for card in my_hand:
      known_cards.add(card_to_int(card))
    for card in comm_cards:
      known_cards.add(card_to_int(card))
    # populate remaining cards randomly
    while (len(comm_cards) < 5):
      while (True):
        # the idea: generate a random card over and over until it's new
        new_card = randint(0, 51)
        if (known_cards.isdisjoint({new_card})):
          known_cards.add(new_card)
          comm_cards.append(int_to_card(new_card))
          break
    #
    ########## BOOTSTRAP OPPONENT POCKETS ##########
    # first, count the number of opponents who haven't folded
    num_ops = int(-1) # starts here to remove ourselves from the count
    for j in range(game.max_players):
      if (not (game.players[j].state == PlayerState.OUT or
               game.players[j].state == PlayerState.SKIP)):
        num_ops += 1
    # next, populate their pockets
    ops_pockets = [[]] * num_ops
    for pocket in ops_pockets:
      while (len(pocket) < 2):
        # the idea: generate a random card over an over until it's new
        while (True):
          new_card = randint(0, 51)
          if (known_cards.isdisjoint({new_card})):
            known_cards.add(new_card)
            pocket.append(int_to_card(new_card))
            break
    # determine hand ranks of all players
    my_hand_rank = evaluate(my_hand, comm_cards)
    ops_hand_ranks = [0] * num_ops
    for j in range(num_ops):
      ops_hand_ranks[j] = evaluate(ops_pockets[j], comm_cards)
    # determine winner, update bootstrap_results
    if (my_hand_rank < min(ops_hand_ranks)):
      bootstrap_results[i] = 1
  # return the win probability
  return mean(bootstrap_results)
