# Tom, Jerry, and Spike: Showcasing Reinforcement Learning Algorithms for Texas Holdem' Poker

## Overview

This is a collaborative project by:
- Kiratiya Barrera
- Wesley Fletcher (to whom correspondance should be addressed at weslfletch@gmail.com)
- Daniel Hu
- Shouro Shuvit
- Tanish Thaker

It is a revisit to a class project for STAT-421: Machine Learning at Texas A&M University, to refine and improve upon our past work and showcase it in a intuitive and expandable environment.

We have primarily employed the following libraries:

- [`texasholdem`](https://github.com/SirRender00/texasholdem) by SirRender00 for the underlying Texas Holdem' match structure
- [`numpy`](https://numpy.org/) for performing computations

These resources have been instrumental to the success of our project.

## Quick Start Guide:

### Classes

This repository contains code for the following classes:

- `PokerBot`, the parent class for all bots
- `MatchHandler`, a high-level wrapper object for automating matches of Texas Holdem' poker with the `-Bot` objects in this repository
- `UserBot`, a `PokerBot` child class that allows for users to participate in matches of Texas Holdem' at the command line in the `MatchHandler` environment
- `TomBot`, a `PokerBot` child class that will either call/check or make a small raise when prompted to make a decision
- `JerryBotRational`, a `PokerBot` child class that dynamically plays using a novel ad-hoc reinforcement learning algorithm

### Running a match

To run a match, first instanciate all desired `PokerBot` child classes, and then instanciate a `MatchHandler` object with the argument of a list of these instanciated objects.

```python
from texasholdem import TexasHoldEm
import numpy as np
from TomBot import TomBot
from UserBot import UserBot
from JerryVersions.JerryBotRational import JerryBotRational
from MatchHandler import MatchHandler

Jerry = JerryBotRational()
Tom = TomBot()
User = UserBot()

my_handler = MatchHandler([Jerry, Tom, User])
```

Calling the `.run_match()` or `.run_matches()` methods on the `MatchHandler` instance will step through an entire match with the corresponding `PokerBot` child class instances. Match history can be accessed in the `.match_histories` data member of the `MatchHandler` instance. `PokerBot` child class instances can be passed into different `MatchHandler` objects, so that training can be done prior to testing.

## About Jerry

Whenever prompted to make a decision, `JerryBotRational` estimates its probability of winning the hand at showdown using Monte-Carlo simulations, then makes its decision by comparing this probability with an adaptive threshold set by approximating the intercept of the regression of expected earnings on probability of winning. An additional non-negative noise term on the Monte-Carlo probability estimate allows for slightly more aggressive play and experimentation.

Before `maturity` many rounds are played by Jerry, decision making is entirely random for the purposes of populating the regression space of earnings on probability of winning. `maturity` is set at Jerry's instanciation.

Details on the algorithms of `JerryBotRational` can be found in `JerryBotDocs.md`.
