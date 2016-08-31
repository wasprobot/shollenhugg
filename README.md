_“Model the mind, not the world!”_

## Neural Networks:
* Translate real-world data (images, sounds, text, time-series etc.) into vectors
* Recognize patterns in these vectors, numerically
* Group unlabeled data according to similarities among example inputs, and
* They classify data when they have a labeled dataset to train on

## Reinforcement Learning:
* Learn based on reward: if an action has a positive impact on the progress toward the goal, continue doing it, otherwise course-correct. Repeat until the goal is achieved
* The actions are based on short- and long-term rewards
* Q-function: maps a state and a proposed-action, to the probable reward(s):
```
Q (state, action) {
  return reward;
}
```
* The prediction of the best path for the agent to take (using a series of Q-function calls) is called a policy

## Neural Networks (agents. A variation is called convolutional networks):
* **In supervised learning:** A NN will label the input parameters. E.g., is this given image a face or not
* **In reinforcement learning:** A NN can take possible “legal” actions and rate them with a weight for success in achieving in the final goal

## Continuous Attributes (e.g., Age, Distance etc.)
* Breakdown these attributes into discrete ranges (e.g., 0-20; 21-30 etc.)
* We never repeat the same question about an attribute in a decision tree

_Overfitting?_

_Pruning?_

## Catechism:
* What kind of information?
* How is it represented?
* How is it used?
* How much knowledge is required?
* What exactly???

## Rules-based expert systems
* Forward Chaining Rule-Based Expert System (e.g., one that can take observations about an animal and deduce what animal it is)
* Backward Chaining.. (e.g., when the system uses the same rules, to answer a question like “Is this animal a Cheetah?”) - * * Deduction System
* Question: Do Rules-based systems have anything to do with common sense?
* We don’t know!
