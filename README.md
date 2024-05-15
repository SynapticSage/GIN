# DREAM Olfaction Challenge

This repository contains code for exploring and analyzing data for the DREAM Olfaction Challenge. The goal of the challenge is to develop models that can predict how close two mixtures of molecules are in the odor perceptual space using physical and chemical features.

Below, providing an abbreviated challenge description from [Synapse website](https://www.synapse.org/#!Synapse:syn53470621/wiki/627282):

```markdown
## Background

Olfaction—the sense of smell—is the least understood of our senses. We use it constantly in our daily lives—choosing food that is not spoiled, as an early-warning sign of a gas leak or a fire, and in the enjoyment of perfume and wine. Recent advances have helped predict what a molecule will smell like, given its chemical structure. This is known as the stimulus-percept problem, which was solved long ago for color vision and tone hearing.

For this challenge, we are providing a large published training set of 500 mixtures measurements obtained from 3 publications, and an unpublished test set of 46 equi-intense mixtures of 10 molecules whose distance was rated by 35 human subjects.

## Task

The goal of the DREAM Olfaction Challenge is to find models that can predict how close two mixtures of molecules are in the odor perceptual space (on a 0-1 scale, where 0 is total overlap and 1 is the furthest away) using physical and chemical features.

## Data Files

- `Dragon_Descriptors.csv`: Provides the physico-chemical Dragon features for all molecules in the training, leaderboard, and test sets, plus extra molecules if needed.
- `Mordred_Descriptors.csv`: Provides the physico-chemical Mordred features for all molecules in the training, leaderboard, and test sets.
- `Morgan_Fingerprint.csv`: Provides the physico-chemical Morgan fingerprints for all molecules in the training, leaderboard, and test sets.
- `Mixure_Definitions_Training_set.csv`: Indicates the composition of each mixture in the training set.
- `TrainingData_mixturedist.csv`: Contains distance measurements between pairs of mixtures in the training set.
- `Mixure_Definitions_Leaderboard_set.csv`: Indicates the composition of each mixture in the leaderboard set.
- `Leaderboard_set_Submission_form.csv`: Contains pairs of mixtures for the leaderboard set and a column for your prediction.
- `Mixure_Definitions_test_set.csv`: Indicates the composition of each mixture in the test set.
- `Test_set_Submission_form.csv`: Contains pairs of mixtures for the test set and a column for your prediction.
```

## Installation

To install the package, run:

```bash
pip install .

