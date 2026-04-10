# Logistic Regression from Scratch with numpy
This project implements logistic regression from scratch using NumPy, without using machine learning libraries such as scikit-learn.

The goal is to understand how a **binary classification model** works, including prediction, error calculation, **gradient-based parameter updates**, and the effect of **learning rate** and **feature scaling**.

This project focuses not only on implementation, but also on understanding how gradients are influenced by both prediction error and input features.

## What I learn
- How logistic regression turns weighted features into probabilities
- Why gradients are computed from error and input features
- How learning rate affects training stability
- Why feature scaling matters in multi-feature models

## Project Structure
- `main.py`: entry point for training and testing
- `model.py`: core logistic regression implementation
- `utils.py`: helper functions
- `experiments`: notes on experiments and observations

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Experiments
I conducted several experiments to better understand how logistic regression behaves:
- The effect of different learning rates on convergence.
- The impact of feature scaling in multi-featue input.
- Model behavior under extreme parameter updates.

See details in: `experiments/lr_experiments.md`