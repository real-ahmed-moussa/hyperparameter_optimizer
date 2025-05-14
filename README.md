# Hyperparameter Optimizer

A lightweight and flexible Python library for hyperparameter tuning using metaheuristic techniques.  
Currently, the library includes **Particle Swarm Optimization (PSO)** and **Pattern Search (PS)** only.

Designed for Scikit-learn-compatible models, this package offers an easy-to-use interface for optimizing model performance in just a few lines.

![PyPI version](https://img.shields.io/pypi/v/hyperparameter-optimizer.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/hyperparameter-optimizer?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/hyperparameter-optimizer)

---

## 📦 Installation

Install from PyPI:

```bash
pip install hyperparameter-optimizer
```

---

## ⚙️ Features

- ✅ Metaheuristic optimization using Particle Swarm Optimization (PSO) and Pattern Search (PS)
- ✅ Supports continuous, integer, and categorical hyperparameters
- ✅ Compatible with any Scikit-learn estimator
- ✅ Custom scoring metrics
- ✅ Cross-validation built-in
- ✅ Verbose logging and full traceability

---

## 📚 API Reference

### `HyperparameterOptimizer`

```python
HyperparameterOptimizer(
                            obj_func,                       # machine learning model or pipeline being created
                            params,                         # dictionary of parameters for which the model is to be optimized.
                            scoring,                        # scoring metric (e.g., 'accuracy')
                            opt_type="max",                 # type of optimization: "max" for maximization (default) or "min" for minimization.
                            cv=5,                           # number of cross-validation folds
                            verbose=1                       # binary with a value of 1 (default) to show iteration information.
)
```

### `optimizePSO`

```python
optimizePSO(
            features,                       # training features.
            target,                         # training target.
            nParticles,                     # number of particles in the swarm.
            bounds,                         # list of (min, max) tuples or a categorical list of choices. 
                                            # if either min/max is non-integer, the algorithm shall consider them as floats (i.e., continuous).
            w=0.5,                          # inertia weight.
            c1=1,                           # cognitive weight.
            c2=1,                           # social weight.
            maxIter=20,                     # maximum number of iterations.
            mutation_prob=0.1               # mtation probability -- for discrete hyperparameters.
)
```

### `optimizePS`

```python
optimizePS(
            features,                       # training features.
            target,                         # training target.
            bounds,                         # list of (min, max) tuples or a categorical list of choices. 
                                            # if either min/max is non-integer, the algorithm shall consider them as floats (i.e., continuous).
            mesh_size_coeff=0.6,            # mesh size coefficient.
            acc_coeff=1,                    # acceleration/mesh expansion coefficient.
            contr_coeff=0.5,                # mesh contraction coefficient.
            search_method="gps",            # new points exploration method; either 'gps' or 'mads'.
            min_mesh_ratio=0.001,           # minimum mesh size as a percentage of parameter range.
            maxIter=25                      # maximum number of iterations.
)
```

---

## 🚀 Quickstart

Here's a basic example of how to use **Hyperparameter Optimizer** with a Scikit-learn model (e.g., `RandomForestClassifier`):

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from hyperparameter_optimizer import HyperparameterOptimizer

# Load data
X, y = load_iris(return_X_y=True)

# Define parameter space
param_space = {
    'n_estimators': [],
    'max_depth': [],
    'criterion': []
}

# Initialize the optimizer
optimizer = HyperparameterOptimizer(
                                        obj_func=RandomForestClassifier(),
                                        params=param_space,
                                        scoring='accuracy',
                                        opt_type='max',
                                        cv=3,
                                        verbose=1
                                    )

# Run optimization
# [1] Particle Swarm Optimization
particles, Gbest_history, Gbest_pos, Gbest_score = optimizer.optimizePSO(
                                                                            features = X,
                                                                            target = y,
                                                                            nParticles = 10,
                                                                            bounds = [(10, 1000), (1, 10), ['gini', 'entropy']],
                                                                            maxIter = 10
                                                                    )
print("Best Params:", Gbest_pos)
print("Best Score:", Gbest_score)

# [2] Pattern Search Optimization
Gbest_history2, Gbest_pos2, Gbest_score2 = optimizer.optimizePS(
                                                                    features = X,
                                                                    target = y,
                                                                    bounds = [(10, 1000), (1, 10), ['gini', 'entropy']],
                                                                    mesh_size_coeff = 0.1,
                                                                    acc_coeff = 1,
                                                                    contr_coeff = 0.50,
                                                                    maxIter = 10
                                                            )
print("Best Params:", Gbest_pos2)
print("Best Score:", Gbest_score2)

```

---

## 🧠 How It Works

This library leverages two optimization techniques—**Particle Swarm Optimization (PSO) and Pattern Search (PS)**—to explore the hyperparameter space effectively.

    Particle Swarm Optimization simulates a group of particles (candidate solutions) moving through the search space. Each particle adjusts its position based on its own best-known position and the best-known position among its peers, enabling a balance between exploration and exploitation as the swarm converges toward an optimal solution.

    Pattern Search is a derivative-free method that explores the neighborhood of a current solution by systematically evaluating a set of search directions. If an improved solution is found, the algorithm moves in that direction and repeats the process, enabling efficient local refinement without relying on gradient information.

Together, these methods provide robust global and local search capabilities for hyperparameter tuning.

---

## 📜 License

This project is licensed under the **MIT License**.  
© 2025 **Dr. Ahmed Moussa**

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📫 Contact

For feedback, bugs, or collaboration ideas:

- **GitHub**: [@real-ahmed-moussa](https://github.com/real-ahmed-moussa)  

---

## ⭐️ Show Your Support

If you find this project useful, consider giving it a ⭐️ on [GitHub](https://github.com/real-ahmed-moussa/hyperparameter-optimizer)!