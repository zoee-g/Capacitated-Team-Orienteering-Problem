# Capacitated-Team-Orienteering-Problem

This is a solver written in Python for the Capacitated Team Orienteering Problem (CTOP) in a last-mile logistics context. Developed as a university assignment for the course: Operations Research and Computational Intelligence.


## Problem Description

A courier company has more orders than its fleet can deliver in a single day. The goal is to select which customers to serve and in what order, maximizing total profit while respecting vehicle capacity and driver shift time constraints.

Two scenarios are solved:

- **Problem 1 (Standard CTOP):** No mandatory customers where we freely select any subset of customers to maximize profit.
- **Problem 2 (Mandatory CTOP):** Certain premium customers must be included in the solution.

## Constraints

- All vehicles start and end at the Depot (Node 0)
- Each customer can be served at most once by one vehicle
- Total demand per route <= vehicle capacity (Q)
- Total travel time per route <= shift time limit (T_max)
- Mandatory nodes (Problem 2) must all be included

## Project Structure
```
├── data/
│   └── ctop_main_instance.txt
├── solutions/
│   ├── solution_no_mandatory.txt
│   └── solution_mandatory.txt
├── src/
│   ├── Main.py
│   ├── Parser.py
│   ├── Solver.py
│   ├── SolutionValidator.py
│   └── SolutionPlotter.py
└── README.md
```

## Algorithm

- **Greedy Construction:** Best insertion heuristic based on profit-to-cost ratio. Mandatory nodes are inserted first (Problem 2).
- **Improvement Phase:** *(to be updated)*

## Requirements

- Python 3.10+
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install matplotlib scikit-learn
```

## Usage

```bash
cd src
python Main.py
```

## Team Members

- Zoi Giagli
- Vasiliki Zagoraiou
