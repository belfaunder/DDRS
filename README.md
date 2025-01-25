Code for the algortihms described in paper Demand Steering in a Last-Mile Delivery Problem with Home and Pickup Point Delivery Options.

## Technologies: ##
Project is created with:

- Python 3.6
- git (apt-get install git)
- Gurobi 8.1.1 (https://packages.gurobi.com/8.1/Gurobi-8.1.1-win64.msi)


## Installation: ##

1. Clone repository: `git clone https://github.com/belfaunder/DDRS.git'
2. Install conda: `https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html`
3. Install Gurobi >= 8.1.1: `https://support.gurobi.com/hc/en-us/articles/12872889819921-How-do-I-use-conda-to-install-Gurobi-in-Anaconda`
4. Install dependencies: conda -r install requirementx.txt
5. Run test file ddrs/src/main/discount_strategy/test/branchAndBound_test.py
6. If you want to use scip benchmark files, you need to install pyscipopt~=8.0.0
7. If you want to use Concorde as a solver, you need to use Python wrapper for Concorde, refer to git@github.com:jvkersch/pyconcorde.git