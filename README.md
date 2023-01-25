## Technologies: ##
Project is created with:

- Python 3.6
- git (apt-get install git)
- Gurobi 8.1.1 (https://packages.gurobi.com/8.1/Gurobi-8.1.1-win64.msi)
- GNU parallel (apt-get install parallel)
- NumPy 1.16.4 (pip3 install numpy)


## Installation: ##

1. Clone repository: `git clone https://belfaunder@bitbucket.org/belfaunder/octsp.git'
2. Install dependencies: `mvn validate` (?)
3. Copy `./etc/local.conf_example` to `./`, rename to `local.conf` and update its contents (?)
4. Compile the code: `mvn package`(?)
5. Run the example benchmark mip.sh under ./scripts/MIP/set1/ (?)