# **Bridging structure and function: A model of sequence learning and prediction in primary visual cortex**
Christian Klos<sup>1,2</sup>, Daniel Miner<sup>1</sup>, Jochen Triesch<sup>1</sup>

**1** *Frankfurt Institute for Advanced Studies, Frankfurt am Main, Germany.*

**2** *Neural Network Dynamics and Computation, Institute of Genetics, University of Bonn, Bonn, Germany.*

---

This repository contains code to reproduce the results of our [article](https://doi.org/10.1371/journal.pcbi.1006187) on sequence learning in primary visual cortex. It is structured as follows:

- *pars.py, pars_exp.py* specify the parameters of our LIF-SORN model and of the sequence learning experiments.
- *lifsorn.py* defines the function that creates the LIF-SORN model.
- *experiments.py* contains wrapper functions for the different experiments.
- *run_seqlearn.py* performs the sequence learning experiments. It creates the data underlying Figures 4-9 and stores it in the *data* subdirectory. See the comments in it for more details.
- *analysis.py* contains helper functions for the analysis of the simulation data.
- *figs1-3.ipynb* runs the network simulation without input and creates Figures 1-3.
- *figX.ipynb* takes the sequence learning data, analyses the data and creates Figure X.

Note that as the original code used for the paper was written in Python 2.7 using Brian1, the figures in the jupyter notebooks look slightly different to the figures in the article.

## Prerequisites

The code was written in Python 3.8 using [Brian2](https://briansimulator.org/) 2.4.2, [NumPy](https://numpy.org/) 1.19, [SciPy](https://www.scipy.org/scipylib/index.html) 1.6 and [Matplotlib](https://matplotlib.org/) 3.3.
