# VanillaSABR
 Didactic, from-scratch implementation of a vanilla SABR model from Hagan's 2002 paper "Managing Smile Risk".
 
 On top of the traditional "vanilla" model, which assumes strictly positive rates, this implementation allows negative rates to be input by adopting a "shifted" version of the smile.

The repository contains the following files:
- SABR.py, that contains the documented SABR class;
- DifferentialEvolution.py, a from-scratch Python3 implementation of the differential evolution algorithm used to calibrate the SABR model;
- Implementation.ipynb, Jupyter notebook in which the model is fitted;
- SABR.pdf, a LateX Beamer presentation that explains the model and showcases results.
