# **Please fill in [this form](https://docs.google.com/document/d/1wGvThrTNFXvXWczOvaEOyBbBf9Sdgj-lV_LYVM7T9po/edit?usp=sharing) to register by March 23rd, 2023.** 


# Possible projects

## 1. Hydrogen in a box 

[Here](https://code.itp.ac.cn/codes/hydrogen-data) you can find some pre-computed data for 64 hydrogen atoms in a box. There are six sets of data at different temperatures (T) and densities (rs). 

Try to set up a machine-learning model to learn the energy and pressure of the system. It makes sense you train on 1200K and 6000K data, and test on 3000K data. 



## 2. A generative model for alanine dipeptide

[Here](https://markovmodel.github.io/mdshare/ALA2/#alanine-dipeptide) you can find the trajectory file of the alanine dipeptide sample from molecular dynamics simulation. 

Please write a generative model to generate more configurations of the same molecule. Think about it, how can this be useful ? 



## 3. Differentiable quantum control 

[Here](https://colab.research.google.com/drive/1T0_sJMwmk7rbpxHMcBZwdD9pnYZx93oh?usp=sharing) is an example code to optimize the pulse to realize a particular quantum gate. 

You can either 

- work on the questions at the end of the notebook 
- or, explore your own idea building on the notebook. For example, adapt it to a more realistic Hamiltonian (Ryderberg atom, transmon, etc). 



## 4. Electrons in a quantum dot

[Here](quantum_dot.ipynb) is an example code that carries variational free energy calculation for a few electrons in a quantum dot. 

You can either 

- work on the questions at the end of the notebook 
- or, explore your own idea building on the notebook. 



## 5. Paper QA

[Here](https://github.com/whitead/paper-qa) is a tool allows you to ask questions about uploaded texts (PDF documents, code repos). 

Play around with it. Explain how it works. Do you find it useful ? 



## 6. Molecular Property Prediction

[Here](https://www.kaggle.com/competitions/champs-scalar-coupling/overview) is a competition on Kaggle. It's scientific ML with data: Use the data to develop an algorithm to predict the interaction between two atoms in a molecule. Background in DFT, MD would be helpful. Please work on the competition and submit your results to the leaderboard.



## 7. Three-body problem

[Here](https://github.com/zaman13/Three-Body-Problem-Gravitational-System/blob/master/Python%20notebook%20files/Earth_Jupiter_Sun_system.ipynb) is a solver of the gravitational three-body system. Due to the system’s chaotic nature, this kind of iterative calculation always has unpredictable and potentially infinite computational costs.

Try to design a neural network trained by the data generated from this traditional solver, and see if you can accelerate the calculation.



## 8. Superconductor discovery

[This paper](https://arxiv.org/abs/2104.11150) contains some ideas of discoverying new superconducting coumpounds with higher Tc.  Try to setup such a system and make your own prediction! 



## 9. Symbolic regression 

[Symbolic Regression](https://github.com/MilesCranmer/PySR) is a method to let you fit data with mathmatical expressions. Play with this package and use it for a cool application. 

## 10. Rubik’s cube

[Reinforcement Learning to solve Rubik’s cube (and other complex problems!)](https://medium.datadriveninvestor.com/reinforcement-learning-to-solve-rubiks-cube-and-other-complex-problems-106424cf26ff). Our class does not cover Reinforcement Learning. However, if you are interested in Rubik's cube and want to explore how Machine Learning can be applied to it without relying on complex group theory, you can see that article and [github code](https://github.com/Shmuma/rl/tree/master/articles/01_rubic).


## 11. Otherwise, welcome to bring your own project! 





