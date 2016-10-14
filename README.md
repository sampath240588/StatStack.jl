# Mixed Model Scoring using Julia GLMM - [Julia](http://julialang.org)

#[![Build Status](https://travis-ci.org/dmbates/MixedModels.jl.svg?branch=master)](https://travis-ci.org/dmbates/MixedModels.jl)
[![Coverage Status](https://img.shields.io/coveralls/dmbates/MixedModels.jl.svg)](https://coveralls.io/r/dmbates/MixedModels.jl?branch=master)
[![MixedModels](http://pkg.julialang.org/badges/MixedModels_0.3.svg)](http://pkg.julialang.org/?pkg=MixedModels&ver=0.3)
[![MixedModels](http://pkg.julialang.org/badges/MixedModels_0.4.svg)](http://pkg.julialang.org/?pkg=MixedModels&ver=0.4)

#### Julia Scoring
#1. Features
- Consumes Fixed and Random effect output from Julia and R glmm models
- generates Dataframe decomposition of predicted values for each observation
- uses coeffiecients for both fixed and random effects for genrating dependent values
- agregates raw and adjusted/modeld values - based on the assumption that the mean of the raw values equates to the mean of the dependent values
- uses NLOpt optimization to calculate confidence interval aproximation.
- uses NLSolve to calculate optimized pvalues for derived weighted breaks
- Uses weighting factor based on Household buyer counts with weight the publisher breaks back to the total campaign
