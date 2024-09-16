BART
================
Rafael Izbicki

This notebook is part of the book “Machine Learning Beyond Point
Predictions: Uncertainty Quantification”, by Rafael Izbicki.

# Introduction

This notebook demonstrates the use of Bayesian Additive Regression Trees
(BART) to model uncertainty. It simulates non-linear data and fits a
BART model, showing how to quantify both total and epistemic uncertainty
through prediction intervals.

``` r
# Load necessary packages
library(ggplot2)
library(MASS)
library(BART)
```

## Simulation of Data

``` r
# Set parameters for the simulation
n <- 50  # Number of data points

# Simulate data
set.seed(2) # For reproducibility
which <- (runif(n) > 0.5)
X <- which * rbeta(n, 5, 1) + (1 - which) * rbeta(n, 1, 10)
Y <- sapply(2 * X^2, function(x) rnorm(1, mean = x, sd = 0.2))  # Y ~ N(x, 0.1)
```

## Prepare Data for the Model

``` r
# Prepare data 
X_mat <- matrix(X, ncol = 1)
X_new <- seq(0, 1, length.out = 100)
X_new_mat <- matrix(X_new, ncol = 1)
```

## Bayesian Additive Regression Trees (BART)

``` r
# Set seed for reproducibility
set.seed(99)

# Fit BART model
post <- wbart(X_mat, Y, X_new_mat, ndpost = 1000)
```

    ## *****Into main of wbart
    ## *****Data:
    ## data:n,p,np: 50, 1, 100
    ## y1,yn: -1.061982, 1.121848
    ## x1,x[n*p]: 0.141994, 0.883559
    ## xp1,xp[np*p]: 0.000000, 1.000000
    ## *****Number of Trees: 200
    ## *****Number of Cut Points: 49 ... 49
    ## *****burn and ndpost: 100, 1000
    ## *****Prior:beta,alpha,tau,nu,lambda: 2.000000,0.950000,0.050756,3.000000,0.014219
    ## *****sigma: 0.270174
    ## *****w (weights): 1.000000 ... 1.000000
    ## *****Dirichlet:sparse,theta,omega,a,b,rho,augment: 0,0,1,0.5,1,1,0
    ## *****nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws: 1000,1000,1000,1000
    ## *****printevery: 100
    ## *****skiptr,skipte,skipteme,skiptreedraws: 1,1,1,1
    ## 
    ## MCMC
    ## done 0 (out of 1100)
    ## done 100 (out of 1100)
    ## done 200 (out of 1100)
    ## done 300 (out of 1100)
    ## done 400 (out of 1100)
    ## done 500 (out of 1100)
    ## done 600 (out of 1100)
    ## done 700 (out of 1100)
    ## done 800 (out of 1100)
    ## done 900 (out of 1100)
    ## done 1000 (out of 1100)
    ## time: 0s
    ## check counts
    ## trcnt,tecnt,temecnt,treedrawscnt: 1000,1000,1000,1000

``` r
# Calculate means and prediction intervals
means <- post$yhat.test.mean
std_dev <- sqrt(mean(post$sigma)^2 + apply(post$yhat.test, 2, sd)^2)
lower_bound <- means - 1.96 * std_dev
upper_bound <- means + 1.96 * std_dev
```

## Total Uncertainty

``` r
# Plot the original data, fitted regression line, and prediction bands
ggplot() +
  geom_point(aes(x = X, y = Y), colour = "#1E88E5", size = 3) + # Original data
  geom_line(aes(x = X_new, y = means), colour = "#D81B60", linewidth = 2) + # Fitted regression
  geom_ribbon(aes(x = X_new, ymin = lower_bound, ymax = upper_bound), 
              fill = "grey20", alpha = 0.2) + # Prediction bands
  labs(x = "x", y = "y") +
  theme_bw() + 
  theme(text = element_text(size = 14), 
        legend.title = element_blank(), 
        legend.position = "top") + 
  coord_cartesian(ylim = c(-1, 3))
```

![](BART_Notebook_files/figure-gfm/plot-1-1.png)<!-- -->

## Epistemic Uncertainty

``` r
std_dev <- apply(post$yhat.test, 2, sd)
lower_bound <- means - 1.96 * std_dev
upper_bound <- means + 1.96 * std_dev
```

``` r
# Plot the original data, fitted regression line, and new prediction bands
ggplot() +
  geom_point(aes(x = X, y = Y), colour = "#1E88E5", size = 3) + # Original data
  geom_line(aes(x = X_new, y = means), colour = "#D81B60", linewidth = 2) + # Fitted regression
  geom_ribbon(aes(x = X_new, ymin = lower_bound, ymax = upper_bound), 
              fill = "grey20", alpha = 0.2) + # Prediction bands
  labs(x = "x", y = "y") +
  theme_bw() + 
  theme(text = element_text(size = 14), 
        legend.title = element_blank(), 
        legend.position = "top") + 
  coord_cartesian(ylim = c(-1, 3))
```

![](BART_Notebook_files/figure-gfm/plot-2-1.png)<!-- -->
