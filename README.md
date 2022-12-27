# Projecting non-parametric NFL player performances from ordinal data

## Abstract

Statisticians revolutionized 
professional baseball and basketball at all levels of team operations. 
However, effectively modeling players’ performances in the National Football 
League remains elusive due to three common pitfalls:
(1) football games contain many variables, (2) datasets are small, and (3) 
patterns are non-parametric. I developed a supervised learning approach, 
which seeks to overcome these challenges. My method translates publicly 
available expert rankings into probability distributions, predicting each 
player’s likely performance in the upcoming week. The distributions are fit 
using kernel density estimation, followed by a series of processing steps to 
reduce overfitting and noise. The results below focus on "fantasy points" as the
metric of player performance, although the method can be adjusted to predict
other dependent variables.

The present repo contains the code used for this Python project. Full details on
the project are provided in this 
[white paper](https://drive.google.com/file/d/1ml9dRE8Aqx3htos0nMnKif_fHV3WQo2P/view?usp=share_link). 
This readme illustrates the primary results (the distributions) and later gives a summary of the repo's code.

## Results

The figure just below provides the final distributions for each position.

Notice that the distributions are oftentimes not smooth, with slight humps and 
plateaus. These humps and plateaus are not noise, as smoothing them out lowers 
model fit on testing data. These patterns arise due to discrete aspects of 
football performance. For example, each touchdown is a discrete outcome worth 
six fantasy points. See the wide receiver (WR) plot in the top right. 
There is first a plateau from 8-13 points and then a second plateau from 17-21 points. 
Wide receiver performances at the second plateau were likely associated with 
one more touchdown that performances at the first plateau. A similar trend is found
for tight ends (TE).

<img src='distribution_images/final_density_distribution.png' width=800>

This second figure provides the cumulative distributions for each position.

<img src='distribution_images/final_cumulative_density_distribution.png' width=800>

## Code organization
### Scraping & Organizing data

The code in directory `scrape_prepare_input` retrieves historic ranking and performance data.
Most notably, it retrieves every FantasyPros expert's ranking of every player 
across every week of every NFL season from 2013-2021. 
Roughly, this corresponds to 60,000 player-weeks and around ten million rankings.
All data scraped are public.
These files also organize the data into a Pandas Dataframe.
I prepared documentation for the most pivotal pieces of code, including: 
* `organize.organize_input.py`
* `scrape.scrape_fantasypros.py`
* `scrape.scrape_scores.py`

### Creating the distributions
Based on the scraped data, the code in directory `make_distributions` creates probability distributions, 
predicting player performance based on their expert ranking. These are described in detail in the white paper. 
I prepared documentation for the most pivotal pieces of code, including: 
* `setup_expert_distributions.py`
* `density.py`
* `rank_concat.py`
* `smooth.py`
* `test_accuracy.py`
* `plot.py`

## General pipeline

`setup_expert_distributions.py` runs the pipeline, it calls all the functions below.

Creating a distribution involves first scraping historic expert ranking data 
using `scrape.scrape_fantasypros.py` and scraping historic fantasy points 
performance data using `scrape.scrape_scores.py`. The scraped data are then 
organized into a single Pandas dataframe using `scrape.organize_input.py`. Each
row of the dataframe represents one player's performance in one week, and their
expert projections for that week.

After the data are loaded, the dataframes data are further organized into numpy arrays. 
Then, functions from `density.py` are called to create the density distributions.
In creating the distributions, `density.py` calls functions from `rank_concat.py` 
to carry out the concatenation procedures described in the white paper. 
The distributions are then smoothed using functions from `smooth.py`. 
Next, the accuracy of the distributions is tested using cross-validation and 
functions from `test_accuracy.py`. Testing accuracy is used to tune various
hyperparameters described in the white paper. Finally, the distributions are
plotted using functions from `plot.py`.

Note that I haven't uploaded my scraped data to the repo, nor the directory I 
use as a cache.

## Monte Carlo Simulation
Sampling from these distributions allows modeling the performance of player groups.
The white paper goes into brief detail about this and its applicability to Daily Fantasy Sports.

However, the present repository only contains code for generating the distributions 
and does not contain any code for Monte Carlo simulations. I have that code in a private repository. 
If you are interested in discussing this, please email me (paulcbogdan@gmail.com). 
Building an efficient large simulation is difficult, so I encourage you to email me if that is your goal.

I am also open to other emails if you are interested in sports analytic consulting.