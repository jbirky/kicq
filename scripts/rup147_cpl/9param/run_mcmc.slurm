#!/bin/bash

## Job Name

#SBATCH --job-name=jbirky_mcmc

## Allocation Definition

## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).

#SBATCH --account=astro
#SBATCH --partition=astro

## Resources

## Total number of Nodes

#SBATCH --nodes=1

## Number of cores per node

#SBATCH --ntasks-per-node=28

## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.

#SBATCH --time=14-00:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.
## For mox, --mem may be more than 100G depending on the memory of your nodes.
## For ikt, --mem may be 58G or more depending on the memory of your nodes.
## See above section on "Specifying memory" for choices for --mem.

#SBATCH --mem=64G

## Specify the working directory for this job

#SBATCH --chdir='/gscratch/stf/jbirky/projects/kicq/scripts/rup147_cpl/9param/'

##turn on e-mail notification

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbirky@uw.edu

## export all your environment variables to the batch job session

#SBATCH --export=all

python rup147_cpl_9par_emcee.py
