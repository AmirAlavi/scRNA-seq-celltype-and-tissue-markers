
## Requirements
This analysis requires
- a Linux terminal (for example, the terminal in Ubuntu, or in MacOS, or the Windows Subsystem for Linux (WSL) in Windows 10)
- Anaconda python installed for whichever of the above operating systems you have above, for example for Ubuntu, or WSL in windows, you can follow [these instructions](https://docs.anaconda.com/anaconda/install/linux/)

**Note:** This analysis pipeline was successfully run on  Ubuntu 16.04.7 LTS via WSL for Windows 10, and has not been tested on the other platforms

## Usage
We are using the [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow management system to make our analysis reproducible. Snakemake is like a super-powered GNU make, if you are familiar with that. Analysis are written as a series of rules that define who to make output files from a series of input files.

### Prepare snakemake environment

1. Once you have Anaconda installed, clone this repository
2. In you terminal, enter the repository, and enter this subdirectory ("labeling_pipeline")
3. Create the snakemake conda environment: `$ conda env create -f environment.yml`
4. Activate the environment: `$ conda activate snakemake`

### Generate labeled human data

Once you've prepared and activated the snakemake environment, you're now all set to run all analysis. To produce all the labeled data, from the root directory of this repository simply run:

```
$ snakemake --cores 1 --use-conda
```

Snakemake will read the rules in `Snakefile` to automatically download all necessary data files, install all software and packages (including the right version of R) into an isolated environment, and run all the scripts in the right sequence to generate your labeled data.

This will take a very long time, and if for some reason execution fails along the way, snakemake will pick up where it left off. Once intermediate steps have finished and created their respective outputs, they don't need to be run again unless you delete them and want to start over.
