# 3d-reactions
Repo for my master's thesis.

Clone the current repo
`git clone https://github.com/avishvj/3d-reactions`
`cd 3d-reactions`

Install and activate the conda environment:
`conda env create -f 3d-rdkit.yml`
`conda activate 3d-rdkit`

To be completed later. Include:
- Versions for: Python, Conda, PyT + PyTG, Windows
- Data included, but mention data processing
- Install .yml file
- Stuff to compare to existing models
- Notebooks: mention creating ipython kernels corresponding to .yml
- Info about work e.g. building on MIT model
- Describe how logger works and how results get saved.
- Submodules: ts_gen. 
    - Reminder: if making submodule changes, check out a branch, make changes, publish change within submodule (ts_gen), then update superproject (3d-reactions) to reference new commit.