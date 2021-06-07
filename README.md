# 3d-reactions
Repo for my master's thesis.

include: python version, conda version, pytorch + PTG versions, windows versions

1. Create conda env `conda create -c rdkit -n rdkit-env rdkit` then activate `conda activate rdkit-env`
2. Install pytorch `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
3. `conda install -c conda-forge scikit-learn`
4. Install pymol `conda install -c schrodinger pymol`


x. commands to download pytorch geometric (+ scatter, etc.) with corresponding cuda versions

x+1. Install tqdm `pip install tqdm`

Maybe? `conda install -c rdkit rdkit=2018.09.1`

Should replace this later with:
- Install .yml file
- Stuff to compare to existing models
- Notebooks: mention creating ipython kernels corresponding to .yml