
### Reaction Dynamics (distances.ipynb, rxn.ipynb)
How does the reactant change into the product? Identify the reaction centre and magnitude of change: bonds break/form; atoms moving; bond and dihedral angles changing. 
- MIT Fig 2: compare distribution of atom distances in reaction core across initial model, final model, ground truth, and linear approximation.
- What I've done: looked at how much reaction centre atoms have changed positions in test AND training data (distances.ipynb). 
    - Bond dynamics: which bonds break/form? How many across training data? Do this through adjacency matrix changes.
    - Atom dynamics: which atoms move and by how much? Do this through 3D distance matrix changes for reaction centre.
**TODO**:
- Alternative distance functions: 3D rbf matrix worth looking at? Looking at topological distances in rxn.ipynb but not sure if useful since basically just about reaction core.
- Angle dynamics: how do bond/dihedral angles change? 
- How interatomic distances/reaction centre bond lengths change. https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html
- How to extract their D_final?

### Learned Model Weights (weights.ipynb) and Embeddings (TODO)
- MIT Fig 3: you want the learned weights between atoms to decrease as the topological distance increases. They plot this for test set. Doesn't make sense to do on 3D distance matrix since could be far but bonded.
- MIT Fig 4: compare cosine distance ("loss") of test set from training set using learned GNN embeddings... why? 
    - They train model and save GNN embeddings. Then run test set through model and...?
    - How is the learned GNN embedding related to D_init and W?
    - Still not sure how they compare training set to test set? What are they comparing. The embeddings or ...? Or are they running the train set and test set through the embeddings created by the train model?

Potential **TODO**:
- Do the same experiments but on different train:test splits to see if similar trends are observed.
