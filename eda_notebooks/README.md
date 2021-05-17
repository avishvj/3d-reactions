### Reaction Dynamics EDA (Training Set)
How does the reactant change into the product? Identify the reaction centre and magnitude of change: bonds break/form; atoms moving; bond and dihedral angles changing. 
- Bond dynamics: which bonds break/form? How many across training data?
- Atom dynamics: which atoms move and by how much? 
- Angle dynamics: how do bond/dihedral angles change? 
- TODO: get bond lengths and look at how they change. https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html

### MIT Model EDA (Evaluating Model)
- Comparing ground truth TS with initial model guess (D_init), final model guess, and average TS based on distance.
    - Could also do this based on different distance functions. So instead of D_init, use the topological matrix or 3D_rbf matrix. This means my model has to produce one of these types of matrices though.
- Fig 2: Lucky compares distribution of atom distances in the reaction core across initial model, final model, ground truth, and linear approximation.
    - Have ground truth and their D_init. Need D_final. Have their best model but need a way to run ts_gen with that best checkpoint.
- Fig 3: you want the learned weights between atoms to decrease as the topological distance increases. They plot this for test set. Doesn't make sense to do on 3D distance matrix since could be far but bonded.
- Fig 4: compare cosine distance ("loss") of test set from training set using learned GNN embeddings... why? 
    - They train model and save GNN embeddings. Then run test set through model and...?
    - How is the learned GNN embedding related to D_init and W?
    - Still not sure how they compare training set to test set? What are they comparing. The embeddings or ...? Or are they running the train set and test set through the embeddings created by the train model?

- Do the same experiments but on different train:test splits to see if similar trends are observed.
