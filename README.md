# __Name_of_our_paper__

## BioEncoder-DTI
* Source code for the paper "__Name_of_our_paper__" .
* BioEncoder-DTI is a model for drug-target interaction(DTI) prediction.

![BioEncoderDTI](img/BioEncoderDTI.png)


## Requirements
* You can create a [conda](https://conda.io/) environment with the required dependencies by the following command:
```
  conda env create -n bedti -f environment.yaml
  conda activate bedti
```

## Dataset
All *Drug-Target Binding Benchmark Dataset* used in this paper are publicly available and and can be accessed here: 
| Data  | Function |
|-------|----------|
|[DAVIS](http://staff.cs.utu.fi/~aatapa/data/DrugTarget/)|```load_DAVIS_DTI("data/davis.txt")``` |
|[KIBA](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)|```load_KIBA_DTI("data/KIBA.txt")``` |

## Usage
```
from BioEncoder.dataset.loader import *
from BioEncoder.encoder.BE import *
from BioEncoder.task.BETrainer import *
import gc
gc.collect()

pdb_ids, drugs, proteins, labels = load_DTI("data/DAVIS/davis.txt", id=True)
BE = BioEncoder()
de = BE.create_encoder(BE.DRUG, "CNN")
pe = BE.create_encoder(BE.PROT_PDB, "GCN")
inter = BE.set_interaction(de, pe, "cat", head=1)
model = BE.build_model()
trainer = BETrainer(BE)
#test
train, val, test = trainer.prepare_datasets(inputs = [drugs[:7], proteins[:7], pdb_ids[:7]], y=labels[:7], split_frac = [0.7, 0.1, 0.2],
                                            input_types=[BE.DRUG, BE.PROT, BE.PROT_PDB])
trainer.train(train, val)
```
