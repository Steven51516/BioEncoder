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
# train, val, test = trainer.prepare_datasets(inputs = [drugs, proteins, pdb_ids], y=labels, split_frac = [0.7, 0.1, 0.2],
#                                             input_types=[BE.DRUG, BE.PROT, BE.PROT_PDB])
trainer.train(train, val)

