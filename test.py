from BioEncoder.dataset.loader import load_DTI
from BioEncoder.encoder import DrugEncoder
from BioEncoder.encoder import ProteinEncoder
from BioEncoder.task.DTI import DTITrainer


drugs, proteins, labels = load_DTI("data/DAVIS/data.txt")
drug_encoder = DrugEncoder("CNN")
target_encoder = ProteinEncoder("CNN")
trainer = DTITrainer(drug_encoder, target_encoder)
train, val, test = trainer.prepare_datasets(drugs, proteins, labels, [0.7, 0.2, 0.1])
trainer.train(train, val)
