import numpy as np


def apply_transform(drugs, featurizer, idx):
    if isinstance(drugs[0], np.ndarray):
        drugs_as_tuples = [tuple(drug) for drug in drugs]
        unique_tuples = list(set(drugs_as_tuples))
        unique_transformed = [featurizer(drug_tuple, idx) for drug_tuple in unique_tuples]
        mapping = dict(zip(unique_tuples, unique_transformed))
        return [mapping[tuple(drug)] for drug in drugs]
    else:
        unique_values = np.unique(drugs)
        unique_transformed = [featurizer(drug, idx) for drug in unique_values]
        mapping = dict(zip(unique_values, unique_transformed))
        return [mapping[drug] for drug in drugs]


def split(df, split_frac, shuffle=True):
    data_splits = []
    remaining_data = df

    for ratio in split_frac[:-1]:  # we leave out the last split ratio
        subset_size = int(len(remaining_data) * ratio / sum(split_frac))
        if shuffle:
            subset = remaining_data.sample(n=subset_size)
        else:
            subset = remaining_data.iloc[:subset_size]

        data_splits.append(subset.reset_index(drop=True))

        remaining_data = remaining_data.drop(subset.index)

        # Adjusting the remaining split fractions
        split_frac = split_frac[1:]

    # Append the remaining data (resetting index) as the last split
    data_splits.append(remaining_data.reset_index(drop=True))

    return data_splits