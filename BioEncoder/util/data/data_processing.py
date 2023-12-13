

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