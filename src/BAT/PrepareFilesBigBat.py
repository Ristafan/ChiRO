def slideWindow(a, size, step):
    corr_size = list(a.shape)
    corr_size[0] = math.ceil(corr_size[0] / step) * step
    c = torch.zeros(corr_size)
    c[:len(a)] = a
    return c.unfold(dimension=0, size=size, step=step)

def prepareSet(group, labels, patch_len, patch_skip, max_seqs=None, min_seqs=None, single_first=False):
    X = []
    Y = []

    sets = group.keys()
    sets = sorted(sets, key=lambda x: x.count(','), reverse=(not single_first))
    spec_count = torch.zeros(len(labels))
    for species in tqdm(list(sets)):
        if max_seqs is not None:
            signal = group.get(species)[:max_seqs * patch_skip]
        else:
            signal = group.get(species)
        signal = torch.as_tensor(np.asarray(signal))
        label = torch.zeros(len(labels))
        for s in species.split(','):
            if s in labels:
                label[labels[s]] = 1
        if max_seqs is not None and min_seqs is not None:
            ma = torch.all(spec_count[torch.nonzero(label)] < max_seqs) # all labels have less then max
            mi = torch.any(spec_count[torch.nonzero(label)] < min_seqs) # one label has less than min
            if label.sum() > 0 and len(signal) > 0 and (mi or ma):
                patches = slideWindow(signal, patch_len, patch_skip)
                X.extend(patches)
                Y.extend([label] * len(patches))
                spec_count += label * len(patches)
        else:
            patches = slideWindow(signal, patch_len, patch_skip)
            X.extend(patches)
            Y.extend([label] * len(patches))


    X, Y = shuffle(X, Y, random_state=42)
    return torch.stack(X), torch.stack(Y)

def prepare(file, labels, patch_len, patch_skip, max_seqs=None, min_seqs=None, single_first=False):
    prepared_hf = h5py.File(file, 'r')
    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, patch_skip, max_seqs, min_seqs, single_first)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, patch_skip, max_seqs, min_seqs, single_first)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, patch_skip, max_seqs, min_seqs, single_first)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val