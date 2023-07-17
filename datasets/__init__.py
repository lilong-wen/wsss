from .train_data import get_train_data


def build_dataset(image_set, args, preprocess_fn):

    if image_set == 'train':
        dataset = get_train_data(args, preprocess_fn)
    else:
        dataset = get_train_data(args, preprocess_fn)

    return dataset