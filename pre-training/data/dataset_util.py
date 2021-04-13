import os


def find_class(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_index = {classes[i]:i for i in range(len(classes))}
    return classes, class_to_index


def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        fnames = os.listdir(os.path.join(root, cname))
        for fname in fnames:
            path = os.path.join(root, cname, fname)
            images.append(path)

    return images