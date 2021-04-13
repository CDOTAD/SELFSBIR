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
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)
        else:
            images.append(c_path)

    return images