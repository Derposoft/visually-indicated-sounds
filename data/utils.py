import pickle as pkl


def load_annotations_and_classmap():
    annotation_file = "vig_annotation.pkl"
    class_map_file = "vig_class_map.pkl"

    with open(annotation_file, "rb") as f:
        annotations = pkl.load(f, encoding="latin-1")

    with open(class_map_file, "rb") as f:
        class_map = pkl.load(f)
    return annotations, class_map


if __name__ == "__main__":
    load_annotations_and_classmap()
