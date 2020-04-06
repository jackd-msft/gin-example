# either of the following work
import core_gin

import gin


def parse_args():
    import sys
    args = sys.argv[1:]
    files = []
    bindings = []
    for arg in args:
        if '=' in arg:
            bindings.append(arg)
        else:
            files.append(arg if arg.endswith('.gin') else (arg + '.gin'))
    gin.parse_config_files_and_bindings(files, bindings)


def main():
    parse_args()
    train_ds, test_ds, num_classes = core_gin.get_datasets()
    inputs_spec = train_ds.element_spec[0]
    model = core_gin.get_classification_model(inputs_spec, num_classes)
    core_gin.fit(model, train_ds, test_ds)
    print(gin.operative_config_str())


if __name__ == "__main__":
    main()
