# TODO this really should go in permacache


import pickle

SYMBOL_RENAME_MAP = {
    ("__main__", "TransducerExperimentResultPacked"): (
        "evallm.experiments.exhaustive_transducer_experiment",
        "TransducerExperimentResultPacked",
    )
}


class renamed_symbol_unpickler(pickle.Unpickler):
    """
    Unpicler that renames modules and symbols as specified in the
    MODULE_RENAME_MAP and SYMBOL_RENAME_MAP dictionaries.
    """

    def find_class(self, module, name):
        if (module, name) in SYMBOL_RENAME_MAP:
            module, name = SYMBOL_RENAME_MAP[(module, name)]
        try:
            return super(renamed_symbol_unpickler, self).find_class(module, name)
        except:
            print("Could not find", (module, name))
            raise


# class remapping_pickle:
#     """
#     An instance of this class will behave like the pickle module, but
#     will use the renamed_symbol_unpickler class instead of the default
#     Unpickler class.
#     """

#     def __getattribute__(self, name):
#         if name == "Unpickler":
#             return renamed_symbol_unpickler
#         return getattr(pickle, name)

#     def __hasattr__(self, name):
#         return hasattr(pickle, name)