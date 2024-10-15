# TODO this really should go in permacache


import pickle
from typing import Dict, Tuple, Union


def renamed_symbol_unpickler(
    symbol_rename_map: Dict[Tuple[str, str], Union[Tuple[str, str], type]]
) -> type:
    """
    Returns an unpickler class that renames symbols as specified in
    the symbol_rename_map dictionaries.

    :param symbol_rename_map: A dictionary mapping (module, name) pairs to
        (new_module, new_name) pairs. Can also map to a type, in which case
        we convert the type to a (module, name) pair.
    """

    symbol_rename_map_string = {}
    for (module, name), new_symbol in symbol_rename_map.items():
        if isinstance(new_symbol, type):
            new_symbol = (new_symbol.__module__, new_symbol.__name__)
        assert (
            isinstance(new_symbol, tuple)
            and len(new_symbol) == 2
            and all(isinstance(x, str) for x in new_symbol)
        ), f"Invalid new symbol: {new_symbol}"
        symbol_rename_map_string[(module, name)] = new_symbol

    class RenamedSymbolUnpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if (module, name) in symbol_rename_map_string:
                module, name = symbol_rename_map_string[(module, name)]
            try:
                return super().find_class(module, name)
            except:
                print("Could not find", (module, name))
                raise

    return RenamedSymbolUnpickler
