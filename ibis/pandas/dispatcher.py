from multipledispatch import Dispatcher


def _create_dispatcher_func(dispatcher):
    return lambda _: dispatcher


def _create_register_func(dispatcher_funcs, types, **kwargs):
    def _(func):

        # if len(dispatcher_funcs) == 3:
        # import pdb; pdb.set_trace()

        for dispatcher_func in dispatcher_funcs:
            dispatcher_func(None).add(types, func, **kwargs)

        return func

    return _


class TwoLevelDispatcher(object):
    """A faster implementation of Dispatch."""

    def __init__(self, name, doc=None):
        self.name = self.__name__ = name
        self.doc = doc
        # The return value of _first_level_dispatcher(arg0) is a single arg
        # function that returns the dispatcher for type(arg0).
        self._meta_dispatcher = Dispatcher(f'{name}_meta')

    def register(self, *types, **kwargs):
        type0 = types[0]

        if isinstance(type0, type):
            type0 = [type0]

        dispatcher_funcs = []

        for t in type0:
            if (t,) in self._meta_dispatcher.funcs:
                dispatcher_func = self._meta_dispatcher.funcs[(t,)]
            else:
                # if t.__name__ == 'Add':
                #    import pdb; pdb.set_trace()

                dispatcher_func = _create_dispatcher_func(
                    Dispatcher(f"{self.name}_{t.__name__}")
                )
                self._meta_dispatcher.register(t)(dispatcher_func)

            dispatcher_funcs.append(dispatcher_func)

        return _create_register_func(dispatcher_funcs, types, **kwargs)

    def __call__(self, *args, **kwargs):
        dispatcher = self._meta_dispatcher(args[0])
        return dispatcher(*args, **kwargs)
