from multipledispatch import Dispatcher


def _create_register_func(dispatcher_funcs, types, **kwargs):
    def _(func):
        for dispatcher_func in dispatcher_funcs:
            dispatcher_func.add(types, func, **kwargs)

        return func

    return _


class TwoLevelDispatcher(Dispatcher):
    """A faster implementation of Dispatch."""

    def __init__(self, name, doc=None):
        super().__init__(name, doc)
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
                # dispatcher_func = _create_dispatcher_func(
                # )
                dispatcher_func = Dispatcher(f"{self.name}_{t.__name__}")

                self._meta_dispatcher.register(t)(dispatcher_func)

            dispatcher_funcs.append(dispatcher_func)

        return _create_register_func(dispatcher_funcs, types, **kwargs)

    def dispatch_iter(self, *types):
        for dispatcher_func in self._meta_dispatcher.dispatch_iter(types[0]):
            func = dispatcher_func.dispatch(*types)
            if func is not None:
                yield func

    def dispatch(self, *types):
        try:
            func = next(self.dispatch_iter(*types))
            return func
        except StopIteration:
            return None

    # def __call__(self, *args, **kwargs):
    #     types = tuple(type(arg) for arg in args)

    #     try:
    #         func = self._cache[types]
    #     except KeyError:
    #         func = self.dispatch(*types)
    #         if not func:
    #             raise NotImplementedError(
    #                 'Could not find signature for %s: <%s>' %
    #                 (self.name, str_signature(types)))
    #         self._cache[types] = func

    #     return func(*args, **kwargs)
