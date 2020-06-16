from multipledispatch import Dispatcher


class TwoLevelDispatcher(Dispatcher):
    """An implementation of multipledispatch.Dispatcher that utilizes two
    levels of dispatching.

    The major change is that this class no longer trigger reorder in
    dispatch_iter. Because the majority of the slowness is happening
    in reorder, this implementation makes dispatch_iter faster.
    Instead, this implementation will trigger reorder in the meta dispatcher
    and second level dispatcher. Because the number of registered signatures
    for each dispatcher is much smaller in this implementation (In pandas
    backend, the number of signatures in one level implementation is
    O(1000), and the max number of signatures for the meta dispatcher and
    second level dispatcher is O(100)), the overall dispatch_iter is faster.

    This implementation consist of three Dispatcher instance:

    (1) This dispatcher, or the instance of this class itself. This class
    inherits Dispatcher to avoid duplicating __call__, cache, ambiguities
    detection, as well as properties like ordering and funcs.

    (2) First level dispatcher, aka, meta dispatcher. This is the dispatcher
    is used to dispatch to the second level dispatcher using the type of the
    first arg.

    (3) Second level dispatcher. This is the actual dispatcher used for linear
    searching of matched function given type of args.

    Implementation notes:

    (1) register:
    This method will now (a) create the second level dispatcher
    if missing and register it with the meta dispatcher. (b) return a function
    decorator that will register with all the second level dispatcher. Note
    that multiple second level dispatcher could be registered with because this
    is supported:

        @foo.register((C1, C2), ...)

    The decorator will also register with this dispatcher so that func and
    ordering works properly.

    (2) dispatcher_iter
    Instead of searching through self.ordering, this method now searches
    through:
    (a) dispatch_iter of the meta dispatcher (to find matching second level
    dispatcher).
    (b) for each second level dispatcher, searches through its dispatch_iter.
    Because dispatch_iter of meta dispatcher and second level dispatcher
    searches through registered functions in proper order (from subclasses to
    base classes).

    (3) ambiguity detection, ordering, and funcs
    Because this dispatcher has the same func and ordering property as
    multipledispatch.Dispatcher. We can completely reuse the ambiguity
    detection logic of Dispatcher. Note:
    (a) we never actually linear search through ordering of this dispatcher
    for dispatching. It's only used for ambiguity detection.
    (b) deleting an entry from func of this dispatcher (i.e. del
    dispatcher.func[A, B]) does not unregister it. Entries from the second
    level dispatcher also needs to be deleted. This is OK because it is not
    public API.

    Difference in behavior:
    (1) ambiguity detection
    Because this implementation doesn't not trigger total reorder of signatures
    in dispatch_iter, ambiguity warning will trigger when user calls
    "ordering", instead of "dispatch".

    """

    def __init__(self, name, doc=None):
        super().__init__(name, doc)
        self._meta_dispatcher = Dispatcher(f'{name}_meta')

    def register(self, *types, **kwargs):
        type0 = types[0]

        if isinstance(type0, type):
            type0 = [type0]

        dispatchers = []

        for t in type0:
            if (t,) in self._meta_dispatcher.funcs:
                dispatcher = self._meta_dispatcher.funcs[(t,)]
            else:
                dispatcher = Dispatcher(f"{self.name}_{t.__name__}")
                self._meta_dispatcher.register(t)(dispatcher)

            dispatchers.append((t, dispatcher))

        def _(func):
            self.add(types, func, **kwargs)
            for t, dispatcher in dispatchers:
                dispatcher.add(tuple([t, *types[1:]]), func, **kwargs)
            return func

        return _

    def __delitem__(self, types):
        del self.funcs[types]
        del self._meta_dispatcher.funcs[types[:1]].funcs[types]
        if not self._meta_dispatcher.funcs[types[:1]].funcs:
            del self._meta_dispatcher.funcs[types[1:]]

    def dispatch_iter(self, *types):
        for dispatcher in self._meta_dispatcher.dispatch_iter(types[0]):
            func = dispatcher.dispatch(*types)
            if func is not None:
                yield func
