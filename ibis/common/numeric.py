from __future__ import annotations

from decimal import Context, Decimal, InvalidOperation


def normalize_decimal(value, precision: int | None = None, scale: int | None = None):
    context = Context(prec=38 if precision is None else precision)

    try:
        if isinstance(value, float):
            out = Decimal(str(value))
        else:
            out = Decimal(value)
    except InvalidOperation:
        raise TypeError(f"Unable to construct decimal from {value!r}")

    out = out.normalize(context=context)
    components = out.as_tuple()
    n_digits = len(components.digits)
    exponent = components.exponent

    if precision is not None and precision < n_digits:
        raise TypeError(
            f"Decimal value {value} has too many digits for precision: {precision}"
        )

    if scale is not None:
        if exponent < -scale:
            raise TypeError(
                f"Normalizing {value} with scale {exponent} to scale -{scale} "
                "would loose precision"
            )

        other = Decimal(10) ** -scale
        try:
            out = out.quantize(other, context=context)
        except InvalidOperation:
            raise TypeError(
                f"Unable to normalize {value!r} as decimal with precision {precision} "
                f"and scale {scale}"
            )

    return out
