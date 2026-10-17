"""Provider-boundary market symbol normalization."""


def canonicalize_class_symbol(symbol: str) -> str:
    """Use the hyphen class separator expected by SEC and Yahoo."""
    return symbol.replace(".", "-")