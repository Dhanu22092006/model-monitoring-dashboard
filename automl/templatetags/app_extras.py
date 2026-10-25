from django import template

register = template.Library()


@register.filter
def get_item(d, key):
    try:
        return d.get(key)
    except AttributeError:
        return None


@register.filter
def num(value, digits=4):
    """Format a number, or an en dash when the value is not available."""
    if value is None or value == "":
        return "–"
    try:
        return f"{float(value):.{int(digits)}f}"
    except (TypeError, ValueError):
        return value


@register.filter
def pct(value):
    if value is None:
        return "–"
    return f"{float(value) * 100:.1f}%"


@register.filter
def kb(value):
    try:
        return f"{float(value) / 1024:.1f} KB"
    except (TypeError, ValueError):
        return "–"
