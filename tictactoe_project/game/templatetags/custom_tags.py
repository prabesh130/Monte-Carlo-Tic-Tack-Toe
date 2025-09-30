from django import template

register = template.Library()

@register.filter
def to(value, arg):
    """Creates a range from value to arg"""
    return range(int(value), int(arg))

@register.filter
def get_item(board, index):
    """Get item from list using index"""
    return board[index]