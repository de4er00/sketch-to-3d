"""Wearing order implied by garment names."""
import re

# a higher rank is worn over a lower one
LAYER_RANK = [
    (r'shirt|blouse|tee|top|tunic|sweater|jumper|turtleneck', 1),
    (r'dress|pinafore|overall|apron|bib|vest|waistcoat|corset|gilet', 2),
    (r'jacket|coat|hoodie|cardigan|parka|blazer|cape|cloak|collar|capelet|shawl|scarf|stole|poncho', 3),
    (r'trouser|jeans|pants|shorts|skirt|leggings|tights|stocking', 1),
    (r'sock|legwarmer|warmer|cuff|gaiter', 2),
    (r'boot|shoe|sneaker|sole|sandal', 3),
]


LEG_GARMENT = r'trouser|jeans|pants|shorts|leggings|tights'


def is_leg_garment(cname):
    return re.search(LEG_GARMENT, str(cname).lower()) is not None


def layer_rank(cname):
    """Wearing order the name implies, None when it says nothing."""
    n = str(cname).lower()
    for pat, rank in LAYER_RANK:
        if re.search(pat, n):
            return rank
    return None
