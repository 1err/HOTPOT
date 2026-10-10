"""The shared ingredient vocabulary. IDs are stable across UI and checkpoints."""

INGREDIENTS = [
    {"id": "beef_rolls", "name": "肥牛卷", "en": "Beef rolls"},
    {"id": "crab_sticks", "name": "蟹棒", "en": "Crab sticks"},
    {"id": "napa_cabbage", "name": "大白菜", "en": "Napa cabbage"},
    {"id": "lotus_root", "name": "藕片", "en": "Lotus root slices"},
    {"id": "soft_tofu", "name": "嫩豆腐", "en": "Soft tofu"},
    {"id": "wide_noodles", "name": "宽粉", "en": "Wide starch noodles"},
    {"id": "udon_noodles", "name": "乌冬面", "en": "Udon noodles"},
    {"id": "shiitake_mushrooms", "name": "香菇", "en": "Shiitake mushrooms"},
    {"id": "beef_meatballs", "name": "牛肉丸", "en": "Beef meatballs"},
]
INGREDIENT_IDS = tuple(item["id"] for item in INGREDIENTS)
SAMPLE_LABELS = (*INGREDIENT_IDS, "background")

# Initial durations supplied by the user; editable independently of recognition.
DEFAULT_DURATIONS = {
    "beef_rolls": 15,
    "crab_sticks": 90,
    "napa_cabbage": 150,
    "lotus_root": 300,
    "soft_tofu": 300,
    "wide_noodles": 480,
    "udon_noodles": 180,
    "shiitake_mushrooms": 300,
    "beef_meatballs": 480,
}
