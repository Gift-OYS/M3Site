reps1 = [
    {
        "model": 0,
        "chain": "",
        "resname": "",
        "style": "cartoon",  # line, stick, sphere, cartoon, surface
        "color": "whiteCarbon",  # blue, red, green, yellow, whiteCarbon
        "residue_range": "",  # 3-15
        "around": 0,  # around range, default 0
        "byres": False,
        "visible": False
    },
]

style_list = ["Cartoon", "Sphere", "Stick", "Line", "Surface"]
color_list = ["White", "Blue", "Red", "Green", "Yellow", "Magenta", "Cyan", "Orange", "Purple", "Gray"]
default_reps = [
    {
        "model": 0,
        "chain": "",
        "resname": "",
        "style": style_list[0][0].lower() + style_list[0][1:],
        "color": color_list[0][0].lower() + color_list[0][1:] + "Carbon",  # whiteCarbon
        "residue_range": "",  # 3-15
        "around": 0,  # around range, default 0
        "byres": False,
        "visible": False
    },
]
model_list = ['M3Site-ESM3-abs', 'M3Site-ESM3-full', 'M3Site-ESM2-abs', 'M3Site-ESM2-full', 'M3Site-ESM1b-abs', 'M3Site-ESM1b-full']
no_cat_dict = {
    'b': 'background',
    '0': 'CRI',
    '1': 'SCI',
    '2': 'PI',
    '3': 'PTCR',
    '4': 'IA',
    '5': 'SSA'
}