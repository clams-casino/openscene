HABITAT_OGN_LABEL_TO_TEXT_PROMPT = {
    # 21 habitat object goal nav categories along with 'other' category
    # Map each raw label to a "gramatically correct" text prompt to be encoded
    # by the text embedding model
    "chair": ["a chair in a scene"],
    "table": ["a table in a scene"],
    "picture": ["a picture in a scene"],
    "cabinet": ["a cabinet in a scene"],
    "cushion": ["a cushion in a scene"],
    "sofa": ["a sofa in a scene"],
    "bed": ["a bed in a scene"],
    "chest_of_drawers": ["a chest of drawers in a scene"],
    "plant": ["a plant in a scene"],
    "sink": ["a sink in a scene"],
    "toilet": ["a toilet in a scene"],
    "stool": ["a stool in a scene"],
    "towel": ["a towel in a scene"],
    "tv_monitor": ["a television monitor in a scene"],
    "shower": ["a shower in a scene"],
    "bathtub": ["a bathtub in a scene"],
    "counter": ["a counter in a scene"],
    "fireplace": ["a fireplace in a scene"],
    "gym_equipment": ["gym equipment in a scene"],
    "seating": ["seating in a scene"],
    "clothes": ["clothes in a scene"],

    "other": ["other in a scene"],
}

HABITAT_OGN_LABELS = (
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",

    "other",
)





MATTERPORT_LABELS_REGIONS = ('bathroom', 'bedroom', 'closet', 'dining room',
                             'garage', 'hallway', 'library', 'laundry room or mudroom', 'kitchen', 
                             'living room', 'meeting room or conference room', 'office', 
                             'porch or terrace or deck or driveway', 'recreation or game room', 'stairs', 'utility room or tool room', 
                             'cinema or home theater or theater', 'gym', 'balcony', 'bar', 'classroom',
                             'spa or sauna', 
                             
                             # these labels are blacklisted later in the evaluation
                             # but we still want to query with them!
                             # so they're NOT commented out
                             'other room', 'junk', 'no label',
                             'dining booth', 'entryway/foyer/lobby', 'outdoor',

                            # these labels are ambiguous and mapped to a less ambiguous similar label
                            #  'familyroom', 'lounge', 'toilet',
)

