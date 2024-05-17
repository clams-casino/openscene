HABITAT_OGN_LABEL_MAPPINGS = {
    "wall": ["wall"],
    "floor": ["floor"],
    # "chair": ["chair"],
    "door": ["door"],
    # "table": ["table"],
    # "picture": ["picture"],
    # "cabinet": ["cabinet"],
    # "cushion": ["cushion"],
    "window": ["window"],
    # "sofa": ["sofa"],
    # "bed": ["bed"],
    "curtain": ["curtain"],
    # "chest_of_drawers": ["chest of drawers"],
    # "plant": ["plant"],
    # "sink": ["sink"],
    "stairs": ["stairs"],
    "ceiling": ["ceiling"],
    # "toilet": ["toilet"],
    # "stool": ["stool"],
    # "towel": ["towel"],
    "mirror": ["mirror"],
    # "tv_monitor": ["television monitor"],
    # "shower": ["shower"],
    "column": ["structural column"],
    # "bathtub": ["bathtub"],
    # "counter": ["countertop"],
    # "fireplace": ["fireplace"],
    "lighting": ["light"],
    "beam": ["structural beam"],
    "railing": ["railing"],
    "shelving": ["shelf"],
    "blinds": ["window blinds"],
    # "gym_equipment": ["gym equipment"],
    # "seating": ["seating"],
    "board_panel": ["board panel"],
    "furniture": ["furniture"],
    "appliances": ["home appliance"],
    # "clothes": ["clothes"],
    "misc": [],
    "objects": [],
    "void": [],
    "unlabeled": [],
}





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

