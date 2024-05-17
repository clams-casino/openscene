HABITAT_OGN_LABEL_MAPPINGS = {
    # 21 habitat object goal nav categories along with 'other' category
    "chair": ["chair"],
    "table": ["table"],
    "picture": ["picture"],
    "cabinet": ["cabinet"],
    "cushion": ["cushion"],
    "sofa": ["sofa"],
    "bed": ["bed"],
    "chest_of_drawers": ["chest of drawers"],
    "plant": ["plant"],
    "sink": ["sink"],
    "toilet": ["toilet"],
    "stool": ["stool"],
    "towel": ["towel"],
    "tv_monitor": ["television monitor"],
    "shower": ["shower"],
    "bathtub": ["bathtub"],
    "counter": ["counter"],
    "fireplace": ["fireplace"],
    "gym_equipment": ["gym equipment"],
    "seating": ["seating"],
    "clothes": ["clothes"],

    "other": ["other"],
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

