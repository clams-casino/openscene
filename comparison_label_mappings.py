HABITAT_OGN_LABELS_TO_TEXT_PROMPTS = {
    # 21 habitat object goal nav categories along with 'other' category
    # Map each raw label to a "gramatically correct" text prompt to be encoded
    # by the text embedding model
    "chair": "a chair in a scene",
    "table": "a table in a scene",
    "picture": "a picture in a scene",
    "cabinet": "a cabinet in a scene",
    "cushion": "a cushion in a scene",
    "sofa": "a sofa in a scene",
    "bed": "a bed in a scene",
    "chest_of_drawers": "a chest of drawers in a scene",
    "plant": "a plant in a scene",
    "sink": "a sink in a scene",
    "toilet": "a toilet in a scene",
    "stool": "a stool in a scene",
    "towel": "a towel in a scene",
    "tv_monitor": "a television monitor in a scene",
    "shower": "a shower in a scene",
    "bathtub": "a bathtub in a scene",
    "counter": "a counter in a scene",
    "fireplace": "a fireplace in a scene",
    "gym_equipment": "gym equipment in a scene",
    "seating": "seating in a scene",
    "clothes": "clothes in a scene",

    "other": "other in a scene",
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



MATTERPORT_REGION_LABELS_TO_TEXT_PROMPTS = {
    # Matterport regions labels along with 'other room' category
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "closet": "closet",
    "dining room": "dining room",
    "garage": "garage",
    "hallway": "hallway",
    "library": "library",
    "laundryroom/mudroom": "laundry room or mudroom",  
    "kitchen": "kitchen",
    "living room": "living room",
    "meetingroom/conferenceroom": "meeting room or conference room",
    "office": "office",
    "porch/terrace/deck/driveway": "porch or terrace or deck or driveway",
    "rec/game": "recreation or game room",
    "stairs": "stairs",
    "utilityroom/toolroom": "utility room or tool room",
    "tv": "cinema or home theater or theater",
    "workout/gym/exercise": "gym",
    "balcony": "balcony",
    "bar": "bar",
    "classroom": "classroom",
    "spa/sauna": "spa or sauna",


    ### Following labels are blacklisted but query them in the text prompt anyways
    "entryway/foyer/lobby": "entryway or foyer or lobby", 
    "outdoor": "outdoor",
    "dining booth": "dining booth",

    "other room": "other room",


    # Following labels are mapped into less ambiguous labels
    # "louge" -> "living room"
    # "familyroom" -> "living room"
    # "toilet" -> "bathroom"
}

MATTERPORT_REGION_LABELS = (
    "bathroom",
    "bedroom",
    "closet",
    "dining room",
    "garage",
    "hallway",
    "library",
    "laundryroom/mudroom",
    "kitchen",
    "living room",
    "meetingroom/conferenceroom",
    "office",
    "porch/terrace/deck/driveway",
    "rec/game",
    "stairs",
    "utilityroom/toolroom",
    "tv",
    "workout/gym/exercise",
    "balcony",
    "bar",
    "classroom",
    "spa/sauna",

    "entryway/foyer/lobby",
    "outdoor",
    "dining booth",

    "other room",
)