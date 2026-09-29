W_WEATHER = 0.40
W_TIME    = 0.10
W_ROAD    = 0.40
W_LANE    = 0.10

WEATHER_SCORES = {
    "clear": 0.20, "sunny": 0.25, "cloudy": 0.35, "overcast": 0.40,
    "rain": 0.90, "snow": 0.95, "fog": 0.60, "mixed": 0.60, "rainstorm": 0.95,
    "sand": 0.60, "sandstorm": 0.60,
}
TIME_SCORES = {"day": 0.30, "dusk": 0.50, "night": 0.80, "dawn": 0.40}
ROAD_SCORES = {
    "highway": 0.30, "urban intersection": 0.75, "residential street": 0.50,
    "rural road": 0.45, "roundabout": 0.60, "tunnel": 0.80, "bridge": 0.45,
    "ramp": 0.55, "construction zone": 0.90, "busy downtown": .90
}


LANE_SCORING_RULES = [
    ("blocked/closed/impassable", 0.90),
    ("lane blocked", 0.90),
    ("lane closed", 0.90),
    ("construction zone", 0.90),

    ("merge", 0.70),
    ("split", 0.70),
    ("exit ramp", 0.70),
    ("entry ramp", 0.70),
    ("lane drop", 0.70),
    ("weaving section", 0.70),
    ("diverging lane", 0.70),

    ("one-way", 0.10),
    ("two/2-lane", 0.20),
    ("multi-lane", 0.30),

    ("no marking", 0.85),
    ("faded", 0.85),
    ("poorly marked", 0.85),
    ("wet surface", 0.85),
    ("icy surface", 0.85),
    ("temporary markings", 0.85),

    ("sharp curve", 0.70),
    ("steep incline", 0.70),
    ("tunnel lane", 0.70),
    ("bridge lane", 0.70),

    ("default", 0.40),
]


ALPHA = 1.0
BETA  = 1.0
PRIORITY = {
    "pedestrian": 1.00, "construction worker": 1.00, "wheelchair user" : 1.00,
    "crossing guard":1.00, "runner":1.00, "scooter rider": 1.00,"motorcyclist":1.00,
    "cyclist":1.00,

    "police car":.9, "ambulance": .9, "fire truck":.9, "emergency vehicle":.9,
    "Construction zone car":.9,

    "Train":.8 , "school bus": .8, "taxis": .8,"van": .8, "tow vehicle":.8,

    "bicycle": 0.7, "scooter": 0.7, "e scooter": 0.7, "motorcycle": 0.7,"delivery robots":0.7,

    "cattle": 0.7, "sheep": 0.7, "horse": 0.7,"deer":0.7,
    "cat": 0.7, "dog": 0.7, "animal other": 0.7,


    "crosswalk sign":0.6, "traffic light":0.6, "stop sign":0.6, "yield sign":0.6,
    "speed bump":0.6, "direction arrow":0.6, "construction zone sign":0.6,
    "road marking faded":0.6,

    "cone":0.5, "barrier":0.5, "barrier gate":0.5, "crossing gate":0.5, " trash bin":0.5,
    "construction equipment":0.5, "garbage bag":0.5, "snow pile":0.5,
    "pothole":0.5, "fallen tree":0.5,

    "Parked car":.6, "parked truck":.6, "parked bus":.6, "parked motorcycle":.6,

    "accident car": .6, "accident truck": .6, "accident bus": .6, "accident motorcycle": .6,

    "Trees": 0.2, "plastic bags" : 0.2, "buildings": 0.2, "fences": 0.2,


}

def as_prompt_dicts():
    return {
        "weights": {"ww": W_WEATHER, "wt": W_TIME, "wr": W_ROAD, "wl": W_LANE},
        "weather_scores": WEATHER_SCORES,
        "time_scores": TIME_SCORES,
        "road_scores": ROAD_SCORES,
        "lane_rules": LANE_SCORING_RULES,
        "alpha": ALPHA,
        "beta":  BETA,
        "priority": PRIORITY
    }