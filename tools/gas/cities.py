"""Fixed demand-weighting cities for the population-weighted degree days (chosen a priori, before any score).
name -> (latitude, longitude, weight = 2020 US Census metro (MSA) population, millions)."""
CITIES = {
    "new_york": (40.7128, -74.0060, 20.1),
    "chicago": (41.8781, -87.6298, 9.6),
    "philadelphia": (39.9526, -75.1652, 6.2),
    "boston": (42.3601, -71.0589, 4.9),
    "detroit": (42.3314, -83.0458, 4.4),
    "minneapolis": (44.9778, -93.2650, 3.7),
    "pittsburgh": (40.4406, -79.9959, 2.4),
    "atlanta": (33.7490, -84.3880, 6.1),
    "dallas": (32.7767, -96.7970, 7.6),
    "houston": (29.7604, -95.3698, 7.1),
}
BASE_F = 65.0      # degree-day base temperature (deg F), US convention
