"""Shared DKASC Alice Springs raw-data schema for PV audit scripts."""

SITE_NAME = "DKASC Site 18"
SITE_RATED_POWER_KW = 5.2

COLUMN_MAP = {
    "timestamp": "timestamp",
    "71_DKA_M2_C_Phase_Active_Power": "Active_Power",
    "101_DKA_WeatherStation_Global_Horizontal_Radiation": "Global_Horizontal_Radiation",
    "101_DKA_WeatherStation_Diffuse_Horizontal_Radiation": "Diffuse_Horizontal_Radiation",
    "101_DKA_WeatherStation_Radiation_Global_Tilted": "Radiation_Global_Tilted",
    "101_DKA_WeatherStation_Radiation_Diffuse_Tilted": "Radiation_Diffuse_Tilted",
    "101_DKA_WeatherStation_Weather_Temperature_Celsius": "Weather_Temperature_Celsius",
    "101_DKA_WeatherStation_Weather_Relative_Humidity": "Weather_Relative_Humidity",
    "101_DKA_WeatherStation_Wind_Speed": "Wind_Speed",
    "101_DKA_WeatherStation_Wind_Direction": "Wind_Direction",
    "101_DKA_WeatherStation_Weather_Daily_Rainfall": "Weather_Daily_Rainfall",
}

REQUIRED_RAW_COLUMNS = list(COLUMN_MAP.keys())
OUTPUT_COLUMNS = list(COLUMN_MAP.values())
NUMERIC_COLUMNS = [name for name in OUTPUT_COLUMNS if name != "timestamp"]
RADIATION_COLUMNS = [
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]
WEATHER_COLUMNS = [
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Wind_Speed",
    "Wind_Direction",
    "Weather_Daily_Rainfall",
]

FEATURE_COLUMNS = [
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
]
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
TARGET_COLUMN = "Active_Power"

SAMPLING_MINUTES = 5
HISTORY_STEPS = 72
HORIZON_STEPS = 3
