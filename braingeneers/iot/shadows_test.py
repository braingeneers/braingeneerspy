'''
How do we want this to flow

Create interaction thing
    - only happens once, when a new device is added to the system
    - contains important imaging data

Create Experiment
    - happens when a new experiment is started
    - Creates a plate or assigns existing plate to experiment
    - assigns current experiment to interaction thing?

Create Plate
    - Plate spawns wells, wells don't exist outside of plates
    - Plates have interaction things
    - Imaging metadata should be transferred to the plate from the interaction thing

Starting an image run
    - Image uuid must be passed to plate object
    - should not be able to start image run if interaction thing has no plate


'''


import shadows as sh
import json

from credentials import API_KEY

# endpoint = "http://localhost:1337/api"
shadow = {
  "welcome": "aws-iot",
  "params": {
    "interval": 1,
    "stack_size": 15,
    "stack_offset": 750,
    "step_size": 50,
    "camera_params": "-t 4000 -awb off -awbg 1,1 -o",
    "light_mode": "Above"
  },
  "connected": 1659472053691,
  "timestamp": "2022-07-27T17:49:40",
  "uuid": "2022-07-11-i-connectoid-3",
  "time": {
    "elapsed": 680368,
    "seconds": 680.368,
    "minutes": 11.339466666666667
  },
  "active_cameras": [
    "11",
    "13",
    "14",
    "15",
    "16",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "41",
    "42",
    "44",
    "45",
    "46"
  ],
  "num_cameras": 22,
  "last-upload": "2022-01-17T23:19:09",
  "experiment-state": "stopped",
  "group-id": "C"
}
token = API_KEY
# Create a shadow object
# instance = sh.DatabaseInteractor(overwrite_endpoint=endpoint, overwrite_api_key=token)
instance = sh.DatabaseInteractor()
# print(json.dumps(instance.get_device_state(13), indent=4))

# print(instance.list_experiments())
# print(instance.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=BioPlateScope"))

# thing1 = instance.create_interaction_thing("BioPlateScope", "StreamTest")
thing2 = instance.create_interaction_thing("BioPlateScope", "Evee")
# thing2.add_to_shadow(shadow)
print(thing2.attributes["name"])
# experiment1 = instance.create_experiment("Feed-Frequency-06-26-2022","Feed frequency experiment")
# experiment2 = instance.create_experiment("Connectoids-06-26-2022","Feed frequency experiment")
# uuids1 = {"uuids":
#             {
#                 "2022-06-26-i-feed-frequency-4": "G",
#                 "2022-06-28-i-feed-frequency-5": "G"
#             }
# }
# uuids2 = {"uuids":
#             {
#                 "2022-06-28-i-connectoid" : "C",
#                 "2022-06-28-i-connectoid-2":"C",
#                 "2022-06-29-i-connectoids":"C",
#                 "2022-06-29-i-connectoid-2":"C",
#                 "2022-07-11-i-connectoid-3":"C"
#             }
# }
# plate1 = instance.create_plate("Fluidic-24-well-06-26-2022",4,6,uuids1)
# plate2 = instance.create_plate("Connectoid-plate-06-26-2022",4,6,uuids2)
# experiment1.add_plate(plate1)
# experiment2.add_plate(plate2)
# thing1.set_current_experiment(experiment1)
# thing2.set_current_experiment(experiment2)
# thing1.set_current_plate(plate1)
# thing2.set_current_plate(plate2)

