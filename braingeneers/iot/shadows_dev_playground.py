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
# endpoint = "http://braingeneers.gi.ucsc.edu:1337/api"
# ENDPOINT = "http://shadows-db.braingeneers.gi.ucsc.edu/api"

# token = API_KEY
# Create a shadow object
# instance = sh.DatabaseInteractor(overwrite_endpoint = ENDPOINT, overwrite_api_key=token)
instance = sh.DatabaseInteractor()
# print(json.dumps(instance.get_device_state(13), indent=4))
# print(instance.list_experiments())
# print(instance.get_device_state_by_name("Evee"))
# print(instance.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=BioPlateScope"))
print(instance.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=BioPlateScope"))

instance.empty_trash()

thing1 = instance.create_interaction_thing("Other", "delete_test")
# shadow = {"params": {
#                 "interval": 1,
#                 "stack_size": 15,
#                 "stack_offset": 750,
#                 "step_size": 50,
#                 "camera_params": "-t 4000 -awb off -awbg 1,1 -o",
#                 "light_mode": "Above"
#             },
#             "connected": 1691797301180,
#             "timestamp": "2023-08-11T23:19:56",
#             "uuid": "2023-09-09-i-test",
#             "time": {
#                 "elapsed": 97924,
#                 "seconds": 97.924,
#                 "minutes": 1.6320666666666668
#             },
#             "num_cameras": 9,
#             "last-upload": "",
#             "experiment-state": "engaged",
#             "group-id": "I"
#         }
# thing1.add_to_shadow(shadow)
print(json.dumps(thing1.to_json(), indent=4))

thing1.move_to_trash()

print(instance.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=Other"))

print(json.dumps(thing1.to_json(), indent=4))

# thing1.recover_from_trash()

print(instance.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=Other"))

# plate = instance.create_plate("ephys-tester",1,1)
# plate.add_thing(thing1)
# plate.add_entry_to_ephys_params(uuid="2023-04-17-e-causal_v1", channels="362-71-31-338-231-315-309-234", timestamp="20230419_153528", data_length="-1")
# thing2 = instance.create_interaction_thing("BioPlateScope", "Evee")
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

