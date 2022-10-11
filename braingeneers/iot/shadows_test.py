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

from credentials import API_KEY

endpoint = "http://localhost:1337/api"
Stream_shadow ={
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
instance = sh.DatabaseInteractor(overwrite_endpoint=endpoint, overwrite_api_key=token)
print(instance.endpoint)
thing = instance.create_interaction_thing("BioPlateScope", "full_test_thing6")
print(thing)

thing2 = instance.create_interaction_thing("BioPlateScope", "Evee")
thing2.add_to_shadow(Stream_shadow)
# # thing.attributes["name"] = "name_change_test"
# thing.push()
# print(thing)
# shadow_add = {"boobbbobob" : "gnarpants", "barf" : "gnar", "gwar":{"hello":"world"}}
thing = instance.create_interaction_thing("BioPlateScope", "full_test_thing6")
thing.add_to_shadow({"group-id": "C"})

plate = instance.create_plate("testy_plate_obj_6", 2, 4)

print(plate)
# plate.pull()
# print(plate)
thing.set_current_plate(plate)
print(thing)
print(plate)

instance.start_image_capture(thing, "test_uuuid_3")
plate.pull()
print(plate)
# thing = instance.add_to_shadow(thing, shadow_add)
# print("updated thing: ", thing)
# experiment = instance.create_experiment("test_experiment_obj_7", "test_description")
# # print(type(experiment))
# thing = instance.add_experiment_to_thing(thing, experiment)
# print("added experiment", thing)
# thing = instance.update_thing_on_database(thing)
# print(thing)

# thing = instance.add_plate_to_thing(thing, plate)
# plate = instance.get_plate(plate.id)
# print(thing)
# print(plate)
# print(type(plate))
# print(type(thing))
# print(type(experiment))
# thing.add_to_shadow("fart","sparkle")

# print(thing.to_json())

# thing.add_to_shadow("barg","sparkleeness")
# thing.add_to_shadow("barg",{"nuts":"sparkle"})
# print(thing.to_json())
# print(instance.create_interaction_thing("test_thingy", "BioPlateScope", "test_description", {"test_key": "test_value"}))

# thing = instance.get_thing_from_database("Forky")
# print(thing.to_json())
# thing.add_to_shadow("fartt","sparklettt")
# print(thing.to_json())
# instance.update_thing_on_database(thing)


# print(plate.to_json())
# instance.sync_plate(plate)
# plate = instance.get_plate(10)
# print(plate)
# experiment = instance.create_experiment("test_experiment_obj_3", "test_description")
# experiment = instance.get_experiment(9)
# print(experiment)
# experiment.plates.append(10)
# experiment.plates.append(6)

# instance.sync_experiment(experiment)
# experiment = instance.get_experiment(9)
# print(experiment)

