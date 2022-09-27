import shadows as sh

from credentials import API_KEY

endpoint = "http://braingeneers.gi.ucsc.edu:1337/api"

token = API_KEY
# Create a shadow object
instance = sh.DatabaseInteractor(endpoint, API_KEY)

# thing = instance.Thing("BioPlateScope", "test_thing")

# print(thing.to_json())

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

# plate = instance.create_plate("test_plate_obj", 2, 4)
# print(plate.to_json())
# instance.sync_plate(plate)
plate = instance.get_plate(10)
print(plate)
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

