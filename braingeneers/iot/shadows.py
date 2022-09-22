import requests
import time

class DatabaseInteractor:
    """
    This class provides methods for interacting with the Strapi Shadows database.

    See documentation at: https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md

    Assumes the following:
        - The Strapi database is running at the endpoint specified in the constructor
        - User has an API key for the Strapi database

    Public functions:

        #
        # List and register IoT devices
        #
        list_devices(**filters)  # list connected devices, filter by one or more state variables.
        create_device(device_name: str, device_type: str)  # create a new device if it doesn't already exist

        #
        # Get/set/update/subscribe to device state variables
        #
        # todo:
            get_device_state(device_name: str)  # returns the device shadow file as a dictionary.
            update_device_state(device: str, device_state: dict)  # updates one or more state variables for a registered device.
            set_device_state(device_name: str, state: dict)  # saves the shadow file, a dict that is JSON serializable.
            subscribe_device_state_change(device: str, device_state_keys: List[str], callback: Callable)  # subscribe to notifications when a device state changes.

    """

    def __init__(self, endpoint, api_token) -> None:
        self.endpoint = endpoint
        self.token = api_token

    def create_interaction_thing(self, name, interaction_type, description="", shadow={}):
        api_url = self.endpoint + "/interaction-things/"
        # print(api_url)
        headers = {"Authorization": "Bearer " + self.token}
        info = {
            "data": {
                "name": name,
                "description": description,
                "type": interaction_type,
                "shadow": shadow
            }
        }

        response = requests.post(api_url, headers=headers, json=info)
        # response = requests.post(api_url, json=info, headers={
        #                         'Authorization': 'bearer ' + self.token})
        return response.json()


    def update_experiment_on_interaction_thing(self, interaction_thing_id, experiment_id):
        url = self.endpoint + "/interaction-things/" + str(interaction_thing_id)
        headers = {"Authorization": "Bearer " + self.token}
        data = {
            "experiment_id": experiment_id
        }
        response = requests.put(url, headers=headers, json=data)
        return response


    def update_plate_on_interaction_thing(self, interaction_thing_id, plate_id):
        url = self.endpoint + "/interaction-things/" + str(interaction_thing_id)
        headers = {"Authorization": "Bearer " + self.token}
        data = {
            "plate_id": plate_id
        }
        response = requests.put(url, headers=headers, json=data)
        return response

    def get_interaction_thing_id_from_name(self, name):
        url = self.endpoint + "/interaction-things?filters[name][$eq]=" + name
        headers = {"Authorization": "Bearer " + self.token}
        response = requests.get(url, headers=headers)
        print(response.json())
        return response.json()['data'][0]['id']

    def update_shadow_without_overwrite(self, interaction_thing_id, shadow):
        url = self.endpoint + "/interaction-things/" + str(interaction_thing_id)
        headers = {"Authorization": "Bearer " + self.token}
        data = {
            "shadow": shadow
        }
        response = requests.put(url, headers=headers, json=data)
        return response

    def list_all_interaction_things(self):
        url = self.endpoint + "/interaction-things"
        headers = {"Authorization": "Bearer " + self.token}
        response = requests.get(url, headers=headers)
        return response.json()

## Experiments

## Experiments
#     Create, delete, update, get, list

    ##create methods:
    def create_experiment(self, name, description):
        api_url = self.endpoint+"/experiments/"
        info = {
            "data": {
                "name": name,
                "description": description
            }
        }
        response = requests.post(api_url, json=info, headers={
                                'Authorization': 'bearer ' + self.token})
        print(response.status_code)
        print(response.json())
        return response.json()['data']['id']

    def generate_plate_for_experiment(self, experiment_id, plate_name):
        plate_id = self.create_plate(plate_name, 2, 3)
        self.add_plate_to_experiment_without_overwriting(experiment_id, plate_id)
        return plate_id

    ## update methods:
    def add_plate_to_experiment(self, experiment_id, plate_id):
        api_url = self.endpoint+"/experiments/"+str(experiment_id)
        info = {
            "data": {
                "plates": [plate_id]
            }
        }
        response = requests.put(api_url, json=info, headers={
            'Authorization': 'bearer ' + self.token})
        print(response.status_code)
        print(response.json())

    def add_plate_to_experiment_without_overwriting(self, experiment_id, plate_id):
        experiment = self.get_experiment(experiment_id) 
        # experiment['data']['attributes']['plates']['data']['id'].append(plate_id) 
        plates = [] 
        for i in experiment['data']['attributes']['plates']['data']:
            print(i['id'])
            plates.append(str(i['id']))
        # print(experiment['data']['attributes']['plates']['data'][0]['id'])  
        plates.append(str(plate_id)) 
        api_url = self.endpoint+"/experiments/"+str(experiment_id)
        info = {
            "data": {
                "plates": plates
            }
        }
        response = requests.put(api_url, json=info, headers={
            'Authorization': 'bearer ' + self.token})
            
        print(response.status_code)
        print(response.json())

    ## get methods:
    def get_experiment(self, experiment_id):
        api_url = self.endpoint+"/experiments/"+str(experiment_id)+"?populate=%2A"
        response = requests.get(api_url, headers={
            'Authorization': 'bearer ' + self.token})
        # print(response.status_code)
        # print(response.json())
        return response.json()

    def list_experiments(self):
        api_url = self.endpoint+"/experiments/"
        response = requests.get(api_url, headers={
            'Authorization': 'bearer ' + self.token})
        # print(response.status_code)
        # print(response.json())
        return response.json()
