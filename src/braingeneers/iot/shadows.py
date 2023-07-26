import requests
import configparser
import os
import io
from typing import Union



class DatabaseInteractor:
    """
    This class provides methods for interacting with the Strapi Shadows database.

    See documentation at: ...

    Assumes the following:
        - ~/.aws/credentials file exists and contains a section [strapi] with the following keys:
            - endpoint: the URL of the Strapi server
            - api_key: the API key for the Strapi server
        
    Public functions:


    """
    def __init__(self , credentials: Union[str, io.IOBase] = None, overwrite_endpoint = None, overwrite_api_key = None) -> None:
        # self.endpoint = endpoint
        # self.token = api_token

        if credentials is None:
            credentials = os.path.expanduser('~/.aws/credentials')  # default credentials location

        if isinstance(credentials, str):
            with open(credentials, 'r') as f:
                self._credentials = f.read()
        else:
            assert hasattr(credentials, 'read'), 'credentials parameter must be a filename string or file-like object.'
            self._credentials = credentials.read()

        config = configparser.ConfigParser()
        config.read_file(io.StringIO(self._credentials))
        assert 'strapi' in config, 'Your AWS credentials file is missing a section [strapi], ' \
                                    'you may have the wrong version of the credentials file.'
        assert 'endpoint' in config['strapi'], 'Your AWS credentials file is malformed, ' \
                                                'endpoint was not found under the [strapi] section.'
        assert 'api_key' in config['strapi'], 'Your AWS credentials file is malformed, ' \
                                                'api_key was not found under the [strapi] section.'

        self.endpoint = config['strapi']['endpoint']
        self.token = config['strapi']['api_key']
        if overwrite_endpoint:
            self.endpoint = overwrite_endpoint
        if overwrite_api_key:
            self.token = overwrite_api_key
    
    class __API_object:
        """
        This class is used to represent objects in the database as python objects
        """
        def __init__(self, endpoint, api_token, api_object_id):
                self.endpoint = endpoint
                self.token = api_token
                self.id = None
                self.attributes = {}
                self.api_object_id = api_object_id

        def __str__(self):
            var_list = filter(lambda x: x not in ["endpoint", "token", "api_object_id"], vars(self))
            return str({var: getattr(self, var) for var in var_list})

        def to_json(self):
            var_list = filter(lambda x: x not in ["endpoint", "token", "api_object_id"], vars(self))
            return {var: getattr(self, var) for var in var_list}

        def parse_API_response(self, response_data):
            """
            parses the response from the API and updates the python object
            """
            self.id = response_data['id']
            self.attributes = response_data['attributes']
            for key in self.attributes:
                # print(key, self.attributes[key])
                if type(self.attributes[key]) is dict and "data" in self.attributes[key]:
                    if self.attributes[key]["data"] is not None and len(self.attributes[key]["data"]) != 0:
                        # print("found data", self.attributes[key]["data"])
                        item_list = []
                        if type(self.attributes[key]["data"]) is list:
                            for item in self.attributes[key]["data"]:
                                # print("item", item)
                                if "id" in item:
                                    item_list.append(item["id"])
                        else:
                            item_list.append(self.attributes[key]["data"]["id"])

                        self.attributes[key] = item_list
                    else:
                        self.attributes[key] = []

        def spawn(self):
            """
            creates a new object in the database
            """
            url = self.endpoint + "/"+self.api_object_id+"?filters[name][$eq]=" + self.attributes["name"] + "&populate=%2A"
            headers = {"Authorization": "Bearer " + self.token}
            response = requests.get(url, headers=headers)
            # print("spawn response " ,response.json())
            if len(response.json()['data']) == 0:
                # thing = self.Thing(type, name)
                api_url = self.endpoint+"/"+self.api_object_id+"?populate=%2A"
                data = {"data": self.attributes}
                response = requests.post(api_url, json=data, headers={
                                        'Authorization': 'bearer ' + self.token})
                # print(response.status_code)
                # print("response after creating new object", response.json())
                if response.status_code == 200:
                    self.parse_API_response(response.json()['data'])
                    # self.id = response.json()['data']['id']
            else:
                print(self.api_object_id + " object already exists")
                # print(response.json())
                try:
                    # print("parse API response", response.json()['data'][0])
                    self.parse_API_response(response.json()['data'][0])
                except KeyError:
                    print("some values are missing")

        def push(self):
            """
            updates the database with the current state of the object
            """
            url = self.endpoint + "/"+self.api_object_id+"/" + str(self.id) + "?populate=%2A"
            headers = {"Authorization": "Bearer " + self.token}
            data = {"data": self.attributes}
            response = requests.put(url, headers=headers, json=data)
            # print(response.json())
            # print(response.status_code)
            self.parse_API_response(response.json()['data'])

        def pull(self):
            """
            updates object with the latest data from the database
            """
            url = self.endpoint + "/"+self.api_object_id+"/" + str(self.id) + "?populate=%2A"
            headers = {"Authorization": "Bearer " + self.token}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                raise Exception("Object not found")
            # print(response.json())
            # print(response.status_code)

            self.parse_API_response(response.json()['data'])

        def get_by_name(self, name):
            """
            gets the object from the database by name
            """
            url = self.endpoint + "/"+self.api_object_id+"?filters[name][$eq]=" + name + "&populate=%2A"
            headers = {"Authorization": "Bearer " + self.token}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                # raise Exception("No object with name " + name + " found")
                raise Exception("no " + self.api_object_id + " object with name " + name)
            else:
                self.parse_API_response(response.json()['data'][0])

    class __Thing(__API_object):
        def __init__(self, endpoint, api_token):
            super().__init__(endpoint, api_token, "interaction-things")

        def add_to_shadow(self, json):
            if self.attributes["shadow"] is None:
                self.attributes["shadow"] = json
            else:
                for key, value in json.items():
                    self.attributes["shadow"][key] = value

            self.push()

        def add_uuid_to_shadow(self, uuid):
            if self.attributes["shadow"] is None:
                self.attributes["shadow"] = {}
            self.attributes["shadow"]["uuid"] = uuid
            self.push()            


        def set_current_plate(self, plate):
            """
            updates the current plate of the thing and adds the plate to the list of all plates historically associated with the thing.
            The plates list relation also updates the thing relation on the plate object itself. 
            """

            if self.attributes["plates"] is None:
                self.attributes["plates"] = []
            self.attributes["plates"].append(plate.id)
            self.attributes["current_plate"] = plate.id

            plate.add_thing(self)
            self.push()

        def set_current_experiment(self, experiment):
            """

            updates the current experiment of the thing
            """
            # if self.attributes["experiments"] is None:
            #     self.attributes["experiments"] = []
            # self.attributes["experiments"].append(experiment.id)
            self.attributes["current_experiment"] = experiment.id
            self.push()

    class __Experiment(__API_object):
        def __init__(self, endpoint, api_token):
            super().__init__(endpoint, api_token, "experiments")

        def add_plate(self, plate):
            # Bidirectional relations have an owner and a related object, plate owns this relation
            plate.add_experiment(self)
            self.pull()

    class __Plate(__API_object):
        def __init__(self, endpoint, api_token):
            super().__init__(endpoint, api_token, "plates")

        def add_thing(self, thing):
            """
            add_thing_to_plate

            adds the thing to the list of things associated with the plate. also updates the plate relation on the thing object itself.
            does not effect current_plate value of thing
            """
            if self.attributes["interaction_things"] is None:
                self.attributes["interaction_things"] = []
            self.attributes["interaction_things"].append(thing.id)
            self.push()

        def add_uuid_to_image_params(self, uuid):
            if self.attributes["image_parameters"] is None:
                self.attributes["image_parameters"] = {}

            # if attributes["image_parameters] has attribute uuid" and it is not none
            if "uuids" not in self.attributes["image_parameters"] or self.attributes["image_parameters"]["uuids"] is None:
                self.attributes["image_parameters"]["uuids"] = {}
            for key, value in uuid.items():
                self.attributes["image_parameters"]["uuids"][key] = value

            self.push()
        
        def add_experiment(self, experiment):
            if self.attributes["experiments"] is None:
                self.attributes["experiments"] = []
            self.attributes["experiments"].append(experiment.id)
            self.push()

    class __Well(__API_object):
        def __init__(self, endpoint, api_token):
            super().__init__(endpoint, api_token, "wells")
        
    class __Sample(__API_object):
        def __init__(self, endpoint, api_token):
            super().__init__(endpoint, api_token, "samples")

    def create_interaction_thing(self, type, name):
        thing = self.__Thing(self.endpoint, self.token)
        thing.attributes["name"] = name
        thing.attributes["type"] = type
        thing.spawn()
        return thing

    def create_plate(self, name, rows, columns, image_params = {}):
        plate = self.__Plate(self.endpoint, self.token)
        plate.attributes["name"] = name
        plate.attributes["rows"] = rows
        plate.attributes["columns"] = columns
        plate.attributes["image_parameters"] = image_params
        plate.spawn()

        if len(plate.attributes["wells"]) == 0 or plate.attributes["wells"] is None:
            for i in range(1, rows+1):
                for j in range(1, columns+1):
                    well = self.__Well(self.endpoint, self.token)
                    well.attributes["name"] = plate.attributes["name"]+"_well_"+str(i)+str(j)
                    well.attributes["position_index"] = str(i) + str(j)
                    well.attributes["plate"] = plate.id
                    well.spawn()

        plate.pull()
        return plate
    
 
    def create_experiment(self, name, description):
        experiment = self.__Experiment(self.endpoint, self.token)
        experiment.attributes["name"] = name
        experiment.attributes["description"] = description
        experiment.spawn()
        return experiment
        

    def start_image_capture(self, thing, uuid):
        thing.add_uuid_to_shadow(uuid)
        group_id = thing.attributes["shadow"]["group-id"]
        value = { uuid : group_id }
        if thing.attributes["current_plate"]:
            plate = self.__Plate(self.endpoint, self.token)
            plate.id = thing.attributes["current_plate"][0]
            plate.pull()
            plate.add_uuid_to_image_params(value)
        else: 
            #raise exception
            raise Exception("no plate associated with thing")


    # a method that returns a list of all experiments by name
    def list_objects(self, api_object_id, filter = "?"):
        """
        when you need a list of the objects in the database

        useful for populating dropdown lists in plotly dash
        """
        url = self.endpoint + "/"+  api_object_id + filter +"&populate=%2A"
        headers = {"Authorization": "Bearer " + self.token}
        response = requests.get(url, headers=headers)
        # print(response.json())
        # print(response.status_code)
        return response.json()['data']

    def list_objects_with_name_and_id(self, api_object_id, filter = "?"):
        """
        when you need a list of the objects in the database

        returns a list of dictionaries with the name and id of each object

        """
        response = self.list_objects(api_object_id, filter)
        return [{"label": x["attributes"]["name"], "value": x["id"]} for x in response]

    def list_experiments(self):
        response = self.list_objects("experiments")
        output = []
        for i in response:
            # print(i["attributes"]["name"])
            output.append(i["attributes"]["name"])
        return output

    def list_BioPlateScopes(self):
        return self.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=BioPlateScope")

    def list_devices_by_type(self, thingTypeName):
        return self.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]="+thingTypeName)

    def get_device_state(self, thing_id):
        thing = self.__Thing(self.endpoint, self.token)
        thing.id = thing_id
        thing.pull()
        return thing.attributes["shadow"]

    def get_device_state_by_name(self, thing_name):
        thing = self.__Thing(self.endpoint, self.token)
        thing.get_by_name(thing_name)
        return thing.attributes["shadow"]



    """
    Getters for objects from their id numbers
    """

    def get_device(self, thing_id= None, name = None):
        if thing_id is None and name is None:
            raise Exception("must provide either thing_id or name")
        if name:
            thing = self.__Thing(self.endpoint, self.token)
            thing.get_by_name(name)
            return thing
        else:
            thing = self.__Thing(self.endpoint, self.token)
            thing.id = thing_id
            thing.pull()
            return thing
    
    def get_plate(self, plate_id):
        plate = self.__Plate(self.endpoint, self.token)
        plate.id = plate_id
        plate.pull()
        return plate

    def get_experiment(self, experiment_id):
        experiment = self.__Experiment(self.endpoint, self.token)
        experiment.id = experiment_id
        experiment.pull()
        return experiment

    def get_sample(self, sample_id):
        sample = self.__Sample(self.endpoint, self.token)
        sample.id = sample_id
        sample.pull()
        return sample

    def get_well(self, well_id):
        well = self.__Well(self.endpoint, self.token)
        well.id = well_id
        well.pull()
        return well