
import requests
import time

# from credentials import API_KEY
endpoint = "http://braingeneers.gi.ucsc.edu:1337/api"
API_KEY = "boof"
token = API_KEY

class DatabaseInteractor:

    def create_device(self, device_name: str, device_type: str, shadow={}, description="") -> None:
        """
        This function creates a new device in the Strapi Shadows Database.

        It may be called once or multiple times, when called multiple times only the first call
        has any effect, subsequent calls will identify that the device already exists and do nothing.

        Will throw an exception if you try to create an existing device_name with a device_type that
        doesn't match the existing device.

        :param device_name: Name of the device, for example 'marvin'
        :param device_type: Device type as defined in AWS, standard device types are ['ephys', 'picroscope', 'feeding']
        """
        api_url = endpoint + "/interaction-things/"
        headers = {"Authorization": "Bearer " + token}
        info = {
            "data": {
                "name": device_name,
                "description": description,
                "type": device_type,
                "shadow": shadow
            }
        }

        response = requests.post(api_url, headers=headers, json=info)
        # response = requests.post(api_url, json=info, headers={
        #                         'Authorization': 'bearer ' + token})
        return response.json()