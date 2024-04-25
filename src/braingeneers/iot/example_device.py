from device import Device

# Or use this instead, if calling this in a repo outside braingeneerspy:
# from braingeneers.iot import Device 

class ExampleDevice(Device): 
    """ Example Device Class
    Demonstrates how to use and inherit the Device class for new application
    """
    def __init__(self, device_name, eggs = 0, ham = 0, spam = 1):
        """Initialize the ExampleDevice class
        Args: 
            device_name (str): name of the device
            ham (str): starting quantity of ham
            eggs (int): starting quantity of eggs
            spam (int): starting quantity of spam

        :param device_specific_handlers: dictionary that maps the device's command keywords 
        to a function call that handles an incomming command.
        """
        self.eggs = eggs
        self.ham = ham
        self.spam = spam

        super().__init__(device_name=device_name, device_type = "Other", primed_default=True)

        self.device_specific_handlers.update({ 
            "ADD": self.handle_add, # new command to add any amount of eggs or ham
            "LIST": self.handle_list # new command to list current amount of eggs and ham by message
        }) 
        return

    @property
    def device_state(self):
        """
        Return the device state as a dictionary. This is used by the parent Device class to update the device shadow.
        Child can add any additional device specific state to the dictionary i.e., "EGGS", "HAM", and "SPAM"
        """
        return { **super().device_state,
                "EGGS": self.eggs, 
                "HAM": self.ham,
                "SPAM": self.spam
                }

    def is_primed(self):
        """
        Modify this function if your device requires a physical prerequsite. 
        In Parent initialization, when primed_default=True no physical prerequsite is required.

        If a physical prerequsite is required, set primed_default=False and modify this function to check for a condition to be met to return True. 

        For example, you may wait for a hardware button press confirming that a physical resource is attached (i.e., a consumable, like fresh media) before allowing 
        the device to be used in an experiment.

        This function should not perform any blocking/loops because it is checked periodically by the parent loop in "IDLE" state!
        """
        return self.primed_default


    def handle_add(self, topic, message):
        """
        Function to handle the ADD command. This function is called by the parent Device class when an ADD command is received.
        Args:
            topic (str): topic of the received message
            message (dict): message received by the device
        ADD assumes that the message contains the keys "EGGS", "HAM", "SPAM", and adds the values to the device's state.
        """
        try:
            self.eggs += message["EGGS"]
            self.ham += message["HAM"]
            self.spam += message["SPAM"]
            self.update_state(self.state) # to update eggs and ham change ASAP in shadow
        except:
            self.mb.publish_message(topic= self.generate_response_topic("ADD", "ERROR"),
                                    message= { "COMMAND": "ADD-ERROR",
                                                "ERROR": f"Missing argument for EGGS, HAM, or SPAM"})
        return

    def handle_list(self, topic, message):
        """
        Function to handle the LIST command. This function is called by the parent Device class when an LIST command is received.
        Args:
            topic (str): topic of the received message
            message (dict): message received by the device
        LIST responds with a message containing the current values for "EGGS", "HAM", and "SPAM".
        """
        self.mb.publish_message(topic=self.generate_response_topic("LIST", "RESPONSE"),
                                message= { "COMMAND": "LIST-RESPONSE",
                                            "EGGS": self.eggs,
                                            "HAM" : self.ham,
                                            "SPAM" : self.spam })
        return    