import base64
import json
import os
import re
import fnmatch
import schedule
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta
import pytz
import diskcache
from functools import wraps

import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers.iot import messaging

BASEPATH_CACHE = os.path.expanduser(os.path.join('..', 'mqtt_device_cache'))
MAX_POOL_THREADS = 1

def retry(initial_delay=5, max_delay=60, max_tries=10):
    """Decorator to retry function execution on exception with an exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for tries in range(max_tries): 
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in {delay} seconds... (Attempt {tries+1}/{max_tries})")
                    time.sleep(delay) # Wait for the current delay duration before retrying
                    delay = min(delay * 2, max_delay) # Double the delay for the next retry, but don't exceed max_delay
            raise # If we've tried the maximum number of times, raise the exception.
        return wrapper
    return decorator

def format_time(t):
    return f":{t.minute:02}" if t.hour == 0 else f"{t.hour}:{t.minute:02}"

def datetime_serializer(obj):
    if isinstance(obj, datetime) and obj != datetime(1900, 1, 1): #{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    return "never"

class Device:
    """ Device Class for mqtt device """

    def __init__(self, device_name, device_type, primed_default=False):
        self.initialize_variables(device_name, device_type, primed_default)
        self.initialize_handlers()
        self.initialize_message_broker()

    def initialize_variables(self, device_name, device_type, primed_default):
            self.device_name = device_name 
            self.device_type = device_type
            self.experiment_uuid = "NONE"
            self.state = "SHUTDOWN"
            self.primed_default = primed_default

            self.root_topic = "telemetry"
            self.logging_token = "log" #for message logging job container to parse and write to csv 
            self.DEFAULT_TOPIC = f"{self.root_topic}/NONE/log/{self.device_name}/+/REQUEST"
            self.DEFAULT_GENERAL_TOPIC = f"{self.root_topic}/NONE/log/+/REQUEST"
            self.to_slack_topic = "telemetry/slack/TOSLACK/iot-experiments" #TODO: no logging token here, do we need one? @David
            self.mqtt_subscribe_topics = [self.DEFAULT_TOPIC, self.DEFAULT_GENERAL_TOPIC] # Working topic that can change

            self.naive_datetime = datetime(1900, 1, 1)
            self.pause_until = self.naive_datetime #None (in the past)
            self.exec_job_name = None
            self.exec_job_until = self.naive_datetime
            self.job_index = iter(range(0, 10**6)) #some large number
            self.who_paused = None
            self.teammates = []
            self.path = os.getcwd()
            self.time = None
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            #self.have_been_waiting_for = #time #track time a device has been in Waiting state; if more than an hour and haven't been able to perform any jobs
            #Why didn't Zambezi acccumulate jobs?
            
            # Initialize disk cache for file uploads and in case of internet outages
            self.cache = diskcache.Cache(f'{BASEPATH_CACHE}')
            # Pass the cache to the Deque constructor
            self.cache_queue = diskcache.Deque(directory=self.cache.directory)
            self.failed_jobs_queue = diskcache.Deque(directory=self.cache.directory)
            self.queue_lock = threading.Lock() #discache is threadafe but still need sync for queue add/remove atmic operations
            self.thread_pool = ThreadPoolExecutor(MAX_POOL_THREADS)
            self.futures_dict = {} #key: job_id, value: future object
            self.exec_thread = None #reference to thread running EXEC
            self.exec_semaphore = threading.Semaphore(1)
            self.stop_event = threading.Event() #signal thread to stop
            self.scheduler = schedule.Scheduler()


    def initialize_handlers(self):
        self.device_specific_handlers = {"TWIDDLE": self.handle_twiddle}
        self.handlers = {
            "START": self.handle_start,
            "PING": self.handle_ping,
            "STATUS": self.handle_status,
            "SCHEDULE": self.handle_schedule,
            "END": self.handle_end,
            "STOP": self.handle_stop,
            "PAUSE": self.handle_pause,
            "RESUME": self.handle_resume,
        }

    def initialize_message_broker(self):            
        #TODO: fix why new devices go straight to database trash and you need to take them out
        self.mb = messaging.MessageBroker(str(self.device_name + "-" + str(uuid.uuid4())))
        try:
            #contains_test_device = any(d['label'] == self.device_name for d in self.mb.list_devices_by_type(self.device_type))
            #self.mb.create_device(self.device_name, self.device_type)
            #TODO: store location in database like "device_type"
            thing1 = self.mb.shadow_interface.create_interaction_thing(self.device_type, self.device_name)
            thing1.recover_from_trash()
        except Exception as e:
            print("Shadows Data: is unreachable, device was not created:", e)
        self.update_state("IDLE")  # Set device state in shadow

    @property
    def mqtt_publish_topic(self):
        return f"{self.root_topic}/{self.experiment_uuid}/{self.logging_token}"

    @property
    def pretty_schedule(self):
        sched_display = []
        for job in self.scheduler.jobs:
            job_name = job.job_func.args[1]
            job_time_unit = {'seconds': 'sec', 'hours': 'hr'}.get(job.unit, job.unit)
            last_run = datetime_serializer(job.last_run)
            next_run = datetime_serializer(job.next_run)
            time = f"@{format_time(job.at_time)} every" if job.at_time else None
            prefix = "once" if job.job_func.__name__ == "run_once" else f"{time or 'every'} {job.interval} {job_time_unit}"
            sched_display.append(f"{prefix}, {job_name}, last:{last_run}, next:{next_run}, tag: {job.tag}")
        return sched_display

    @property
    def device_state(self):
        return {
            "STATE": self.state,
            "UUID": self.experiment_uuid,
            "TEAMMATES": self.teammates,
            "SCHEDULE": self.pretty_schedule,
            "PAUSE_UNTIL": datetime_serializer(self.pause_until),
            "WHO_PAUSED": self.who_paused,
            "EXEC_JOB": self.exec_job_name, 
            "EXEC_UNTIL": datetime_serializer(self.exec_job_until)
        }


    #----------------Helper Functions ----------------------
    # Placeholder function for override by child class: check requirements for the device to be primed
    def is_primed(self):
        #print("\nNO CONDITIONS SPECIFIED FOR PRIMING\n")
        return self.primed_default

    def update_state(self, new_state):
        print(f"UPDATE_STATE: {self.state} --> {new_state}")
        self.state = new_state 
        try:
            self.mb.update_device_state(self.device_name, self.device_state) 
        except Exception as e:
            print("Shadows Database is unreachable, device state was not updated online:", e)

    def generate_job_tag(self):
        return str(next(self.job_index))
    
    def peek_jobs(self, lookahead_seconds = 10):
        return any(job.next_run <= datetime.now() + timedelta(seconds=lookahead_seconds) for job in self.scheduler.jobs)

    def is_my_topic(self, topic):
        return self.device_name in topic.split('/')
    
    def is_my_default_topic(self, topic):
        return self.DEFAULT_TOPIC == topic

    def is_general_experiment_topic(self, topic):
        pattern = f"^{re.escape(self.root_topic)}/{re.escape(self.experiment_uuid)}/{re.escape(self.logging_token)}/.*REQUEST$"
        return bool(re.match(pattern, topic))
    
    def get_command_key_from_topic(self, topic):
        return topic.split('/')[-2]

    def get_command_value_from_topic(self, topic):
        return topic.split('/')[-1]

    def generate_response_topic(self, response_cmnd_key = None, response_cmnd_value = None):
        if response_cmnd_key == None or response_cmnd_value == None:
            return '/'.join([self.root_topic, self.experiment_uuid, self.logging_token])
        return '/'.join([self.root_topic, self.experiment_uuid, self.logging_token, self.device_name, response_cmnd_key, response_cmnd_value])

    def get_curr_timestamp(self):
        return (datetime.now(tz=pytz.timezone('US/Pacific')).strftime('%Y-%m-%d-T%H%M%S-')) 

    def _unpack_string(self, unpack_item, match_items):
        """
        Unpacks items from match_items based on a string or regex pattern.
        Raises an error if the pattern is invalid or if no matches are found.

        Args:
            unpack_item (str): A string or regex pattern to match against match_items.
            match_items (list): A list of items to match the unpack_item against.

        Returns:
            list: A sorted list of items from match_items that match the unpack_item.
        """
        if not isinstance(unpack_item, str): 
            raise ValueError(f"Item {unpack_item} should be a string. Forfeiting entire command list.")

        try:
            if unpack_item.startswith("^") and unpack_item.endswith("$"):
                regex = re.compile(unpack_item)
                filtered = [item for item in match_items if regex.match(item)]
            else:
                filtered = fnmatch.filter(match_items, unpack_item)
        except re.error:
            raise ValueError(f"Invalid regex {unpack_item}")

        if not filtered:
            raise ValueError(f"Item {unpack_item} not found in match_items. Forfeiting entire command list.")

        return sorted(filtered)

    def _enqueue_messages_to_schedule(self, topic, message, unpack_field, unpack_items):
        """
        Updates the message dictionary's specified field with each item from unpack_items,
        printing the updated message each time.

        Args:
            message (dict): The message to update.
            unpack_field (str): The field within the message to update.
            unpack_items (list): A list of new values to sequentially assign to unpack_field.
        """
        print("ENQUEUEING!")
        for unpack_item in unpack_items:
            new_message = message.copy()
            new_message[unpack_field] = unpack_item
            #self.add_to_scheduler(new_message, param, time_unit, at_time))
            schedule_message = {
                "COMMAND": "SCHEDULE-REQUEST",
                "TYPE": "ADD",
                "EVERY_X_SECONDS": "1",
                "AT": ":01",
                "FLAG": "ONCE",
                "FROM": self.device_name,
                "DO": json.dumps(new_message)
                }
            
            print(schedule_message)
            job_tag = self.generate_job_tag()
            self.scheduler.every(1).second.do(self.run_once, topic, new_message, job_tag).tag=(job_tag)

        self.update_state(self.state)
        return


    def unpack(self, topic, message, unpack_field, match_items, sort_all=False, enqueue_schedule=True):
        """
        Unpacks items based on patterns specified in a message field against a list of items.
        Filters and optionally sorts matched items from match_items based on patterns found in message[unpack_field].
        Uses _stuff_messages to display how each matched item updates the message.

        Args:
            message (dict): The dictionary containing the unpack_field.
            unpack_field (str): Key in message whose value is a string or list of strings or regex patterns.
            match_items (list): Items to match against the patterns.
            sort_all (bool): If True, sorts the entire result list; otherwise, maintains order of first appearance.

        Returns:
            boolean: True if anything has been unpacked into individual messages; False otherwise.

        Usage:
        try:
            unpackable, unpacked_list = self.unpack(message, "WELL_ID", self.wells, sort_all=True, enqueue_schedule=True)
            if not unpackable: #It's a single literal value, execute command now
        except ValueError as e:
            print("Error:", e)
        """
        unpack_items = message.get(unpack_field)

        #unpack items should not contain '?'
        if isinstance(unpack_items, str) and unpack_items.find("?") == -1 and unpack_items in match_items:
            return False, []

        if not isinstance(unpack_items, (str, list)):
            raise ValueError("unpack_items must be a string or a list of strings")

        filtered = []
        seen = set()

        if isinstance(unpack_items, str):
            unpack_items = [unpack_items]  # Treat single string as a list with one item

        for item in unpack_items:
            current_matches = self._unpack_string(item, match_items)
            for match in current_matches:
                if match not in seen:
                    seen.add(match)
                    filtered.append(match)

        if sort_all:
            filtered.sort()

        print(filtered)

        if enqueue_schedule: 
            self._enqueue_messages_to_schedule(topic, message, unpack_field, filtered)

        return True, filtered


    def set_mqtt_subscribe_topics(self, topics = None): # topics must be an array
        # unsubscribe from old topic
        for topic in self.mqtt_subscribe_topics:
            self.mb.unsubscribe_message(topic)
        # construct new topic
        if topics == None:
            base_topic = '/'.join([self.root_topic, self.experiment_uuid, self.logging_token])
            self.mqtt_subscribe_topics = [
                f"{base_topic}/{self.device_name}/+/REQUEST", # device topic
                f"{base_topic}/+/REQUEST" # general experiment topic
            ]
        else:
            self.mqtt_subscribe_topics = topics
        # subscribe to new topic
        for topic in self.mqtt_subscribe_topics:
            self.mb.subscribe_message(topic, callback=self.consume_mqtt_message)
        return

    def s3_basepath(self, UUID):
        base_path = 's3://braingeneers/'
        if UUID == "NONE": return base_path + 'integrated/'
        match = re.search(r'-[a-z]*-', UUID)
        if not match: return base_path + 'integrated/'
        type_mapping = {
            "-e-": "ephys/",
            "-f-": "fluidics/",
            "-i-": "imaging/",
        }
        return base_path + type_mapping.get(match.group(0), 'integrated/')

    def signal_handler(self, sig, frame):
        """ Gracefully shutdown the device on a SIGTERM or SIGINT signal. Cleanup or finalization code goes here. """
        self.shutdown()
        sys.exit(0)
        
    def find_job(self, tag):
        for job in self.scheduler.jobs:
            if hasattr(job, 'tag') and job.tag == tag: 
                yield job

    def run_once(self, topic, message, tag):
        for job in self.find_job(tag):
            print(f"Removing job {job.tag} from schedule")
            self.scheduler.cancel_job(job)
        self.consume_mqtt_message(topic, message)

    def consume_mqtt_message(self, topic, message):
        """
        Callback function for MQTT messages, delegates message handling to the appropriate function.

        :param topic: String containing MQTT topic on which the message was received. 
        :param message: Dictionary containing the MQTT message. The first key in the dictionary is used
                        to determine the message type and appropriate handler function.
        :return: None
        """
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t New unsorted message:\n{topic}\n{message}\n")

        try:
            cmnd_key = self.get_command_key_from_topic(topic)
            print(f"cmnd_key: {cmnd_key}")
            # Call the corresponding handler if it exists, otherwise handle the other cases
            if cmnd_key in self.handlers:
                self.handlers[cmnd_key](topic, message)
            else: 
                self.handle_transition_to_device_cases(topic, message)
        except Exception as e:
            print(f"Error while processing message: {e}")
        return

    def handle_ping(self, topic, message):
        self.update_state(self.state)
        self.mb.publish_message(topic= self.generate_response_topic("PING", "RESPONSE"),
                                message={"COMMAND":"PING-RESPONSE", 
                                        "FROM" : self.device_name })

    def handle_status(self, topic, message):
            message.update(self.device_state)
            self.mb.publish_message(topic=self.generate_response_topic("STATUS", "RESPONSE"), 
                                    message={"COMMAND":"STATUS-RESPONSE", 
                                            "FROM" : self.device_name} )

    def handle_schedule(self, topic, message):
        if not self.is_my_topic(topic) or message["FROM"] == self.device_name:
            return

        if self.state not in ["READY", "WAITING", "EXEC"]:
            self.publish_schedule_error(f"Device is {self.state}. No experiment to schedule")
            return

        # TODO: Right now device puts coommand blindly on the schedule, if command
        # has wrong syntax error will occur at a future time when the command is scheduled to run.
        # Solution: implement pre-parse/checking of the command to make sure it's good to run in the future, 
        # and return any errors immediately

        #Solution: Check key is in handlers (mqtt and device specific)

        if message["TYPE"] == 'ADD':
            self.handle_add_schedule(message)
        elif message["TYPE"] == 'CLEAR':
            self.handle_clear_schedule(message)
        elif message["TYPE"] == 'GET':
            self.handle_get_schedule()
        else:
            self.publish_schedule_error(f"Undefined schedule command, {message['SCHEDULE']}")

        self.update_state(self.state)  # update schedule on shadow, without state changes


    def handle_add_schedule(self, message):
        print("Schedule Job: ", message['DO'])
        sched_params = [
            ("EVERY_X_HOURS", "hours", "AT"),
            ("EVERY_X_DAYS", "days", "AT"),
            ("EVERY_X_MINUTES", "minutes", None),
            ("EVERY_X_SECONDS", "seconds", None)
        ]
        for param, time_unit, at_time in sched_params:
            if param in message:
                print(f"Got {param}")
                self.add_to_scheduler(message, param, time_unit, at_time)
                return
        self.publish_schedule_error("Must specify either EVERY_X_HOURS, EVERY_X_MINUTES, EVERY_X_DAYS or EVERY_X_SECONDS")
        
    def add_to_scheduler(self, message, param, time_unit, at_time):
        job_tag = self.generate_job_tag()
        schedule_time = int(message[param])
        response_topic = self.generate_response_topic(*message['DO']["COMMAND"].split("-")) #get the command from the message
        job_function = self.run_once if ('FLAGS' in message and message['FLAGS'] == 'ONCE') else self.consume_mqtt_message  #run once or regularly

        arguments = (job_function, response_topic, message['DO'])
        if job_function == self.run_once: arguments += (job_tag,)

        if at_time: #hours, days
            self.scheduler.every(schedule_time).__getattribute__(time_unit).at(message[at_time]).do(*arguments).tag=(job_tag)
        else: #minutes, seconds
            self.scheduler.every(schedule_time).__getattribute__(time_unit).do(*arguments).tag=(job_tag)

        self.mb.publish_message(topic=self.generate_response_topic("SCHEDULE", "COMPLETE"), message={"COMMAND": "SCHEDULE-COMPLETE", "FROM": self.device_name})

    def handle_clear_schedule(self, message):
        if "TAG" in message.keys() and re.match(r"^\d+$", message["TAG"]): #tag is numeric
            for job in self.find_job(message["TAG"]): self.scheduler.cancel_job(job)
        else:
            self.scheduler.clear()
        self.mb.publish_message(topic=self.generate_response_topic("SCHEDULE", "COMPLETE"), 
                                message={"COMMAND": "SCHEDULE-COMPLETE", "FROM": self.device_name})

    def handle_get_schedule(self):
        self.mb.publish_message(topic=self.generate_response_topic("SCHEDULE", "RESPONSE"), 
                                message={"COMMAND": "SCHEDULE-RESPONSE", "FROM": self.device_name, "SCHEDULE": str(self.scheduler.jobs)})

    def publish_schedule_error(self, error_message):
        self.mb.publish_message(topic=self.generate_response_topic("SCHEDULE", "ERROR"), 
                                message={"COMMAND": "SCHEDULE-ERROR", "ERROR": error_message, "FROM": self.device_name})
            
    def handle_stop(self, topic, message):
        if self.state == "EXEC":
            # handle quitting exec immediately
            print("Hit STOP while in EXEC State!")
            self.stop_event.set() #send signal to stop thread
            self.exec_thread.join() #wait for thread to finish
            self.stop_event.clear() #re-load the signal for next time

            self.update_state("READY")
            self.mb.publish_message(topic=self.generate_response_topic("STOP", "COMPLETE"),
                                    message={"COMMAND":"STOP-COMPLETE",
                                            "TIMESTAMP": datetime_serializer(datetime.now()),
                                            "FROM" : self.device_name })
        else:
            self.mb.publish_message(topic=self.generate_response_topic("STOP", "ERROR"),
                                    message={"COMMAND":"STOP-ERROR", 
                                            "ERROR": f"In state {self.state} and not executing anything to stop", 
                                            "FROM" : self.device_name })

    def handle_pause(self, topic, message):
        if (self.state in ["READY", "EXEC", "WAITING"]) and (message["FROM"] != self.device_name):
            requested_pause_until = datetime.now() + timedelta(seconds=int(message["SECONDS"]))

            #only update t_pause if greater than current pause value
            #unless its coming from the same sender, then update regardless
            if self.pause_until < requested_pause_until or self.who_paused is message["FROM"]:

                self.who_paused = message["FROM"]
                print("\n Pausing for seconds:", message["SECONDS"])
                self.pause_until = requested_pause_until

                #if self.exec_running: 
                if self.state == "EXEC":
                    self.update_state(self.state) #keep being in EXEC
                else: 
                    self.update_state("WAITING")

                self.mb.publish_message(topic=self.generate_response_topic("PAUSE", "ACK"),
                                        message={"COMMAND":"PAUSE-ACK", 
                                                "TIMESTAMP": datetime_serializer(datetime.now()),
                                                "FROM" : self.device_name })
            else:
                self.mb.publish_message(topic=self.generate_response_topic("PAUSE", "ERROR"),
                                        message={"COMMAND":"PAUSE-ERROR", 
                                                "ERROR": f"Already paused until {self.pause_until} by {self.who_paused}", 
                                                "FROM" : self.device_name })                  

        else:
            self.mb.publish_message(topic=self.generate_response_topic("PAUSE", "ERROR"),
                                    message={"COMMAND":"PAUSE-ERROR", 
                                            "ERROR": "No experiment to pause or this device is the PAUSE sender", 
                                            "FROM" : self.device_name })

    def handle_resume(self, topic, message):
        # Device is WAITING
        if self.state == "WAITING": #self.pause_until < datetime.now()
            if message["FROM"] == self.who_paused: 
                print("\n Resuming from pause")
                self.pause_until = self.naive_datetime
                self.who_paused = None
                
                #if self.exec_running: 
                #TODO: WHY AM I CHECKING THIS??
                if self.state == "EXEC":
                    self.update_state(self.state) #keep being in EXEC
                else: 
                    self.update_state("READY")
                
                self.mb.publish_message(topic=self.generate_response_topic("RESUME", "ACK"),
                                        message={"COMMAND":"RESUME-ACK", 
                                                "FROM" : self.device_name })
            else:
                self.mb.publish_message(topic=self.generate_response_topic("RESUME", "ERROR"),
                                        message={"COMMAND":"RESUME-ERROR", 
                                                "ERROR": f"Can't resume me because you're not {self.who_paused}",  
                                                "FROM" : self.device_name })
        else: #Not WAITING
            self.mb.publish_message(topic=self.generate_response_topic("RESUME", "ERROR"),
                                    message={"COMMAND":"RESUME-ERROR", 
                                            "ERROR": "Not WAITING, nothing to resume", 
                                            "FROM" : self.device_name })      

    def handle_twiddle(self, topic, message):
        # A comand function for testing that does nothing for a specified amount of time
        print("Handling twiddle!")
        if "SECONDS" in message.keys():
            twiddle_for = message["SECONDS"]
            twiddle_until = datetime.now() + timedelta(seconds=int(twiddle_for))
            while not self.stop_event.is_set() and datetime.now() < twiddle_until:
                time.sleep(1)
        else:
            self.mb.publish_message(topic=self.generate_response_topic("TWIDDLE", "ERROR"),
                                    message={"COMMAND":"TWIDDLE-ERROR", 
                                            "ERROR": "Invalid message format", 
                                            "FROM" : self.device_name })

    def handle_device_specific(self, topic, message):
        cmnd_key = self.get_command_key_from_topic(topic)

        # Call the corresponding handler if it exists
        # The child class (i.e., a specific device) must specify its functions in device_specific_handlers 
        # Also, use while not self.stop_event.is_set()  for stop conction
        
        if cmnd_key in self.device_specific_handlers:
            try:
                #TODO: WHy are ACK and COMPLETE being published at the same time (both after handler finishes)
            
                # self.exec_running = True
                self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "ACK"),
                                        message={"COMMAND": f"{cmnd_key}-ACK", "TIMESTAMP": datetime_serializer(datetime.now()), "FROM" : self.device_name})
                
                self.exec_job_name = str(message)

                if "SECONDS" in message.keys(): 
                    self.exec_job_until = datetime.now() + timedelta(seconds=int(message["SECONDS"]))
                if "MINUTES" in message.keys(): 
                    self.exec_job_until = datetime.now() + timedelta(minutes=int(message["MINUTES"]))

                self.update_state(self.state)

                self.device_specific_handlers[cmnd_key](topic, message)

                self.exec_job_name = None
                self.exec_job_until = self.naive_datetime

                if self.stop_event.is_set():
                    self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "STOPPED"),
                                            message={"COMMAND": f"{cmnd_key}-STOPPED", "TIMESTAMP": datetime_serializer(datetime.now()), "FROM" : self.device_name}) 
                else:
                    self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "COMPLETE"),
                                            message={"COMMAND": f"{cmnd_key}-COMPLETE", "TIMESTAMP": datetime_serializer(datetime.now()), "FROM" : self.device_name}) 
                
                if datetime.now() < self.pause_until: self.update_state("WAITING")
                else: self.update_state("READY")

                # self.exec_running = False 
            except Exception as e:
                print(f"ERROR executing, command not supported: {e}")
                self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "ERROR"), 
                                        message= {"COMMAND": f"{cmnd_key}-ERROR",
                                                "ERROR": "EXEC job crashed, see device's printed output for error",  
                                                "FROM" : self.device_name})
                self.exec_job_name = None
                self.exec_job_until = self.naive_datetime
                self.update_state("READY")

        else:
            self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "ERROR"), 
                                    message= {"COMMAND": f"{cmnd_key}-ERROR",
                                                "ERROR": "Command not supported",  
                                                "FROM" : self.device_name})
            self.update_state("READY")
        return


    def handle_transition_to_device_cases(self, topic, message):   #spin this off in a thread?
        
        cmnd_key = self.get_command_key_from_topic(topic)
        cmnd_type = self.get_command_value_from_topic(topic)

        if self.is_my_topic(topic):
            
            if self.state == "READY":
                self.update_state("EXEC")
                self.exec_thread = threading.Thread(target=self.handle_device_specific, args=(topic, message))
                self.exec_thread.start()
                return

            elif self.state in ["WAITING", "EXEC"]:
                print(f"Error: command received while in {self.state}")

                job_tag = self.generate_job_tag()
                self.scheduler.every(1).second.do(self.run_once, topic, message, job_tag).tag=(job_tag)
                self.update_state(self.state)
                self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "ERROR"),
                                        message={"COMMAND": f"{cmnd_key}-ERROR", 
                                                "ERROR": f"Command recieved while device is {self.state}, scheduling after {self.state} complete",
                                                "UNTIL": json.dumps(self.pause_until, default=datetime_serializer) })
                return

            else:
                print(f"Error: command received while in {self.state}")
                self.mb.publish_message(topic=self.generate_response_topic(cmnd_key, "ERROR"),
                                        message={"COMMAND": f"{cmnd_key}-ERROR",  
                                                "ERROR": f"Current state {self.state} is an invalid state for accepting commands",
                                                "UNTIL": json.dumps(self.pause_until, default=datetime_serializer)}) 
                return   

    def handle_start(self, topic, message):
        if self.is_my_topic(topic):

            if self.state == "PRIMED":
                pattern = r"\d{4}-\d{2}-\d{2}-[efi]+-[a-zA-Z0-9]+" 
                regex = re.compile(pattern)

                if regex.match(message['UUID']): #check UUID payload regex -- else invalid UUID
                    
                    self.experiment_uuid = message["UUID"] # uuid payload
                    print("\nNew experiment UUID:", self.experiment_uuid)

                    if message.get("TEAMMATES") != None:
                        self.teammates = message["TEAMMATES"] # set teammates
                        print("Teammates:", self.teammates)


                    #self.set_mqtt_publish_topic()
                    self.set_mqtt_subscribe_topics()
                    print("Listening to messages on " + str(self.mqtt_subscribe_topics))

                    self.update_state("READY")
                    self.mb.publish_message(topic=self.generate_response_topic("START", "COMPLETE"),
                                            message={"COMMAND":"START-COMPLETE", 
                                                    "FROM" : self.device_name })
                else:
                    self.mb.publish_message(topic=self.generate_response_topic("START", "ERROR"),
                                        message={"COMMAND":"START-ERROR", 
                                                "ERROR": f"Invalid UUID {message['UUID']}, please use format YYYY-MM-DD-efi-experiment-name", 
                                                "FROM" : self.device_name })


            elif self.state == "IDLE":
                self.mb.publish_message(topic=self.generate_response_topic("START", "ERROR"),
                                        message={"COMMAND":"START-ERROR", 
                                                "ERROR": "Not PRIMED", 
                                                "FROM" : self.device_name })
            
            else:
                if self.experiment_uuid != "NONE":
                    self.mb.publish_message(topic=self.generate_response_topic("START", "ERROR"),
                                        message={"COMMAND":"START-ERROR", 
                                                    "ERROR": f"Already in experiment {self.experiment_uuid}", 
                                                    "FROM" : self.device_name })

    def handle_end(self, topic, message):
        if self.state == "PRIMED":
            print("handle end - primed case")
            self.update_state("IDLE")
            self.mb.publish_message(topic=self.generate_response_topic("END", "COMPLETE"),
                                message={"COMMAND":"END-COMPLETE", 
                                        "FROM" : self.device_name })
            return

        if self.state == "EXEC":
            print("handle end - exec case")

            # handle quitting exec immediately
            print("Hit END while in EXEC State!")
            self.stop_event.set() #send signal to stop thread
            self.exec_thread.join() #wait for thread to finish
            self.stop_event.clear() #re-load the signal for next time

        print("\nEnded experiment UUID:", self.experiment_uuid)

        self.experiment_uuid = "NONE"
        self.teammates = []
        self.pause_until = self.naive_datetime #None/in the past
        self.who_paused = None

        #self.set_mqtt_publish_topic()
        self.set_mqtt_subscribe_topics()

        self.scheduler.clear()
        self.update_state("IDLE")

        self.mb.publish_message(topic=self.generate_response_topic("END", "COMPLETE"),
                                message={"COMMAND":"END-COMPLETE", 
                                        "FROM" : self.device_name })
        return

    def start_mqtt(self):  
        # Blocking function (spins in while loop)
        # do you want to unsubscribe from old UUIDs?
        #Start listening to messages on default topic
        for subscribe_topic in self.mqtt_subscribe_topics:
            self.mb.subscribe_message(topic=subscribe_topic, callback=self.consume_mqtt_message)
        
        print("Now start listening on MQTT...")
        print("Listening to messages on: " + str(self.mqtt_subscribe_topics))

        # Note: Calbacks handle state transitions based on MQTT messages in the background
        # This function works in parallel handle all other state trasition cases and keep checking the scheduler
        # for any commands to execute
        while True:
            if self.state == "IDLE" and self.is_primed(): 
                self.update_state("PRIMED")

            if self.state == "READY" and datetime.now() < self.pause_until: 
                self.update_state("WAITING")

            if self.state == "WAITING" and datetime.now() >= self.pause_until: 
                self.pause_until = self.naive_datetime #None/in the past
                self.who_paused = None
                self.update_state("READY")

            #do scheduled and coordinate if waiting
            #Make sure schedule respects the pause
            if self.state == "READY":
                pending_jobs = [job for job in self.scheduler.get_jobs() if job.next_run <= datetime.now()]

                #print(pending_jobs)

                if pending_jobs:
                    print("Getting pending jobs:", self.scheduler.get_jobs())
                    print("Pending jobs: ", pending_jobs)
                    
                    try:
                        self.scheduler.run_pending()                            
                    except Exception as e:
                        print(f"Error while running scheduled task: {e}")
                    #self.update_state("READY")

            time.sleep(1)

    def shutdown(self):
        # TODO: finish shutdown funcunction and use it somewhere to terminate device
        self.update_state("SHUTDOWN")
        self.mb.shutdown() #shutdown message broker
        self.cache.close() #close the cache 

        #wait to shut down s3 uplod/download thread
        print("Waiting for S3 thread to finish...")
        print("Received shutdown signal...")
        self.thread_pool.shutdown(wait=True)
        print("ThreadPoolExecutor shut down.")

        # TODO: Remove yourself from the database when getting shut down
        #shut down shadow

    # Move these functions to a more general utility class?
    def post_to_slack(self, text, image_path = None):
        print("Sending message to Slack...")
        message = {"message": text}
        if image_path:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
                message.update({
                    "filename": image_path,
                    "image": encoded_image
                })
        # Publish the image to a Slack channel 
        self.mb.publish_message(self.to_slack_topic, message)
        return

#------------------Helper Functions------------------#

    @retry(initial_delay=5, max_delay=60, max_tries=60)
    def _transfer_file(self, source, destination, is_upload=True):
        operation = 'writing' if is_upload else 'reading'
        try:
            print(f"Attempting {operation} file from {source} to {destination}")
            with smart_open.open(source, 'rb') as source_file:
                content = source_file.read()
            with smart_open.open(destination, 'wb') as dest_file:
                dest_file.write(content)
        except Exception as e:
            print(f"Error {operation} file: {e}")
            raise

    def _direct_upload_file(self, s3_location, local_file_path, delete_local=False, announce_completion=None):
        base_filename = os.path.basename(local_file_path)
        s3_path = os.path.join(s3_location, base_filename)

        #self._transfer_file(s3_path, local_file_path)
        self._transfer_file(local_file_path, s3_path, is_upload=True)

        # this will not execute if there's a failure in the upload or download process above
        if delete_local:
            self.cache.delete(base_filename)
            os.remove(local_file_path)

        if announce_completion:
            topic, message = announce_completion
            self.mb.publish_message(topic=topic, message=message, confirm_receipt=True)

        return s3_path

    def _direct_download_file(self, s3_path, local_file_path):
        self._transfer_file(s3_path, local_file_path, is_upload=False)
        return local_file_path

    def _s3_job_worker(self):
        print("Entered the _s3_job_worker!")

        with self.queue_lock:
            print("Entered inside a lock!")
            print("QUEUE", list(self.cache_queue))
            try:
                job_type, args, task_id = self.cache_queue.popleft() # Remove the processed job
            except Exception as e:
                print(f"Exception when popping from cache_queue: {e}")
                return

            #job_type, args, task_id = self.cache_queue.popleft() # Remove the processed job
            future = self.futures_dict[task_id]
            
            print(f"Processing job {task_id} of type {job_type} with args {args}")

        if future is None:
            print(f"Error: No future found for task ID {task_id}")
            return

        try:
            if job_type == "upload":
                result = self._direct_upload_file(*args)
                future.set_result(result)
            elif job_type == "download":
                result = self._direct_download_file(*args)
                future.set_result(result)
        except Exception as e:
            print(f"{job_type} failed due to: {e}")
            future.set_exception(e)
            self.failed_jobs_queue.append(args)
        finally:
            with self.queue_lock:
                del self.futures_dict[task_id]
   

    def _enqueue_file_task(self, args, wait_for_completion=False):
        print("Entered the _enqueue_file_task!")

        future = Future()
        task_id = str(uuid.uuid4())

        with self.queue_lock:
            self.cache_queue.append((*args, task_id))
            self.futures_dict[task_id] = future

        # Submitting the task to the thread pool
        self.thread_pool.submit(self._s3_job_worker) #(args, task_id)

        if wait_for_completion:
            return future.result()  # This will block until the future has a result
        else:
            return future  # Return the future immediately, letting the caller decide what to do with it


    def upload_file(self, s3_location, local_file_path, delete_local=False, announce_completion = None, wait_for_completion = False):
        """ Uploads a file to S3 and publishes a message to a topic upon completion """
        args = "upload", (s3_location, local_file_path, delete_local, announce_completion)
        return self._enqueue_file_task(args, wait_for_completion)

    def download_file(self, s3_path, local_path = BASEPATH_CACHE, wait_for_completion = True):
        """ Downloads a file from S3 """
        if local_path == ".":  local_file_path = os.path.join(os.getcwd(), s3_path.split('/')[-1])
        else: local_file_path = os.path.join(local_path, s3_path.split('/')[-1])
       
        args = "download", (s3_path, local_file_path)
        return self._enqueue_file_task(args, wait_for_completion)


    def append_to_s3_file(self, s3_path, data_to_append, local_path = BASEPATH_CACHE):
        #Best used for appending to small files, for larger files consider multipart upload in boto3 instead (?)
        
        # Read the existing content of the file
        try:
            with smart_open.open(s3_path, 'r') as f:
                existing_content = f.read()
        except Exception as e:
            print(f"Error fetching file: {e}")
            existing_content = "" # blank file
            print(f"Creating a new file: {s3_path}")
        
        # Append the new data
        updated_content = existing_content + data_to_append
        
        # Write the updated content back to the file on S3
        with smart_open.open(s3_path, 'wb') as f:
            f.write(updated_content.encode('utf-8'))
        return

    def read_s3_file(self, s3_path):
      # sets the default PRP endpoint
        with smart_open.open(s3_path, 'r') as f:
            txt = f.read()
            return txt
