
import os
import json
import webbrowser
import importlib.resources
import configparser
import datetime
import requests
import argparse


def authenticate_and_get_token():
    """
    Directs users to a URL to authenticate and get a JWT token.
    Once the token has been obtained manually it will refresh automatically every month.
    By default, the token is valid for 4 months from issuance.
    Returns token data as a dict containing `access_token` and `expires_at` keys.
    """
    PACKAGE_NAME = "braingeneers.iot"

    url = 'https://service-accounts.braingeneers.gi.ucsc.edu/generate_token'
    print(f'Please visit the following URL to generate your JWT token: {url}')
    webbrowser.open(url)

    token_json = input('Please paste the JSON token issued by the page and press Enter:\n')
    try:
        token_data = json.loads(token_json)
    except json.JSONDecodeError:
        raise ValueError('Invalid JSON. Please make sure you have copied the token correctly.')

    config_dir = os.path.join(importlib.resources.files(PACKAGE_NAME), 'service_account')
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, 'config.json')

    with open(config_file, 'w') as f:
        json.dump(token_data, f)

    print('Token has been saved successfully.')
    return token_data


def update_config_file(file_path, section, key, new_value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        section_found = False
        for line in lines:
            if line.strip() == f'[{section}]':
                section_found = True
            if section_found and line.strip().startswith(key):
                line = f'{key} = {new_value}\n'
                section_found = False  # Reset the flag
            file.write(line)


def picroscope_authenticate_and_update_token(credentials_file):
    """
    Authentication and update service-account token for legacy picroscope environment. This updates the AWS credentials file
    with the JWT token and updates it if it has <3 months before expiration. This function can be run as a cron job.
    """
    # Check if the JWT token exists and if it exists in the credentials file if it's expired.
    # The credentials file section is [strapi] with `api_key` containing the jwt token, and `api_key_expires` containing
    # the expiration date in ISO format.
    config_file_path = os.path.expanduser(credentials_file)

    config = configparser.ConfigParser()
    with open(config_file_path, 'r') as f:
        config.read_string(f.read())

    assert 'strapi' in config, \
        'Your AWS credentials file is missing a section [strapi], you may have the wrong version of the credentials file.'

    token_exists = 'api_key' in config['strapi']
    expire_exists = 'api_key_expires' in config['strapi']

    if expire_exists:
        expiration_str = config['strapi']['api_key_expires']
        expiration_str = expiration_str.split(' ')[0] + ' ' + expiration_str.split(' ')[1]  # Remove timezone
        expiration_date = datetime.datetime.fromisoformat(expiration_str)
        days_remaining = (expiration_date - datetime.datetime.now()).days
        print('Days remaining for token:', days_remaining)
    else:
        days_remaining = -1

    # check if api_key_expires exists, if not, it's expired, else check if it has <90 days remaining on it
    manual_refresh = not token_exists \
                     or not expire_exists \
                     or (datetime.datetime.fromisoformat(config['strapi']['api_key_expires']) - datetime.datetime.now()).days < 0
    auto_refresh = (token_exists and expire_exists) \
                   and (datetime.datetime.fromisoformat(config['strapi']['api_key_expires']) - datetime.datetime.now()).days < 90

    if manual_refresh or auto_refresh:
        token_data = authenticate_and_get_token() if manual_refresh else requests.get(url).json()
        update_config_file(config_file_path, 'strapi', 'api_key', token_data['access_token'])
        update_config_file(config_file_path, 'strapi', 'api_key_expires', token_data['expires_at'])
        print(f'JWT token has been updated in {config_file_path}')
    else:
        print('JWT token is still valid, no action taken.')


def parse_args():
    """
    Two commands are available:

        # Authenticate and obtain a JWT service account token for braingeneerspy
        python -m braingeneers.iot.authenticate

        # Authenticate and obtain a JWT service account token for picroscope specific environment
        python -m braingeneers.iot.authenticate picroscope
    """
    parser = argparse.ArgumentParser(description='JWT Service Account Token Management')
    parser.add_argument('config', nargs='?', choices=['picroscope'], help='Picroscope specific JWT token configuration.')
    parser.add_argument('--credentials', default='~/.aws/credentials', help='Path to the AWS credentials file, only used for picroscope authentication.')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config == 'picroscope':
        credentials_file = args.credentials
        picroscope_authenticate_and_update_token(credentials_file)
    else:
        authenticate_and_get_token()


if __name__ == '__main__':
    main()
