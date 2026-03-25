from __future__ import annotations

import argparse
import configparser
import datetime
import json
import time
import webbrowser
from pathlib import Path

import requests

from braingeneers.mcp.service_profiles import get_default_service_profile
from braingeneers.utils.auth_endpoints import (
    MCP_HELPER_CLIENT_ID,
    MCP_REQUESTED_SCOPE,
    OAUTH2_BROKER_DEVICE_AUTHORIZATION_URL,
    OAUTH2_BROKER_ISSUER_URL,
    OAUTH2_BROKER_JWKS_URL,
    OAUTH2_BROKER_TOKEN_URL,
    SERVICE_ACCOUNT_TOKEN_URL,
)
from braingeneers.utils.oidc_tokens import normalize_oidc_token_response
from braingeneers.utils.auth_storage import (
    save_token,
    service_account_token_path,
    user_token_path,
)


def _open_and_parse_json(url: str) -> dict:
    print(f"Please visit the following URL to generate your token(s): {url}")
    webbrowser.open(url)
    token_json = input(
        "Please paste the JSON token payload issued by the page and press Enter:\n"
    )
    try:
        return json.loads(token_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid JSON. Please make sure you have copied the token correctly."
        ) from exc


def _request_device_authorization(*, scope: str) -> dict:
    response = requests.post(
        OAUTH2_BROKER_DEVICE_AUTHORIZATION_URL,
        data={
            "client_id": MCP_HELPER_CLIENT_ID,
            "scope": scope,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Failed to start Keycloak device authorization flow. "
            f"Response: {response.text}"
        )
    return response.json()


def _poll_for_user_token(device_authorization: dict, *, audience: str, scope: str) -> dict:
    device_code = str(device_authorization["device_code"])
    interval_seconds = int(device_authorization.get("interval", 5))
    expires_in = int(device_authorization["expires_in"])
    deadline = time.monotonic() + expires_in

    while time.monotonic() < deadline:
        time.sleep(interval_seconds)
        response = requests.post(
            OAUTH2_BROKER_TOKEN_URL,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": MCP_HELPER_CLIENT_ID,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return normalize_oidc_token_response(
                response.json(),
                issuer_url=OAUTH2_BROKER_ISSUER_URL,
                jwks_url=OAUTH2_BROKER_JWKS_URL,
                token_url=OAUTH2_BROKER_TOKEN_URL,
                client_id=MCP_HELPER_CLIENT_ID,
                audience=audience,
                requested_scope=scope,
            )

        body = response.json()
        error = body.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval_seconds += 5
            continue
        if error == "access_denied":
            raise PermissionError("Keycloak device authorization was denied.")
        if error == "expired_token":
            raise PermissionError(
                "Keycloak device authorization expired before it completed."
            )
        raise RuntimeError(
            "Keycloak device authorization failed. "
            f"Response: {response.text}"
        )

    raise PermissionError(
        "Keycloak device authorization timed out before it completed."
    )


def _open_keycloak_device_flow(*, audience: str) -> dict:
    scope = MCP_REQUESTED_SCOPE
    device_authorization = _request_device_authorization(scope=scope)
    verification_uri_complete = device_authorization.get("verification_uri_complete")
    verification_uri = device_authorization.get("verification_uri")
    user_code = device_authorization.get("user_code")

    if verification_uri_complete:
        print(
            "Please complete the Braingeneers MCP login in your browser at:\n"
            f"{verification_uri_complete}"
        )
        webbrowser.open(str(verification_uri_complete))
    else:
        print(
            "Please complete the Braingeneers MCP login in your browser."
        )
        if verification_uri:
            print(f"Verification URL: {verification_uri}")
        if user_code:
            print(f"User code: {user_code}")
        if verification_uri:
            webbrowser.open(str(verification_uri))

    return _poll_for_user_token(
        device_authorization,
        audience=audience,
        scope=scope,
    )


def authenticate_and_get_tokens() -> dict:
    """
    Bootstraps both the traditional broad service-account token and the newer
    interactive user token used by the MCP helper. The broad token continues to
    come from the existing browser-authenticated service-accounts page, while
    the user token comes directly from the Keycloak broker via device flow.
    """
    service_profile = get_default_service_profile()
    print("Step 1/2: bootstrap the broad Braingeneers service-account token.")
    service_account_token = _open_and_parse_json(SERVICE_ACCOUNT_TOKEN_URL)

    print("Step 2/2: bootstrap the interactive Braingeneers MCP user token.")
    user_token = _open_keycloak_device_flow(audience=service_profile.audience)
    user_token["selected_service"] = service_profile.name

    save_token(service_account_token_path(), service_account_token)
    save_token(user_token_path(), user_token)
    print("Service-account token and interactive user token have been saved successfully.")
    return {
        "service_account_token": service_account_token,
        "user_token": user_token,
    }


def authenticate_and_get_service_account_token() -> dict:
    """
    Legacy service-account-only bootstrap path. This remains useful for
    picroscope and other callers that only need the broad web-service token.
    """
    token_data = _open_and_parse_json(SERVICE_ACCOUNT_TOKEN_URL)
    save_token(service_account_token_path(), token_data)
    print("Service-account token has been saved successfully.")
    return token_data


def update_config_file(file_path, section, key, new_value):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(file_path, "w", encoding="utf-8") as file:
        section_found = False
        for line in lines:
            if line.strip() == f"[{section}]":
                section_found = True
            if section_found and line.strip().startswith(key):
                line = f"{key} = {new_value}\n"
                section_found = False
            file.write(line)


def picroscope_authenticate_and_update_token(credentials_file):
    """
    Authentication and update service-account token for legacy picroscope environment.
    This updates the AWS credentials file with the JWT token and refreshes it if it has
    <3 months before expiration. This function can be run as a cron job.
    """
    config_file_path = credentials_file.expanduser()

    config = configparser.ConfigParser()
    with config_file_path.open("r", encoding="utf-8") as f:
        config.read_string(f.read())

    assert "strapi" in config, (
        "Your AWS credentials file is missing a section [strapi], "
        "you may have the wrong version of the credentials file."
    )

    token_exists = "api_key" in config["strapi"]
    expire_exists = "api_key_expires" in config["strapi"]

    if expire_exists:
        expiration_str = config["strapi"]["api_key_expires"]
        expiration_str = (
            expiration_str.split(" ")[0] + " " + expiration_str.split(" ")[1]
        )
        expiration_date = datetime.datetime.fromisoformat(expiration_str).replace(
            tzinfo=datetime.timezone.utc
        )
        days_remaining = (expiration_date - datetime.datetime.now(datetime.timezone.utc)).days
        print("Days remaining for token:", days_remaining)
    else:
        days_remaining = -1

    manual_refresh = (
        not token_exists
        or not expire_exists
        or (
            datetime.datetime.fromisoformat(config["strapi"]["api_key_expires"]).replace(
                tzinfo=datetime.timezone.utc
            )
            - datetime.datetime.now(datetime.timezone.utc)
        ).days
        < 0
    )
    auto_refresh = (
        token_exists
        and expire_exists
        and (
            datetime.datetime.fromisoformat(config["strapi"]["api_key_expires"]).replace(
                tzinfo=datetime.timezone.utc
            )
            - datetime.datetime.now(datetime.timezone.utc)
        ).days
        < 90
    )

    if manual_refresh or auto_refresh:
        token_data = (
            authenticate_and_get_service_account_token()
            if manual_refresh
            else requests.get(
                SERVICE_ACCOUNT_TOKEN_URL,
                headers={
                    "Authorization": f"Bearer {config['strapi']['api_key']}",
                },
                timeout=30,
            ).json()
        )
        update_config_file(
            config_file_path,
            "strapi",
            "api_key",
            token_data["access_token"],
        )
        update_config_file(
            config_file_path,
            "strapi",
            "api_key_expires",
            token_data["expires_at"],
        )
        print(f"JWT token has been updated in {config_file_path}")
    else:
        print("JWT token is still valid, no action taken.")


def parse_args():
    """
    Two commands are available:

        # Bootstrap both the broad service-account token and the interactive user
        # token used by the braingeneers.mcp stdio helper
        python -m braingeneers.iot.authenticate

        # Authenticate and obtain only the broad service-account token for the
        # legacy picroscope environment
        python -m braingeneers.iot.authenticate picroscope
    """

    parser = argparse.ArgumentParser(description="Braingeneers token bootstrap")
    parser.add_argument(
        "config",
        nargs="?",
        choices=["picroscope"],
        help="Picroscope specific service-account token configuration.",
    )
    parser.add_argument(
        "--credentials",
        default="~/.aws/credentials",
        type=Path,
        help="Path to the AWS credentials file, only used for picroscope authentication.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config == "picroscope":
        picroscope_authenticate_and_update_token(args.credentials)
    else:
        authenticate_and_get_tokens()


if __name__ == "__main__":
    main()
