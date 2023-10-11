# For writing commands that will be executed after the container is created

# The given command installs a Python package located in the current directory in editable mode and includes all optional extras specified in the [all] section of braingeneers.
python3 -m pip install -e ".[all]"
