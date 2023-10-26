# For writing commands that will be executed after the container is created

# Uninstalls the braingeneerspy package (pre-installed in the research Docker image) from the environment
python3 -m pip uninstall braingeneerspy

# Installs a Python package located in the current directory in editable mode and includes all optional extras specified in the [all] section of braingeneers.
python3 -m pip install -e ".[all]"
