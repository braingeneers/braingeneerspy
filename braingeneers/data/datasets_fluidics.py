def get_fluidics_data(uuid):
    full_path = '/public/groups/braingeneers/Fluidics/derived/' + uuid + '/parameters.txt'
    with open(full_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data
