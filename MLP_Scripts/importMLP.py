import MLP2PLC
import os

# Open a select file dialog which only accepts *.project files
selected_file = system.ui.open_file_dialog("MLP Selection",
                                           None, None,
                                           "MLP model as JSON|*.json")

# Remember to check if the user canceled the dialog
if selected_file is not None:
    # get model into memory
    file_split = selected_file.split(os.path.sep)
    MLPfile = file_split[-1]
    MLPpath = os.path.sep.join(file_split[:-1])
    MLP = MLP2PLC.MLPloader()
    MLP.loadJsonModel(MLPpath, MLPfile)
    MLPwrite = MLP2PLC.MLPwriter(MLP)
    # create CoDeSys FB
    # get CoDeSys application
    app = projects.primary.active_application
    MLP_FB = app.create_pou(name=MLPwrite.Name)
    MLP_FB.textual_declaration.replace(MLPwrite.Code['Header'])
    MLP_FB.textual_implementation.replace(MLPwrite.Code['Body'])
