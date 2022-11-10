# How to use MLP functionality

1. Train a MLP in a tool which is supported
    1. orange3
    2. sklearn
2. Import trained MLP into MLPloader object of MLP2PLC module (located in Scripts folder of the library)
    1. An orange3 model can be loaded as pkclm file via loadOrangeModel function
    2. A sklearn model can be loaded directly from training script via loadSKLearnModel function
3. Export MLP to JSON file
4. Open your CoDeSys Gateway Project
5. Call importMLP.py (also located in Scripts folder) Script with CoDeSys Scriptengine
    1. A file dialog is opened by the script
    2. Point to the JSON file of your MLP model
    3. A FB of your model is created automaticely
6. Use the FB in your Project

# Example JSON export

```python
from Scripts import MLP2PLC

MLP = MLP2PLC.MLPloader()

# orange3
MLP.loadOrangeModel("location/of/your/model", "yourModel.pkcls")
# sklearn
MLP.loadSKLearnModel(yourSKLearnModelObj, "optionalModelName")

MLP.exportToJson()
```
