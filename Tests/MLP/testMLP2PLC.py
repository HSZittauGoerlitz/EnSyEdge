from Scripts import MLP2PLC


addName = "3Layer"

MLP = MLP2PLC.MLPloader()
MLP.loadOrangeModel("./Tests/MLP/MLPvalidation",
                    "WineTestMLP_" + addName + ".pkcls")

writeMLP = MLP2PLC.MLPwriter(MLP)
print("\n--------\n|HEADER|\n--------\n")
print(writeMLP.Code["Header"])
print("\n------\n|BODY|\n------\n")
print(writeMLP.Code["Body"])

# Test Json Ex-/Import
MLP.exportToJson()
MLP.loadJsonModel("./Tests/MLP/MLPvalidation",
                  "WineTestMLP_" + addName + ".json")
writeMLP_sjon = MLP2PLC.MLPwriter(MLP)
print("\n--------------\n|JSON-Version|\n--------------")
print("--------\n|HEADER|\n--------\n")
print(writeMLP_sjon.Code["Header"])
print("\n------\n|BODY|\n------\n")
print(writeMLP_sjon.Code["Body"])
