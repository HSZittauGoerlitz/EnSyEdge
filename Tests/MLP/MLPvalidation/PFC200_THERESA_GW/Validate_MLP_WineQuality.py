from getpass import getpass
import logging
import numpy as np
from opcua import Client, ua
from opcua.ua.uaerrors import UaStatusCodeError
import os
from passlib.hash import pbkdf2_sha256
import pickle
import time


PW_HASH = ("$pbkdf2-sha256$29000$0vq/F.Kcs3bO2Xtv7f3f2w$"
           "5EqLg3Y4cmLdJabcKkXWShUL2xfJ3qP1v1OkdfLg1uI"
           )

MLP_IN_NAMES = ["in_1", "in_2", "in_3", "in_4", "in_5", "in_6", "in_7",
                "in_8", "in_9", "in_10", "in_11", "out_1"]

############
# Set Path #
############
# Set Path information, that this script can be run without workspace config
os.chdir("D:/Filr/Netzwerkordner/ZI - IPM/IPMSYS2/KWL/14 THERESA_next/"
         "04 BeSiX/05 Code_Allgemein/GatewayLib/Tests/MLP/MLPvalidation"
         )


####################
# Helper Functions #
####################
# hide password
def getGWpassword():
    count = 4
    print("Please enter OPC UA password of THERESA GateWay")
    pw = getpass(prompt="")
    while not pbkdf2_sha256.verify(pw, PW_HASH):
        if count <= 0:
            exit()
        else:
            print("Wrong password, "
                  "number of tries remaining: %i" % count)
            pw = getpass(prompt="")
        count -= 1

    return pw


# common server browsing functions
def find_PrgNode(Node, prgName="PLC_PRG"):
    # Catch bad node configuration
    try:
        children = Node.get_children()
    except UaStatusCodeError:
        return None

    # BFS Search
    for child in children:
        # Catch bad node configuration
        try:
            if child.get_display_name().Text == prgName:
                return child
        except UaStatusCodeError:
            continue

    # nothing found -> look at each child
    for child in children:
        search_result = find_PrgNode(child, prgName)
        if search_result is not None:
            return search_result

    return None


def get_DeviceSetNode(client):
    logger.info("Loading DeviceSetNode")
    # get starting node
    objects = client.get_objects_node()

    for node in objects.get_children():
        if node.get_display_name().Text == "DeviceSet":
            return node

    logger.error("DeviseSet Node not found")


def get_MLPnode(prgNode, MLPname):
    for child in prgNode.get_children():
        if child.get_display_name().Text == MLPname:
            return child

    logger.error("MLP Node '{}' not found".format(MLPname))


def get_rcNode(prgNode, nodeName):
    for child in prgNode.get_children():
        if child.get_display_name().Text == nodeName:
            return child

    logger.error("Recalculation Node '{}' not found".format(nodeName))


# test function for MLP
def testMLP(client, inNodes, MLPnodes, rcNode, models, nTestCases=100):
    logger.info("Start test")
    # MLP data was normalised to [0, 1] range
    logger.info("Generating input values")

    nModels = len(models)
    test_input = np.random.random((nTestCases, 11, nModels))

    logger.info("Getting reference Predictions")
    refPred = []
    for i, model in enumerate(models):
        refPred.append(model.predict(test_input[:, :, i]))

    refPred = np.array(refPred)

    plcPred = np.zeros_like(refPred)

    logger.info("Getting Predictions of MLP variants on PLC")
    for i in range(nTestCases):
        if i % 100 == 0:
            logger.info("Test Progress {}"
                        .format(float(i)/float(nTestCases) * 100.))

        plcPred[:, i] = call_PLC_MLP(test_input[i, :, :], client,
                                     inNodes, MLPnodes, rcNode)

    minOut = plcPred.min()
    maxOut = plcPred.max()
    e = np.abs(refPred - plcPred)

    logger.info("Test of all MLP finished")
    logger.info("min. MLP output value: {}".format(minOut))
    logger.info("max. MLP output value: {}".format(maxOut))
    logger.info("max. error: {}".format(e.max()))
    logger.info("max. rel. error: {} %"
                .format((e.max() / maxOut * 100.)))
    logger.info("mean. rel. error: {} %"
                .format((e.mean() / maxOut * 100.)))


def call_PLC_MLP(inputs, client, inNodes, MLPnodes, rcNode):
    # set new inputs
    writeInputs(inputs, client, inNodes, rcNode)

    # wait twice cycle time, just to be sure
    time.sleep(0.01)

    return getOutputs(client, MLPnodes)


def writeInputs(inputs, client, inNodes, rcNode):
    # seperate write of MLPinputs and recalculation flag,
    # be sure that calculation is only started when all inputs are up to date
    writeRealArray(client, inNodes, inputs)
    rcNode.set_attribute(ua.AttributeIds.Value,
                         ua.DataValue(ua.Variant(True,
                                                 ua.VariantType.Boolean)))


def writeRealArray(client, inNodes, inputs):
    # adaption of set_values method of opc ua lib
    nodeids = [node.nodeid for node in inNodes]
    dvs = [ua.DataValue(ua.Variant(val, ua.VariantType.Float))
           for val in inputs.T.tolist()]

    results = client.uaclient.set_attributes(nodeids, dvs,
                                             ua.AttributeIds.Value)

    for result in results:
        result.check()


def getOutputs(client, MLPnodes):
    results = client.get_values(MLPnodes)

    return [result.out_1 for result in results]


##########################
# Config and Preparation #
##########################
# Import Orange3 Model
file_loc = ""
cert = file_loc + "../pyOPCUA/cert.der"
key = file_loc + "../pyOPCUA/key.pem"

models = []

for i in range(1, 4):
    with open(file_loc + "WineTestMLP_{}Layer.pkcls".format(i), 'rb') as f:
        models.append(pickle.load(f).skl_model)

# configure client
logger = logging.Logger('Log', logging.INFO)
logger.addHandler(logging.StreamHandler())

client = Client(url='opc.tcp://141.46.119.188:4840/')
client.set_user("admin")
client.set_password(getGWpassword())
client.set_security_string(",".join(["Basic256Sha256", "SignAndEncrypt",
                                     cert, key])
                           )
client.application_uri = "urn:WineTestComm:client"


############
# Run Test #
############
# start communication
if __name__ == "__main__":
    try:
        logger.info("Connecting to PLC")
        client.connect()
        # init method for custom types
        client.load_type_definitions()
        objects = client.get_objects_node()
        # work in DeviceSet Branch, since it has less side nodes
        devSetNode = get_DeviceSetNode(client)
        logger.info("Loading Program Node")
        prgNode = find_PrgNode(devSetNode, "MLPinference")
        InNodes = []
        MLPnodes = []
        logger.info("Loading Nodes")
        for i in range(1, 4):
            InNodes.append(get_MLPnode(prgNode, "inMLP{}".format(i)))
            MLPnodes.append(get_MLPnode(prgNode, "MLP{}".format(i)))
        rcNode = get_rcNode(prgNode, "recalculate")
        # Run Test
        testMLP(client, InNodes, MLPnodes, rcNode, models, 10000)

    finally:
        client.disconnect()
        logger.info("PLC disconnected successfully")
