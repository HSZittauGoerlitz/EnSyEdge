import json
import pickle
import struct


class MLPloader:
    """ Class for managing MLP data """

    def exportToJson(self):
        """ Export MLP as json file, so that it can be read
            without other dependencies
            -> Json Model can be read by CoDeSys Script Engine directly
        """
        model_data = {}
        model_data['nInput'] = self.nInput
        model_data['nHidden'] = self.nHidden
        model_data['Hidden'] = self.Hidden
        model_data['Output'] = self.Output

        with open(self.path + self.name + '.json', 'w') as jFile:
            json.dump(model_data, jFile, indent=2)

    def loadJsonModel(self, path, filename):
        """ Load MLP model exported by MLPloader (for use in CoDeSys)

        Args:
            path (str): Path to file
            filename (str): File where the model is saved (json-File)
        """
        self._checkFilename(filename, 'json')
        self._checkPath(path)

        with open(self.path + filename, 'rb') as jFile:
            model_data = json.load(jFile)

        self.nInput = model_data['nInput']
        self.nHidden = model_data['nHidden']
        self.Hidden = model_data['Hidden']
        self.Output = model_data['Output']

    def loadOrangeModel(self, path, filename):
        """ Load MLP model created with orange3 and initialise MLPcreator object

        Args:
            path (str): Path to file
            filename (str): File where the model was dumped (pkcls-File)
        """
        self._checkFilename(filename, 'pkcls')
        self._checkPath(path)

        # Import Orange3 Model
        with open(self.path + filename, 'rb') as f:
            model = pickle.load(f)

        # get skl model
        sklModel = model.skl_model

        self._getDataFromSKLearnModel(sklModel)

    def loadSKLearnModel(self, SKLmodel, name):
        """Load MLP model created with sklearn and initialise MLPcreator object

        Args:
            SKLmodel (sklmodel obj): MLP model created with sklearn
            name (str): Name of the model (is also used as CoDeSys FB-Name)
        """
        self.name = name
        self._getDataFromSKLearnModel(SKLmodel)

    def _checkFilename(self, filename, ending):
        """ Check / prepare given filename for use in MLPloader

        Args:
            filename (str): Name of file for model to load
            ending (str): Type of file where model is saved

        Returns:
            str: Prepared file name
        """
        if filename.split(".")[-1] != ending:
            filename += "." + ending

        self._filenameToName(filename)

        return filename

    def _checkPath(self, path):
        """ Check / prepare given path for use in MLPloader

        Args:
            path (str): Path to file
        """
        # replace backslash
        path = path.replace('\\', '/')
        # replace double slash
        path = path.replace('//', '/')
        # check for correct path ending
        if path[-1] != '/':
            path += '/'

        self.path = path

    def _filenameToName(self, filename):
        """ Prepare filename and set it as name

        Args:
            filename (str): (Path to File and) Name of File
        """
        # remove possible file ending
        name = "".join(filename.split(".")[:-1])
        # remove folder informations
        name = name.replace("\\", "/").split("/")[-1]

        self.name = name

    def _getDataFromSKLearnModel(self, model):
        """ Initialise MLPcreator object with data from sklearn model

        Args:
            model (obj): MLP Model saved with sklearn
        """
        self.nInput = model.n_features_in_
        self.nHidden = len(model.hidden_layer_sizes)
        # create hidden layers
        self.Hidden = self.nHidden * [None]
        for idxHL in range(self.nHidden):
            neurons = model.hidden_layer_sizes[idxHL] * [None]
            for idxN in range(model.hidden_layer_sizes[idxHL]):
                neurons[idxN] = self._createNeuron(
                                  model.coefs_[idxHL][:, idxN],
                                  model.intercepts_[idxHL][idxN])
            self.Hidden[idxHL] = self._createHiddenLayer(
                neurons, model.hidden_layer_sizes[idxHL], model.activation)
        # create output layer
        neurons = model.n_outputs_ * [None]
        for idxN in range(model.n_outputs_):
            neurons[idxN] = self._createNeuron(
                                model.coefs_[idxHL+1][:, idxN],
                                model.intercepts_[idxHL+1][idxN])
        self.Output = self._createHiddenLayer(neurons, model.n_outputs_,
                                              model.out_activation_)

    def _createNeuron(self, weights, bias):
        """ Create and return dict with relevant neuron data
                - get rid of numpy dependency -> convert to list
                - save parameters with high accuracy -> convert to hex strings

        Args:
            weights (np.array): Array of real numbers with all input weights
                                (sequence of weights hast to be equal
                                 to input sequence)
            bias (real): Bias value of neuron

        Returns:
            dict: Neuron data
        """
        return {"weights": [self._float_to_hex(w) for w in weights],
                "bias": self._float_to_hex(bias)}

    def _createHiddenLayer(self, neurons, nNeurons, activation,
                           normalisation="None", threshold=0.0):
        """ Create and return dict with relevant hidden layer data

        Args:
            neurons (list): List of neurons belonging to hidden layer
            nNeurons (int): Number of neurons in layer
            activation (str): Activation function used for neurons
                              of this layer
            normalisation (str, optional): Optional normalisation function
                                           applied on all neuron outputs
                                           of this layer
                                           Defaults to "None"
            threshold (float, optional): Threshold value for corresponding
                                         activation function
                                         Defaults to 0.0.
        """
        return {"neurons": neurons,
                "nNeurons": nNeurons,
                "activation": activation.lower(),
                "normalisation": normalisation,
                "threshold": threshold}

    def _float_to_hex(self, f):
        return hex(struct.unpack('<I', struct.pack('<f', f))[0]).replace(
                 "0x", "16#")


class MLPwriter:
    """ Create MLP FB code for PLC from MLPloader data """
    # Mapping of activation functions
    activation_map = {"threshold": 0,
                      "logistic": 1,
                      "tanh": 2,
                      "identity": 3,
                      "exponential": 4,
                      "reciprocal": 5,
                      "square": 6,
                      "Gauss": 7,
                      "sine": 8,
                      "cosine": 9,
                      "Elliott": 10,
                      "arctan": 11,
                      "rectifier": 12,
                      "relu": 12  # common term for rectifier
                      }

    # Mapping of normalisation functions
    normalisation_map = {"None": 0,
                         "softmax": 1,
                         "simplemax": 2
                         }

    def __init__(self, MLP, name=None):
        """ Create MLPwriter Object

        Args:
            MLP (MLPloader): MLP model provided by MLPloader class
            name (str, optional): [description]. Defaults to None.
        """
        if name is None:
            self.Name = MLP.name
        else:
            self.Name = name

        self.MLP = MLP
        self.Code = {"Header": "",
                     "Body": ""}

        self._writeHeader()
        self._writeBody()

    def _writeBody(self):
        """ Create code for Body and save in self.Code dict
        """
        # read FB inputs and set to MLP input array
        code = "(* get inputs *)\n"
        for idxIn in range(1, self.MLP.nInput+1):
            code += "input[{0}] := in_{0};\n".format(idxIn)
        # MLP inference
        code += "\n(* MLP inference *)\n"
        code += "EEL.CallANN(ANN:=MLP);\n"
        # provide MLP outputs at FB outputs
        code += "\n(* set outputs *)\n"
        for idxOut in range(1, self.MLP.Output["nNeurons"]+1):
            code += "out_{0} := out_neurons[{0}].output;\n".format(idxOut)

        self.Code["Body"] = code

    def _writeHeader(self):
        """ Create code for header and save in self.Code dict
        """
        code = "FUNCTION_BLOCK {}\n\n".format(self.Name)
        # All inputs as separate vars -> hold all arrays in FB
        code += "VAR_INPUT\n"
        for idxIn in range(1, self.MLP.nInput+1):
            code += "    in_{}: REAL;\n".format(idxIn)
        code += "END_VAR\n\n"

        # Create variables for dynamic data
        code += "VAR\n"
        # input array to store data
        code += "(* input array *)\n"
        code += "    input: ARRAY[1..{}] OF REAL;\n".format(self.MLP.nInput)

        # build up ANN layers
        # Hidden Layer(s)
        code += "\n(* hidden layer *)\n"
        # ensure to keep IEEE float representation by hex values
        # save layer names for array creation
        hlArray = self.MLP.nHidden * [None]
        for idxHL, hidden in enumerate(self.MLP.Hidden, 1):
            code += "(* Layer {} *)\n".format(idxHL)
            code += "(* Neurons *)\n"
            # save neuron names for array creation
            nArray = hidden["nNeurons"] * [None]
            # create neuron data
            for idxN, neuron in enumerate(hidden["neurons"], 1):
                code += ("    neuron_{0}_{1}: EEL.Neuron := (weights:="
                         "ADR(weights_{0}_{1}), bias:=(dw:={2}), "
                         "output:=0.0);\n"
                         ).format(idxHL, idxN, neuron["bias"])
                nArray[idxN-1] = "neuron_{}_{}".format(idxHL, idxN)
            # collect neurons in array
            code += ("    hl_{0}_neurons: ARRAY[1..{1}] OF "
                     "EEL.Neuron := ["
                     ).format(idxHL, hidden["nNeurons"])
            code += ", ".join(nArray) + "];\n"
            code += "(* Layer *)\n"
            # create layer data
            code += ("    hidden_layer_{0}: EEL.Layer := ("
                     "neurons:=ADR(hl_{0}_neurons), nNeurons:={1},"
                     "activation:={2}, threshold:={3});\n\n"
                     ).format(idxHL, hidden["nNeurons"],
                              self.activation_map[hidden["activation"]],
                              hidden["threshold"]
                              )
            hlArray[idxHL-1] = "hidden_layer_{}".format(idxHL)

        # collect hidden layer in array
        code += "\n    hlArray: ARRAY[1..{}] of EEL.Layer := [".format(
          self.MLP.nHidden)
        code += ", ".join(hlArray) + "];\n\n"

        # Output Layer
        code += "(* output layer *)\n"
        # ensure to keep IEEE float representation by hex values
        # save neuron names for array creation
        nArray = self.MLP.Output["nNeurons"] * [None]
        # create neuron data
        for idxN, neuron in enumerate(self.MLP.Output["neurons"], 1):
            code += ("    neuron_out_{0}: EEL.Neuron := (weights:="
                     "ADR(weights_out_{0}), bias:=(dw:={1}), output:=0.0);\n"
                     ).format(idxN, neuron["bias"])
            nArray[idxN-1] = "neuron_out_{}".format(idxN)
        # collect neurons in array
        code += ("    out_neurons: ARRAY[1..{}] OF "
                 "EEL.Neuron := ["
                 ).format(self.MLP.Output["nNeurons"])
        code += ", ".join(nArray) + "];\n"
        # create out layer data
        code += ("    out_layer: EEL.Layer := ("
                 "neurons:=ADR(out_neurons), nNeurons:={},"
                 "activation:={}, threshold:={});\n"
                 ).format(self.MLP.Output["nNeurons"],
                          self.activation_map[self.MLP
                                              .Output["activation"]],
                          self.MLP.Output["threshold"]
                          )

        # ANN
        code += "\n(* MLP representation *)\n"
        code += ("    MLP: EEL.ANN := (input:=ADR(input), "
                 "hidden:=ADR(hlArray), output:=ADR(out_layer), "
                 "nInput:={}, nHidden:={});\n"
                 ).format(self.MLP.nInput, self.MLP.nHidden)

        code += "END_VAR\n\n"

        # provide easy access to output value(s)
        code += "VAR_OUTPUT\n"
        for idxOut in range(1, self.MLP.Output["nNeurons"]+1):
            code += "    out_{}: REAL;\n".format(idxOut)
        code += "END_VAR\n\n"

        # Store MLP data as const arrays
        # ensure to keep IEEE float representation by hex values
        code += "VAR CONSTANT\n"
        for idxHL, hidden in enumerate(self.MLP.Hidden, 1):
            for idxN, neuron in enumerate(hidden["neurons"], 1):
                code += ("    weights_{}_{}: ARRAY[1..{}] OF "
                         "EEL.dwREAL := ").format(idxHL, idxN,
                                                  len(neuron["weights"]))
                weights = neuron["weights"]
                code += "[(dw:="
                code += '), (dw:='.join(weights)
                code += ")];\n"

        for idxN, neuron in enumerate(self.MLP.Output["neurons"], 1):
            code += ("    weights_out_{}: ARRAY[1..{}] OF "
                     "EEL.dwREAL :=").format(
                        idxN, len(neuron["weights"]))
            weights = neuron["weights"]
            code += "[(dw:="
            code += '), (dw:='.join(weights)
            code += ")];\n"

        code += "END_VAR\n\n"

        self.Code["Header"] = code
