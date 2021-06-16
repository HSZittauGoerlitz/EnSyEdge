import pickle


# Import Orange3 Model
with open("./Tests/MLP/WineTestMLP.pkcls", 'rb') as f:
    model = pickle.load(f)

# get skl model
sklModel = model.skl_model

# Print parameters
print("Number of inputs: %i" % (sklModel.n_features_in_))
print("Number of outputs: %i" % (sklModel.n_outputs_))
print("\nHidden Layers\n_____________")
print("\nShape")
print(sklModel.hidden_layer_sizes)
print("Number of HL: %i" % (len(sklModel.hidden_layer_sizes)))
print("\nWeights")
print(sklModel.coefs_[0:3])
print("Shape first")
print(sklModel.coefs_[0].shape)
print("Shape second")
print(sklModel.coefs_[1].shape)
print("Shape third")
print(sklModel.coefs_[2].shape)
print("\nBiases")
print(sklModel.intercepts_[0:3])
print("\Output Layer\n____________")
print("\nWeights")
print(sklModel.coefs_[3])
print("\nBiases")
print(sklModel.intercepts_[3])
print("\nActivation\n__________")
print("Activation: %s" % (sklModel.activation))
print("Out: %s" % (sklModel.out_activation_))

print("\nTest Neuron 1.1 Weights")
print(sklModel.coefs_[0][:, 0])
