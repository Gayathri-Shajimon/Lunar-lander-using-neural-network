import numpy as np
import math


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self._lambda = 0.8 
        
    def activation_function(self, input_data, inp_weights):
        v_results = np.dot(input_data, np.transpose(inp_weights))
        hidden_layer_values = [(lambda v: 1 / (1 + math.exp(-(self._lambda) * v)))(v) for v in v_results]
        return hidden_layer_values
    
    def normalisation(self, minimumx, maximumx, initial_input):
        norm = []
        for i in range(2):
            norm.append(abs((minimumx - initial_input[i])/(maximumx - minimumx)))
        return norm 
            
    def denormalisation(self, vel_min_x, vel_max_x, output_hidden_layer_values):
        denorm = []
        for i in range(2):
            denorm.append(((output_hidden_layer_values[i] * (vel_max_x - vel_min_x)) + vel_min_x))
        return denorm    
        
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        minimumx = -297.001526
        maximumx = 219.661910
        minimumy = 65.144014
        maximumy = 641.871080
        vel_min_x = -3.763255
        vel_max_x = 5.694243
        vel_min_y = -4.949881
        vel_max_y = 4.969426
        input_list_norm = []
        initial_input = []
        _input = []
        output_denormlized = []
        print("input_row", input_row)
        
        #Split the data and appending to the list as float
        input_list = list(input_row.split(","))
        for i in input_list:
            initial_input.append(float(i))
            
        #Normalising input data
        _input = self.normalisation(minimumx, maximumx, initial_input)
        input_weights = [[43.426083134511735, 58.25390206623468], [31.58419916820029, 85.60551531816098], [0.05181178476703328, 0.0719787890912831], [150.2369799376772, 59.46559625048118], [208.17246018730776, 58.88306178587619], [82.15717246581637, 69.9440333265969], [44.83079259306777, 27.201105167224394]]
        output_weights = [[0.061225855104314865, 0.12948792570663156, 0.03258232327650565, 0.09936517071722867, 0.11310558291783818, 0.1128361016249679, 0.014862731858851043], [0.21946836173171885, 0.16620958754463727, 0.06059668186829589, 0.024244249146698996, 0.024003141816401362, 0.1416806663216751, 0.05518983077341342]]
       
        #calculating activation function for input layer
        hidden_layer_values = self.activation_function(_input,input_weights)
        
        #calculating activation function for output layer
        output_hidden_layer_values = self.activation_function(hidden_layer_values,output_weights)
        
        #Denormalising output
        output_denormlized = self.denormalisation(vel_min_x, vel_max_x, output_hidden_layer_values)
        print("output_denormlized",output_denormlized)
        return output_denormlized[0], output_denormlized[1]

