"""
The implicit architecture in chapter 2.
"""

import numpy as np

graph = []
layers = [8,60,30,1] # number of neurons per layer
input_num = layers[0]
output_num = layers[-1]

# x --- 1:input_num

activations = []
# x
for i in range(input_num):
    activations_neuron = [('0',False,0.0)]
    activations.append(activations_neuron)

# hiddens
for layer in layers[1:-1]:
    for neuron in range(layer):
        activations_neuron = [('2',True)]
        activations.append(activations_neuron)
            
# output
for i in range(output_num):
    activations_neuron = [('0',True)]
    activations.append(activations_neuron)

neuron_num = len(activations)
#for activations_neuron in activations:
#    print(activations_neuron)
#print(f"neuron_num: {neuron_num}\n")

graph = []

#
# Main graph
#

# inputs and hiddens
ind_to = 0; ind_from = 0
for j in range (len(layers)-1):
    
    ind_to=ind_from+layers[j]
    for k in range(layers[j]):
        neighbors = []
        for l in range(layers[j+1]):
            neighbors.append((ind_to+l+1,1,True))
        if (j>0):
            for l in range(layers[j]):
                #if (l!=k):
                neighbors.append((ind_from+l+1,1,True))
        graph.append(neighbors)
    ind_from=ind_from+layers[j]
    
    
# output
for i in range(output_num):
    neighbors = []
    graph.append(neighbors)

#for neighbors in graph:
#    print(neighbors)
#for activations_neuron in activations:
#    print(activations)

#print("\n")
#print(len(graph))


to_file_graph = []
to_file_logic = []
to_file_fixwb = []
line_ind = 0

for line_ind in range(len(graph)):
    
    neighbors = graph[line_ind]
    line_graph = str(len(neighbors))+ " ### "
    line_fixwb = ""
    line_logic = ""
    
    # neighbors
    for neighbor in neighbors:
        line_graph += f"{neighbor[0]} {neighbor[1]}; "
        line_logic += f"{int(neighbor[2])} "
        if (neighbor[2]==False):
            line_fixwb += f"{neighbor[3]} "
            
    # activations
    line_graph += "### "
    line_logic += "### "
    line_fixwb += "### "
    activations_neuron = activations[line_ind]
    
    line_graph += str(len(activations_neuron)) + " ### "
    for activation in activations_neuron:
        line_graph += f"{activation[0]} "
        line_logic += f"{int(activation[1])} "
        if (activation[1]==False):
            line_fixwb += f"{activation[2]} "
    
    to_file_graph.append(line_graph)
    to_file_logic.append(line_logic)
    to_file_fixwb.append(line_fixwb)


#for line_graph in to_file_graph:
#    print(line_graph)
#print("\n")
    
#for line_logic in to_file_logic:
#    print(line_logic)
#print("\n")
    
#for line_fixwb in to_file_fixwb:
#    print(line_fixwb)
#print("\n")
    
print(f"Neuron num: {neuron_num}")

trainable_parameters = 0
for line in to_file_logic:
    trainable_parameters += line.count('1')

L = len(layers)
H = sum(layers[1:-1])

print(f"Layer num: {L}")
print(f"Trainable parameters: {trainable_parameters}")

graph_datas = f"graph_idense_I{input_num}_O{output_num}_L{L}_H{H}.dat"
f = open(graph_datas,"w")
for line in to_file_graph:
    f.write(line+"\n")
f.close()

logic_datas = f"logic_idense_I{input_num}_O{output_num}_L{L}_H{H}.dat"
f = open(logic_datas,"w")
for line in to_file_logic:
    f.write(line+"\n")
f.close()

fixwb_datas = f"fixwb_idense_I{input_num}_O{output_num}_L{L}_H{H}.dat"
f = open(fixwb_datas,"w")
for line in to_file_fixwb:
    f.write(line+"\n")
f.close()

print(f"Id: *idense_I{input_num}_O{output_num}_L{L}_H{H}.dat")

 
