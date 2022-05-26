"""
The proposed architecture in chapter 3.
"""

import numpy as np

input_num = 8 # number of input neurons
output_num = 1 # number of output neurons
H = 20 # size of the cells (denoted by H in the thesis as well)
neuron_num = input_num + output_num + 3*H

graph = []

#
# Main graph
#

# inputs
for i in range(input_num):
    neighbors = []
    
    # ---> r
    for j in range(H):
        neighbors.append((j+input_num+1,1,True))
        
    # ---> h
    for j in range(H):
        neighbors.append((j+input_num+1+2*H,1,True))
    
    graph.append(neighbors)

# r
for i in range(H):
    neighbors = []
    # ---> c
    neighbors.append((i+1+input_num+H,1,False,1.0))
    
    graph.append(neighbors)

# c
for i in range(H):
    neighbors = []
    # ---> h
    for j in range(H):
        neighbors.append((j+input_num+1+2*H,1,True))
    
    graph.append(neighbors)
    
# h
for i in range(H):
    neighbors = []
        
    # ---> output
    for j in range(output_num):
        neighbors.append((j+1+input_num+3*H,1,True))
    
    # ---> r
    for j in range(H):
        neighbors.append((j+1+input_num,1,True))
                
    # ---> c
    neighbors.append((i+1+input_num+H,2,True))
    # ---> c
    #for j in range(H):
    #    neighbors.append((j+1+input_num,2,True))
    
    graph.append(neighbors)

# outputs
for i in range(output_num):
    neighbors = []
    
    graph.append(neighbors)
    
#  
#
# Activations
#

activations = []

# inputs
for i in range(input_num):
    activations_neuron = [('0',True)]
    activations.append(activations_neuron)
    
# r
for i in range(H):
    activations_neuron = [('2',True)]
    activations.append(activations_neuron)

# c
for i in range(H):
    activations_neuron = [('0',False,0.0),('1',True)]
    activations.append(activations_neuron)
   
# h    
for i in range(H):
    activations_neuron = [('1',True)]
    activations.append(activations_neuron)

# outputs
for i in range(output_num):
    activations_neuron = [('0',True)]
    activations.append(activations_neuron)

print(f"act_len: {len(activations)}")
print(f"graph: {len(graph)}")
#print(f"graph_trans: {len(trans_graph)}")
print("\n")
#
# Creating the files to save
#

to_file_graph = []
to_file_logic = []
to_file_fixwb = []
line_ind = 0
    

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
    
#
# Saving to file and output to screen
#
print(f"Neuron num: {neuron_num}")

trainable_parameters = 0
for line in to_file_logic:
    trainable_parameters += line.count('1')

print(f"Hidden num: {H}")
print(f"Trainable parameters: {trainable_parameters}")

graph_datas = f"graph_igru_I{input_num}_O{output_num}_H{H}.dat"
f = open(graph_datas,"w")
for line in to_file_graph:
    f.write(line+"\n")
f.close()

logic_datas = f"logic_igru_I{input_num}_O{output_num}_H{H}.dat"
f = open(logic_datas,"w")
for line in to_file_logic:
    f.write(line+"\n")
f.close()

fixwb_datas = f"fixwb_igru_I{input_num}_O{output_num}_H{H}.dat"
f = open(fixwb_datas,"w")
for line in to_file_fixwb:
    f.write(line+"\n")
f.close()

print(f"*_igru_I{input_num}_O{output_num}_H{H}.dat")
