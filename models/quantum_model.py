import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

def QuantumCircut(weights, n_qubits, n_layers, z):
    for qubit in range(n_qubits):
        qml.RY(z[qubit], wires=[qubit])

    for layer in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(*weights[layer][j], wires=j)

        for j in range(len(z) - 1):
            qml.CNOT(wires=[j, j + 1])
                
    return qml.probs(wires=list(range(n_qubits)))

def RotCNOTGate(n_qubits, weights):
    for j in range(n_qubits):
        qml.Rot(*weights[0][j], wires=j)
    
    for j in range(n_qubits - 1):
        qml.CNOT(wires=[j, j +1])   
        
    for j in range(n_qubits):
        qml.Rot(*weights[1][j], wires=j)
    
def QuantumExponentialCircut(weights, n_qubits, n_layers, z):
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RY(2**layer*z[qubit], wires=[qubit])

        RotCNOTGate(n_qubits, weights[layer])
    
    for j in range(n_qubits - 1):
        qml.CNOT(wires=[j, j + 1])

    return qml.probs(wires=list(range(n_qubits)))

def PostProcess(probs, n_qubits, n_ancillas):
    probs_given_ancilla_0 = probs[: 2 ** (n_qubits - n_ancillas)]
    mass = torch.sum(probs_given_ancilla_0)
    if not torch.isfinite(mass) or mass <= 0:
        raise ValueError("ancilla-zero probability mass must be finite and positive")
    post_measurement_probs = probs_given_ancilla_0 / mass
    peak = torch.max(post_measurement_probs)
    if not torch.isfinite(peak) or peak <= 0:
        raise ValueError("post-selected probabilities must have a positive maximum")
    post_processed_patch = ((post_measurement_probs / peak) - 0.5) * 2
    return post_processed_patch

class QuantumGenerator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout, device):
        super(QuantumGenerator, self).__init__()

        self.vertexes       = vertexes
        self.edges          = edges
        self.nodes          = nodes
        self.n_qubits       = z_dim
        self.n_ancillas     = 2
        self.n_layers       = 3
        self.quantum_params = torch.nn.Parameter(torch.rand(self.n_layers, self.n_qubits, 3), requires_grad=True)
        self.q_device       = qml.device("default.qubit", wires=self.n_qubits)
        self.qlayer         = qml.QNode(QuantumCircut, self.q_device, interface='torch')
        self.device         = device
        
        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        
        layers       = layers[1:] #self.qlayer is going outside main pipeline
        self.layers  = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout    = nn.Dropout(p=dropout)

    def forward(self, x, mode='Default'):
        #print(x.shape)
        array    = []
        in_size  = x.shape[0]
        out_size = 2 ** (self.n_qubits - self.n_ancillas)
        if mode == 'Default':
            for i in range(in_size):
                array.append(PostProcess(self.qlayer(self.quantum_params, self.n_qubits, self.n_layers, x[i]), self.n_qubits,  self.n_ancillas).float())
        elif mode == 'IBM':
            for i in range(in_size):
                array.append(PostProcess(x[i], self.n_qubits,  self.n_ancillas).float())
        else:
            raise ValueError(f"unsupported quantum generator mode: {mode!r}")
        
        x = torch.cat(array).to(self.device).view(in_size, out_size)
        output = self.layers(x)
        #print(output.shape)
        edges_logits = self.edges_layer(output)\
                       .view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits

class QuantumExponentialGenerator(nn.Module):
    """Quantum Exponential Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout, device):
        super(QuantumExponentialGenerator, self).__init__()

        self.vertexes       = vertexes
        self.edges          = edges
        self.nodes          = nodes
        self.n_qubits       = z_dim
        self.n_ancillas     = 2
        self.n_layers       = 2
        self.quantum_params = torch.nn.Parameter(torch.rand(self.n_layers, 2, self.n_qubits, 3), requires_grad=True)
        self.q_device       = qml.device("default.qubit", wires=self.n_qubits)
        self.qlayer         = qml.QNode(QuantumExponentialCircut, self.q_device, interface='torch')
        self.device         = device
        
        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        
        layers       = layers[1:] #self.qlayer is going outside main pipeline
        self.layers  = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout    = nn.Dropout(p=dropout)

    def forward(self, x):
        #print(x.shape)
        array    = []
        in_size  = x.shape[0]
        out_size = 2 ** (self.n_qubits - self.n_ancillas)
        for i in range(in_size):
            array.append(PostProcess(self.qlayer(self.quantum_params, self.n_qubits, self.n_layers, x[i]), self.n_qubits,  self.n_ancillas).float())
            
        x = torch.cat(array).to(self.device).view(in_size, out_size)
        output = self.layers(x)
        #print(output.shape)
        edges_logits = self.edges_layer(output)\
                       .view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits