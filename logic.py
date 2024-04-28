import base64
import json
import numpy as np
import operator
import pandas as pd

from qiskit import *
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
from io import BytesIO
from typing import Tuple

from utils import gen_predefined_payoffs, Protocol, generate_unitary_gate

QiskitRuntimeService.save_account(channel="ibm_quantum", token="e80dbd26cacf8a4d94a4f02674f6c4a13d355c29f95830845a84744b9303cde25fd3bd7ed47a3daea7e52aa6fcca544b3ae1424356655aee19b2fb512b75c834", set_as_default=True, overwrite=True)
service = QiskitRuntimeService(channel="ibm_quantum")


class PayoffTable:
    def __init__(self, n_players, n_choices, payoff=None):
        self.n_players = n_players
        self.n_choices = n_choices
        self.payoff = payoff

    def set_payoffs(self, choices: str, payoff: Tuple):
        self.payoff[choices] = payoff

    def get_payoffs(self, choices: str):
        return self.payoff[choices]

    def get_payoff_table(self):
        return self.payoff


class QuantumGame:
    def __init__(self, player_gates, protocol: Protocol = Protocol.EWL, backend: str = 'qasm_simulator'):
        self.protocol = protocol
        self.backend = backend
        self.player_gates = player_gates
        self.num_players = self._get_num_players()
        self.J, self.Jdg = self._make_J_operators()
        self.circ = self._make_circuit(player_gates)

    def _get_num_players(self):
        counter = 0
        for gates in self.player_gates:
            if len(gates) > 0:
                counter += 1
        return counter

    def _make_J_operators(self):
        I = np.identity(1 << self.num_players)
        X = np.array([[0, 1], [1, 0]])
        tensorX = X

        for i in range(self.num_players - 1):
            tensorX = np.kron(tensorX, X)

        J = Operator(1 / np.sqrt(2) * (I + 1j * tensorX))
        Jdg = Operator(1 / np.sqrt(2) * (I - 1j * tensorX))

        return J, Jdg

    def _make_decomposed_J_operators(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.num_players + 1)
        circ.cx(0, self.num_players)
        circ.h(0)
        circ.x(self.num_players)
        for i in range(1, self.num_players):
            circ.ccx(0, self.num_players, i)
        circ.x(self.num_players)
        circ.x(0)
        for i in range(1, self.num_players):
            circ.ccx(0, self.num_players, i)
        circ.x(0)
        circ.s(0)
        return circ

    def _make_circuit(self, player_gates):
        if self.num_players == 2 or str(self.backend) == 'qasm_simulator':
            circ = QuantumCircuit(self.num_players, self.num_players)
            circ.append(self.J, range(self.num_players))
            circ.barrier()
    
            for i in range(self.num_players):
                circ = self._add_player_gates(circ, i, player_gates[i])
            circ.barrier()
    
            if self.protocol == Protocol.EWL:
                circ.append(self.Jdg, range(self.num_players))
                circ.barrier()
            circ.measure(range(self.num_players), range(self.num_players))
            return circ
    
        else:
            circ = QuantumCircuit(self.num_players + 1, self.num_players + 1)
            decomposed_J = self._make_decomposed_J_operators()
            for gate in decomposed_J:
                circ.append(gate, decomposed_J.qubits)
            circ.barrier()
    
            for i in range(self.num_players):
                circ = self._add_player_gates(circ, i, player_gates[i])
            circ.barrier()
    
            if self.protocol == Protocol.EWL:
                decomposed_J_inv = decomposed_J.inverse()
                for gate in decomposed_J_inv:
                    circ.append(gate, decomposed_J_inv.qubits)
                circ.barrier()
            circ.measure(range(self.num_players + 1), range(self.num_players + 1))
            return circ


    def _add_player_gates(self, circ, player_num, gates):
        for i in range(len(gates)):
            circ.append(gates[i], [player_num])
        return circ

    def draw_circuit(self):
        return self.circ.draw(output='mpl')


class Game:
    def __init__(self, game_name, protocol, num_players, payoff_table=None, group='open', backend='simulator'):
        self._game_name = game_name
        self._num_players = num_players
        self._n_players, self._n_choices, self._payoff_table = self._generate_payoff_table(
            self._game_name, self._num_players, payoff_table)
        self._protocol = Protocol[protocol]
        self._quantum_game = None
        self._final_results = None
        self._backend = self._set_backend(group, backend)

    def set_protocol(self, protocol, backend, group='open'):
        self._protocol = Protocol[protocol]
        self._backend = self._set_backend(group, backend)

    def _set_backend(self, group, backend):
        if self._protocol == Protocol.Classical:
            return "Classical"
        if backend == "simulator":
            return service.backend("ibmq_qasm_simulator")
        else:
            return service.least_busy(operational=True, simulator=False)

    def _generate_payoff_table(self, game_name, num_players, payoff_input):
        payoff_table = gen_predefined_payoffs(game_name, num_players, payoff_input)
        n_players = num_players
        n_choices = int(len(payoff_table) ** (1 / n_players))
        payoff_table = PayoffTable(n_players, n_choices, payoff_table)
        return n_players, n_choices, payoff_table

    def display_payoffs(self):
        print('Game: ' + self._game_name)
        print('Payoffs: ')
        choices = list(self._payoff_table.payoff.keys())
        payoffs = list(self._payoff_table.payoff.values())
        payoff_table = pd.DataFrame({'outcome': choices, 'payoffs': payoffs})
        payoff_table = payoff_table.sort_values(by=['outcome'])
        return payoff_table

    def format_choices(self, player_choices):
        formatted_player_choices = []
        for choice in player_choices:
            if isinstance(choice, list):
                formatted_player_choices.append(choice)
            else:
                formatted_player_choices.append([choice])
        if len(formatted_player_choices) != self._n_players:
            raise ValueError(f'The number of choices ({len(formatted_player_choices)}) does not match the number of'
                             f' players ({self._n_players})')
        return formatted_player_choices

    def _generate_quantum_circuit(self, player_gates):
        if self._protocol == Protocol.Classical:
            return None
        player_gate_objects = []
        for i in range(len(player_gates)):
            player_gate_objects.append([])
            for j in player_gates[i]:
                player_gate_objects[i].append(generate_unitary_gate(j))
        self._quantum_game = QuantumGame(player_gate_objects, self._protocol, self._backend)
        self._quantum_game.circ.draw()
        return self._quantum_game.circ

    def _generate_final_choices(self, player_choices, n_times):
        if self._protocol == Protocol.Classical:
            player_choices_str = ''
            for player_choice in player_choices:
                for choice in player_choice:
                    player_choices_str += str(choice)
            return {player_choices_str: n_times}
        else:
            print('Transpiling circuit ....')
            transpiled_circuit = transpile(self._quantum_game.circ, backend=self._backend)
            print('Executing transpiled circuit ....')
            job_sim = self._backend.run(transpiled_circuit, shots=n_times)
            print('Circuit running ...')
            job_result = job_sim.result()
            print('Circuit finished running, getting counts ...')
            counts = job_result.get_counts(transpiled_circuit)
            counts_inverted = {}
            for key, value in counts.items():
                if self._num_players == 2 or str(self._backend) == 'qasm_simulator':
                    counts_inverted[key[::-1]] = value
                else:
                    counts_inverted[key[:0:-1]] = counts_inverted.get(key[:0:-1], 0) + value
            return counts_inverted

    def _generate_final_results(self, results):
        outcome = []
        num_times = []
        payoffs = []
        winners = []

        for curr_choices, curr_num_times in results.items():
            outcome.append(curr_choices)
            num_times.append(curr_num_times)
            curr_payoffs = self._payoff_table.get_payoffs(curr_choices)  # Get payoffs based on outcomes
            payoffs.append(curr_payoffs)
            max_payoff = max(curr_payoffs)
            winning_players = ''
            total_winners = 0
            for j in range(len(curr_payoffs)):
                if curr_payoffs[j] == max_payoff:
                    total_winners += 1
                    if winning_players == '':
                        winning_players += f'Player {j+1}'
                    else:
                        winning_players += f' and {j+1}'
            if total_winners == self._n_players:
                winning_players = 'No winners'
            winners.append(winning_players)
        payoff_json = json.dumps(self._payoff_table.get_payoff_table())
        return {'outcome': outcome, 'payoffs': payoffs, 'players': self._num_players, 'game': self._game_name,
                'payoff_matrix': payoff_json, 'winners': winners, 'num_times': num_times, 'backend': str(self._backend)}

    def base64_figure(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        fig_str = base64.b64encode(buf.getvalue())
        fig_str = fig_str.decode("utf-8")

        return fig_str

    def play_game(self, player_choices, n_times=50):
        player_choices = self.format_choices(player_choices)
        self.quantum_circuit = self._generate_quantum_circuit(player_choices)
        final_choices = self._generate_final_choices(player_choices, n_times)
        self._final_results = self._generate_final_results(final_choices)

        # Generate graph(s)
        if self._protocol != Protocol.Classical:
            circuit_fig = self.quantum_circuit.draw(output='mpl')
            circuit_fig.suptitle("Full Circuit for Players", fontsize=25)

            probability_graph_img = plot_histogram(final_choices)
            probability_graph_img.suptitle("Probability Graph", fontsize=25)
            axes = probability_graph_img.get_axes()
            for t in axes[0].get_xticklabels():
                t.set_rotation(0)

            self._final_results['full_circ_str'] = self.base64_figure(circuit_fig)
            self._final_results['graph_str'] = self.base64_figure(probability_graph_img)

        return final_choices, self._final_results

    def show_results(self):
        winners = []
        payoffs = []
        for i in self._final_results['outcome']:
            payoff_dict = eval(self._final_results['payoff_matrix'])
            curr_payoffs = payoff_dict[i]
            payoffs.append(curr_payoffs)
            max_payoff = max(curr_payoffs)
            winning_players = ''
            total_winners = 0
            for j in range(len(curr_payoffs)):
                if curr_payoffs[j] == max_payoff:
                    total_winners += 1
                    if winning_players == '':
                        winning_players += f'Player {j+1}'
                    else:
                        winning_players += f' and {j+1}'
            if total_winners == self._n_players:
                winning_players = 'No winners'
            winners.append(winning_players)
        df = pd.DataFrame({'Outcome':self._final_results['outcome'], 'Payoffs':payoffs, 'Winners':winners, 'num_times':self._final_results['num_times']})
        return df
