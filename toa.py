"""
Muhammad Abd-Ur-Rahman 23K-0760
Ammar Zulfiqar 23K-0908
Abdul-Hadi 23K-0928
Ali Yehya 23K-0569

Regex to DFA Converter using McNaughton-Yamada-Thompson Algorithm - GUI Version
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import graphviz
from collections import deque
from collections import defaultdict

def minimize_dfa(start_state, transitions, accept_states):
  """Minimize DFA using Hopcroft's Algorithm."""
  # 1. Gather all states
  states = set()
  symbols = set()
  for (from_state, symbol), to_state in transitions.items():
    states.add(from_state)
    states.add(to_state)
    symbols.add(symbol)
  
  # 2. Initial partition: accepting and non-accepting states
  accepting = set(accept_states)
  non_accepting = states - accepting
  
  # Start with two partitions (accepting and non-accepting)
  partitions = []
  if accepting:
    partitions.append(accepting)
  if non_accepting:
    partitions.append(non_accepting)
  
  # 3. Refine partitions until no more refinement is possible
  waiting = list(partitions)  # Start with all partitions in waiting list
  
  while waiting:
    split_set = waiting.pop(0)
    
    for symbol in symbols:
      # Find states that can reach split_set via symbol
      predecessors = {}
      for state in states:
        target = transitions.get((state, symbol))
        if target is not None:
          for i, partition in enumerate(partitions):
            if target in partition:
              # Group states by which partition they end up in
              if i not in predecessors:
                predecessors[i] = set()
              predecessors[i].add(state)
              break
      
      # Check if any partition needs to be split
      new_partitions = []
      for partition in partitions:
        # Split partition based on which states can reach split_set
        partition_splits = {}
        for state in partition:
          found = False
          for i, pred_set in predecessors.items():
            if state in pred_set:
              if i not in partition_splits:
                partition_splits[i] = set()
              partition_splits[i].add(state)
              found = True
              break
          
          # States that don't transition to any partition via symbol
          if not found:
            if -1 not in partition_splits:
              partition_splits[-1] = set()
            partition_splits[-1].add(state)
        
        # If partition was split
        if len(partition_splits) > 1:
          for new_set in partition_splits.values():
            new_partitions.append(new_set)
            # Add new set to waiting if not already there
            if new_set != split_set and new_set not in waiting:
              waiting.append(new_set)
        else:
          new_partitions.append(partition)
      
      partitions = new_partitions
  
  # 4. Create the minimized DFA
  # Map each state to its partition representative
  state_mapping = {}
  for i, partition in enumerate(partitions):
    for state in partition:
      state_mapping[state] = f"q{i}"
  
  # Map transitions
  new_transitions = {}
  for (from_state, symbol), to_state in transitions.items():
    new_from = state_mapping[from_state]
    new_to = state_mapping[to_state]
    new_transitions[(new_from, symbol)] = new_to
  
  # Map accepting states and start state
  new_accept_states = {state_mapping[s] for s in accept_states}
  new_start_state = state_mapping[start_state]
  
  return new_start_state, new_transitions, new_accept_states


class NFAState:
  """NFA state with epsilon transitions."""
  def __init__(self, state_id):
    self.id = state_id
    self.transitions = {}  # symbol -> set of states
    self.epsilon_transitions = set()  # set of states
    self.is_accept = False

class DFAState:
  """DFA state representing a set of NFA states."""
  def __init__(self, state_id, nfa_states):
    self.id = state_id
    self.nfa_states = frozenset(nfa_states)  # Frozenset for hashability
    self.transitions = {}  # symbol -> DFAState
    self.is_accept = any(state.is_accept for state in nfa_states)

class NFA:
  """Non-deterministic Finite Automaton with epsilon transitions."""
  def __init__(self):
    self.states = []
    self.start_state = None
    self.accept_state = None
    self.next_state_id = 0
    self.alphabet = set()

  def create_state(self):
    """Create a new NFA state with a unique ID."""
    state = NFAState(self.next_state_id)
    self.next_state_id += 1
    self.states.append(state)
    return state

  def add_transition(self, from_state, symbol, to_state):
    """Add a transition from one state to another on the given symbol."""
    if symbol != 'Îµ':  # Not an epsilon transition
      self.alphabet.add(symbol)
      
    if symbol not in from_state.transitions:
      from_state.transitions[symbol] = set()
    from_state.transitions[symbol].add(to_state)

  def add_epsilon_transition(self, from_state, to_state):
    """Add an epsilon transition from one state to another."""
    from_state.epsilon_transitions.add(to_state)

class Thompson:
  """Thompson's construction algorithm for converting regex to NFA."""
  def __init__(self):
    self.operators = {'|', '*', '(', ')', '+'}
    self.precedence = {'|': 1, '.': 2, '*': 3, '+': 3}
    self.next_nfa_id = 0

  def build_nfa(self, regex, alphabet):
    """Build an NFA from the given regex using Thompson's construction."""
    if not regex:
      # Empty regex - NFA that accepts empty string
      nfa = NFA()
      start = nfa.create_state()
      nfa.start_state = start
      nfa.accept_state = start
      start.is_accept = True
      return nfa

    # Convert infix regex to postfix
    postfix = self._to_postfix(regex)
    
    # Build NFA using Thompson's construction
    stack = []
    
    for char in postfix:
      if char not in self.operators and char != '.':  # Regular character
        # Create a simple NFA for a single character
        nfa = self._create_basic_nfa(char)
        stack.append(nfa)
      elif char == '|':  # Alternation (Union)
        if len(stack) < 2:
          raise ValueError("Invalid regex: not enough operands for '|'")
        nfa2 = stack.pop()
        nfa1 = stack.pop()
        stack.append(self._create_union_nfa(nfa1, nfa2))
      elif char == '.':  # Concatenation
        if len(stack) < 2:
          raise ValueError("Invalid regex: not enough operands for concatenation")
        nfa2 = stack.pop()
        nfa1 = stack.pop()
        stack.append(self._create_concat_nfa(nfa1, nfa2))
      elif char == '*':  # Kleene star (zero or more)
        if len(stack) < 1:
          raise ValueError("Invalid regex: not enough operands for '*'")
        nfa1 = stack.pop()
        stack.append(self._create_kleene_star_nfa(nfa1))
      elif char == '+':  # One or more
        if len(stack) < 1:
          raise ValueError("Invalid regex: not enough operands for '+'")
        nfa1 = stack.pop()
        stack.append(self._create_one_or_more_nfa(nfa1))
    
    if len(stack) != 1:
      raise ValueError("Invalid regex: too many operands")
    
    nfa = stack.pop()
    
    # Add the alphabet
    for symbol in alphabet:
      nfa.alphabet.add(symbol)
      
    return nfa

  def _to_postfix(self, regex):
    """Convert infix regex notation to postfix using Shunting Yard algorithm."""
    # Explicitly add concatenation operator '.'
    output = []
    explicit_regex = ''
    i = 0
    while i < len(regex):
      explicit_regex += regex[i]
      if i < len(regex) - 1:
        if (regex[i] not in '(|' and regex[i+1] not in ')|*+') or \
           (regex[i] in ')*+' and regex[i+1] not in ')|*+'):
          explicit_regex += '.'
      i += 1
      
    # Shunting Yard algorithm
    output = []
    operator_stack = []
    
    for char in explicit_regex:
      if char not in self.operators and char != '.':
        output.append(char)
      elif char == '(':
        operator_stack.append(char)
      elif char == ')':
        while operator_stack and operator_stack[-1] != '(':
          output.append(operator_stack.pop())
        if operator_stack and operator_stack[-1] == '(':
          operator_stack.pop()  # Discard the '('
      else:  # Operator
        while (operator_stack and operator_stack[-1] != '(' and 
             self.precedence.get(char, 0) <= self.precedence.get(operator_stack[-1], 0)):
          output.append(operator_stack.pop())
        operator_stack.append(char)
    
    while operator_stack:
      output.append(operator_stack.pop())
      
    return ''.join(output)

  def _create_basic_nfa(self, char):
    """Create a basic NFA that accepts only the given character."""
    nfa = NFA()
    start = nfa.create_state()
    accept = nfa.create_state()
    
    nfa.add_transition(start, char, accept)
    
    nfa.start_state = start
    nfa.accept_state = accept
    accept.is_accept = True
    
    return nfa

  def _create_union_nfa(self, nfa1, nfa2):
    """Create an NFA that accepts either NFA1 or NFA2 (union)."""
    nfa = NFA()
    start = nfa.create_state()
    accept = nfa.create_state()
    
    # Copy all states from nfa1 and nfa2 to new NFA
    nfa.states.extend(nfa1.states)
    nfa.states.extend(nfa2.states)
    
    # Add epsilon transitions from new start state to both NFAs' start states
    nfa.add_epsilon_transition(start, nfa1.start_state)
    nfa.add_epsilon_transition(start, nfa2.start_state)
    
    # Add epsilon transitions from both NFAs' accept states to new accept state
    nfa.add_epsilon_transition(nfa1.accept_state, accept)
    nfa.add_epsilon_transition(nfa2.accept_state, accept)
    
    # Update accept states
    nfa1.accept_state.is_accept = False
    nfa2.accept_state.is_accept = False
    accept.is_accept = True
    
    nfa.start_state = start
    nfa.accept_state = accept
    
    return nfa

  def _create_concat_nfa(self, nfa1, nfa2):
    """Create an NFA that accepts NFA1 followed by NFA2 (concatenation)."""
    nfa = NFA()
    
    # Copy all states from nfa1 and nfa2 to new NFA
    nfa.states.extend(nfa1.states)
    nfa.states.extend(nfa2.states)
    
    # Add epsilon transition from nfa1's accept state to nfa2's start state
    nfa.add_epsilon_transition(nfa1.accept_state, nfa2.start_state)
    
    # Update accept states
    nfa1.accept_state.is_accept = False
    nfa2.accept_state.is_accept = True
    
    nfa.start_state = nfa1.start_state
    nfa.accept_state = nfa2.accept_state
    
    return nfa

  def _create_kleene_star_nfa(self, nfa1):
    """Create an NFA that accepts zero or more repetitions of NFA1 (Kleene Star)."""
    nfa = NFA()
    start = nfa.create_state()
    accept = nfa.create_state()
    
    # Copy all states from nfa1 to new NFA
    nfa.states.extend(nfa1.states)
    
    # Add epsilon transitions:
    # From new start to new accept (skip the loop)
    nfa.add_epsilon_transition(start, accept)
    # From new start to nfa1's start (enter the loop)
    nfa.add_epsilon_transition(start, nfa1.start_state)
    # From nfa1's accept to new accept (exit after one iteration)
    nfa.add_epsilon_transition(nfa1.accept_state, accept)
    # From nfa1's accept to nfa1's start (continue the loop)
    nfa.add_epsilon_transition(nfa1.accept_state, nfa1.start_state)
    
    # Update accept states
    nfa1.accept_state.is_accept = False
    accept.is_accept = True
    
    nfa.start_state = start
    nfa.accept_state = accept
    
    return nfa

  def _create_one_or_more_nfa(self, nfa1):
    """Create an NFA that accepts one or more repetitions of NFA1 ('+' operator)."""
    nfa = NFA()
    start = nfa.create_state()
    accept = nfa.create_state()
    
    # Copy all states from nfa1 to new NFA
    nfa.states.extend(nfa1.states)
    
    # Add epsilon transitions:
    # From new start to nfa1's start
    nfa.add_epsilon_transition(start, nfa1.start_state)
    # From nfa1's accept to new accept
    nfa.add_epsilon_transition(nfa1.accept_state, accept)
    # From nfa1's accept to nfa1's start (for repetition)
    nfa.add_epsilon_transition(nfa1.accept_state, nfa1.start_state)
    
    # Update accept states
    nfa1.accept_state.is_accept = False
    accept.is_accept = True
    
    nfa.start_state = start
    nfa.accept_state = accept
    
    return nfa

class SubsetConstruction:
  """Subset construction algorithm for converting NFA to DFA."""
  def __init__(self, nfa, alphabet):
    self.nfa = nfa
    self.alphabet = alphabet
    self.dfa_states = {}  # Maps frozenset of NFA states to DFA state ID
    self.dfa_transitions = {}  # Maps (from_state_id, symbol) to to_state_id
    self.dfa_accept_states = set()  # Set of DFA accept state IDs
    self.next_dfa_state_id = 0

  def convert(self):
    """Convert NFA to DFA using the subset construction algorithm."""
    # Start with the epsilon closure of the NFA start state
    initial_states = self._epsilon_closure({self.nfa.start_state})
    initial_state_id = self._get_dfa_state_id(initial_states)
    
    # Initialize DFA start state
    start_state_id = initial_state_id
    
    # Check if initial state is an accept state
    if any(state.is_accept for state in initial_states):
      self.dfa_accept_states.add(initial_state_id)
    
    # Process queue of DFA states
    queue = deque([initial_states])
    processed = set([frozenset(initial_states)])
    
    while queue:
      current_nfa_states = queue.popleft()
      current_dfa_state_id = self._get_dfa_state_id(current_nfa_states)
      
      # Process each symbol in the alphabet
      for symbol in self.alphabet:
        # Find next set of states through symbol transitions
        next_nfa_states = self._move(current_nfa_states, symbol)
        # Add epsilon transitions
        next_nfa_states = self._epsilon_closure(next_nfa_states)
        
        if not next_nfa_states:
          continue
        
        # Get or create the DFA state for this set of NFA states
        next_dfa_state_id = self._get_dfa_state_id(next_nfa_states)
        
        # Add the transition
        self.dfa_transitions[(current_dfa_state_id, symbol)] = next_dfa_state_id
        
        # If we haven't processed this state yet, add it to the queue
        if frozenset(next_nfa_states) not in processed:
          queue.append(next_nfa_states)
          processed.add(frozenset(next_nfa_states))
          
          # Check if this is an accept state
          if any(state.is_accept for state in next_nfa_states):
            self.dfa_accept_states.add(next_dfa_state_id)
    
    return start_state_id, self.dfa_transitions, self.dfa_accept_states

  def _epsilon_closure(self, nfa_states):
    """Find all states reachable from the given states through epsilon transitions."""
    result = set(nfa_states)
    stack = list(nfa_states)
    
    while stack:
      state = stack.pop()
      for next_state in state.epsilon_transitions:
        if next_state not in result:
          result.add(next_state)
          stack.append(next_state)
    
    return result

  def _move(self, nfa_states, symbol):
    """Find all states reachable from the given states through transitions on the given symbol."""
    result = set()
    
    for state in nfa_states:
      if symbol in state.transitions:
        result.update(state.transitions[symbol])
    
    return result

  def _get_dfa_state_id(self, nfa_states):
    """Get the DFA state ID for the given set of NFA states."""
    frozen_states = frozenset(nfa_states)
    
    if frozen_states not in self.dfa_states:
      self.dfa_states[frozen_states] = f"q{self.next_dfa_state_id}"
      self.next_dfa_state_id += 1
    
    return self.dfa_states[frozen_states]

class DFAVisualizer:
  """Visualize a DFA using Graphviz."""
  def __init__(self, start_state, transitions, accept_states):
    self.start_state = start_state
    self.transitions = transitions
    self.accept_states = accept_states
    
  def visualize(self, output_file="dfa_graph"):
    """Create a visual representation of the DFA."""
    g = graphviz.Digraph('DFA', filename=output_file, format='png')
    g.attr(rankdir='LR', size='8,5')
    
    # Collect all states
    states = set()
    states.add(self.start_state)
    for (from_state, _), to_state in self.transitions.items():
      states.add(from_state)
      states.add(to_state)
    
    # Add states to the graph
    for state in states:
      if state in self.accept_states:
        g.node(state, shape='doublecircle')
      else:
        g.node(state, shape='circle')
    
    # Add a special starting node
    g.node('', shape='none')
    g.edge('', self.start_state)
    
    # Add transitions
    transition_dict = {}
    for (from_state, symbol), to_state in self.transitions.items():
      if (from_state, to_state) not in transition_dict:
        transition_dict[(from_state, to_state)] = []
      transition_dict[(from_state, to_state)].append(symbol)
    
    for (from_state, to_state), symbols in transition_dict.items():
      # Sort and combine symbols for cleaner label
      symbols.sort()
      label = ','.join(symbols)
      g.edge(from_state, to_state, label=label)
    
    # Render the graph
    g.render(cleanup=True)
    return f"{output_file}.png"

class RegexToDFAApp:
  def __init__(self, root):
    self.root = root
    self.root.title("Regex to DFA Converter - Thompson Construction")
    
    # Set window size and position
    window_width = 1000
    window_height = 700
    screen_width = self.root.winfo_screenwidth()
    screen_height = self.root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    self.dfa_image_path = None
    self.create_widgets()
    
  def create_widgets(self):
    # Header with student info
    header_frame = ttk.Frame(self.root, padding="10")
    header_frame.pack(fill=tk.X)
    
    header_text = """
Muhammad Abd-Ur-Rahman 23K-0760
Ammar Zulfiqar 23K-0908
Abdul-Hadi 23K-0928
Ali Yehya 23K-0569

Regex to DFA Converter using McNaughton-Yamada-Thompson Algorithm
Plus Minimization using Hopcroft's Algorithm
    """
    header_label = ttk.Label(header_frame, text=header_text, justify=tk.CENTER, font=("Arial", 12))
    header_label.pack()
    
    # Main content
    main_frame = ttk.Frame(self.root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Left panel - Input
    input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
    input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Regex input
    regex_frame = ttk.Frame(input_frame)
    regex_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(regex_frame, text="Regular Expression:").pack(anchor=tk.W)
    self.regex_entry = ttk.Entry(regex_frame, width=40)
    self.regex_entry.pack(fill=tk.X, pady=5)
    
    # Alphabet input
    alphabet_frame = ttk.Frame(input_frame)
    alphabet_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(alphabet_frame, text="Alphabet (space-separated characters):").pack(anchor=tk.W)
    self.alphabet_entry = ttk.Entry(alphabet_frame, width=40)
    self.alphabet_entry.pack(fill=tk.X, pady=5)
    
    # Extended operators
    operator_frame = ttk.LabelFrame(input_frame, text="Supported Operators", padding="5")
    operator_frame.pack(fill=tk.X, padx=5, pady=5)
    
    operator_info = """
Supported Operators:
() - Grouping
| - Alternation (OR)
* - Kleene star (zero or more)
+ - Kleene plus (one or more)
    """
    operator_label = ttk.Label(operator_frame, text=operator_info, justify=tk.LEFT)
    operator_label.pack(anchor=tk.W)
    
    # Examples
    examples_frame = ttk.LabelFrame(input_frame, text="Examples", padding="5")
    examples_frame.pack(fill=tk.X, padx=5, pady=10)
    
    example_regex = [
      ("a*b", "a b"),
      ("a|b", "a b"),
      ("(a|b)*abb", "a b"),
      ("ab+c", "a b c")
    ]
    
    for i, (regex, alphabet) in enumerate(example_regex):
      example_btn = ttk.Button(
        examples_frame, 
        text=f"Example {i+1}: {regex}",
        command=lambda r=regex, a=alphabet: self.load_example(r, a)
      )
      example_btn.pack(fill=tk.X, pady=2)
    
    # Convert button
    convert_btn = ttk.Button(input_frame, text="Convert to DFA", command=self.convert_regex_to_dfa)
    convert_btn.pack(fill=tk.X, padx=5, pady=10)
    
    # Status
    self.status_var = tk.StringVar()
    self.status_var.set("Ready")
    status_label = ttk.Label(input_frame, textvariable=self.status_var, font=("Arial", 10))
    status_label.pack(fill=tk.X, padx=5, pady=10)
    
    # Right panel - Output
    output_frame = ttk.LabelFrame(main_frame, text="DFA Visualization", padding="10")
    output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Image display
    self.image_frame = ttk.Frame(output_frame)
    self.image_frame.pack(fill=tk.BOTH, expand=True)
    
    self.image_label = ttk.Label(self.image_frame)
    self.image_label.pack(fill=tk.BOTH, expand=True)
    
    # DFA information
    self.info_text = tk.Text(output_frame, height=10, width=50, state=tk.DISABLED)
    self.info_text.pack(fill=tk.X, padx=5, pady=5)
    
  def load_example(self, regex, alphabet):
    self.regex_entry.delete(0, tk.END)
    self.regex_entry.insert(0, regex)
    
    self.alphabet_entry.delete(0, tk.END)
    self.alphabet_entry.insert(0, alphabet)
    
  def convert_regex_to_dfa(self):
    regex = self.regex_entry.get().strip()
    alphabet_input = self.alphabet_entry.get().strip()
    alphabet = alphabet_input.split()
    
    if not regex:
      messagebox.showerror("Error", "Please enter a regular expression.")
      return
    
    if not alphabet:
      messagebox.showerror("Error", "Please enter the alphabet.")
      return
    
    # Validate that regex only uses the provided alphabet and operators
    valid_chars = set(alphabet + ['|', '*', '(', ')', '+'])
    if any(c not in valid_chars for c in regex if c not in [' ', '\t', '\n']):
      invalid_chars = [c for c in regex if c not in valid_chars and c not in [' ', '\t', '\n']]
      messagebox.showerror("Error", f"Regex contains characters not in the alphabet: {', '.join(invalid_chars)}")
      return
    
    # Remove whitespace from regex
    regex = ''.join(regex.split())
    
    # Use a thread to avoid freezing the UI
    threading.Thread(target=self.process_conversion, args=(regex, alphabet)).start()
  
  def process_conversion(self, regex, alphabet):
    try:
      self.status_var.set("Converting regex to NFA using Thompson's construction...")
      self.root.update_idletasks()
      
      # Build NFA using Thompson's construction
      thompson = Thompson()
      nfa = thompson.build_nfa(regex, alphabet)
      
      self.status_var.set("Converting NFA to DFA using subset construction...")
      self.root.update_idletasks()
      
      # Convert NFA to DFA using subset construction
      # subset = SubsetConstruction(nfa, alphabet)
      # start_state, transitions, accept_states = subset.convert()

      subset = SubsetConstruction(nfa, alphabet)
      start_state, transitions, accept_states = subset.convert()
      start_state, transitions, accept_states = minimize_dfa(start_state, transitions, accept_states)
      
      self.status_var.set("Generating visualization...")
      self.root.update_idletasks()
      
      # Generate the visualization
      output_file = f"dfa_thompson_{int(time.time())}"
      visualizer = DFAVisualizer(start_state, transitions, accept_states)
      self.dfa_image_path = visualizer.visualize(output_file)
      
      # Update the UI
      self.root.after(0, lambda: self.update_ui_with_results(start_state, transitions, accept_states))
      
    except Exception as e:
      self.root.after(0, lambda e=e: self.show_error(str(e)))
  
  def update_ui_with_results(self, start_state, transitions, accept_states):
    try:
      # Load and display the image
      pil_image = Image.open(self.dfa_image_path)
      self.display_image(pil_image)
      
      # Update info text
      self.info_text.config(state=tk.NORMAL)
      self.info_text.delete(1.0, tk.END)
      
      # Calculate number of states
      states = set()
      states.add(start_state)
      for (from_state, _), to_state in transitions.items():
        states.add(from_state)
        states.add(to_state)
      
      info = f"States: {len(states)}\n"
      info += f"Start state: {start_state}\n"
      info += f"Accept states: {', '.join(sorted(accept_states))}\n\n"
      info += "Transitions:\n"
      
      # Group transitions by from_state for cleaner output
      state_transitions = {}
      for (from_state, symbol), to_state in sorted(transitions.items()):
        if from_state not in state_transitions:
          state_transitions[from_state] = []
        state_transitions[from_state].append((symbol, to_state))
      
      for from_state, transitions_list in sorted(state_transitions.items()):
        for symbol, to_state in sorted(transitions_list):
          info += f"  {from_state} --({symbol})--> {to_state}\n"
      
      self.info_text.insert(tk.END, info)
      self.info_text.config(state=tk.DISABLED)
      
      self.status_var.set("Conversion completed successfully.")
      
    except Exception as e:
      self.show_error(f"Error displaying results: {str(e)}")
  
  def display_image(self, pil_image):
    # Calculate size to fit in the frame
    frame_width = self.image_frame.winfo_width()
    frame_height = self.image_frame.winfo_height()
    
    if frame_width <= 1 or frame_height <= 1:  # Frame not yet drawn
      self.root.update_idletasks()
      frame_width = self.image_frame.winfo_width()
      frame_height = self.image_frame.winfo_height()
    
    img_width, img_height = pil_image.size
    
    scale = min(frame_width / img_width, frame_height / img_height)
    new_width = int(img_width * scale * 0.9)  # 90% of available space
    new_height = int(img_height * scale * 0.9)
    
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    self.tk_image = ImageTk.PhotoImage(resized_image)
    
    self.image_label.config(image=self.tk_image)
    self.image_label.image = self.tk_image
  
  def show_error(self, message):
    messagebox.showerror("Error", message)
    self.status_var.set("Error occurred. See details.")

def main():
  root = tk.Tk()
  app = RegexToDFAApp(root)
  root.mainloop()

if __name__ == "__main__":
  main() 