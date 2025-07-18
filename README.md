# Theory-of-Automata-Project

A Python GUI application that converts a regular expression into a minimized DFA, using McNaughton-Yamada-Thompson construction to build an ε-NFA, subset construction to generate a DFA, and Hopcroft's algorithm to minimize it. The resulting DFA is rendered using Graphviz and displayed in a Tkinter GUI.

👨‍💻 Authors
Muhammad Abd-Ur-Rahman — 23K-0760

Ammar Zulfiqar — 23K-0908

Abdul-Hadi — 23K-0928

Ali Yehya — 23K-0569

🚀 Features
Convert regular expressions to minimized DFA

Visualize DFA as a state graph using Graphviz

Interactive Tkinter GUI for input/output

Supports |, *, +, concatenation, and parentheses

Example regexes for quick testing

Real-time status updates during conversion

📸 GUI Preview
(You can include a screenshot here)

🧮 Supported Operators
| → Union (OR)

* → Kleene Star (zero or more)

+ → One or more

() → Grouping

Concatenation is implicit (e.g., ab = a followed by b)

📚 How It Works
Regex → ε-NFA: Using Thompson's Construction

ε-NFA → DFA: Using Subset Construction

DFA → Minimized DFA: Using Hopcroft’s Algorithm

Graph Rendering: With graphviz.Digraph to a .png

GUI Display: Shown inside a Tkinter window using PIL

🛠 Requirements
Install dependencies using pip:

bash
Copy
Edit
pip install graphviz pillow
🧰 System Requirements:
Python 3.8+

graphviz installed and added to system PATH (for rendering images)

On Ubuntu/Debian:

bash
Copy
Edit
sudo apt install graphviz
On Windows:
Install Graphviz from the official website and add it to PATH.

🧪 Example Inputs
You can test with the following regex/alphabet pairs:

Regex	Alphabet
a*b	a b
`a	b`
`(a	b)*abb`
ab+c	a b c

🖥️ How to Run
bash
Copy
Edit
python toa.py
A GUI window will open. Enter your regex and alphabet, click Convert to DFA, and view the resulting minimized DFA graph and transitions.

📦 File Structure
graphql
Copy
Edit
toa.py                   # Main application (logic + GUI)
output.png               # Generated DFA image (auto-deleted after display)
🧾 Library Versions Used
Library	Version
tkinter	Built-in (Python standard library)
Pillow	10.3.0
graphviz	0.20.1
Python	3.10.x+ recommended
