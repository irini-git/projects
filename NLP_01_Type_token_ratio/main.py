from type_token_ratio import Documents, Graphs
import glob

# Constants
AUTHOR = "Agatha Christie"

# Files in the directory
FILES = glob.glob(f"./data/{AUTHOR}*.txt")

# Calculate type to token ratio
for f in FILES:
    filename = f[7:]
    book = Documents(f'./data/{filename}')

# Plot  type to token ratio
graph = Graphs()
graph.ttr_scatter_plot()

# END ------------------------------------