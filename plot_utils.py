import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Preferred defaults

def setup_mpl_defaults():
  from matplotlib import rc
  from matplotlib import rcParams
  rc("text", usetex=True)
  rcParams["backend"] = "ps"
  rcParams["text.latex.preamble"] = ["\usepackage{gensymb}"]
  rcParams["font.size"] = 12
  rcParams["legend.fontsize"] = 10
  #rc("font", **{"family":"serif", "serif":["Computer Modern Roman"],
            #"monospace": ["Computer Modern Typewriter"]})

# More helpers

def plot_line_graph(xs, ys_list, legend_labels):
  plt.figure()
  for ys, legend_label in zip(ys_list, legend_labels):
    plt.plot(xs, ys, label=legend_label)
  plt.legend()

def save_graph_pdf(output_file):
  pp = PdfPages(output_file)
  plt.savefig(pp, format="pdf", bbox_inches="tight")
  pp.close()
