#!/usr/bin/env python3

import pandas
from plotnine import *

# read dataset
data = pandas.read_csv('./output/2022-11-23_20-39-53_single_node_performance/single_node_performance_with_papi.csv')
grouped = data.groupby(data.type)
print(grouped.head())

def format_number(x):
  if x == 10:
    return '10'
  if x == 20:
    return '20'
  if x == 30:
    return '30'
  if x == 250_000:
    return '250k'
  if x == 500_000:
    return '500k'
  if x == 750_000:
    return '750k'
  if x == 1_000_000:
    return '1M'
  if x == 1_250_000:
    return '1.25M'
  if x == 2_000_000:
    return '2M'
  if x == 3_000_000:
    return '3M'
  if x == 4_000_000:
    return '4M'
  if x == 5_000_000:
    return '5M'
  if x == 100_000_000_000:
    return '100G'
  if x == 200_000_000_000:
    return '200G'
  if x == 300_000_000_000:
    return '300G'
  if x == 400_000_000_000:
    return '400G'
  if x == 500_000_000_000:
    return '500G'
  return str(x)

def format_breakpoints(breakpoints):
  return list(map(format_number, breakpoints))

def plot_single_node_performance(property, title, bidirectional_color_scale):
  plot = (ggplot(data)
    + facet_grid(facets='~type') # create subplot for each type
    + aes(
      # use factor to display all values on the axes
      x='factor(control_qubit)',
      y='factor(target_qubit)',
      fill=property,
    )
    + labs(
      x='Control qubit (0–24)',
      y='Target qubit (0–24)',
      fill=title,
    )
    + scale_fill_cmap(
      # see palettes at https://matplotlib.org/2.0.2/users/colormaps.html
      cmap_name='RdYlBu' if bidirectional_color_scale else 'magma',
      trans='reverse',
      labels=format_breakpoints,
      **(dict(limits=[0, 2]) if bidirectional_color_scale else {}),
    )
    + guides(fill=guide_colorbar(
      reverse=True,
      direction='horizontal',
      label_position='top',
      barheight=50,
      barwidth=10,
    ))
    + geom_tile(aes(width=0.9, height=0.9)) # resize to leave gaps between tiles
    + theme(
      figure_size=(20, 5),
      panel_background=element_rect(fill='white'),
      text=element_text(family='TeX Gyre Schola', size=25),

      # axes
      axis_ticks=element_blank(),
      axis_text=element_blank(),

      # legend
      legend_text=element_text(size=15),
      legend_position='top',

      # subplot titles
      strip_background=element_text(color='white'),
    ))

  return plot

def generate_plot(property, title, filename, shorten_numbers=False):
  output_directory = '../../master-thesis/Bilder/'
  plot = plot_single_node_performance(property, title, shorten_numbers)
  plot.save(output_directory + filename + '.png', dpi=300, limitsize=False)
  # plot.save(output_directory + filename + '.pdf', limitsize=False)

generate_plot('time_in_sec', 'Time (seconds)', 'plot_single_node_time')
generate_plot('l2_norm', 'L2 norm', 'plot_single_node_norm', True)
generate_plot('cycles', 'CPU cycles', 'plot_single_node_cycles')
generate_plot('PAPI_L1_TCM', 'Level 1 cache misses', 'plot_single_node_l1_misses')
generate_plot('PAPI_L2_TCM', 'Level 2 cache misses', 'plot_single_node_l2_misses')
generate_plot('PAPI_L3_TCM', 'Level 3 cache misses', 'plot_single_node_l3_misses')
