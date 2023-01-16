#!/usr/bin/env python3

import pandas
from plotnine import *

# read dataset
data = pandas.read_csv('./output/2022-11-23_20-39-53_single_node_performance/single_node_performance_with_papi.csv')
grouped = data.groupby(data.type)
print(grouped.head())

aggregated = grouped.aggregate({
  'time_in_sec': ['mean', 'median', 'std', 'min', 'max'],
  'l2_norm': ['mean', 'median', 'std', 'min', 'max'],
  'cycles': ['mean', 'median', 'std', 'min', 'max'],
  'PAPI_L1_TCM': ['mean', 'median', 'std', 'min', 'max'],
  'PAPI_L2_TCM': ['mean', 'median', 'std', 'min', 'max'],
  'PAPI_L3_TCM': ['mean', 'median', 'std', 'min', 'max'],
})

output_directory = '../../master-thesis/Bilder/'

def format_number(x):
  if x == 0:
    return '0'
  if x == 10:
    return '10'
  if x == 20:
    return '20'
  if x == 30:
    return '30'
  if x == 40:
    return '40'
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
  if x == 6_000_000:
    return '6M'
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
  if x == 600_000_000_000:
    return '600G'
  return str(x)

def format_breakpoints(breakpoints):
  return list(map(format_number, breakpoints))

def plot_single_node_performance(property, title, filename, bidirectional_color_scale=False):
  plot = (ggplot(data)
    + facet_grid(facets='~type') # create subplot for each type
    + aes(
      # use factor to display all values on the axes
      x='factor(target_qubit)',
      y='factor(control_qubit)',
      fill=property,
    )
    + labs(
      x='Target qubit (0–24)',
      y='Control qubit (0–24)',
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

  plot.save(output_directory + filename + '.png', dpi=300, limitsize=False)
  # plot.save(output_directory + filename + '.pdf', limitsize=False)

def plot_single_node_performance_aggregated(property, title, filename):
  print(property)
  print(aggregated[property])

  plot = (ggplot(data)
    + aes(
      # use factor to display all values on the axes
      x='type',
      y=property,
    )
    + labs(
      x=None,
      y=title,
    )
    + geom_boxplot(
      color='black',
      width=0.25,
    )
    + scale_y_continuous(labels=format_breakpoints)
    + theme(
      figure_size=(15, 5),
      panel_background=element_rect(fill='white'),
      panel_grid_major=element_line(colour="#d3d3d3"),
      panel_grid_minor=element_blank(),
      axis_ticks=element_blank(),
      axis_text=element_text(color='black'),
      text=element_text(family='TeX Gyre Schola', size=25),
    ))

  plot.save(output_directory + filename + '.png', dpi=300, limitsize=False)
  # plot.save(output_directory + filename + '.pdf', limitsize=False)

def plot_single_node_cache_misses(filename):
  print('Cache misses')
  print(aggregated[['PAPI_L1_TCM', 'PAPI_L2_TCM', 'PAPI_L3_TCM']])

  plot = (ggplot(data.melt(id_vars=['type'], value_vars=['PAPI_L1_TCM', 'PAPI_L2_TCM', 'PAPI_L3_TCM'], var_name='property', value_name='cache_misses'))
    + facet_grid(facets='~type') # create subplot for each type
    + aes(
      # use factor to display all values on the axes
      x='factor(property)',
      y='cache_misses',
      fill='factor(property)',
    )
    + labs(
      x=None,
      y='Cache misses',
    )
    + geom_boxplot(
      color='black',
      width=0.6,
    )
    + scale_fill_manual(values=['#f8961e', '#90be6d', '#4d908e'], guide=False)
    + scale_y_continuous(labels=format_breakpoints)
    + scale_x_discrete(labels=['L1', 'L2', 'L3'])
    + theme(
      figure_size=(15, 5),
      panel_background=element_rect(fill='white'),
      panel_grid_major=element_line(colour="#d3d3d3"),
      panel_grid_minor=element_blank(),
      axis_ticks=element_blank(),
      axis_text=element_text(color='black'),
      text=element_text(family='TeX Gyre Schola', size=25),

      # subplot titles
      strip_background=element_text(color='white'),
    ))

  plot.save(output_directory + filename + '.png', dpi=300, limitsize=False)
  # plot.save(output_directory + filename + '.pdf', limitsize=False)

plot_single_node_performance('time_in_sec', 'Time (seconds)', 'plot_single_node_time')
plot_single_node_performance('cycles', 'CPU cycles', 'plot_single_node_cycles')
plot_single_node_performance('PAPI_L1_TCM', 'Level 1 cache misses', 'plot_single_node_l1_misses')
plot_single_node_performance('PAPI_L2_TCM', 'Level 2 cache misses', 'plot_single_node_l2_misses')
plot_single_node_performance('PAPI_L3_TCM', 'Level 3 cache misses', 'plot_single_node_l3_misses')
plot_single_node_performance('l2_norm', 'L2 norm', 'plot_single_node_norm', True)

plot_single_node_performance_aggregated('time_in_sec', 'Time (seconds)', 'plot_single_node_time_aggregated')
plot_single_node_performance_aggregated('cycles', 'CPU cycles', 'plot_single_node_cycles_aggregated')
plot_single_node_performance_aggregated('l2_norm', 'L2 norm', 'plot_single_node_norm_aggregated')

plot_single_node_cache_misses('plot_single_node_cache_misses_aggregated')
