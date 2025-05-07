#!/usr/bin/env python
"""
Visualization utilities for the traffic management system.
Provides functions for visualizing traffic data, predictions, and simulation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Set up color schemes
COLORS = {
    'congestion': ['#4daf4a', '#fee08b', '#e41a1c'],  # Green to yellow to red
    'prediction': '#3182bd',                          # Blue
    'actual': '#333333',                              # Black
    'highlight': '#ff7f00'                            # Orange
}

def setup_plot_style():
    """Set up the plot style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#dddddd'
    plt.rcParams['axes.facecolor'] = '#f5f5f5'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def visualize_time_series(df: pd.DataFrame, x_col: str, y_cols: List[str], 
                        title: str = 'Time Series Visualization', 
                        x_label: str = 'Time', y_label: str = 'Value',
                        save_path: Optional[str] = None):
    """
    Visualize time series data.
    
    Args:
        df (pd.DataFrame): DataFrame containing time series data
        x_col (str): Column name for x-axis (usually timestamp)
        y_cols (list): List of column names for y-axis
        title (str): Plot title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        save_path (str, optional): Path to save the figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    
    # Check if x_col is a datetime
    if pd.api.types.is_datetime64_dtype(df[x_col]):
        formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
        ax.xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
    
    # Plot each y column
    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x_col], df[col], label=col, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved time series visualization to {save_path}")
    
    return fig, ax

def visualize_congestion_heatmap(df: pd.DataFrame, segment_col: str, time_col: str, 
                               value_col: str = 'congestion', 
                               title: str = 'Traffic Congestion Heatmap',
                               save_path: Optional[str] = None):
    """
    Create a heatmap visualization of traffic congestion over time for different segments.
    
    Args:
        df (pd.DataFrame): DataFrame containing traffic data
        segment_col (str): Column name for road segments
        time_col (str): Column name for timestamps
        value_col (str): Column name for congestion values
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    setup_plot_style()
    
    # Pivot the data for the heatmap
    pivot_df = df.pivot(index=segment_col, columns=time_col, values=value_col)
    
    # Create custom colormap for congestion (green to yellow to red)
    cmap = LinearSegmentedColormap.from_list('congestion', COLORS['congestion'], N=100)
    
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(pivot_df, cmap=cmap, vmin=0, vmax=1, 
                    linewidths=0.5, cbar_kws={'label': 'Congestion Level'})
    
    plt.title(title, fontsize=14)
    plt.ylabel('Road Segment')
    plt.xlabel('Time')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved congestion heatmap to {save_path}")
    
    return plt.gcf(), ax

def visualize_network(graph: nx.Graph, node_attribute: Optional[str] = None, 
                    edge_attribute: Optional[str] = None, 
                    title: str = 'Traffic Network Visualization',
                    save_path: Optional[str] = None):
    """
    Visualize the traffic network as a graph.
    
    Args:
        graph (nx.Graph): NetworkX graph of the traffic network
        node_attribute (str, optional): Node attribute to visualize through colors
        edge_attribute (str, optional): Edge attribute to visualize through colors
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    setup_plot_style()
    
    plt.figure(figsize=(16, 12))
    
    # Get positions from node attributes if available, otherwise use spring layout
    if all('pos' in data for _, data in graph.nodes(data=True)):
        pos = {node: data['pos'] for node, data in graph.nodes(data=True)}
    else:
        pos = nx.spring_layout(graph, seed=42)
    
    # Node visualization
    if node_attribute and node_attribute in next(iter(graph.nodes(data=True)))[1]:
        node_values = [data[node_attribute] for _, data in graph.nodes(data=True)]
        nodes = nx.draw_networkx_nodes(graph, pos, node_size=200, node_color=node_values, cmap=plt.cm.viridis)
        plt.colorbar(nodes, label=node_attribute)
    else:
        nx.draw_networkx_nodes(graph, pos, node_size=200, node_color='skyblue')
    
    # Edge visualization
    if edge_attribute and all(edge_attribute in graph[u][v] for u, v in graph.edges()):
        edge_values = [graph[u][v][edge_attribute] for u, v in graph.edges()]
        cmap = LinearSegmentedColormap.from_list('congestion', COLORS['congestion'], N=100)
        edges = nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_values, 
                                     edge_cmap=cmap, arrows=True, arrowsize=15)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(edge_values), max(edge_values)))
        sm.set_array([])
        plt.colorbar(sm, label=edge_attribute)
    else:
        nx.draw_networkx_edges(graph, pos, width=1, edge_color='gray', arrows=True)
    
    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved network visualization to {save_path}")
    
    return plt.gcf(), plt.gca()

def visualize_prediction_vs_actual(actual: Union[np.ndarray, List], 
                                 prediction: Union[np.ndarray, List],
                                 title: str = 'Prediction vs. Actual',
                                 x_label: str = 'Time Step',
                                 y_label: str = 'Value',
                                 save_path: Optional[str] = None):
    """
    Visualize prediction vs. actual values.
    
    Args:
        actual (array): Actual values
        prediction (array): Predicted values
        title (str): Plot title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        save_path (str, optional): Path to save the figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    
    # Convert to numpy arrays
    actual = np.array(actual)
    prediction = np.array(prediction)
    
    # Plot actual and prediction
    time_steps = np.arange(len(actual))
    ax.plot(time_steps, actual, label='Actual', color=COLORS['actual'], linewidth=2)
    ax.plot(time_steps, prediction, label='Prediction', color=COLORS['prediction'], 
            linewidth=2, linestyle='--')
    
    # Calculate error metrics
    mae = np.mean(np.abs(actual - prediction))
    rmse = np.sqrt(np.mean(np.square(actual - prediction)))
    
    # Add text with metrics
    ax.text(0.02, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
           transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved prediction vs. actual visualization to {save_path}")
    
    return fig, ax

def visualize_route(graph: nx.Graph, path: List[str], title: str = 'Route Visualization',
                  save_path: Optional[str] = None):
    """
    Visualize a route on the traffic network.
    
    Args:
        graph (nx.Graph): NetworkX graph of the traffic network
        path (list): List of node IDs representing the route
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    setup_plot_style()
    
    plt.figure(figsize=(16, 12))
    
    # Get positions
    if all('pos' in data for _, data in graph.nodes(data=True)):
        pos = {node: data['pos'] for node, data in graph.nodes(data=True)}
    else:
        pos = nx.spring_layout(graph, seed=42)
    
    # Draw background network
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='lightgray')
    nx.draw_networkx_edges(graph, pos, width=1, edge_color='lightgray', alpha=0.5)
    
    # Create route edges
    route_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    
    # Highlight route
    nx.draw_networkx_nodes(graph, pos, nodelist=path, node_size=200, 
                           node_color=COLORS['highlight'])
    nx.draw_networkx_edges(graph, pos, edgelist=route_edges, width=3, 
                           edge_color=COLORS['highlight'])
    
    # Highlight origin and destination
    nx.draw_networkx_nodes(graph, pos, nodelist=[path[0]], node_size=300, node_color='green')
    nx.draw_networkx_nodes(graph, pos, nodelist=[path[-1]], node_size=300, node_color='red')
    
    # Add labels
    route_labels = {node: node for node in path}
    nx.draw_networkx_labels(graph, pos, labels=route_labels, font_size=10)
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved route visualization to {save_path}")
    
    return plt.gcf(), plt.gca()

def visualize_simulation_results(results: Dict, save_path: Optional[str] = None):
    """
    Visualize simulation results with multiple metrics.
    
    Args:
        results (dict): Dictionary containing simulation results
        save_path (str, optional): Path to save the figure
    """
    setup_plot_style()
    
    # Extract statistics from results
    stats = results.get('stats', {})
    
    # Check if we have statistics to plot
    if not stats or all(len(stat) == 0 for stat in stats.values()):
        logger.warning("No statistics available for visualization")
        return None, None
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot travel time
    if 'avg_travel_time' in stats and stats['avg_travel_time']:
        times, values = zip(*stats['avg_travel_time'])
        axs[0, 0].plot(times, values, 'o-', color=COLORS['highlight'])
        axs[0, 0].set_title('Average Travel Time')
        axs[0, 0].set_xlabel('Simulation Time (s)')
        axs[0, 0].set_ylabel('Travel Time (s)')
    
    # Plot waiting time
    if 'avg_waiting_time' in stats and stats['avg_waiting_time']:
        times, values = zip(*stats['avg_waiting_time'])
        axs[0, 1].plot(times, values, 'o-', color=COLORS['prediction'])
        axs[0, 1].set_title('Average Waiting Time')
        axs[0, 1].set_xlabel('Simulation Time (s)')
        axs[0, 1].set_ylabel('Waiting Time (s)')
    
    # Plot throughput
    if 'throughput' in stats and stats['throughput']:
        times, values = zip(*stats['throughput'])
        axs[1, 0].plot(times, values, 'o-', color='green')
        axs[1, 0].set_title('Throughput (Vehicles per Minute)')
        axs[1, 0].set_xlabel('Simulation Time (s)')
        axs[1, 0].set_ylabel('Vehicles')
    
    # Plot congestion
    if 'congestion' in stats and stats['congestion']:
        times, values = zip(*stats['congestion'])
        axs[1, 1].plot(times, values, 'o-', color='red')
        axs[1, 1].set_title('Average Congestion Level')
        axs[1, 1].set_xlabel('Simulation Time (s)')
        axs[1, 1].set_ylabel('Congestion (0-1)')
        axs[1, 1].set_ylim(0, 1)
    
    fig.suptitle('Simulation Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved simulation results visualization to {save_path}")
    
    return fig, axs

def create_simulation_animation(simulation_data: List[Dict], interval: int = 200, 
                              save_path: Optional[str] = None):
    """
    Create an animation of a traffic simulation.
    
    Args:
        simulation_data (list): List of dictionaries with simulation state at each time step
        interval (int): Interval between frames in milliseconds
        save_path (str, optional): Path to save the animation
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    setup_plot_style()
    
    if not simulation_data:
        logger.warning("No simulation data available for animation")
        return None
    
    # Extract graph from first frame
    graph = simulation_data[0].get('graph')
    
    if not graph:
        logger.warning("No graph found in simulation data")
        return None
    
    # Get positions from node attributes if available, otherwise use spring layout
    if all('pos' in data for _, data in graph.nodes(data=True)):
        pos = {node: data['pos'] for node, data in graph.nodes(data=True)}
    else:
        pos = nx.spring_layout(graph, seed=42)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Initialize plot elements
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='lightgray', ax=ax)
    edges = nx.draw_networkx_edges(graph, pos, width=1.0, edge_color='lightgray', arrows=True, arrowsize=15, ax=ax)
    labels = nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black', ax=ax)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.8))
    
    # Vehicle count text
    vehicle_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8))
    
    # Color map for congestion
    cmap = LinearSegmentedColormap.from_list('congestion', COLORS['congestion'], N=100)
    
    def init():
        """Initialize the animation."""
        ax.set_title('Traffic Simulation', fontsize=14)
        ax.axis('off')
        return nodes, edges, time_text, vehicle_text
    
    def update(frame_idx):
        """Update the animation for each frame."""
        ax.clear()
        
        # Get data for current frame
        frame_data = simulation_data[frame_idx]
        current_time = frame_data.get('time', 0)
        vehicles = frame_data.get('vehicles', {})
        traffic_data = frame_data.get('traffic_data', {})
        
        # Update time text
        time_text = ax.text(0.02, 0.98, f'Time: {current_time:.1f}s', transform=ax.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))
        
        # Update vehicle count text
        active_vehicles = sum(1 for v in vehicles.values() if v.get('status') == 'moving')
        arrived_vehicles = sum(1 for v in vehicles.values() if v.get('status') == 'arrived')
        waiting_vehicles = sum(1 for v in vehicles.values() if v.get('status') == 'waiting')
        
        vehicle_text = ax.text(0.02, 0.93, 
                             f'Vehicles: {active_vehicles} active, {arrived_vehicles} arrived, {waiting_vehicles} waiting',
                             transform=ax.transAxes, fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.8))
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='lightgray', ax=ax)
        
        # Color edges based on congestion
        edge_colors = []
        for u, v in graph.edges():
            edge_data = traffic_data.get((u, v), {})
            congestion = edge_data.get('congestion', 0.0)
            edge_colors.append(congestion)
        
        # Draw edges with congestion colors
        edges = nx.draw_networkx_edges(graph, pos, width=2.0, edge_color=edge_colors, 
                                     edge_cmap=cmap, edge_vmin=0, edge_vmax=1,
                                     arrows=True, arrowsize=15, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black', ax=ax)
        
        # Draw vehicles
        for vehicle_id, vehicle in vehicles.items():
            if vehicle.get('status') == 'moving':
                # Get vehicle position
                current_edge = vehicle.get('current_edge')
                if current_edge:
                    source, target = current_edge
                    progress = vehicle.get('progress', 0.0)
                    
                    # Interpolate position along the edge
                    source_pos = pos[source]
                    target_pos = pos[target]
                    vehicle_pos = (
                        source_pos[0] + progress * (target_pos[0] - source_pos[0]),
                        source_pos[1] + progress * (target_pos[1] - source_pos[1])
                    )
                    
                    # Draw vehicle
                    ax.plot(vehicle_pos[0], vehicle_pos[1], 'o', color='blue', markersize=5)
        
        # Add colorbar for congestion
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Congestion Level')
        
        ax.set_title('Traffic Simulation', fontsize=14)
        ax.axis('off')
        
        return edges, time_text, vehicle_text
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(simulation_data),
        init_func=init, blit=False, interval=interval
    )
    
    # Save animation if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer='ffmpeg', fps=30)
        logger.info(f"Saved simulation animation to {save_path}")
    
    plt.close()
    
    return ani 