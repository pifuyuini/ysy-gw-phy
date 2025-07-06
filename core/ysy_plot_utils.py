"""
ysy_plot_utils.py

Copyright (c) 2025 pifuyuini

Author: pifuyuini
Email: You can contact me via Github
Version: 1.5.0
Date: 2025-05-05

Description:
    This module provides a set of utility functions and styling presets for creating high-quality, customizable
    plots using Matplotlib. It is designed to streamline the process of producing publication-ready visualizations
    by applying consistent themes, color palettes, and configuration options.

    The module supports fine-tuned control of figure aesthetics, custom color cycles inspired by the "firefly" theme,
    and simple yet powerful APIs for quick plotting.

    Key Features:
    - `ysy_settings`: Configures global Matplotlib settings with options for DPI, color, font, and marker styles.
    - `science_settings`: Minimalist scientific plotting style optimized for paper-ready figures.
    - `firefly`: Provides a predefined set of color codes or palettes based on a personalized theme.
    - `firefly_color_theme`: Generates a Matplotlib colormap and gradient based on the firefly palette.
    - `plot`: High-level wrapper for quickly drawing single or multiple curves or scatter plots with legends.
    - `temp_style`: A context manager for temporarily applying custom Matplotlib styles by combining preset themes and user-defined style strings.
    - Auto support for Jupyter Notebook display and marker cycling.
    - Support for additional Matplotlib style registration via temporary style files.

License:
    This file is licensed under the MIT License. You may use, modify, and distribute it under the terms of the license.
    For more details, see the LICENSE file in the project root.

Usage:
    Example usage of the module:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import ysy_plot_utils as ypu

    >>> with ypu.temp_style(["ysy_academic", "catppuccin_latte"]):
    ...     def model(x, p): 
    ...         return x ** (2 * p + 1) / (1 + x ** (2 * p))
    ...     x = np.linspace(0.75, 1.25, 201)
    ...     fig, ax = plt.subplots()
    ...     for p in [10, 15, 20, 25, 30, 40, 50, 100]:      
    ...         ax.plot(x, model(x, p), label=p)
    ...     ax.legend(title='Order')
    ...     ax.set(xlabel='Voltage (mV)')
    ...     ax.set(ylabel='Current ($\\mu$A)')
    ...     ax.autoscale(tight=True)
    ...     plt.show()

Changelog:
    v1.2.0:
        - Added `firefly_color_theme` for custom colormap generation.
        - Enhanced `plot` to support plotting data points.
        - Imported `LinearSegmentedColormap`.

    v1.3.0:
        - Added new Matplotlib theme support via `mplstyle` (VERY IMPORTANT).
        - Adjusted behavior of `firefly_color_theme` and `plot` for greater flexibility.

    v1.5.0:
        Optimize code structure and user experience, add new themes, and add a new theme directory.
"""

__version__ = "1.5.0"



# Import necessary package
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
## v1.3新增
import matplotlib.style as mplstyle
import tempfile, os, contextlib


# Settings

## Personal
def ysy_settings(dpi=300, color=True, font=True, jupyter=False, marker=False):
    '''
    Configures Matplotlib plot settings to ensure consistent and high-quality visualizations.

    This function creates a dictionary of configuration settings for Matplotlib. It allows customization of 
    figure appearance, gridlines, fonts, colors, markers, and other plot elements. You can toggle specific 
    settings based on your requirements, making it ideal for publication-ready plots or tailored visualizations.

    Args:
        dpi (int, optional): Dots per inch (DPI) for figure resolution. Defaults to 300.
        color (bool, optional): Enables a predefined color cycle for plots. Defaults to True.
        font (bool, optional): Configures font family and style. Defaults to True.
        jupyter (bool, optional): Adapts settings for Jupyter Notebook display, with smaller sizes and simpler fonts. Defaults to False.
        marker (bool, optional): Adds a custom marker style cycle to the plots. Defaults to False.

    Returns:
        dict: A dictionary containing Matplotlib configuration settings. This can be directly applied using `matplotlib.rcParams.update()`.

    Configuration Overview:
        - **Figure**: Default figure size and DPI.
        - **Axes**: Title and label font sizes, grid visibility and style.
        - **Ticks**: Major and minor tick sizes and widths, as well as label font sizes.
        - **Grid**: Customizable gridlines, including color, style, and transparency.
        - **Legend**: Frame visibility, shadow, font size, and fancy box styling.
        - **Font**: Default font family and support for mathematical text rendering.
        - **Lines**: Default line width and optional marker styles.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from ysy_plot_utils import ysy_settings
        >>> config = ysy_settings(dpi=150, color=True, font=False, marker=True)
        >>> plt.rcParams.update(config)
        >>> plt.plot([0, 1, 2], [0, 1, 4])
        >>> plt.show()

    '''
    # Global
    config = {
        # Default 
        # figure
        'figure.figsize':(12.5, 9), 
        # label
        'axes.labelsize':27,
        'axes.titlesize': 30,
        # x
        'xtick.major.size': 9,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 4.5,
        'xtick.minor.width': 1.5,
        # y
        'ytick.major.size': 9,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 4.5,
        'ytick.minor.width': 1.5,
        # xy label
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        # grid
        'axes.grid' : True,
        'axes.axisbelow' : True,
        'grid.linestyle': '--',
        'grid.color': 'k',
        'grid.alpha': 0.5,
        'grid.linewidth': 0.5,
        # legend
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.fancybox': True,
        'legend.numpoints': 1,
        'legend.shadow': True,
        'legend.fontsize': 25,
        'legend.title_fontsize': 25,
        # font
        'font.family': 'Times New Roman',
        'axes.formatter.use_mathtext': True,
        'mathtext.fontset': 'cm',
        'text.usetex': False,
        # line
        'lines.linewidth': 3.,
        }
    # Additional
    # dpi
    if dpi != None:
        add_dpi = {'figure.dpi': dpi}
        config.update(add_dpi)
    # color
    if color:
        add_color = {'axes.prop_cycle': cycler('color', firefly(requirement='cycle'))}
        config.update(add_color)
    # font
    if font:
        add_font = {
            'font.serif': ['cmr10', 'Computer Modern Serif', 'DejaVu Serif'],
            'font.family': 'serif'
        }
        config.update(add_font)
    # marker
    if marker:
        add_marker = {
            'axes.prop_cycle': (cycler('marker', ['o', 's', '^', 'v', '<', '>', 'd']) + 
                    cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']) + 
                    cycler('ls', [' ', ' ', ' ', ' ', ' ', ' ', ' '])),
            'lines.markersize': 3,
        }
        config.update(add_marker)
    # Jupyter Notebook
    if jupyter:
        add_jupyter = {
            'figure.figsize': (8, 6),
            'xtick.major.size': 6,
            'xtick.major.width': 1,
            'xtick.minor.size': 3,
            'xtick.minor.width': 1,
            'ytick.major.size': 6,
            'ytick.major.width': 1,
            'ytick.minor.size': 3,
            'ytick.minor.width': 1,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'legend.title_fontsize': 16,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2.,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'dejavusans',
            'text.usetex': False,
        }
        config.update(add_jupyter)
    return config

## Science
def science_settings(dpi=None, frame_on=False):
    '''
    Configures Matplotlib settings optimized for scientific plots.

    This function provides a set of Matplotlib configuration options tailored for creating high-quality 
    scientific visualizations. It ensures consistent styling with a focus on precision and clarity, suitable 
    for academic publications or professional presentations.

    Args:
        dpi (int, optional): Dots per inch (DPI) for figure resolution. Defaults to 300.
        frame_on (bool, optional): Toggles the display of a legend frame. Defaults to False.

    Returns:
        dict: A dictionary of Matplotlib configuration settings. This can be applied using `matplotlib.rcParams.update()`.

    Configuration Details:
        - **Color Cycle**: A predefined set of colors for plots.
        - **Figure**: Default size optimized for compact scientific plots (3.5 x 2.625 inches).
        - **Ticks**: Major and minor tick sizes, widths, and visibility for both x and y axes.
        - **Axes**: Thin axis lines for a cleaner look.
        - **Gridlines**: Lightweight gridlines with consistent width.
        - **Legend**: Optionally includes a frame with adjustable transparency.
        - **Savefig**: Tight bounding box and minimal padding for saving figures.
        - **Font**: Serif font family and support for mathematical text rendering.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from ysy_plot_utils import science_settings
        >>> config = science_settings(dpi=150, frame_on=True)
        >>> plt.rcParams.update(config)
        >>> plt.plot([0, 1, 2], [0, 1, 4], label='Example Line')
        >>> plt.legend()
        >>> plt.savefig('example_plot.png')
        >>> plt.show()

    Notes:
        - This configuration emphasizes compact and precise plots with minimal clutter.
        - The `dpi` parameter allows for high-resolution outputs suitable for publications.
        - The `frame_on` parameter provides flexibility to include or exclude a legend frame.

    '''
    config = {
    'axes.prop_cycle': cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']),
    'figure.figsize': (3.5, 2.625),
    'xtick.direction': 'in',
    'xtick.major.size': 3,
    'xtick.major.width': 0.5,
    'xtick.minor.size': 1.5,
    'xtick.minor.width': 0.5,
    'xtick.minor.visible': True,
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.major.size': 3,
    'ytick.major.width': 0.5,
    'ytick.minor.size': 1.5,
    'ytick.minor.width': 0.5,
    'ytick.minor.visible': True,
    'ytick.right': True,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'legend.frameon': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    }
    # Additional
    # dpi
    if dpi != None:
        add_dpi = {'figure.dpi': dpi}
        config.update(add_dpi)
    # legend frame
    if frame_on:
        add_legend_frame = {
            'legend.frameon': True,
            'legend.framealpha': 1.0,
        }
        config.update(add_legend_frame)
    return config


# Color

## color-firefly1
def firefly(requirement=None):
    '''
    Provides a predefined set of colors inspired by the firefly theme.

    This function returns color codes in hexadecimal format based on the specified `requirement`. It is 
    designed to offer a palette for styling plots and visualizations. The function can return the full 
    color palette, a subset for cycling, or a specific color code.

    Args:
        requirement (str, optional): Specifies the type of color output. Defaults to `None`.
            - `None`: Returns the full list of firefly colors.
            - `'cycle'`: Returns a subset of the palette suitable for color cycling in plots.
            - Specific color name (`'fgrayblue'`, `'fgraygreen'`, `'fred'`, `'fbluegreen'`, `'fblack'`, `'fsilver'`): Returns the corresponding hex color code.

    Returns:
        list or str: 
            - A list of hex color codes if `requirement` is `None` or `'cycle'`.
            - A single hex color code if a specific color name is provided.

    Full Palette (Default):
        1. `#475d7b` (Gray Blue)
        2. `#97c6c0` (Gray Green)
        3. `#e26e1b` (Deep Orange Yellow)
        4. `#4df8e8` (Blue Green)
        5. `#3e324a` (Purple Black)
        6. `#e6e4e0` (Silver White)

    Color Cycle (Subset for `requirement='cycle'`):
        - `#475d7b`, `#97c6c0`, `#e26e1b`, `#4df8e8`

    Color Names (For specific `requirement` values):
        - `'fgrayblue'`: `#475d7b`
        - `'fgraygreen'`: `#97c6c0`
        - `'fred'`: `#e26e1b`
        - `'fbluegreen'`: `#4df8e8`
        - `'fblack'`: `#3e324a`
        - `'fsilver'`: `#e6e4e0`

    Examples:
        >>> firefly()
        ['#475d7b', '#97c6c0', '#e26e1b', '#4df8e8', '#3e324a', '#e6e4e0']

        >>> firefly('cycle')
        ['#475d7b', '#97c6c0', '#e26e1b', '#4df8e8']

        >>> firefly('fred')
        '#e26e1b'

    Notes:
        - The function is flexible and can be used to style Matplotlib plots with custom color palettes.
        - Ensure the `requirement` string matches one of the predefined keys for specific color retrieval.

    '''
    if requirement == None:
        colors = ['#475d7b', '#97c6c0', '#e26e1b', '#4df8e8', '#3e324a', '#e6e4e0']
        return colors
    elif requirement == 'cycle':
        colors = ['#475d7b', '#97c6c0', '#e26e1b', '#4df8e8']
        return colors
    else:
        color_dictionary = {
            'fgrayblue': '#475d7b', 
            'fgraygreen': '#97c6c0', 
            'fred': '#e26e1b', 
            'fbluegreen': '#4df8e8', 
            'fblack': '#3e324a', 
            'fsilver': '#e6e4e0', 
        }
        return color_dictionary[requirement]
    
## color-firefly2
def firefly_color_theme(cmap_or_cycle=None, dark_or_light='dark', color_sample_num=10, set_color_cycle=False, reverse=False):
    """
    Generates and applies a custom color theme inspired by the 'Firefly Gallery' project.

    This function provides an option to create either a color map or a color cycle based on the 'Firefly' theme.
    The theme is designed to work in both dark and light modes, allowing users to select the appropriate theme 
    based on their preference. The function also supports automatically updating Matplotlib's color cycle.

    Args:
        cmap_or_cycle (str, optional): Determines the output of the function.
            - `'cycle'`: Returns a list of sampled colors for use as a color cycle.
            - `'cmap'`: Returns the generated color map.
            - `None`: No output, just updates the Matplotlib color cycle if `set_color_cycle` is True.
        dark_or_light (str, optional): Specifies the color theme to use.
            - `'dark'` (default): Uses the dark theme with colors `#FF8B4D` and `#5AFFCC`.
            - `'light'`: Uses the light theme with colors `#FF7C3A` and `#1AFFB2`.
        color_sample_num (int, optional): The number of colors to sample from the colormap. Default is 10.
        set_color_cycle (bool, optional): If `True`, the function updates Matplotlib's color cycle with the generated colors. Default is `False`.

    Returns:
        list or LinearSegmentedColormap or None: 
            - A list of colors if `cmap_or_cycle` is `'cycle'`.
            - A `LinearSegmentedColormap` if `cmap_or_cycle` is `'cmap'`.
            - `None` if no output is requested but the color cycle is updated in Matplotlib.

    Example:
        >>> from your_module import firefly_color_theme
        >>> firefly_color_theme('cycle')  # Get color cycle
        >>> firefly_color_theme('cmap')   # Get color map
        >>> firefly_color_theme(set_color_cycle=True)  # Automatically apply color cycle to Matplotlib

    Notes:
        - To apply the generated color cycle to Matplotlib, ensure that this function is called after `ysy_settings`.
        - The colors are sampled from a linear gradient colormap created using two sets of colors, one for dark and one for light themes.

    Update:
        - Add "reverse".
    """
    # 定义深色和浅色主题的颜色
    dark_colors = ["#FF8B4D", "#5AFFCC"]  # 深色主题
    light_colors = ["#FF7C3A", "#1AFFB2"]  # 浅色主题
    # 根据选择的主题设置主题颜色
    theme_colors = dark_colors
    if dark_or_light == 'light':  # 如果用户选择了浅色主题
        theme_colors = light_colors
    # 颜色反向
    if reverse:
        theme_colors = theme_colors[::-1]
    # 创建自定义的线性渐变 colormap，基于主题颜色
    firefly_cmap = LinearSegmentedColormap.from_list("firefly_theme", theme_colors)
    # 从渐变 colormap 中采样一定数量的颜色作为颜色循环
    firefly_cycle = [firefly_cmap(i / (color_sample_num - 1)) for i in range(color_sample_num)]
    # 设置颜色循环配置
    add_color = {'axes.prop_cycle': cycler('color', firefly_cycle)}
    # 如果需要设置颜色循环，更新 Matplotlib 的 rcParams
    if set_color_cycle:
        plt.rcParams.update(add_color)
    # 根据 cmap_or_cycle 参数返回不同的结果
    if cmap_or_cycle == 'cycle':
        return firefly_cycle  # 返回颜色循环
    elif cmap_or_cycle == 'cmap':
        return firefly_cmap  # 返回 colormap
    return None  # 如果没有特定要求，返回 None


# Plot

# Standardized Plot
def plot(x, y, legend_name, plot_title='', x_label='X', y_label='Y', plot_type='curve', legend_title='', data_point=None):
    '''
    A utility function to plot data using Matplotlib.

    This function generates a customizable plot (curve or scatter) for single or multiple datasets. 
    It provides options for setting plot titles, axis labels, legend names, and legend titles.

    Args:
        x (list or array-like): The x-axis data.
        y (list, array-like, or tuple of lists/arrays): The y-axis data. If `y` is a tuple, multiple datasets are plotted.
        legend_name (str or list of str): The legend label(s) for the dataset(s). Must match the length of `y` if `y` is a tuple.
        plot_title (str, optional): The title of the plot. Defaults to an empty string.
        x_label (str, optional): The label for the x-axis. Defaults to 'X'.
        y_label (str, optional): The label for the y-axis. Defaults to 'Y'.
        plot_type (str, optional): The type of plot to generate. Options are:
            - `'curve'` (default): Plots lines using `plt.plot`.
            - `'scatter'`: Plots points using `plt.scatter`.
            - `'with point'` (Only available for single drawing mode): 
        legend_title (str, optional): The title of the legend box. Defaults to an empty string.
        data_point (tuple of lists/arrays): Plot a curve with the data points. You must pass in something of the form (x_data_point, y_data_point). Also, only single plotting mode is available.

    Returns:
        None: The function does not return a value but displays the generated plot.

    Examples:
        Plotting a single dataset as a curve:
        >>> x = [1, 2, 3, 4]
        >>> y = [10, 20, 30, 40]
        >>> plot(x, y, legend_name='Dataset 1', plot_title='Sample Plot', x_label='Time', y_label='Value')

        Plotting multiple datasets as scatter plots:
        >>> x = [1, 2, 3, 4]
        >>> y = ([10, 20, 30, 40], [15, 25, 35, 45])
        >>> legend_names = ['Dataset 1', 'Dataset 2']
        >>> plot(x, y, legend_name=legend_names, plot_type='scatter', legend_title='Groups')

    Notes:
        - The function automatically handles multiple datasets if `y` is a tuple.
        - Ensure that the length of `legend_name` matches the number of datasets in `y` when `y` is a tuple.
        - The plot is immediately displayed using `plt.show()`.

    '''
    plt.figure()
    if isinstance(y, tuple) or isinstance(y, list):
        for i in range(len(y)):
            if plot_type == 'curve':
                plt.plot(x, y[i], label=legend_name[i])
            elif plot_type == 'scatter':
                plt.scatter(x, y[i], label=legend_name[i])
    else:
        if plot_type == 'curve':
            plt.plot(x, y, label=legend_name, zorder=1)
            if data_point != None:
                plt.scatter(data_point[0], data_point[1], label='Data Point', color=firefly('fblack'), s=150, marker='x', zorder=2)
        elif plot_type == 'scatter':
            plt.scatter(x, y, label=legend_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend(title=legend_title)
    plt.show()
    return None


# === 以下v1.3是新增功能 ===
# Style Manager

## 临时样式组装加载器
@contextlib.contextmanager
def temp_style(style_keys=None, extra_style: str = ""):
    """
    Temporarily applies a custom matplotlib style composed from preset and/or extra style definitions.

    This context manager dynamically creates a temporary `.mplstyle` file by combining predefined style
    snippets (from `PRESET_STYLES`) and any additional user-defined style string, then applies it using
    `matplotlib.style.use`. Upon exiting the context, it restores the default style and removes the temporary file.

    Parameters
    ----------
    style_keys : list[str], optional
        A list of keys referencing style snippets in the `PRESET_STYLES` dictionary. These styles will be
        merged in the given order. If None, only `extra_style` will be used.

    extra_style : str, optional
        Additional matplotlib style definitions provided as a raw string, which will be appended to the
        combined style content.

    Raises
    ------
    ValueError
        If any key in `style_keys` is not found in the `PRESET_STYLES` dictionary.

    Yields
    ------
    None
        Allows temporary application of the combined style inside a `with` block.

    Example (Recommended)
    -------
    >>> with ypu.temp_style(["ysy_academic", "catppuccin_latte"]):
    ...     plt.plot(x, y)
    ...     plt.show()
    
    """
    combined_style = ""
    if style_keys:
        for key in style_keys:
            if key in PRESET_STYLES:
                combined_style += PRESET_STYLES[key] + "\n"
            else:
                raise ValueError(f"Unknown style key: {key}")
    combined_style += extra_style

    with tempfile.NamedTemporaryFile('w+', suffix='.mplstyle', delete=False) as tmp:
        tmp.write(combined_style)
        tmp_path = tmp.name

    try:
        mplstyle.use(tmp_path)
        yield
    finally:
        mplstyle.reload_library()
        mplstyle.use('default')
        os.remove(tmp_path)

## 内置样式
def print_preset_styles():
    print('=== Recommended Loading Format ===')
    print('with ypu.temp_style(["ysy_academic", "sky"]):')
    print()
    print('=== Drawing Layout ===')
    print("ysy_common, ysy_jupyter, sci_common, science(recommended), ysy_academic(recommended)")
    print()
    print('=== Color Themes ===')
    print(
        '''catppuccin_mocha(dark, recommended), 
catppuccin_latte(recommended), 
ysy_firefly_1(lightweight), 
science_color(lightweight), 
catppuccin_farppe(dark), 
nord_light, nord_dark(dark), 
school_light, 
sky(lightweight, recommended), 
ysy_firefly_2(grayscale, lightweight), 
cold_nature(lightweight), 
mondrian_dunghuang(lightweight), 
sky2(lightweight)'''
    )
    return None

PRESET_STYLES = {
    "ysy_common": """
figure.figsize         : 12.5, 9
figure.dpi             : 300

axes.labelsize         : 27
axes.titlesize         : 30

xtick.major.size       : 9
xtick.major.width      : 1.5
xtick.minor.size       : 4.5
xtick.minor.width      : 1.5

ytick.major.size       : 9
ytick.major.width      : 1.5
ytick.minor.size       : 4.5
ytick.minor.width      : 1.5

xtick.labelsize        : 25
ytick.labelsize        : 25

axes.grid              : True
axes.axisbelow         : True
grid.linestyle         : --
grid.alpha             : 0.75
grid.linewidth         : 1

legend.frameon         : True
legend.framealpha      : 1.0
legend.fancybox        : True
legend.numpoints       : 1
legend.shadow          : True
legend.fontsize        : 25
legend.title_fontsize  : 25

lines.linewidth        : 3.0

# Font settings
font.family            : serif
font.serif             : cmr10, Computer Modern Serif, DejaVu Serif
axes.formatter.use_mathtext : True
mathtext.fontset       : cm
text.usetex            : False

""",
    "ysy_jupyter": """
figure.figsize         : 8, 6
figure.dpi             : 300

axes.labelsize         : 16
axes.titlesize         : 16
axes.linewidth         : 1

xtick.major.size       : 6
xtick.major.width      : 1
xtick.minor.size       : 3
xtick.minor.width      : 1
xtick.labelsize        : 16

ytick.major.size       : 6
ytick.major.width      : 1
ytick.minor.size       : 3
ytick.minor.width      : 1
ytick.labelsize        : 16

axes.grid              : True
axes.axisbelow         : True
grid.linestyle         : --
grid.alpha             : 0.75
grid.linewidth         : 1

legend.frameon         : True
legend.framealpha      : 1.0
legend.fancybox        : True
legend.numpoints       : 1
legend.shadow          : True
legend.fontsize        : 16
legend.title_fontsize  : 16

lines.linewidth        : 2.0

font.family            : sans-serif
mathtext.fontset       : dejavusans
text.usetex            : False

""",
    "catppuccin_mocha": """
# Catppuccin Mocha color theme (dark UI style)
axes.prop_cycle: cycler('color', ['89b4fa', 'fab387', 'a6e3a1', 'f38ba8', 'cba6f7', 'eba0ac', 'f5c2e7', 'f5e0dc', '94e2d5', 'b4befe'])

# Font color: Text
text.color: cdd6f4
axes.labelcolor: cdd6f4
xtick.labelcolor: cdd6f4
ytick.labelcolor: cdd6f4

# Background color: Base
figure.facecolor: 1e1e2e
axes.facecolor: 1e1e2e
savefig.facecolor: 1e1e2e

# Edge color: Surface 0
axes.edgecolor: 313244
legend.edgecolor: 313244
xtick.color: 313244
ytick.color: 313244
patch.edgecolor: 313244
hatch.color: 313244

# Grid color: Surface 0
grid.color: 313244

# Boxplots: Overlay 0
boxplot.flierprops.color: 6c7086
boxplot.flierprops.markerfacecolor: 6c7086
boxplot.flierprops.markeredgecolor: 6c7086
boxplot.boxprops.color: 6c7086
boxplot.whiskerprops.color: 6c7086
boxplot.capprops.color: 6c7086
boxplot.medianprops.color: 6c7086
boxplot.meanprops.color: 6c7086
boxplot.meanprops.markerfacecolor: 6c7086
boxplot.meanprops.markeredgecolor: 6c7086

""",
    "catppuccin_latte": """
# Note: this explicitly only alters colors and should be combined with
# other stylesheets if you want to edit other aspects, like linewidth,
# margins, etc.

# Original matplotlib cycle is blue, orange, green, red, purple, brown, pink, grey, light green, light blue
# Main colors: blue, peach, green, red, mauve, maroon, pink, rosewater, teal, lavender
axes.prop_cycle: cycler('color', ['1e66f5', 'fe640b', '40a02b', 'd20f39', '8839ef', 'e64553', 'ea76cb', 'dc8a78', '179299', '7287fd'])

# Font color: Text
text.color: 4c4f69
axes.labelcolor: 4c4f69
xtick.labelcolor: 4c4f69
ytick.labelcolor: 4c4f69

# Background color: Base
figure.facecolor: eff1f5
axes.facecolor: eff1f5
savefig.facecolor: eff1f5

# Edge color: Surface 0
axes.edgecolor: ccd0da
legend.edgecolor: ccd0da
xtick.color: ccd0da
ytick.color: ccd0da
patch.edgecolor: ccd0da
hatch.color: ccd0da

# Grid color: Surface 0
grid.color: ccd0da

# Boxplots: Overlay 0
boxplot.flierprops.color: 9ca0b0
boxplot.flierprops.markerfacecolor: 9ca0b0
boxplot.flierprops.markeredgecolor: 9ca0b0
boxplot.boxprops.color: 9ca0b0
boxplot.whiskerprops.color: 9ca0b0
boxplot.capprops.color: 9ca0b0
boxplot.medianprops.color: 9ca0b0
boxplot.meanprops.color: 9ca0b0
boxplot.meanprops.markerfacecolor: 9ca0b0
boxplot.meanprops.markeredgecolor: 9ca0b0

""",
    "ysy_firefly_1": """
axes.prop_cycle : cycler('color', ['475d7b', '97c6c0', 'e26e1b', '4df8e8', '3e324a', '6b8fb4', 'f1b349', 'a081af'])
grid.color: k

""",
    "science_color": """
axes.prop_cycle : cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])
grid.color: k

""",
    "sci_common": """
figure.figsize        : 3.375, 2.5
figure.dpi            : 300
savefig.dpi           : 300

font.family           : serif
font.size             : 8
axes.labelsize        : 8
axes.titlesize        : 8
xtick.labelsize       : 7
ytick.labelsize       : 7
legend.fontsize       : 7

lines.linewidth       : 1
lines.markersize      : 3

axes.linewidth        : 0.5
xtick.major.size      : 3
xtick.major.width     : 0.5
ytick.major.size      : 3
ytick.major.width     : 0.5

axes.grid             : False
legend.frameon        : False
pdf.fonttype          : 42
ps.fonttype           : 42
text.usetex           : False

""",
    "science": """
figure.figsize           : 3.5, 2.625

xtick.direction          : in
xtick.major.size         : 3
xtick.major.width        : 0.5
xtick.minor.size         : 1.5
xtick.minor.width        : 0.5
xtick.minor.visible      : True
xtick.top                : True

ytick.direction          : in
ytick.major.size         : 3
ytick.major.width        : 0.5
ytick.minor.size         : 1.5
ytick.minor.width        : 0.5
ytick.minor.visible      : True
ytick.right              : True

axes.linewidth           : 0.5
grid.linewidth           : 0.5
lines.linewidth          : 1.0

legend.frameon           : False
savefig.bbox             : tight
savefig.pad_inches       : 0.05

font.family              : serif
mathtext.fontset         : dejavuserif

""",
    "catppuccin_farppe": """
# Note: this explicitly only alters colors and should be combined with
# other stylesheets if you want to edit other aspects, like linewidth,
# margins, etc.

# Original matplotlib cycle is blue, orange, green, red, purple, brown, pink, grey, light green, light blue
# Main colors: blue, peach, green, red, mauve, maroon, pink, rosewater, teal, lavender
axes.prop_cycle: cycler('color', ['8caaee', 'ef9f76', 'a6d189', 'e78284', 'ca9ee6', 'ea999c', 'f4b8e4', 'f2d5cf', '81c8be', 'babbf1'])

# Font color: Text
text.color: c6d0f5
axes.labelcolor: c6d0f5
xtick.labelcolor: c6d0f5
ytick.labelcolor: c6d0f5

# Background color: Base
figure.facecolor: 303446
axes.facecolor: 303446
savefig.facecolor: 303446

# Edge color: Surface 0
axes.edgecolor: 414559
legend.edgecolor: 414559
xtick.color: 414559
ytick.color: 414559
patch.edgecolor: 414559
hatch.color: 414559

# Grid color: Surface 0
grid.color: 414559

# Boxplots: Overlay 0
boxplot.flierprops.color: 737994
boxplot.flierprops.markerfacecolor: 737994
boxplot.flierprops.markeredgecolor: 737994
boxplot.boxprops.color: 737994
boxplot.whiskerprops.color: 737994
boxplot.capprops.color: 737994
boxplot.medianprops.color: 737994
boxplot.meanprops.color: 737994
boxplot.meanprops.markerfacecolor: 737994
boxplot.meanprops.markeredgecolor: 737994

""",
    "nord_light": """
# Note: this explicitly only alters colors and should be combined with
# other stylesheets if you want to edit other aspects, like linewidth,
# margins, etc.

# Original matplotlib cycle is blue, orange, green, red, purple, brown, pink, grey, light green, light blue
# Main colors: frost + aurora + polar night accents
axes.prop_cycle: cycler('color', ['5e81ac', '88c0d0', 'a3be8c', 'd08770', 'bf616a', 'b48ead', 'ebcb8b', '81a1c1', '8fbcbb', '4c566a'])

# Font color: Text
text.color: 2e3440
axes.labelcolor: 2e3440
xtick.labelcolor: 2e3440
ytick.labelcolor: 2e3440

# Background color: Base
figure.facecolor: eceff4
axes.facecolor: eceff4
savefig.facecolor: eceff4

# Edge color: Surface 0
axes.edgecolor: d8dee9
legend.edgecolor: d8dee9
xtick.color: d8dee9
ytick.color: d8dee9
patch.edgecolor: d8dee9
hatch.color: d8dee9

# Grid color: Surface 0
grid.color: d8dee9

# Boxplots: Overlay 0
boxplot.flierprops.color: 4c566a
boxplot.flierprops.markerfacecolor: 4c566a
boxplot.flierprops.markeredgecolor: 4c566a
boxplot.boxprops.color: 4c566a
boxplot.whiskerprops.color: 4c566a
boxplot.capprops.color: 4c566a
boxplot.medianprops.color: 4c566a
boxplot.meanprops.color: 4c566a
boxplot.meanprops.markerfacecolor: 4c566a
boxplot.meanprops.markeredgecolor: 4c566a

""",
    "nord_dark": """
# Note: this explicitly only alters colors and should be combined with
# other stylesheets if you want to edit other aspects, like linewidth,
# margins, etc.

# Original matplotlib cycle is blue, orange, green, red, purple, brown, pink, grey, light green, light blue
# Main colors: polar night + frost + aurora
axes.prop_cycle: cycler('color', ['81a1c1', '8fbcbb', 'a3be8c', 'b48ead', 'd08770', 'bf616a', '5e81ac', '88c0d0', 'ebcb8b', 'eceff4'])

# Font color: Text
text.color: d8dee9
axes.labelcolor: d8dee9
xtick.labelcolor: d8dee9
ytick.labelcolor: d8dee9

# Background color: Base
figure.facecolor: 2e3440
axes.facecolor: 2e3440
savefig.facecolor: 2e3440

# Edge color: Surface 0
axes.edgecolor: 4c566a
legend.edgecolor: 4c566a
xtick.color: 4c566a
ytick.color: 4c566a
patch.edgecolor: 4c566a
hatch.color: 4c566a

# Grid color: Surface 0
grid.color: 4c566a

# Boxplots: Overlay 0
boxplot.flierprops.color: 616e88
boxplot.flierprops.markerfacecolor: 616e88
boxplot.flierprops.markeredgecolor: 616e88
boxplot.boxprops.color: 616e88
boxplot.whiskerprops.color: 616e88
boxplot.capprops.color: 616e88
boxplot.medianprops.color: 616e88
boxplot.meanprops.color: 616e88
boxplot.meanprops.markerfacecolor: 616e88
boxplot.meanprops.markeredgecolor: 616e88

""",
    "school_light": """
# school_light: Elegant light-colored UI theme based on gray-blue and rose accents

axes.prop_cycle: cycler('color', ['3e324a', '5a6b84', '1c2f62', 'a52a2a', 'd94352', 'c0e4d0', 'bcc7d6'])
""",
    "ysy_academic": """
figure.figsize         : 4.7, 2.9
figure.dpi             : 300

axes.labelsize         : 10.5
axes.titlesize         : 11.5
axes.linewidth         : 0.75

xtick.major.size       : 3
xtick.major.width      : 0.75
xtick.minor.size       : 1.5
xtick.minor.width      : 0.5
xtick.minor.visible    : False
xtick.top              : False

ytick.major.size       : 3
ytick.major.width      : 0.75
ytick.minor.size       : 1.5
ytick.minor.width      : 0.5
ytick.minor.visible    : False
ytick.right            : False

xtick.labelsize        : 9.5
ytick.labelsize        : 9.5

axes.grid              : False
axes.axisbelow         : True
grid.linestyle         : --
grid.alpha             : 0.85
grid.linewidth         : 0.75

legend.frameon         : True
legend.framealpha      : 1
legend.fancybox        : True
legend.numpoints       : 1
legend.shadow          : False
legend.fontsize        : 10
legend.title_fontsize  : 10

lines.linewidth        : 1.5
lines.markersize       : 5

# Font settings
font.family            : serif
axes.formatter.use_mathtext : True
mathtext.fontset       : cm
text.usetex            : False

""",
    "sky": """
# Sky Color theme, based on SkyRelax.
axes.prop_cycle: cycler('color', ['4c55bc', 'ffbe98', 'fc8f9b', 'ad69a2', 'f39477', 'eca4b8', '8c9cc1', 'c8ead4', '63d7fe', '388ef7', 'f7786b', '91dce8', 'd16d7c', '766c9b', '53181f', 'f7b9c2', '555d8b'])
""",
    "ysy_firefly_2": """
# YSY Firefly 2 color theme (grayish teal tone, light UI)
# Base color: #97c6c0, with grayscale variation for high contrast on white backgrounds
# Color sequence: grayscale-modulated variants of #97c6c0
axes.prop_cycle: cycler('color', ['3e5754', '567e79', '6e9f99', '85b6b1', '97c6c0', 'a7d1cb', 'bedfd8', 'd3eae5', 'e8f4f2', 'f5faf9'])
""",
    "cold_nature": """
# Cold Nature color theme (natural dusk palette, light UI) (Mixed with Memphis color) (v1.5 Test)
# Plot color cycle
axes.prop_cycle: cycler('color', ['403990', '80A6E2', 'FBDD85', 'F46F43', 'CF3D3E', '077ABD', 'B7AACB', 'F77A82', '89D0C2', 'EFD55E'])
""",
    "mondrian_dunghuang": """
# Mondiran+DungHuang color theme (light UI) (v1.5 Test)
axes.prop_cycle: cycler('color', ['0000ff', 'a64c3b', '8fbfb0', 'f20d0d', '6b3b32', 'b78f68', 'cabeaf', '25d4d0', 'ffff00'])
""",
    "sky2": """
# Additional sky color theme (v1.5 Test)
axes.prop_cycle: cycler('color', ['388ef7', 'f7786b', '91dce8', 'd16d7c', '766c9b', '53181f', 'f7b9c2', '555d8b'])
""",
    # 其他样式继续添加
}
