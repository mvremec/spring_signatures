from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
def plot_signature_map_with_hist3(signature):
    level=0.15
    fontsize = 8
    fig, axs = plt.subplot_mosaic(
        """ 
        AAA
        BCD
        """, figsize=(7,6), gridspec_kw={"height_ratios": [3, 1]})
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    austria = world[world['name'] == 'Austria']  # Select Austria from the world GeoDataFrame
    austria = austria.to_crs("3416")  # Set the Coordinate Reference System (CRS) for both Austria and your GeoPandas DataFrame if they differ
    ax = austria.plot(color='lightgray', edgecolor='black', ax=axs["A"])  # Plot the Austria map

    colrs = cm.get_cmap('RdBu', 8)  # get the coloramp from red to blue
    colrs1 = [colrs(i) for i in range(0, 8)]
    #schp = shp_change[(shp_change[signature]>1)|(shp_change[signature]<-1)].plot(signature, ax=axs["A"], legend=False, alpha=0.8, s=30, cmap=colrs, vmax=20, vmin=-20)  # plot the decade % change
    shp_change[(shp_change[signature]>-1)&(shp_change[signature]<1)].plot(ax=axs["A"], legend=False, marker="+", markersize=30, color="k",  linewidth=1)  # plot the decade % change between 1 and -1
    shp_ele1[(shp_ele1[signature]>1)|(shp_ele1[signature]<-1)].plot(signature, ax=axs["A"], legend=False, alpha=0.8, s=30, cmap=colrs, vmax=20, vmin=-20, marker="o")  # plot the decade % change
    shp_p1[shp_p1[signature]<level].plot(ax=axs["A"], legend=True, color="none", edgecolor="black", markersize=30, marker="o")  # plot the significant changes
    
    shp_ele2[(shp_ele2[signature]>1)|(shp_ele2[signature]<-1)].plot(signature, ax=axs["A"], legend=False, alpha=0.8, s=30, cmap=colrs, vmax=20, vmin=-20, marker="v")  # plot the decade % change
    shp_p2[shp_p2[signature]<level].plot(ax=axs["A"], legend=True, color="none", edgecolor="black", markersize=30, marker="v")  # plot the significant changes
    
    schp = shp_ele3[(shp_ele3[signature]>1)|(shp_ele3[signature]<-1)].plot(signature, ax=axs["A"], legend=False, alpha=0.8, s=30, cmap=colrs, vmax=20, vmin=-20, marker="^")  # plot the decade % change
    shp_p3[shp_p3[signature]<level].plot(ax=axs["A"], legend=True, color="none", edgecolor="black", markersize=30, marker="^")  # plot the significant changes
    
    #

    def plot_hist(input1, axs):
        bins1 = np.linspace(-20, 20, 9)  # Define the bin boundaries
        input1_clipped = np.clip(input1, bins1[0], bins1[-1])  # Clip the input data to the bin range
        axs1 = axs.hist(input1_clipped, bins=bins1)
        n, bins, patches = axs.hist(input1_clipped, bins=bins1)  # Use the clipped data
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Compute the bin centers
        col = bin_centers - min(bin_centers)  # Shift values to be ≥ 0
        col /= max(col)  # Scale values to be between 0 and 1
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', colrs(c)) 
        axs.set_xlim(-20, 20)
        axs.set_ylim(0, 6)
        return axs1

    plot_hist(shp_ele1[signature], axs["B"])
    plot_hist(shp_ele2[signature], axs["C"])
    plot_hist(shp_ele3[signature], axs["D"])

    cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.025])
    sm = ScalarMappable(cmap=colrs, norm=plt.Normalize(vmin=-20, vmax=20))
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"1996-2020 trend [% decade$^{-1}$]")

    for i in ("B", "C", "D"):
        axs[i].axvline(0, linestyle="--", color="gray", alpha=0.6)
        axs[i].tick_params(direction="in")
    axs["A"].tick_params(direction="in")
    axs["B"].set_ylabel("Stations")
    axs["C"].set_ylabel("")
    axs["D"].set_ylabel("")
    #axs["A"].text(0.05, 0.88, signature, transform=axs["A"].transAxes, fontsize=12)
    axs["B"].text(0.05, 0.88, "Elevation<1300 m", transform=axs["B"].transAxes, fontsize=fontsize)
    axs["C"].text(0.05, 0.88, "1300<Elevation<1800 m", transform=axs["C"].transAxes, fontsize=fontsize)
    axs["D"].text(0.05, 0.88, "Elevation>1800 m", transform=axs["D"].transAxes, fontsize=fontsize)
    legend_elements = [Line2D([0], [0], marker='o', markerfacecolor='none', color="black", linestyle='None', markersize=5),
                   Line2D([0], [0], marker='v',  markerfacecolor='none', color="black", linestyle='None', markersize=5),
                   Line2D([0], [0], marker='^',  markerfacecolor='none', color="black", linestyle='None', markersize=5),
                   Line2D([0], [0], marker='+',  markerfacecolor='none', color="black", linestyle='None', markersize=5)]
    axs["A"].legend(handles=legend_elements, labels=["Elevation<1300 m", "1300<Elevation<1800 m", "Elevation>1800 m", "-1%>trend<1%"], loc="upper left", title=signature)
    # Create a scale bar
    scalebar = ScaleBar(dx=1) # dx is the distance between two ticks on the scale bar. You need to adjust this according to your CRS.
    axs["A"].set_ylim(280000, 575000)
    axs["A"].axis("off")
    plt.subplots_adjust(hspace=-0.2)


def data_availability_plot(series, title = None, ax = None, figsize = [5,5], xlabel = "Time [Year]", ylabel = "Monitoring point"):

    """ 
    Plot the data availability for a given time series. The function uses the `pcolormesh` function from matplotlib to plot the data availability.

    Parameters:
    ----------
    series (pandas Series):
        The time series data to be plotted.
    title (str, optional):
        The title of the plot.
    ax (matplotlib Axes, optional):
        The Axes object to plot on. If not provided, a new figure will be created.
    figsize (tuple, optional): default = (10, 5)
        A tuple specifying the size of the figure. If not provided, the default is (5, 5).
    xlabel (str, optional): default = "Time [Days]"
        The label for the x-axis.
    ylabel (str, optional): default = "Monitoring point"
        The label for the y-axis.

    Returns:
    -------
    matplotlib Axes: The Axes object with the plotted data.
    """

    data = series

    if ax is None:

        fig = plt.subplots(figsize = figsize)
        ax = ax or plt.gca()
    
    bounds = np.array([0, 10000, np.inf])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', ['lightgray', 'darkgray'], N=256)
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', ['gray', 'gray'], N=256)
    ax.pcolormesh(data.values.T, norm = norm, cmap = cmap);#'tab20b');
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
    xticks = ax.get_xticks().astype(int)
    ax.set_xticklabels(data.index.year[xticks[:-1]]);

    if title is not None:
        ax.set_title("{}".format(title))