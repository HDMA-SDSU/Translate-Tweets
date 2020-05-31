import geopandas
import pandas as pd
import pandas_alive
import contextily
import matplotlib.pyplot as plt


region_gdf = geopandas.read_file('C:\\Users\\jackm\\Documents\\GitHub\\pandas-alive\\data\\geo-data\\italy-with-regions\\reg2011_g.shp')
region_gdf.NOME_REG = region_gdf.NOME_REG.str.lower().str.title()
region_gdf = region_gdf.replace('Trentino-Alto Adige/Sudtirol','Trentino-Alto Adige')
region_gdf = region_gdf.replace("Valle D'Aosta/Vallée D'Aoste\r\nValle D'Aosta/Vallée D'Aoste","Valle d'Aosta")
# gdf = geopandas.read_file('data/nsw-covid19-cases-by-postcode.gpkg')

italy_df = pd.read_csv('data\Regional Data - Sheet1.csv',index_col=0,header=1,parse_dates=[0])

cases_df = italy_df.iloc[:,:3]
cases_df['Date'] = cases_df.index
pivoted = cases_df.pivot(values='New positives',index='Date',columns='Region')
pivoted.columns = pivoted.columns.astype(str)
pivoted = pivoted.rename(columns={'nan':'Unknown Region'})

cases_gdf = pivoted.T
cases_gdf['geometry'] = cases_gdf.index.map(region_gdf.set_index('NOME_REG')['geometry'].to_dict())
# cases_gdf[cases_gdf['geometry'].isna()]
cases_gdf = cases_gdf[cases_gdf['geometry'].notna()]
# print(type(cases_gdf))
cases_gdf = geopandas.GeoDataFrame(cases_gdf, crs=region_gdf.crs, geometry=cases_gdf.geometry)

gdf = cases_gdf

# gdf.index = gdf/.postcode
# gdf = gdf.drop('postcode',axis=1)

map_chart = gdf.plot_animated(basemap_format={'source':contextily.providers.CartoDB.Voyager},cmap='viridis')


# cases_df = pd.read_csv('data/nsw-covid-cases-by-postcode.csv',index_col=0,parse_dates=[0])
cases_df = pivoted

from datetime import datetime

bar_chart = cases_df.sum(axis=1).plot_animated(
    kind='line',
    label_events={
        'Schools Close':datetime.strptime("4/03/2020", "%d/%m/%Y"),
        'Country Lockdown':datetime.strptime("9/03/2020", "%d/%m/%Y"),
        'Production Recommences':datetime.strptime("26/04/2020", "%d/%m/%Y"),
        'Lockdown Ends':datetime.strptime("03/05/2020", "%d/%m/%Y"),

    },
    fill_under_line_color="blue",
    enable_progress_bar=True
)

map_chart.ax.set_title('Cases by Location')

# grouped_df = pd.read_csv('data/nsw-covid-cases-by-postcode.csv', index_col=0, parse_dates=[0])

line_chart = (
    cases_df.sum(axis=1)
    .cumsum()
    .fillna(0)
    .plot_animated(kind="line", period_label=False, title="Cumulative Total Cases")
)


def current_total(values):
    total = values.sum()
    s = f'Total : {int(total)}'
    return {'x': .85, 'y': .2, 's': s, 'ha': 'right', 'size': 11}

race_chart = cases_df.cumsum().plot_animated(
    n_visible=5, title="Cases by Region", period_label=False,period_summary_func=current_total
)

import time

timestr = time.strftime("%d/%m/%Y")

plots = [bar_chart, line_chart, map_chart, race_chart]

def update_all_graphs(frame):
    for plot in plots:
        try:
            plot.anim_func(frame)
        except Exception as e:
            print(e)
            raise UserWarning(
                f"Ensure all plots share index length {[plot.get_frames() for plot in plots]}"
            )

# Otherwise titles overlap and adjust_subplot does nothing
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

rcParams.update({"figure.autolayout": False})

figs = plt.figure()
gs = figs.add_gridspec(2, 3, hspace=0.5)
f3_ax1 = figs.add_subplot(gs[0, :])
f3_ax1.set_title(bar_chart.title)
bar_chart.ax = f3_ax1

f3_ax2 = figs.add_subplot(gs[1, 0])
f3_ax2.set_title(line_chart.title)
line_chart.ax = f3_ax2

f3_ax3 = figs.add_subplot(gs[1, 1])
f3_ax3.set_title(map_chart.title)
map_chart.ax = f3_ax3

f3_ax4 = figs.add_subplot(gs[1, 2])
f3_ax4.set_title(race_chart.title)
race_chart.ax = f3_ax4

axes = [f3_ax1, f3_ax2, f3_ax3, f3_ax4]
timestr = cases_df.index.max().strftime("%d/%m/%Y")
figs.suptitle(f"Italy COVID-19 Confirmed Cases up to {timestr}")

for num, plot in enumerate(plots):
    axes[num] = plot.apply_style(axes[num])
    if plot.fixed_max:
        # Hodgepodge way of fixing this, should refactor to contain all figures and axes
        # TODO plot.axes_format(self,ax) and pass current ax or desired ax
        if plot.__class__.__name__ == "BarChartRace" and plot.orientation == "h":
            axes[num].set_xlim(axes[num].get_xlim()[0], plot.df.values.max() * 1.1)
        elif plot.__class__.__name__ == "BarChartRace" and plot.orientation == "v":
            axes[num].set_ylim(axes[num].get_ylim()[0], plot.df.values.max() * 1.1)

    axes[num].set_title(plot.title)
    plot.ax = axes[num]

    plot.init_func()

fps = 1000 / plots[0].period_length * plots[0].steps_per_period
interval = plots[0].period_length / plots[0].steps_per_period
anim = FuncAnimation(
    figs,
    update_all_graphs,
    min([max(plot.get_frames()) for plot in plots]),
    interval=interval,
)

anim.save("test.mp4", dpi=144)