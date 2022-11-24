```python
import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(19680801)
N = 20 
theta = np.linspace(0.0,2*np.pi,N,endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
colors = plt.cm.viridis(radii / 10.)
ax = plt.subplot(projection='polar')
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)
plt.show()
```

    Matplotlib is building the font cache; this may take a moment.
    


    
![png](output_0_1.png)
    



```python
import time
import tqdm
count = 0 
for i in tqdm.tqdm(range(100)):
    count += i 
    time.sleep(0.4)
print(f'{count = }')
```

    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:40<00:00,  2.45it/s]

    count = 4950
    

    
    


```python
animals = ['python', 'viber', 'cobra']
def add_snake(snake_type):
    animals.extend(snake_type)
    print(animals)
    
add_snake('Boa')
```

    ['python', 'viber', 'cobra', 'B', 'o', 'a']
    


```python
print([number for number in range(25,35,2)])
```

    [25, 27, 29, 31, 33]
    


```python
plt.get_cmap('viridis')
```




<div style="vertical-align: middle;"><strong>viridis</strong> </div><div class="cmap"><img alt="viridis colormap" title="viridis" style="border: 1px solid #555;" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAABACAYAAABsv8+/AAAAFnRFWHRUaXRsZQB2aXJpZGlzIGNvbG9ybWFwrE0mCwAAABx0RVh0RGVzY3JpcHRpb24AdmlyaWRpcyBjb2xvcm1hcAtjl3IAAAAwdEVYdEF1dGhvcgBNYXRwbG90bGliIHYzLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZwld89MAAAAydEVYdFNvZnR3YXJlAE1hdHBsb3RsaWIgdjMuNC4zLCBodHRwczovL21hdHBsb3RsaWIub3JnJ/ts9AAAAiJJREFUeJzt1kGSmzAURdEv2FqWkP0vJfQgMhQCGceV2Ttn4pL0EVQPum771X5vVVXVWv39XfrPeV193V5zS98f1sf5/fPjey73zu6/3Hv/uz2cz57f9vP68rxO9+/zre7nhvvG+et6vH92bw3PDfcsD+eX59+/53n96f3362/f87/vf5yr93Of72/fPV9P89tX3zGeH3OT8/07Zs+/32+TuXZZD8/VODf8W5uuH/b7vctlfuv7NazH8/t7ZnP7bz2cD3NL+/Ph3Hl+/efz83vWun/vuL++nquH9eu9w/uu6/vvOO49f/8xf77vOj+8b7Y/fMfse9ca/y7nv+d62a++X+f1vt+G/b7u+/u6TxzzS//tc2053QMABBEAABBIAABAIAEAAIEEAAAEEgAAEEgAAEAgAQAAgQQAAAQSAAAQSAAAQCABAACBBAAABBIAABBIAABAIAEAAIEEAAAEEgAAEEgAAEAgAQAAgQQAAAQSAAAQSAAAQCABAACBBAAABBIAABBIAABAIAEAAIEEAAAEEgAAEEgAAEAgAQAAgQQAAAQSAAAQSAAAQCABAACBBAAABBIAABBIAABAIAEAAIEEAAAEEgAAEEgAAEAgAQAAgQQAAAQSAAAQSAAAQCABAACBBAAABBIAABBIAABAIAEAAIEEAAAEEgAAEEgAAEAgAQAAgQQAAAQSAAAQSAAAQCABAACBBAAABBIAABBIAABAoB9ucImHxcKZtAAAAABJRU5ErkJggg=="></div><div style="vertical-align: middle; max-width: 514px; display: flex; justify-content: space-between;"><div style="float: left;"><div title="#440154ff" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #440154ff;"></div> under</div><div style="margin: 0 auto; display: inline-block;">bad <div title="#00000000" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #00000000;"></div></div><div style="float: right;">over <div title="#fde725ff" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #fde725ff;"></div></div>




```python
plt.get_cmap('viridis_r')
```




<div style="vertical-align: middle;"><strong>viridis_r</strong> </div><div class="cmap"><img alt="viridis_r colormap" title="viridis_r" style="border: 1px solid #555;" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAABACAYAAABsv8+/AAAAGHRFWHRUaXRsZQB2aXJpZGlzX3IgY29sb3JtYXA0MKMeAAAAHnRFWHREZXNjcmlwdGlvbgB2aXJpZGlzX3IgY29sb3JtYXB2q0WVAAAAMHRFWHRBdXRob3IATWF0cGxvdGxpYiB2My40LjMsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcJXfPTAAAAMnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHYzLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZyf7bPQAAAImSURBVHic7dZBdpwwEEXRwmvJ/reVVTTKoMEcilbjdob/3omPCklgD5K3PP7+GVVVj7FWVdVaY/v5XD9GW+/Pt/n3uuq0/7Gtj3lb17LNl22+nObrOK+P+Vebf72eb+u1bubt3HHP/nyZ7N/f+/47puvv7+jf1daX+/vfp8/7PW3e/943z/v7Zu+Z7Rs3943Zuaq3+451ndaXffX6+Zicu873C9q+6vN9fT5fl/PndbVzt/Pq8+ePZXrufH6ZPL/Ob/ZPv+Nu/ul3TOb1ev7T8/+/b/zy3Kfrz95zrMfv7l/fP799zzp7b/892n8Il++d7Z/tO9+39PmPn0/m+/fdPJ+eX9fTelz2r21/m6/9XN+/vr13TN7z/FccAIgiAAAgkAAAgEACAAACCQAACCQAACCQAACAQAIAAAIJAAAIJAAAIJAAAIBAAgAAAgkAAAgkAAAgkAAAgEACAAACCQAACCQAACCQAACAQAIAAAIJAAAIJAAAIJAAAIBAAgAAAgkAAAgkAAAgkAAAgEACAAACCQAACCQAACCQAACAQAIAAAIJAAAIJAAAIJAAAIBAAgAAAgkAAAgkAAAgkAAAgEACAAACCQAACCQAACCQAACAQAIAAAIJAAAIJAAAIJAAAIBAAgAAAgkAAAgkAAAgkAAAgEACAAACCQAACCQAACCQAACAQAIAAAIJAAAIJAAAIJAAAIBAAgAAAgkAAAgkAAAgkAAAgED/AE+UgCcBrBTgAAAAAElFTkSuQmCC"></div><div style="vertical-align: middle; max-width: 514px; display: flex; justify-content: space-between;"><div style="float: left;"><div title="#fde725ff" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #fde725ff;"></div> under</div><div style="margin: 0 auto; display: inline-block;">bad <div title="#00000000" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #00000000;"></div></div><div style="float: right;">over <div title="#440154ff" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #440154ff;"></div></div>




```python
import numpy as np 
import matplotlib.pyplot as plt 
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
x, y = np.meshgrid(x, y)
z = f(x, y)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.contour3D(x,y,z,50, cmap="binary")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(x,y,z, color = "black")
ax.set_title("wireframe")
plt.show()
ax = plt.axes(projection="3d")
ax.plot_surface(x, y, z, rstride=1, cstride = 1, cmap="viridis", edgecolor = "none")
ax.set_title("surface")
plt.show()
```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



```python
import numpy as np 
import matplotlib.pyplot as plt

N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)

area = 200 * r**2 
colors = theta 

plt.figure(figsize=(12,12))

axes_1 = plt.subplot(1,3,1, projection="polar")
axes_1.scatter(theta, r, c=colors, s=area, cmap="hsv", alpha=0.75)
axes_1.set_title("polar plot")

axes_2 = plt.subplot(1,3,2, projection="polar")
axes_2.scatter(theta, r, c=colors, s=area, cmap="hsv", alpha=0.75)
axes_2.set_rorigin(-2.5)
axes_2.set_theta_zero_location("W", offset=10)
axes_2.set_title("polar with offset origin")

axes_3 = plt.subplot(1,3,3, projection="polar")
axes_3.scatter(theta, r, c=colors, s=area, cmap="hsv", alpha=0.75)
axes_3.set_thetamin(45)
axes_3.set_thetamax(135)
axes_3.set_title("polar confined to a sector")
plt.show()
```


    
![png](output_7_0.png)
    



```python
import pandas as pd 
from pandas.plotting import scatter_matrix
df = pd.read_csv("C:/Users/Mua/Downloads/penguins.csv.csv").dropna()
color_codes = {"Adelie": 1, "Chinstrap" : 2, "Gentoo" : 3}
df["species"] = df["species"].map(color_codes)

x = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]

y = df["species"] 

sm = pd.plotting.scatter_matrix(x, c=y, hist_kwds={"bins":20}, figsize=(7,7))
```


    
![png](output_8_0.png)
    



```python
import pandas as pd 
import numpy as np 

df = pd.DataFrame(np.random.randn(1000,2), columns=["a", "b"])
df["b"] = df["b"] + np.arange(1000)

df.plot.hexbin(x="a", y="b", gridsize=25)
```




    <AxesSubplot:xlabel='a', ylabel='b'>




    
![png](output_9_1.png)
    



```python
import matplotlib.pyplot as plt 

labels = "Python","Java", "C++", "C#","JavaScript", "PHP"
sizes = [24.8, 9.4, 31.3, 10.8, 11.3, 12.4]

explode = (0.1, 0, 0, 0, 0, 0.0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)

plt.show()
```


    
![png](output_10_0.png)
    



```python
import seaborn as sns 

pengunis = sns.load_dataset("penguins")
pengunis.head(2)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns 

pengunis = sns.load_dataset("penguins")
pengunis.head(2)

sns.pairplot(pengunis, hue="sex")
```




    <seaborn.axisgrid.PairGrid at 0x20e445aa100>




    
![png](output_12_1.png)
    



```python
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(0, 10)
y = np.sin(x)
z = np.cos(x)

fig, ax_main = plt.subplots()

ax_inset = fig.add_axes(rect=[0.18, 0.18, 0.2, 0.2])
ax_main.plot(x, y, color="red")
ax_main.set_title("sin(x)")

ax_inset.plot(x, z, color="green")
ax_inset.set_title("cos(x)")

plt.show()
```


    
![png](output_13_0.png)
    

