import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#pd.read_csv
uv_data = pd.read_fwf("./joshua_image/new_obs.txt", skiprows=22)


print("uv_data", uv_data)

print("UV", uv_data[['U (lambda)', 'V (lambda)']])
#print("V", uv_data['V (lambda)'])

plt.scatter(uv_data['U (lambda)'], uv_data['V (lambda)'])
plt.title("uv coverage")
plt.show()
