
#Dataset 1: blobs dataset

# Assuming df_rank is your DataFrame
data = {
    'Comparison': ['K-Means', 'DBScan'],
    'f1_score': [ 0.00, 0.0090],
    'rand_score': [0.44, 0.9161],
    'normalized_mutual_info_score': [0.63,  0.8868]
}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_rank = pd.DataFrame(data)

plt.figure(figsize=(16, 7))
Frame = df_rank['Comparison']

f1 = df_rank["f1_score"]
rand = df_rank["rand_score"]
normalized_mutual_info = df_rank["normalized_mutual_info_score"]

frame_axis = np.arange(len(Frame))

plt.bar(frame_axis + 0.1, f1, 0.1, label='f1_score', color="pink")
plt.bar(frame_axis + 0.2, rand, 0.1, label='rand_score', color="lightpink")
plt.bar(frame_axis, normalized_mutual_info, 0.1, label='normalized_mutual_info_score', color="deeppink")

plt.xticks(frame_axis, Frame)
plt.xlabel("Algorithm", fontsize=20)
plt.ylabel("Ratio")
plt.title("Dataset 1: blobs dataset", fontsize=25, fontweight="bold")
plt.legend()
plt.show()

###############################

#Dataset 2: Aniso dataset

# Assuming df_rank is your DataFrame
data = {
    'Comparison': ['K-Means', 'DBScan'],
    'f1_score': [  0.34,  0.0093],
    'rand_score': [0.93,  0.9302],
    'normalized_mutual_info_score': [0.91, 0.8990]
}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_rank = pd.DataFrame(data)

plt.figure(figsize=(16, 7))
Frame = df_rank['Comparison']

f1 = df_rank["f1_score"]
rand = df_rank["rand_score"]
normalized_mutual_info = df_rank["normalized_mutual_info_score"]

frame_axis = np.arange(len(Frame))

plt.bar(frame_axis + 0.1, f1, 0.1, label='f1_score', color="pink")
plt.bar(frame_axis + 0.2, rand, 0.1, label='rand_score', color="lightpink")
plt.bar(frame_axis, normalized_mutual_info, 0.1, label='normalized_mutual_info_score', color="deeppink")

plt.xticks(frame_axis, Frame)
plt.xlabel("Algorithm", fontsize=20)
plt.ylabel("Ratio")
plt.title("Dataset 2: Aniso dataset", fontsize=25, fontweight="bold")
plt.legend()
plt.show()

###############################

#Dataset 3: Noisy moons dataset

# Assuming df_rank is your DataFrame
data = {
    'Comparison': ['K-Means', 'DBScan'],
    'f1_score': [  0.48,   0.0318],
    'rand_score': [0.30,  0.8719],
    'normalized_mutual_info_score': [0.34, 0.7920]
}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_rank = pd.DataFrame(data)

plt.figure(figsize=(16, 7))
Frame = df_rank['Comparison']

f1 = df_rank["f1_score"]
rand = df_rank["rand_score"]
normalized_mutual_info = df_rank["normalized_mutual_info_score"]

frame_axis = np.arange(len(Frame))

plt.bar(frame_axis + 0.1, f1, 0.1, label='f1_score', color="pink")
plt.bar(frame_axis + 0.2, rand, 0.1, label='rand_score', color="lightpink")
plt.bar(frame_axis, normalized_mutual_info, 0.1, label='normalized_mutual_info_score', color="deeppink")

plt.xticks(frame_axis, Frame)
plt.xlabel("Algorithm", fontsize=20)
plt.ylabel("Ratio")
plt.title("Dataset 3: Noisy moons dataset", fontsize=25, fontweight="bold")
plt.legend()
plt.show()


###############################

#Dataset 4: Noisy Circles dataset

# Assuming df_rank is your DataFrame
data = {
    'Comparison': ['K-Means', 'DBScan'],
    'f1_score': [   0.27, 0.0044],
    'rand_score': [-0.00, 0.9934],
    'normalized_mutual_info_score': [ 0.0,  0.9858]
}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_rank = pd.DataFrame(data)

plt.figure(figsize=(16, 7))
Frame = df_rank['Comparison']

f1 = df_rank["f1_score"]
rand = df_rank["rand_score"]
normalized_mutual_info = df_rank["normalized_mutual_info_score"]

frame_axis = np.arange(len(Frame))

plt.bar(frame_axis + 0.1, f1, 0.1, label='f1_score', color="pink")
plt.bar(frame_axis + 0.2, rand, 0.1, label='rand_score', color="lightpink")
plt.bar(frame_axis, normalized_mutual_info, 0.1, label='normalized_mutual_info_score', color="deeppink")

plt.xticks(frame_axis, Frame)
plt.xlabel("Algorithm", fontsize=20)
plt.ylabel("Ratio")
plt.title("Dataset 4: Noisy Circles dataset", fontsize=25, fontweight="bold")
plt.legend()
plt.show()