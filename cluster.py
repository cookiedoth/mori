import numpy as np
from sklearn.cluster import KMeans

features = np.load('image_features.npz')['features']
kmeans = KMeans(n_clusters=15, random_state=0, n_init="auto").fit(features)

print(kmeans.labels_)
np.savez('kmeans_labels.npz', features=kmeans.labels_)

kmeans_group = np.load('kmeans_labels.npz')['features']
colors_pal = [(50,150,77), (181,224,164), (27,81,29), (111,239,112), (115,138,105), (30,239,201), (25,71,125), (122,160,211), (98,46,134), (136,114,228), (206,43,188), (228,181,255), (87,53,246), (189,104,150), (199,206,53)]
colors_pal = [(v[0], v[1], v[2], 255) for v in colors_pal]
colors_pal = np.array(colors_pal) / 255.
np.savez('color_palette.npz', colors=colors_pal)
color = np.array([colors_pal[i] for i in kmeans_group])

print(color)